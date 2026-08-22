"""The evaluation loop. Every model, every protocol, every fold.

The structure is deliberately boring. What matters is that a single function
runs the leaky protocol and the honest ones through identical code, so the gap
between them cannot be an artefact of two different evaluation paths and that
every fold is checked for the leakage its protocol claims to exclude before a
single number comes out of it.

Out-of-fold predictions are pooled and kept. Every cell appears in exactly one
test fold, so the pooled vector is a complete leave-one-group-out prediction for
the whole dataset and it is what the cost models are priced against. Pricing
against a single fold would put the business numbers at the mercy of which
batches happened to land in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cobalt.config import Config
from cobalt.data.splits import TASK_PROTOCOLS, assert_no_leakage, build_splits
from cobalt.evaluate.metrics import aggregate, interval_metrics, regression_metrics
from cobalt.features import early_life as early_life_features
from cobalt.features import online as online_features
from cobalt.models.conformal import ConformalRegressor
from cobalt.models.registry import build

#: Which column each protocol groups on when calibrating a conformal interval.
#: ``None`` means calibrate on random rows, which is the right thing to do only
#: when the protocol itself splits on random rows.
_CALIBRATION_GROUP = {
    "random_cycle": None,
    "cell_independent": "cell_id",
    "batch_independent": "batch_id",
}

_TASK_TARGET = {"early_life": "log_cycle_life", "online_soh": "log_remaining_cycles"}
_TASK_FEATURES = {
    "early_life": early_life_features.FEATURE_NAMES,
    "online_soh": online_features.FEATURE_NAMES,
}


@dataclass
class OutOfFoldPredictions:
    """Pooled predictions for one model under one protocol."""

    log_truth: np.ndarray
    log_prediction: np.ndarray
    log_lower: np.ndarray | None
    log_upper: np.ndarray | None
    cell_id: np.ndarray
    batch_id: np.ndarray
    #: The label in its natural units, carried through for the cost models.
    natural_truth: np.ndarray

    def as_frame(self) -> pd.DataFrame:
        data: dict[str, Any] = {
            "cell_id": self.cell_id,
            "batch_id": self.batch_id,
            "log_truth": self.log_truth,
            "log_prediction": self.log_prediction,
            "truth": self.natural_truth,
        }
        if self.log_lower is not None and self.log_upper is not None:
            data["log_lower"] = self.log_lower
            data["log_upper"] = self.log_upper
        return pd.DataFrame(data)


def _model_kwargs(config: Config) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "rated_capacity_ah": 2.20,
        "end_of_life_soh": config.fleet.end_of_life_soh,
    }


def _feature_view(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    """Just the model-visible columns, in a fixed order.

    Selecting by an explicit tuple rather than by dropping the label columns is
    the difference between adding a diagnostic column to the table and silently
    feeding it to the model.
    """
    names = list(_TASK_FEATURES[task])
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"feature table for {task} is missing {missing}")
    return frame[names]


def _natural_truth(frame: pd.DataFrame, task: str) -> np.ndarray:
    column = "cycle_life" if task == "early_life" else "remaining_cycles"
    return frame[column].to_numpy(dtype=float)


def evaluate_model(
    task: str,
    frame: pd.DataFrame,
    protocol: str,
    model_name: str,
    config: Config,
    with_intervals: bool = False,
    grouped_calibration: bool = True,
) -> tuple[dict[str, float], OutOfFoldPredictions, dict[str, float] | None]:
    """Cross-validate one model under one protocol.

    Returns the aggregated metrics, the pooled out-of-fold predictions and, when
    intervals were requested, the aggregated interval metrics.
    """
    features = _feature_view(frame, task)
    target = frame[_TASK_TARGET[task]].to_numpy(dtype=float)
    natural = _natural_truth(frame, task)

    splits = build_splits(frame, protocol, n_folds=config.evaluation.n_folds, seed=config.seed)

    n = len(frame)
    pooled_prediction = np.full(n, np.nan)
    pooled_lower = np.full(n, np.nan)
    pooled_upper = np.full(n, np.nan)

    fold_metrics: list[dict[str, float]] = []
    fold_intervals: list[dict[str, float]] = []

    for split in splits:
        assert_no_leakage(frame, split)

        train_features = features.iloc[split.train_index]
        test_features = features.iloc[split.test_index]
        train_target = target[split.train_index]
        test_target = target[split.test_index]

        base = build(task, model_name, **_model_kwargs(config))

        if with_intervals:
            group_column = _CALIBRATION_GROUP[protocol] if grouped_calibration else None
            groups = (
                frame.iloc[split.train_index][group_column].to_numpy()
                if group_column is not None
                else None
            )
            model = ConformalRegressor(
                base,
                alpha=config.evaluation.alpha,
                calibration_fraction=config.evaluation.calibration_fraction,
                grouped=grouped_calibration,
                seed=config.seed + split.fold,
            )
            model.fit(train_features, train_target, groups=groups)
            prediction = model.predict(test_features)
            lower, upper = model.predict_interval(test_features)
            pooled_lower[split.test_index] = lower
            pooled_upper[split.test_index] = upper
            fold_intervals.append(
                interval_metrics(test_target, lower, upper, config.evaluation.alpha)
            )
        else:
            base.fit(train_features, train_target)
            prediction = base.predict(test_features)

        pooled_prediction[split.test_index] = prediction
        fold_metrics.append(regression_metrics(test_target, prediction))

    if np.isnan(pooled_prediction).any():
        # Folds partition the groups, so every row must be tested exactly once.
        # If that stops being true the pooled cost numbers quietly become a
        # subset average, so fail here instead.
        missing = int(np.isnan(pooled_prediction).sum())
        raise AssertionError(
            f"{protocol}/{model_name}: {missing} of {n} rows were never in a test fold"
        )

    predictions = OutOfFoldPredictions(
        log_truth=target,
        log_prediction=pooled_prediction,
        log_lower=pooled_lower if with_intervals else None,
        log_upper=pooled_upper if with_intervals else None,
        cell_id=frame["cell_id"].to_numpy(),
        batch_id=frame["batch_id"].to_numpy(),
        natural_truth=natural,
    )

    intervals = aggregate(fold_intervals) if with_intervals else None
    return aggregate(fold_metrics), predictions, intervals


def run_task(task: str, frame: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Every model under every protocol valid for the task.

    The deployed model is additionally run with conformal intervals twice: once
    calibrated the way the protocol demands, once calibrated on random rows. The
    second is the mistake and the coverage gap between them is what it costs.
    """
    protocols = TASK_PROTOCOLS[task]
    model_names = (
        config.evaluation.early_life_models
        if task == "early_life"
        else config.evaluation.online_models
    )
    deployed = (
        config.evaluation.deployed_early_life_model
        if task == "early_life"
        else config.evaluation.deployed_online_model
    )
    if deployed not in model_names:
        raise ValueError(
            f"deployed model {deployed!r} for {task} is not in the evaluated set {model_names}"
        )

    out: dict[str, Any] = {
        "task": task,
        "n_rows": int(len(frame)),
        "n_cells": int(frame["cell_id"].nunique()),
        "n_batches": int(frame["batch_id"].nunique()),
        "deployed_model": deployed,
        "alpha": config.evaluation.alpha,
        "protocols": {},
    }

    for protocol in protocols:
        models: dict[str, Any] = {}
        pooled: dict[str, OutOfFoldPredictions] = {}

        for model_name in model_names:
            metrics, predictions, _ = evaluate_model(
                task, frame, protocol, model_name, config, with_intervals=False
            )
            models[model_name] = metrics
            pooled[model_name] = predictions

        _, deployed_predictions, matched = evaluate_model(
            task,
            frame,
            protocol,
            deployed,
            config,
            with_intervals=True,
            grouped_calibration=True,
        )
        _, mismatched_predictions, mismatched = evaluate_model(
            task,
            frame,
            protocol,
            deployed,
            config,
            with_intervals=True,
            grouped_calibration=False,
        )

        out["protocols"][protocol] = {
            "models": models,
            "intervals": {
                "matched_calibration": matched,
                "random_calibration": mismatched,
            },
        }
        pooled[f"{deployed}+conformal"] = deployed_predictions
        pooled[f"{deployed}+conformal[random-cal]"] = mismatched_predictions
        out["protocols"][protocol]["_pooled"] = pooled

    return out


def build_feature_tables(dataset, config: Config) -> dict[str, Any]:
    """Both feature tables plus what each one had to exclude."""
    early_frame, early_excluded = early_life_features.build_table(dataset)
    online_frame, online_excluded = online_features.build_table(
        dataset,
        window_cycles=config.online.window_cycles,
        stride=config.online.stride,
    )
    return {
        "early_life": {"frame": early_frame, "excluded": early_excluded},
        "online_soh": {"frame": online_frame, "excluded": online_excluded},
    }
