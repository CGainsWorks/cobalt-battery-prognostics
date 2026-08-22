"""Fitting the models that actually get shipped, and writing their model card.

The evaluation refits a model per fold to measure it. Deployment fits once, on
everything, and that is a different object with different provenance. Keeping
the two apart matters: the number quoted for a deployed model must come from
cross-validation, never from the fit it was trained with, and the easiest way to
get that wrong is to let one function do both.

Every artefact written here carries a model card next to it. The card states
that the training data is simulated, in the first field, because a serialised
model outlives the conversation in which somebody explained that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cobalt.config import Config
from cobalt.features import early_life as early_life_features
from cobalt.features import online as online_features
from cobalt.models.conformal import ConformalRegressor
from cobalt.models.registry import build, is_baseline

EARLY_LIFE_ARTEFACT = "early_life_model.joblib"
ONLINE_ARTEFACT = "online_model.joblib"
MODEL_CARD = "model_card.json"


@dataclass
class DeployedModel:
    """A fitted conformal model plus what a caller needs to use it correctly."""

    task: str
    model_name: str
    feature_names: tuple[str, ...]
    alpha: float
    conformal: ConformalRegressor
    training_rows: int
    #: Label distribution the model saw, for the no-features fallback policy.
    training_log_target: np.ndarray

    def predict(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        """Point estimate and interval, in natural units rather than log10."""
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise KeyError(f"{self.task} model needs missing features: {missing}")
        ordered = features[list(self.feature_names)]
        centre = self.conformal.predict(ordered)
        lower, upper = self.conformal.predict_interval(ordered)
        return {
            "point": np.power(10.0, centre),
            "lower": np.power(10.0, lower),
            "upper": np.power(10.0, upper),
            "log_point": centre,
            "log_lower": lower,
            "log_upper": upper,
        }


def _fit(
    task: str,
    model_name: str,
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    target_column: str,
    group_column: str,
    config: Config,
) -> DeployedModel:
    if is_baseline(model_name):
        raise ValueError(
            f"{model_name} is a baseline and is reported but never deployed. "
            f"Set deployed_{task}_model in the config to a learned model."
        )

    features = frame[list(feature_names)]
    target = frame[target_column].to_numpy(dtype=float)
    groups = frame[group_column].to_numpy()

    base = build(
        task,
        model_name,
        seed=config.seed,
        rated_capacity_ah=2.20,
        end_of_life_soh=config.fleet.end_of_life_soh,
    )
    conformal = ConformalRegressor(
        base,
        alpha=config.evaluation.alpha,
        calibration_fraction=config.evaluation.calibration_fraction,
        grouped=True,
        seed=config.seed,
    )
    conformal.fit(features, target, groups=groups)

    return DeployedModel(
        task=task,
        model_name=model_name,
        feature_names=feature_names,
        alpha=config.evaluation.alpha,
        conformal=conformal,
        training_rows=len(frame),
        training_log_target=target,
    )


def train_and_save(
    config: Config,
    early_frame: pd.DataFrame,
    online_frame: pd.DataFrame,
    summary: dict[str, Any],
    directory: Path,
) -> dict[str, str]:
    """Fit both deployed models on the full dataset and write them with a card."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    early = _fit(
        "early_life",
        config.evaluation.deployed_early_life_model,
        early_frame,
        early_life_features.FEATURE_NAMES,
        "log_cycle_life",
        "batch_id",
        config,
    )
    online = _fit(
        "online_soh",
        config.evaluation.deployed_online_model,
        online_frame,
        online_features.FEATURE_NAMES,
        "log_remaining_cycles",
        "batch_id",
        config,
    )

    joblib.dump(early, directory / EARLY_LIFE_ARTEFACT)
    joblib.dump(online, directory / ONLINE_ARTEFACT)
    card = build_model_card(config, early, online, summary)
    (directory / MODEL_CARD).write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    return {
        "early_life": str(directory / EARLY_LIFE_ARTEFACT),
        "online_soh": str(directory / ONLINE_ARTEFACT),
        "model_card": str(directory / MODEL_CARD),
    }


def build_model_card(
    config: Config,
    early: DeployedModel,
    online: DeployedModel,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Provenance, honest headline numbers, and the limitation stated first."""

    def headline(task: str, model: str) -> dict[str, Any]:
        protocols = summary["tasks"][task]["protocols"]
        # Quote the hardest protocol available for the task. Quoting the best one
        # is the default failure mode of a model card and it is why they are
        # usually worthless.
        for protocol in ("batch_independent", "cell_independent", "random_cycle"):
            if protocol in protocols:
                metrics = protocols[protocol]["models"][model]
                return {
                    "protocol": protocol,
                    "rmse_cycles": round(metrics["rmse_cycles"], 1),
                    "mape": round(metrics["mape"], 4),
                    "alpha_lambda": round(metrics["alpha_lambda"], 4),
                    "spearman": round(metrics["spearman"], 4),
                    "picp": round(
                        protocols[protocol]["intervals"]["matched_calibration"]["picp"], 4
                    ),
                }
        raise KeyError(f"no protocols recorded for {task}")

    return {
        "data_source": "SIMULATED. No physical cells were tested. See docs/LIMITATIONS.md.",
        "config_name": config.name,
        "seed": config.seed,
        "intended_use": (
            "Demonstration of a leakage-free evaluation protocol and a cost-aware "
            "decision layer for battery prognostics. Not qualified for use on real packs."
        ),
        "out_of_scope": [
            "Any safety-related decision.",
            "Any real cell chemistry, which this has never seen.",
            "Warranty or resale pricing without recalibration on measured data.",
        ],
        "models": {
            "early_life": {
                "name": early.model_name,
                "target": "log10 cycles to 80% of nameplate capacity",
                "features": list(early.feature_names),
                "training_rows": early.training_rows,
                "conformal_alpha": early.alpha,
                "conformal_calibration_rows": early.conformal.n_calibration,
                "reported": headline("early_life", early.model_name),
            },
            "online_soh": {
                "name": online.model_name,
                "target": "log10 remaining cycles to end of life",
                "features": list(online.feature_names),
                "training_rows": online.training_rows,
                "conformal_alpha": online.alpha,
                "conformal_calibration_rows": online.conformal.n_calibration,
                "reported": headline("online_soh", online.model_name),
            },
        },
        "decision_layer": {
            "lot_acceptance": "cobalt.business.grading, cost matrix in the config",
            "maintenance_scheduling": "cobalt.business.cost, newsvendor quantile "
            f"{config.replacement.optimal_quantile:.3f}",
        },
    }


def load_deployed(directory: Path) -> tuple[DeployedModel, DeployedModel]:
    """Load both models, with a message that names the fix if they are absent."""
    directory = Path(directory)
    early_path = directory / EARLY_LIFE_ARTEFACT
    online_path = directory / ONLINE_ARTEFACT
    missing = [str(path) for path in (early_path, online_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"no trained models at {directory} (missing {', '.join(missing)}). "
            f"Run `cobalt evaluate --config config/default.yaml` first, or "
            f"`make demo` from a clean checkout."
        )
    return joblib.load(early_path), joblib.load(online_path)
