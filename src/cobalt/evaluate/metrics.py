"""Scoring.

Everything is fitted in log10 cycles, because cycle life spans roughly an order
of magnitude across a fleet and a squared error in linear cycles would let the
long-lived cells set the whole objective. Everything is *reported* in both
spaces, because a maintenance planner does not think in decades.

Four families of number, and they disagree with each other often enough to be
worth keeping all four:

* **Error size.** RMSE and MAE, in log space and in cycles.
* **Relative error.** MAPE, and the alpha-lambda accuracy the prognostics
  literature uses: the share of predictions landing inside a tolerance band
  around the truth.
* **Ranking.** Spearman correlation. Triage is a sorting problem, and a model
  can rank a fleet correctly while being biased on every single cell.
* **Interval quality.** Coverage against the width it cost.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

#: Tolerance for the alpha-lambda metric: within 20% of true remaining life.
ALPHA_TOLERANCE = 0.20


def _to_cycles(log_cycles: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(log_cycles, dtype=float))


def regression_metrics(log_truth: np.ndarray, log_prediction: np.ndarray) -> dict[str, float]:
    """The full scorecard for one set of predictions."""
    log_truth = np.asarray(log_truth, dtype=float)
    log_prediction = np.asarray(log_prediction, dtype=float)
    if log_truth.shape != log_prediction.shape:
        raise ValueError(f"shape mismatch: {log_truth.shape} against {log_prediction.shape}")
    if log_truth.size == 0:
        raise ValueError("no predictions to score")

    residual = log_prediction - log_truth
    truth_cycles = _to_cycles(log_truth)
    prediction_cycles = _to_cycles(log_prediction)
    cycle_residual = prediction_cycles - truth_cycles

    relative = np.abs(cycle_residual) / np.maximum(truth_cycles, 1.0)

    # Spearman is undefined when either side is constant. The prediction side is
    # exactly what the fleet_mean baseline produces, and the truth side turns up
    # on a fold where every held-out cell happens to share a label. Report zero
    # rather than a NaN that then propagates into every aggregate in the table.
    constant = np.allclose(log_prediction, log_prediction[0]) or np.allclose(
        log_truth, log_truth[0]
    )
    if constant or log_truth.size < 3:
        rank_correlation = 0.0
    else:
        rank_correlation = float(spearmanr(log_truth, log_prediction).statistic)

    return {
        "n": int(log_truth.size),
        "rmse_log": float(np.sqrt(np.mean(residual**2))),
        "mae_log": float(np.mean(np.abs(residual))),
        "rmse_cycles": float(np.sqrt(np.mean(cycle_residual**2))),
        "mae_cycles": float(np.mean(np.abs(cycle_residual))),
        "mape": float(np.mean(relative)),
        "alpha_lambda": float(np.mean(relative <= ALPHA_TOLERANCE)),
        "bias_cycles": float(np.mean(cycle_residual)),
        "spearman": rank_correlation,
    }


def interval_metrics(
    log_truth: np.ndarray,
    log_lower: np.ndarray,
    log_upper: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Coverage and width, plus the shortfall against the advertised level.

    ``coverage_gap`` is the number that matters. A negative value means the
    interval promised more than it delivered, which is worse than a wide
    interval: a wide interval is visibly cautious, and an undercovering one is
    invisibly wrong.
    """
    log_truth = np.asarray(log_truth, dtype=float)
    log_lower = np.asarray(log_lower, dtype=float)
    log_upper = np.asarray(log_upper, dtype=float)

    inside = (log_truth >= log_lower) & (log_truth <= log_upper)
    picp = float(np.mean(inside))
    nominal = 1.0 - alpha

    lower_cycles = _to_cycles(log_lower)
    upper_cycles = _to_cycles(log_upper)

    return {
        "nominal_coverage": float(nominal),
        "picp": picp,
        "coverage_gap": float(picp - nominal),
        "mean_width_log": float(np.mean(log_upper - log_lower)),
        "median_width_cycles": float(np.median(upper_cycles - lower_cycles)),
    }


def aggregate(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation across folds.

    Weighted by nothing: each fold is one observation of the protocol, and a
    fold with more test cells is not a more reliable estimate of the *protocol's*
    behaviour, only of that fold's. The standard deviation is reported alongside
    because on the harder protocols it is large enough to matter.
    """
    if not fold_metrics:
        raise ValueError("nothing to aggregate")

    keys = [key for key in fold_metrics[0] if key != "n"]
    out: dict[str, float] = {"n_folds": len(fold_metrics)}
    out["n"] = int(sum(int(fold.get("n", 0)) for fold in fold_metrics))
    for key in keys:
        values = np.asarray([fold[key] for fold in fold_metrics], dtype=float)
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=0))
    return out
