"""Prediction intervals and how to get them wrong.

A point estimate of remaining life is not a decision. "About 900 cycles" and
"between 300 and 1500 cycles" carry the same number and imply completely
different amounts of spare stock, so the interval is the deliverable and the
point estimate is a summary of it.

Split conformal prediction is used here because it makes one assumption,
exchangeability between the calibration set and the test set and because that
assumption is precisely the one this repository is about. Calibrate on cells
drawn at random from batches the model has already seen and the residuals are
smaller than they will be on a new lot, so the interval comes out too narrow and
the coverage it advertises is not the coverage it delivers.

So :class:`ConformalRegressor` takes a ``groups`` argument and holds out whole
groups for calibration when it is given one. `cobalt.evaluate.protocol` runs it
both ways and reports the coverage gap, which is the concrete cost of a
mismatched calibration set.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

#: Groups that must remain in the fit after grouped calibration has taken its
#: share. Below this the fit is starved of between-group variety and the
#: grouped-against-random comparison measures the starvation rather than the
#: calibration. See docs/EVALUATION_PROTOCOL.md.
MIN_FIT_GROUPS = 4


class ConformalRegressor:
    """Split-conformal wrapper around any model with ``fit`` and ``predict``.

    Parameters
    ----------
    base:
        The underlying regressor. Refitted on the proper training subset only,
        so the calibration residuals are genuinely out of sample.
    alpha:
        Miscoverage rate. ``0.1`` asks for 90% intervals.
    calibration_fraction:
        Share of the training data reserved for calibration.
    grouped:
        When true, calibration holds out whole groups rather than random rows.
        Set this to match the evaluation protocol or the intervals will be
        confident about the wrong thing.
    """

    supports_uncertainty = True

    def __init__(
        self,
        base,
        alpha: float = 0.10,
        calibration_fraction: float = 0.30,
        grouped: bool = True,
        seed: int = 0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0.0 < calibration_fraction < 1.0:
            raise ValueError(f"calibration_fraction must be in (0, 1), got {calibration_fraction}")
        self.base = base
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.grouped = grouped
        self.seed = seed
        self.name = f"{base.name}+conformal" + ("" if grouped else "[random-cal]")
        self.quantile = float("nan")
        self.n_calibration = 0
        self.n_calibration_groups = 0
        self.n_fit_groups = 0
        self.group_warning: str | None = None

    def _calibration_mask(self, n: int, groups: np.ndarray | None) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        mask = np.zeros(n, dtype=bool)

        if self.grouped and groups is not None:
            unique = np.unique(groups)
            n_held = max(1, int(round(self.calibration_fraction * len(unique))))
            if n_held >= len(unique):
                # Every group in calibration leaves nothing to fit on. Fall back
                # to random rows and record it in the name so it shows up in the
                # results rather than only in a log line.
                self.name += "[degraded-to-random-cal]"
                return self._calibration_mask(n, None)
            held = rng.choice(unique, size=n_held, replace=False)
            mask[np.isin(groups, held)] = True

            self.n_calibration_groups = n_held
            self.n_fit_groups = len(unique) - n_held
            if self.n_fit_groups < MIN_FIT_GROUPS:
                # Grouped calibration is not free. It takes whole groups out of
                # the fit and when there are only a handful to begin with the
                # model loses more from the missing diversity than the
                # calibration gains from being out of group. Below this many
                # groups the comparison between grouped and random calibration
                # stops measuring what it is meant to measure, so say so rather
                # than let it be read as a result.
                self.group_warning = (
                    f"only {self.n_fit_groups} groups left to fit on after holding out "
                    f"{n_held} for calibration; grouped calibration needs at least "
                    f"{MIN_FIT_GROUPS} to be worth doing"
                )
            return mask

        n_held = max(2, int(round(self.calibration_fraction * n)))
        mask[rng.choice(n, size=min(n_held, n - 1), replace=False)] = True
        return mask

    def fit(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> ConformalRegressor:
        target = np.asarray(target, dtype=float)
        mask = self._calibration_mask(len(features), groups)

        proper = features.iloc[~mask]
        calibration = features.iloc[mask]
        if len(proper) < 2 or len(calibration) < 2:
            raise ValueError(
                f"calibration split left {len(proper)} training and "
                f"{len(calibration)} calibration rows, which is not enough to fit"
            )

        self.base.fit(proper, target[~mask])
        residuals = np.abs(target[mask] - self.base.predict(calibration))

        # The finite-sample conformal quantile. The (n+1) is not a rounding
        # detail: without it the interval undercovers on small calibration sets,
        # which is exactly the regime a held-out-batch calibration lands in.
        n = len(residuals)
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile = float(np.quantile(residuals, level, method="higher"))
        self.n_calibration = n
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.base.predict(features)

    def predict_interval(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds, in whatever units the target was fitted in."""
        if not math.isfinite(self.quantile):
            raise RuntimeError("predict_interval called before fit")
        centre = self.predict(features)
        return centre - self.quantile, centre + self.quantile


def implied_sigma(half_width: float, alpha: float) -> float:
    """Turn a symmetric conformal half-width into a Gaussian scale.

    Conformal prediction gives one interval at one level and says nothing about
    the shape of the distribution inside it. The cost models need an arbitrary
    quantile, not the 90% bound, so something has to bridge the gap.

    The bridge here is the crudest defensible one: assume the residuals are
    roughly Gaussian in log space and read off the scale that would produce this
    half-width at this level. That assumption is wrong in the tails and it is
    stated wherever the result is used. It is still a large improvement on the
    alternative in common use, which is to pretend the point estimate is the
    whole answer.

    A conformalised quantile regressor would remove the assumption entirely and
    `docs/LIMITATIONS.md` lists it as the obvious next piece of work.
    """
    if half_width <= 0.0:
        return 0.0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return half_width / max(z, 1e-9)


def quantile_from_interval(
    centre: np.ndarray,
    half_width: float,
    alpha: float,
    quantile: float,
) -> np.ndarray:
    """The ``quantile``-th percentile of the implied predictive distribution."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    sigma = implied_sigma(half_width, alpha)
    return np.asarray(centre, dtype=float) + float(norm.ppf(quantile)) * sigma


def coverage(lower: np.ndarray, upper: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of truths inside the interval. Compare against ``1 - alpha``."""
    inside = (truth >= lower) & (truth <= upper)
    return float(np.mean(inside))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average width. Coverage without width is free, so the two travel together."""
    return float(np.mean(upper - lower))
