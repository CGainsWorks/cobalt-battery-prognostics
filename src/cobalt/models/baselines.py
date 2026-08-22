"""The baselines a learned model has to beat to be worth deploying.

Two of them and they are not straw men.

``capacity_extrapolation``
    What a competent engineer does with a spreadsheet: fit the observed capacity
    fade, extend the line, read off where it crosses the end-of-life threshold.
    It needs no training data at all, which is a real advantage when a new
    chemistry arrives. It fails for a specific and interesting reason and the
    failure is the argument for the whole repository: fade over the first hundred
    cycles is nearly linear and nearly flat, so the extrapolation is a long lever
    on a small, noisy slope and it is blind to the knee entirely.

``fleet_mean``
    Predict the training mean and ignore the features. Any model that cannot
    beat this has learned nothing and on the harder protocols that is a live
    possibility rather than a formality.

Both implement the same two-method interface as the learned models, so the
evaluation loop does not need to know which kind it is holding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FleetMean:
    """Predict the mean of the training labels. The floor."""

    name = "fleet_mean"
    supports_uncertainty = False

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> FleetMean:
        self._mean = float(np.mean(target))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self._mean)


class CapacityExtrapolation:
    """Extrapolate the early capacity fade to the end-of-life threshold.

    Fitting happens only to pick up the fleet's median cycle life, which is used
    as the fallback when the observed slope is non-negative. A cell whose
    capacity has not measurably fallen in a hundred cycles gives a slope of
    roughly zero and dividing by it produces an infinite life rather than an
    error, which is precisely the sort of thing that reaches production.
    """

    name = "capacity_extrapolation"
    supports_uncertainty = False

    #: How far past the longest life in training the extrapolation may reach,
    #: in decades. 0.1 dex is about 26%.
    HORIZON_MARGIN_DEX = 0.1

    def __init__(
        self,
        rated_capacity_ah: float,
        end_of_life_soh: float = 0.80,
        observation_cycles: int = 100,
        reference_cycle: int = 2,
    ) -> None:
        self.rated_capacity_ah = rated_capacity_ah
        self.end_of_life_soh = end_of_life_soh
        self.observation_cycles = observation_cycles
        self.reference_cycle = reference_cycle
        self._fallback_log_life = np.log10(800.0)
        self._cap_log_life = np.log10(20_000.0)

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> CapacityExtrapolation:
        self._fallback_log_life = float(np.median(target))
        # A planning horizon. Extrapolating a slope measured over ninety-eight
        # cycles out to twenty thousand is arithmetic, not a prediction and
        # letting it through would flatter every other model in the table by
        # comparison. A real engineer would say "beyond the horizon" and stop, so
        # the baseline does too.
        self._cap_log_life = float(np.max(target)) + self.HORIZON_MARGIN_DEX
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        capacity_now = features["capacity_cycle_100"].to_numpy(dtype=float)
        fade = features["capacity_fade_2_to_100"].to_numpy(dtype=float)
        span = float(self.observation_cycles - self.reference_cycle)

        fade_per_cycle = fade / span
        threshold = self.end_of_life_soh * self.rated_capacity_ah
        headroom = capacity_now - threshold

        with np.errstate(divide="ignore", invalid="ignore"):
            remaining = headroom / fade_per_cycle
        life = self.observation_cycles + remaining

        # A flat or rising fade curve gives a nonsensical life. Fall back to the
        # fleet median rather than emitting an infinity that poisons the metric.
        usable = np.isfinite(life) & (fade_per_cycle > 0) & (life > self.observation_cycles)
        out = np.where(usable, life, 10.0**self._fallback_log_life)
        return np.clip(np.log10(np.clip(out, 1.0, None)), 0.0, self._cap_log_life)


class SlopeExtrapolation:
    """The online-task equivalent: extend the recent fade slope to the threshold.

    Same idea and the same weakness but with a better vantage point, because by
    mid-life the slope is large enough to measure. It is a genuinely strong
    baseline late in a pack's life and a poor one early and the evaluation
    reports the two separately for exactly that reason.
    """

    name = "slope_extrapolation"
    supports_uncertainty = False

    #: Same planning horizon as the early-life extrapolation and for the same
    #: reason: early in life the fade slope is barely distinguishable from zero
    #: and dividing by it produces a number with no information in it.
    HORIZON_MARGIN_DEX = 0.1

    def __init__(self, end_of_life_soh: float = 0.80) -> None:
        self.end_of_life_soh = end_of_life_soh
        self._fallback_log_remaining = np.log10(300.0)
        self._cap_log_remaining = np.log10(20_000.0)

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> SlopeExtrapolation:
        self._fallback_log_remaining = float(np.median(target))
        self._cap_log_remaining = float(np.max(target)) + self.HORIZON_MARGIN_DEX
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        soh_now = features["soh_now"].to_numpy(dtype=float)
        slope = features["soh_slope"].to_numpy(dtype=float)
        headroom = soh_now - self.end_of_life_soh

        with np.errstate(divide="ignore", invalid="ignore"):
            remaining = headroom / (-slope)

        usable = np.isfinite(remaining) & (slope < 0) & (remaining > 0)
        out = np.where(usable, remaining, 10.0**self._fallback_log_remaining)
        return np.clip(np.log10(np.clip(out, 1.0, None)), 0.0, self._cap_log_remaining)
