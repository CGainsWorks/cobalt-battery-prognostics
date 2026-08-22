"""Maintenance scheduling: when do you book the replacement?

The online model predicts remaining useful life. The decision it feeds is a
date in a workshop diary and the cost of that decision is asymmetric in a way
that has a closed-form answer nobody uses.

Book the replacement too late and the pack runs past end of life: derated
range, elevated failure risk, warranty exposure. Book it too early and you throw
away usable cycles you have already paid for. Write those as linear costs per
cycle and the expected cost is the pinball loss,

    L(d) = c_late * max(0, d - T) + c_early * max(0, T - d)

whose minimiser is not the mean prediction and not the median prediction. It is
the quantile of the predictive distribution at

    q* = c_early / (c_early + c_late)

With the defaults here that is around the fifteenth percentile, which says the
booking should sit well below the model's central estimate. This is the
newsvendor problem and a fleet that schedules on the point estimate is
answering a question nobody asked.

That is why the conformal interval is not decoration. The lower bound is an
approximation to the quantile the cost function actually wants and the
comparison between the two policies is the clearest single number in the
repository: same model, same predictions, different reading of them, materially
different bill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cobalt.models.conformal import quantile_from_interval


@dataclass(frozen=True)
class ReplacementCostModel:
    """Cost per cycle of being wrong in each direction.

    Defaults describe a small commercial pack: roughly two and a half thousand
    pounds, a nominal thousand-cycle life, so a cycle of unused residual life is
    worth a couple of pounds. Running past end of life is charged at the
    expected cost of an unplanned failure per cycle of exposure plus the
    operational penalty of a derated pack.

    Illustrative figures. They set the shape of the answer, not its truth.
    """

    #: Pounds per cycle of usable life discarded by replacing early.
    cost_per_cycle_early: float = 2.40
    #: Pounds per cycle the pack runs beyond end of life.
    cost_per_cycle_late: float = 14.00
    #: Fixed cost of the replacement itself. Identical under every policy, so it
    #: cancels in comparisons and it is here to keep the absolute figures honest.
    fixed_replacement_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_per_cycle_early <= 0 or self.cost_per_cycle_late <= 0:
            raise ValueError("both per-cycle costs must be positive")

    @property
    def optimal_quantile(self) -> float:
        """The quantile of remaining life the booking should be made at."""
        return self.cost_per_cycle_early / (self.cost_per_cycle_early + self.cost_per_cycle_late)

    def cost(self, booked_cycles: np.ndarray, true_remaining: np.ndarray) -> np.ndarray:
        """Pounds per pack for a set of bookings against what actually happened."""
        booked = np.asarray(booked_cycles, dtype=float)
        truth = np.asarray(true_remaining, dtype=float)
        if booked.shape != truth.shape:
            raise ValueError(f"shape mismatch: {booked.shape} against {truth.shape}")

        late = np.maximum(0.0, booked - truth)
        early = np.maximum(0.0, truth - booked)
        return (
            self.cost_per_cycle_late * late
            + self.cost_per_cycle_early * early
            + self.fixed_replacement_cost
        )


@dataclass
class PolicyResult:
    """What one booking policy cost on one test set."""

    policy: str
    cost_per_pack: float
    total_cost: float
    n_packs: int
    mean_booked_cycles: float
    late_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "cost_per_pack": round(self.cost_per_pack, 2),
            "total_cost": round(self.total_cost, 2),
            "n_packs": self.n_packs,
            "mean_booked_cycles": round(self.mean_booked_cycles, 1),
            "late_fraction": round(self.late_fraction, 4),
        }


def score_policy(
    name: str,
    booked_cycles: np.ndarray,
    true_remaining: np.ndarray,
    model: ReplacementCostModel,
) -> PolicyResult:
    """Apply the cost model to one policy's bookings."""
    booked = np.maximum(0.0, np.asarray(booked_cycles, dtype=float))
    truth = np.asarray(true_remaining, dtype=float)
    costs = model.cost(booked, truth)
    return PolicyResult(
        policy=name,
        cost_per_pack=float(np.mean(costs)),
        total_cost=float(np.sum(costs)),
        n_packs=int(truth.size),
        mean_booked_cycles=float(np.mean(booked)),
        late_fraction=float(np.mean(booked > truth)),
    )


def compare_policies(
    log_truth: np.ndarray,
    log_prediction: np.ndarray,
    model: ReplacementCostModel,
    log_lower_bound: np.ndarray | None = None,
    training_log_truth: np.ndarray | None = None,
    alpha: float = 0.10,
) -> list[PolicyResult]:
    """Score every booking policy on the same predictions.

    ``oracle`` is the floor and is unreachable: it books on the true remaining
    life and therefore costs nothing but the fixed charge. It is here so the
    other policies can be read as a fraction of the value available rather than
    as bare pounds.
    """
    truth = np.power(10.0, np.asarray(log_truth, dtype=float))
    prediction = np.power(10.0, np.asarray(log_prediction, dtype=float))

    results = [
        score_policy("oracle", truth, truth, model),
        score_policy("point_estimate", prediction, truth, model),
    ]

    if log_lower_bound is not None:
        lower = np.power(10.0, np.asarray(log_lower_bound, dtype=float))
        results.append(score_policy("conformal_lower_bound", lower, truth, model))

        # The policy the cost function actually asks for. The 90% lower bound is
        # a fixed level chosen for readability; the newsvendor quantile is
        # derived from the two costs and with these figures it sits well above
        # the bound. Booking on the bound is cautious in the right direction and
        # still the wrong amount.
        half_width = float(
            np.mean(np.asarray(log_prediction, dtype=float) - np.asarray(log_lower_bound))
        )
        booking = quantile_from_interval(
            np.asarray(log_prediction, dtype=float),
            half_width,
            alpha,
            model.optimal_quantile,
        )
        results.append(score_policy("newsvendor_quantile", np.power(10.0, booking), truth, model))

    if training_log_truth is not None and len(training_log_truth) > 0:
        # No features at all: book everything at the cost-optimal quantile of the
        # training label distribution. If a model cannot beat this, the features
        # are not paying for themselves.
        constant = float(
            np.quantile(
                np.power(10.0, np.asarray(training_log_truth, dtype=float)),
                model.optimal_quantile,
            )
        )
        results.append(score_policy("fleet_quantile", np.full(truth.shape, constant), truth, model))

    return results


def qualification_time_saving(
    median_cycle_life: float,
    observation_cycles: int,
    hours_per_cycle: float,
    channels: int = 1,
) -> dict[str, float]:
    """Test-channel time saved by grading early instead of cycling to end of life.

    This is the part of the business case that does not depend on the model being
    good, only on it being usable: the alternative to a hundred-cycle test is a
    test that runs until the cell dies and cycler channels are the constrained
    resource in every battery laboratory.
    """
    if observation_cycles <= 0 or median_cycle_life <= 0 or hours_per_cycle <= 0:
        raise ValueError("all durations must be positive")
    if channels < 1:
        raise ValueError("channels must be at least one")

    full_hours = median_cycle_life * hours_per_cycle / channels
    early_hours = observation_cycles * hours_per_cycle / channels
    return {
        "full_test_days": round(full_hours / 24.0, 1),
        "early_test_days": round(early_hours / 24.0, 1),
        "days_saved": round((full_hours - early_hours) / 24.0, 1),
        "fraction_saved": round(1.0 - early_hours / full_hours, 4),
    }
