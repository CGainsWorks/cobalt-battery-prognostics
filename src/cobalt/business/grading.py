"""Lot acceptance: what grade does this cell get, and what does getting it wrong cost?

The early-life model exists to support one decision. A lot of cells arrives,
each one gets a grade, and the grade decides where it goes: a premium
application that needs long life, a standard one, or the second-life channel.

The costs are wildly asymmetric and that asymmetry is the entire point.
Undergrading a good cell loses the margin between two price bands, which is
tens of pounds. Overgrading a poor one puts a cell that will not last into an
application that assumed it would, and the bill for that arrives as a field
replacement with labour and downtime attached.

A model selected on RMSE does not know any of this. Two models with identical
RMSE can differ by a factor of three in cost per cell, depending on which
direction they err in, and this module is what makes that visible.

The grade boundaries and the cost figures are illustrative, chosen to be the
right order of magnitude for a small commercial pack. They are configuration,
not findings, and `config/default.yaml` is where they live.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRADES = ("A", "B", "C")


@dataclass(frozen=True)
class GradingPolicy:
    """Cycle-life boundaries between grades, and what misgrading costs.

    ``cost_matrix`` is indexed ``[true_grade][assigned_grade]`` in the order of
    :data:`GRADES`. The diagonal is zero by definition: grading a cell correctly
    costs nothing beyond the test that produced the grade.
    """

    #: Minimum cycle life for grade A, then for grade B. Below both is grade C.
    grade_a_min_cycles: int = 1200
    grade_b_min_cycles: int = 700

    cost_matrix: tuple[tuple[float, ...], ...] = (
        # true A: sold below its worth
        (0.0, 55.0, 140.0),
        # true B: promoted into an application it cannot serve, or demoted
        (260.0, 0.0, 85.0),
        # true C: the expensive corner, a short-lived cell in a premium slot
        (620.0, 310.0, 0.0),
    )

    def __post_init__(self) -> None:
        if self.grade_a_min_cycles <= self.grade_b_min_cycles:
            raise ValueError(
                f"grade A threshold ({self.grade_a_min_cycles}) must exceed "
                f"grade B ({self.grade_b_min_cycles})"
            )
        if len(self.cost_matrix) != len(GRADES):
            raise ValueError(f"cost matrix must be {len(GRADES)}x{len(GRADES)}")
        for row in self.cost_matrix:
            if len(row) != len(GRADES):
                raise ValueError(f"cost matrix must be {len(GRADES)}x{len(GRADES)}")

    def grade_of(self, cycle_life: np.ndarray) -> np.ndarray:
        """Grade index for each cycle life. 0 is A, 2 is C."""
        cycle_life = np.asarray(cycle_life, dtype=float)
        grade = np.full(cycle_life.shape, 2, dtype=int)
        grade[cycle_life >= self.grade_b_min_cycles] = 1
        grade[cycle_life >= self.grade_a_min_cycles] = 0
        return grade


@dataclass
class GradingResult:
    """Outcome of grading one set of cells with one model."""

    cost_per_cell: float
    total_cost: float
    n_cells: int
    accuracy: float
    overgraded: int
    undergraded: int
    confusion: list[list[int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "cost_per_cell": round(self.cost_per_cell, 2),
            "total_cost": round(self.total_cost, 2),
            "n_cells": self.n_cells,
            "accuracy": round(self.accuracy, 4),
            "overgraded": self.overgraded,
            "undergraded": self.undergraded,
            "confusion": self.confusion,
        }


def grade_from_prediction(
    log_predicted_life: np.ndarray,
    policy: GradingPolicy,
    log_lower_bound: np.ndarray | None = None,
) -> np.ndarray:
    """Assign grades from model output.

    When a conformal lower bound is supplied the grade is taken from *that*
    rather than from the point estimate. It is the one-line change that turns a
    regression into a decision procedure that respects the cost asymmetry:
    overgrading is the expensive error, so grade on the pessimistic end of the
    interval and stop paying for it.
    """
    basis = log_predicted_life if log_lower_bound is None else log_lower_bound
    return policy.grade_of(np.power(10.0, np.asarray(basis, dtype=float)))


def minimum_expected_cost_grade(
    log_predicted_life: np.ndarray,
    sigma_log: float,
    policy: GradingPolicy,
) -> np.ndarray:
    """Assign the grade with the lowest expected cost under the cost matrix.

    This is the rule the cost matrix actually implies, and it is worth spelling
    out why neither of the obvious alternatives is it.

    The **point estimate** rule reads the predicted life and looks up the grade.
    It behaves as though the prediction were certain, so it happily assigns
    grade A to a cell that is one standard deviation from being a C, and the
    cost matrix charges six hundred pounds for that.

    The **lower bound** rule grades on the pessimistic end of the interval. It
    fixes the overgrading and then overshoots, because undergrading is not free
    either: at a 90% bound almost everything lands in C and the lost margin adds
    up to more than the failures it prevented. That is not a hypothetical, it is
    what the results table in this repository shows.

    What the matrix actually asks for is an expectation. Take the predictive
    distribution, integrate the probability of each true grade, and pick the
    assignment minimising expected cost. With three grades and a Gaussian in log
    space it is one line of arithmetic per cell, and it dominates both.
    """
    from scipy.stats import norm

    centre = np.asarray(log_predicted_life, dtype=float)
    boundaries = np.log10([policy.grade_b_min_cycles, policy.grade_a_min_cycles])

    if sigma_log <= 0.0:
        return grade_from_prediction(centre, policy)

    # P(true grade == C), == B, == A, from the two boundaries in log space.
    below_b = norm.cdf((boundaries[0] - centre) / sigma_log)
    below_a = norm.cdf((boundaries[1] - centre) / sigma_log)
    probability = np.stack([1.0 - below_a, below_a - below_b, below_b], axis=1)

    costs = np.asarray(policy.cost_matrix, dtype=float)
    expected = probability @ costs
    return np.asarray(np.argmin(expected, axis=1), dtype=int)


def evaluate_grading(
    true_cycle_life: np.ndarray,
    assigned_grade: np.ndarray,
    policy: GradingPolicy,
) -> GradingResult:
    """Score an assignment against the grades the cells actually earned."""
    true_grade = policy.grade_of(true_cycle_life)
    assigned_grade = np.asarray(assigned_grade, dtype=int)
    if true_grade.shape != assigned_grade.shape:
        raise ValueError(f"shape mismatch: {true_grade.shape} against {assigned_grade.shape}")

    costs = np.asarray(policy.cost_matrix, dtype=float)
    per_cell = costs[true_grade, assigned_grade]

    confusion = np.zeros((len(GRADES), len(GRADES)), dtype=int)
    for actual, predicted in zip(true_grade, assigned_grade):
        confusion[actual, predicted] += 1

    # A lower grade index is a better grade, so assigning an index below the
    # truth is claiming the cell is better than it is.
    overgraded = int(np.sum(assigned_grade < true_grade))
    undergraded = int(np.sum(assigned_grade > true_grade))

    return GradingResult(
        cost_per_cell=float(np.mean(per_cell)),
        total_cost=float(np.sum(per_cell)),
        n_cells=int(true_grade.size),
        accuracy=float(np.mean(assigned_grade == true_grade)),
        overgraded=overgraded,
        undergraded=undergraded,
        confusion=confusion.tolist(),
    )


def constant_grade_baseline(true_cycle_life: np.ndarray, policy: GradingPolicy) -> GradingResult:
    """Grade every cell B and never test anything. The do-nothing option.

    Worth reporting because on a fleet with a healthy median it is not a
    disaster, and any model that costs more than this is worse than no model.
    """
    assigned = np.full(np.asarray(true_cycle_life).shape, 1, dtype=int)
    return evaluate_grading(true_cycle_life, assigned, policy)
