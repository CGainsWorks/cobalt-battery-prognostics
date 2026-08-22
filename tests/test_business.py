"""The cost models, where a prediction becomes a decision and a bill.

These are the tests with closed-form answers, so they check arithmetic against
theory rather than against a stored output. The newsvendor optimum in particular
is a textbook result and a numerical search over the same cost function has to
land on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cobalt.business.cost import (
    ReplacementCostModel,
    compare_policies,
    qualification_time_saving,
    score_policy,
)
from cobalt.business.grading import (
    GRADES,
    GradingPolicy,
    constant_grade_baseline,
    evaluate_grading,
    grade_from_prediction,
    minimum_expected_cost_grade,
)

# -- replacement scheduling ---------------------------------------------


def test_the_optimal_quantile_is_the_newsvendor_ratio():
    model = ReplacementCostModel(cost_per_cycle_early=2.40, cost_per_cycle_late=14.00)
    assert model.optimal_quantile == pytest.approx(2.40 / 16.40)


def test_a_numerical_search_finds_the_newsvendor_quantile():
    """The theory says q* = c_early / (c_early + c_late). Check it empirically."""
    model = ReplacementCostModel(cost_per_cycle_early=3.0, cost_per_cycle_late=12.0)
    rng = np.random.default_rng(0)
    truth = rng.lognormal(mean=np.log(600.0), sigma=0.5, size=40_000)

    candidates = np.quantile(truth, np.linspace(0.02, 0.60, 59))
    costs = [float(np.mean(model.cost(np.full(truth.shape, d), truth))) for d in candidates]
    best = candidates[int(np.argmin(costs))]
    expected = float(np.quantile(truth, model.optimal_quantile))

    assert best == pytest.approx(expected, rel=0.06)


def test_being_late_costs_more_than_being_early():
    model = ReplacementCostModel(cost_per_cycle_early=2.0, cost_per_cycle_late=10.0)
    truth = np.array([100.0])
    late = float(model.cost(np.array([150.0]), truth)[0])
    early = float(model.cost(np.array([50.0]), truth)[0])
    assert late == pytest.approx(500.0)
    assert early == pytest.approx(100.0)
    assert late > early


def test_a_perfect_booking_costs_only_the_fixed_charge():
    model = ReplacementCostModel(fixed_replacement_cost=40.0)
    truth = np.array([100.0, 300.0])
    assert np.allclose(model.cost(truth, truth), 40.0)


def test_cost_model_rejects_negative_prices():
    with pytest.raises(ValueError, match="positive"):
        ReplacementCostModel(cost_per_cycle_early=0.0)


def test_cost_model_rejects_mismatched_shapes():
    model = ReplacementCostModel()
    with pytest.raises(ValueError, match="shape mismatch"):
        model.cost(np.array([1.0, 2.0]), np.array([1.0]))


def test_score_policy_reports_how_often_it_ran_late():
    model = ReplacementCostModel()
    result = score_policy(
        "test", np.array([50.0, 150.0, 250.0]), np.array([100.0, 100.0, 100.0]), model
    )
    assert result.late_fraction == pytest.approx(2 / 3)
    assert result.n_packs == 3
    assert "late_fraction" in result.as_dict()


def test_the_oracle_is_the_floor():
    model = ReplacementCostModel()
    rng = np.random.default_rng(1)
    log_truth = np.log10(rng.uniform(100, 900, 300))
    log_prediction = log_truth + rng.normal(0, 0.08, 300)

    results = {
        result.policy: result.cost_per_pack
        for result in compare_policies(
            log_truth,
            log_prediction,
            model,
            log_lower_bound=log_prediction - 0.12,
            training_log_truth=log_truth,
            alpha=0.10,
        )
    }
    assert results["oracle"] == pytest.approx(0.0)
    for policy, cost in results.items():
        assert cost >= results["oracle"] - 1e-9, policy


def test_the_newsvendor_policy_beats_the_point_estimate_under_asymmetric_costs():
    """The whole argument of cobalt.business.cost, as an assertion."""
    model = ReplacementCostModel(cost_per_cycle_early=2.40, cost_per_cycle_late=14.00)
    rng = np.random.default_rng(2)
    log_truth = np.log10(rng.uniform(200, 1400, 4000))
    log_prediction = log_truth + rng.normal(0, 0.10, 4000)

    results = {
        result.policy: result.cost_per_pack
        for result in compare_policies(
            log_truth, log_prediction, model, log_lower_bound=log_prediction - 0.164, alpha=0.10
        )
    }
    assert results["newsvendor_quantile"] < results["point_estimate"]


def test_qualification_saving_is_the_fraction_of_the_test_skipped():
    saving = qualification_time_saving(
        median_cycle_life=1000.0, observation_cycles=100, hours_per_cycle=2.0
    )
    assert saving["fraction_saved"] == pytest.approx(0.9)
    assert saving["full_test_days"] == pytest.approx(1000 * 2 / 24, rel=1e-3)


def test_more_channels_shorten_both_tests_equally():
    one = qualification_time_saving(1000.0, 100, 2.0, channels=1)
    eight = qualification_time_saving(1000.0, 100, 2.0, channels=8)
    # The function rounds to a tenth of a day, so the tolerance has to clear that.
    assert eight["full_test_days"] == pytest.approx(one["full_test_days"] / 8, rel=5e-3)
    assert eight["fraction_saved"] == pytest.approx(one["fraction_saved"])


def test_qualification_rejects_nonsense():
    with pytest.raises(ValueError, match="positive"):
        qualification_time_saving(0.0, 100, 2.0)
    with pytest.raises(ValueError, match="channels"):
        qualification_time_saving(1000.0, 100, 2.0, channels=0)


# -- lot acceptance -----------------------------------------------------


@pytest.fixture
def policy() -> GradingPolicy:
    return GradingPolicy(grade_a_min_cycles=1200, grade_b_min_cycles=700)


def test_grade_boundaries_are_applied_at_the_right_places(policy):
    grades = policy.grade_of(np.array([699, 700, 1199, 1200, 5000]))
    assert list(grades) == [2, 1, 1, 0, 0]


def test_grading_policy_rejects_inverted_boundaries():
    with pytest.raises(ValueError, match="must exceed"):
        GradingPolicy(grade_a_min_cycles=100, grade_b_min_cycles=900)


def test_grading_policy_rejects_a_wrong_sized_matrix():
    with pytest.raises(ValueError, match="3x3"):
        GradingPolicy(cost_matrix=((0.0, 1.0), (1.0, 0.0)))


def test_correct_grading_is_free(policy):
    truth = np.array([1500, 900, 400])
    result = evaluate_grading(truth, policy.grade_of(truth), policy)
    assert result.total_cost == 0.0
    assert result.accuracy == 1.0
    assert result.overgraded == 0 and result.undergraded == 0


def test_overgrading_costs_more_than_undergrading(policy):
    """A short-lived cell in a premium slot is the expensive error."""
    truth = np.array([400])
    overgraded = evaluate_grading(truth, np.array([0]), policy)
    truth_good = np.array([1500])
    undergraded = evaluate_grading(truth_good, np.array([2]), policy)
    assert overgraded.cost_per_cell > undergraded.cost_per_cell


def test_confusion_matrix_totals_match_the_cell_count(policy):
    truth = np.array([1500, 900, 400, 1300, 800])
    result = evaluate_grading(truth, np.array([0, 1, 2, 1, 0]), policy)
    assert sum(sum(row) for row in result.confusion) == len(truth)
    assert len(result.confusion) == len(GRADES)


def test_grading_rejects_mismatched_shapes(policy):
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_grading(np.array([1000, 500]), np.array([0]), policy)


def test_the_lower_bound_rule_is_more_pessimistic_than_the_point_estimate(policy):
    log_prediction = np.log10(np.array([1250.0, 1250.0, 1250.0]))
    log_lower = np.log10(np.array([900.0, 900.0, 900.0]))
    from_point = grade_from_prediction(log_prediction, policy)
    from_bound = grade_from_prediction(log_prediction, policy, log_lower_bound=log_lower)
    assert list(from_point) == [0, 0, 0]
    assert list(from_bound) == [1, 1, 1]


def test_expected_cost_grading_matches_the_point_estimate_when_certain(policy):
    log_prediction = np.log10(np.array([1500.0, 900.0, 400.0]))
    assert list(minimum_expected_cost_grade(log_prediction, 0.0, policy)) == [0, 1, 2]


def test_uncertainty_pushes_a_borderline_cell_down_a_grade(policy):
    """Just over the A boundary, with enough spread to be a C, should not be an A."""
    log_prediction = np.log10(np.array([1210.0]))
    confident = minimum_expected_cost_grade(log_prediction, 0.001, policy)
    uncertain = minimum_expected_cost_grade(log_prediction, 0.25, policy)
    assert confident[0] == 0
    assert uncertain[0] > confident[0]


def test_expected_cost_grading_beats_the_point_estimate_on_average(policy):
    """The rule the cost matrix implies must actually be cheaper than ignoring it."""
    rng = np.random.default_rng(3)
    truth = rng.uniform(300, 2000, 6000)
    sigma = 0.12
    log_prediction = np.log10(truth) + rng.normal(0, sigma, truth.size)

    point = evaluate_grading(truth, grade_from_prediction(log_prediction, policy), policy)
    expected = evaluate_grading(
        truth, minimum_expected_cost_grade(log_prediction, sigma, policy), policy
    )
    assert expected.cost_per_cell < point.cost_per_cell


def test_the_do_nothing_baseline_grades_everything_b(policy):
    truth = np.array([1500, 900, 400])
    result = constant_grade_baseline(truth, policy)
    assert result.confusion[0][1] == 1
    assert result.confusion[1][1] == 1
    assert result.confusion[2][1] == 1
