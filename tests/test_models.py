"""Models, baselines and conformal calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobalt.models.baselines import CapacityExtrapolation, FleetMean, SlopeExtrapolation
from cobalt.models.conformal import (
    ConformalRegressor,
    coverage,
    implied_sigma,
    mean_interval_width,
    quantile_from_interval,
)
from cobalt.models.registry import TASKS, available, build, is_baseline
from cobalt.models.regressors import gradient_boosting, ridge


@pytest.fixture
def linear_problem() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A well-behaved regression with a group structure, for the plumbing tests."""
    rng = np.random.default_rng(0)
    n = 400
    groups = np.array([f"G{i % 8}" for i in range(n)])
    features = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    target = 2.0 * features["a"] - 1.0 * features["b"] + rng.normal(0, 0.3, n)
    return features, target.to_numpy(), groups


# -- registry -----------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_every_registered_model_can_be_built(task):
    for name in available(task):
        model = build(task, name, seed=0, rated_capacity_ah=2.2, end_of_life_soh=0.8)
        assert hasattr(model, "fit") and hasattr(model, "predict")


def test_unknown_names_are_rejected_with_the_alternatives():
    with pytest.raises(ValueError, match="Available"):
        build("early_life", "transformer")
    with pytest.raises(ValueError, match="unknown task"):
        build("telepathy", "ridge")


def test_baselines_are_identified_as_such():
    assert is_baseline("fleet_mean")
    assert is_baseline("capacity_extrapolation")
    assert not is_baseline("ridge")


# -- baselines ----------------------------------------------------------


def test_fleet_mean_predicts_the_mean(linear_problem):
    features, target, _ = linear_problem
    model = FleetMean().fit(features, target)
    prediction = model.predict(features)
    assert np.allclose(prediction, np.mean(target))
    assert len(prediction) == len(features)


def test_capacity_extrapolation_reaches_the_right_answer_on_a_clean_case():
    """A cell fading exactly 0.001 Ah per cycle from 2.0 Ah, threshold 1.76 Ah."""
    model = CapacityExtrapolation(rated_capacity_ah=2.2, end_of_life_soh=0.80)
    model.fit(pd.DataFrame({"x": [0]}), np.array([np.log10(400.0)]))
    features = pd.DataFrame({"capacity_cycle_100": [2.000], "capacity_fade_2_to_100": [0.098]})
    predicted = float(10.0 ** model.predict(features)[0])
    # (2.000 - 1.76) / 0.001 = 240 cycles beyond cycle 100.
    assert predicted == pytest.approx(340.0, rel=0.02)


def test_capacity_extrapolation_survives_a_flat_fade_curve():
    model = CapacityExtrapolation(rated_capacity_ah=2.2)
    model.fit(pd.DataFrame({"x": [0]}), np.array([np.log10(900.0)]))
    features = pd.DataFrame(
        {
            "capacity_cycle_100": [2.10, 2.10, 2.10],
            # Zero fade, negative fade and a fade that puts the cell already dead.
            "capacity_fade_2_to_100": [0.0, -0.05, 1.0],
        }
    )
    prediction = model.predict(features)
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()


def test_extrapolation_respects_the_planning_horizon():
    """A tiny slope must not extrapolate to a number nobody would act on."""
    model = CapacityExtrapolation(rated_capacity_ah=2.2)
    model.fit(pd.DataFrame({"x": [0]}), np.array([np.log10(1000.0)]))
    features = pd.DataFrame({"capacity_cycle_100": [2.10], "capacity_fade_2_to_100": [1e-6]})
    predicted = float(10.0 ** model.predict(features)[0])
    assert predicted <= 1000.0 * 10**model.HORIZON_MARGIN_DEX + 1


def test_slope_extrapolation_handles_a_rising_state_of_health():
    model = SlopeExtrapolation(end_of_life_soh=0.80)
    model.fit(pd.DataFrame({"x": [0]}), np.array([np.log10(200.0)]))
    features = pd.DataFrame({"soh_now": [0.9, 0.9], "soh_slope": [0.0, +1e-4]})
    prediction = model.predict(features)
    assert np.isfinite(prediction).all()


# -- regressors ---------------------------------------------------------


def test_ridge_learns_a_linear_relationship(linear_problem):
    features, target, _ = linear_problem
    model = ridge().fit(features, target)
    residual = model.predict(features) - target
    assert float(np.std(residual)) < 0.5


def test_coefficients_are_exposed_for_linear_models(linear_problem):
    features, target, _ = linear_problem
    model = ridge().fit(features, target)
    coefficients = model.coefficients
    assert coefficients is not None
    # 'a' has twice the influence of 'b' and 'c' has none.
    assert abs(coefficients["a"]) > abs(coefficients["b"]) > abs(coefficients["c"])


def test_tree_models_report_no_coefficients(linear_problem):
    features, target, _ = linear_problem
    model = gradient_boosting().fit(features, target)
    assert model.coefficients is None


def test_a_reordered_feature_frame_is_rejected(linear_problem):
    """Silently reordering columns would rescale every coefficient."""
    features, target, _ = linear_problem
    model = ridge().fit(features, target)
    with pytest.raises(ValueError, match="different feature set"):
        model.predict(features[["c", "b", "a"]])


def test_a_missing_feature_names_itself(linear_problem):
    features, target, _ = linear_problem
    model = ridge().fit(features, target)
    with pytest.raises(ValueError, match="missing="):
        model.predict(features[["a", "b"]])


# -- conformal ----------------------------------------------------------


def test_conformal_covers_at_about_the_nominal_rate(linear_problem):
    features, target, groups = linear_problem
    train = slice(0, 300)
    test = slice(300, None)

    model = ConformalRegressor(ridge(), alpha=0.10, grouped=False, seed=0)
    model.fit(features.iloc[train], target[train])
    lower, upper = model.predict_interval(features.iloc[test])

    achieved = coverage(lower, upper, target[test])
    assert 0.80 <= achieved <= 1.0
    assert mean_interval_width(lower, upper) > 0


def test_grouped_calibration_holds_out_whole_groups(linear_problem):
    features, target, groups = linear_problem
    model = ConformalRegressor(ridge(), alpha=0.10, grouped=True, seed=1)
    model.fit(features, target, groups=groups)
    assert model.n_calibration > 0
    # Group sizes are 50 here, so a grouped calibration set is a multiple of 50.
    assert model.n_calibration % 50 == 0


def test_random_calibration_is_named_as_such(linear_problem):
    features, target, _ = linear_problem
    model = ConformalRegressor(ridge(), grouped=False, seed=0)
    assert "random-cal" in model.name


def test_calibration_that_would_consume_every_group_degrades_loudly(linear_problem):
    features, target, groups = linear_problem
    model = ConformalRegressor(ridge(), calibration_fraction=0.99, grouped=True, seed=0)
    model.fit(features, target, groups=groups)
    assert "degraded-to-random-cal" in model.name


def test_interval_before_fit_is_an_error(linear_problem):
    features, _, _ = linear_problem
    model = ConformalRegressor(ridge())
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict_interval(features)


def test_conformal_rejects_nonsense_settings():
    with pytest.raises(ValueError, match="alpha"):
        ConformalRegressor(ridge(), alpha=1.5)
    with pytest.raises(ValueError, match="calibration_fraction"):
        ConformalRegressor(ridge(), calibration_fraction=0.0)


def test_implied_sigma_inverts_the_normal_quantile():
    sigma = implied_sigma(half_width=1.6449, alpha=0.10)
    assert sigma == pytest.approx(1.0, rel=1e-3)
    assert implied_sigma(0.0, 0.10) == 0.0


def test_quantile_from_interval_is_monotonic_in_the_quantile():
    centre = np.zeros(5)
    low = quantile_from_interval(centre, half_width=0.5, alpha=0.10, quantile=0.15)
    mid = quantile_from_interval(centre, half_width=0.5, alpha=0.10, quantile=0.50)
    high = quantile_from_interval(centre, half_width=0.5, alpha=0.10, quantile=0.85)
    assert float(low[0]) < float(mid[0]) < float(high[0])
    assert float(mid[0]) == pytest.approx(0.0)


def test_quantile_from_interval_rejects_an_impossible_quantile():
    with pytest.raises(ValueError, match="quantile"):
        quantile_from_interval(np.zeros(3), 0.5, 0.1, 1.0)


def test_conformal_refuses_a_calibration_split_that_leaves_nothing_to_fit():
    features = pd.DataFrame({"a": [1.0, 2.0]})
    target = np.array([1.0, 2.0])
    model = ConformalRegressor(ridge(), calibration_fraction=0.5, grouped=False, seed=0)
    with pytest.raises(ValueError, match="not enough to fit"):
        model.fit(features, target)
