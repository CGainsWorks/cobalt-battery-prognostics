"""Scoring arithmetic and configuration loading."""

from __future__ import annotations

import numpy as np
import pytest

from cobalt.config import Config, load_config, load_resolved, repository_root
from cobalt.evaluate.metrics import (
    ALPHA_TOLERANCE,
    aggregate,
    interval_metrics,
    regression_metrics,
)

# -- metrics ------------------------------------------------------------


def test_a_perfect_prediction_scores_perfectly():
    truth = np.log10(np.array([100.0, 500.0, 900.0]))
    metrics = regression_metrics(truth, truth)
    assert metrics["rmse_log"] == pytest.approx(0.0)
    assert metrics["rmse_cycles"] == pytest.approx(0.0)
    assert metrics["mape"] == pytest.approx(0.0)
    assert metrics["alpha_lambda"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_errors_are_reported_in_cycles_as_well_as_log_space():
    truth = np.log10(np.array([100.0, 100.0]))
    prediction = np.log10(np.array([110.0, 90.0]))
    metrics = regression_metrics(truth, prediction)
    assert metrics["mae_cycles"] == pytest.approx(10.0)
    assert metrics["rmse_cycles"] == pytest.approx(10.0)


def test_alpha_lambda_counts_predictions_inside_the_tolerance_band():
    truth = np.log10(np.array([1000.0] * 4))
    # 10% out, 19% out, 21% out, 50% out. Tolerance is 20%.
    prediction = np.log10(np.array([1100.0, 1190.0, 1210.0, 1500.0]))
    metrics = regression_metrics(truth, prediction)
    assert ALPHA_TOLERANCE == 0.20
    assert metrics["alpha_lambda"] == pytest.approx(0.5)


def test_bias_keeps_its_sign():
    truth = np.log10(np.array([100.0, 100.0]))
    optimistic = regression_metrics(truth, np.log10(np.array([120.0, 130.0])))
    assert optimistic["bias_cycles"] > 0


def test_a_constant_prediction_gets_a_rank_correlation_of_zero():
    """Spearman is undefined on a constant, and a NaN would poison every aggregate."""
    truth = np.log10(np.array([100.0, 500.0, 900.0, 1200.0]))
    metrics = regression_metrics(truth, np.full(4, np.log10(400.0)))
    assert metrics["spearman"] == 0.0


def test_metrics_reject_mismatched_or_empty_input():
    with pytest.raises(ValueError, match="shape mismatch"):
        regression_metrics(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="no predictions"):
        regression_metrics(np.array([]), np.array([]))


def test_interval_metrics_report_the_coverage_shortfall():
    truth = np.log10(np.array([100.0, 200.0, 300.0, 400.0]))
    lower = truth - 0.05
    upper = truth + 0.05
    inside = interval_metrics(truth, lower, upper, alpha=0.10)
    assert inside["picp"] == pytest.approx(1.0)
    assert inside["coverage_gap"] == pytest.approx(0.10)

    # Half the truths fall outside.
    shifted = lower.copy()
    shifted[:2] = truth[:2] + 0.02
    outside = interval_metrics(truth, shifted, upper, alpha=0.10)
    assert outside["picp"] < 0.9
    assert outside["coverage_gap"] < 0


def test_aggregate_reports_a_mean_and_a_spread():
    folds = [
        {"n": 10, "rmse_cycles": 100.0},
        {"n": 10, "rmse_cycles": 200.0},
    ]
    summary = aggregate(folds)
    assert summary["rmse_cycles"] == pytest.approx(150.0)
    assert summary["rmse_cycles_std"] == pytest.approx(50.0)
    assert summary["n_folds"] == 2
    assert summary["n"] == 20


def test_aggregate_rejects_an_empty_list():
    with pytest.raises(ValueError, match="nothing to aggregate"):
        aggregate([])


# -- config -------------------------------------------------------------


def test_the_shipped_configs_load():
    root = repository_root()
    for name in ("default.yaml", "ci.yaml"):
        config = load_config(root / "config" / name)
        assert isinstance(config, Config)
        assert config.fleet.n_batches >= config.evaluation.n_folds


def test_the_default_config_can_support_its_own_fold_count():
    """Four-fold batch-independent CV needs at least four batches."""
    config = load_config(repository_root() / "config" / "default.yaml")
    assert config.fleet.n_batches >= config.evaluation.n_folds


def test_the_deployed_models_are_in_the_evaluated_set():
    for name in ("default.yaml", "ci.yaml"):
        config = load_config(repository_root() / "config" / name)
        assert config.evaluation.deployed_early_life_model in config.evaluation.early_life_models
        assert config.evaluation.deployed_online_model in config.evaluation.online_models


def test_lists_become_tuples_including_nested_ones():
    """The grading cost matrix is a tuple of tuples and YAML gives lists of lists."""
    config = load_config(repository_root() / "config" / "default.yaml")
    assert isinstance(config.grading.cost_matrix, tuple)
    assert all(isinstance(row, tuple) for row in config.grading.cost_matrix)
    assert isinstance(config.fleet.curve_cycles, tuple)


def test_an_unknown_top_level_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nnonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_config(path)


def test_an_unknown_section_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("fleet:\n  n_batchez: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_a_missing_config_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="no config at"):
        load_config(tmp_path / "absent.yaml")


def test_a_non_mapping_config_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(path)


def test_paths_are_resolved_against_the_repository_not_the_cwd():
    config = load_resolved("config/default.yaml")
    from pathlib import Path

    assert Path(config.paths.results_dir).is_absolute()
    assert repository_root() in Path(config.paths.results_dir).parents


def test_config_serialises_to_plain_types():
    config = load_config(repository_root() / "config" / "ci.yaml")
    payload = config.as_dict()
    import json

    json.dumps(payload)  # must not raise
    assert payload["fleet"]["n_batches"] == config.fleet.n_batches
