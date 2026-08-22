"""The whole pipeline, end to end, on a fleet small enough to be a test.

This is the one that catches integration breaks: a feature renamed in one place,
a protocol added without a label, a model that fits but cannot be serialised. It
runs the same functions the command line runs, in the same order.

Marked slow because it simulates a fleet. It still finishes in well under a
minute and continuous integration runs the whole suite rather than a subset.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cobalt.cli import build_parser, main
from cobalt.data.schema import FleetDataset
from cobalt.data.splits import TASK_PROTOCOLS
from cobalt.deploy import load_deployed, train_and_save
from cobalt.evaluate import report as report_module
from cobalt.evaluate.protocol import build_feature_tables, run_task
from cobalt.models.registry import is_baseline

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def evaluated(request):
    """Run both tasks through every protocol once and reuse the result."""
    dataset = request.getfixturevalue("tiny_dataset")
    config = request.getfixturevalue("tiny_config")
    tables = build_feature_tables(dataset, config)
    results = {
        task: run_task(task, tables[task]["frame"], config) for task in ("early_life", "online_soh")
    }
    summary = report_module.build_summary(
        config, dataset, results, {task: tables[task]["excluded"] for task in tables}
    )
    return config, dataset, tables, results, summary


def test_every_protocol_for_every_task_produced_metrics(evaluated):
    _, _, _, results, _ = evaluated
    for task, result in results.items():
        assert set(result["protocols"]) == set(TASK_PROTOCOLS[task])
        for payload in result["protocols"].values():
            for metrics in payload["models"].values():
                assert metrics["n"] > 0
                assert np.isfinite(metrics["rmse_cycles"])


def test_every_row_was_tested_exactly_once(evaluated):
    """Pooled predictions back the cost models, so partial coverage is silent damage."""
    _, _, _, results, _ = evaluated
    for result in results.values():
        for payload in result["protocols"].values():
            for predictions in payload["_pooled"].values():
                assert np.isfinite(predictions.log_prediction).all()
                assert len(predictions.log_prediction) == result["n_rows"]


def test_the_deployed_model_carries_intervals(evaluated):
    _, _, _, results, _ = evaluated
    for result in results.values():
        deployed = result["deployed_model"]
        for payload in result["protocols"].values():
            conformal = payload["_pooled"][f"{deployed}+conformal"]
            assert conformal.log_lower is not None
            assert (conformal.log_lower <= conformal.log_prediction).all()
            assert (conformal.log_prediction <= conformal.log_upper).all()


def test_the_summary_is_json_serialisable(evaluated):
    *_, summary = evaluated
    json.dumps(summary)  # must not raise
    assert "_pooled" not in json.dumps(summary)


def test_the_summary_holds_both_cost_analyses(evaluated):
    *_, summary = evaluated
    grading = summary["business"]["grading"]
    replacement = summary["business"]["replacement"]
    assert "batch_independent" in grading
    assert "batch_independent" in replacement
    assert replacement["batch_independent"]["oracle"]["cost_per_pack"] == 0.0


def test_a_model_beats_the_do_nothing_baseline_on_cost(evaluated):
    """A weak claim on a tiny fleet but if it fails the wiring is wrong."""
    *_, summary = evaluated
    grading = summary["business"]["grading"]["batch_independent"]
    assert (
        grading["minimum_expected_cost"]["cost_per_cell"]
        <= (grading["grade_everything_b"]["cost_per_cell"])
    )


def test_figures_and_markdown_are_written(evaluated, tmp_path):
    config, dataset, _, results, summary = evaluated
    pooled = {
        task: {p: payload["_pooled"] for p, payload in results[task]["protocols"].items()}
        for task in results
    }

    report_module.write_summary(summary, tmp_path / "summary.json")
    report_module.write_evaluation_markdown(summary, tmp_path / "EVALUATION.md")
    report_module.figure_capacity_fade(dataset, tmp_path / "fade.png")
    report_module.figure_delta_q(dataset, tmp_path / "delta_q.png")
    report_module.figure_protocol_gap(summary, tmp_path / "gap.png")
    report_module.figure_cost(summary, tmp_path / "cost.png")
    report_module.figure_parity(
        pooled["online_soh"],
        config.evaluation.deployed_online_model,
        tmp_path / "parity.png",
        "remaining life",
    )
    report_module.write_prediction_tables(
        pooled["early_life"], config.evaluation.deployed_early_life_model, tmp_path, "early"
    )

    for name in (
        "summary.json",
        "EVALUATION.md",
        "fade.png",
        "delta_q.png",
        "gap.png",
        "cost.png",
        "parity.png",
    ):
        assert (tmp_path / name).stat().st_size > 0, name
    assert list(tmp_path.glob("early_*.csv"))


def test_models_round_trip_through_disk(evaluated, tmp_path):
    config, _, tables, _, summary = evaluated
    written = train_and_save(
        config,
        tables["early_life"]["frame"],
        tables["online_soh"]["frame"],
        summary,
        tmp_path,
    )
    assert set(written) == {"early_life", "online_soh", "model_card"}

    early, online = load_deployed(tmp_path)
    features = tables["online_soh"]["frame"][list(online.feature_names)].head(5)
    prediction = online.predict(features)
    assert len(prediction["point"]) == 5
    assert (prediction["lower"] <= prediction["point"]).all()
    assert (prediction["point"] <= prediction["upper"]).all()

    early_features = tables["early_life"]["frame"][list(early.feature_names)].head(3)
    assert len(early.predict(early_features)["point"]) == 3


def test_a_deployed_model_needing_a_missing_feature_says_which(evaluated, tmp_path):
    config, _, tables, _, summary = evaluated
    train_and_save(
        config, tables["early_life"]["frame"], tables["online_soh"]["frame"], summary, tmp_path
    )
    _, online = load_deployed(tmp_path)
    frame = tables["online_soh"]["frame"][list(online.feature_names)].head(2)
    with pytest.raises(KeyError, match="missing features"):
        online.predict(frame.drop(columns=[online.feature_names[0]]))


def test_baselines_are_never_deployed(evaluated, tmp_path):
    """Reported, yes. Shipped, no."""
    from dataclasses import replace

    config, _, tables, _, summary = evaluated
    baseline_config = replace(
        config,
        evaluation=replace(config.evaluation, deployed_early_life_model="capacity_extrapolation"),
    )
    assert is_baseline("capacity_extrapolation")
    with pytest.raises(ValueError, match="never deployed"):
        train_and_save(
            baseline_config,
            tables["early_life"]["frame"],
            tables["online_soh"]["frame"],
            summary,
            tmp_path,
        )


def test_loading_absent_models_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="make demo"):
        load_deployed(tmp_path)


def test_a_deployed_model_not_in_the_evaluated_set_is_rejected(evaluated):
    from dataclasses import replace

    config, _, tables, _, _ = evaluated
    broken = replace(
        config,
        evaluation=replace(config.evaluation, deployed_online_model="gradient_boosting"),
    )
    with pytest.raises(ValueError, match="is not in the evaluated set"):
        run_task("online_soh", tables["online_soh"]["frame"], broken)


def test_grouped_calibration_warns_when_there_are_too_few_groups(evaluated):
    """A real trap, found by this test and now a stated precondition.

    Grouped calibration takes whole groups out of the fit. On the four-batch
    fleet this suite uses, a two-fold batch split leaves two training batches
    and calibration takes one, so the model is fitted on a single batch. It then
    covers *worse* than a randomly calibrated one, not because the grouping
    argument is wrong but because the fit has been starved of the very
    between-group variety the argument is about.

    The shipped configs clear the threshold comfortably. This asserts that a
    configuration which does not is flagged rather than quietly reported.
    """
    from cobalt.models.conformal import MIN_FIT_GROUPS, ConformalRegressor
    from cobalt.models.regressors import ridge

    config, _, tables, _, _ = evaluated
    frame = tables["online_soh"]["frame"]

    model = ConformalRegressor(ridge(), alpha=0.10, grouped=True, seed=0)
    model.fit(
        frame[list(tables["online_soh"]["frame"].columns[6:])],
        frame["log_remaining_cycles"].to_numpy(dtype=float),
        groups=frame["batch_id"].to_numpy(),
    )
    if model.n_fit_groups < MIN_FIT_GROUPS:
        assert model.group_warning is not None
        assert str(MIN_FIT_GROUPS) in model.group_warning
    else:
        assert model.group_warning is None


def test_grouped_calibration_really_holds_out_whole_groups(evaluated):
    """The mechanism, asserted where the tiny fleet cannot confound it."""
    from cobalt.models.conformal import ConformalRegressor
    from cobalt.models.regressors import ridge

    config, _, tables, _, _ = evaluated
    frame = tables["online_soh"]["frame"]
    groups = frame["batch_id"].to_numpy()
    sizes = {group: int((groups == group).sum()) for group in set(groups)}

    model = ConformalRegressor(ridge(), alpha=0.10, grouped=True, seed=0)
    model.fit(
        frame[list(frame.columns[6:])],
        frame["log_remaining_cycles"].to_numpy(dtype=float),
        groups=groups,
    )
    assert model.n_calibration_groups >= 1
    # The calibration set is a union of whole groups, so its size is a sum of
    # group sizes rather than an arbitrary row count.
    assert model.n_calibration in {
        sum(subset)
        for subset in [
            [sizes[g] for g in combination]
            for combination in _combinations(sorted(sizes), model.n_calibration_groups)
        ]
    }


def _combinations(items, size):
    from itertools import combinations

    return list(combinations(items, size))


def test_the_shipped_configs_leave_enough_groups_to_fit_on():
    """The precondition above, checked against what actually ships."""
    from cobalt.config import load_config, repository_root
    from cobalt.models.conformal import MIN_FIT_GROUPS

    for name in ("default.yaml", "ci.yaml"):
        config = load_config(repository_root() / "config" / name)
        held_out_per_fold = config.fleet.n_batches // config.evaluation.n_folds
        training_groups = config.fleet.n_batches - held_out_per_fold
        calibration_groups = round(config.evaluation.calibration_fraction * training_groups)
        assert training_groups - calibration_groups >= MIN_FIT_GROUPS, name


# -- the command line ----------------------------------------------------


def test_the_parser_exposes_every_command():
    parser = build_parser()
    for command in ("generate", "evaluate", "report", "stream", "serve"):
        args = parser.parse_args([command])
        assert hasattr(args, "handler")


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_generate_writes_a_dataset_and_then_leaves_it_alone(tmp_path, monkeypatch):
    config_path = tmp_path / "tiny.yaml"
    data_dir = tmp_path / "data"
    config_path.write_text(
        "name: tiny\n"
        "seed: 3\n"
        "paths:\n"
        f"  data_dir: {data_dir.as_posix()}\n"
        "fleet:\n"
        "  n_batches: 2\n"
        "  cells_per_batch: 2\n"
        "  max_cycles: 130\n"
        "  site_temperature_mean_c: 48.0\n"
        "  site_charge_c_rate_choices: [5.0, 6.0]\n",
        encoding="utf-8",
    )

    assert main(["generate", "--config", str(config_path)]) == 0
    dataset = FleetDataset.load(data_dir)
    assert len(dataset.cells) == 4

    written = (data_dir / "cycles.csv.gz").stat().st_mtime_ns
    assert main(["generate", "--config", str(config_path)]) == 0
    assert (data_dir / "cycles.csv.gz").stat().st_mtime_ns == written


def test_evaluate_on_a_missing_dataset_exits_with_a_message(tmp_path, capsys):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text(
        f"name: empty\npaths:\n  data_dir: {(tmp_path / 'nothing').as_posix()}\n",
        encoding="utf-8",
    )
    assert main(["evaluate", "--config", str(config_path)]) == 2
    assert "cobalt generate" in capsys.readouterr().err
