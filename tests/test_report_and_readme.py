"""Reporting and the check that keeps the README honest.

The last two tests are the ones that matter to a reader. One asserts the
committed results exist and the README's tables were generated from them. The
other asserts the README says, in the first screen, that the data is simulated.
Both would be easy to break by accident and neither would look broken.
"""

from __future__ import annotations

import json

import pytest

from cobalt.check_readme import check, extract_block
from cobalt.config import load_resolved, repository_root
from cobalt.evaluate.report import apply_blocks, render_tables

BLOCK_NAMES = (
    "fleet",
    "early_life",
    "early_life_alpha",
    "online",
    "online_alpha",
    "coverage",
    "grading",
    "replacement",
)


@pytest.fixture(scope="module")
def committed_summary() -> dict:
    config = load_resolved("config/default.yaml")
    from pathlib import Path

    path = Path(config.paths.results_dir) / "summary.json"
    if not path.exists():
        pytest.skip(f"no committed results at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# -- the committed results ----------------------------------------------


def test_the_committed_results_exist_and_are_from_the_default_config(committed_summary):
    assert committed_summary["config_name"] == "default"
    assert committed_summary["data_source"] == "simulated"
    assert committed_summary["fleet"]["n_cells"] > 0


def test_the_readme_matches_the_committed_results(committed_summary):
    """The check continuous integration runs, run here too so it fails locally first."""
    from pathlib import Path

    config = load_resolved("config/default.yaml")
    problems = check(
        repository_root() / "README.md",
        Path(config.paths.results_dir) / "summary.json",
    )
    assert not problems, "\n".join(problems)


def test_the_readme_says_the_data_is_simulated_early():
    """The limitation has to be visible without scrolling, not in an appendix."""
    text = (repository_root() / "README.md").read_text(encoding="utf-8")
    head = text[:2500].lower()
    assert "simulat" in head
    assert "no hardware" in head or "no cells" in head


def test_every_block_is_present_in_the_readme(committed_summary):
    text = (repository_root() / "README.md").read_text(encoding="utf-8")
    for name in BLOCK_NAMES:
        block = extract_block(text, name)
        assert block.startswith("|"), f"block {name} is not a table"


def test_the_readme_leads_with_the_batch_independent_number(committed_summary):
    """The honest column must not be buried to the right of a flattering one.

    Checked structurally rather than by eye: the batch-independent column has to
    exist in the headline tables and the hardest protocol has to be the last
    column, which is where a reader's eye lands after the comparison.
    """
    text = (repository_root() / "README.md").read_text(encoding="utf-8")
    header = extract_block(text, "online").splitlines()[0]
    assert header.rstrip().endswith("Batch-independent |")


# -- rendering ----------------------------------------------------------


def test_every_block_renders_as_a_markdown_table(committed_summary):
    tables = render_tables(committed_summary)
    assert set(tables) == set(BLOCK_NAMES)
    for name, table in tables.items():
        lines = table.splitlines()
        assert lines[0].startswith("|"), name
        assert set(lines[1].replace(" ", "")) <= {"|", "-"}, name
        assert len(lines) > 2, name


def test_apply_blocks_replaces_only_the_marked_region():
    text = "before\n<!-- BEGIN:fleet -->\nold\n<!-- END:fleet -->\nafter\n"
    updated = apply_blocks(text, {"fleet": "| a |\n| --- |\n| b |"})
    assert updated.startswith("before\n")
    assert updated.endswith("after\n")
    assert "old" not in updated
    assert "| b |" in updated


def test_apply_blocks_is_idempotent():
    text = "<!-- BEGIN:fleet -->\nold\n<!-- END:fleet -->\n"
    table = "| a |\n| --- |\n| b |"
    once = apply_blocks(text, {"fleet": table})
    assert apply_blocks(once, {"fleet": table}) == once


def test_a_missing_marker_is_an_error():
    with pytest.raises(ValueError, match="BEGIN:fleet"):
        apply_blocks("nothing here", {"fleet": "| a |"})
    with pytest.raises(ValueError, match="no <!-- BEGIN:x"):
        extract_block("nothing here", "x")


def test_check_reports_a_stale_block(tmp_path, committed_summary):
    readme = tmp_path / "README.md"
    body = "".join(f"<!-- BEGIN:{name} -->\nstale\n<!-- END:{name} -->\n" for name in BLOCK_NAMES)
    readme.write_text(body, encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(committed_summary), encoding="utf-8")

    problems = check(readme, summary_path)
    assert len(problems) == len(BLOCK_NAMES)
    assert "is stale" in problems[0]


def test_check_says_what_to_run_when_there_are_no_results(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("nothing", encoding="utf-8")
    problems = check(readme, tmp_path / "absent.json")
    assert len(problems) == 1
    assert "cobalt evaluate" in problems[0]


# -- the model card ------------------------------------------------------


def test_the_committed_model_card_leads_with_the_limitation():
    from pathlib import Path

    config = load_resolved("config/default.yaml")
    path = Path(config.paths.artifacts_dir) / "model_card.json"
    if not path.exists():
        pytest.skip("no committed model card")
    card = json.loads(path.read_text(encoding="utf-8"))
    assert list(card)[0] == "data_source"
    assert "SIMULATED" in card["data_source"]
    assert card["out_of_scope"]


# -- the headline claims, checked against the committed full-scale results ---


def test_the_leaky_protocol_flatters_the_online_model(committed_summary):
    """The claim the README leads with. If it stops holding, the README is wrong."""
    protocols = committed_summary["tasks"]["online_soh"]["protocols"]
    deployed = committed_summary["tasks"]["online_soh"]["deployed_model"]

    leaky = protocols["random_cycle"]["models"][deployed]
    honest = protocols["batch_independent"]["models"][deployed]

    assert leaky["rmse_cycles"] < honest["rmse_cycles"]
    assert leaky["alpha_lambda"] > honest["alpha_lambda"]


def test_holding_out_batches_is_harder_than_holding_out_cells(committed_summary):
    for task in ("early_life", "online_soh"):
        protocols = committed_summary["tasks"][task]["protocols"]
        deployed = committed_summary["tasks"][task]["deployed_model"]
        cell = protocols["cell_independent"]["models"][deployed]["rmse_cycles"]
        batch = protocols["batch_independent"]["models"][deployed]["rmse_cycles"]
        assert batch > cell, task


def test_random_calibration_undercovers_on_a_new_batch(committed_summary):
    """The coverage table's claim, on the fleet that is large enough to make it."""
    for task in ("early_life", "online_soh"):
        intervals = committed_summary["tasks"][task]["protocols"]["batch_independent"]["intervals"]
        nominal = 1.0 - committed_summary["tasks"][task]["alpha"]
        assert intervals["random_calibration"]["picp"] < intervals["matched_calibration"]["picp"]
        assert intervals["random_calibration"]["picp"] < nominal


def test_the_cost_optimal_policies_are_the_cheapest_ones(committed_summary):
    """The business tables' claim: reading the prediction correctly is worth money."""
    replacement = committed_summary["business"]["replacement"]
    for protocol in ("random_cycle", "cell_independent", "batch_independent"):
        policies = replacement[protocol]
        assert (
            policies["newsvendor_quantile"]["cost_per_pack"]
            < policies["point_estimate"]["cost_per_pack"]
        ), protocol
        assert (
            policies["newsvendor_quantile"]["cost_per_pack"]
            < policies["fleet_quantile"]["cost_per_pack"]
        ), protocol

    grading = committed_summary["business"]["grading"]
    for protocol in ("cell_independent", "batch_independent"):
        rules = grading[protocol]
        assert (
            rules["minimum_expected_cost"]["cost_per_cell"]
            < rules["grade_everything_b"]["cost_per_cell"]
        ), protocol
        assert (
            rules["minimum_expected_cost"]["cost_per_cell"]
            <= rules["point_estimate"]["cost_per_cell"]
        ), protocol


def test_the_newsvendor_policy_runs_late_about_as_often_as_it_intends(committed_summary):
    """The theory says the booking should run late q* of the time. Check it did."""
    replacement = committed_summary["business"]["replacement"]
    target = replacement["optimal_quantile"]
    for protocol in ("cell_independent", "batch_independent"):
        achieved = replacement[protocol]["newsvendor_quantile"]["late_fraction"]
        assert abs(achieved - target) < 0.08, (protocol, achieved, target)
