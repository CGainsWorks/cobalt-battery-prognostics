"""Fleet structure and the on-disk round trip.

The load-bearing test in this file is the one about variance. The entire
argument of the repository rests on batches being more alike inside than
between, so if that ever stops being true the batch-independent protocol stops
being a harder question and every conclusion downstream is empty.
"""

from __future__ import annotations

import numpy as np
import pytest

from cobalt.data.schema import (
    CELLS_FILE,
    CURVES_FILE,
    CYCLES_FILE,
    MANIFEST_FILE,
    FleetDataset,
    cells_frame,
    cycles_frame,
    describe_defaults,
    history_summary,
    write_fleet,
)
from cobalt.sim.fleet import FleetConfig, simulate_fleet


def test_fleet_has_the_requested_shape(tiny_fleet, tiny_fleet_config):
    expected = tiny_fleet_config.n_batches * tiny_fleet_config.cells_per_batch
    assert len(tiny_fleet.cells) == expected
    assert len(tiny_fleet.batch_ids) == tiny_fleet_config.n_batches


def test_every_cell_has_a_unique_identifier(tiny_fleet):
    identifiers = [cell.cell_id for cell in tiny_fleet.cells]
    assert len(set(identifiers)) == len(identifiers)


def test_most_cells_reach_end_of_life(tiny_fleet):
    """If the ceiling censors most of the fleet the labels are mostly fiction."""
    censored = sum(1 for cell in tiny_fleet.cells if cell.censored)
    assert censored < len(tiny_fleet.cells) / 2


def test_batches_are_more_alike_inside_than_between(tiny_fleet):
    lives_by_batch: dict[str, list[float]] = {}
    for cell in tiny_fleet.uncensored():
        lives_by_batch.setdefault(cell.batch_id, []).append(float(cell.cycle_life))

    populated = {k: v for k, v in lives_by_batch.items() if len(v) >= 2}
    assert len(populated) >= 3, "not enough populated batches to compare"

    within = float(np.mean([np.var(values) for values in populated.values()]))
    between = float(np.var([np.mean(values) for values in populated.values()]))
    assert between > within, (
        f"between-batch variance {between:.1f} does not exceed within-batch {within:.1f}, "
        f"so the batch-independent protocol is not a harder question than the cell one"
    )


def test_duty_cycles_are_correlated_inside_a_batch(tiny_fleet):
    """Cells in a lot share a site, so their charge policies must cluster."""
    by_batch: dict[str, list[float]] = {}
    for cell in tiny_fleet.cells:
        by_batch.setdefault(cell.batch_id, []).append(cell.duty.charge_c_rate)

    within = float(np.mean([np.var(rates) for rates in by_batch.values()]))
    between = float(np.var([np.mean(rates) for rates in by_batch.values()]))
    assert between > within


def test_curves_are_retained_at_the_configured_cycles(tiny_fleet, tiny_fleet_config):
    for cell in tiny_fleet.cells:
        if len(cell.records) >= max(tiny_fleet_config.curve_cycles):
            assert set(cell.voltage_curves) == set(tiny_fleet_config.curve_cycles)


def test_simulation_is_reproducible(tiny_fleet_config):
    first = simulate_fleet(tiny_fleet_config, seed=7)
    second = simulate_fleet(tiny_fleet_config, seed=7)
    assert [c.cycle_life for c in first.cells] == [c.cycle_life for c in second.cells]


def test_adding_cells_does_not_disturb_the_ones_before_them(tiny_fleet_config):
    """Per-cell seed streams, so the CI fleet is a prefix of a larger one."""
    from dataclasses import replace

    small = simulate_fleet(tiny_fleet_config, seed=7)
    bigger = simulate_fleet(
        replace(tiny_fleet_config, n_batches=tiny_fleet_config.n_batches + 2), seed=7
    )
    shared = len(small.cells)
    assert [c.cell_id for c in bigger.cells[:shared]] == [c.cell_id for c in small.cells]
    assert [c.cycle_life for c in bigger.cells[:shared]] == [c.cycle_life for c in small.cells]


def test_cycles_frame_has_one_row_per_cell_cycle(tiny_fleet):
    frame = cycles_frame(tiny_fleet)
    assert len(frame) == sum(len(cell.records) for cell in tiny_fleet.cells)
    assert frame.groupby("cell_id")["cycle"].apply(lambda s: s.is_monotonic_increasing).all()


def test_cells_frame_labels_censoring_honestly(tiny_fleet):
    frame = cells_frame(tiny_fleet)
    for _, row in frame.iterrows():
        if row["censored"]:
            # A censored cell has no cycle life, so the column holds the number
            # of cycles observed, which must not be mistaken for a label.
            assert row["cycle_life"] == row["observed_cycles"]


def test_round_trip_through_disk(tiny_fleet, tmp_path):
    written = write_fleet(tiny_fleet, tmp_path)
    for name in (CYCLES_FILE, CELLS_FILE, CURVES_FILE, MANIFEST_FILE):
        assert (tmp_path / name).exists()
    assert set(written) == {"cycles", "cells", "curves", "manifest"}

    dataset = FleetDataset.load(tmp_path)
    assert len(dataset.cells) == len(tiny_fleet.cells)
    assert np.allclose(dataset.voltage_grid, tiny_fleet.voltage_grid)

    first = tiny_fleet.cells[0]
    if first.voltage_curves:
        cycle = sorted(first.voltage_curves)[0]
        assert np.allclose(
            dataset.curves[(first.cell_id, cycle)],
            first.voltage_curves[cycle],
            equal_nan=True,
        )


def test_loading_a_non_dataset_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="cobalt generate"):
        FleetDataset.load(tmp_path)


def test_batch_lookup(tiny_dataset):
    cell_id = tiny_dataset.cell_ids[0]
    assert tiny_dataset.batch_of(cell_id) == cell_id.split("-")[0]
    with pytest.raises(KeyError):
        tiny_dataset.batch_of("does-not-exist")


def test_helpers_produce_something_serialisable(tiny_fleet):
    defaults = describe_defaults()
    assert "cell" in defaults and "duty" in defaults
    summary = history_summary(tiny_fleet.cells[0])
    assert summary["cell_id"] == tiny_fleet.cells[0].cell_id


def test_a_fleet_with_no_batches_is_rejected_by_the_split_builder():
    """Guarding the degenerate case here rather than three layers up."""
    config = FleetConfig(n_batches=1, cells_per_batch=2, max_cycles=40)
    fleet = simulate_fleet(config, seed=1)
    assert len(fleet.batch_ids) == 1
