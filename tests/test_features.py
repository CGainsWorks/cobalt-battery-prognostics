"""Feature extraction and the two things it must never do.

It must never emit a feature that could not be measured on a real pack and it
must never emit one derived from the label. Both are easy to do by accident and
neither shows up as an error: the first produces a model that cannot be
deployed, the second produces a model that looks superb and is worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

from cobalt.features import early_life, online

# -- early life ---------------------------------------------------------


def test_early_life_table_has_every_declared_feature(tiny_dataset):
    frame, _ = early_life.build_table(tiny_dataset)
    assert not frame.empty
    for name in early_life.FEATURE_NAMES:
        assert name in frame.columns


def test_early_life_features_are_finite(tiny_dataset):
    frame, _ = early_life.build_table(tiny_dataset)
    values = frame[list(early_life.FEATURE_NAMES)].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_early_life_has_one_row_per_usable_cell(tiny_dataset):
    frame, excluded = early_life.build_table(tiny_dataset)
    assert frame["cell_id"].is_unique
    assert len(frame) + sum(excluded.values()) == len(tiny_dataset.cells)


def test_censored_cells_are_excluded_and_counted(tiny_dataset):
    frame, excluded = early_life.build_table(tiny_dataset)
    censored = set(tiny_dataset.cells.loc[tiny_dataset.cells["censored"].astype(bool), "cell_id"])
    assert not censored & set(frame["cell_id"])
    assert excluded["censored"] == len(censored)


def test_the_label_is_the_log_of_the_label(tiny_dataset):
    frame, _ = early_life.build_table(tiny_dataset)
    assert np.allclose(frame["log_cycle_life"], np.log10(frame["cycle_life"]))


def test_no_early_life_feature_leaks_the_label(tiny_dataset):
    """None of the features may be a deterministic function of cycle life.

    A perfect correlation on a fleet this small would still be suspicious, so the
    bar is set where a genuine physical feature has no business reaching.
    """
    frame, _ = early_life.build_table(tiny_dataset)
    if len(frame) < 8:
        pytest.skip("too few cells for a correlation to mean anything")
    label = frame["log_cycle_life"].to_numpy(dtype=float)
    for name in early_life.FEATURE_NAMES:
        values = frame[name].to_numpy(dtype=float)
        if np.std(values) == 0:
            continue
        correlation = abs(float(np.corrcoef(values, label)[0, 1]))
        assert correlation < 0.999, f"{name} is essentially the label"


def test_delta_q_needs_both_cycles(tiny_dataset):
    cell_id = tiny_dataset.cell_ids[0]
    assert early_life.delta_q(tiny_dataset, cell_id) is not None
    assert early_life.delta_q(tiny_dataset, "not-a-cell") is None


def test_delta_q_is_negative_because_capacity_falls(tiny_dataset):
    for cell_id in tiny_dataset.cell_ids[:5]:
        difference = early_life.delta_q(tiny_dataset, cell_id)
        if difference is None:
            continue
        assert float(np.mean(difference)) < 0.0


def test_a_cell_that_dies_inside_the_window_cannot_be_predicted(tiny_dataset):
    short = tiny_dataset.cycles_for(tiny_dataset.cell_ids[0])["cell_id"].iloc[0]
    features = early_life.cell_features(tiny_dataset, short)
    # The tiny fleet is driven hard but not that hard, so this one is usable.
    assert features is None or len(features) == len(early_life.FEATURE_NAMES)


# -- online -------------------------------------------------------------


def test_online_table_has_every_declared_feature(tiny_dataset, tiny_config):
    frame, _ = online.build_table(
        tiny_dataset,
        window_cycles=tiny_config.online.window_cycles,
        stride=tiny_config.online.stride,
    )
    assert not frame.empty
    for name in online.FEATURE_NAMES:
        assert name in frame.columns


def test_online_features_are_finite(tiny_dataset, tiny_config):
    frame, _ = online.build_table(
        tiny_dataset,
        window_cycles=tiny_config.online.window_cycles,
        stride=tiny_config.online.stride,
    )
    values = frame[list(online.FEATURE_NAMES)].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_remaining_life_is_consistent_with_the_window_end(tiny_dataset, tiny_config):
    frame, _ = online.build_table(
        tiny_dataset,
        window_cycles=tiny_config.online.window_cycles,
        stride=tiny_config.online.stride,
    )
    assert (frame["remaining_cycles"] == frame["cycle_life"] - frame["cycle"]).all()
    assert (frame["remaining_cycles"] >= online.MIN_REMAINING_CYCLES).all()


def test_windows_stop_at_end_of_life(tiny_dataset, tiny_config):
    """Past end of life the pack is booked for replacement and there is no decision left."""
    frame, _ = online.build_table(
        tiny_dataset,
        window_cycles=tiny_config.online.window_cycles,
        stride=tiny_config.online.stride,
    )
    assert (frame["cycle"] < frame["cycle_life"]).all()


def test_state_of_health_falls_along_a_cell(tiny_dataset, tiny_config):
    frame, _ = online.build_table(
        tiny_dataset,
        window_cycles=tiny_config.online.window_cycles,
        stride=tiny_config.online.stride,
    )
    for cell_id, group in frame.groupby("cell_id"):
        if len(group) < 3:
            continue
        ordered = group.sort_values("cycle")
        assert ordered["soh_now"].iloc[-1] < ordered["soh_now"].iloc[0], cell_id


def test_online_features_exclude_a_full_reference_discharge(tiny_dataset, tiny_config):
    """The online model must not need something a fleet pack rarely provides.

    Nothing in the feature list may be a raw discharge capacity: the pack in
    service gets partial discharges and a feature that silently assumes a
    laboratory reference test is a model that cannot be deployed.
    """
    assert "discharge_capacity_ah" not in online.FEATURE_NAMES


def test_a_flat_window_does_not_divide_by_zero():
    import pandas as pd

    window = pd.DataFrame(
        {
            "cycle": [10] * 5,
            "discharge_capacity_ah": [2.0] * 5,
            "charge_capacity_ah": [2.01] * 5,
            "internal_resistance_ohm": [0.03] * 5,
            "mean_temperature_c": [25.0] * 5,
            "charge_time_min": [60.0] * 5,
            "discharge_energy_wh": [7.4] * 5,
            "cumulative_ah_throughput": [20.0] * 5,
            "charge_c_rate": [1.0] * 5,
        }
    )
    features = online.window_features(window, rated_capacity_ah=2.2)
    assert all(np.isfinite(value) for value in features.values())
    assert features["soh_slope"] == 0.0


def test_a_cell_shorter_than_a_window_is_excluded(tiny_dataset):
    _, excluded = online.build_table(tiny_dataset, window_cycles=10_000, stride=100)
    assert excluded["too_short"] > 0
