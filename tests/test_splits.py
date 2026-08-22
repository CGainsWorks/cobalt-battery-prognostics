"""The split protocols, which are the part it would be easiest to get quietly wrong.

Every test here is about one question: does a split honour the claim its name
makes? A protocol is a string in a config file and the difference between an
honest number and a worthless one is whether that string is wired to the
grouping it names. Nothing about a wrong wiring looks wrong at runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobalt.data.splits import (
    PROTOCOLS,
    TASK_PROTOCOLS,
    Split,
    assert_no_leakage,
    build_splits,
    group_splits,
    random_row_splits,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = []
    for batch in range(6):
        for cell in range(4):
            for cycle in range(5):
                rows.append(
                    {
                        "batch_id": f"B{batch}",
                        "cell_id": f"B{batch}-C{cell}",
                        "cycle": cycle,
                        "value": batch * 10 + cell,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_folds_partition_the_rows_exactly_once(frame, protocol):
    splits = build_splits(frame, protocol, n_folds=3, seed=0)
    covered = np.concatenate([split.test_index for split in splits])
    assert sorted(covered) == list(range(len(frame)))


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_train_and_test_never_overlap(frame, protocol):
    for split in build_splits(frame, protocol, n_folds=3, seed=0):
        assert not set(split.train_index) & set(split.test_index)


def test_cell_independent_shares_no_cell(frame):
    for split in build_splits(frame, "cell_independent", n_folds=3, seed=0):
        train = set(frame.iloc[split.train_index]["cell_id"])
        test = set(frame.iloc[split.test_index]["cell_id"])
        assert not train & test
        assert_no_leakage(frame, split)


def test_batch_independent_shares_no_batch(frame):
    for split in build_splits(frame, "batch_independent", n_folds=3, seed=0):
        train = set(frame.iloc[split.train_index]["batch_id"])
        test = set(frame.iloc[split.test_index]["batch_id"])
        assert not train & test
        assert_no_leakage(frame, split)


def test_batch_independent_is_also_cell_independent(frame):
    """Holding out a batch necessarily holds out its cells."""
    for split in build_splits(frame, "batch_independent", n_folds=3, seed=0):
        train = set(frame.iloc[split.train_index]["cell_id"])
        test = set(frame.iloc[split.test_index]["cell_id"])
        assert not train & test


def test_random_cycle_does_leak(frame):
    """Stated as a test so nobody 'fixes' it later thinking it is a bug."""
    shared = 0
    for split in random_row_splits(frame, n_folds=3, seed=0):
        train = set(frame.iloc[split.train_index]["cell_id"])
        test = set(frame.iloc[split.test_index]["cell_id"])
        shared += len(train & test)
    assert shared > 0


def test_assert_no_leakage_catches_a_mislabelled_split(frame):
    """A split that claims a grouping it does not have must be rejected."""
    honest = build_splits(frame, "random_cycle", n_folds=2, seed=0)[0]
    liar = Split(
        protocol="batch_independent",
        fold=0,
        train_index=honest.train_index,
        test_index=honest.test_index,
        train_groups=(),
        test_groups=(),
    )
    with pytest.raises(AssertionError, match="claims batch_independent"):
        assert_no_leakage(frame, liar)


def test_split_rejects_overlapping_indices():
    with pytest.raises(ValueError, match="shared rows"):
        Split(
            protocol="cell_independent",
            fold=0,
            train_index=np.array([0, 1, 2]),
            test_index=np.array([2, 3]),
            train_groups=(),
            test_groups=(),
        )


def test_too_many_folds_for_the_groups_is_an_error(frame):
    with pytest.raises(ValueError, match="cannot build"):
        group_splits(frame, "batch_id", "batch_independent", n_folds=99, seed=0)


def test_unknown_protocol_is_rejected(frame):
    with pytest.raises(ValueError, match="unknown protocol"):
        build_splits(frame, "vibes", n_folds=2, seed=0)


def test_splits_are_deterministic(frame):
    first = build_splits(frame, "batch_independent", n_folds=3, seed=5)
    second = build_splits(frame, "batch_independent", n_folds=3, seed=5)
    for left, right in zip(first, second):
        assert np.array_equal(left.test_index, right.test_index)


def test_a_different_seed_gives_a_different_partition(frame):
    first = build_splits(frame, "batch_independent", n_folds=3, seed=1)
    second = build_splits(frame, "batch_independent", n_folds=3, seed=2)
    assert any(
        set(left.test_groups) != set(right.test_groups) for left, right in zip(first, second)
    )


def test_the_early_life_task_does_not_offer_the_leaky_protocol():
    """One row per cell, so a random row split is already cell-independent.

    Offering both would imply a distinction that does not exist and would put a
    meaningless 'leaky' column in the results table.
    """
    assert "random_cycle" not in TASK_PROTOCOLS["early_life"]
    assert "random_cycle" in TASK_PROTOCOLS["online_soh"]


def test_assert_no_leakage_is_a_no_op_for_the_random_protocol(frame):
    split = random_row_splits(frame, n_folds=2, seed=0)[0]
    assert_no_leakage(frame, split)


def test_split_name_is_readable(frame):
    split = build_splits(frame, "cell_independent", n_folds=2, seed=0)[0]
    assert split.name == "cell_independent/fold0"
