"""Train and test splits, including the ones that lie.

Three protocols, and the whole point of the repository is the distance between
them:

``random_cycle``
    Shuffle every cell-cycle row and cut. Cycle 604 of a cell can be in training
    while cycle 605 of the same cell is in test. Those two rows are nearly
    identical, so the model gets to interpolate between neighbours it has
    already seen. This is a leaky protocol and it is reported here purely so the
    size of the leak is visible.

``cell_independent``
    Hold out whole cells. No cycle of a test cell appears in training. Honest
    about cells, still optimistic about batches, because a held-out cell has
    siblings from the same coating run and the same depot in training.

``batch_independent``
    Hold out whole batches. This is the arrival of a new manufacturing lot at a
    new site, which is the situation a fleet operator is actually in, and it is
    the number this repository leads with.

Every function returns index arrays into the frame it was handed, and every one
is deterministic given the seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PROTOCOLS = ("random_cycle", "cell_independent", "batch_independent")

#: Which protocols are meaningful for which task. The early-life task has one
#: row per cell, so a random row split *is* a cell-independent split and
#: offering both would imply a distinction that does not exist.
TASK_PROTOCOLS = {
    "early_life": ("cell_independent", "batch_independent"),
    "online_soh": ("random_cycle", "cell_independent", "batch_independent"),
}


@dataclass(frozen=True)
class Split:
    """One train/test partition, with enough provenance to audit it later."""

    protocol: str
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train_index) & set(self.test_index)
        if overlap:
            raise ValueError(f"{self.protocol} fold {self.fold} has {len(overlap)} shared rows")

    @property
    def name(self) -> str:
        return f"{self.protocol}/fold{self.fold}"


def _grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Partition the unique group labels into ``n_folds`` roughly equal blocks."""
    unique = np.unique(groups)
    if len(unique) < n_folds:
        raise ValueError(
            f"cannot build {n_folds} folds from {len(unique)} groups. "
            f"Reduce n_folds or generate a fleet with more batches."
        )
    order = np.random.default_rng(seed).permutation(unique)
    return [np.asarray(block) for block in np.array_split(order, n_folds)]


def group_splits(
    frame: pd.DataFrame,
    group_column: str,
    protocol: str,
    n_folds: int = 4,
    seed: int = 0,
) -> list[Split]:
    """Leave-one-block-of-groups-out folds over ``group_column``."""
    groups = frame[group_column].to_numpy()
    index = np.arange(len(frame))
    splits: list[Split] = []
    for fold, held_out in enumerate(_grouped_folds(groups, n_folds, seed)):
        mask = np.isin(groups, held_out)
        splits.append(
            Split(
                protocol=protocol,
                fold=fold,
                train_index=index[~mask],
                test_index=index[mask],
                train_groups=tuple(sorted(set(groups[~mask]))),
                test_groups=tuple(sorted(set(held_out))),
            )
        )
    return splits


def random_row_splits(frame: pd.DataFrame, n_folds: int = 4, seed: int = 0) -> list[Split]:
    """Plain shuffled k-fold over rows. Leaky by construction for the online task."""
    index = np.random.default_rng(seed).permutation(len(frame))
    splits: list[Split] = []
    for fold, held_out in enumerate(np.array_split(index, n_folds)):
        mask = np.zeros(len(frame), dtype=bool)
        mask[held_out] = True
        splits.append(
            Split(
                protocol="random_cycle",
                fold=fold,
                train_index=np.arange(len(frame))[~mask],
                test_index=np.arange(len(frame))[mask],
                train_groups=(),
                test_groups=(),
            )
        )
    return splits


def build_splits(
    frame: pd.DataFrame,
    protocol: str,
    n_folds: int = 4,
    seed: int = 0,
) -> list[Split]:
    """Dispatch to the right protocol. Raises on an unknown name rather than guessing."""
    if protocol == "random_cycle":
        return random_row_splits(frame, n_folds=n_folds, seed=seed)
    if protocol == "cell_independent":
        return group_splits(frame, "cell_id", protocol, n_folds=n_folds, seed=seed)
    if protocol == "batch_independent":
        return group_splits(frame, "batch_id", protocol, n_folds=n_folds, seed=seed)
    raise ValueError(f"unknown protocol {protocol!r}, expected one of {PROTOCOLS}")


def assert_no_leakage(frame: pd.DataFrame, split: Split) -> None:
    """Check a split really honours its own claim.

    Cheap, and it has already earned its place: a protocol is one string in a
    config file and the difference between an honest number and a fraudulent one
    is whether that string is wired to the grouping the name implies.
    """
    if split.protocol == "random_cycle":
        return

    column = "cell_id" if split.protocol == "cell_independent" else "batch_id"
    train_groups = set(frame.iloc[split.train_index][column])
    test_groups = set(frame.iloc[split.test_index][column])
    shared = train_groups & test_groups
    if shared:
        raise AssertionError(
            f"{split.name} claims {split.protocol} but shares {len(shared)} "
            f"{column} values across the split: {sorted(shared)[:5]}"
        )
