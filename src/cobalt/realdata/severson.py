"""Loader for the Severson et al. (2019) cycle-life dataset.

Commercial LFP/graphite 18650 cells (A123 APR18650M1A, 1.1 Ah nominal): 129 in the
files after the authors' corrections, 124 of which reach end of life, fast-charged
under 70 distinct policies and discharged at 4C in a 30 degC
chamber, in three test campaigns ("batches") run months apart. Published by
the Toyota Research Institute with K. A. Severson, P. M. Attia et al.,
*Data-driven prediction of battery cycle life before capacity degradation*,
Nature Energy 4, 383 to 391 (2019). See `docs/datasets/severson-2019.md` for
the dataset card, the licence position and every preprocessing decision.

Three steps, each one a command:

``download``
    The three batch files from the publisher, about 8.3 GB, into
    ``data/real/severson/raw/`` (git-ignored and docker-ignored).
``extract``
    Read only what this repository uses out of the MATLAB v7.3 (HDF5) files and
    write it in exactly the format :class:`~cobalt.data.schema.FleetDataset`
    reads, into ``data/real/severson/extract/``. That extract is a few
    megabytes, is what CI caches and is never committed.
``evaluate``
    See :mod:`cobalt.realdata.evaluate`.

Preprocessing follows the original authors' own loading code
(``LoadData.m`` and ``Load Data.ipynb`` in their public repository) wherever it
makes a decision, so the cell set matches the paper's and is not a choice made
here after looking at results:

* five batch 1 cells continued into batch 2 under new channel numbers and are
  re-joined to their batch 1 history;
* six batch 3 channels the authors flagged as noisy are dropped;
* cycle life is the first cycle at which discharge capacity falls below 0.88 Ah
  (80% of nominal). A cell whose test stopped with its last capacity at or below
  0.885 Ah is labelled one cycle past its last record, which is the authors'
  rule; a cell that ended above that never reached end of life and is marked
  censored, which the feature builders then exclude with a count.

Nothing here pickles or executes anything from the downloaded files.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cobalt.data.schema import CELLS_FILE, CURVES_FILE, CYCLES_FILE, MANIFEST_FILE
from cobalt.realdata.download import download
from cobalt.textio import write_csv, write_json

NOMINAL_CAPACITY_AH = 1.1
END_OF_LIFE_CAPACITY_AH = 0.88
#: Last-record capacity at or below which a stopped test counts as having reached
#: end of life. From the authors' LoadData.m.
FINISHED_ENDCAP_AH = 0.885
#: Every cell in the dataset was discharged at 4C (Severson et al., Methods).
DISCHARGE_C_RATE = 4.0
#: Discharge curves retained for the early-life features, by cycle number.
CURVE_CYCLES = (2, 10, 100)

_URL = "https://data.matr.io/1/api/v1/file/{}/download"


@dataclass(frozen=True)
class Batch:
    batch_id: str
    prefix: str
    filename: str
    file_id: str
    size_bytes: int

    @property
    def url(self) -> str:
        return _URL.format(self.file_id)


BATCHES = (
    Batch(
        "2017-05-12",
        "b1",
        "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "5c86c0b5fa2ede00015ddf66",
        3_025_320_241,
    ),
    Batch(
        "2017-06-30",
        "b2",
        "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        "5c86bf13fa2ede00015ddd82",
        2_007_331_155,
    ),
    Batch(
        "2018-04-12",
        "b3",
        "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
        "5c86bd64fa2ede00015ddbb2",
        3_236_690_412,
    ),
)

#: Batch 2 channels that are the continuation of batch 1 cells, from the
#: authors' Load Data.ipynb: (batch 1 cell, batch 2 continuation).
CONTINUATIONS = (
    ("b1c0", "b2c7"),
    ("b1c1", "b2c8"),
    ("b1c2", "b2c9"),
    ("b1c3", "b2c15"),
    ("b1c4", "b2c16"),
)
#: Batch 3 channels the authors removed as noisy, from the same notebook.
NOISY_CELLS = ("b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43")

_SUMMARY_KEYS = {
    "discharge_capacity_ah": "QDischarge",
    "charge_capacity_ah": "QCharge",
    "internal_resistance_ohm": "IR",
    "mean_temperature_c": "Tavg",
    "max_temperature_c": "Tmax",
    "charge_time_min": "chargetime",
}


@dataclass
class RawCell:
    cell_id: str
    batch_id: str
    policy: str
    series: dict[str, np.ndarray]
    curves: dict[int, np.ndarray]


def download_all(raw_dir: Path, connections: int = 8, log=print) -> list[Path]:
    return [
        download(batch.url, Path(raw_dir) / batch.filename, connections=connections, log=log)
        for batch in BATCHES
    ]


def first_c_rate(policy: str) -> float:
    """The first fast-charge step, in C. ``"5.4C(40%)-3.6C"`` gives 5.4."""
    match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)C", policy)
    if match is None:
        raise ValueError(f"cannot read a C-rate from policy {policy!r}")
    return float(match.group(1))


def _read_batch(path: Path, batch: Batch, log=print) -> tuple[dict[str, RawCell], np.ndarray]:
    import h5py

    started = time.perf_counter()
    cells: dict[str, RawCell] = {}
    with h5py.File(path, "r") as handle:
        group = handle["batch"]
        voltage = handle[group["Vdlin"][0, 0]][()].ravel().astype(float)
        mid_voltage = 0.5 * (voltage[1:] + voltage[:-1])
        for i in range(group["summary"].shape[0]):
            summary = handle[group["summary"][i, 0]]
            cycles = handle[group["cycles"][i, 0]]
            policy = handle[group["policy_readable"][i, 0]][()].tobytes()[::2].decode()
            series = {name: summary[key][0, :].astype(float) for name, key in _SUMMARY_KEYS.items()}
            references = cycles["Qdlin"][:, 0]
            n = min(len(series["discharge_capacity_ah"]), len(references))
            series = {name: values[:n] for name, values in series.items()}

            energy = np.empty(n)
            curves: dict[int, np.ndarray] = {}
            for j in range(n):
                q = handle[references[j]][()].ravel().astype(float)
                # Energy between 3.5 V and 2.0 V, the window the publisher
                # interpolated Qdlin over, by the midpoint rule.
                energy[j] = float(np.sum(mid_voltage * np.diff(q)))
                if j + 1 in CURVE_CYCLES:
                    curves[j + 1] = q
            series["discharge_energy_wh"] = energy
            cell_id = f"{batch.prefix}c{i}"
            cells[cell_id] = RawCell(cell_id, batch.batch_id, policy, series, curves)
    log(f"{batch.filename}: {len(cells)} cells in {time.perf_counter() - started:.0f}s")
    return cells, voltage


def _join(first: RawCell, continuation: RawCell) -> RawCell:
    series = {
        name: np.concatenate([first.series[name], continuation.series[name]])
        for name in first.series
    }
    return RawCell(first.cell_id, first.batch_id, first.policy, series, first.curves)


#: Capacity above this is a logging glitch: 118% of nominal and no cell in the
#: dataset delivers more than about 1.1 Ah on a real cycle.
MAX_PLAUSIBLE_CAPACITY_AH = 1.3
#: Discharge energy further than this from the cell's own median, as a fraction
#: of it, is a glitch. Capacity fade to end of life moves it by about 20%.
MAX_ENERGY_DEVIATION = 0.5
#: Charge time more than this many times the cell's own median, or less than
#: its reciprocal, is a glitch. Real charge time roughly doubles towards end of
#: life as the constant-voltage phase lengthens, so a tighter bound would cut
#: genuine ageing signal; the glitches are tens of times the median.
MAX_CHARGE_TIME_RATIO = 3.0


def implausible_records(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Masks of cycle records that cannot be physical, one per rule.

    About 0.1% of records in the published summaries are glitches: a discharge
    capacity of 2.9 Ah from a 1.1 Ah cell, a charge time of 1,290 minutes, a
    negative discharge energy. Left in, one of them dominates every slope and
    mean in the fifty-cycle window it lands in. The rules are deliberately wide,
    so they catch impossible values and nothing that could be ageing. Every
    record they drop is counted in the manifest.
    """
    energy = frame["discharge_energy_wh"]
    energy_median = float(energy.median())
    charge_time = frame["charge_time_min"]
    ratio = charge_time / float(charge_time.median())

    return {
        "capacity_above_1_3_ah": (frame["discharge_capacity_ah"] > MAX_PLAUSIBLE_CAPACITY_AH)
        | (frame["charge_capacity_ah"] > MAX_PLAUSIBLE_CAPACITY_AH),
        "resistance_not_positive": ~(frame["internal_resistance_ohm"] > 0),
        "charge_time_implausible": ~(
            (ratio <= MAX_CHARGE_TIME_RATIO) & (ratio >= 1.0 / MAX_CHARGE_TIME_RATIO)
        ),
        "energy_off_median": ~(
            (energy - energy_median).abs() <= MAX_ENERGY_DEVIATION * abs(energy_median)
        ),
    }


def label(discharge_capacity: np.ndarray) -> tuple[int, bool]:
    """``(cycle_life, censored)`` by the authors' rule.

    Position 1 of every test run is an empty record (all zeros) and a joined
    continuation brings a second one with it, so empty records are skipped
    rather than read as a capacity of zero.
    """
    capacity = np.asarray(discharge_capacity, dtype=float)
    recorded = np.isfinite(capacity) & (capacity > 0.01)
    recorded[0] = False
    below = np.flatnonzero(recorded & (capacity < END_OF_LIFE_CAPACITY_AH))
    if below.size:
        return int(below[0]) + 1, False
    n = len(capacity)
    last = capacity[np.flatnonzero(recorded)[-1]] if recorded.any() else np.inf
    if last <= FINISHED_ENDCAP_AH:
        return n + 1, False
    return n, True


def extract(
    raw_dir: Path,
    out_dir: Path,
    delete_raw: bool = False,
    fetch: bool = False,
    verify_size: bool = True,
    connections: int = 8,
    log=print,
) -> dict[str, object]:
    """Read the three batches and write a FleetDataset-format extract.

    With ``fetch`` a missing batch is downloaded just before it is read, and
    with ``delete_raw`` it is deleted straight after, so a CI runner never holds
    more than one 3 GB file at a time.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    missing = [b.filename for b in BATCHES if not (raw_dir / b.filename).exists()]
    if missing and not fetch:
        raise FileNotFoundError(
            f"{raw_dir} is missing {', '.join(missing)}. Run `cobalt real-data download` first."
        )

    cells: dict[str, RawCell] = {}
    grid: np.ndarray | None = None
    for batch in BATCHES:
        path = raw_dir / batch.filename
        if fetch:
            download(batch.url, path, connections=connections, log=log)
        if verify_size and path.stat().st_size != batch.size_bytes:
            raise OSError(
                f"{path.name} is {path.stat().st_size} bytes, expected {batch.size_bytes}"
            )
        read, voltage = _read_batch(path, batch, log=log)
        if grid is None:
            grid = voltage
        elif not np.allclose(grid, voltage):
            raise ValueError(f"{batch.filename} uses a different voltage grid")
        cells.update(read)
        if delete_raw:
            path.unlink()
            log(f"deleted {path.name} to save disk")
    assert grid is not None

    for first, continuation in CONTINUATIONS:
        cells[first] = _join(cells[first], cells.pop(continuation))
    for noisy in NOISY_CELLS:
        del cells[noisy]

    cycle_rows: list[pd.DataFrame] = []
    cell_rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {"__voltage_grid__": grid}
    empty_records = 0
    implausible: dict[str, int] = {}
    for cell in cells.values():
        series = cell.series
        n = len(series["discharge_capacity_ah"])
        cycle_life, censored = label(series["discharge_capacity_ah"])
        c_rate = first_c_rate(cell.policy)
        frame = pd.DataFrame({"cycle": np.arange(1, n + 1), **series})
        empty = ~(frame["discharge_capacity_ah"] > 0.01) | ~np.isfinite(
            frame["discharge_capacity_ah"]
        )
        empty_records += int(empty.sum())
        frame = frame[~empty].copy()
        bad = implausible_records(frame)
        for rule, mask in bad.items():
            implausible[rule] = implausible.get(rule, 0) + int(mask.sum())
        frame = frame[~np.logical_or.reduce(list(bad.values()))].copy()
        frame["cumulative_ah_throughput"] = frame["discharge_capacity_ah"].cumsum()
        frame.insert(0, "batch_id", cell.batch_id)
        frame.insert(0, "cell_id", cell.cell_id)
        frame["charge_c_rate"] = c_rate
        frame["discharge_c_rate"] = DISCHARGE_C_RATE
        frame["soh"] = frame["discharge_capacity_ah"] / NOMINAL_CAPACITY_AH
        cycle_rows.append(frame)
        cell_rows.append(
            {
                "cell_id": cell.cell_id,
                "batch_id": cell.batch_id,
                "cycle_life": cycle_life,
                "censored": censored,
                "observed_cycles": int(len(frame)),
                "nameplate_capacity_ah": NOMINAL_CAPACITY_AH,
                "charge_c_rate": c_rate,
                "discharge_c_rate": DISCHARGE_C_RATE,
                "charge_policy": cell.policy,
            }
        )
        for cycle, curve in cell.curves.items():
            curves[f"{cell.cell_id}|{cycle}"] = curve

    cycles_frame = pd.concat(cycle_rows, ignore_index=True)
    columns = [
        "cell_id",
        "batch_id",
        "cycle",
        "discharge_capacity_ah",
        "charge_capacity_ah",
        "internal_resistance_ohm",
        "mean_temperature_c",
        "max_temperature_c",
        "charge_c_rate",
        "discharge_c_rate",
        "charge_time_min",
        "discharge_energy_wh",
        "cumulative_ah_throughput",
        "soh",
    ]
    cells_frame = pd.DataFrame(cell_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(cycles_frame[columns], out_dir / CYCLES_FILE)
    write_csv(cells_frame, out_dir / CELLS_FILE)
    np.savez_compressed(out_dir / CURVES_FILE, **curves)

    finished = cells_frame[~cells_frame["censored"]]
    manifest = {
        "dataset": "severson-2019",
        "source": "https://data.matr.io/1/",
        "files": [{"name": b.filename, "bytes": b.size_bytes, "url": b.url} for b in BATCHES],
        "n_cells": int(len(cells_frame)),
        "n_batches": int(cells_frame["batch_id"].nunique()),
        "cells_per_batch": {
            str(k): int(v) for k, v in cells_frame.groupby("batch_id").size().items()
        },
        "censored_cells": sorted(cells_frame.loc[cells_frame["censored"], "cell_id"]),
        "continuations_joined": [list(pair) for pair in CONTINUATIONS],
        "noisy_cells_removed": list(NOISY_CELLS),
        "empty_cycle_records_dropped": empty_records,
        "implausible_cycle_records_dropped": implausible,
        "cell_cycles": int(len(cycles_frame)),
        "cycle_life_min": int(finished["cycle_life"].min()),
        "cycle_life_median": int(finished["cycle_life"].median()),
        "cycle_life_max": int(finished["cycle_life"].max()),
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "end_of_life_capacity_ah": END_OF_LIFE_CAPACITY_AH,
    }
    write_json(out_dir / MANIFEST_FILE, manifest)
    log(
        f"wrote {manifest['n_cells']} cells ({len(manifest['censored_cells'])} censored), "
        f"{manifest['cell_cycles']:,} cell-cycles to {out_dir}"
    )
    return manifest
