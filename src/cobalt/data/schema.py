"""On-disk representation of a simulated fleet.

Three files, because they have three different shapes and three different sizes:

``cycles.csv.gz``
    One row per cell-cycle. Everything a battery management system logs. This is
    the big one and it is the input to the online state-of-health task.

``cells.csv``
    One row per cell. Identity, batch, how it was used and the label: cycles to
    end of life. This is the input to the early-life task.

``curves.npz``
    The discharge capacity-against-voltage curves retained at a handful of early
    cycles, plus the shared voltage grid. Only the delta-Q features need these
    and keeping them out of the wide table stops it doubling in size.

Nothing here is committed to the repository. `make data` regenerates all of it
from a seed in under a minute, so the alternative is carrying tens of megabytes
of derived files that a one-line config change invalidates.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from cobalt.sim.cell import CellParameters, DutyCycle
from cobalt.sim.fleet import CellHistory, Fleet, FleetConfig

CYCLES_FILE = "cycles.csv.gz"
CELLS_FILE = "cells.csv"
CURVES_FILE = "curves.npz"
MANIFEST_FILE = "manifest.json"

#: Columns a model is allowed to see. `true_` columns are ground truth for the
#: evaluation and are filtered out before any feature is computed.
OBSERVABLE_CYCLE_COLUMNS = (
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
)


def cycles_frame(fleet: Fleet) -> pd.DataFrame:
    """Long table, one row per cell-cycle."""
    rows: list[dict[str, object]] = []
    for cell in fleet.cells:
        rated = cell.parameters.nameplate_capacity_ah
        for record in cell.records:
            row = asdict(record)
            row["cell_id"] = cell.cell_id
            row["batch_id"] = cell.batch_id
            row["soh"] = record.discharge_capacity_ah / rated
            row["true_soh"] = record.true_capacity_ah / rated
            rows.append(row)
    frame = pd.DataFrame(rows)
    leading = ["cell_id", "batch_id", "cycle"]
    return frame[leading + [c for c in frame.columns if c not in leading]]


def cells_frame(fleet: Fleet) -> pd.DataFrame:
    """Wide table, one row per cell, carrying the cycle-life label."""
    rows: list[dict[str, object]] = []
    for cell in fleet.cells:
        observed = len(cell.records)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "batch_id": cell.batch_id,
                "cycle_life": cell.cycle_life if cell.cycle_life is not None else observed,
                "censored": cell.censored,
                "observed_cycles": observed,
                "nameplate_capacity_ah": cell.parameters.nameplate_capacity_ah,
                "charge_c_rate": cell.duty.charge_c_rate,
                "discharge_c_rate": cell.duty.discharge_c_rate,
                "ambient_temperature_c": cell.duty.ambient_temperature_c,
                "depth_of_discharge": cell.duty.depth_of_discharge,
                "rest_hours_per_cycle": cell.duty.rest_hours_per_cycle,
            }
        )
    return pd.DataFrame(rows)


def write_fleet(fleet: Fleet, directory: Path) -> dict[str, str]:
    """Write all three files plus a manifest and return the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    cycles_path = directory / CYCLES_FILE
    cells_path = directory / CELLS_FILE
    curves_path = directory / CURVES_FILE
    manifest_path = directory / MANIFEST_FILE

    cycles_frame(fleet).to_csv(cycles_path, index=False, float_format="%.6g")
    cells_frame(fleet).to_csv(cells_path, index=False, float_format="%.6g")

    arrays: dict[str, np.ndarray] = {"__voltage_grid__": fleet.voltage_grid}
    for cell in fleet.cells:
        for cycle, curve in cell.voltage_curves.items():
            arrays[f"{cell.cell_id}|{cycle}"] = curve
    np.savez_compressed(curves_path, **arrays)

    manifest = {
        "seed": fleet.seed,
        "n_cells": len(fleet.cells),
        "n_batches": len(fleet.batch_ids),
        "censored_cells": sum(1 for cell in fleet.cells if cell.censored),
        "config": _config_to_json(fleet.config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "cycles": str(cycles_path),
        "cells": str(cells_path),
        "curves": str(curves_path),
        "manifest": str(manifest_path),
    }


def _config_to_json(config: FleetConfig) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in asdict(config).items():
        out[key] = list(value) if isinstance(value, tuple) else value
    return out


class FleetDataset:
    """Loaded view of a written fleet, with the curve lookup already unpacked."""

    def __init__(
        self,
        cycles: pd.DataFrame,
        cells: pd.DataFrame,
        curves: dict[tuple[str, int], np.ndarray],
        voltage_grid: np.ndarray,
        manifest: dict[str, object],
    ) -> None:
        self.cycles = cycles
        self.cells = cells
        self.curves = curves
        self.voltage_grid = voltage_grid
        self.manifest = manifest

    @classmethod
    def load(cls, directory: Path) -> FleetDataset:
        directory = Path(directory)
        missing = [
            name
            for name in (CYCLES_FILE, CELLS_FILE, CURVES_FILE, MANIFEST_FILE)
            if not (directory / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{directory} is not a fleet dataset, missing {', '.join(missing)}. "
                f"Run `cobalt generate --out {directory}` first."
            )

        cycles = pd.read_csv(directory / CYCLES_FILE)
        cells = pd.read_csv(directory / CELLS_FILE)
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))

        with np.load(directory / CURVES_FILE) as handle:
            voltage_grid = handle["__voltage_grid__"]
            curves = {}
            for key in handle.files:
                if key == "__voltage_grid__":
                    continue
                cell_id, cycle = key.rsplit("|", 1)
                curves[(cell_id, int(cycle))] = handle[key]

        return cls(cycles, cells, curves, voltage_grid, manifest)

    def cycles_for(self, cell_id: str) -> pd.DataFrame:
        return self.cycles[self.cycles["cell_id"] == cell_id].sort_values("cycle")

    @property
    def cell_ids(self) -> list[str]:
        return list(self.cells["cell_id"])

    def batch_of(self, cell_id: str) -> str:
        matches = self.cells.loc[self.cells["cell_id"] == cell_id, "batch_id"]
        if matches.empty:
            raise KeyError(cell_id)
        return str(matches.iloc[0])


def describe_defaults() -> dict[str, object]:
    """The default cell and duty cycle, for the model card and the docs."""
    return {"cell": asdict(CellParameters()), "duty": asdict(DutyCycle())}


def history_summary(cell: CellHistory) -> dict[str, object]:
    """Compact human-readable summary of one simulated cell."""
    return {
        "cell_id": cell.cell_id,
        "batch_id": cell.batch_id,
        "cycle_life": cell.cycle_life,
        "censored": cell.censored,
        "observed_cycles": len(cell.records),
        "charge_c_rate": round(cell.duty.charge_c_rate, 3),
        "ambient_temperature_c": round(cell.duty.ambient_temperature_c, 2),
    }
