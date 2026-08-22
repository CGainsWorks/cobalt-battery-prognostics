"""Features for the online task: how many cycles has this pack got left?

The early-life task grades a lot before it ships. This one runs for the rest of
the pack's life, on whatever the battery management system happened to log and
answers the question a maintenance planner asks: not "how healthy is it" but
"when do I have to have a replacement on the shelf".

Each row is a trailing window of ``WINDOW_CYCLES`` cycles ending at some cycle
``t`` and the label is the remaining useful life, ``cycle_life - t``. Windows
are taken every ``STRIDE`` cycles rather than every cycle, because consecutive
windows overlap by all but one sample and a hundred near-duplicate rows per cell
buys nothing but confidence in a number that is not real.

That overlap is also exactly why the ``random_cycle`` protocol is a lie for this
task. Two windows from the same cell twenty cycles apart share thirty of their
fifty samples. Split them at random across train and test and the model is being
scored on rows it has all but memorised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cobalt.data.schema import FleetDataset

#: Length of the trailing observation window, in cycles.
WINDOW_CYCLES = 50
#: Gap between consecutive window end points.
STRIDE = 25
#: Do not emit windows past end of life. Beyond it the pack is scheduled for
#: replacement and the prediction has no decision hanging off it.
MIN_REMAINING_CYCLES = 1

FEATURE_NAMES = (
    "soh_now",
    "soh_window_start",
    "soh_slope",
    "soh_curvature",
    "resistance_now",
    "resistance_ratio",
    "resistance_slope",
    "mean_temperature",
    "max_temperature",
    "mean_charge_time",
    "charge_time_slope",
    "mean_coulombic_efficiency",
    "log_throughput_ah",
    "log_cycle_index",
    "charge_c_rate",
    "energy_per_ah",
)


def _slope(x: np.ndarray, y: np.ndarray, degree: int = 1) -> np.ndarray:
    """Polynomial coefficients, highest order first, with a flat-input guard."""
    if len(x) <= degree or np.allclose(x, x[0]):
        return np.zeros(degree + 1)
    return np.polyfit(x, y, degree)


def window_features(window: pd.DataFrame, rated_capacity_ah: float) -> dict[str, float]:
    """Features from one trailing window.

    Note what is *not* here: the discharge capacity of a full reference
    discharge. A pack in service rarely gets one, so the state of health this
    uses is the one the management system estimates from partial discharges
    and every feature below is something a CAN bus already carries.
    """
    cycle = window["cycle"].to_numpy(dtype=float)
    capacity = window["discharge_capacity_ah"].to_numpy(dtype=float)
    resistance = window["internal_resistance_ohm"].to_numpy(dtype=float)
    temperature = window["mean_temperature_c"].to_numpy(dtype=float)
    charge_time = window["charge_time_min"].to_numpy(dtype=float)
    charge_capacity = window["charge_capacity_ah"].to_numpy(dtype=float)
    energy = window["discharge_energy_wh"].to_numpy(dtype=float)

    soh = capacity / rated_capacity_ah
    quadratic = _slope(cycle - cycle[0], soh, degree=2)

    return {
        "soh_now": float(soh[-1]),
        "soh_window_start": float(soh[0]),
        "soh_slope": float(quadratic[1]),
        # The knee is a change in the *second* derivative of the fade curve, so
        # this is the term that carries the warning rather than the level.
        "soh_curvature": float(quadratic[0]),
        "resistance_now": float(resistance[-1]),
        "resistance_ratio": float(resistance[-1] / max(resistance[0], 1e-9)),
        "resistance_slope": float(_slope(cycle - cycle[0], resistance)[0]),
        "mean_temperature": float(np.mean(temperature)),
        "max_temperature": float(np.max(temperature)),
        "mean_charge_time": float(np.mean(charge_time)),
        "charge_time_slope": float(_slope(cycle - cycle[0], charge_time)[0]),
        "mean_coulombic_efficiency": float(np.mean(capacity / np.maximum(charge_capacity, 1e-9))),
        "log_throughput_ah": float(
            np.log10(max(float(window["cumulative_ah_throughput"].iloc[-1]), 1e-6))
        ),
        "log_cycle_index": float(np.log10(max(cycle[-1], 1.0))),
        "charge_c_rate": float(window["charge_c_rate"].iloc[-1]),
        "energy_per_ah": float(np.mean(energy / np.maximum(capacity, 1e-9))),
    }


def build_table(
    dataset: FleetDataset,
    window_cycles: int = WINDOW_CYCLES,
    stride: int = STRIDE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Feature table for the online task, many rows per cell.

    Rows carry ``cell_id`` and ``batch_id`` so the grouped protocols in
    `cobalt.data.splits` have something to group on and ``cycle`` so a
    prediction can be traced back to the window it came from.
    """
    rows: list[dict[str, object]] = []
    excluded = {"censored": 0, "too_short": 0}

    for _, cell in dataset.cells.iterrows():
        cell_id = str(cell["cell_id"])
        if bool(cell["censored"]):
            excluded["censored"] += 1
            continue

        cycle_life = int(cell["cycle_life"])
        rated = float(cell["nameplate_capacity_ah"])
        cycles = dataset.cycles_for(cell_id).reset_index(drop=True)
        if len(cycles) < window_cycles + stride:
            excluded["too_short"] += 1
            continue

        indexed = cycles.set_index("cycle", drop=False)
        last_end = cycle_life - MIN_REMAINING_CYCLES
        for end in range(window_cycles, last_end + 1, stride):
            window = indexed.loc[end - window_cycles + 1 : end]
            if len(window) < window_cycles:
                continue
            remaining = cycle_life - end
            rows.append(
                {
                    "cell_id": cell_id,
                    "batch_id": str(cell["batch_id"]),
                    "cycle": end,
                    "cycle_life": cycle_life,
                    "remaining_cycles": remaining,
                    "log_remaining_cycles": float(np.log10(max(remaining, 1))),
                    **window_features(window, rated),
                }
            )

    frame = pd.DataFrame(
        rows,
        columns=[
            "cell_id",
            "batch_id",
            "cycle",
            "cycle_life",
            "remaining_cycles",
            "log_remaining_cycles",
            *FEATURE_NAMES,
        ],
    )
    return frame, excluded
