"""Features from the first hundred cycles of a cell's life.

The commercial question this answers: a lot of cells arrives and somebody has
to decide what it is worth. Cycling a cell to end of life to find out takes
months of a test channel's time. Predicting from the first hundred cycles takes
a few days.

The feature set is the one from Severson et al., *Data-driven prediction of
battery cycle life before capacity degradation* (Nature Energy, 2019), which
found that the *variance* of the difference between two early discharge curves
carries almost all of the signal and that it does so long before the capacity
fade curve shows anything at all. Reimplemented here against simulated cells,
not against their data.

The key quantity is

    dQ(V) = Q_100(V) - Q_10(V)

the difference in discharged capacity at matched terminal voltage between cycle
100 and cycle 10. Its variance across the voltage window is the single strongest
feature in the original paper, because it picks up the resistance rise and the
shape change that a scalar capacity measurement averages away.

**It is not the strongest feature here and that is a finding about the
simulator rather than about batteries.** Fitted on this data the elastic net
leans on ``log_abs_fade_slope`` and ``charge_time_drift`` instead and gives
delta-Q a small coefficient. The reason is that the simulated capacity fade is
far smoother than a real cell's: the plain slope over a hundred cycles is a
clean, low-noise estimate here, whereas on real hardware it is buried in
measurement noise and cell-to-cell scatter, which is precisely the condition
that makes delta-Q worth the trouble. A simulator cannot reproduce the
motivation for a feature it has designed away and `docs/LIMITATIONS.md` records
this as one of the specific things real cells would change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cobalt.data.schema import FleetDataset

#: Cycle indices the delta-Q feature is taken between. Must both be present in
#: the retained curves, which `FleetConfig.curve_cycles` controls.
DELTA_Q_EARLY_CYCLE = 10
DELTA_Q_LATE_CYCLE = 100

#: Observation window. A cell that dies inside this cannot be predicted from it,
#: and is excluded with a count rather than dropped quietly.
OBSERVATION_CYCLES = 100

FEATURE_NAMES = (
    "log_var_delta_q",
    "log_abs_min_delta_q",
    "log_abs_mean_delta_q",
    "log_norm_delta_q",
    "capacity_cycle_2",
    "capacity_cycle_100",
    "capacity_fade_2_to_100",
    "capacity_max_minus_cycle_2",
    "log_abs_fade_slope",
    "log_abs_late_fade_slope",
    "resistance_cycle_2",
    "resistance_cycle_100",
    "resistance_rise_2_to_100",
    "mean_temperature",
    "max_temperature",
    "temperature_integral",
    "mean_charge_time",
    "charge_time_drift",
    "mean_coulombic_efficiency",
    "charge_c_rate",
)


def _safe_log10(value: float, floor: float = 1e-12) -> float:
    """log10 of a magnitude, floored so a degenerate cell gives a number not a warning."""
    return float(np.log10(max(abs(value), floor)))


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope, or zero when there is nothing to fit."""
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def delta_q(dataset: FleetDataset, cell_id: str) -> np.ndarray | None:
    """``Q_100(V) - Q_10(V)`` on the shared voltage grid, or ``None`` if unavailable.

    Grid points either curve never reached are dropped rather than filled, so the
    variance is taken over the voltage window both cycles genuinely covered.
    """
    early = dataset.curves.get((cell_id, DELTA_Q_EARLY_CYCLE))
    late = dataset.curves.get((cell_id, DELTA_Q_LATE_CYCLE))
    if early is None or late is None:
        return None
    difference = late - early
    difference = difference[np.isfinite(difference)]
    return difference if difference.size >= 8 else None


def cell_features(dataset: FleetDataset, cell_id: str) -> dict[str, float] | None:
    """The full early-life feature row for one cell, or ``None`` if it died too soon."""
    cycles = dataset.cycles_for(cell_id)
    window = cycles[cycles["cycle"] <= OBSERVATION_CYCLES]
    if len(window) < OBSERVATION_CYCLES:
        return None

    difference = delta_q(dataset, cell_id)
    if difference is None:
        return None

    cycle_index = window["cycle"].to_numpy(dtype=float)
    capacity = window["discharge_capacity_ah"].to_numpy(dtype=float)
    resistance = window["internal_resistance_ohm"].to_numpy(dtype=float)
    temperature = window["mean_temperature_c"].to_numpy(dtype=float)
    charge_time = window["charge_time_min"].to_numpy(dtype=float)
    charge_capacity = window["charge_capacity_ah"].to_numpy(dtype=float)

    at = {int(c): i for i, c in enumerate(cycle_index)}
    capacity_2 = capacity[at[2]]
    capacity_100 = capacity[at[OBSERVATION_CYCLES]]
    resistance_2 = resistance[at[2]]
    resistance_100 = resistance[at[OBSERVATION_CYCLES]]

    fade_mask = cycle_index >= 2
    late_mask = cycle_index >= OBSERVATION_CYCLES - 20

    features = {
        # The delta-Q block. In the original paper the variance alone gets within
        # a few per cent of the full model. Here it is real but secondary, for the
        # reason set out in the module docstring.
        "log_var_delta_q": _safe_log10(float(np.var(difference))),
        "log_abs_min_delta_q": _safe_log10(float(np.min(difference))),
        "log_abs_mean_delta_q": _safe_log10(float(np.mean(difference))),
        "log_norm_delta_q": _safe_log10(float(np.linalg.norm(difference))),
        # Capacity fade, which is what an engineer looks at and which is
        # deliberately near-flat this early in life.
        "capacity_cycle_2": float(capacity_2),
        "capacity_cycle_100": float(capacity_100),
        "capacity_fade_2_to_100": float(capacity_2 - capacity_100),
        "capacity_max_minus_cycle_2": float(np.max(capacity) - capacity_2),
        "log_abs_fade_slope": _safe_log10(_slope(cycle_index[fade_mask], capacity[fade_mask])),
        "log_abs_late_fade_slope": _safe_log10(_slope(cycle_index[late_mask], capacity[late_mask])),
        # Power fade. Resistance moves earlier than capacity does.
        "resistance_cycle_2": float(resistance_2),
        "resistance_cycle_100": float(resistance_100),
        "resistance_rise_2_to_100": float(resistance_100 - resistance_2),
        # Thermal history, the strongest driver of the calendar term.
        "mean_temperature": float(np.mean(temperature)),
        "max_temperature": float(np.max(temperature)),
        "temperature_integral": float(np.trapezoid(temperature, cycle_index)),
        # Charge behaviour. Charge time creeping up is resistance rise seen from
        # the charger rather than from a dedicated measurement.
        "mean_charge_time": float(np.mean(charge_time)),
        "charge_time_drift": _slope(cycle_index, charge_time),
        "mean_coulombic_efficiency": float(np.mean(capacity / np.maximum(charge_capacity, 1e-9))),
        # Known at commissioning, not measured: the charge policy the site runs.
        "charge_c_rate": float(window["charge_c_rate"].iloc[0]),
    }
    return features


def build_table(dataset: FleetDataset) -> tuple[pd.DataFrame, dict[str, int]]:
    """Feature table for the early-life task, one row per usable cell.

    Returns the table and a count of what was excluded and why. Cells that reach
    end of life inside the observation window cannot be predicted from that
    window and cells that were still alive at the cycle ceiling have no label.
    """
    rows: list[dict[str, object]] = []
    excluded = {"died_within_window": 0, "censored": 0, "missing_curves": 0}

    for _, cell in dataset.cells.iterrows():
        cell_id = str(cell["cell_id"])
        if bool(cell["censored"]):
            excluded["censored"] += 1
            continue

        features = cell_features(dataset, cell_id)
        if features is None:
            cycles = dataset.cycles_for(cell_id)
            key = "died_within_window" if len(cycles) < OBSERVATION_CYCLES else "missing_curves"
            excluded[key] += 1
            continue

        rows.append(
            {
                "cell_id": cell_id,
                "batch_id": str(cell["batch_id"]),
                "cycle_life": int(cell["cycle_life"]),
                "log_cycle_life": float(np.log10(max(1, int(cell["cycle_life"])))),
                **features,
            }
        )

    frame = pd.DataFrame(
        rows, columns=["cell_id", "batch_id", "cycle_life", "log_cycle_life", *FEATURE_NAMES]
    )
    return frame, excluded
