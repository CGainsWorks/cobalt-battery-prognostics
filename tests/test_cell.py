"""The cell model has to behave like a cell before anything above it means much."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cobalt.sim.cell import (
    Cell,
    CellParameters,
    DutyCycle,
    open_circuit_voltage,
    state_of_charge_at_voltage,
)


def run(cell: Cell, cycles: int) -> list:
    return [cell.step() for _ in range(cycles)]


def test_open_circuit_voltage_is_monotonic():
    soc = np.linspace(0.0, 1.0, 500)
    voltage = open_circuit_voltage(soc)
    assert np.all(np.diff(voltage) > 0)
    assert 3.15 < voltage[0] < 3.25
    assert 4.15 < voltage[-1] < 4.25


def test_voltage_and_state_of_charge_round_trip():
    soc = np.linspace(0.02, 0.98, 40)
    recovered = state_of_charge_at_voltage(open_circuit_voltage(soc))
    # Two independent monotone interpolants, not an analytic inverse pair, so
    # the round trip is close rather than exact. Half a percent of state of
    # charge is far below the measurement noise on the same quantity.
    assert np.allclose(recovered, soc, atol=8e-3)


def test_state_of_charge_at_voltage_clips_outside_the_window():
    assert state_of_charge_at_voltage(1.0) == pytest.approx(0.0)
    assert state_of_charge_at_voltage(9.9) == pytest.approx(1.0)


def test_capacity_falls_and_resistance_rises_monotonically():
    cell = Cell(CellParameters(), DutyCycle(), rng=np.random.default_rng(0))
    records = run(cell, 300)
    capacity = [record.true_capacity_ah for record in records]
    resistance = [record.true_resistance_ohm for record in records]

    # The noise-free series must be monotonic. The measured ones need not be,
    # and asserting on those would be asserting on the noise.
    assert all(later <= earlier + 1e-12 for earlier, later in zip(capacity, capacity[1:]))
    assert all(later >= earlier - 1e-12 for earlier, later in zip(resistance, resistance[1:]))


def test_measurement_noise_is_actually_present():
    cell = Cell(CellParameters(), DutyCycle(), rng=np.random.default_rng(1))
    records = run(cell, 60)
    measured = np.array([record.discharge_capacity_ah for record in records])
    truth = np.array([record.true_capacity_ah for record in records])
    assert not np.allclose(measured, truth)
    assert np.std(measured - truth) > 1e-4


def test_hotter_cells_age_faster():
    """Arrhenius, in the direction the Arrhenius term says."""
    cool = Cell(
        CellParameters(), DutyCycle(ambient_temperature_c=15.0), rng=np.random.default_rng(2)
    )
    hot = Cell(
        CellParameters(), DutyCycle(ambient_temperature_c=50.0), rng=np.random.default_rng(2)
    )
    run(cool, 250)
    run(hot, 250)
    assert hot.state_of_health < cool.state_of_health


def test_faster_charging_ages_faster():
    gentle = Cell(CellParameters(), DutyCycle(charge_c_rate=0.5), rng=np.random.default_rng(3))
    aggressive = Cell(CellParameters(), DutyCycle(charge_c_rate=4.0), rng=np.random.default_rng(3))
    run(gentle, 250)
    run(aggressive, 250)
    assert aggressive.state_of_health < gentle.state_of_health


def test_the_knee_makes_fade_accelerate():
    """Late fade must be steeper than early fade, or there is no knee.

    Run to end of life rather than to a fixed cycle count: where the knee falls
    depends on the duty cycle, so a fixed horizon tests the horizon rather than
    the knee.
    """
    cell = Cell(CellParameters(), DutyCycle(charge_c_rate=2.0), rng=np.random.default_rng(4))
    capacity = []
    while cell.state_of_health > 0.78 and cell.cycle < 4000:
        capacity.append(cell.step().true_capacity_ah)
    capacity = np.array(capacity)
    assert len(capacity) > 400, "the cell never reached end of life"

    early = capacity[50] - capacity[150]
    late = capacity[-101] - capacity[-1]
    assert late > early * 1.5


def test_resistance_rise_brings_the_cut_off_forward():
    cell = Cell(CellParameters(), DutyCycle(discharge_c_rate=2.0), rng=np.random.default_rng(5))
    first = cell.cutoff_state_of_charge()
    run(cell, 700)
    assert cell.cutoff_state_of_charge() > first


def test_deliverable_capacity_never_exceeds_true_capacity():
    cell = Cell(CellParameters(), DutyCycle(discharge_c_rate=3.0), rng=np.random.default_rng(6))
    for _ in range(200):
        cell.step()
        assert cell.deliverable_capacity_ah() <= cell.true_capacity_ah + 1e-12


def test_discharge_curve_is_descending_in_voltage_and_ascending_in_capacity():
    cell = Cell(CellParameters(), DutyCycle(), rng=np.random.default_rng(7))
    run(cell, 20)
    capacity, voltage = cell.discharge_curve(points=128)
    assert np.all(np.diff(capacity) >= -1e-12)
    assert np.all(np.diff(voltage) <= 1e-12)


def test_capacity_at_voltage_is_nan_outside_the_window():
    cell = Cell(CellParameters(), DutyCycle(), rng=np.random.default_rng(8))
    run(cell, 10)
    grid = np.array([5.0, 3.7, 1.0])
    values = cell.capacity_at_voltage(grid)
    assert math.isnan(values[0])
    assert not math.isnan(values[1])
    assert math.isnan(values[2])


def test_delta_q_grows_with_age():
    """The feature the early-life model leans on must carry a signal."""
    cell = Cell(CellParameters(), DutyCycle(charge_c_rate=2.0), rng=np.random.default_rng(9))
    grid = np.linspace(3.82, 3.34, 60)

    run(cell, 10)
    reference = cell.capacity_at_voltage(grid)
    run(cell, 90)
    early = cell.capacity_at_voltage(grid)
    run(cell, 300)
    late = cell.capacity_at_voltage(grid)

    assert np.nanvar(late - reference) > np.nanvar(early - reference)


def test_simulation_is_deterministic_given_a_seed():
    def trace() -> list[float]:
        cell = Cell(CellParameters(), DutyCycle(), rng=np.random.default_rng(1234))
        return [record.discharge_capacity_ah for record in run(cell, 50)]

    assert trace() == trace()


def test_scaled_multiplies_rather_than_replaces():
    base = CellParameters(k_sei=4.0e-3)
    scaled = base.scaled(k_sei=2.0)
    assert scaled.k_sei == pytest.approx(8.0e-3)
    # Everything else is untouched.
    assert scaled.k_plate == base.k_plate


def test_knee_multiplier_does_not_overflow_late_in_a_long_run():
    """knee_sharpness is large and the exponential either side of it must be safe."""
    cell = Cell(
        CellParameters(knee_sharpness=400.0),
        DutyCycle(charge_c_rate=5.0, ambient_temperature_c=55.0),
        rng=np.random.default_rng(11),
    )
    for _ in range(1500):
        record = cell.step()
        assert math.isfinite(record.true_capacity_ah)
        assert math.isfinite(record.true_resistance_ohm)
