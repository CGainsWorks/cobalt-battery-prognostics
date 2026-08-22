"""The hardware abstraction layer, including the half of it that does not exist.

The last two tests are the important ones. They assert that the hardware-backed
source raises rather than returning something plausible, which is the difference
between an honest gap and a silent lie. If somebody implements `SerialBms` they
will fail, and that is the correct signal.
"""

from __future__ import annotations

import pytest

from cobalt.hal.base import BmsSource, CycleTelemetry, NoHardwareError
from cobalt.hal.serial_bms import SerialBms
from cobalt.hal.simulated import ReplayBms, SimulatedBms
from cobalt.sim.cell import CellParameters, DutyCycle


def test_simulated_source_satisfies_the_protocol():
    assert isinstance(SimulatedBms(), BmsSource)
    assert isinstance(ReplayBms("P1", []), BmsSource)
    assert isinstance(SerialBms(port="COM3", pack_id="P1"), BmsSource)


def test_simulated_source_streams_cycles_in_order():
    with SimulatedBms(pack_id="P1", seed=1, max_cycles=25) as source:
        cycles = []
        while (record := source.read_cycle()) is not None:
            cycles.append(record)
    assert [record.cycle for record in cycles] == list(range(1, 26))
    assert all(record.pack_id == "P1" for record in cycles)


def test_the_simulated_source_stops_at_the_configured_state_of_health():
    source = SimulatedBms(seed=2, max_cycles=6000, stop_at_soh=0.85)
    source.open()
    last = None
    while (record := source.read_cycle()) is not None:
        last = record
    source.close()
    assert last is not None
    assert last.discharge_capacity_ah / 2.20 < 0.90


def test_opening_twice_does_not_restart_the_cell():
    source = SimulatedBms(seed=3, max_cycles=10)
    source.open()
    first = source.read_cycle()
    source.open()
    second = source.read_cycle()
    source.close()
    assert first is not None and second is not None
    assert second.cycle == first.cycle + 1


def test_closing_resets_the_source():
    source = SimulatedBms(seed=3, max_cycles=10)
    source.open()
    source.read_cycle()
    source.close()
    source.open()
    assert source.read_cycle().cycle == 1


def test_the_same_seed_gives_the_same_stream():
    def stream() -> list[float]:
        with SimulatedBms(seed=42, max_cycles=15) as source:
            out = []
            while (record := source.read_cycle()) is not None:
                out.append(record.discharge_capacity_ah)
            return out

    assert stream() == stream()


def test_a_configured_duty_cycle_reaches_the_telemetry():
    duty = DutyCycle(charge_c_rate=2.5, discharge_c_rate=1.5)
    with SimulatedBms(parameters=CellParameters(), duty=duty, max_cycles=3) as source:
        record = source.read_cycle()
    assert record.charge_c_rate == pytest.approx(2.5)
    assert record.discharge_c_rate == pytest.approx(1.5)


def test_telemetry_serialises_to_a_dictionary():
    with SimulatedBms(max_cycles=1) as source:
        record = source.read_cycle()
    payload = record.as_dict()
    assert payload["pack_id"] == record.pack_id
    assert set(payload) >= {"cycle", "discharge_capacity_ah", "internal_resistance_ohm"}


def test_replay_returns_exactly_what_it_was_given():
    with SimulatedBms(seed=5, max_cycles=8) as source:
        recorded = []
        while (record := source.read_cycle()) is not None:
            recorded.append(record)

    replay = ReplayBms("P2", recorded)
    replay.open()
    played = []
    while (record := replay.read_cycle()) is not None:
        played.append(record)
    assert played == recorded
    assert replay.read_cycle() is None


def test_replay_can_be_rewound():
    telemetry = [
        CycleTelemetry(
            pack_id="P3",
            cycle=i,
            discharge_capacity_ah=2.0,
            charge_capacity_ah=2.01,
            internal_resistance_ohm=0.03,
            mean_temperature_c=25.0,
            max_temperature_c=26.0,
            charge_c_rate=1.0,
            discharge_c_rate=1.0,
            charge_time_min=60.0,
            discharge_energy_wh=7.4,
            cumulative_ah_throughput=2.0 * i,
        )
        for i in range(1, 4)
    ]
    replay = ReplayBms("P3", telemetry)
    replay.open()
    assert replay.read_cycle().cycle == 1
    replay.open()
    assert replay.read_cycle().cycle == 1


# -- the gap ------------------------------------------------------------


def test_the_hardware_source_refuses_to_open():
    source = SerialBms(port="COM3", pack_id="PACK-1")
    with pytest.raises(NoHardwareError, match="docs/HARDWARE.md"):
        source.open()


def test_the_hardware_source_refuses_to_read():
    source = SerialBms(port="/dev/ttyUSB0", pack_id="PACK-2")
    with pytest.raises(NoHardwareError):
        source.read_cycle()


def test_the_hardware_error_is_both_a_missing_feature_and_a_link_failure():
    """Callers check for either, and both should catch this."""
    assert issubclass(NoHardwareError, NotImplementedError)
    assert issubclass(NoHardwareError, RuntimeError)


def test_the_hardware_source_can_still_be_configured_and_inspected():
    """Wiring and dependency injection must be testable without a cycler."""
    source = SerialBms(port="COM7", pack_id="PACK-3", baudrate=57_600)
    described = source.describe()
    assert described["implemented"] is False
    assert described["baudrate"] == 57_600
    assert "HARDWARE.md" in str(described["reason"])


def test_closing_an_unopened_hardware_source_is_safe():
    """So a `finally: source.close()` in a caller stays correct."""
    SerialBms(port="COM9", pack_id="PACK-4").close()
