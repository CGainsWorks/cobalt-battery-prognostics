"""The CAN link: the DBC, the simulated BMS node and the receiver, end to end.

Everything here runs on python-can's in-process ``virtual`` interface, so it
needs no kernel module, no adapter and no privileges. Continuous integration
additionally runs the node and the receiver as separate processes over a Linux
``vcan0`` interface when the runner allows it; that is a workflow step rather
than a test because it needs root to create the interface.

The tolerance used throughout is half of each signal's least significant bit,
read from the DBC itself. Anything wider would hide a wrong scale factor and
anything narrower would be asking the wire to carry more precision than the
DBC says it does.
"""

from __future__ import annotations

import dataclasses
import json
import uuid

import can
import pytest

from cobalt.cli import main as cli_main
from cobalt.hal import can_protocol as protocol
from cobalt.hal.base import BmsSource, CycleTelemetry
from cobalt.hal.can_bms import CanBms
from cobalt.hal.can_node import SimulatedBmsNode
from cobalt.hal.simulated import SimulatedBms


def _channel() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


def _direct(seed: int, max_cycles: int) -> list[CycleTelemetry]:
    with SimulatedBms(pack_id="P1", seed=seed, max_cycles=max_cycles) as source:
        out = []
        while (record := source.read_cycle()) is not None:
            out.append(record)
    return out


def _over_can(seed: int, max_cycles: int, bus_wrapper=None) -> tuple[list[CycleTelemetry], CanBms]:
    channel = _channel()
    receiver = CanBms(pack_id="P1", channel=channel, first_frame_timeout_s=5.0, idle_timeout_s=0.5)
    receiver.open()
    sender = can.Bus(interface="virtual", channel=channel)
    bus = bus_wrapper(sender) if bus_wrapper else sender
    node = SimulatedBmsNode(SimulatedBms(pack_id="P1", seed=seed, max_cycles=max_cycles), bus)
    node.start()
    received = []
    try:
        while (record := receiver.read_cycle()) is not None:
            received.append(record)
    finally:
        node.stop()
        sender.shutdown()
        receiver.close()
    assert node.error is None
    return received, receiver


# -- the DBC -------------------------------------------------------------------


def test_the_dbc_loads_strictly_with_unique_ids():
    database = protocol.database()
    ids = [message.frame_id for message in database.messages]
    assert len(ids) == len(set(ids)) == len(protocol.LIVE_MESSAGES + protocol.SUMMARY_MESSAGES)
    assert all(message.length == 8 for message in database.messages)


def test_every_telemetry_field_has_exactly_one_signal():
    fields = {f.name for f in dataclasses.fields(CycleTelemetry)} - {"pack_id", "cycle"}
    assert set(protocol.SUMMARY_FIELDS) == fields


def test_every_frame_carries_a_counter_and_a_checksum_in_byte_seven():
    for message in protocol.database().messages:
        checksum = message.get_signal_by_name("Checksum")
        counter = message.get_signal_by_name("RollingCounter")
        assert (checksum.start, checksum.length) == (56, 8)
        assert counter.length == 4


def test_a_summary_frame_round_trips_within_half_a_bit():
    values = {"CycleIndex": 1234, "DischargeCapacity": 2.07391, "ChargeCapacity": 2.08012}
    frame_id, data = protocol.encode(protocol.CYCLE_CAPACITY, values, counter=21)
    name, decoded = protocol.decode(frame_id, data)
    assert name == protocol.CYCLE_CAPACITY
    assert decoded["CycleIndex"] == 1234
    assert decoded["RollingCounter"] == 21 % 16
    for key in ("DischargeCapacity", "ChargeCapacity"):
        assert abs(decoded[key] - values[key]) <= 0.00005 + 1e-12
    assert protocol.checksum_ok(frame_id, data)


def test_an_out_of_range_value_is_refused_rather_than_clamped():
    with pytest.raises(Exception, match="DischargeCapacity|range|7"):
        protocol.encode(
            protocol.CYCLE_CAPACITY,
            {"CycleIndex": 1, "DischargeCapacity": 7.0, "ChargeCapacity": 2.0},
            counter=0,
        )


# -- the receiver, frame by frame ---------------------------------------------------


def _frame(name: str, values: dict[str, float], counter: int = 0) -> can.Message:
    frame_id, data = protocol.encode(name, values, counter)
    return can.Message(arbitration_id=frame_id, data=data, is_extended_id=False)


def _summary_frames(cycle_index: int, counter: int = 0) -> list[can.Message]:
    values = {
        protocol.CYCLE_CAPACITY: {"DischargeCapacity": 2.1, "ChargeCapacity": 2.11},
        protocol.CYCLE_RESISTANCE: {"InternalResistance": 0.031, "ChargeTime": 61.5},
        protocol.CYCLE_THERMAL: {"MeanTemperature": 27.2, "MaxTemperature": 29.0},
        protocol.CYCLE_RATES: {"ChargeCRate": 1.0, "DischargeCRate": 1.0, "DischargeEnergy": 7.7},
        protocol.CYCLE_THROUGHPUT: {"CumulativeThroughput": 2.1 * (cycle_index + 1)},
    }
    return [
        _frame(name, {"CycleIndex": cycle_index, **signals}, counter)
        for name, signals in values.items()
    ]


def test_a_corrupted_frame_is_counted_and_ignored():
    receiver = CanBms(pack_id="P1")
    frames = _summary_frames(5)
    corrupted = bytearray(frames[0].data)
    corrupted[2] ^= 0x10
    frames[0] = can.Message(
        arbitration_id=frames[0].arbitration_id, data=bytes(corrupted), is_extended_id=False
    )
    results = [receiver.handle(frame) for frame in frames]
    assert all(result is None for result in results)
    assert receiver.stats.checksum_errors == 1


def test_frames_this_protocol_does_not_define_are_ignored():
    receiver = CanBms(pack_id="P1")
    assert (
        receiver.handle(can.Message(arbitration_id=0x7E8, data=bytes(8), is_extended_id=False))
        is None
    )
    assert receiver.handle(can.Message(arbitration_id=0x18FF50E5, data=bytes(8))) is None
    assert receiver.stats.foreign_frames == 2
    assert receiver.stats.frames_decoded == 0


def test_the_five_summary_frames_make_one_cycle_in_any_order():
    receiver = CanBms(pack_id="P1")
    frames = _summary_frames(42)
    results = [receiver.handle(frame) for frame in reversed(frames)]
    assert results[:-1] == [None] * 4
    telemetry = results[-1]
    assert telemetry.cycle == 42
    assert telemetry.pack_id == "P1"
    assert telemetry.discharge_capacity_ah == pytest.approx(2.1, abs=5e-5)


def test_the_cycle_index_is_unwrapped_past_sixteen_bits():
    receiver = CanBms(pack_id="P1")
    cycles = []
    for index in (65534, 65535, 0, 1):
        for frame in _summary_frames(index):
            out = receiver.handle(frame)
        cycles.append(out.cycle)
    assert cycles == [65534, 65535, 65536, 65537]


def test_a_counter_gap_is_reported_as_lost_frames():
    receiver = CanBms(pack_id="P1")
    receiver.handle(_frame(protocol.PACK_STATUS, _status(), counter=3))
    receiver.handle(_frame(protocol.PACK_STATUS, _status(), counter=7))
    assert receiver.stats.counter_gaps == 1
    assert receiver.stats.frames_lost_estimate == 3


def _status() -> dict[str, float]:
    return {"PackVoltage": 14.8, "PackCurrent": -2.2, "StateOfCharge": 50.0, "OperatingMode": 2}


def test_a_quiet_bus_ends_the_stream():
    receiver = CanBms(pack_id="P1", channel=_channel(), first_frame_timeout_s=0.2)
    with receiver:
        assert receiver.read_cycle() is None


def test_the_receiver_satisfies_the_source_protocol_and_describes_itself():
    receiver = CanBms(pack_id="P9", interface="socketcan", channel="vcan0")
    assert isinstance(receiver, BmsSource)
    described = receiver.describe()
    assert described["open"] is False
    assert described["channel"] == "vcan0"
    receiver.close()


# -- end to end over the virtual bus ---------------------------------------------


def test_telemetry_survives_the_bus_within_half_a_bit_per_field():
    direct = _direct(seed=11, max_cycles=60)
    received, receiver = _over_can(seed=11, max_cycles=60)

    assert [r.cycle for r in received] == [r.cycle for r in direct] == list(range(1, 61))
    for field_name in protocol.SUMMARY_FIELDS:
        half_bit = protocol.resolution(field_name) / 2 + 1e-9
        worst = max(
            abs(getattr(a, field_name) - getattr(b, field_name)) for a, b in zip(received, direct)
        )
        assert worst <= half_bit, f"{field_name} off by {worst}, more than half a bit"

    stats = receiver.stats
    assert (
        stats.checksum_errors == stats.frames_lost_estimate == stats.incomplete_cycles_dropped == 0
    )
    assert stats.cycles_assembled == 60
    assert stats.by_message[protocol.CELL_VOLTAGES] == 60 * 4


def test_live_frames_carry_plausible_values():
    _, receiver = _over_can(seed=3, max_cycles=5)
    live = receiver.live
    assert live.operating_mode == protocol.MODE_DISCHARGE
    assert live.pack_current_a == pytest.approx(-2.2, abs=0.01)
    assert len(live.cell_voltages_v) == protocol.SERIES_CELLS
    assert all(2.8 < v < 4.4 for v in live.cell_voltages_v)
    assert live.pack_voltage_v == pytest.approx(sum(live.cell_voltages_v), abs=0.02)
    assert live.temperature_min_c <= live.temperature_mean_c <= live.temperature_max_c


class _LossyBus:
    """Drops chosen frames on the way out, to stand in for a lossy link."""

    def __init__(self, bus: can.BusABC, drop: set[int]) -> None:
        self._bus = bus
        self._drop = drop
        self._sent = 0

    def send(self, message: can.Message, timeout: float | None = None) -> None:
        index = self._sent
        self._sent += 1
        if index not in self._drop:
            self._bus.send(message, timeout)


def test_a_lost_summary_frame_drops_that_cycle_and_says_so():
    # 17 frames per cycle with two live samples per phase: 12 live, then 5 summary.
    # Frame 12 of cycle 3 (zero-based frame 2 * 17 + 12) is that cycle's first summary frame.
    lost = 2 * 17 + 12
    received, receiver = _over_can(
        seed=5, max_cycles=8, bus_wrapper=lambda bus: _LossyBus(bus, {lost})
    )
    assert [r.cycle for r in received] == [1, 2, 4, 5, 6, 7, 8]
    assert receiver.stats.incomplete_cycles_dropped == 1
    assert receiver.stats.frames_lost_estimate == 1


def test_the_deployed_model_gives_the_same_answer_over_can(tmp_path):
    """The whole ingestion path: node, bus, DBC, receiver, features, model.

    Predictions are compared rather than asserted equal. The wire quantises
    every field to half a bit and the deployed gradient-boosted model has
    thresholds, so a quantised input can land on the other side of one. On the
    committed 1,200-cycle demo the largest change is 1.2% of the estimate, and
    the 3% tolerance here is set from that, not tuned to pass.
    """
    direct_log = tmp_path / "direct.json"
    can_log = tmp_path / "can.json"
    common = ["stream", "--max-cycles", "200", "--config", "config/default.yaml"]
    assert cli_main([*common, "--output", str(direct_log)]) == 0
    assert (
        cli_main([*common, "--source", "can", "--idle-timeout", "0.5", "--output", str(can_log)])
        == 0
    )

    direct = json.loads(direct_log.read_text(encoding="utf-8"))
    over_can = json.loads(can_log.read_text(encoding="utf-8"))
    assert over_can["source"] == "cobalt.hal.can_bms.CanBms"
    assert over_can["cycles_streamed"] == direct["cycles_streamed"] == 200
    assert over_can["link"]["incomplete_cycles_dropped"] == 0
    assert len(over_can["predictions"]) == len(direct["predictions"]) > 0
    for a, b in zip(direct["predictions"], over_can["predictions"]):
        assert a["cycle"] == b["cycle"]
        assert a["state_of_health"] == pytest.approx(b["state_of_health"], abs=1e-3)
        assert a["remaining_estimate"] == pytest.approx(b["remaining_estimate"], rel=0.03)
