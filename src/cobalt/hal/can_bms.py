"""Cycle telemetry from a battery management system over CAN.

This replaces the old ``SerialBms`` stub. It listens on any python-can bus,
decodes frames with the repository's DBC through cantools, checks every frame's
checksum and rolling counter, keeps the latest live status and assembles the
five cycle summary frames of each cycle into one
:class:`~cobalt.hal.base.CycleTelemetry`. Everything upstream of it is
unchanged: the features, the models and the service never learn the telemetry
crossed a bus.

What it has been run against, honestly:

* python-can's in-process ``virtual`` interface, with
  :class:`~cobalt.hal.can_node.SimulatedBmsNode` in a thread. Every test and
  ``cobalt stream --source can`` use this.
* a Linux ``vcan0`` interface through ``socketcan``, with the simulated node
  running as a separate process. Continuous integration does this when the
  runner can load the ``vcan`` kernel module.

It has **not** been run against a physical CAN adapter or a real BMS. Nothing
in the code is specific to the virtual interfaces (the interface and channel
are constructor arguments and a PCAN, Kvaser or SocketCAN adapter is the same
call) but that is a statement about the code rather than a result.

Frame loss is detected rather than hidden. A gap in a message's rolling
counter counts the frames that went missing; a cycle whose five summary frames
never all arrive is dropped and counted, so the estimator sees a missing cycle
index rather than a cycle stitched from two different ones. Both counts are in
:attr:`CanBms.stats`, which is what the frame-loss acceptance test in
``docs/HARDWARE.md`` asks for.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import can

from cobalt.hal import can_protocol as protocol
from cobalt.hal.base import BmsError, BmsSource, CycleTelemetry


@dataclass
class LinkStats:
    """What came off the bus, for the stream log and the acceptance tests."""

    frames_received: int = 0
    frames_decoded: int = 0
    foreign_frames: int = 0
    checksum_errors: int = 0
    counter_gaps: int = 0
    frames_lost_estimate: int = 0
    cycles_assembled: int = 0
    incomplete_cycles_dropped: int = 0
    by_message: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LiveStatus:
    """The most recent live frames, decoded. ``None`` until the first one arrives."""

    pack_voltage_v: float | None = None
    pack_current_a: float | None = None
    state_of_charge_pct: float | None = None
    operating_mode: int | None = None
    cell_voltages_v: tuple[float, ...] | None = None
    temperature_mean_c: float | None = None
    temperature_max_c: float | None = None
    temperature_min_c: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CanBms(BmsSource):
    """Reads cycle telemetry for one pack from a CAN bus."""

    def __init__(
        self,
        pack_id: str,
        interface: str = "virtual",
        channel: str = "cobalt-bms",
        bus: can.BusABC | None = None,
        first_frame_timeout_s: float = 10.0,
        idle_timeout_s: float = 2.0,
        bitrate: int | None = None,
    ) -> None:
        self._pack_id = pack_id
        self.interface = interface
        self.channel = channel
        self.bitrate = bitrate
        self.first_frame_timeout_s = first_frame_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self._injected_bus = bus
        self._bus: can.BusABC | None = None
        self._reader: can.BufferedReader | None = None
        self._notifier: can.Notifier | None = None
        self._seen_any = False
        self._last_counter: dict[int, int] = {}
        self._pending: dict[int, dict[str, dict[str, float]]] = {}
        self._cycle_epoch = 0
        self._last_index: int | None = None
        self.stats = LinkStats()
        self.live = LiveStatus()

    @property
    def pack_id(self) -> str:
        return self._pack_id

    # -- lifecycle ------------------------------------------------------------

    def open(self) -> None:
        """Open the bus and start draining it in the background.

        A python-can Notifier thread moves frames off the socket into an
        unbounded queue as they arrive, so the estimator can spend tens of
        milliseconds on a prediction without the kernel's receive buffer
        overflowing behind it. Reading straight from the socket instead lost
        about 6% of frames on a vcan interface at two thousand frames a second.
        """
        if self._bus is not None:
            return
        if self._injected_bus is not None:
            bus = self._injected_bus
        else:
            kwargs: dict[str, object] = {"interface": self.interface, "channel": self.channel}
            if self.bitrate is not None:
                kwargs["bitrate"] = self.bitrate
            try:
                bus = can.Bus(**kwargs)
            except (can.CanError, OSError, ValueError, ImportError) as error:
                raise BmsError(
                    f"cannot open CAN {self.interface}:{self.channel}: {error}"
                ) from error
            bus.set_filters(
                [
                    {"can_id": frame_id, "can_mask": 0x7FF, "extended": False}
                    for frame_id in protocol.frame_ids()
                ]
            )
        self._bus = bus
        self._reader = can.BufferedReader()
        self._notifier = can.Notifier(bus, [self._reader], timeout=0.05)

    def close(self) -> None:
        if self._notifier is not None:
            self._notifier.stop()
            self._notifier = None
        if self._bus is not None and self._injected_bus is None:
            self._bus.shutdown()
        self._bus = None
        self._reader = None

    def __enter__(self) -> CanBms:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- frames ---------------------------------------------------------------

    def _track_counter(self, arbitration_id: int, counter: int) -> None:
        last = self._last_counter.get(arbitration_id)
        if last is not None:
            missing = (counter - last - 1) % protocol.COUNTER_MODULUS
            if missing:
                self.stats.counter_gaps += 1
                self.stats.frames_lost_estimate += missing
        self._last_counter[arbitration_id] = counter

    def _update_live(self, name: str, values: dict[str, float]) -> None:
        if name == protocol.PACK_STATUS:
            self.live.pack_voltage_v = values["PackVoltage"]
            self.live.pack_current_a = values["PackCurrent"]
            self.live.state_of_charge_pct = values["StateOfCharge"]
            self.live.operating_mode = int(values["OperatingMode"])
        elif name == protocol.CELL_VOLTAGES:
            self.live.cell_voltages_v = tuple(
                values[f"Cell{i + 1}Voltage"] for i in range(protocol.SERIES_CELLS)
            )
        elif name == protocol.TEMPERATURES:
            self.live.temperature_mean_c = values["CellTemperatureMean"]
            self.live.temperature_max_c = values["CellTemperatureMax"]
            self.live.temperature_min_c = values["CellTemperatureMin"]

    def _absolute_cycle(self, index: int) -> int:
        """Unwrap the 16-bit cycle index into a monotonic cycle count."""
        if (
            self._last_index is not None
            and index < self._last_index - protocol.CYCLE_INDEX_MODULUS // 2
        ):
            self._cycle_epoch += 1
        self._last_index = index
        return self._cycle_epoch * protocol.CYCLE_INDEX_MODULUS + index

    def _accept_summary(self, name: str, values: dict[str, float]) -> CycleTelemetry | None:
        cycle = self._absolute_cycle(int(values["CycleIndex"]))
        parts = self._pending.setdefault(cycle, {})
        parts[name] = values
        if len(parts) < len(protocol.SUMMARY_MESSAGES):
            return None

        del self._pending[cycle]
        stale = [pending for pending in self._pending if pending < cycle]
        for pending in stale:
            del self._pending[pending]
        self.stats.incomplete_cycles_dropped += len(stale)

        fields = {
            attribute: parts[message][signal]
            for attribute, (message, signal) in protocol.SUMMARY_FIELDS.items()
        }
        self.stats.cycles_assembled += 1
        return CycleTelemetry(pack_id=self._pack_id, cycle=cycle, **fields)

    def handle(self, frame: can.Message) -> CycleTelemetry | None:
        """Process one frame. Returns telemetry when it completes a cycle."""
        self.stats.frames_received += 1
        if frame.is_extended_id or frame.is_error_frame or frame.is_remote_frame:
            self.stats.foreign_frames += 1
            return None
        data = bytes(frame.data)
        decoded = protocol.decode(frame.arbitration_id, data) if len(data) == 8 else None
        if decoded is None:
            self.stats.foreign_frames += 1
            return None
        if not protocol.checksum_ok(frame.arbitration_id, data):
            self.stats.checksum_errors += 1
            return None

        name, values = decoded
        self.stats.frames_decoded += 1
        self.stats.by_message[name] = self.stats.by_message.get(name, 0) + 1
        self._track_counter(frame.arbitration_id, int(values["RollingCounter"]))

        if name in protocol.SUMMARY_MESSAGES:
            return self._accept_summary(name, values)
        self._update_live(name, values)
        return None

    def read_cycle(self) -> CycleTelemetry | None:
        """Block until the next complete cycle arrives.

        Returns ``None`` when the bus goes quiet: nothing at all within
        ``first_frame_timeout_s`` of opening, or nothing within
        ``idle_timeout_s`` once traffic has been seen. A BMS does not announce
        the end of a pack's life, so silence is the only end-of-stream signal
        there is.
        """
        if self._bus is None:
            self.open()
        assert self._reader is not None

        while True:
            timeout = self.idle_timeout_s if self._seen_any else self.first_frame_timeout_s
            deadline = time.monotonic() + timeout
            frame = None
            while frame is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                frame = self._reader.get_message(timeout=remaining)
            self._seen_any = True
            telemetry = self.handle(frame)
            if telemetry is not None:
                return telemetry

    def describe(self) -> dict[str, object]:
        """Configuration and link health. Works without an open bus."""
        return {
            "pack_id": self._pack_id,
            "protocol": "CAN, cobalt_bms.dbc",
            "interface": self.interface,
            "channel": self.channel,
            "bitrate": self.bitrate,
            "open": self._bus is not None,
            "link": self.stats.as_dict(),
        }
