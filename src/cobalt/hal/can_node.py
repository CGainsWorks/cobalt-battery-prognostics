"""A simulated battery management system that talks CAN.

This is the far end of the link :class:`~cobalt.hal.can_bms.CanBms` listens to.
It owns a :class:`~cobalt.hal.simulated.SimulatedBms`, steps it one cycle at a
time and puts what a BMS would put on the bus onto a python-can bus: live
status frames through each charge and discharge, then the five cycle summary
frames. Every frame is encoded through the same DBC the receiver decodes with.

**Everything this node sends is simulated.** The cell is `cobalt.sim.cell.Cell`.
The module is four of those in series, which share one simulated ageing state:
each series cell's voltage is the simulated cell's terminal voltage plus a fixed
per-cell offset of a few millivolts drawn from the seed, which stands in for
cell-to-cell mismatch and is not a model of it. The cycle summaries are per
series cell, which is what the estimator was trained on.

Run it in a thread on python-can's in-process ``virtual`` interface, which is
what the tests and ``cobalt stream --source can`` do, or as its own process on
a Linux ``vcan`` interface with ``cobalt bms-node``, which is what continuous
integration does when the runner allows it.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field

import can
import numpy as np

from cobalt.hal import can_protocol as protocol
from cobalt.hal.base import CycleTelemetry
from cobalt.hal.simulated import SimulatedBms
from cobalt.sim.cell import open_circuit_voltage

#: Standard deviation of the fixed per-cell voltage offset, volts. Illustrative.
CELL_OFFSET_SIGMA_V = 0.004


@dataclass
class NodeStats:
    cycles_sent: int = 0
    frames_sent: int = 0
    by_message: dict[str, int] = field(default_factory=dict)


class SimulatedBmsNode:
    """Publishes a simulated pack's life onto a CAN bus, one cycle at a time."""

    def __init__(
        self,
        source: SimulatedBms,
        bus: can.BusABC,
        live_samples_per_phase: int = 2,
        frame_gap_s: float = 0.0,
        seed: int = 0,
    ) -> None:
        if live_samples_per_phase < 1:
            raise ValueError("live_samples_per_phase must be at least 1")
        self.source = source
        self.bus = bus
        self.live_samples_per_phase = live_samples_per_phase
        self.frame_gap_s = frame_gap_s
        self.stats = NodeStats()
        self._counters = dict.fromkeys(protocol.LIVE_MESSAGES + protocol.SUMMARY_MESSAGES, 0)
        self._offsets = np.random.default_rng(seed).normal(
            0.0, CELL_OFFSET_SIGMA_V, protocol.SERIES_CELLS
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    # -- frames -------------------------------------------------------------

    def _send(self, name: str, signals: dict[str, float]) -> None:
        arbitration_id, data = protocol.encode(name, signals, self._counters[name])
        self._counters[name] = (self._counters[name] + 1) % protocol.COUNTER_MODULUS
        self.bus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False))
        self.stats.frames_sent += 1
        self.stats.by_message[name] = self.stats.by_message.get(name, 0) + 1
        if self.frame_gap_s > 0:
            time.sleep(self.frame_gap_s)

    def _live(self, telemetry: CycleTelemetry) -> None:
        """Status frames through one charge and one discharge.

        State of charge is swept linearly between the discharge cut-off and
        full, and each cell's terminal voltage is its open-circuit voltage
        plus or minus the IR drop at the phase current.
        """
        cell = self.source.cell
        assert cell is not None
        rated = cell.rated_capacity_ah
        resistance = cell.true_resistance_ohm
        floor = float(cell.cutoff_state_of_charge())
        n = self.live_samples_per_phase
        spread = max(0.0, telemetry.max_temperature_c - telemetry.mean_temperature_c)

        phases = (
            (protocol.MODE_CHARGE, telemetry.charge_c_rate * rated, np.linspace(floor, 1.0, n)),
            (
                protocol.MODE_DISCHARGE,
                -telemetry.discharge_c_rate * rated,
                np.linspace(1.0, floor, n),
            ),
        )
        for mode, current, socs in phases:
            for soc in socs:
                ocv = float(open_circuit_voltage(soc))
                cells = ocv + current * resistance + self._offsets
                self._send(
                    protocol.PACK_STATUS,
                    {
                        "PackVoltage": round(float(np.sum(cells)), 2),
                        "PackCurrent": round(current, 2),
                        "StateOfCharge": round(float(soc) * 100.0 * 2) / 2,
                        "OperatingMode": mode,
                    },
                )
                self._send(
                    protocol.CELL_VOLTAGES,
                    {f"Cell{i + 1}Voltage": round(float(v), 3) for i, v in enumerate(cells)},
                )
                self._send(
                    protocol.TEMPERATURES,
                    {
                        "CellTemperatureMean": round(telemetry.mean_temperature_c, 2),
                        "CellTemperatureMax": round(telemetry.max_temperature_c, 2),
                        "CellTemperatureMin": round(telemetry.mean_temperature_c - spread, 2),
                    },
                )

    def _summary(self, telemetry: CycleTelemetry) -> None:
        index = telemetry.cycle % protocol.CYCLE_INDEX_MODULUS
        grouped: dict[str, dict[str, float]] = {name: {} for name in protocol.SUMMARY_MESSAGES}
        for attribute, (name, signal) in protocol.SUMMARY_FIELDS.items():
            grouped[name][signal] = float(getattr(telemetry, attribute))
        for name in protocol.SUMMARY_MESSAGES:
            self._send(name, {"CycleIndex": index, **grouped[name]})

    # -- running ------------------------------------------------------------

    def run(self) -> NodeStats:
        """Publish every cycle the source has, then return. Blocking."""
        try:
            with self.source:
                while not self._stop.is_set():
                    telemetry = self.source.read_cycle()
                    if telemetry is None:
                        break
                    self._live(telemetry)
                    self._summary(telemetry)
                    self.stats.cycles_sent += 1
        except BaseException as error:  # surfaced to the caller through .error
            self.error = error
            raise
        return self.stats

    def start(self) -> SimulatedBmsNode:
        """Run in a background thread. For the in-process virtual bus."""

        def target() -> None:
            # run() keeps the exception on self.error for the caller to inspect.
            with contextlib.suppress(BaseException):
                self.run()

        self._thread = threading.Thread(target=target, name="simulated-bms-node", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self.join(timeout_s)

    def join(self, timeout_s: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout_s)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
