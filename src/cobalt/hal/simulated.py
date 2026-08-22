"""A simulated pack presented as if it were a live battery management system.

This is what stands in for hardware. It runs the same :class:`~cobalt.sim.cell.Cell`
the training fleet was built from, one cycle at a time, and emits telemetry
through the same interface a serial link would.

It is the honest kind of stand-in: the model on the other side of the interface
cannot tell the difference, which is the whole purpose, but that says nothing
about whether the model would work on a real cell. It says the plumbing works.
"""

from __future__ import annotations

import numpy as np

from cobalt.hal.base import BmsSource, CycleTelemetry
from cobalt.sim.cell import Cell, CellParameters, DutyCycle


class SimulatedBms(BmsSource):
    """Streams a freshly simulated cell, one cycle per :meth:`read_cycle`."""

    def __init__(
        self,
        pack_id: str = "SIM-0001",
        parameters: CellParameters | None = None,
        duty: DutyCycle | None = None,
        seed: int = 0,
        max_cycles: int = 3000,
        stop_at_soh: float = 0.72,
    ) -> None:
        self._pack_id = pack_id
        self._parameters = parameters if parameters is not None else CellParameters()
        self._duty = duty if duty is not None else DutyCycle()
        self._seed = seed
        self._max_cycles = max_cycles
        self._stop_at_soh = stop_at_soh
        self._cell: Cell | None = None

    @property
    def pack_id(self) -> str:
        return self._pack_id

    def open(self) -> None:
        if self._cell is None:
            self._cell = Cell(self._parameters, self._duty, rng=np.random.default_rng(self._seed))

    def close(self) -> None:
        self._cell = None

    def read_cycle(self) -> CycleTelemetry | None:
        if self._cell is None:
            self.open()
        assert self._cell is not None

        if self._cell.cycle >= self._max_cycles or self._cell.state_of_health < self._stop_at_soh:
            return None

        record = self._cell.step()
        return CycleTelemetry(
            pack_id=self._pack_id,
            cycle=record.cycle,
            discharge_capacity_ah=record.discharge_capacity_ah,
            charge_capacity_ah=record.charge_capacity_ah,
            internal_resistance_ohm=record.internal_resistance_ohm,
            mean_temperature_c=record.mean_temperature_c,
            max_temperature_c=record.max_temperature_c,
            charge_c_rate=record.charge_c_rate,
            discharge_c_rate=record.discharge_c_rate,
            charge_time_min=record.charge_time_min,
            discharge_energy_wh=record.discharge_energy_wh,
            cumulative_ah_throughput=record.cumulative_ah_throughput,
        )

    def __enter__(self) -> SimulatedBms:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ReplayBms(BmsSource):
    """Replays telemetry that already exists, for tests and for reprocessing logs."""

    def __init__(self, pack_id: str, telemetry: list[CycleTelemetry]) -> None:
        self._pack_id = pack_id
        self._telemetry = list(telemetry)
        self._position = 0

    @property
    def pack_id(self) -> str:
        return self._pack_id

    def open(self) -> None:
        self._position = 0

    def close(self) -> None:
        self._position = len(self._telemetry)

    def read_cycle(self) -> CycleTelemetry | None:
        if self._position >= len(self._telemetry):
            return None
        record = self._telemetry[self._position]
        self._position += 1
        return record
