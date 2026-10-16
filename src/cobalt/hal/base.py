"""The seam between this repository and a real battery.

Everything upstream of here consumes :class:`CycleTelemetry` and nothing
upstream knows or cares where it came from. There are three implementations:

* :class:`~cobalt.hal.simulated.SimulatedBms` replays a simulated cell and is
  what the evaluation, the figures and the README's numbers run on;
* :class:`~cobalt.hal.simulated.ReplayBms` replays telemetry that already
  exists;
* :class:`~cobalt.hal.can_bms.CanBms` decodes it off a CAN bus with the
  repository's DBC. It has only ever been run against a simulated BMS node, on
  python-can's virtual bus and on a Linux vcan interface, never against a
  physical adapter. `docs/HARDWARE.md` says what that would take.

Keeping the boundary this narrow is the point: the feature extraction, the
models, the service and the tests are all written against telemetry rather than
against a simulator, so the transport underneath can change without them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CycleTelemetry:
    """One cycle's worth of battery management system output.

    Every field here is something a production BMS already puts on a CAN bus or
    a serial link. Nothing on this record requires a laboratory instrument and
    that constraint is what makes the online model deployable rather than an
    exercise.
    """

    pack_id: str
    cycle: int
    discharge_capacity_ah: float
    charge_capacity_ah: float
    internal_resistance_ohm: float
    mean_temperature_c: float
    max_temperature_c: float
    charge_c_rate: float
    discharge_c_rate: float
    charge_time_min: float
    discharge_energy_wh: float
    cumulative_ah_throughput: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@runtime_checkable
class BmsSource(Protocol):
    """Anything that can produce cycle telemetry for a pack."""

    @property
    def pack_id(self) -> str:
        """Identifier the telemetry is tagged with."""

    def open(self) -> None:
        """Acquire whatever the source needs. Must be idempotent."""

    def close(self) -> None:
        """Release it. Must be safe to call without a matching open."""

    def read_cycle(self) -> CycleTelemetry | None:
        """Next cycle, or ``None`` when the source is exhausted."""


class BmsError(RuntimeError):
    """Raised when a source cannot deliver telemetry."""
