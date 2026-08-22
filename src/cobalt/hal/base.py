"""The seam between this repository and a real battery.

Everything upstream of here consumes :class:`CycleTelemetry` and nothing
upstream knows or cares where it came from. There are two implementations. One
replays a simulated cell and is what every test, figure and number in this
repository runs on. The other talks to a real battery management system over a
serial link and is **not implemented**, because there was no hardware.

That is the honest state of it. The interface exists, the simulated side is
complete and exercised and the hardware side is a documented gap with a
specific shape rather than a vague intention. `docs/HARDWARE.md` lists the rig
it needs and exactly which methods have to be filled in.

Keeping the boundary this narrow is the point: when a cycler does arrive, the
work is one class and the feature extraction, the models, the service and the
tests are all already written against telemetry rather than against a simulator.
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


class NoHardwareError(BmsError, NotImplementedError):
    """Raised by every hardware-backed source in this repository.

    Deliberately a subclass of both, so a caller that is checking for a missing
    feature and a caller that is checking for a dead link both catch it and so
    it never gets mistaken for a bug in the pipeline.
    """
