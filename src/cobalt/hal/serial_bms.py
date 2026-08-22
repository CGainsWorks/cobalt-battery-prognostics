"""The hardware-backed source. Not implemented and this file explains why.

**There was no hardware.** No cycler, no thermal chamber, no cells. Every number
in this repository comes from the simulator in `cobalt.sim` and this module is
the place where that limitation is concrete rather than a caveat in a README.

Rather than delete it, the interface is defined and every method raises
:class:`~cobalt.hal.base.NoHardwareError` with a message saying what it would
need. That is a more useful artefact than an empty space: it fixes the shape of
the work, it keeps the rest of the codebase honestly written against an
interface rather than against a simulator and it is checked by a test that
asserts these methods raise rather than silently returning something plausible.

What would actually be needed, in order:

1. **A cycler.** Something like a Neware CT-4008 or a Maccor 4300, eight to
   sixty-four channels. The channel count is what sets how many cells can be
   aged in parallel and cells have to be aged in parallel because a single
   channel takes months per cell.
2. **A thermal chamber** per temperature point. The Arrhenius term is the single
   largest driver in the model and it cannot be characterised at one temperature.
3. **Cells.** Thirty to fifty per lot, at least three lots from different
   coating runs. Fewer lots than that and the batch-independent protocol this
   entire repository argues for cannot be run at all, which would be an
   unusually pointed way to fail.
4. **A BMS link.** CAN via python-can and a DBC, or Modbus RTU over RS-485.
   :meth:`SerialBms.read_cycle` is where the frames become a
   :class:`~cobalt.hal.base.CycleTelemetry`.
5. **Reference performance tests.** A slow full discharge every twenty-five
   cycles, which is what makes ``discharge_capacity_ah`` a measurement rather
   than an estimate.

Nothing else in the codebase changes. The features, the split protocols, the
models, the conformal calibration, the cost models and the service all consume
telemetry and never touch the simulator.

See `docs/HARDWARE.md` for the full rig specification and the acceptance tests
that would tell you the link is working.
"""

from __future__ import annotations

from cobalt.hal.base import BmsSource, CycleTelemetry, NoHardwareError

_MESSAGE = (
    "SerialBms is not implemented: this repository was built without access to "
    "cells, a cycler or a battery management system. Every result in it comes "
    "from cobalt.sim. See docs/HARDWARE.md for the rig this needs and use "
    "cobalt.hal.simulated.SimulatedBms in the meantime."
)


class SerialBms(BmsSource):
    """Reads cycle telemetry from a battery management system over a serial link.

    Constructing one is deliberately allowed, so that configuration wiring and
    dependency injection can be tested without hardware. Every method that would
    need to touch a real device raises.
    """

    def __init__(
        self,
        port: str,
        pack_id: str,
        baudrate: int = 115_200,
        timeout_s: float = 2.0,
        protocol: str = "modbus-rtu",
    ) -> None:
        self.port = port
        self._pack_id = pack_id
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.protocol = protocol

    @property
    def pack_id(self) -> str:
        return self._pack_id

    def open(self) -> None:
        raise NoHardwareError(_MESSAGE)

    def close(self) -> None:
        # Closing something that was never opened is a no-op rather than an
        # error, so a `finally: source.close()` in a caller stays correct.
        return None

    def read_cycle(self) -> CycleTelemetry | None:
        raise NoHardwareError(_MESSAGE)

    def describe(self) -> dict[str, object]:
        """Configuration summary. Works without hardware, on purpose.

        Being able to inspect a configured-but-unopenable source is what lets the
        service report which packs it is meant to be talking to when the link is
        down, instead of failing to start.
        """
        return {
            "pack_id": self._pack_id,
            "port": self.port,
            "baudrate": self.baudrate,
            "timeout_s": self.timeout_s,
            "protocol": self.protocol,
            "implemented": False,
            "reason": "no hardware available, see docs/HARDWARE.md",
        }
