"""The CAN protocol between a battery management system and the estimator.

The frames are defined in ``cobalt_bms.dbc`` next to this file and nowhere else.
Both ends load that one file through cantools, so the simulated node and the
receiver cannot disagree about a scale factor or a bit position: if the DBC is
wrong, it is wrong for both and the round-trip tests catch it against the
unencoded values.

Two kinds of traffic share the bus:

* **Live frames** (``BMS_PackStatus``, ``BMS_CellVoltages``,
  ``BMS_Temperatures``), sent several times per charge and discharge phase. They
  carry what a BMS always has: pack voltage and current, state of charge, each
  series cell's voltage and the temperature spread.
* **Cycle summary frames** (five of them, joined by ``CycleIndex``), sent once
  at the end of each full cycle. Together they carry exactly the fields of
  :class:`~cobalt.hal.base.CycleTelemetry` and they are what the remaining-life
  model consumes.

Every frame ends with a four-bit rolling counter and an eight-bit checksum in
byte 7. The checksum is a ones-complement byte sum over data bytes 0 to 6 and
both bytes of the arbitration ID. It is a deliberately simple scheme that
catches corruption and mis-addressed frames; it is not AUTOSAR E2E and does not
claim to be.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import cantools
from cantools.database.can import Database, Message

DBC_RESOURCE = "cobalt_bms.dbc"

PACK_STATUS = "BMS_PackStatus"
CELL_VOLTAGES = "BMS_CellVoltages"
TEMPERATURES = "BMS_Temperatures"
LIVE_MESSAGES = (PACK_STATUS, CELL_VOLTAGES, TEMPERATURES)

CYCLE_CAPACITY = "BMS_CycleCapacity"
CYCLE_RESISTANCE = "BMS_CycleResistance"
CYCLE_THERMAL = "BMS_CycleThermal"
CYCLE_RATES = "BMS_CycleRates"
CYCLE_THROUGHPUT = "BMS_CycleThroughput"
SUMMARY_MESSAGES = (CYCLE_CAPACITY, CYCLE_RESISTANCE, CYCLE_THERMAL, CYCLE_RATES, CYCLE_THROUGHPUT)

#: Series cells in the module the DBC describes. Fixed by ``BMS_CellVoltages``.
SERIES_CELLS = 4

#: Operating modes, as enumerated by the ``VAL_`` table in the DBC.
MODE_IDLE, MODE_CHARGE, MODE_DISCHARGE, MODE_FAULT = 0, 1, 2, 3

#: Which DBC signal fills which :class:`CycleTelemetry` field.
SUMMARY_FIELDS = {
    "discharge_capacity_ah": (CYCLE_CAPACITY, "DischargeCapacity"),
    "charge_capacity_ah": (CYCLE_CAPACITY, "ChargeCapacity"),
    "internal_resistance_ohm": (CYCLE_RESISTANCE, "InternalResistance"),
    "charge_time_min": (CYCLE_RESISTANCE, "ChargeTime"),
    "mean_temperature_c": (CYCLE_THERMAL, "MeanTemperature"),
    "max_temperature_c": (CYCLE_THERMAL, "MaxTemperature"),
    "charge_c_rate": (CYCLE_RATES, "ChargeCRate"),
    "discharge_c_rate": (CYCLE_RATES, "DischargeCRate"),
    "discharge_energy_wh": (CYCLE_RATES, "DischargeEnergy"),
    "cumulative_ah_throughput": (CYCLE_THROUGHPUT, "CumulativeThroughput"),
}

COUNTER_MODULUS = 16
CYCLE_INDEX_MODULUS = 1 << 16


@lru_cache(maxsize=1)
def database() -> Database:
    """The parsed DBC. Strict, so an overlapping signal is a load error."""
    text = resources.files("cobalt.hal").joinpath(DBC_RESOURCE).read_text(encoding="utf-8")
    return cantools.database.load_string(text, database_format="dbc", strict=True)


def message(name: str) -> Message:
    return database().get_message_by_name(name)


def checksum(arbitration_id: int, data: bytes) -> int:
    """Ones-complement byte sum over data bytes 0 to 6 and the two ID bytes."""
    total = sum(data[:7]) + (arbitration_id & 0xFF) + ((arbitration_id >> 8) & 0xFF)
    return (~total) & 0xFF


def encode(name: str, signals: dict[str, float], counter: int) -> tuple[int, bytes]:
    """Encode one frame, stamp its counter and checksum and return ``(id, data)``.

    Values outside a signal's range are an error rather than a silent clamp: a
    BMS that saturates a field without saying so is how a 7 Ah cell turns into a
    6.55 Ah one in the logs.
    """
    definition = message(name)
    payload = dict(signals)
    payload["RollingCounter"] = counter % COUNTER_MODULUS
    payload["Checksum"] = 0
    data = bytearray(definition.encode(payload, scaling=True, strict=True))
    data[7] = checksum(definition.frame_id, bytes(data))
    return definition.frame_id, bytes(data)


def decode(arbitration_id: int, data: bytes) -> tuple[str, dict[str, float]] | None:
    """Decode a frame this protocol defines, or ``None`` for anything else on the bus."""
    try:
        definition = database().get_message_by_frame_id(arbitration_id)
    except KeyError:
        return None
    values = definition.decode(bytes(data), decode_choices=False, scaling=True)
    return definition.name, {key: float(value) for key, value in values.items()}


def checksum_ok(arbitration_id: int, data: bytes) -> bool:
    return len(data) == 8 and checksum(arbitration_id, data) == data[7]


def frame_ids() -> tuple[int, ...]:
    return tuple(message(name).frame_id for name in LIVE_MESSAGES + SUMMARY_MESSAGES)


def resolution(field: str) -> float:
    """Quantisation step of a telemetry field on the wire, for tolerance checks."""
    name, signal = SUMMARY_FIELDS[field]
    return float(message(name).get_signal_by_name(signal).scale)
