"""Where ingested telemetry lives while the service is running.

In memory, in a dictionary, behind a lock. That is the correct amount of
engineering for this repository and it is a deliberate choice rather than a
missing feature: swapping it for a time-series database is a real decision with
real operational consequences, and making it here on no evidence would be
guessing.

What it does do is bound itself. A pack that streams for a year is a hundred
thousand records, and the online model only ever looks at a trailing window, so
the store keeps a ring of the most recent cycles and drops the rest. An
unbounded list in a long-running service is a slow leak that looks fine in
testing and falls over in the third week.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import pandas as pd

from cobalt.hal.base import CycleTelemetry

#: Retained cycles per pack. Comfortably more than any window the model uses, so
#: there is history to plot and to recompute from after a model change.
DEFAULT_RETENTION_CYCLES = 400


@dataclass(frozen=True)
class PackSummary:
    pack_id: str
    cycles_held: int
    first_cycle: int
    last_cycle: int
    latest_soh: float


class TelemetryStore:
    """Thread-safe, bounded, per-pack cycle history."""

    def __init__(
        self,
        retention_cycles: int = DEFAULT_RETENTION_CYCLES,
        rated_capacity_ah: float = 2.20,
    ) -> None:
        if retention_cycles < 2:
            raise ValueError("retention_cycles must be at least 2")
        self.retention_cycles = retention_cycles
        self.rated_capacity_ah = rated_capacity_ah
        self._packs: dict[str, deque[CycleTelemetry]] = {}
        self._lock = threading.Lock()

    def ingest(self, telemetry: list[CycleTelemetry]) -> int:
        """Append records, keeping each pack's history in cycle order.

        Out-of-order and duplicate cycles are both real on a flaky link, so the
        ring is re-sorted and de-duplicated on the way in rather than trusting
        the sender. Later records win, because a retransmission is usually a
        correction.
        """
        if not telemetry:
            return 0

        with self._lock:
            for record in telemetry:
                ring = self._packs.setdefault(record.pack_id, deque(maxlen=self.retention_cycles))
                ring.append(record)

            touched = {record.pack_id for record in telemetry}
            for pack_id in touched:
                ring = self._packs[pack_id]
                latest: dict[int, CycleTelemetry] = {}
                for record in ring:
                    latest[record.cycle] = record
                ordered = [latest[cycle] for cycle in sorted(latest)]
                self._packs[pack_id] = deque(
                    ordered[-self.retention_cycles :], maxlen=self.retention_cycles
                )

        return len(telemetry)

    def pack_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._packs)

    def history(self, pack_id: str) -> list[CycleTelemetry]:
        with self._lock:
            ring = self._packs.get(pack_id)
            return list(ring) if ring else []

    def frame(self, pack_id: str) -> pd.DataFrame:
        """History as the frame the feature builders expect.

        The column names match the simulated dataset exactly, which is what lets
        one feature function serve both training and serving. A separate
        serving-time feature path is the classic way to ship a training/serving
        skew, so there is not one.
        """
        history = self.history(pack_id)
        if not history:
            return pd.DataFrame()
        frame = pd.DataFrame([record.as_dict() for record in history])
        return frame.rename(columns={"pack_id": "cell_id"}).sort_values("cycle")

    def summary(self, pack_id: str) -> PackSummary | None:
        history = self.history(pack_id)
        if not history:
            return None
        return PackSummary(
            pack_id=pack_id,
            cycles_held=len(history),
            first_cycle=history[0].cycle,
            last_cycle=history[-1].cycle,
            latest_soh=history[-1].discharge_capacity_ah / self.rated_capacity_ah,
        )

    def drop(self, pack_id: str) -> bool:
        with self._lock:
            return self._packs.pop(pack_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._packs.clear()
