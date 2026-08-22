"""The fleet service.

Four things a maintenance planner needs and one thing an auditor needs:

* ``POST /v1/packs/{pack_id}/cycles`` take telemetry off the link
* ``GET  /v1/packs/{pack_id}/health`` state of health, remaining life, and a date
* ``GET  /v1/fleet/risk`` the whole fleet, soonest first
* ``DELETE /v1/packs/{pack_id}`` forget a pack
* ``GET  /v1/model`` the model card, including the fact that it was trained
  on simulated data

The health response leads with ``book_replacement_after_cycles``, not with the
point estimate. That is the newsvendor quantile from `cobalt.business.cost`,
approximated by the conformal lower bound, and it is the number the decision
actually hangs on. The point estimate is returned too, one field further down,
because somebody will want it and it should not be the first thing they see.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from cobalt import __version__
from cobalt.config import Config, load_resolved, repository_root
from cobalt.deploy import MODEL_CARD, DeployedModel, load_deployed
from cobalt.features.online import window_features
from cobalt.hal.base import CycleTelemetry
from cobalt.service.store import TelemetryStore

#: Environment overrides, so the container and the tests can point the service at
#: a different config or artefact directory without editing anything.
CONFIG_ENV = "COBALT_CONFIG"
ARTEFACTS_ENV = "COBALT_ARTEFACTS"


class TelemetryIn(BaseModel):
    """One cycle, as it arrives over the wire."""

    cycle: int = Field(ge=0)
    discharge_capacity_ah: float = Field(gt=0)
    charge_capacity_ah: float = Field(gt=0)
    internal_resistance_ohm: float = Field(gt=0)
    mean_temperature_c: float
    max_temperature_c: float
    charge_c_rate: float = Field(gt=0)
    discharge_c_rate: float = Field(gt=0)
    charge_time_min: float = Field(gt=0)
    discharge_energy_wh: float = Field(ge=0)
    cumulative_ah_throughput: float = Field(ge=0)

    def to_telemetry(self, pack_id: str) -> CycleTelemetry:
        return CycleTelemetry(pack_id=pack_id, **self.model_dump())


class IngestResponse(BaseModel):
    pack_id: str
    accepted: int
    cycles_held: int


class HealthResponse(BaseModel):
    pack_id: str
    status: str
    cycles_held: int
    latest_cycle: int | None = None
    state_of_health: float | None = None
    book_replacement_after_cycles: float | None = None
    remaining_cycles_estimate: float | None = None
    remaining_cycles_upper: float | None = None
    interval_confidence: float | None = None
    cycles_needed: int | None = None
    note: str | None = None


class FleetRiskEntry(BaseModel):
    pack_id: str
    status: str
    state_of_health: float | None = None
    book_replacement_after_cycles: float | None = None


class FleetRiskResponse(BaseModel):
    packs: list[FleetRiskEntry]
    ranked_by: str = "book_replacement_after_cycles, soonest first"


class ServiceState:
    """Everything the request handlers need, built once at startup."""

    def __init__(self, config: Config, artefacts_dir: Path) -> None:
        self.config = config
        self.artefacts_dir = Path(artefacts_dir)
        self.store = TelemetryStore()
        self._models: tuple[DeployedModel, DeployedModel] | None = None

    @property
    def models(self) -> tuple[DeployedModel, DeployedModel]:
        if self._models is None:
            self._models = load_deployed(self.artefacts_dir)
        return self._models

    @property
    def online_model(self) -> DeployedModel:
        return self.models[1]

    def model_card(self) -> dict[str, Any]:
        path = self.artefacts_dir / MODEL_CARD
        if not path.exists():
            raise FileNotFoundError(f"no model card at {path}")
        import json

        return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _default_state() -> ServiceState:
    config_path = os.environ.get(CONFIG_ENV, str(repository_root() / "config" / "default.yaml"))
    config = load_resolved(config_path)
    artefacts = Path(os.environ.get(ARTEFACTS_ENV, config.paths.artifacts_dir))
    return ServiceState(config, artefacts)


def get_state() -> ServiceState:
    """Dependency seam. Tests override this to inject their own state."""
    return _default_state()


app = FastAPI(
    title="Cobalt battery prognostics",
    version=__version__,
    summary="Remaining useful life and replacement scheduling for a battery fleet",
    description=(
        "Every model behind this service was trained on **simulated** cells. "
        "See `GET /v1/model` and docs/LIMITATIONS.md before believing anything it says."
    ),
)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    """Liveness. Deliberately does not touch the models, so it stays up when they are down."""
    return {"status": "ok", "version": __version__}


@app.get("/v1/model", tags=["ops"])
def model_card(state: ServiceState = Depends(get_state)) -> dict[str, Any]:
    try:
        return state.model_card()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{error}. Run `make demo` or `cobalt evaluate` to produce it.",
        ) from error


@app.post(
    "/v1/packs/{pack_id}/cycles",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["telemetry"],
)
def ingest(
    pack_id: str,
    cycles: list[TelemetryIn],
    state: ServiceState = Depends(get_state),
) -> IngestResponse:
    if not cycles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="no cycles in the request"
        )
    accepted = state.store.ingest([record.to_telemetry(pack_id) for record in cycles])
    summary = state.store.summary(pack_id)
    return IngestResponse(
        pack_id=pack_id,
        accepted=accepted,
        cycles_held=summary.cycles_held if summary else 0,
    )


def _predict_health(state: ServiceState, pack_id: str) -> HealthResponse:
    summary = state.store.summary(pack_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no telemetry for pack {pack_id}"
        )

    window_cycles = state.config.online.window_cycles
    if summary.cycles_held < window_cycles:
        return HealthResponse(
            pack_id=pack_id,
            status="insufficient_data",
            cycles_held=summary.cycles_held,
            latest_cycle=summary.last_cycle,
            state_of_health=round(summary.latest_soh, 4),
            cycles_needed=window_cycles - summary.cycles_held,
            note=(
                f"the online model needs a {window_cycles}-cycle trailing window; "
                f"state of health above is the latest measurement, not a prediction"
            ),
        )

    frame = state.store.frame(pack_id).tail(window_cycles)
    features = window_features(frame, state.store.rated_capacity_ah)

    import pandas as pd

    model = state.online_model
    prediction = model.predict(pd.DataFrame([features]))

    # The lower bound is the deliverable. Booking on the point estimate is the
    # mistake this repository is partly about, so it is not the leading field.
    booking = float(np.round(prediction["lower"][0], 1))
    return HealthResponse(
        pack_id=pack_id,
        status="ok",
        cycles_held=summary.cycles_held,
        latest_cycle=summary.last_cycle,
        state_of_health=round(summary.latest_soh, 4),
        book_replacement_after_cycles=booking,
        remaining_cycles_estimate=float(np.round(prediction["point"][0], 1)),
        remaining_cycles_upper=float(np.round(prediction["upper"][0], 1)),
        interval_confidence=round(1.0 - model.alpha, 3),
    )


@app.get("/v1/packs/{pack_id}/health", response_model=HealthResponse, tags=["prognostics"])
def pack_health(pack_id: str, state: ServiceState = Depends(get_state)) -> HealthResponse:
    try:
        return _predict_health(state, pack_id)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


@app.get("/v1/fleet/risk", response_model=FleetRiskResponse, tags=["prognostics"])
def fleet_risk(state: ServiceState = Depends(get_state)) -> FleetRiskResponse:
    """Every pack, soonest booking first. Packs without enough history sort last."""
    entries: list[FleetRiskEntry] = []
    for pack_id in state.store.pack_ids():
        try:
            health = _predict_health(state, pack_id)
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
            ) from error
        entries.append(
            FleetRiskEntry(
                pack_id=pack_id,
                status=health.status,
                state_of_health=health.state_of_health,
                book_replacement_after_cycles=health.book_replacement_after_cycles,
            )
        )

    entries.sort(
        key=lambda entry: (
            entry.book_replacement_after_cycles is None,
            entry.book_replacement_after_cycles or 0.0,
        )
    )
    return FleetRiskResponse(packs=entries)


@app.delete("/v1/packs/{pack_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["telemetry"])
def forget_pack(pack_id: str, state: ServiceState = Depends(get_state)) -> None:
    if not state.store.drop(pack_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no telemetry for pack {pack_id}"
        )
