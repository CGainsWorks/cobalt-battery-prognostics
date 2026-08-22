"""The fleet service, driven through the app rather than through a socket.

No port is bound and no server process is started, so nothing here can leak a
listener into the rest of the suite. The state is injected through the same
dependency the production path uses, which means these tests exercise the real
wiring rather than a parallel test harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cobalt.config import Config, OnlineConfig, PathsConfig
from cobalt.deploy import MODEL_CARD, train_and_save
from cobalt.evaluate.protocol import build_feature_tables, run_task
from cobalt.evaluate.report import build_summary
from cobalt.features.online import window_features
from cobalt.hal.simulated import SimulatedBms
from cobalt.service.api import ServiceState, app, get_state


@pytest.fixture(scope="module")
def trained(tmp_path_factory, request) -> tuple[Config, Path]:
    """Train the deployed models once on the tiny fleet, for the whole module."""
    tiny_dataset = request.getfixturevalue("tiny_dataset")
    tiny_config = request.getfixturevalue("tiny_config")

    tables = build_feature_tables(tiny_dataset, tiny_config)
    results = {
        task: run_task(task, tables[task]["frame"], tiny_config)
        for task in ("early_life", "online_soh")
    }
    summary = build_summary(
        tiny_config,
        tiny_dataset,
        results,
        {task: tables[task]["excluded"] for task in tables},
    )

    artefacts = tmp_path_factory.mktemp("artefacts")
    train_and_save(
        tiny_config,
        tables["early_life"]["frame"],
        tables["online_soh"]["frame"],
        summary,
        artefacts,
    )
    return tiny_config, artefacts


@pytest.fixture
def client(trained):
    config, artefacts = trained
    state = ServiceState(config, artefacts)
    app.dependency_overrides[get_state] = lambda: state
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def telemetry(trained) -> list[dict]:
    """A pack's worth of cycles, as JSON payloads."""
    config, _ = trained
    needed = config.online.window_cycles + 20
    with SimulatedBms(pack_id="P1", seed=11, max_cycles=needed) as source:
        rows = []
        while (record := source.read_cycle()) is not None:
            payload = record.as_dict()
            payload.pop("pack_id")
            rows.append(payload)
    return rows


# -- ops ----------------------------------------------------------------


def test_liveness_does_not_depend_on_the_models(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_model_card_leads_with_the_data_source(client, trained):
    response = client.get("/v1/model")
    assert response.status_code == 200
    card = response.json()
    assert "SIMULATED" in card["data_source"]
    assert card["models"]["online_soh"]["name"]
    # The card must quote the hardest protocol it has, not the flattering one.
    assert card["models"]["online_soh"]["reported"]["protocol"] == "batch_independent"


def test_a_missing_model_card_is_a_503_that_says_how_to_fix_it(trained, tmp_path):
    config, _ = trained
    app.dependency_overrides[get_state] = lambda: ServiceState(config, tmp_path)
    with TestClient(app) as test_client:
        response = test_client.get("/v1/model")
    app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "cobalt evaluate" in response.json()["detail"]


# -- telemetry ----------------------------------------------------------


def test_ingest_accepts_cycles_and_reports_what_it_holds(client, telemetry):
    response = client.post("/v1/packs/P1/cycles", json=telemetry[:10])
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] == 10
    assert body["cycles_held"] == 10


def test_an_empty_ingest_is_rejected(client):
    assert client.post("/v1/packs/P1/cycles", json=[]).status_code == 400


def test_malformed_telemetry_is_rejected(client, telemetry):
    bad = dict(telemetry[0])
    bad["discharge_capacity_ah"] = -1.0
    assert client.post("/v1/packs/P1/cycles", json=[bad]).status_code == 422


def test_duplicate_cycles_are_deduplicated(client, telemetry):
    client.post("/v1/packs/DUP/cycles", json=telemetry[:5])
    response = client.post("/v1/packs/DUP/cycles", json=telemetry[:5])
    assert response.json()["cycles_held"] == 5


def test_out_of_order_cycles_are_sorted(client, telemetry):
    shuffled = list(reversed(telemetry[:8]))
    client.post("/v1/packs/ORD/cycles", json=shuffled)
    health = client.get("/v1/packs/ORD/health").json()
    assert health["latest_cycle"] == max(row["cycle"] for row in shuffled)


def test_forgetting_a_pack(client, telemetry):
    client.post("/v1/packs/GONE/cycles", json=telemetry[:5])
    assert client.delete("/v1/packs/GONE").status_code == 204
    assert client.delete("/v1/packs/GONE").status_code == 404


# -- prognostics --------------------------------------------------------


def test_health_for_an_unknown_pack_is_a_404(client):
    assert client.get("/v1/packs/NOBODY/health").status_code == 404


def test_a_short_history_says_how_many_more_cycles_it_needs(client, telemetry, trained):
    config, _ = trained
    short = telemetry[: config.online.window_cycles - 5]
    client.post("/v1/packs/SHORT/cycles", json=short)
    body = client.get("/v1/packs/SHORT/health").json()

    assert body["status"] == "insufficient_data"
    assert body["cycles_needed"] == 5
    assert body["book_replacement_after_cycles"] is None
    # State of health is still reported, because it is a measurement not a prediction.
    assert body["state_of_health"] is not None


def test_a_full_history_produces_a_booking_inside_its_own_interval(client, telemetry):
    client.post("/v1/packs/P1/cycles", json=telemetry)
    body = client.get("/v1/packs/P1/health").json()

    assert body["status"] == "ok"
    assert body["book_replacement_after_cycles"] > 0
    assert (
        body["book_replacement_after_cycles"]
        <= body["remaining_cycles_estimate"]
        <= body["remaining_cycles_upper"]
    )
    assert body["interval_confidence"] == pytest.approx(0.90)


def test_the_service_and_the_library_agree_on_the_same_window(client, telemetry, trained):
    """No training-serving skew: one feature function, one answer."""
    config, artefacts = trained
    client.post("/v1/packs/SKEW/cycles", json=telemetry)
    served = client.get("/v1/packs/SKEW/health").json()

    from cobalt.deploy import load_deployed

    _, online_model = load_deployed(artefacts)
    frame = pd.DataFrame(telemetry).tail(config.online.window_cycles)
    features = window_features(frame, rated_capacity_ah=2.20)
    direct = online_model.predict(pd.DataFrame([features]))

    # The API rounds to a tenth of a cycle on the way out, so compare at that
    # resolution rather than at floating point resolution.
    assert served["remaining_cycles_estimate"] == pytest.approx(
        round(float(direct["point"][0]), 1), abs=1e-9
    )


def test_fleet_risk_ranks_the_soonest_booking_first(client, telemetry, trained):
    config, _ = trained
    for pack_id, seed in (("A", 21), ("B", 22), ("C", 23)):
        with SimulatedBms(
            pack_id=pack_id, seed=seed, max_cycles=config.online.window_cycles + 30
        ) as source:
            rows = []
            while (record := source.read_cycle()) is not None:
                payload = record.as_dict()
                payload.pop("pack_id")
                rows.append(payload)
        client.post(f"/v1/packs/{pack_id}/cycles", json=rows)

    packs = client.get("/v1/fleet/risk").json()["packs"]
    bookings = [
        p["book_replacement_after_cycles"]
        for p in packs
        if p["book_replacement_after_cycles"] is not None
    ]
    assert bookings == sorted(bookings)


def test_packs_without_enough_history_sort_last(client, telemetry, trained):
    config, _ = trained
    client.post("/v1/packs/FULL/cycles", json=telemetry)
    client.post("/v1/packs/STUB/cycles", json=telemetry[:3])

    packs = client.get("/v1/fleet/risk").json()["packs"]
    identifiers = [p["pack_id"] for p in packs]
    statuses = {p["pack_id"]: p["status"] for p in packs}
    assert statuses["STUB"] == "insufficient_data"
    assert identifiers.index("FULL") < identifiers.index("STUB")


def test_the_model_card_on_disk_matches_what_the_service_serves(client, trained):
    _, artefacts = trained
    on_disk = json.loads((artefacts / MODEL_CARD).read_text(encoding="utf-8"))
    assert client.get("/v1/model").json() == on_disk


def test_service_state_can_be_pointed_at_a_different_artefact_directory(trained, tmp_path):
    config, artefacts = trained
    state = ServiceState(
        Config(
            name=config.name,
            seed=config.seed,
            paths=PathsConfig(artifacts_dir=str(tmp_path)),
            fleet=config.fleet,
            evaluation=config.evaluation,
            online=OnlineConfig(window_cycles=config.online.window_cycles),
        ),
        artefacts,
    )
    assert state.online_model is not None
