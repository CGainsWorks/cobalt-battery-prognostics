"""The real-data loader and evaluation, on synthetic files shaped like the real ones.

The Severson files are 8.3 GB and are never in the repository, so these tests
write small HDF5 files with the same layout the publisher's MATLAB v7.3 files
have (a ``batch`` struct of object references) and run the loader on those.
The values are synthetic and carry no meaning; what is tested is the plumbing:
the reference chasing, the continuation joins, the removals, the labelling rule,
the glitch filter and that the extract is a dataset the unchanged evaluation
protocol accepts.

The real files are exercised by the `real-data` CI job, which re-runs the
evaluation and compares it with the committed results/severson/summary.json.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from cobalt.config import EvaluationConfig, load_resolved, repository_root
from cobalt.data.schema import FleetDataset
from cobalt.realdata import evaluate as real_evaluate
from cobalt.realdata import severson

GRID = np.linspace(3.5, 2.0, 40)


class _Writer:
    """Builds one MATLAB-v7.3-shaped HDF5 batch file."""

    def __init__(self, path: Path) -> None:
        self.file = h5py.File(path, "w")
        self.file.create_group("#refs#")
        self.count = 0

    def ref(self, data) -> h5py.Reference:
        name = f"#refs#/d{self.count}"
        self.count += 1
        self.file[name] = data
        return self.file[name].ref

    def group_ref(self, members: dict) -> h5py.Reference:
        name = f"#refs#/g{self.count}"
        self.count += 1
        group = self.file.create_group(name)
        for key, value in members.items():
            group[key] = value
        return group.ref

    def write(self, cells: list[dict]) -> None:
        summaries, cycles, policies, grids = [], [], [], []
        for cell in cells:
            qd = cell["qd"]
            n = len(qd)
            summaries.append(
                self.group_ref(
                    {
                        "QDischarge": qd[None, :],
                        "QCharge": (qd * 1.001)[None, :],
                        "IR": np.where(qd > 0, 0.0165, 0.0)[None, :],
                        "Tavg": np.where(qd > 0, 32.0, 0.0)[None, :],
                        "Tmin": np.where(qd > 0, 30.0, 0.0)[None, :],
                        "Tmax": np.where(qd > 0, 35.0, 0.0)[None, :],
                        "chargetime": np.where(qd > 0, 10.5, 0.0)[None, :],
                        "cycle": np.arange(1, n + 1, dtype=float)[None, :],
                    }
                )
            )
            curve_refs = [self.ref(np.linspace(0.0, max(q, 0.0), len(GRID))[None, :]) for q in qd]
            cycles.append(
                self.group_ref({"Qdlin": np.array(curve_refs, dtype=h5py.ref_dtype).reshape(-1, 1)})
            )
            policies.append(
                self.ref(
                    np.frombuffer(cell["policy"].encode("utf-16-le"), dtype=np.uint16)[:, None]
                )
            )
            grids.append(self.ref(GRID[None, :]))
        batch = self.file.create_group("batch")
        for key, refs in (
            ("summary", summaries),
            ("cycles", cycles),
            ("policy_readable", policies),
            ("Vdlin", grids),
        ):
            batch.create_dataset(key, data=np.array(refs, dtype=h5py.ref_dtype).reshape(-1, 1))
        self.file.close()


def _fade(start: float, end: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """An empty first record, then a near-linear fade with a little noise."""
    values = np.linspace(start, end, n - 1) + rng.normal(0, 0.0005, n - 1)
    return np.concatenate([[0.0], values])


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(3)
    directory = tmp_path_factory.mktemp("raw")
    b1, b2, b3 = severson.BATCHES

    batch1 = []
    for i in range(6):
        # Cells 0 to 4 continue into batch 2, so no end of life inside batch 1.
        # Cell 5 stops well above end of life and is censored.
        qd = _fade(1.07, 1.00, 110, rng) if i < 5 else _fade(1.07, 0.97, 140, rng)
        batch1.append({"qd": qd, "policy": "3.6C(80%)-3.6C"})
    _Writer(directory / b1.filename).write(batch1)

    batch2 = []
    for i in range(17):
        if i in (7, 8, 9, 15, 16):
            qd = _fade(1.00, 0.875, 120, rng)
        else:
            life = 130 + 5 * i
            qd = _fade(1.08, 0.875, life, rng)
        batch2.append({"qd": qd, "policy": f"{4 + i % 3}C(40%)-3.6C"})
    # One glitch: a 2.9 Ah record, which the filter must drop.
    batch2[0]["qd"][50] = 2.9
    _Writer(directory / b2.filename).write(batch2)

    batch3 = []
    for i in range(44):
        life = 125 + 3 * i
        batch3.append({"qd": _fade(1.08, 0.875, life, rng), "policy": "5.3C(54%)-4C-newstructure"})
    _Writer(directory / b3.filename).write(batch3)
    return directory


@pytest.fixture(scope="module")
def extract_dir(raw_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("extract")
    severson.extract(raw_dir, out, verify_size=False, log=lambda _: None)
    return out


def test_policies_give_their_first_charge_rate():
    assert severson.first_c_rate("5.4C(40%)-3.6C") == 5.4
    assert severson.first_c_rate("8C(15%)-3.6C") == 8.0
    assert severson.first_c_rate("5.3C(54%)-4C-newstructure") == 5.3
    with pytest.raises(ValueError):
        severson.first_c_rate("rest")


def test_the_label_follows_the_authors_rule():
    # Crosses 0.88 at the fourth record (cycle 4).
    assert severson.label(np.array([0.0, 1.0, 0.9, 0.87, 0.86])) == (4, False)
    # Stopped just above 0.88: one past the last record, as the authors label it.
    assert severson.label(np.array([0.0, 1.0, 0.9, 0.882])) == (5, False)
    # Stopped well above: censored.
    assert severson.label(np.array([0.0, 1.0, 0.97])) == (3, True)
    # An empty record mid-series is not a capacity of zero.
    assert severson.label(np.array([0.0, 1.0, 0.0, 0.95, 0.87])) == (5, False)


def test_the_glitch_filter_catches_impossible_values_only():
    frame = pd.DataFrame(
        {
            "discharge_capacity_ah": [1.05, 1.04, 2.9, 1.03, 0.95],
            "charge_capacity_ah": [1.05, 1.04, 1.04, 1.03, 0.95],
            "internal_resistance_ohm": [0.016, 0.0, 0.016, 0.016, 0.017],
            "charge_time_min": [10.0, 10.0, 10.0, 500.0, 21.0],
            "discharge_energy_wh": [3.1, 3.1, 3.1, 3.0, -4.0],
        }
    )
    masks = severson.implausible_records(frame)
    dropped = np.logical_or.reduce(list(masks.values()))
    # Rows 1 to 4 are glitches; the 21-minute charge (a real end-of-life
    # symptom, about twice the median) must survive the charge-time rule.
    assert list(dropped) == [False, True, True, True, True]
    assert not masks["charge_time_implausible"].iloc[4]


def test_the_extract_applies_the_authors_corrections(extract_dir: Path):
    dataset = FleetDataset.load(extract_dir)
    cells = dataset.cells.set_index("cell_id")
    manifest = dataset.manifest

    # 6 + (17 - 5 continuations) + (44 - 6 noisy) cells.
    assert len(cells) == 56
    assert not any(c in cells.index for c in ("b2c7", "b2c15", "b3c37", "b3c2"))
    assert manifest["censored_cells"] == ["b1c5"]
    assert manifest["implausible_cycle_records_dropped"]["capacity_above_1_3_ah"] == 1

    # A joined cell keeps its batch 1 identity and carries both runs: 109 real
    # records from batch 1 and 119 from batch 2 (each run's first is empty).
    joined = dataset.cycles_for("b1c0")
    assert cells.loc["b1c0", "batch_id"] == "2017-05-12"
    assert len(joined) == 109 + 119
    assert int(cells.loc["b1c0", "cycle_life"]) > 110
    assert joined["cumulative_ah_throughput"].is_monotonic_increasing


def test_the_extract_has_the_telemetry_columns_and_the_curves(extract_dir: Path):
    dataset = FleetDataset.load(extract_dir)
    assert (dataset.cycles["discharge_c_rate"] == severson.DISCHARGE_C_RATE).all()
    assert (dataset.cells["nameplate_capacity_ah"] == severson.NOMINAL_CAPACITY_AH).all()
    assert np.allclose(dataset.voltage_grid, GRID)
    assert ("b3c0", 10) in dataset.curves and ("b3c0", 100) in dataset.curves
    energy = dataset.cycles["discharge_energy_wh"]
    assert energy.between(1.0, 4.0).all()


def test_the_unchanged_protocol_runs_on_the_extract(extract_dir: Path, tmp_path: Path):
    config = load_resolved(repository_root() / "config" / "severson.yaml")
    config = replace(
        config,
        paths=replace(config.paths, data_dir=str(extract_dir)),
        evaluation=EvaluationConfig(
            n_folds=3,
            early_life_models=("fleet_mean", "ridge"),
            online_models=("fleet_mean", "ridge"),
            deployed_early_life_model="ridge",
            deployed_online_model="ridge",
        ),
    )
    summary = real_evaluate.evaluate(config, output_dir=tmp_path, figures=False, log=lambda _: None)

    assert summary["data_source"] == "real"
    assert set(summary["tasks"]) == {"early_life", "online_soh"}
    early = summary["tasks"]["early_life"]
    assert early["n_batches"] == 3
    assert set(early["protocols"]) == {"cell_independent", "batch_independent"}
    assert any("starved" in note for note in summary["notes"])

    table = real_evaluate.table_headline(summary)
    assert table.count("\n") == 1 + 2 * 2
    assert (tmp_path / "summary.json").exists()


def test_summaries_compare_within_tolerance_and_flag_drift():
    base = {"a": 1.0, "b": [0.5, {"c": 100.0}], "name": "x"}
    assert real_evaluate.compare(base, {"a": 1.0005, "b": [0.5, {"c": 100.1}], "name": "x"}) == []
    drift = real_evaluate.compare(base, {"a": 1.1, "b": [0.5, {"c": 100.0}], "name": "y"})
    assert len(drift) == 2
    assert real_evaluate.compare(base, {"a": 1.0}) == [
        "b: present in only one summary",
        "name: present in only one summary",
    ]


def test_the_committed_real_results_render_into_the_readme(repository_root_path: Path):
    """The README block is generated from the committed summary, never typed."""
    from cobalt.check_readme import extract_block

    summary_path = repository_root_path / "results" / "severson" / "summary.json"
    if not summary_path.exists():
        pytest.skip("no committed real-data results in this copy")
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    readme = (repository_root_path / "README.md").read_text(encoding="utf-8")
    assert extract_block(readme, real_evaluate.README_BLOCK) == real_evaluate.table_headline(
        summary
    )
    assert summary["data_source"] == "real"


def test_gradient_boosting_figures_get_the_documented_slack_and_nothing_else_does():
    """Histogram gradient boosting differs slightly between processors.

    The numbers below are the ones a GitHub runner produced against the
    committed summary: 1.3% on a mean RMSE, over 20% on a fold spread. Those
    must pass. The same 1.3% on a linear model is drift and must not.
    """

    def summary(gbm_rmse: float, gbm_std: float, enet_rmse: float, picp: float) -> dict:
        return {
            "tasks": {
                "online_soh": {
                    "protocols": {
                        "random_cycle": {
                            "models": {
                                "gradient_boosting": {
                                    "rmse_cycles": gbm_rmse,
                                    "rmse_cycles_std": gbm_std,
                                },
                                "ridge": {"rmse_cycles": enet_rmse},
                            },
                            "intervals": {"matched_calibration": {"picp": picp}},
                        }
                    }
                }
            }
        }

    committed = summary(70.5982, 5.51748, 173.0, 0.900249)
    assert real_evaluate.compare(committed, summary(69.6875, 6.8036, 173.0, 0.90271)) == []
    drift = real_evaluate.compare(committed, summary(69.6875, 6.8036, 175.3, 0.90271))
    assert drift == [
        "tasks.online_soh.protocols.random_cycle.models.ridge.rmse_cycles: 173.0 against 175.3"
    ]
    assert real_evaluate.compare(committed, summary(60.0, 5.5, 173.0, 0.90)) != []
