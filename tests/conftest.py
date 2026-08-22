"""Shared fixtures.

One tiny fleet, simulated once per session and reused. It is driven hard enough
that every cell reaches end of life inside a few hundred cycles, because a test
suite that takes four minutes to simulate a realistic fleet is a test suite
people stop running.

The fleet is small enough that no accuracy assertion would mean anything, so no
test in this suite asserts on accuracy. They assert on shapes, invariants,
contracts and the leakage guarantees, which is what tests are for here. The
numbers are the evaluation's job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cobalt.config import Config, EvaluationConfig, OnlineConfig, PathsConfig
from cobalt.data.schema import FleetDataset, write_fleet
from cobalt.sim.fleet import FleetConfig, simulate_fleet


@pytest.fixture(scope="session")
def tiny_fleet_config() -> FleetConfig:
    """A fleet that ages fast: hot, fast-charged and shallow on cycles."""
    return FleetConfig(
        n_batches=4,
        cells_per_batch=4,
        end_of_life_soh=0.80,
        max_cycles=420,
        cycles_past_eol=10,
        site_temperature_mean_c=46.0,
        site_temperature_sigma_c=4.0,
        site_charge_c_rate_choices=(3.0, 4.0, 5.0, 6.0),
        rest_hours_mean=6.0,
        curve_cycles=(2, 10, 100),
    )


@pytest.fixture(scope="session")
def tiny_fleet(tiny_fleet_config: FleetConfig):
    return simulate_fleet(tiny_fleet_config, seed=99)


@pytest.fixture(scope="session")
def tiny_dataset(tiny_fleet, tmp_path_factory: pytest.TempPathFactory) -> FleetDataset:
    """The tiny fleet written to disk and read back, so the round trip is exercised."""
    directory = tmp_path_factory.mktemp("fleet")
    write_fleet(tiny_fleet, directory)
    return FleetDataset.load(directory)


@pytest.fixture(scope="session")
def tiny_config(tiny_fleet_config: FleetConfig, tmp_path_factory: pytest.TempPathFactory) -> Config:
    """A config matching the tiny fleet, with three folds over four batches."""
    root = tmp_path_factory.mktemp("run")
    return Config(
        name="test",
        seed=99,
        paths=PathsConfig(
            data_dir=str(root / "data"),
            results_dir=str(root / "results"),
            analysis_dir=str(root / "analysis"),
            artifacts_dir=str(root / "artifacts"),
        ),
        fleet=tiny_fleet_config,
        evaluation=EvaluationConfig(
            n_folds=2,
            alpha=0.10,
            calibration_fraction=0.30,
            early_life_models=("fleet_mean", "capacity_extrapolation", "ridge"),
            online_models=("fleet_mean", "ridge"),
            deployed_early_life_model="ridge",
            deployed_online_model="ridge",
        ),
        online=OnlineConfig(window_cycles=30, stride=20),
    )


@pytest.fixture(scope="session")
def repository_root_path() -> Path:
    return Path(__file__).resolve().parents[1]
