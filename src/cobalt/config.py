"""Configuration.

One YAML file describes a whole run: the fleet to simulate, the protocols and
models to evaluate, the window the online task uses and the commercial
assumptions the cost models are built on.

Two files ship. ``config/default.yaml`` produces every number in the README.
``config/ci.yaml`` is the same pipeline on a smaller fleet so continuous
integration proves the wiring in a couple of minutes without pretending the
numbers mean anything.

Unknown keys are an error rather than a shrug. A typo in a config file that is
silently ignored produces a run that looks fine and answers a different
question, which is the worst failure mode available here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from cobalt.business.cost import ReplacementCostModel
from cobalt.business.grading import GradingPolicy
from cobalt.sim.fleet import FleetConfig


@dataclass(frozen=True)
class EvaluationConfig:
    n_folds: int = 4
    #: Miscoverage rate for the conformal intervals. 0.10 asks for 90%.
    alpha: float = 0.10
    calibration_fraction: float = 0.30
    early_life_models: tuple[str, ...] = (
        "fleet_mean",
        "capacity_extrapolation",
        "ridge",
        "elastic_net",
        "gradient_boosting",
    )
    online_models: tuple[str, ...] = (
        "fleet_mean",
        "slope_extrapolation",
        "ridge",
        "gradient_boosting",
    )
    #: Model carried through to the cost analysis, the service and the model card.
    deployed_early_life_model: str = "elastic_net"
    deployed_online_model: str = "gradient_boosting"


@dataclass(frozen=True)
class OnlineConfig:
    window_cycles: int = 50
    stride: int = 25


@dataclass(frozen=True)
class QualificationConfig:
    """How long a cycler channel takes per cycle, for the test-time saving."""

    hours_per_cycle: float = 1.6
    channels: int = 1


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data/fleet"
    results_dir: str = "results"
    analysis_dir: str = "analysis"
    artifacts_dir: str = "artifacts"


@dataclass(frozen=True)
class Config:
    """The whole run."""

    name: str = "default"
    seed: int = 20260822
    paths: PathsConfig = field(default_factory=PathsConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    grading: GradingPolicy = field(default_factory=GradingPolicy)
    replacement: ReplacementCostModel = field(default_factory=ReplacementCostModel)
    qualification: QualificationConfig = field(default_factory=QualificationConfig)

    def resolved(self, root: Path) -> Config:
        """Return a copy with every path made absolute against ``root``.

        Relative paths in a config file are relative to the repository, not to
        wherever the command happened to be run from. Getting this wrong writes
        results into a user's home directory and it is not obvious for a while.
        """
        root = Path(root)
        paths = PathsConfig(
            **{
                key: str((root / value).resolve()) if not Path(value).is_absolute() else value
                for key, value in asdict(self.paths).items()
            }
        )
        return _replace_frozen(self, paths=paths)

    def as_dict(self) -> dict[str, Any]:
        return _to_plain(self)


def _replace_frozen(instance: Any, **updates: Any) -> Any:
    current = {f.name: getattr(instance, f.name) for f in fields(instance)}
    current.update(updates)
    return type(instance)(**current)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def _build(cls: type, payload: dict[str, Any] | None, where: str) -> Any:
    """Instantiate a dataclass from a mapping, rejecting keys it does not have."""
    if payload is None:
        return cls()
    if not isinstance(payload, dict):
        raise TypeError(f"{where} must be a mapping, got {type(payload).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(
            f"{where} has unknown keys: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        # Tuple-typed fields come out of YAML as lists and a dataclass that
        # advertises a tuple and holds a list compares unequal to itself after a
        # round trip through the manifest. The nesting matters too: the grading
        # cost matrix is a tuple of tuples and YAML gives a list of lists.
        default = known[key].default
        kwargs[key] = _as_tuple(value) if isinstance(default, tuple) else value
    return cls(**kwargs)


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def load_config(path: str | Path) -> Config:
    """Read a config file. Raises on anything it does not recognise."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a mapping at the top level")

    sections = {
        "paths": PathsConfig,
        "fleet": FleetConfig,
        "evaluation": EvaluationConfig,
        "online": OnlineConfig,
        "grading": GradingPolicy,
        "replacement": ReplacementCostModel,
        "qualification": QualificationConfig,
    }
    scalar_keys = {"name", "seed"}
    unknown = set(payload) - set(sections) - scalar_keys
    if unknown:
        raise ValueError(
            f"{path} has unknown top-level keys: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(set(sections) | scalar_keys))}"
        )

    built = {
        name: _build(cls, payload.get(name), f"{path}:{name}") for name, cls in sections.items()
    }
    return Config(
        name=str(payload.get("name", path.stem)),
        seed=int(payload.get("seed", 20260822)),
        **built,
    )


def repository_root() -> Path:
    """The repository root, found from this file rather than from the cwd."""
    return Path(__file__).resolve().parents[2]


def load_resolved(path: str | Path) -> Config:
    """Load a config and make its paths absolute. The usual entry point."""
    return load_config(path).resolved(repository_root())
