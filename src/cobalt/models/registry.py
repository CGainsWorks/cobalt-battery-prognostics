"""One place that knows which models exist and which task each one fits.

Kept separate so the evaluation loop, the command line and the service all name
models the same way and so an unknown name fails at the point somebody typed it
rather than three stages later.
"""

from __future__ import annotations

from collections.abc import Callable

from cobalt.models.baselines import CapacityExtrapolation, FleetMean, SlopeExtrapolation
from cobalt.models.regressors import elastic_net, gradient_boosting, ridge

TASKS = ("early_life", "online_soh")

#: Models available for each task. The baselines come first because that is the
#: order the results tables read best in: floor, engineering rule of thumb, then
#: the learned models that have to justify themselves against both.
_EARLY_LIFE: dict[str, Callable[..., object]] = {
    "fleet_mean": lambda seed=0, **_: FleetMean(),
    "capacity_extrapolation": lambda seed=0, rated_capacity_ah=2.20, end_of_life_soh=0.80, **_: (
        CapacityExtrapolation(rated_capacity_ah=rated_capacity_ah, end_of_life_soh=end_of_life_soh)
    ),
    "ridge": lambda seed=0, **_: ridge(seed),
    "elastic_net": lambda seed=0, **_: elastic_net(seed),
    "gradient_boosting": lambda seed=0, **_: gradient_boosting(seed),
}

_ONLINE_SOH: dict[str, Callable[..., object]] = {
    "fleet_mean": lambda seed=0, **_: FleetMean(),
    "slope_extrapolation": lambda seed=0, end_of_life_soh=0.80, **_: SlopeExtrapolation(
        end_of_life_soh=end_of_life_soh
    ),
    "ridge": lambda seed=0, **_: ridge(seed),
    "gradient_boosting": lambda seed=0, **_: gradient_boosting(seed),
}

_BY_TASK = {"early_life": _EARLY_LIFE, "online_soh": _ONLINE_SOH}


def available(task: str) -> tuple[str, ...]:
    """Model names valid for a task, in report order."""
    if task not in _BY_TASK:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    return tuple(_BY_TASK[task])


def build(task: str, name: str, **kwargs):
    """Construct a model by task and name.

    ``kwargs`` are passed to every factory and ignored by the ones that do not
    need them, so the caller does not have to know which models take a rated
    capacity and which do not.
    """
    models = _BY_TASK.get(task)
    if models is None:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    factory = models.get(name)
    if factory is None:
        raise ValueError(
            f"unknown model {name!r} for task {task!r}. Available: {', '.join(models)}"
        )
    return factory(**kwargs)


def is_baseline(name: str) -> bool:
    """Baselines are reported but never shipped, so the service refuses them."""
    return name in {"fleet_mean", "capacity_extrapolation", "slope_extrapolation"}
