"""The learned models.

Three, deliberately small. The dataset here is a few hundred cells and a few
thousand windows, the features are hand-built and physically motivated and
nothing about that shape calls for a deep network. A linear model on good
features is the right first answer and the original Severson result was an
elastic net, so matching it is also the honest comparison.

Every model takes a feature frame rather than an array, so a column can be
missing loudly instead of silently shifting the meaning of every coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FrameRegressor:
    """Wraps an sklearn estimator so it works on named columns.

    The column order is captured at fit time and enforced at predict time. That
    check has caught a real bug in this repository already: the online feature
    builder emits its columns in dictionary order and a later edit that added a
    feature in the middle silently reordered the matrix for anything that had
    already been fitted.
    """

    supports_uncertainty = False

    def __init__(self, name: str, estimator: Pipeline) -> None:
        self.name = name
        self.estimator = estimator
        self.columns: list[str] = []

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> FrameRegressor:
        self.columns = list(features.columns)
        self.estimator.fit(features.to_numpy(dtype=float), np.asarray(target, dtype=float))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != self.columns:
            missing = set(self.columns) - set(features.columns)
            extra = set(features.columns) - set(self.columns)
            raise ValueError(
                f"{self.name} was fitted on a different feature set. "
                f"missing={sorted(missing)} unexpected={sorted(extra)}"
            )
        return np.asarray(self.estimator.predict(features.to_numpy(dtype=float)), dtype=float)

    @property
    def coefficients(self) -> dict[str, float] | None:
        """Standardised coefficients, when the final step is linear."""
        final = self.estimator[-1]
        raw = getattr(final, "coef_", None)
        if raw is None:
            return None
        return {name: float(value) for name, value in zip(self.columns, np.ravel(raw))}


def ridge(seed: int = 0) -> FrameRegressor:
    """Ridge with the penalty chosen by internal cross-validation.

    The alpha grid is wide because the honest protocols leave few effective
    degrees of freedom and the right answer there is a much heavier penalty than
    the leaky protocol wants.
    """
    del seed  # RidgeCV is deterministic; the argument keeps the factories uniform.
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 4, 40))),
        ]
    )
    return FrameRegressor("ridge", pipeline)


def elastic_net(seed: int = 0) -> FrameRegressor:
    """Elastic net, the model the original early-life result used.

    Worth keeping even where it is not the winner, because it is the one that
    says which features it is using. On this data it keeps about half of them
    and puts most of its weight on the capacity fade slope and the charge-time
    drift, not on the delta-Q block that dominates the published real-cell
    result. See the note in `cobalt.features.early_life` for why that difference
    is a property of the simulator rather than a contradiction of the paper.
    """
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                    # An integer here means "this many alphas on the default
                    # path". scikit-learn 1.7 folded the old `n_alphas` argument
                    # into `alphas`, so this spelling is 1.7 and later only.
                    alphas=60,
                    cv=5,
                    max_iter=20_000,
                    random_state=seed,
                ),
            ),
        ]
    )
    return FrameRegressor("elastic_net", pipeline)


def gradient_boosting(seed: int = 0) -> FrameRegressor:
    """Histogram gradient boosting. The one that wins on the leaky protocol.

    Its margin over the linear models shrinks sharply once whole batches are held
    out, which is the expected behaviour and is itself a useful diagnostic: a
    flexible model given a leaky split spends its capacity on the leak.
    """
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=400,
                    learning_rate=0.06,
                    max_leaf_nodes=15,
                    min_samples_leaf=8,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=seed,
                ),
            ),
        ]
    )
    return FrameRegressor("gradient_boosting", pipeline)
