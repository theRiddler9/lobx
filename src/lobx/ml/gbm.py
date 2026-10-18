"""LightGBM: gradient-boosted decision trees.

Boosting builds many small decision trees one after another, each one correcting the mistakes of
the ones before it. Unlike logistic regression it can capture non-linear effects and interactions
(for example "imbalance matters more when the spread is wide").
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from lobx.ml.dataset import Dataset
from lobx.ml.model import NUM_CLASSES, FloatArray

# Small trees, a low learning rate and a minimum leaf size keep the model from memorizing noise.
# num_threads=1 and deterministic=True make training repeatable. Chosen once; never tuned on the
# test seeds.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": NUM_CLASSES,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "seed": 0,
    "deterministic": True,
    "num_threads": 1,
    "verbosity": -1,
}


class LightGBMModel:
    """A multiclass LightGBM model, trained with early stopping on validation seeds."""

    kind = "lightgbm"

    def __init__(self, rounds: int = 400, patience: int = 20) -> None:
        """Create an unfitted model: at most ``rounds`` trees, early-stopping after ``patience``."""
        self._rounds = rounds
        self._patience = patience
        self.booster: lgb.Booster | None = None

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Fit on ``train``; if ``validation`` is given, stop adding trees when it stops improving.

        ``validation`` must come from different seeds than ``train``: early stopping on rows of the
        same runs would reward memorizing them.
        """
        if len(train) == 0:
            raise ValueError("cannot fit on an empty dataset")
        train_set = lgb.Dataset(train.X, label=train.y)
        valid_sets: list[lgb.Dataset] = []
        callbacks: list[Any] = []
        if validation is not None and len(validation) > 0:
            valid_sets = [lgb.Dataset(validation.X, label=validation.y, reference=train_set)]
            callbacks = [lgb.early_stopping(self._patience, verbose=False)]
        self.booster = lgb.train(
            DEFAULT_PARAMS,
            train_set,
            num_boost_round=self._rounds,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return class probabilities, shape ``(len(X), 3)``."""
        if self.booster is None:
            raise RuntimeError("fit the model before predicting")
        proba = np.asarray(self.booster.predict(X), dtype=np.float64)
        return proba.reshape(len(X), NUM_CLASSES)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly description (LightGBM's own text format, as a string)."""
        if self.booster is None:
            raise RuntimeError("fit the model before saving")
        return {"kind": self.kind, "model": self.booster.model_to_string()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LightGBMModel:
        """Rebuild a model written by ``to_dict``."""
        model = cls()
        model.booster = lgb.Booster(model_str=data["model"])
        return model
