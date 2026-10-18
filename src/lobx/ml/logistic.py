"""Multinomial logistic regression: the simple, readable model.

It learns one weight per feature per class and turns the weighted sum into probabilities with the
softmax function. It can only draw straight-line boundaries between classes, which makes it a good
test of whether the features carry signal at all, and easy to inspect: a feature's weight says how
much it pushes each outcome.

scikit-learn does the fitting; the fitted numbers are copied out so that prediction is plain numpy
and the model saves as plain JSON (no pickle).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from lobx.ml.dataset import Dataset
from lobx.ml.model import NUM_CLASSES, FloatArray


class LogisticModel:
    """Standardize the features, then fit a multinomial logistic regression."""

    kind = "logistic"

    def __init__(self, c: float = 1.0) -> None:
        """Create an unfitted model; ``c`` is the inverse regularization strength."""
        self._c = c
        self.mean: FloatArray | None = None
        self.scale: FloatArray | None = None
        self.coef: FloatArray | None = None  # shape (classes seen, features)
        self.intercept: FloatArray | None = None
        self.classes: list[int] = []

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Fit on ``train``. ``validation`` is unused: there is nothing to stop early."""
        if len(train) == 0:
            raise ValueError("cannot fit on an empty dataset")
        # Features have very different scales (spread in ticks, imbalance in [-1, 1]); putting them
        # on one scale lets a single regularization strength treat them fairly.
        self.mean = train.X.mean(axis=0)
        scale = train.X.std(axis=0)
        self.scale = np.where(scale > 0, scale, 1.0)  # a constant column must not divide by zero
        fitted = LogisticRegression(C=self._c, max_iter=1000)
        fitted.fit((train.X - self.mean) / self.scale, train.y)
        self.coef = fitted.coef_.astype(np.float64)
        self.intercept = fitted.intercept_.astype(np.float64)
        self.classes = [int(c) for c in fitted.classes_]

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return class probabilities: softmax of the per-class weighted sums."""
        if self.mean is None or self.scale is None or self.coef is None or self.intercept is None:
            raise RuntimeError("fit the model before predicting")
        scores = ((X - self.mean) / self.scale) @ self.coef.T + self.intercept
        if scores.shape[1] == 1:
            # With only two classes seen, scikit-learn stores one row: the score of the second class
            # against the first. Softmax of (0, score) is exactly the logistic function.
            scores = np.column_stack([np.zeros(len(X)), scores[:, 0]])
        scores -= scores.max(
            axis=1, keepdims=True
        )  # subtracting the max keeps exp() from overflowing
        weights = np.exp(scores)
        seen = weights / weights.sum(axis=1, keepdims=True)
        proba = np.zeros((len(X), NUM_CLASSES))
        proba[:, self.classes] = seen  # a class missing from training gets probability 0
        return proba

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the fitted model."""
        if self.mean is None or self.scale is None or self.coef is None or self.intercept is None:
            raise RuntimeError("fit the model before saving")
        return {
            "kind": self.kind,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "classes": self.classes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticModel:
        """Rebuild a model written by ``to_dict``."""
        model = cls()
        model.mean = np.array(data["mean"], dtype=np.float64)
        model.scale = np.array(data["scale"], dtype=np.float64)
        model.coef = np.array(data["coef"], dtype=np.float64)
        model.intercept = np.array(data["intercept"], dtype=np.float64)
        model.classes = [int(c) for c in data["classes"]]
        return model
