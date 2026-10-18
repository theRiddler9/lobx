"""The model interface, and the naive baseline every other model has to beat.

A model maps feature rows to class probabilities: an array of shape ``(rows, 3)`` whose columns
are the probabilities of (down, flat, up), in the order of ``labels.CLASS_NAMES``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from lobx.ml.dataset import Dataset
from lobx.ml.labels import CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)
FloatArray = npt.NDArray[np.float64]


class Classifier(Protocol):
    """Anything that can be fitted to a ``Dataset`` and then predict class probabilities."""

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Learn from ``train``. ``validation`` (other seeds) may be used to stop training early."""
        ...

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return an array of shape ``(len(X), 3)`` of class probabilities."""
        ...


class NaiveModel:
    """Always predicts the training label frequencies, ignoring the features.

    This is the bar to clear. If a model cannot beat "guess the most common outcome", its features
    carry no usable information.
    """

    kind = "naive"

    def __init__(self) -> None:
        """Create an unfitted model."""
        self.frequencies: FloatArray | None = None

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Count how often each class occurs in ``train``."""
        if len(train) == 0:
            raise ValueError("cannot fit on an empty dataset")
        counts = np.bincount(train.y, minlength=NUM_CLASSES).astype(np.float64)
        self.frequencies = counts / counts.sum()

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return the same class frequencies for every row."""
        if self.frequencies is None:
            raise RuntimeError("fit the model before predicting")
        return np.tile(self.frequencies, (len(X), 1))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the fitted model."""
        if self.frequencies is None:
            raise RuntimeError("fit the model before saving")
        return {"kind": self.kind, "frequencies": self.frequencies.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NaiveModel:
        """Rebuild a model written by ``to_dict``."""
        model = cls()
        model.frequencies = np.array(data["frequencies"], dtype=np.float64)
        return model
