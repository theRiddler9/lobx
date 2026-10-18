"""Judge a model on held-out data: accuracy, log loss, and calibration.

*Accuracy* alone can mislead when one outcome is common (guessing "flat" every time can look good),
so a model is always compared with the naive baseline. *Log loss* punishes confident wrong answers
and rewards honest probabilities. *Calibration* asks whether "70%" really means 70%: among rows
where the model said about 0.7, did the event happen about 70% of the time?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lobx.ml.dataset import Dataset
from lobx.ml.labels import DOWN, UP
from lobx.ml.model import Classifier, FloatArray

_EPS = 1e-12  # keeps log(0) out of the log loss


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Rows whose predicted probability fell in ``[low, high)``."""

    low: float
    high: float
    mean_predicted: float
    observed_rate: float
    count: int


@dataclass(frozen=True, slots=True)
class Evaluation:
    """How one model did on one dataset."""

    rows: int
    accuracy: float
    log_loss: float
    # Among rows where the mid really moved, how often the model got the direction right.
    directional_accuracy: float
    # Expected calibration error for the "up" and "down" probabilities: lower is better.
    ece_up: float
    ece_down: float


def accuracy(proba: FloatArray, y: npt.NDArray[np.int64]) -> float:
    """Return the fraction of rows whose most probable class is the true class."""
    return float((proba.argmax(axis=1) == y).mean())


def log_loss(proba: FloatArray, y: npt.NDArray[np.int64]) -> float:
    """Return the mean negative log probability given to the true class (lower is better)."""
    true_proba = proba[np.arange(len(y)), y]
    return float(-np.log(np.clip(true_proba, _EPS, 1.0)).mean())


def directional_accuracy(proba: FloatArray, y: npt.NDArray[np.int64]) -> float:
    """Among rows where the mid moved, how often ``p(up) > p(down)`` matches the real direction."""
    moved = y != 1
    if not moved.any():
        return float("nan")
    predicted_up = proba[moved, UP] > proba[moved, DOWN]
    return float((predicted_up == (y[moved] == UP)).mean())


def calibration_table(
    predicted: FloatArray, happened: npt.NDArray[np.bool_], bins: int = 10
) -> list[CalibrationBin]:
    """Group rows by predicted probability and compare with how often the event really happened."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    # digitize puts p=1.0 in the last bin by clipping the index.
    which = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
    table = []
    for b in range(bins):
        in_bin = which == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        table.append(
            CalibrationBin(
                float(edges[b]),
                float(edges[b + 1]),
                float(predicted[in_bin].mean()),
                float(happened[in_bin].mean()),
                count,
            )
        )
    return table


def expected_calibration_error(table: list[CalibrationBin]) -> float:
    """Return the row-weighted average gap between predicted and observed rate (0 is perfect)."""
    total = sum(b.count for b in table)
    return sum(b.count * abs(b.mean_predicted - b.observed_rate) for b in table) / total


def evaluate(model: Classifier, data: Dataset) -> Evaluation:
    """Score ``model`` on ``data`` (which must be seeds the model never trained or stopped on)."""
    proba = model.predict_proba(data.X)
    return Evaluation(
        rows=len(data),
        accuracy=accuracy(proba, data.y),
        log_loss=log_loss(proba, data.y),
        directional_accuracy=directional_accuracy(proba, data.y),
        ece_up=expected_calibration_error(calibration_table(proba[:, UP], data.y == UP)),
        ece_down=expected_calibration_error(calibration_table(proba[:, DOWN], data.y == DOWN)),
    )
