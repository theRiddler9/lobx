"""Save and load fitted models as JSON (human-readable, no pickle, safe to open)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lobx.errors import LobxError
from lobx.ml.gbm import LightGBMModel
from lobx.ml.logistic import LogisticModel
from lobx.ml.model import Classifier, NaiveModel

_KINDS: dict[str, Any] = {
    NaiveModel.kind: NaiveModel,
    LogisticModel.kind: LogisticModel,
    LightGBMModel.kind: LightGBMModel,
}


class ModelFormatError(LobxError, ValueError):
    """A model file could not be read: bad JSON or an unknown model kind."""


def save_model(model: Classifier, path: Path) -> None:
    """Write a fitted model to ``path`` as JSON."""
    to_dict = getattr(model, "to_dict", None)
    if to_dict is None:
        raise TypeError(f"{type(model).__name__} cannot be saved: it has no to_dict()")
    path.write_text(json.dumps(to_dict()))


def load_model(path: Path) -> Classifier:
    """Read a model written by ``save_model``."""
    try:
        data = json.loads(path.read_text())
        kind = data["kind"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelFormatError(f"{path} is not a LOBX model file: {error!r}") from error
    if kind not in _KINDS:
        raise ModelFormatError(f"{path} has unknown model kind {kind!r}; known: {sorted(_KINDS)}")
    model: Classifier = _KINDS[kind].from_dict(data)
    return model
