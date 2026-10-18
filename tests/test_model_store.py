from pathlib import Path

import numpy as np
import pytest

from lobx.ml.dataset import Dataset
from lobx.ml.gbm import LightGBMModel
from lobx.ml.logistic import LogisticModel
from lobx.ml.model import Classifier, NaiveModel
from lobx.ml.store import ModelFormatError, load_model, save_model


def _data() -> Dataset:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=600)
    X = np.column_stack([y + rng.normal(0, 0.5, size=600), rng.normal(size=600)])
    return Dataset(X, y.astype(np.int64), np.zeros(600, dtype=np.int64))


@pytest.mark.parametrize("make", [NaiveModel, LogisticModel, LightGBMModel])
def test_a_loaded_model_predicts_exactly_like_the_original(tmp_path: Path, make: type) -> None:
    data = _data()
    model: Classifier = make()
    model.fit(data)
    path = tmp_path / "model.json"
    save_model(model, path)
    reloaded = load_model(path)
    assert type(reloaded) is make
    assert np.allclose(reloaded.predict_proba(data.X), model.predict_proba(data.X))


def test_unknown_kind_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text('{"kind": "magic"}')
    with pytest.raises(ModelFormatError, match="unknown model kind 'magic'"):
        load_model(path)


def test_garbage_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text("not json")
    with pytest.raises(ModelFormatError, match="not a LOBX model file"):
        load_model(path)


def test_saving_an_unfitted_model_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        save_model(LogisticModel(), tmp_path / "m.json")
