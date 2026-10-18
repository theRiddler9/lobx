import numpy as np
import pytest

from lobx.ml.dataset import Dataset
from lobx.ml.model import NaiveModel


def make_dataset(y: list[int]) -> Dataset:
    n = len(y)
    return Dataset(np.zeros((n, 2)), np.array(y, dtype=np.int64), np.zeros(n, dtype=np.int64))


def test_naive_model_predicts_training_frequencies_for_every_row() -> None:
    model = NaiveModel()
    model.fit(make_dataset([0, 1, 1, 1, 2, 2, 1, 1]))  # 1 down, 5 flat, 2 up
    proba = model.predict_proba(np.zeros((3, 2)))
    assert proba.shape == (3, 3)
    assert np.allclose(proba, [[0.125, 0.625, 0.25]] * 3)


def test_naive_model_must_be_fitted_first() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        NaiveModel().predict_proba(np.zeros((1, 2)))


def test_naive_model_rejects_an_empty_dataset() -> None:
    with pytest.raises(ValueError, match="empty"):
        NaiveModel().fit(make_dataset([]))


def _separable(n: int = 600, seed: int = 0) -> Dataset:
    """Three classes whose first feature clearly separates them; the second is pure noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    X = np.column_stack([y + rng.normal(0, 0.3, size=n), rng.normal(size=n)])
    return Dataset(X, y.astype(np.int64), np.zeros(n, dtype=np.int64))


def test_logistic_model_learns_a_separable_problem() -> None:
    from lobx.ml.logistic import LogisticModel

    train, test = _separable(seed=0), _separable(seed=1)
    model = LogisticModel()
    model.fit(train)
    proba = model.predict_proba(test.X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba.argmax(axis=1) == test.y).mean() > 0.9


def test_logistic_model_handles_a_class_missing_from_training() -> None:
    from lobx.ml.logistic import LogisticModel

    data = _separable()
    keep = data.y != 1
    train = Dataset(data.X[keep], data.y[keep], data.seeds[keep])
    model = LogisticModel()
    model.fit(train)
    proba = model.predict_proba(data.X)
    assert proba.shape == (len(data), 3)
    assert (proba[:, 1] == 0).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_model_tolerates_a_constant_feature() -> None:
    from lobx.ml.logistic import LogisticModel

    data = _separable()
    train = Dataset(np.column_stack([data.X, np.ones(len(data))]), data.y, data.seeds)
    model = LogisticModel()
    model.fit(train)
    assert np.isfinite(model.predict_proba(train.X)).all()


def test_logistic_model_must_be_fitted_first() -> None:
    from lobx.ml.logistic import LogisticModel

    with pytest.raises(RuntimeError, match="fit"):
        LogisticModel().predict_proba(np.zeros((1, 2)))

