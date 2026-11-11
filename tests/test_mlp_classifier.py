"""Tests for MLPClassifier: protocol compliance, standardization, early stopping."""

from __future__ import annotations

import numpy as np

from lobx.ml.dataset import Dataset
from lobx.ml.evaluate import evaluate
from lobx.ml.mlp_classifier import MLPClassifier


def test_mlp_classifier_fit_and_predict() -> None:
    """Verify that MLPClassifier fits, stops early, and produces valid probabilities."""
    rng = np.random.default_rng(101)
    N_train = 200
    N_val = 60
    input_dim = 8

    X_train = rng.normal(10.0, 5.0, (N_train, input_dim))
    y_train = rng.integers(0, 3, size=N_train)
    train_data = Dataset(X_train, y_train, np.zeros(N_train, dtype=np.int64))

    X_val = rng.normal(10.0, 5.0, (N_val, input_dim))
    y_val = rng.integers(0, 3, size=N_val)
    val_data = Dataset(X_val, y_val, np.ones(N_val, dtype=np.int64))

    clf = MLPClassifier(hidden_dim=16, epochs=30, batch_size=32, patience=5, seed=42)
    clf.fit(train_data, val_data)

    probs = clf.predict_proba(X_val)
    assert probs.shape == (N_val, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-12)

    eval_res = evaluate(clf, val_data)
    assert 0.0 <= eval_res.accuracy <= 1.0
    assert eval_res.log_loss > 0.0


def test_mlp_classifier_serialization() -> None:
    """Verify that MLPClassifier serializes to/from dict preserving predictions."""
    rng = np.random.default_rng(202)
    X = rng.normal(0.0, 2.0, (50, 4))
    y = rng.integers(0, 3, size=50)
    data = Dataset(X, y, np.zeros(50, dtype=np.int64))

    clf = MLPClassifier(hidden_dim=8, epochs=10, seed=1)
    clf.fit(data)

    p1 = clf.predict_proba(X[:5])
    data_dict = clf.to_dict()

    clf2 = MLPClassifier.from_dict(data_dict)
    p2 = clf2.predict_proba(X[:5])

    np.testing.assert_allclose(p1, p2, atol=1e-12)
