"""Tests for sequence datasets, sliding windows, and causal 1D-CNN."""

from __future__ import annotations

import numpy as np

from lobx.ml.sequence_models import SnapshotCNN1D, is_torch_available, make_sliding_windows


def test_make_sliding_windows_shapes_and_causality() -> None:
    """Verify that sliding windows strictly align past and current observations."""
    N = 25
    F = 4
    W = 5

    X = np.arange(N * F, dtype=np.float64).reshape(N, F)
    y = np.arange(N, dtype=np.int64)

    seq_data = make_sliding_windows(X, y, window_len=W)
    assert seq_data.X_seq.shape == (N - W + 1, W, F)
    assert seq_data.y.shape == (N - W + 1,)

    # The label for window 0 must correspond to index W - 1
    assert seq_data.y[0] == y[W - 1]
    # The last row of window 0 must be X[W - 1]
    np.testing.assert_array_equal(seq_data.X_seq[0, -1], X[W - 1])
    # The first row of window 0 must be X[0]
    np.testing.assert_array_equal(seq_data.X_seq[0, 0], X[0])


def test_snapshot_cnn1d_forward_axioms() -> None:
    """Verify that SnapshotCNN1D outputs valid probability distributions."""
    B = 12
    W = 10
    F = 8
    rng = np.random.default_rng(303)
    X_seq = rng.normal(0.0, 1.0, (B, W, F))

    cnn = SnapshotCNN1D(window_len=W, num_features=F, num_filters=16, kernel_size=3, seed=42)
    probs = cnn.forward(X_seq)

    assert probs.shape == (B, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-12)


def test_torch_availability_helper() -> None:
    """Verify that the torch availability flag behaves consistently."""
    avail = is_torch_available()
    assert isinstance(avail, bool)
