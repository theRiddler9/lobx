"""Temporal sequence models for limit order book forecasting over windows of snapshots.

Supports both a pure-NumPy causal 1D-CNN implementation and an optional PyTorch
LSTM/1D-CNN adapter for deep sequential representations without lookahead leakage.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from lobx.ml.model import NUM_CLASSES, FloatArray


@dataclass(frozen=True, slots=True)
class SequenceData:
    """Sliding-window sequence data.

    Shapes:
        X_seq: (N_samples, window_len, num_features)
        y: (N_samples,)
    """

    X_seq: np.ndarray
    y: np.ndarray


def make_sliding_windows(
    X: FloatArray,
    y: np.ndarray,
    window_len: int = 10,
) -> SequenceData:
    """Construct causal sliding windows of length ``window_len`` from sequential snapshot features.

    For each index i >= window_len - 1, the sample features are X[i - window_len + 1 : i + 1].
    Crucially, only past and current snapshots are included; never future snapshots.
    """
    N, F = X.shape
    if window_len > N:
        return SequenceData(
            np.empty((0, window_len, F), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    num_windows = N - window_len + 1
    X_seq = np.empty((num_windows, window_len, F), dtype=np.float64)
    for i in range(num_windows):
        X_seq[i] = X[i : i + window_len]

    y_seq = y[window_len - 1 :]
    return SequenceData(X_seq, y_seq)


class SnapshotCNN1D:
    """A causal 1D Temporal Convolutional network over limit order book sequences.

    Architecture:
        Input: (Batch, window_len, num_features)
        Conv1D: filters across temporal window with ReLU
        Global Temporal Max-Pool: captures peak order flow impulses
        Linear: (num_filters -> NUM_CLASSES)
        Softmax: (Down, Flat, Up)
    """

    def __init__(
        self,
        window_len: int = 10,
        num_features: int = 8,
        num_filters: int = 16,
        kernel_size: int = 3,
        seed: int = 42,
    ) -> None:
        """Initialize weights using He / Kaiming normal distributions."""
        self.window_len = window_len
        self.num_features = num_features
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.rng = np.random.default_rng(seed)

        # Conv filters: shape (num_filters, kernel_size, num_features)
        fan_in = kernel_size * num_features
        self.W_conv = self.rng.normal(
            0.0, np.sqrt(2.0 / fan_in), (num_filters, kernel_size, num_features)
        )
        self.b_conv = np.zeros(num_filters, dtype=np.float64)

        # Dense layer: shape (num_filters, NUM_CLASSES)
        self.W_dense = self.rng.normal(
            0.0, np.sqrt(2.0 / (num_filters + NUM_CLASSES)), (num_filters, NUM_CLASSES)
        )
        self.b_dense = np.zeros(NUM_CLASSES, dtype=np.float64)

    def forward(self, X_seq: np.ndarray) -> np.ndarray:
        """Forward pass over sequence batch.

        Args:
            X_seq: Array of shape (B, window_len, num_features).

        Returns:
            Class probabilities of shape (B, NUM_CLASSES).
        """
        B = X_seq.shape[0]
        if B == 0:
            return np.empty((0, NUM_CLASSES), dtype=np.float64)

        # Output time steps after valid convolution
        out_time = self.window_len - self.kernel_size + 1
        conv_out = np.zeros((B, out_time, self.num_filters), dtype=np.float64)

        # Valid 1D convolution over temporal dimension
        for t in range(out_time):
            window_slice = X_seq[:, t : t + self.kernel_size, :]  # (B, K, F)
            # Dot product with each filter
            for f in range(self.num_filters):
                kernel = self.W_conv[f]  # (K, F)
                conv_out[:, t, f] = np.sum(window_slice * kernel, axis=(1, 2)) + self.b_conv[f]

        # ReLU activation
        conv_relu = np.maximum(0.0, conv_out)

        # Global temporal max pooling: (B, num_filters)
        pooled = np.max(conv_relu, axis=1)

        # Linear projection to logits: (B, NUM_CLASSES)
        logits = np.dot(pooled, self.W_dense) + self.b_dense

        # Numerically stable Softmax
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shift_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.asarray(probs, dtype=np.float64)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        return self.forward(X_seq)


def is_torch_available() -> bool:
    """Return True if PyTorch is installed and importable in this environment."""
    return importlib.util.find_spec("torch") is not None
