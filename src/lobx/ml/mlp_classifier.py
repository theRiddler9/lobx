"""Classifier implementation wrapping NumPyMLP with standardization and early stopping.

Conforms to the Classifier protocol used by the LOBX ML evaluation pipeline.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from lobx.ml.dataset import Dataset
from lobx.ml.model import NUM_CLASSES, FloatArray
from lobx.ml.numpy_mlp import NumPyMLP


class MLPClassifier:
    """A standardizing, early-stopping Classifier powered by NumPyMLP."""

    kind = "mlp"

    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 80,
        batch_size: int = 128,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
        patience: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize the MLP classifier with architecture and training hyperparameters."""
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed

        self.mlp: NumPyMLP | None = None
        self.mean: FloatArray | None = None
        self.std: FloatArray | None = None

    def _standardize(self, X: FloatArray) -> FloatArray:
        """Standardize features using training mean and standard deviation."""
        assert self.mean is not None and self.std is not None
        safe_std = np.where(self.std < 1e-8, 1.0, self.std)
        return np.asarray((X - self.mean) / safe_std, dtype=np.float64)

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Fit the MLP on train with early stopping on validation seeds.

        Args:
            train: Training dataset.
            validation: Validation dataset from disjoint seeds.
        """
        if len(train) == 0:
            raise ValueError("cannot fit on an empty dataset")

        input_dim = train.X.shape[1]
        self.mean = np.mean(train.X, axis=0)
        self.std = np.std(train.X, axis=0)

        X_train = self._standardize(train.X)
        y_train = train.y

        X_val: FloatArray | None = None
        y_val: np.ndarray | None = None
        if validation is not None and len(validation) > 0:
            X_val = self._standardize(validation.X)
            y_val = validation.y

        self.mlp = NumPyMLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=NUM_CLASSES,
            seed=self.seed,
        )

        best_loss = float("inf")
        best_w1 = copy.deepcopy(self.mlp.W1)
        best_b1 = copy.deepcopy(self.mlp.b1)
        best_w2 = copy.deepcopy(self.mlp.W2)
        best_b2 = copy.deepcopy(self.mlp.b2)
        patience_counter = 0

        N = X_train.shape[0]
        rng = np.random.default_rng(self.seed)

        for _ in range(self.epochs):
            indices = rng.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            num_batches = int(np.ceil(N / self.batch_size))
            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, N)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self.mlp.forward(X_batch)
                grads = self.mlp.backward(y_batch)
                self.mlp.update(grads, lr=self.lr, weight_decay=self.weight_decay)

            if X_val is not None and y_val is not None:
                val_probs = self.mlp.forward(X_val)
                val_loss = self.mlp.cross_entropy_loss(val_probs, y_val)
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_w1 = copy.deepcopy(self.mlp.W1)
                    best_b1 = copy.deepcopy(self.mlp.b1)
                    best_w2 = copy.deepcopy(self.mlp.W2)
                    best_b2 = copy.deepcopy(self.mlp.b2)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if X_val is not None and y_val is not None:
            self.mlp.W1 = best_w1
            self.mlp.b1 = best_b1
            self.mlp.W2 = best_w2
            self.mlp.b2 = best_b2

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return class probabilities matrix of shape (len(X), 3)."""
        if self.mlp is None or self.mean is None or self.std is None:
            raise RuntimeError("fit the model before predicting")
        X_norm = self._standardize(X)
        return self.mlp.predict_proba(X_norm)

    def to_dict(self) -> dict[str, Any]:
        """Serialize model parameters to a JSON-compatible dictionary."""
        if self.mlp is None or self.mean is None or self.std is None:
            raise RuntimeError("fit the model before saving")
        return {
            "kind": self.kind,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "W1": self.mlp.W1.tolist(),
            "b1": self.mlp.b1.tolist(),
            "W2": self.mlp.W2.tolist(),
            "b2": self.mlp.b2.tolist(),
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLPClassifier:
        """Deserialize model from a dictionary."""
        model = cls(hidden_dim=data["hidden_dim"])
        model.mean = np.array(data["mean"], dtype=np.float64)
        model.std = np.array(data["std"], dtype=np.float64)
        input_dim = len(model.mean)
        model.mlp = NumPyMLP(input_dim=input_dim, hidden_dim=model.hidden_dim)
        model.mlp.W1 = np.array(data["W1"], dtype=np.float64)
        model.mlp.b1 = np.array(data["b1"], dtype=np.float64)
        model.mlp.W2 = np.array(data["W2"], dtype=np.float64)
        model.mlp.b2 = np.array(data["b2"], dtype=np.float64)
        return model
