"""A 2-layer Multi-Layer Perceptron built strictly from scratch using NumPy.

Implements forward propagation, numerically stable Softmax and Cross-Entropy loss,
exact analytical backpropagation via the chain rule, and parameter optimization.
"""

from __future__ import annotations

import numpy as np


class NumPyMLP:
    """Two-layer neural network with ReLU activations and Softmax output.

    Architecture:
        x (input_dim) -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> Softmax -> p (output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 3,
        seed: int = 42,
    ) -> None:
        """Initialize MLP weights and biases using He and Xavier normal initializations."""
        self.rng = np.random.default_rng(seed)
        # He / Kaiming normal initialization for ReLU layer
        self.W1: np.ndarray = self.rng.normal(
            0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)
        )
        self.b1: np.ndarray = np.zeros((1, hidden_dim), dtype=np.float64)

        # Xavier / Glorot initialization for output layer
        self.W2: np.ndarray = self.rng.normal(
            0.0, np.sqrt(2.0 / (hidden_dim + output_dim)), (hidden_dim, output_dim)
        )
        self.b2: np.ndarray = np.zeros((1, output_dim), dtype=np.float64)

        # Cache for backpropagation
        self._cache_X: np.ndarray | None = None
        self._cache_z1: np.ndarray | None = None
        self._cache_a1: np.ndarray | None = None
        self._cache_probs: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass and return predicted class probabilities.

        Args:
            X: Input matrix of shape (N, input_dim).

        Returns:
            Probabilities matrix of shape (N, output_dim).
        """
        X_mat = np.asarray(X, dtype=np.float64)
        z1 = np.dot(X_mat, self.W1) + self.b1
        a1 = np.maximum(0.0, z1)  # ReLU activation

        z2 = np.dot(a1, self.W2) + self.b2

        # Numerically stable Softmax: subtract max per row to prevent exp overflow
        shift_z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exp_scores = np.exp(shift_z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self._cache_X = X_mat
        self._cache_z1 = z1
        self._cache_a1 = a1
        self._cache_probs = probs
        return np.asarray(probs, dtype=np.float64)

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
        """Compute categorical cross-entropy loss.

        Args:
            probs: Predicted probabilities (N, num_classes).
            y: Target class indices (N,) with integer labels in [0, num_classes-1].
            eps: Epsilon clipping to prevent log(0).
        """
        N = probs.shape[0]
        clipped_probs = np.clip(probs, eps, 1.0 - eps)
        # Select probability corresponding to the correct class for each sample
        correct_logprobs = -np.log(clipped_probs[np.arange(N), y])
        return float(np.mean(correct_logprobs))

    def backward(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Compute analytical gradients of cross-entropy loss with respect to parameters.

        Args:
            y: True class labels (N,) with values in [0, output_dim-1].

        Returns:
            Dictionary mapping parameter names ('W1', 'b1', 'W2', 'b2') to their gradients.
        """
        assert self._cache_X is not None
        assert self._cache_z1 is not None
        assert self._cache_a1 is not None
        assert self._cache_probs is not None
        N = self._cache_X.shape[0]

        # Derivative of Softmax + Cross-Entropy loss w.r.t logits z2: (probs - y_one_hot) / N
        dz2 = self._cache_probs.copy()
        dz2[np.arange(N), y] -= 1.0
        dz2 /= N

        # Gradients for W2 and b2
        dW2 = np.dot(self._cache_a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Backprop through ReLU: dz1 = (dz2 @ W2.T) * (z1 > 0)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self._cache_z1 > 0)

        # Gradients for W1 and b1
        dW1 = np.dot(self._cache_X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def update(
        self,
        grads: dict[str, np.ndarray],
        lr: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> None:
        """Update parameters with gradient descent and L2 regularization."""
        self.W1 -= lr * (grads["W1"] + weight_decay * self.W1)
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * (grads["W2"] + weight_decay * self.W2)
        self.b2 -= lr * grads["b2"]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        """Train the MLP using mini-batch gradient descent."""
        N = X.shape[0]
        history: list[float] = []

        for _ in range(epochs):
            indices = self.rng.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(N / batch_size))

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, N)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                probs = self.forward(X_batch)
                loss = self.cross_entropy_loss(probs, y_batch)
                epoch_loss += loss * (end - start)

                grads = self.backward(y_batch)
                self.update(grads, lr=lr, weight_decay=weight_decay)

            history.append(epoch_loss / N)

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (argmax of probabilities)."""
        probs = self.forward(X)
        return np.asarray(np.argmax(probs, axis=1))
