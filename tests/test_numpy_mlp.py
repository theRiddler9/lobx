"""Mathematical gradient checking and correctness tests for the NumPy MLP."""

from __future__ import annotations

import numpy as np

from lobx.ml.numpy_mlp import NumPyMLP


def _numerical_gradient(
    mlp: NumPyMLP,
    param_name: str,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute numerical gradient for a parameter matrix via two-sided finite difference."""
    param = getattr(mlp, param_name)
    num_grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        orig_val = param[idx]

        # f(theta + eps)
        param[idx] = orig_val + eps
        probs_plus = mlp.forward(X)
        loss_plus = mlp.cross_entropy_loss(probs_plus, y)

        # f(theta - eps)
        param[idx] = orig_val - eps
        probs_minus = mlp.forward(X)
        loss_minus = mlp.cross_entropy_loss(probs_minus, y)

        # Restore original value
        param[idx] = orig_val

        # Central difference approximation: (f(theta+eps) - f(theta-eps)) / (2*eps)
        num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return np.asarray(num_grad, dtype=np.float64)


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative error between analytical and numerical gradients."""
    diff = np.linalg.norm(a - b)
    denom = np.linalg.norm(a) + np.linalg.norm(b) + 1e-15
    return float(diff / denom)


def test_analytical_gradient_check() -> None:
    """Verify that analytical backpropagation gradients match finite-difference numerical gradients.

    The relative error between analytical and numerical gradients must be < 1e-6.
    """
    rng = np.random.default_rng(123)
    N = 20
    input_dim = 6
    hidden_dim = 8
    output_dim = 3

    X = rng.normal(0.0, 1.0, (N, input_dim))
    y = rng.integers(0, output_dim, size=N)

    mlp = NumPyMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, seed=42)

    # 1. Analytical backward pass
    mlp.forward(X)
    analytical_grads = mlp.backward(y)

    # 2. Compare against numerical gradients for each parameter tensor
    for param_name in ["W1", "b1", "W2", "b2"]:
        num_grad = _numerical_gradient(mlp, param_name, X, y, eps=1e-6)
        ana_grad = analytical_grads[param_name]

        rel_err = _relative_error(ana_grad, num_grad)
        assert rel_err < 1e-6, (
            f"Gradient check failed for {param_name}: relative error {rel_err:.2e} >= 1e-6"
        )


def test_softmax_probability_axioms() -> None:
    """Verify that forward pass produces mathematically valid probability distributions."""
    rng = np.random.default_rng(456)
    X = rng.normal(0.0, 2.0, (50, 8))

    mlp = NumPyMLP(input_dim=8, hidden_dim=16, output_dim=3, seed=99)
    probs = mlp.forward(X)

    # Probabilities in [0, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Row sums equal 1.0
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


def test_mlp_can_overfit_small_dataset() -> None:
    """Verify that gradient descent drives loss down on separable data."""
    rng = np.random.default_rng(789)
    X = np.vstack(
        [
            rng.normal(-2.0, 0.5, (20, 4)),
            rng.normal(0.0, 0.5, (20, 4)),
            rng.normal(2.0, 0.5, (20, 4)),
        ]
    )
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)

    mlp = NumPyMLP(input_dim=4, hidden_dim=16, output_dim=3, seed=42)
    history = mlp.fit(X, y, epochs=100, batch_size=20, lr=0.1, weight_decay=1e-5)

    # Loss must decrease substantially
    assert history[-1] < history[0] * 0.1
    # Predictions must achieve > 95% accuracy
    preds = mlp.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95, f"Expected acc >= 0.95, got {acc}"
