"""Tests for PPO Agent: GAE advantages, surrogate loss, policy updates, and execution."""

from __future__ import annotations

import numpy as np

from lobx.rl.environment import EnvConfig
from lobx.rl.ppo import PPOAgent, Transition


def test_ppo_agent_policy_and_value_axioms() -> None:
    """Verify that PPO agent policy outputs valid probability distributions."""
    agent = PPOAgent(obs_dim=5, num_actions=3, hidden_dim=16, seed=123)
    obs = np.array([1.0, 0.8, 2.0, 0.1, -0.5])

    probs = agent.policy(obs)
    assert probs.shape == (3,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-12)

    val = agent.value(obs)
    assert isinstance(val, float)

    action, log_prob = agent.select_action(obs)
    assert action in (0, 1, 2)
    assert log_prob <= 0.0


def test_ppo_gae_and_train_step() -> None:
    """Verify that Generalized Advantage Estimation and training steps run cleanly."""
    agent = PPOAgent(obs_dim=5, num_actions=3, hidden_dim=16, seed=456)
    transitions: list[Transition] = []

    for i in range(5):
        obs = np.random.normal(0.0, 1.0, 5)
        transitions.append(
            Transition(
                obs=obs,
                action=i % 3,
                reward=float(i * 0.1),
                val=0.5,
                log_prob=-1.09,
                done=i == 4,
            )
        )

    advs, returns = agent.compute_gae(transitions, last_val=0.0)
    assert advs.shape == (5,)
    assert returns.shape == (5,)

    loss = agent.train_step(transitions, last_val=0.0)
    assert isinstance(loss, float)


def test_ppo_training_on_execution_env() -> None:
    """Verify that PPO agent trains on ExecutionEnv without errors."""
    agent = PPOAgent(obs_dim=5, num_actions=3, hidden_dim=16, seed=789)
    cfg = EnvConfig(parent_qty=200, num_steps=4, slice_size=50, seed=42)

    losses = agent.train_on_seeds([42, 43], num_episodes_per_seed=1, env_config=cfg)
    assert len(losses) == 2
    for loss in losses:
        assert isinstance(loss, float)
