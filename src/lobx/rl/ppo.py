"""Proximal Policy Optimization (PPO) Actor-Critic Agent for optimal trade execution.

Implements clipped surrogate policy objective, Generalized Advantage Estimation (GAE),
and discrete action selection (Passive, Aggressive, Hold) over the ExecutionEnv MDP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lobx.rl.environment import EnvConfig, ExecutionEnv


@dataclass(slots=True)
class Transition:
    """A single environment step transition."""

    obs: np.ndarray
    action: int
    reward: float
    val: float
    log_prob: float
    done: bool


class PPOAgent:
    """Actor-Critic agent trained with Proximal Policy Optimization."""

    def __init__(
        self,
        obs_dim: int = 5,
        num_actions: int = 3,
        hidden_dim: int = 32,
        lr_actor: float = 0.01,
        lr_critic: float = 0.02,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        seed: int = 42,
    ) -> None:
        """Initialize PPO policy and value networks."""
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rng = np.random.default_rng(seed)

        # Policy (Actor) network: Linear(obs_dim, hidden) -> ReLU -> Linear(hidden, num_actions)
        self.W_actor1 = self.rng.normal(0.0, np.sqrt(2.0 / obs_dim), (obs_dim, hidden_dim))
        self.b_actor1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W_actor2 = self.rng.normal(
            0.0, np.sqrt(2.0 / (hidden_dim + num_actions)), (hidden_dim, num_actions)
        )
        self.b_actor2 = np.zeros(num_actions, dtype=np.float64)

        # Value (Critic) network: Linear(obs_dim, hidden) -> ReLU -> Linear(hidden, 1)
        self.W_critic1 = self.rng.normal(0.0, np.sqrt(2.0 / obs_dim), (obs_dim, hidden_dim))
        self.b_critic1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W_critic2 = self.rng.normal(0.0, np.sqrt(2.0 / (hidden_dim + 1)), (hidden_dim, 1))
        self.b_critic2 = np.zeros(1, dtype=np.float64)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        """Return action probabilities given observation."""
        obs_vec = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        h = np.maximum(0.0, np.dot(obs_vec, self.W_actor1) + self.b_actor1)
        logits = np.dot(h, self.W_actor2) + self.b_actor2
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shift)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.asarray(probs[0], dtype=np.float64)

    def value(self, obs: np.ndarray) -> float:
        """Estimate state value V(s)."""
        obs_vec = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        h = np.maximum(0.0, np.dot(obs_vec, self.W_critic1) + self.b_critic1)
        v = np.dot(h, self.W_critic2) + self.b_critic2
        return float(v[0, 0])

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, float]:
        """Sample an action and return action index and its log probability."""
        probs = self.policy(obs)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(self.rng.choice(self.num_actions, p=probs))
        log_prob = float(np.log(np.clip(probs[action], 1e-12, 1.0)))
        return action, log_prob

    def compute_gae(
        self,
        transitions: list[Transition],
        last_val: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE) and discounted targets."""
        num_steps = len(transitions)
        advantages = np.zeros(num_steps, dtype=np.float64)
        last_gae = 0.0

        for t in reversed(range(num_steps)):
            next_val = last_val if t == num_steps - 1 else transitions[t + 1].val
            done = transitions[t].done
            delta = (
                transitions[t].reward
                + self.gamma * next_val * (1.0 - float(done))
                - transitions[t].val
            )
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - float(done)) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array([tr.val for tr in transitions])
        return advantages, returns

    def train_step(self, transitions: list[Transition], last_val: float) -> float:
        """Execute one PPO optimization step over collected transitions."""
        if not transitions:
            return 0.0

        advantages, returns = self.compute_gae(transitions, last_val)
        # Normalize advantages
        adv_std = np.std(advantages)
        if adv_std > 1e-8:
            advantages = (advantages - np.mean(advantages)) / adv_std

        total_loss = 0.0
        for i, tr in enumerate(transitions):
            # Compute current policy probability
            probs = self.policy(tr.obs)
            curr_log_prob = float(np.log(np.clip(probs[tr.action], 1e-12, 1.0)))
            ratio = np.exp(curr_log_prob - tr.log_prob)

            # Clipped surrogate objective
            surr1 = ratio * advantages[i]
            surr2 = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[i]
            policy_loss = -float(np.minimum(surr1, surr2))

            # Critic value loss: MSE between V(s) and target return
            val = self.value(tr.obs)
            val_loss = float(0.5 * (val - returns[i]) ** 2)

            total_loss += policy_loss + val_loss

            # Analytical gradient update for policy
            obs_vec = tr.obs.reshape(1, -1)
            h_actor = np.maximum(0.0, np.dot(obs_vec, self.W_actor1) + self.b_actor1)
            dlogits = probs.copy()
            dlogits[tr.action] -= 1.0
            grad_scale = float(advantages[i])
            self.W_actor2 -= self.lr_actor * np.dot(h_actor.T, dlogits.reshape(1, -1)) * grad_scale
            self.b_actor2 -= self.lr_actor * dlogits * grad_scale

            # Critic update
            h_critic = np.maximum(0.0, np.dot(obs_vec, self.W_critic1) + self.b_critic1)
            val_err = float(val - returns[i])
            self.W_critic2 -= self.lr_critic * h_critic.T * val_err
            self.b_critic2 -= self.lr_critic * val_err

        return total_loss / len(transitions)

    def train_on_seeds(
        self,
        seeds: list[int],
        num_episodes_per_seed: int = 1,
        env_config: EnvConfig | None = None,
    ) -> list[float]:
        """Train PPO policy across simulation seeds."""
        losses: list[float] = []
        cfg = env_config or EnvConfig()

        for seed in seeds:
            for _ in range(num_episodes_per_seed):
                env = ExecutionEnv(cfg)
                obs, _ = env.reset(seed=seed)
                transitions: list[Transition] = []
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action, log_prob = self.select_action(obs)
                    val = self.value(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    transitions.append(
                        Transition(
                            obs=obs,
                            action=action,
                            reward=reward,
                            val=val,
                            log_prob=log_prob,
                            done=done,
                        )
                    )
                    obs = next_obs

                last_val = self.value(obs) if not (terminated or truncated) else 0.0
                step_loss = self.train_step(transitions, last_val)
                losses.append(step_loss)

        return losses
