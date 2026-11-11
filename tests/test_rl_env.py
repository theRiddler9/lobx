"""Tests for the Gymnasium-compatible Reinforcement Learning execution environment."""

from __future__ import annotations

import unittest

import numpy as np

from lobx.rl.environment import EnvConfig, ExecutionEnv


class TestExecutionEnv(unittest.TestCase):
    def test_env_reset(self) -> None:
        cfg = EnvConfig(parent_qty=1000, num_steps=20, seed=42)
        env = ExecutionEnv(cfg)
        obs, info = env.reset(seed=42)

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (5,))  # [qty_ratio, time_ratio, spread, imb, mom]
        self.assertTrue(np.isclose(obs[0], 1.0))
        self.assertTrue(np.isclose(obs[1], 1.0))
        self.assertIn("arrival_mid", info)

    def test_env_steps_and_termination(self) -> None:
        cfg = EnvConfig(parent_qty=500, num_steps=5, slice_size=100, seed=42)
        env = ExecutionEnv(cfg)
        obs, _ = env.reset(seed=42)

        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(1)
            total_reward += reward
            steps += 1

        self.assertEqual(steps, 5)
        self.assertTrue(terminated or truncated)
        self.assertIn("executed_qty", info)
        self.assertIn("shortfall_bps", info)
        self.assertEqual(info["executed_qty"], 500)

    def test_env_passive_and_hold_actions(self) -> None:
        cfg = EnvConfig(parent_qty=500, num_steps=10, slice_size=50, seed=123)
        env = ExecutionEnv(cfg)
        env.reset(seed=123)

        # Action 2: Hold (do nothing)
        obs, reward, terminated, truncated, info = env.step(2)
        self.assertFalse(terminated)
        self.assertEqual(info["executed_qty"], 0)

        # Action 0: Passive order
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)


if __name__ == "__main__":
    unittest.main()
