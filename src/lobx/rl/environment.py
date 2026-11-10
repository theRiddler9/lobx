"""Gymnasium-compatible trade execution environment driven by the C++ engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from lobx._native import NativeMarket, NativeSimConfig
from lobx.types import Side


@dataclass(slots=True)
class EnvConfig:
    """Configuration for an execution environment episode."""

    parent_qty: int = 1000
    num_steps: int = 20
    slice_size: int = 100
    seed: int = 42
    side: Side = Side.BUY
    step_duration_ns: int = 500_000_000  # 500 ms per step


class ExecutionEnv:
    """Gymnasium-compatible optimal execution environment."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        self.market: NativeMarket | None = None
        self.arrival_mid: float = 10_000.0
        self.rem_qty: int = self.cfg.parent_qty
        self.executed_qty: int = 0
        self.cash_flow: int = 0
        self.current_step: int = 0
        self._last_mid: float = self.arrival_mid

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        used_seed = seed if seed is not None else self.cfg.seed
        sim_cfg = NativeSimConfig()
        sim_cfg.seed = used_seed
        sim_cfg.duration_ns = (self.cfg.num_steps + 5) * self.cfg.step_duration_ns

        self.market = NativeMarket(sim_cfg)
        for _ in range(15):  # warm up market depth
            self.market.step()

        bb = self.market.best_bid() or 9999
        ba = self.market.best_ask() or 10001
        self.arrival_mid = (bb + ba) / 2.0
        self._last_mid = self.arrival_mid
        self.rem_qty = self.cfg.parent_qty
        self.executed_qty = 0
        self.cash_flow = 0
        self.current_step = 0

        return self._get_obs(), {"arrival_mid": self.arrival_mid}

    def _get_obs(self) -> np.ndarray:
        bb = (self.market.best_bid() if self.market else None) or int(self.arrival_mid - 1)
        ba = (self.market.best_ask() if self.market else None) or int(self.arrival_mid + 1)
        mid = (bb + ba) / 2.0
        spread = float(ba - bb)
        imbalance = 0.0  # top level imbalance
        momentum = float(mid - self._last_mid)
        self._last_mid = mid

        rem_qty_ratio = float(self.rem_qty) / max(1, self.cfg.parent_qty)
        rem_time_ratio = float(self.cfg.num_steps - self.current_step) / max(1, self.cfg.num_steps)

        return np.array([rem_qty_ratio, rem_time_ratio, spread, imbalance, momentum], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.market is not None, "Call reset() before step()"
        self.current_step += 1
        qty_to_trade = min(self.rem_qty, self.cfg.slice_size)
        step_exec_qty = 0
        step_cash = 0

        if action == 1 and qty_to_trade > 0:  # Aggressive: immediate fill at ask/bid
            ba = self.market.best_ask() or int(self.arrival_mid + 1)
            step_exec_qty = qty_to_trade
            step_cash = step_exec_qty * ba
        elif action == 0 and qty_to_trade > 0:  # Passive: fill if market trades through
            bb = self.market.best_bid() or int(self.arrival_mid - 1)
            step_exec_qty = qty_to_trade
            step_cash = step_exec_qty * bb

        self.executed_qty += step_exec_qty
        self.cash_flow += step_cash
        self.rem_qty -= step_exec_qty

        # Advance background market simulation
        target_ts = self.current_step * self.cfg.step_duration_ns
        while self.market.current_time() < target_ts and self.market.step():
            pass

        terminated = self.rem_qty == 0
        truncated = self.current_step >= self.cfg.num_steps and not terminated

        # Handle deadline enforcement if truncated
        if truncated and self.rem_qty > 0:
            penalty_px = (self.market.best_ask() or int(self.arrival_mid + 5)) + 10
            self.cash_flow += self.rem_qty * penalty_px
            self.executed_qty += self.rem_qty
            self.rem_qty = 0

        # Implementation shortfall in bps: (exec_price - arrival_mid) / arrival_mid * 10,000
        avg_px = (self.cash_flow / max(1, self.executed_qty)) if self.executed_qty > 0 else self.arrival_mid
        shortfall_bps = ((avg_px - self.arrival_mid) / self.arrival_mid) * 10_000.0
        reward = -shortfall_bps * 0.01  # scale reward

        info = {
            "executed_qty": self.executed_qty,
            "rem_qty": self.rem_qty,
            "shortfall_bps": shortfall_bps,
            "avg_price": avg_px,
        }

        return self._get_obs(), reward, terminated, truncated, info
