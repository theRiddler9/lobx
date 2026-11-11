"""Model-driven passive execution strategy featuring quote shading and dynamic sizing.

Protects liquidity provision against adverse selection from informed traders (Glosten-Milgrom 1985).
When the model predicts imminent adverse price moves, the strategy shades its quotes back
from the touch (or pauses posting) to avoid toxic fills. When conditions are safe, it sizes up
to capture the spread.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from lobx.ml.features import FLOW_WINDOWS_NS, RETURN_WINDOWS, extract_features
from lobx.ml.labels import DOWN, UP
from lobx.ml.model import Classifier
from lobx.sim.actions import Action, Cancel, Submit
from lobx.sim.agents.base import MarketView
from lobx.strategies.base import ParentOrder
from lobx.strategies.twap import TwapStrategy
from lobx.types import BookSnapshot, Side

SNAPSHOT_DEPTH = 5
HISTORY = max(RETURN_WINDOWS) + 1
FLOW_LOOKBACK_NS = max(FLOW_WINDOWS_NS)
DEFAULT_DEADLINE_MARGIN_NS = 5_000_000_000


@dataclass(slots=True)
class _RestingLimitOrder:
    """State of an active resting limit order."""

    price: int
    posted: int
    executed_at_post: int


class ModelPassiveStrategy:
    """Adaptive execution with quote shading and sizing against adverse selection."""

    def __init__(
        self,
        model: Classifier | None,
        num_slices: int = 10,
        shade_threshold: float = 0.1,
        shade_ticks: int = 1,
        size_boost_factor: float = 0.5,
        deadline_margin_ns: int = DEFAULT_DEADLINE_MARGIN_NS,
    ) -> None:
        """Initialize the model-driven passive execution strategy.

        Args:
            model: Fitted Classifier predicting (down, flat, up) probabilities. If None,
                runs in unguided passive mode without quote shading.
            num_slices: TWAP schedule divisions.
            shade_threshold: Adverse probability differential threshold triggering quote shading.
            shade_ticks: Number of ticks to retreat from the touch when toxic flow is predicted.
            size_boost_factor: Fractional increase in passive order size during favorable regimes.
            deadline_margin_ns: Time buffer before deadline when remaining shares are swept.
        """
        self.model = model
        self.num_slices = num_slices
        self.shade_threshold = shade_threshold
        self.shade_ticks = shade_ticks
        self.size_boost_factor = size_boost_factor
        self.deadline_margin_ns = deadline_margin_ns

        self._pace = TwapStrategy(num_slices)
        self._parent: ParentOrder | None = None
        self._history: deque[BookSnapshot] = deque(maxlen=HISTORY)
        self._resting: _RestingLimitOrder | None = None

    def reset(self, parent: ParentOrder) -> None:
        """Reset state for a new parent order."""
        self._parent = parent
        self._pace.reset(parent)
        self._history.clear()
        self._resting = None

    def observe(self, ts: int, view: MarketView) -> None:
        """Record sequential snapshot for causal feature extraction."""
        self._history.append(replace(view.snapshot(SNAPSHOT_DEPTH), ts=ts))

    def probabilities(self, ts: int, view: MarketView) -> tuple[float, float, float] | None:
        """Extract features strictly with data <= ts and query model probabilities."""
        if self.model is None or len(self._history) < HISTORY:
            return None
        tape = view.fills_after(ts - FLOW_LOOKBACK_NS)
        row = extract_features(list(self._history), tape)[-1]
        if np.isnan(row).any():
            return None
        down, flat, up = self.model.predict_proba(row.reshape(1, -1))[0]
        return float(down), float(flat), float(up)

    def adverse_signal(self, probs: tuple[float, float, float]) -> float:
        """Compute P(adverse move) - P(favorable move)."""
        assert self._parent is not None
        if self._parent.side is Side.SELL:
            # Sellers are hurt by price drops (down)
            return probs[DOWN] - probs[UP]
        else:
            # Buyers are hurt by price increases (up)
            return probs[UP] - probs[DOWN]

    def step(self, ts: int, view: MarketView) -> list[Action]:
        """Determine execution actions at timestamp ts."""
        if self._parent is None:
            raise RuntimeError("call reset(parent) before step()")
        if self.model is not None:
            self.observe(ts, view)

        parent = self._parent
        if ts < parent.start_ns:
            return []

        executed = parent.executed(view)
        remaining = parent.qty - executed
        if remaining <= 0:
            return self._cancel_all(view)

        # 1. Deadline sweep
        if ts >= parent.end_ns - self.deadline_margin_ns:
            return self._aggressive_sweep(view, remaining)

        # 2. Pacing and catch-up check
        behind = max(self._pace.target_at(ts) - executed, 0)
        slice_ns = (parent.end_ns - parent.start_ns) // self.num_slices
        overdue = self._pace.target_at(ts - slice_ns) - executed
        if overdue > 0:
            # Sweeping overdue backlog to prevent lagging behind schedule
            return self._aggressive_sweep(view, min(behind, remaining))

        # 3. Model-driven passive quote shading & sizing
        probs = self.probabilities(ts, view)
        adv_signal = self.adverse_signal(probs) if probs is not None else 0.0

        # Quote shading: if adverse flow is toxic, retreat price or avoid posting
        shade_offset = 0
        target_qty = behind

        if adv_signal > self.shade_threshold:
            # Toxic regime: shade quote away from touch to prevent adverse fills
            shade_offset = self.shade_ticks
        elif adv_signal < -self.shade_threshold:
            # Favorable / safe regime: size up to earn spread on more volume
            target_qty = min(remaining, int(behind * (1.0 + self.size_boost_factor)))

        return self._passive_quote(view, target_qty, executed, shade_offset)

    def _cancel_all(self, view: MarketView) -> list[Action]:
        assert self._parent is not None
        actions: list[Action] = [Cancel(i) for i in view.open_orders(self._parent.agent_id)]
        self._resting = None
        return actions

    def _aggressive_sweep(self, view: MarketView, qty: int) -> list[Action]:
        assert self._parent is not None
        actions = self._cancel_all(view)
        if qty > 0:
            actions.append(Submit.market(self._parent.side, qty))
        return actions

    def _passive_quote(
        self,
        view: MarketView,
        target_qty: int,
        executed: int,
        shade_offset: int,
    ) -> list[Action]:
        assert self._parent is not None
        side = self._parent.side
        open_ids = view.open_orders(self._parent.agent_id)
        if not open_ids:
            self._resting = None

        # Determine touch price
        touch = view.best_ask() if side is Side.SELL else view.best_bid()
        if touch is None:
            touch = view.reference_price()

        # Quote shading price calculation
        # If selling: retreat higher (+ shade_offset ticks)
        # If buying: retreat lower (- shade_offset ticks)
        quote_px = touch + shade_offset if side is Side.SELL else touch - shade_offset

        actions: list[Action] = []
        resting = self._resting
        if resting is not None:
            unfilled = resting.posted - (executed - resting.executed_at_post)
            # If price matches and remaining quantity is sufficient, hold queue priority
            if resting.price == quote_px and unfilled >= target_qty:
                return []
            actions.extend(Cancel(i) for i in open_ids)

        self._resting = None
        if target_qty > 0:
            actions.append(Submit.limit(side, quote_px, target_qty))
            self._resting = _RestingLimitOrder(quote_px, target_qty, executed)

        return actions
