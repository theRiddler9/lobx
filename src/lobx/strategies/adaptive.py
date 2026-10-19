"""An execution strategy that watches the order book and uses an ML model to adapt."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from lobx.ml.features import extract_features
from lobx.ml.labels import DOWN, UP
from lobx.ml.model import Classifier
from lobx.sim.actions import Action, Cancel, Submit
from lobx.sim.agents.base import MarketView
from lobx.sim.clock import NS_PER_S
from lobx.strategies.base import ParentOrder
from lobx.strategies.twap import TwapStrategy
from lobx.types import BookSnapshot, Side

HISTORY = 26  # enough snapshots for the 25-step return feature
SNAPSHOT_DEPTH = 5
FLOW_LOOKBACK_NS = 5 * NS_PER_S  # the longest trade-flow feature window
DEFAULT_DEADLINE_MARGIN_NS = 5 * NS_PER_S


@dataclass(slots=True)
class _RestingOrder:
    price: int
    posted: int
    executed_at_post: int


class AdaptiveStrategy:
    """An execution strategy that uses an order-flow model to decide when to trade."""

    def __init__(
        self,
        model: Classifier | None,
        num_slices: int,
        threshold: float,
        deadline_margin_ns: int = DEFAULT_DEADLINE_MARGIN_NS,
    ) -> None:
        """Create the strategy."""
        self._model = model
        self._num_slices = num_slices
        self._deadline_margin_ns = deadline_margin_ns
        self._pace = TwapStrategy(num_slices)
        self._threshold = threshold
        self._parent: ParentOrder | None = None
        self._history: deque[BookSnapshot] = deque(maxlen=HISTORY)
        self._resting: _RestingOrder | None = None

    def reset(self, parent: ParentOrder) -> None:
        """Prepare to execute parent; forget any earlier observations."""
        self._parent = parent
        self._pace.reset(parent)
        self._history.clear()
        self._resting = None

    def observe(self, ts: int, view: MarketView) -> None:
        """Remember the book as of ts. Must be called once per snapshot interval, in order."""
        self._history.append(replace(view.snapshot(SNAPSHOT_DEPTH), ts=ts))

    def probabilities(self, ts: int, view: MarketView) -> tuple[float, float, float] | None:
        """Return the model's (down, flat, up) probabilities now, or None if it cannot tell."""
        if self._model is None or len(self._history) < HISTORY:
            return None
        tape = view.fills_after(ts - FLOW_LOOKBACK_NS)
        row = extract_features(list(self._history), tape)[-1]
        if np.isnan(row).any():
            return None
        down, flat, up = self._model.predict_proba(row.reshape(1, -1))[0]
        return float(down), float(flat), float(up)

    def step(self, ts: int, view: MarketView) -> list[Action]:
        """Observe the market and return the child orders for time ts."""
        if self._parent is None:
            raise RuntimeError("call reset(parent) before step()")
        if self._model is not None:
            self.observe(ts, view)
        return []

    def _touch_price(self, view: MarketView) -> int:
        """Return the best price on our own side: the ask for a seller, the bid for a buyer."""
        assert self._parent is not None
        touch = view.best_ask() if self._parent.side is Side.SELL else view.best_bid()
        return touch if touch is not None else view.reference_price()

    def _aggressive_actions(self, view: MarketView, quantity: int) -> list[Action]:
        """Cancel our resting order, then trade quantity at once with a market order."""
        assert self._parent is not None
        actions: list[Action] = [Cancel(i) for i in view.open_orders(self._parent.agent_id)]
        self._resting = None
        if quantity > 0:
            actions.append(Submit.market(self._parent.side, quantity))
        return actions

    def _passive_actions(self, view: MarketView, behind: int, executed: int) -> list[Action]:
        """Keep a limit order at the touch covering behind shares, replacing it if needed."""
        assert self._parent is not None
        side = self._parent.side
        open_ids = view.open_orders(self._parent.agent_id)
        if not open_ids:
            self._resting = None
        touch = self._touch_price(view)
        actions: list[Action] = []
        resting = self._resting
        if resting is not None:
            unfilled = resting.posted - (executed - resting.executed_at_post)
            off_touch = resting.price > touch if side is Side.SELL else resting.price < touch
            if unfilled >= behind and not off_touch:
                return []
            actions.extend(Cancel(order_id) for order_id in open_ids)
        self._resting = None
        if behind > 0:
            actions.append(Submit.limit(side, touch, behind))
            self._resting = _RestingOrder(touch, behind, executed)
        return actions
