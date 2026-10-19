"""The adaptive strategy: follow a TWAP pace, but choose *how* to trade each share using a model.

TWAP decides how much should be done by when. The model predicts the next second's mid-price move.
For a seller that gives a simple trade-off:

* if the price is likely to fall, hit the bid now (a *market* order): speed beats price;
* otherwise rest a *limit* order at the best ask and wait: if it fills, the seller earns the spread
  instead of paying it, and waiting costs little because the price is not expected to fall.

The strategy always finishes by the deadline (see ``deadline_margin_ns``).
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

SNAPSHOT_DEPTH = 5  # the depth the model's imbalance features were trained with
# Features need this many past snapshots: the longest return window, plus the current one.
HISTORY = max(RETURN_WINDOWS) + 1
# Trade-flow features look this far back (the longest flow window).
FLOW_LOOKBACK_NS = max(FLOW_WINDOWS_NS)
DEFAULT_DEADLINE_MARGIN_NS = 5_000_000_000  # finish this long before the deadline


@dataclass(slots=True)
class _RestingOrder:
    """What the strategy remembers about the one limit order it has in the book."""

    price: int
    posted: int  # quantity when it was placed
    executed_at_post: int  # shares of the parent already done at that moment


class AdaptiveStrategy:
    """Model-guided execution on top of a TWAP schedule."""

    def __init__(
        self,
        model: Classifier | None,
        num_slices: int,
        threshold: float,
        deadline_margin_ns: int = DEFAULT_DEADLINE_MARGIN_NS,
    ) -> None:
        """Create the strategy.

        With ``model=None`` there are no forecasts and the strategy is purely passive (resting
        orders plus the catch-up and deadline rules): the *ablation* that shows how much of the
        adaptive strategy's result comes from the model and how much from passivity alone.
        ``num_slices`` sets the TWAP pace. ``threshold`` is how much more likely an adverse move
        must be than a favourable one before the strategy trades aggressively.
        ``deadline_margin_ns`` is how long before the deadline everything left is crossed with
        market orders, leaving time to retry if a thin book does not fill them all.
        """
        self._model = model
        self._num_slices = num_slices
        self._deadline_margin_ns = deadline_margin_ns
        self._pace = TwapStrategy(num_slices)
        self._threshold = threshold
        self._parent: ParentOrder | None = None
        self._history: deque[BookSnapshot] = deque(maxlen=HISTORY)
        self._resting: _RestingOrder | None = None

    def reset(self, parent: ParentOrder) -> None:
        """Prepare to execute ``parent``; forget any earlier observations."""
        self._parent = parent
        self._pace.reset(parent)
        self._history.clear()
        self._resting = None

    def observe(self, ts: int, view: MarketView) -> None:
        """Remember the book as of ``ts``. Must be called once per snapshot interval, in order.

        The history is what the model's features are computed from, so it has to be built the same
        way as the training data: evenly spaced snapshots stamped with the wake time.
        """
        self._history.append(replace(view.snapshot(SNAPSHOT_DEPTH), ts=ts))

    def probabilities(self, ts: int, view: MarketView) -> tuple[float, float, float] | None:
        """Return the model's (down, flat, up) probabilities now, or ``None`` if it cannot tell.

        Only data at or before ``ts`` is used: the stored snapshots and the trade tape up to now.
        ``None`` means there is not yet enough history, or the book is one-sided.
        """
        if self._model is None or len(self._history) < HISTORY:
            return None
        tape = view.fills_after(ts - FLOW_LOOKBACK_NS)
        row = extract_features(list(self._history), tape)[-1]
        if np.isnan(row).any():
            return None
        down, flat, up = self._model.predict_proba(row.reshape(1, -1))[0]
        return float(down), float(flat), float(up)

    def step(self, ts: int, view: MarketView) -> list[Action]:
        """Observe the market and return the child orders for time ``ts``."""
        if self._parent is None:
            raise RuntimeError("call reset(parent) before step()")
        if self._model is not None:  # the history only feeds the model
            self.observe(ts, view)
        parent = self._parent
        if ts < parent.start_ns:
            return []  # still warming up the history: nothing to execute yet
        executed = parent.executed(view)
        remaining = parent.qty - executed
        if remaining <= 0:
            return self._aggressive_actions(view, 0)  # done: just take back any resting order
        # Deadline: the parent order must finish. Crossing the spread for everything left costs
        # more than waiting, but an unfinished order has an unbounded cost.
        if ts >= parent.end_ns - self._deadline_margin_ns:
            return self._aggressive_actions(view, remaining)
        behind = max(self._pace.target_at(ts) - executed, 0)
        # Catch-up: passive orders may simply not fill. Anything still owed from a full slice ago
        # is overdue and is traded at once, so passivity can delay a share by at most one slice.
        slice_ns = (parent.end_ns - parent.start_ns) // self._num_slices
        overdue = self._pace.target_at(ts - slice_ns) - executed
        if overdue > 0:
            return self._aggressive_actions(view, behind)
        probabilities = self.probabilities(ts, view)
        # The decision rule. An adverse move that is clearly more likely than a favourable one means
        # waiting is expensive: trade what is owed now. Otherwise (or if the model cannot tell)
        # wait passively; deadlines and catch-up rules stop passivity from running out of time.
        if (
            behind > 0
            and probabilities is not None
            and self.adverse_signal(probabilities) > self._threshold
        ):
            return self._aggressive_actions(view, behind)
        return self._passive_actions(view, behind, executed)

    def _touch_price(self, view: MarketView) -> int:
        """Return the best price on *our own* side: the ask for a seller, the bid for a buyer.

        Resting there joins the queue of orders already at that price without crossing the spread.
        If that side of the book is empty the reference price is used instead.
        """
        assert self._parent is not None
        touch = view.best_ask() if self._parent.side is Side.SELL else view.best_bid()
        return touch if touch is not None else view.reference_price()

    def _aggressive_actions(self, view: MarketView, quantity: int) -> list[Action]:
        """Cancel our resting order, then trade ``quantity`` at once with a market order.

        The cancel comes first so the resting order cannot fill *as well*, which could sell more
        than the parent order. Actions run in order within one instant, so nothing can slip between
        the cancel and the market order.
        """
        assert self._parent is not None
        actions: list[Action] = [Cancel(i) for i in view.open_orders(self._parent.agent_id)]
        self._resting = None
        if quantity > 0:
            actions.append(Submit.market(self._parent.side, quantity))
        return actions

    def _passive_actions(self, view: MarketView, behind: int, executed: int) -> list[Action]:
        """Keep a limit order at the touch covering ``behind`` shares, replacing it if needed.

        A limit order that fills saves the half-spread a market order would pay, but only the order
        at the front of the queue fills first, and a cancel loses the place in the queue. So the
        existing order is kept unless it is too small or no longer at the touch.
        """
        assert self._parent is not None
        side = self._parent.side
        open_ids = view.open_orders(self._parent.agent_id)
        if not open_ids:
            self._resting = None  # it filled completely (or was cancelled)
        touch = self._touch_price(view)
        actions: list[Action] = []
        resting = self._resting
        if resting is not None:
            # Only this order trades passively, so its unfilled part is what was posted minus all
            # the shares done since.
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

    def adverse_signal(self, probabilities: tuple[float, float, float]) -> float:
        """Return P(price moves against us) - P(price moves in our favour).

        A seller is hurt by a fall, a buyer by a rise. Positive means trouble is more likely than
        luck.
        """
        if self._parent is None:
            raise RuntimeError("call reset(parent) before use")
        against, favour = (
            (probabilities[DOWN], probabilities[UP])
            if self._parent.side is Side.SELL
            else (probabilities[UP], probabilities[DOWN])
        )
        return against - favour
