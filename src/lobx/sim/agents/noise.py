"""The noise trader: random, uninformed order flow.

Noise traders are the background of the market. They have no view about where the price is going;
they place orders at random times (a Poisson process), on a random side, mostly passive limit
orders a few ticks away from the current price and sometimes market orders that take liquidity.
Their randomness is what makes the order book move at all, and the heavy-tailed size of their
market orders is what gives the price occasional large jumps, as real markets have.
"""

from __future__ import annotations

from numpy.random import Generator

from lobx.sim.actions import Action, Cancel, Submit
from lobx.sim.agents.base import MarketView
from lobx.sim.agents.timing import exponential_gap_ns
from lobx.types import Side


class NoiseTrader:
    """An agent that trades at random, with no opinion about the price."""

    def __init__(
        self,
        agent_id: int,
        mean_interval_ns: int,
        p_market: float = 0.15,
        mean_offset_ticks: float = 3.0,
        max_limit_qty: int = 10,
        max_market_qty: int = 100,
        pareto_alpha: float = 1.5,
        market_qty_scale: float = 2.0,
        max_open_orders: int = 6,
    ) -> None:
        """Create a noise trader.

        ``mean_interval_ns`` is the average time between wakes. ``p_market`` is the chance that an
        order is a market order. A limit order is placed ``mean_offset_ticks`` away from the
        reference price on average. Market order sizes follow a Pareto (heavy-tailed) law, scaled
        by ``market_qty_scale`` and capped at ``max_market_qty``. The trader keeps at most
        ``max_open_orders`` resting orders, retiring the oldest first.
        """
        if mean_interval_ns < 1 or max_limit_qty < 1 or max_market_qty < 1 or max_open_orders < 1:
            raise ValueError("interval, quantities and max_open_orders must all be at least 1")
        if not 0.0 <= p_market <= 1.0:
            raise ValueError(f"p_market must be between 0 and 1, got {p_market}")
        if mean_offset_ticks < 1.0 or pareto_alpha <= 0.0 or market_qty_scale <= 0.0:
            raise ValueError("mean_offset_ticks must be >= 1, pareto_alpha and scale must be > 0")
        self.agent_id = agent_id
        self.mean_interval_ns = mean_interval_ns
        self.p_market = p_market
        self.mean_offset_ticks = mean_offset_ticks
        self.max_limit_qty = max_limit_qty
        self.max_market_qty = max_market_qty
        self.pareto_alpha = pareto_alpha
        self.market_qty_scale = market_qty_scale
        self.max_open_orders = max_open_orders

    def first_wake(self, rng: Generator) -> int:
        """Return the time of the first turn: one random gap after time 0."""
        return exponential_gap_ns(rng, self.mean_interval_ns)

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        """Return the next turn: one random gap after ``ts``. Noise traders never retire."""
        return ts + exponential_gap_ns(rng, self.mean_interval_ns)

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Place one random order, first cancelling the oldest if too many are resting."""
        actions: list[Action] = []
        open_ids = view.open_orders(self.agent_id)
        if len(open_ids) >= self.max_open_orders:
            # Without a cap, stale orders would pile up and the book would grow deeper and deeper,
            # so the market would behave differently at the end of a run than at the start.
            actions.append(Cancel(open_ids[0]))
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        if rng.random() < self.p_market:
            actions.append(Submit.market(side, self._market_qty(rng)))
        else:
            reference = view.reference_price()
            # geometric(p) is 1, 2, 3, ... with mean 1/p: at least one tick away from the reference.
            offset = int(rng.geometric(1.0 / self.mean_offset_ticks))
            price = reference - offset if side is Side.BUY else reference + offset
            qty = int(rng.integers(1, self.max_limit_qty + 1))
            actions.append(Submit.limit(side, max(1, price), qty))
        return actions

    def _market_qty(self, rng: Generator) -> int:
        """Draw a heavy-tailed size: mostly small, occasionally very large, capped."""
        size = 1 + int(rng.pareto(self.pareto_alpha) * self.market_qty_scale)
        return min(size, self.max_market_qty)
