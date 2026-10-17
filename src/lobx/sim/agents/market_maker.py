"""The market maker: quotes both sides of the book and manages its inventory.

A market maker earns the spread by always offering to buy a little below the price and to sell a
little above it. Its risk is inventory: if everybody sells to it, it ends up long a falling asset.
The standard remedy (the idea behind Avellaneda and Stoikov's model) is to *skew* the quotes by the
position: when long, shift both quotes down, so the ask is more likely to be hit and the bid less
likely, and the inventory drifts back towards zero. It also stops quoting the side that would push
it past a hard position limit.

This is a simplified version of that idea, meant to give the simulated market realistic depth and a
positive spread. It is not a trading strategy.
"""

from __future__ import annotations

from numpy.random import Generator

from lobx.sim.actions import Action, Cancel, Submit
from lobx.sim.agents.base import MarketView
from lobx.sim.agents.timing import jittered_gap_ns
from lobx.types import Side


class MarketMaker:
    """An agent that keeps several price levels of quotes on both sides, skewed by inventory."""

    def __init__(
        self,
        agent_id: int,
        interval_ns: int,
        half_spread_ticks: int = 2,
        levels: int = 3,
        level_gap_ticks: int = 2,
        quote_qty: int = 10,
        skew_ticks_per_share: float = 0.02,
        max_position: int = 500,
    ) -> None:
        """Create a market maker.

        Every ``interval_ns`` (give or take some jitter) it pulls its old quotes and posts new
        ones: ``levels`` price levels per side, the first ``half_spread_ticks`` from its
        reservation price and each further level ``level_gap_ticks`` beyond the last, each for
        ``quote_qty`` shares. The reservation price is the reference price shifted down by
        ``skew_ticks_per_share`` for every share of long inventory (up for short).
        """
        if interval_ns < 1 or half_spread_ticks < 1 or levels < 1 or level_gap_ticks < 1:
            raise ValueError("interval, half spread, levels and level gap must all be at least 1")
        if quote_qty < 1 or max_position < 1 or skew_ticks_per_share < 0.0:
            raise ValueError("quote_qty and max_position must be >= 1 and the skew >= 0")
        self.agent_id = agent_id
        self.interval_ns = interval_ns
        self.half_spread_ticks = half_spread_ticks
        self.levels = levels
        self.level_gap_ticks = level_gap_ticks
        self.quote_qty = quote_qty
        self.skew_ticks_per_share = skew_ticks_per_share
        self.max_position = max_position

    def first_wake(self, rng: Generator) -> int:
        """Wake at time 0, before anyone else, so the book has quotes when trading starts."""
        return 0

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        """Re-quote one (jittered) interval after ``ts``. Market makers never retire."""
        return ts + jittered_gap_ns(rng, self.interval_ns)

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Cancel every resting quote, then post fresh ones around the reservation price."""
        # Cancels first: the old quotes may be stale, and stale quotes are free money for others.
        actions: list[Action] = [Cancel(order_id) for order_id in view.open_orders(self.agent_id)]
        position = view.position(self.agent_id)
        reservation = view.reference_price() - self.skew_ticks_per_share * position
        for level in range(self.levels):
            offset = self.half_spread_ticks + level * self.level_gap_ticks
            bid = max(1, round(reservation - offset))
            ask = max(bid + 1, round(reservation + offset))
            if position < self.max_position:  # at the limit, stop adding to a long position
                actions.append(Submit.limit(Side.BUY, bid, self.quote_qty))
            if position > -self.max_position:  # and likewise for a short one
                actions.append(Submit.limit(Side.SELL, ask, self.quote_qty))
        return actions
