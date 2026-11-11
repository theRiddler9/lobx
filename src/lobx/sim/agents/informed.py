"""An informed trading agent possessing private directional signals.

Informed traders introduce adverse selection into the market book (Glosten-Milgrom 1985).
When private information indicates an imminent downward price shift, informed traders
aggressively sell into resting bids, inflicting adverse selection on resting liquidity.
"""

from __future__ import annotations

from numpy.random import Generator

from lobx.sim.actions import Action, Submit
from lobx.sim.agents.base import MarketView
from lobx.sim.clock import NS_PER_S
from lobx.types import Side


class InformedTrader:
    """An agent that trades aggressively in the direction of its private information signal."""

    def __init__(
        self,
        agent_id: int,
        interval_ns: int = 2 * NS_PER_S,
        trade_qty: int = 10,
        signal_strength: float = 0.7,
    ) -> None:
        """Initialize the informed trader.

        Args:
            agent_id: Unique agent identifier.
            interval_ns: Average interval between trading decisions.
            trade_qty: Quantity traded per aggressive order.
            signal_strength: Confidence threshold required to act on directional signal.
        """
        self.agent_id = agent_id
        self.interval_ns = interval_ns
        self.trade_qty = trade_qty
        self.signal_strength = signal_strength
        self._last_ref_price: int | None = None

    def first_wake(self, rng: Generator) -> int:
        """Return the absolute timestamp (ns) for the first turn."""
        return int(rng.exponential(self.interval_ns))

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Examine market information and submit aggressive orders if private signal triggers."""
        del ts
        ref = view.reference_price()
        if self._last_ref_price is None:
            self._last_ref_price = ref
            return []

        # Directional drift signal based on reference price change and order book imbalance
        drift = ref - self._last_ref_price
        self._last_ref_price = ref

        snap = view.snapshot(3)
        bid_vol = sum(q for _, q in snap.bids)
        ask_vol = sum(q for _, q in snap.asks)
        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        # Composite informed signal combining drift and shallow book pressure
        # Informed traders anticipate price drops when negative drift and sell pressure align
        draw = rng.uniform(0.0, 1.0)
        if (drift < 0 or imbalance < -0.2) and draw < self.signal_strength:
            # Informed selling: hit resting bids with aggressive market sell
            return [Submit.market(Side.SELL, self.trade_qty)]
        elif (drift > 0 or imbalance > 0.2) and draw < self.signal_strength:
            # Informed buying: lift resting asks with aggressive market buy
            return [Submit.market(Side.BUY, self.trade_qty)]

        return []

    def next_wake(self, ts: int, rng: Generator) -> int:
        """Return the absolute timestamp of the next turn."""
        return ts + int(rng.exponential(self.interval_ns))
