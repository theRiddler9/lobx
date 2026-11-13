"""Lead-Lag microstructure agents for cross-asset statistical arbitrage.

Implements latency-sensitive arbitrageurs and correlated market makers modeling
cross-asset order flow imbalance transmission and quote staleness.
"""

from __future__ import annotations

from lobx.sim.actions import Action, Cancel, Submit
from lobx.sim.cross_market import CrossMarketView
from lobx.types import Side


class LeadLagArbitrageAgent:
    """Exploits short-horizon lead-lag cross-correlation between two assets.

    Monitors order book depth imbalance on a liquid leader asset (e.g. LEAD)
    to predict imminent price discovery drift in a correlated follower asset (e.g. LAG).
    """

    def __init__(
        self,
        agent_id: int,
        lead_symbol: str = "LEAD",
        lag_symbol: str = "LAG",
        imbalance_threshold: float = 0.3,
        trade_qty: int = 10,
        max_inventory: int = 100,
    ) -> None:
        """Initialize LeadLagArbitrageAgent with symbols and trading parameters."""
        self.agent_id = agent_id
        self.lead_symbol = lead_symbol
        self.lag_symbol = lag_symbol
        self.imbalance_threshold = imbalance_threshold
        self.trade_qty = trade_qty
        self.max_inventory = max_inventory
        self.signals_generated = 0
        self.trades_executed = 0

    def wake(self, ts: int, view: CrossMarketView) -> list[Action]:
        """Detect cross-book imbalance and emit directional orders."""
        imb_lead = view.imbalance(self.lead_symbol, depth=1)
        pos_lag = view.position(self.agent_id, self.lag_symbol)
        actions: list[Action] = []

        # Cancels any old resting orders
        resting = view.resting_orders(self.agent_id, self.lag_symbol)
        for oid in resting:
            actions.append(Cancel(order_id=oid, symbol=self.lag_symbol))

        # Strong buy imbalance on leader -> Expect follower to move up -> Buy follower
        if imb_lead >= self.imbalance_threshold and pos_lag < self.max_inventory:
            self.signals_generated += 1
            actions.append(
                Submit.market(
                    side=Side.BUY,
                    qty=self.trade_qty,
                    symbol=self.lag_symbol,
                )
            )
            self.trades_executed += 1

        # Strong sell imbalance on leader -> Expect follower to move down -> Sell follower
        elif imb_lead <= -self.imbalance_threshold and pos_lag > -self.max_inventory:
            self.signals_generated += 1
            actions.append(
                Submit.market(
                    side=Side.SELL,
                    qty=self.trade_qty,
                    symbol=self.lag_symbol,
                )
            )
            self.trades_executed += 1

        # Inventory unwinding / rebalancing when leader is neutral
        elif abs(imb_lead) < 0.1:
            mid_lag = view.mid_price(self.lag_symbol)
            if mid_lag is not None:
                if pos_lag > 0:
                    # Sell inventory passively at or above mid
                    price = int(mid_lag + 1)
                    qty = min(pos_lag, self.trade_qty)
                    actions.append(
                        Submit.limit(
                            side=Side.SELL,
                            price=price,
                            qty=qty,
                            symbol=self.lag_symbol,
                        )
                    )
                elif pos_lag < 0:
                    # Cover short inventory passively at or below mid
                    price = int(mid_lag - 1)
                    qty = min(-pos_lag, self.trade_qty)
                    actions.append(
                        Submit.limit(
                            side=Side.BUY,
                            price=price,
                            qty=qty,
                            symbol=self.lag_symbol,
                        )
                    )

        return actions


class CorrelatedMarketMaker:
    """Quotes the lagging asset with quotes shaded towards the leading asset's mid-price."""

    def __init__(
        self,
        agent_id: int,
        symbol: str = "LAG",
        anchor_symbol: str = "LEAD",
        half_spread: int = 2,
        quote_qty: int = 50,
    ) -> None:
        """Initialize CorrelatedMarketMaker."""
        self.agent_id = agent_id
        self.symbol = symbol
        self.anchor_symbol = anchor_symbol
        self.half_spread = half_spread
        self.quote_qty = quote_qty

    def wake(self, ts: int, view: CrossMarketView) -> list[Action]:
        """Cancel existing quotes and post fresh quotes reflecting anchor mid-price."""
        actions: list[Action] = []

        # Cancel current resting quotes
        for oid in view.resting_orders(self.agent_id, self.symbol):
            actions.append(Cancel(order_id=oid, symbol=self.symbol))

        # Anchor price is guided by the leader asset's mid-price
        anchor_mid = view.mid_price(self.anchor_symbol)
        current_mid = view.mid_price(self.symbol)
        target_mid = anchor_mid if anchor_mid is not None else current_mid

        if target_mid is not None:
            bid_p = int(target_mid - self.half_spread)
            ask_p = int(target_mid + self.half_spread)
            if ask_p > bid_p:
                actions.append(
                    Submit.limit(
                        side=Side.BUY,
                        price=bid_p,
                        qty=self.quote_qty,
                        symbol=self.symbol,
                    )
                )
                actions.append(
                    Submit.limit(
                        side=Side.SELL,
                        price=ask_p,
                        qty=self.quote_qty,
                        symbol=self.symbol,
                    )
                )

        return actions
