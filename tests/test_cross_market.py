"""Tests for CrossMarket multi-asset simulation environment."""

from lobx.sim.actions import Submit
from lobx.sim.cross_market import (
    CrossMarket,
    CrossMarketConfig,
    CrossMarketView,
)
from lobx.types import Side


class DummyTrader:
    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id

    def wake(self, ts: int, view: CrossMarketView) -> list:
        # Submit buy on LEAD and sell on LAG
        return [
            Submit.limit(Side.BUY, price=9990, qty=10, symbol="LEAD"),
            Submit.limit(Side.SELL, price=10010, qty=10, symbol="LAG"),
        ]


def test_cross_market_initialization_and_liquidity_seeding() -> None:
    cfg = CrossMarketConfig(symbols=("LEAD", "LAG"), duration_ns=1_000_000_000)
    market = CrossMarket(cfg)
    assert market.symbols == ("LEAD", "LAG")

    market.seed_initial_liquidity(depth_per_side=3, qty_per_level=50)

    assert market.best_bid("LEAD") == 9998
    assert market.best_ask("LEAD") == 10002
    assert market.spread("LEAD") == 4
    assert market.mid_price("LEAD") == 10000.0

    snap_lag = market.snapshot("LAG", depth=3)
    assert len(snap_lag.bids) == 3
    assert len(snap_lag.asks) == 3


def test_cross_market_agent_execution_and_positions() -> None:
    cfg = CrossMarketConfig(symbols=("LEAD", "LAG"), duration_ns=500_000_000)
    market = CrossMarket(cfg)
    market.seed_initial_liquidity(depth_per_side=3, qty_per_level=100)

    trader = DummyTrader(agent_id=1)
    market.add_agent(trader, wake_interval_ns=100_000_000)

    res = market.run()
    assert res.duration_ns >= 500_000_000
    assert 1 in res.agent_positions
    # Resting orders were placed
    assert len(market.resting_orders(1, "LEAD")) > 0
    assert len(market.resting_orders(1, "LAG")) > 0
