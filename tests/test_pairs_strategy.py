"""Tests for PairsExecutionStrategy and dynamic leg risk management."""

from lobx.sim.cross_market import CrossMarket, CrossMarketConfig
from lobx.strategies.pairs import PairsExecutionStrategy, PairsParentOrder
from lobx.types import Side


def test_pairs_execution_strategy_lifecycle() -> None:
    cfg = CrossMarketConfig(symbols=("SPY", "QQQ"), duration_ns=1_000_000_000)
    market = CrossMarket(cfg)
    market.seed_initial_liquidity(depth_per_side=5, qty_per_level=500)

    parent = PairsParentOrder(
        agent_id=7,
        symbol_a="SPY",
        side_a=Side.BUY,
        qty_a=100,
        symbol_b="QQQ",
        side_b=Side.SELL,
        qty_b=100,
        start_ns=0,
        end_ns=500_000_000,
        max_leg_imbalance=30,
    )

    strat = PairsExecutionStrategy(parent, num_slices=2)
    market.add_agent(strat, wake_interval_ns=250_000_000)

    _ = market.run()

    metrics = strat.compute_metrics(market)
    assert metrics.executed_a == 100
    assert metrics.executed_b == 100
    assert metrics.max_unhedged_qty <= 50
    assert metrics.arrival_mid_a == 10000.0
    assert metrics.arrival_mid_b == 10000.0
    assert isinstance(metrics.combined_shortfall_bps, float)
