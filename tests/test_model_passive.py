"""Tests for ModelPassiveStrategy: quote shading, dynamic sizing, and deadline compliance."""

from __future__ import annotations

from lobx.engine.book_py import PythonOrderBook
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import MarketConfig, build_background_agents
from lobx.strategies.agent import StrategyAgent
from lobx.strategies.base import ParentOrder
from lobx.strategies.model_passive import ModelPassiveStrategy
from lobx.types import Side


def test_model_passive_strategy_completes_parent_order() -> None:
    """Verify that ModelPassiveStrategy completes 100% of parent order by deadline."""
    market_cfg = MarketConfig(seed=42, duration_ns=60 * NS_PER_S, num_informed_traders=2)
    bg_agents = build_background_agents(market_cfg)

    parent = ParentOrder(
        agent_id=len(bg_agents),
        side=Side.SELL,
        qty=500,
        start_ns=10 * NS_PER_S,
        end_ns=50 * NS_PER_S,
    )

    strat = ModelPassiveStrategy(
        model=None,
        num_slices=5,
        shade_threshold=0.1,
        shade_ticks=1,
    )
    exec_agent = StrategyAgent(
        strategy=strat,
        parent=parent,
        num_slices=5,
        wake_interval_ns=1 * NS_PER_S,
    )

    book = PythonOrderBook()
    res = Market(market_cfg, [*bg_agents, exec_agent], book=book).run()

    # Verify that the execution agent finished completely
    executed = res.peak_abs_position[parent.agent_id]
    assert executed == parent.qty, f"Expected {parent.qty} executed, got {executed}"


def test_model_passive_strategy_adverse_signal_calculation() -> None:
    """Verify that adverse signal computes correct differential for buyers and sellers."""
    strat = ModelPassiveStrategy(model=None)
    parent_sell = ParentOrder(agent_id=0, side=Side.SELL, qty=100, start_ns=0, end_ns=100)
    strat.reset(parent_sell)

    # For sellers: adverse is DOWN (index 0), favorable is UP (index 2)
    probs = (0.7, 0.1, 0.2)  # down, flat, up
    assert strat.adverse_signal(probs) == 0.7 - 0.2

    parent_buy = ParentOrder(agent_id=0, side=Side.BUY, qty=100, start_ns=0, end_ns=100)
    strat.reset(parent_buy)

    # For buyers: adverse is UP (index 2), favorable is DOWN (index 0)
    assert strat.adverse_signal(probs) == 0.2 - 0.7
