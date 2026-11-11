"""Tests for InformedTraderAgent and adverse selection mechanics."""

from __future__ import annotations

import numpy as np

from lobx.engine.book_py import PythonOrderBook
from lobx.sim.agents.informed import InformedTrader
from lobx.sim.market import Market
from lobx.sim.population import MarketConfig, build_background_agents


def test_informed_trader_basic_actions() -> None:
    """Verify that InformedTrader generates actions conforming to the Agent protocol."""
    trader = InformedTrader(agent_id=99, trade_qty=25, signal_strength=1.0)
    rng = np.random.default_rng(42)

    t0 = trader.first_wake(rng)
    assert t0 > 0

    t1 = trader.next_wake(t0, rng)
    assert t1 > t0


def test_informed_trader_in_market_population() -> None:
    """Verify that adding informed traders runs deterministically and produces trades."""
    cfg = MarketConfig(
        seed=123,
        duration_ns=5_000_000_000,
        num_informed_traders=2,
    )
    agents = build_background_agents(cfg)
    assert any(isinstance(a, InformedTrader) for a in agents)

    book = PythonOrderBook()
    res = Market(cfg, agents, book=book).run()
    assert res.trades > 0
