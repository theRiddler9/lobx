"""Tests for LeadLagArbitrageAgent and CorrelatedMarketMaker."""

from lobx.sim.actions import Submit
from lobx.sim.agents.lead_lag import CorrelatedMarketMaker, LeadLagArbitrageAgent
from lobx.sim.cross_market import CrossMarket, CrossMarketConfig
from lobx.types import Side


def test_lead_lag_arbitrage_triggers_on_imbalance() -> None:
    cfg = CrossMarketConfig(symbols=("LEAD", "LAG"), duration_ns=1_000_000_000)
    market = CrossMarket(cfg)
    market.seed_initial_liquidity(depth_per_side=5, qty_per_level=100)

    arb = LeadLagArbitrageAgent(
        agent_id=10,
        lead_symbol="LEAD",
        lag_symbol="LAG",
        imbalance_threshold=0.3,
        trade_qty=20,
    )

    # Initially, book is balanced (imbalance = 0.0)
    actions = arb.wake(0, market)
    assert len(actions) == 0

    # Create artificial strong buy imbalance on LEAD by adding large bid size
    from lobx.types import Order, OrderType

    market._multi_book.submit(
        "LEAD",
        Order(999, 99, Side.BUY, OrderType.LIMIT, 9998, 500, 10),
    )

    imb = market.imbalance("LEAD", depth=1)
    assert imb > 0.3

    # Now agent wakes and sees positive imbalance
    actions2 = arb.wake(20, market)
    assert len(actions2) == 1
    assert isinstance(actions2[0], Submit)
    assert actions2[0].side is Side.BUY
    assert actions2[0].symbol == "LAG"
    assert actions2[0].qty == 20


def test_correlated_market_maker_quotes() -> None:
    cfg = CrossMarketConfig(symbols=("LEAD", "LAG"), duration_ns=1_000_000_000)
    market = CrossMarket(cfg)
    market.seed_initial_liquidity(depth_per_side=2, qty_per_level=50)

    cmm = CorrelatedMarketMaker(
        agent_id=5,
        symbol="LAG",
        anchor_symbol="LEAD",
        half_spread=3,
        quote_qty=25,
    )

    actions = cmm.wake(0, market)
    # Quotes both sides around LEAD mid (10000)
    submits = [a for a in actions if isinstance(a, Submit)]
    assert len(submits) == 2
    bids = [s for s in submits if s.side is Side.BUY]
    asks = [s for s in submits if s.side is Side.SELL]
    assert len(bids) == 1
    assert len(asks) == 1
    assert bids[0].price == 9997
    assert asks[0].price == 10003
