"""Tests for MultiOrderBook routing, metrics, and error handling."""

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.multi_book import MultiOrderBook, UnknownSymbolError
from lobx.types import Order, OrderType, Side


def test_multi_book_initialization_and_registration() -> None:
    mb = MultiOrderBook(["SPY", "QQQ"])
    assert set(mb.symbols) == {"SPY", "QQQ"}

    mb.add_symbol("AAPL")
    assert "AAPL" in mb.symbols
    assert isinstance(mb.get_book("AAPL"), PythonOrderBook)


def test_multi_book_unknown_symbol_raises() -> None:
    mb = MultiOrderBook(["SPY"])
    with pytest.raises(UnknownSymbolError, match="MSFT"):
        mb.get_book("MSFT")

    with pytest.raises(UnknownSymbolError):
        mb.snapshot("MSFT")


def test_multi_book_routing_submits_and_cancels() -> None:
    mb = MultiOrderBook(["SPY", "QQQ"])

    order_spy = Order(
        order_id=1,
        agent_id=10,
        side=Side.BUY,
        type=OrderType.LIMIT,
        price=10000,
        qty=50,
        ts=100,
    )
    fills_spy = mb.submit("SPY", order_spy)
    assert len(fills_spy) == 0
    assert mb.best_bid("SPY") == 10000
    assert mb.best_bid("QQQ") is None

    # Cancel on SPY
    assert mb.cancel("SPY", order_id=1, ts=200)
    assert mb.best_bid("SPY") is None


def test_multi_book_metrics() -> None:
    mb = MultiOrderBook(["LEAD", "LAG"])

    # Empty book metrics
    assert mb.mid_price("LEAD") is None
    assert mb.spread("LEAD") is None
    assert mb.imbalance("LEAD") == 0.0

    # Insert buy and sell on LEAD
    mb.submit(
        "LEAD",
        Order(1, 10, Side.BUY, OrderType.LIMIT, price=5000, qty=100, ts=1),
    )
    mb.submit(
        "LEAD",
        Order(2, 10, Side.SELL, OrderType.LIMIT, price=5004, qty=300, ts=2),
    )

    assert mb.best_bid("LEAD") == 5000
    assert mb.best_ask("LEAD") == 5004
    assert mb.mid_price("LEAD") == 5002.0
    assert mb.spread("LEAD") == 4
    # (100 - 300) / (100 + 300) = -200 / 400 = -0.5
    assert mb.imbalance("LEAD", depth=1) == pytest.approx(-0.5)

    # all_snapshots
    snaps = mb.all_snapshots(depth=2)
    assert "LEAD" in snaps
    assert "LAG" in snaps
    assert len(snaps["LEAD"].bids) == 1
    assert len(snaps["LAG"].bids) == 0
