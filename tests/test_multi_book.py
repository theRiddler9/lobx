"""Tests for MultiOrderBook routing, metrics, and error handling."""

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.multi_book import MultiOrderBook, UnknownSymbolError
from lobx.engine.native import NativeMultiOrderBook, native_available
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


@pytest.mark.skipif(not native_available(), reason="native engine not available")
def test_native_multi_book_differential() -> None:
    py_mb = MultiOrderBook(["SPY", "QQQ"])
    cpp_mb = NativeMultiOrderBook(["SPY", "QQQ"])

    orders = [
        ("SPY", Order(1, 1, Side.BUY, OrderType.LIMIT, 10000, 100, 10)),
        ("QQQ", Order(2, 2, Side.BUY, OrderType.LIMIT, 20000, 50, 15)),
        ("SPY", Order(3, 3, Side.SELL, OrderType.LIMIT, 10005, 100, 20)),
        ("QQQ", Order(4, 4, Side.SELL, OrderType.LIMIT, 20010, 50, 25)),
        ("SPY", Order(5, 5, Side.BUY, OrderType.LIMIT, 10005, 40, 30)),
        ("QQQ", Order(6, 6, Side.SELL, OrderType.MARKET, None, 20, 35)),
    ]

    for sym, order in orders:
        py_fills = py_mb.submit(sym, order)
        cpp_fills = cpp_mb.submit(sym, order)
        assert len(py_fills) == len(cpp_fills)
        for pf, cf in zip(py_fills, cpp_fills, strict=True):
            assert pf.maker_order_id == cf.maker_order_id
            assert pf.taker_order_id == cf.taker_order_id
            assert pf.price == cf.price
            assert pf.qty == cf.qty

    for sym in ["SPY", "QQQ"]:
        assert py_mb.best_bid(sym) == cpp_mb.best_bid(sym)
        assert py_mb.best_ask(sym) == cpp_mb.best_ask(sym)
        assert py_mb.mid_price(sym) == cpp_mb.mid_price(sym)
        assert py_mb.imbalance(sym) == pytest.approx(cpp_mb.imbalance(sym))
        py_snap = py_mb.snapshot(sym, 5)
        cpp_snap = cpp_mb.snapshot(sym, 5)
        assert py_snap.bids == cpp_snap.bids
        assert py_snap.asks == cpp_snap.asks
