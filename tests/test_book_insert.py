"""Tests for resting limit orders and book snapshots (no matching involved)."""

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.types import Order, OrderType, Side


def _limit(order_id: int, side: Side, price: int, qty: int, ts: int = 0) -> Order:
    return Order(order_id, 1, side, OrderType.LIMIT, price, qty, ts)


def test_empty_book_has_no_best_prices() -> None:
    book = PythonOrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    snap = book.snapshot(5)
    assert snap.bids == () and snap.asks == () and snap.ts == 0


def test_resting_orders_set_best_prices() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 99, 5))
    book.submit(_limit(2, Side.SELL, 101, 5))
    assert book.best_bid() == 99
    assert book.best_ask() == 101


def test_snapshot_aggregates_quantity_per_level_best_first() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 99, 5))
    book.submit(_limit(2, Side.BUY, 99, 3))
    book.submit(_limit(3, Side.BUY, 98, 4))
    book.submit(_limit(4, Side.SELL, 101, 2))
    book.submit(_limit(5, Side.SELL, 102, 6))
    snap = book.snapshot(5)
    assert snap.bids == ((99, 8), (98, 4))
    assert snap.asks == ((101, 2), (102, 6))


def test_snapshot_respects_depth() -> None:
    book = PythonOrderBook()
    for i, price in enumerate((99, 98, 97), start=1):
        book.submit(_limit(i, Side.BUY, price, 1))
    assert book.snapshot(2).bids == ((99, 1), (98, 1))


def test_snapshot_ts_follows_the_last_call() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 99, 5, ts=42))
    assert book.snapshot(1).ts == 42


def test_snapshot_rejects_non_positive_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        PythonOrderBook().snapshot(0)
