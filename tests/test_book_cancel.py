"""Tests for cancelling resting orders."""

from lobx.engine.book_py import PythonOrderBook
from lobx.types import Order, OrderType, Side


def _limit(order_id: int, side: Side, price: int, qty: int, ts: int = 0) -> Order:
    return Order(order_id, 1, side, OrderType.LIMIT, price, qty, ts)


def test_cancel_removes_a_resting_order() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 99, 5))
    assert book.cancel(1, ts=5) is True
    assert book.best_bid() is None
    assert book.snapshot(1).bids == ()


def test_cancel_only_removes_the_named_order_from_a_shared_level() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 99, 5))
    book.submit(_limit(2, Side.BUY, 99, 3))
    assert book.cancel(1, ts=5) is True
    assert book.snapshot(1).bids == ((99, 3),)


def test_cancelled_order_no_longer_matches() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 5))
    book.cancel(1, ts=1)
    assert book.submit(_limit(2, Side.BUY, 100, 5, ts=2)) == []


def test_cancel_unknown_order_returns_false() -> None:
    assert PythonOrderBook().cancel(42, ts=0) is False


def test_cancel_twice_returns_false_the_second_time() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 99, 5))
    assert book.cancel(1, ts=1) is True
    assert book.cancel(1, ts=2) is False


def test_cancel_of_a_fully_filled_order_returns_false() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 5))
    book.submit(_limit(2, Side.BUY, 100, 5, ts=1))
    assert book.cancel(1, ts=2) is False


def test_cancel_after_a_partial_fill_removes_the_remainder() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 10))
    book.submit(_limit(2, Side.BUY, 100, 4, ts=1))
    assert book.cancel(1, ts=2) is True
    assert book.best_ask() is None


def test_cancel_updates_the_snapshot_timestamp() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 99, 5, ts=1))
    book.cancel(1, ts=8)
    assert book.snapshot(1).ts == 8
