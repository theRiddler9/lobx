"""Tests for partial fills on both the resting and the incoming side."""

from lobx.engine.book_py import PythonOrderBook
from lobx.types import Order, OrderType, Side


def _limit(order_id: int, side: Side, price: int, qty: int, ts: int = 0) -> Order:
    return Order(order_id, 1, side, OrderType.LIMIT, price, qty, ts)


def test_resting_order_partially_filled_keeps_its_place_in_the_queue() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 10))
    book.submit(_limit(2, Side.SELL, 100, 10))
    first = book.submit(_limit(3, Side.BUY, 100, 4))
    assert [(f.maker_order_id, f.qty) for f in first] == [(1, 4)]
    assert book.snapshot(1).asks == ((100, 16),)
    # Order 1 still has 6 left and still sits ahead of order 2.
    second = book.submit(_limit(4, Side.BUY, 100, 8))
    assert [(f.maker_order_id, f.qty) for f in second] == [(1, 6), (2, 2)]
    assert book.snapshot(1).asks == ((100, 8),)


def test_incoming_order_larger_than_the_book_rests_its_remainder() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 3))
    fills = book.submit(_limit(2, Side.BUY, 100, 10))
    assert [(f.maker_order_id, f.qty) for f in fills] == [(1, 3)]
    assert book.snapshot(1).bids == ((100, 7),)
    assert book.best_ask() is None


def test_one_incoming_order_can_produce_several_fills_in_execution_order() -> None:
    book = PythonOrderBook()
    for order_id, price in ((1, 100), (2, 101), (3, 102)):
        book.submit(_limit(order_id, Side.SELL, price, 2))
    fills = book.submit(_limit(4, Side.BUY, 102, 5))
    assert [(f.maker_order_id, f.price, f.qty) for f in fills] == [
        (1, 100, 2),
        (2, 101, 2),
        (3, 102, 1),
    ]
    assert book.snapshot(1).asks == ((102, 1),)


def test_filled_quantity_never_exceeds_the_incoming_quantity() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 50))
    fills = book.submit(_limit(2, Side.BUY, 100, 7))
    assert sum(f.qty for f in fills) == 7
    assert book.snapshot(1).asks == ((100, 43),)
