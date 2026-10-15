"""Tests for market orders."""

from lobx.engine.book_py import PythonOrderBook
from lobx.types import Order, OrderType, Side


def _limit(order_id: int, side: Side, price: int, qty: int, ts: int = 0) -> Order:
    return Order(order_id, 1, side, OrderType.LIMIT, price, qty, ts)


def _market(order_id: int, side: Side, qty: int, ts: int = 0) -> Order:
    return Order(order_id, 1, side, OrderType.MARKET, None, qty, ts)


def test_market_buy_walks_the_asks_best_price_first() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 101, 3))
    book.submit(_limit(2, Side.SELL, 100, 3))
    fills = book.submit(_market(3, Side.BUY, 5, ts=9))
    assert [(f.maker_order_id, f.price, f.qty, f.ts) for f in fills] == [
        (2, 100, 3, 9),
        (1, 101, 2, 9),
    ]
    assert book.snapshot(1).asks == ((101, 1),)


def test_market_sell_walks_the_bids_best_price_first() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 99, 3))
    book.submit(_limit(2, Side.BUY, 100, 3))
    fills = book.submit(_market(3, Side.SELL, 4))
    assert [(f.maker_order_id, f.price, f.qty) for f in fills] == [(2, 100, 3), (1, 99, 1)]


def test_market_order_remainder_is_discarded_not_rested() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 3))
    fills = book.submit(_market(2, Side.BUY, 10))
    assert sum(f.qty for f in fills) == 3
    assert book.best_bid() is None
    assert book.best_ask() is None


def test_market_order_into_an_empty_book_does_nothing() -> None:
    book = PythonOrderBook()
    assert book.submit(_market(1, Side.BUY, 5)) == []
    assert book.snapshot(5).bids == ()


def test_market_order_takes_the_taker_side_from_the_order() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 100, 5))
    fills = book.submit(_market(2, Side.SELL, 5))
    assert fills[0].taker_side is Side.SELL
