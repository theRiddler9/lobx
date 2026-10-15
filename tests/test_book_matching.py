"""Tests for matching crossing limit orders."""

from lobx.engine.book_py import PythonOrderBook
from lobx.types import Fill, Order, OrderType, Side


def _limit(order_id: int, side: Side, price: int, qty: int, ts: int = 0, agent: int = 1) -> Order:
    return Order(order_id, agent, side, OrderType.LIMIT, price, qty, ts)


def test_non_crossing_limit_order_rests_without_fills() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 101, 5))
    assert book.submit(_limit(2, Side.BUY, 100, 5)) == []
    assert book.best_bid() == 100
    assert book.best_ask() == 101


def test_crossing_buy_trades_at_the_resting_price() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 5))
    fills = book.submit(_limit(2, Side.BUY, 105, 5, ts=7))
    # The buyer was willing to pay 105 but the resting sell is at 100: trade at 100.
    assert fills == [Fill(1, 2, Side.BUY, 100, 5, 7)]
    assert book.best_ask() is None
    assert book.best_bid() is None


def test_crossing_sell_trades_at_the_resting_price() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.BUY, 100, 5))
    fills = book.submit(_limit(2, Side.SELL, 95, 5, ts=3))
    assert fills == [Fill(1, 2, Side.SELL, 100, 5, 3)]
    assert book.best_bid() is None


def test_order_at_exactly_the_opposite_price_trades() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 5))
    assert len(book.submit(_limit(2, Side.BUY, 100, 5))) == 1


def test_better_prices_match_first() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 102, 5))
    book.submit(_limit(2, Side.SELL, 100, 5))
    book.submit(_limit(3, Side.SELL, 101, 5))
    fills = book.submit(_limit(4, Side.BUY, 102, 10))
    assert [(f.maker_order_id, f.price) for f in fills] == [(2, 100), (3, 101)]
    assert book.best_ask() == 102


def test_earlier_orders_at_the_same_price_match_first() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 5, ts=1))
    book.submit(_limit(2, Side.SELL, 100, 5, ts=2))
    fills = book.submit(_limit(3, Side.BUY, 100, 5, ts=3))
    assert [f.maker_order_id for f in fills] == [1]
    assert book.snapshot(1).asks == ((100, 5),)


def test_limit_price_stops_the_walk_through_the_book() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 5))
    book.submit(_limit(2, Side.SELL, 103, 5))
    fills = book.submit(_limit(3, Side.BUY, 101, 20))
    assert [f.price for f in fills] == [100]
    # The unfilled 15 rests as a bid at the order's own limit price.
    assert book.snapshot(1).bids == ((101, 15),)
    assert book.snapshot(1).asks == ((103, 5),)


def test_self_trades_are_allowed() -> None:
    book = PythonOrderBook()
    book.submit(_limit(1, Side.SELL, 100, 5, agent=9))
    fills = book.submit(_limit(2, Side.BUY, 100, 5, agent=9))
    assert len(fills) == 1
