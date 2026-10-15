"""Tests for BookSide ordering and level management."""

from lobx.engine.book_side import BookSide
from lobx.engine.price_level import RestingOrder
from lobx.types import Side


def _add(side: BookSide, order_id: int, price: int, qty: int = 1) -> None:
    order = RestingOrder(order_id, 1, side.side, price, qty, 0)
    side.get_or_create(price).add(order)


def test_empty_side_has_no_best_price() -> None:
    side = BookSide(Side.BUY)
    assert side.is_empty()
    assert side.best_price() is None
    assert side.best_level() is None


def test_best_bid_is_the_highest_price() -> None:
    bids = BookSide(Side.BUY)
    for order_id, price in enumerate((99, 101, 100), start=1):
        _add(bids, order_id, price)
    assert bids.best_price() == 101
    assert [lvl.price for lvl in bids.levels_best_first()] == [101, 100, 99]


def test_best_ask_is_the_lowest_price() -> None:
    asks = BookSide(Side.SELL)
    for order_id, price in enumerate((102, 100, 101), start=1):
        _add(asks, order_id, price)
    assert asks.best_price() == 100
    assert [lvl.price for lvl in asks.levels_best_first()] == [100, 101, 102]


def test_get_or_create_returns_the_same_level_for_one_price() -> None:
    bids = BookSide(Side.BUY)
    assert bids.get_or_create(100) is bids.get_or_create(100)
    assert bids.get(100) is not None
    assert bids.get(101) is None


def test_remove_if_empty_only_removes_empty_levels() -> None:
    bids = BookSide(Side.BUY)
    _add(bids, 1, 100)
    bids.remove_if_empty(100)
    assert bids.best_price() == 100
    level = bids.get(100)
    assert level is not None
    level.remove(1)
    bids.remove_if_empty(100)
    assert bids.is_empty()
    assert bids.get(100) is None
