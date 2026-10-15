"""Tests for the FIFO price level."""

from lobx.engine.price_level import PriceLevel, RestingOrder
from lobx.types import Side


def _order(order_id: int, qty: int, price: int = 100) -> RestingOrder:
    return RestingOrder(order_id=order_id, agent_id=1, side=Side.BUY, price=price, qty=qty, ts=0)


def test_new_level_is_empty() -> None:
    level = PriceLevel(100)
    assert level.is_empty()
    assert level.total_qty == 0
    assert len(level) == 0


def test_add_tracks_total_quantity() -> None:
    level = PriceLevel(100)
    level.add(_order(1, 5))
    level.add(_order(2, 7))
    assert level.total_qty == 12
    assert len(level) == 2


def test_front_is_the_earliest_order() -> None:
    level = PriceLevel(100)
    level.add(_order(1, 5))
    level.add(_order(2, 7))
    assert level.front().order_id == 1


def test_fill_front_partial_keeps_order_at_front() -> None:
    level = PriceLevel(100)
    level.add(_order(1, 5))
    level.add(_order(2, 7))
    filled = level.fill_front(3)
    assert filled.order_id == 1
    assert filled.qty == 2
    assert level.front().order_id == 1
    assert level.total_qty == 9


def test_fill_front_complete_removes_order() -> None:
    level = PriceLevel(100)
    level.add(_order(1, 5))
    level.add(_order(2, 7))
    filled = level.fill_front(5)
    assert filled.qty == 0
    assert level.front().order_id == 2
    assert level.total_qty == 7


def test_remove_from_the_middle_keeps_fifo_order() -> None:
    level = PriceLevel(100)
    for order_id in (1, 2, 3):
        level.add(_order(order_id, 1))
    removed = level.remove(2)
    assert removed is not None and removed.order_id == 2
    assert level.total_qty == 2
    assert level.front().order_id == 1
    level.fill_front(1)
    assert level.front().order_id == 3


def test_remove_unknown_order_returns_none() -> None:
    level = PriceLevel(100)
    assert level.remove(99) is None
