"""Tests for the validation rules in docs/contracts/shared-contracts.md."""

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.errors import DuplicateOrderIdError, InvalidOrderError
from lobx.types import Order, OrderType, Side


def _limit(order_id: int = 1, price: int | None = 100, qty: int = 5, ts: int = 0) -> Order:
    return Order(order_id, 1, Side.BUY, OrderType.LIMIT, price, qty, ts)


def _market(order_id: int = 1, price: int | None = None, qty: int = 5, ts: int = 0) -> Order:
    return Order(order_id, 1, Side.BUY, OrderType.MARKET, price, qty, ts)


@pytest.mark.parametrize("qty", [0, -1])
def test_non_positive_quantity_is_rejected(qty: int) -> None:
    with pytest.raises(InvalidOrderError, match="qty"):
        PythonOrderBook().submit(_limit(qty=qty))


def test_bool_quantity_is_rejected() -> None:
    with pytest.raises(InvalidOrderError, match="qty"):
        PythonOrderBook().submit(_limit(qty=True))


def test_float_quantity_is_rejected() -> None:
    with pytest.raises(InvalidOrderError, match="qty"):
        PythonOrderBook().submit(_limit(qty=1.5))  # type: ignore[arg-type]


@pytest.mark.parametrize("price", [None, 0, -5])
def test_limit_order_needs_a_positive_price(price: int | None) -> None:
    with pytest.raises(InvalidOrderError, match="price"):
        PythonOrderBook().submit(_limit(price=price))


def test_float_price_is_rejected() -> None:
    with pytest.raises(InvalidOrderError, match="price"):
        PythonOrderBook().submit(_limit(price=100.5))  # type: ignore[arg-type]


def test_market_order_must_not_carry_a_price() -> None:
    with pytest.raises(InvalidOrderError, match="market order"):
        PythonOrderBook().submit(_market(price=100))


def test_duplicate_resting_order_id_is_rejected() -> None:
    book = PythonOrderBook()
    book.submit(_limit(order_id=1))
    with pytest.raises(DuplicateOrderIdError, match="already resting"):
        book.submit(_limit(order_id=1, price=99, ts=1))


def test_order_id_can_be_reused_after_the_order_is_gone() -> None:
    book = PythonOrderBook()
    book.submit(_limit(order_id=1))
    book.cancel(1, ts=1)
    book.submit(_limit(order_id=1, ts=2))


def test_timestamp_going_backwards_is_rejected() -> None:
    book = PythonOrderBook()
    book.submit(_limit(order_id=1, ts=10))
    with pytest.raises(InvalidOrderError, match="ts"):
        book.submit(_limit(order_id=2, ts=9))


def test_equal_timestamps_are_allowed() -> None:
    book = PythonOrderBook()
    book.submit(_limit(order_id=1, ts=10))
    book.submit(_limit(order_id=2, price=99, ts=10))


def test_negative_timestamp_is_rejected() -> None:
    with pytest.raises(InvalidOrderError, match="ts"):
        PythonOrderBook().submit(_limit(ts=-1))


def test_cancel_with_a_timestamp_in_the_past_is_rejected() -> None:
    book = PythonOrderBook()
    book.submit(_limit(ts=10))
    with pytest.raises(InvalidOrderError, match="ts"):
        book.cancel(1, ts=5)


def test_rejected_order_leaves_the_book_untouched() -> None:
    book = PythonOrderBook()
    book.submit(_limit(order_id=1, ts=10))
    before = book.snapshot(5)
    with pytest.raises(InvalidOrderError):
        book.submit(_limit(order_id=2, qty=0, ts=11))
    assert book.snapshot(5) == before


def test_rejected_order_does_not_advance_the_clock() -> None:
    book = PythonOrderBook()
    book.submit(_limit(order_id=1, ts=10))
    with pytest.raises(InvalidOrderError):
        book.submit(_limit(order_id=2, qty=0, ts=500))
    book.submit(_limit(order_id=3, price=99, ts=11))  # would fail if ts 500 had been recorded
