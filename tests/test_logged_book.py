"""Tests for the logging wrapper around an engine."""

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.logged_book import LoggedBook
from lobx.engine.protocol import OrderBook
from lobx.errors import InvalidOrderError
from lobx.types import (
    CancelEvent,
    Fill,
    FillEvent,
    Order,
    OrderType,
    Side,
    SubmitEvent,
)


def _limit(order_id: int, agent: int, side: Side, price: int, qty: int, ts: int) -> Order:
    return Order(order_id, agent, side, OrderType.LIMIT, price, qty, ts)


def _book() -> LoggedBook:
    return LoggedBook(PythonOrderBook())


def test_it_is_an_order_book() -> None:
    assert isinstance(_book(), OrderBook)


def test_resting_order_logs_one_submit_and_no_fills() -> None:
    book = _book()
    order = _limit(1, 1, Side.BUY, 100, 5, 0)
    assert book.submit(order) == []
    assert list(book.log) == [SubmitEvent(order)]


def test_fills_are_logged_right_after_their_submit_in_execution_order() -> None:
    book = _book()
    book.submit(_limit(1, 1, Side.SELL, 101, 3, 0))
    book.submit(_limit(2, 2, Side.SELL, 102, 3, 1))
    taker = _limit(3, 3, Side.BUY, 102, 5, 2)
    fills = book.submit(taker)
    assert fills == [Fill(1, 3, Side.BUY, 101, 3, 2), Fill(2, 3, Side.BUY, 102, 2, 2)]
    assert list(book.log)[2:] == [
        SubmitEvent(taker),
        FillEvent(fills[0], self_trade=False),
        FillEvent(fills[1], self_trade=False),
    ]


def test_self_trade_is_flagged() -> None:
    book = _book()
    book.submit(_limit(1, 7, Side.SELL, 100, 2, 0))
    book.submit(_limit(2, 7, Side.BUY, 100, 2, 1))
    assert [e.self_trade for e in book.log if isinstance(e, FillEvent)] == [True]


def test_self_trade_is_flagged_even_when_the_maker_is_completed_by_that_fill() -> None:
    # The maker is forgotten by the tracker once it is fully filled, but the flag must still
    # be computed from its owner. Order 1 (agent 7) is filled completely by order 2 (agent 7).
    book = _book()
    book.submit(_limit(1, 7, Side.SELL, 100, 2, 0))
    book.submit(_limit(2, 8, Side.BUY, 100, 1, 1))  # partial: agent 8, not a self-trade
    book.submit(_limit(3, 7, Side.BUY, 100, 1, 2))  # completes order 1, same agent
    flags = [e.self_trade for e in book.log if isinstance(e, FillEvent)]
    assert flags == [False, True]


def test_cancel_is_logged_with_its_outcome() -> None:
    book = _book()
    book.submit(_limit(1, 1, Side.BUY, 100, 5, 0))
    assert book.cancel(1, 1) is True
    assert book.cancel(1, 2) is False
    assert list(book.log)[1:] == [CancelEvent(1, 1, True), CancelEvent(1, 2, False)]


def test_rejected_order_is_not_logged() -> None:
    book = _book()
    with pytest.raises(InvalidOrderError):
        book.submit(_limit(1, 1, Side.BUY, 100, 0, 0))
    assert len(book.log) == 0
    assert len(book.tracker) == 0


def test_listener_hears_every_trade_with_both_agents() -> None:
    book = _book()
    heard: list[tuple[Fill, int, int]] = []
    book.add_fill_listener(lambda fill, maker, taker: heard.append((fill, maker, taker)))
    book.submit(_limit(1, 4, Side.SELL, 100, 5, 0))
    book.submit(_limit(2, 9, Side.BUY, 100, 5, 1))
    assert heard == [(Fill(1, 2, Side.BUY, 100, 5, 1), 4, 9)]


def test_tracker_follows_the_book() -> None:
    book = _book()
    book.submit(_limit(1, 1, Side.SELL, 100, 5, 0))
    book.submit(_limit(2, 2, Side.BUY, 100, 2, 1))
    assert book.tracker.remaining(1) == 3
    assert book.tracker.open_order_ids(2) == ()  # fully filled, nothing rests
    book.cancel(1, 2)
    assert len(book.tracker) == 0


def test_reads_are_passed_through() -> None:
    book = _book()
    book.submit(_limit(1, 1, Side.BUY, 99, 5, 0))
    book.submit(_limit(2, 1, Side.SELL, 101, 4, 1))
    assert (book.best_bid(), book.best_ask()) == (99, 101)
    snapshot = book.snapshot(5)
    assert snapshot.bids == ((99, 5),) and snapshot.asks == ((101, 4),)
