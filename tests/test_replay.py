"""Tests for replaying an event log into an engine."""

import dataclasses
from pathlib import Path

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.event_log import EventLog
from lobx.engine.logged_book import LoggedBook
from lobx.engine.replay import replay
from lobx.errors import ReplayMismatchError
from lobx.types import (
    CancelEvent,
    Event,
    Fill,
    FillEvent,
    Order,
    OrderType,
    Side,
    SubmitEvent,
)

_DEPTH = 50


def _limit(order_id: int, agent: int, side: Side, price: int, qty: int, ts: int) -> Order:
    return Order(order_id, agent, side, OrderType.LIMIT, price, qty, ts)


def _recorded_run() -> tuple[LoggedBook, EventLog]:
    """A small run with partial fills, a market order, a self-trade and cancels."""
    book = LoggedBook(PythonOrderBook(), EventLog(seed=1, config={"demo": True}))
    book.submit(_limit(1, 1, Side.SELL, 101, 5, 0))
    book.submit(_limit(2, 2, Side.SELL, 102, 5, 1))
    book.submit(_limit(3, 3, Side.BUY, 99, 8, 2))
    book.submit(_limit(4, 4, Side.BUY, 101, 7, 3))  # sweeps order 1, rests 2 at 101
    book.submit(Order(5, 1, Side.SELL, OrderType.MARKET, None, 20, 4))  # hits bids, partly
    book.submit(_limit(6, 2, Side.SELL, 110, 3, 5))
    book.submit(_limit(7, 2, Side.BUY, 110, 1, 6))  # self-trade: agent 2 buys from agent 2
    book.cancel(6, 7)
    book.cancel(6, 8)  # already gone: logged as not cancelled
    return book, book.log


def _with_event_changed(log: EventLog, index: int, new: Event | None) -> EventLog:
    """Copy ``log`` with one event replaced (or removed if ``new`` is ``None``)."""
    copy = EventLog(log.seed, log.config)
    for i, event in enumerate(log):
        if i != index:
            copy.append(event)
        elif new is not None:
            copy.append(new)
    return copy


def _first_fill_index(log: EventLog) -> int:
    return next(i for i, e in enumerate(log) if isinstance(e, FillEvent))


def test_replay_rebuilds_the_final_book_exactly() -> None:
    original, log = _recorded_run()
    rebuilt = replay(log)
    assert rebuilt.snapshot(_DEPTH) == original.snapshot(_DEPTH)


def test_replay_into_a_given_engine_returns_that_engine() -> None:
    _, log = _recorded_run()
    target = PythonOrderBook()
    assert replay(log, target) is target


def test_replay_of_a_log_that_went_through_a_file(tmp_path: Path) -> None:
    original, log = _recorded_run()
    log.write(tmp_path / "run.eventlog")
    rebuilt = replay(EventLog.read(tmp_path / "run.eventlog"))
    assert rebuilt.snapshot(_DEPTH) == original.snapshot(_DEPTH)


def test_on_event_is_called_once_per_event_with_the_current_book() -> None:
    _, log = _recorded_run()
    seen: list[Event] = []
    replay(log, on_event=lambda event, _book: seen.append(event))
    assert seen == list(log)


def test_empty_log_replays_to_an_empty_book() -> None:
    book = replay(EventLog())
    assert (book.best_bid(), book.best_ask()) == (None, None)


def test_changed_fill_price_is_detected() -> None:
    _, log = _recorded_run()
    index = _first_fill_index(log)
    event = log[index]
    assert isinstance(event, FillEvent)
    forged = FillEvent(
        dataclasses.replace(event.fill, price=event.fill.price + 1), event.self_trade
    )
    with pytest.raises(ReplayMismatchError, match=rf"event #{index}: fill differs"):
        replay(_with_event_changed(log, index, forged))


def test_a_missing_fill_is_detected() -> None:
    _, log = _recorded_run()
    index = _first_fill_index(log)
    with pytest.raises(ReplayMismatchError, match="more fill"):
        replay(_with_event_changed(log, index, None))


def test_an_invented_fill_is_detected() -> None:
    _, log = _recorded_run()
    extra = FillEvent(Fill(1, 2, Side.BUY, 100, 1, 0), self_trade=False)
    # Put it right after the first (resting, fill-less) submit.
    broken = EventLog(log.seed, log.config)
    for i, event in enumerate(log):
        broken.append(event)
        if i == 0:
            broken.append(extra)
    with pytest.raises(ReplayMismatchError, match="did not produce"):
        replay(broken)


def test_flipped_cancel_result_is_detected() -> None:
    _, log = _recorded_run()
    index = next(i for i, e in enumerate(log) if isinstance(e, CancelEvent) and e.cancelled)
    event = log[index]
    assert isinstance(event, CancelEvent)
    flipped = CancelEvent(event.order_id, event.ts, cancelled=False)
    with pytest.raises(ReplayMismatchError, match="returned True, the log recorded False"):
        replay(_with_event_changed(log, index, flipped))


def test_flipped_self_trade_flag_is_detected() -> None:
    _, log = _recorded_run()
    index, event = next(
        (i, e) for i, e in enumerate(log) if isinstance(e, FillEvent) and e.self_trade
    )
    assert isinstance(event, FillEvent)
    with pytest.raises(ReplayMismatchError, match="self_trade flag"):
        replay(_with_event_changed(log, index, FillEvent(event.fill, self_trade=False)))


def test_an_order_the_engine_rejects_is_reported_as_a_mismatch() -> None:
    log = EventLog()
    log.append(SubmitEvent(_limit(1, 1, Side.BUY, 100, 0, 0)))  # qty 0 is invalid
    with pytest.raises(ReplayMismatchError, match="rejected order 1"):
        replay(log)
