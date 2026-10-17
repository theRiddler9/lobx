"""Replay an event log into an engine and check that the engine reproduces what was recorded.

Replaying feeds every ``SubmitEvent`` and ``CancelEvent`` to a fresh engine, in order. The
``FillEvent`` lines are *not* fed in; they are the answer key. After each submit the engine's own
fills are compared, one by one, with the fills the log recorded. If anything differs the replay
stops with a ``ReplayMismatchError`` that names the position of the first difference.

This is what makes the log trustworthy: a log that replays cleanly is a complete description of the
run, and the same log can be replayed on the Python engine and the C++ engine to check they agree.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.event_log import EventLog
from lobx.engine.order_tracker import OrderTracker
from lobx.engine.protocol import OrderBook
from lobx.errors import InvalidOrderError, ReplayMismatchError
from lobx.types import CancelEvent, Event, Fill, FillEvent, SubmitEvent


def replay(
    log: EventLog,
    book: OrderBook | None = None,
    on_event: Callable[[Event, OrderBook], None] | None = None,
) -> OrderBook:
    """Replay ``log`` into ``book`` (a new ``PythonOrderBook`` by default) and return it.

    ``on_event(event, book)`` is called after each event is processed and verified, with the book
    in the state it has after that event. It is how analysis code walks through a run.

    Raises ``ReplayMismatchError`` if the engine disagrees with the log.
    """
    engine: OrderBook = book if book is not None else PythonOrderBook()
    tracker = OrderTracker()
    # Fills the engine produced for the latest submit that the log has not yet "used up", each
    # with the self-trade flag we expect. The log must list exactly these, in this order.
    pending: deque[tuple[Fill, bool]] = deque()

    for index, event in enumerate(log):
        if isinstance(event, FillEvent):
            _check_fill(index, event, pending)
        else:
            if pending:
                raise ReplayMismatchError(
                    f"event #{index}: the engine produced {len(pending)} more fill(s) than the log "
                    f"recorded for the previous order"
                )
            if isinstance(event, SubmitEvent):
                pending.extend(_apply_submit(index, event, engine, tracker))
            else:
                _apply_cancel(index, event, engine, tracker)
        if on_event is not None:
            on_event(event, engine)

    if pending:
        raise ReplayMismatchError(
            f"the log ends but the engine produced {len(pending)} more fill(s) than recorded"
        )
    return engine


def _apply_submit(
    index: int, event: SubmitEvent, engine: OrderBook, tracker: OrderTracker
) -> list[tuple[Fill, bool]]:
    order = event.order
    try:
        fills = engine.submit(order)
    except InvalidOrderError as exc:
        raise ReplayMismatchError(
            f"event #{index}: the engine rejected order {order.order_id}, "
            f"which the log recorded as accepted ({exc})"
        ) from exc
    # The expected self-trade flag has to be read before the tracker is updated, because a maker
    # that this order completes is forgotten by that update.
    expected = [(fill, tracker.agent_of(fill.maker_order_id) == order.agent_id) for fill in fills]
    tracker.on_submit(order, fills)
    return expected


def _apply_cancel(index: int, event: CancelEvent, engine: OrderBook, tracker: OrderTracker) -> None:
    cancelled = engine.cancel(event.order_id, event.ts)
    if cancelled != event.cancelled:
        raise ReplayMismatchError(
            f"event #{index}: cancel of order {event.order_id} returned {cancelled}, "
            f"the log recorded {event.cancelled}"
        )
    tracker.on_cancel(event.order_id, cancelled)


def _check_fill(index: int, event: FillEvent, pending: deque[tuple[Fill, bool]]) -> None:
    if not pending:
        raise ReplayMismatchError(
            f"event #{index}: the log records a fill the engine did not produce: {event.fill}"
        )
    expected_fill, expected_self_trade = pending.popleft()
    if event.fill != expected_fill:
        raise ReplayMismatchError(
            f"event #{index}: fill differs. engine: {expected_fill}; log: {event.fill}"
        )
    if event.self_trade != expected_self_trade:
        raise ReplayMismatchError(
            f"event #{index}: self_trade flag is {event.self_trade} in the log "
            f"but the owners say {expected_self_trade}"
        )
