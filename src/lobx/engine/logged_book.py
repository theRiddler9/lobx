"""A wrapper that records everything an engine does into an event log.

``LoggedBook`` implements the same ``OrderBook`` protocol as the engine it wraps, so it can stand in
for it anywhere. Wrapping instead of building the logging into each engine means the Python engine
and the C++ engine (phase 3) get identical logging for free, and the engines stay free of any
bookkeeping that is not matching.
"""

from __future__ import annotations

from collections.abc import Callable

from lobx.engine.event_log import EventLog
from lobx.engine.order_tracker import OrderTracker
from lobx.engine.protocol import OrderBook
from lobx.errors import SimulationError
from lobx.types import BookSnapshot, CancelEvent, Fill, FillEvent, Order, SubmitEvent

# Called once per trade with (fill, maker_agent_id, taker_agent_id).
FillListener = Callable[[Fill, int, int], None]


class LoggedBook:
    """An ``OrderBook`` that logs every accepted submit, every cancel, and every fill."""

    def __init__(self, book: OrderBook, log: EventLog | None = None) -> None:
        """Wrap ``book``. Events go to ``log``, or to a new empty log if none is given."""
        self._book = book
        self.log = log if log is not None else EventLog()
        # Read-only for callers: who owns each resting order, and which orders an agent has open.
        self.tracker = OrderTracker()
        self._listeners: list[FillListener] = []

    def add_fill_listener(self, listener: FillListener) -> None:
        """Call ``listener(fill, maker_agent_id, taker_agent_id)`` for every trade from now on."""
        self._listeners.append(listener)

    def submit(self, order: Order) -> list[Fill]:
        """Submit ``order`` to the wrapped engine and log the result.

        If the engine rejects the order it raises before anything is logged, so the log only ever
        contains orders that really entered the book.
        """
        fills = self._book.submit(order)
        self.log.append(SubmitEvent(order))
        for fill in fills:
            # Ask the tracker *before* it is updated: a maker that this very fill completes is
            # forgotten afterwards, and we still need to know whose it was.
            maker_agent = self.tracker.agent_of(fill.maker_order_id)
            if maker_agent is None:
                raise SimulationError(
                    f"engine filled order {fill.maker_order_id}, which is not open"
                )
            self.log.append(FillEvent(fill, self_trade=maker_agent == order.agent_id))
            for listener in self._listeners:
                listener(fill, maker_agent, order.agent_id)
        self.tracker.on_submit(order, fills)
        return fills

    def cancel(self, order_id: int, ts: int) -> bool:
        """Cancel through the wrapped engine and log the request with its outcome."""
        cancelled = self._book.cancel(order_id, ts)
        self.log.append(CancelEvent(order_id, ts, cancelled))
        self.tracker.on_cancel(order_id, cancelled)
        return cancelled

    def snapshot(self, depth: int) -> BookSnapshot:
        """Return the wrapped engine's snapshot."""
        return self._book.snapshot(depth)

    def best_bid(self) -> int | None:
        """Return the wrapped engine's best bid."""
        return self._book.best_bid()

    def best_ask(self) -> int | None:
        """Return the wrapped engine's best ask."""
        return self._book.best_ask()
