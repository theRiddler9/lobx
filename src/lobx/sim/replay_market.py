"""Replay market: replays historical L2/L3 order streams with queue-position strategy execution.

Coordinates historical limit order book events (e.g. LOBSTER benchmark data) with
an execution strategy (TWAP, VWAP, Adaptive). Child orders placed by the strategy are
tracked via ``QueueTracker`` to enforce strict FIFO priority before filling.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from lobx.data.lobster import (
    LobsterBookSnapshot,
    LobsterMessage,
    LobsterType,
    iter_lobster_messages,
    iter_lobster_orderbook,
)
from lobx.engine.book_py import PythonOrderBook
from lobx.engine.native import NativeOrderBook, native_available
from lobx.engine.protocol import OrderBook
from lobx.eval.metrics import implementation_shortfall_bps
from lobx.sim.actions import Cancel, Submit
from lobx.sim.agents.base import MarketView
from lobx.sim.queue_position import CancellationModel, QueueTracker
from lobx.strategies.base import ExecutionStrategy, ParentOrder
from lobx.types import BookSnapshot, Fill, Order, OrderType, Side


@dataclass(frozen=True, slots=True)
class HistoricalRunResult:
    """The outcome of executing a strategy against historical order book replay."""

    executed: int
    arrival_mid2: int
    shortfall_bps: float | None
    fills: tuple[Fill, ...]
    events_replayed: int
    total_market_volume: int


class ReplayMarket(MarketView):
    """An event-driven replay market that plays historical LOBSTER order streams."""

    def __init__(
        self,
        message_path: Path | str,
        book_path: Path | str | None = None,
        book: OrderBook | None = None,
        strategy: ExecutionStrategy | None = None,
        parent: ParentOrder | None = None,
        wake_interval_ns: int = 1_000_000_000,
        cancellation_model: CancellationModel = CancellationModel.PROPORTIONAL,
        opening_price: int = 10_000,
    ) -> None:
        """Initialize historical replay market."""
        self.message_path = Path(message_path)
        self.book_path = Path(book_path) if book_path else None
        self.book = book or (NativeOrderBook() if native_available() else PythonOrderBook())
        self.strategy = strategy
        self.parent = parent
        self.wake_interval_ns = wake_interval_ns
        self.queue_tracker = QueueTracker(cancellation_model=cancellation_model)
        self.opening_price = opening_price

        self._ts: int = 0
        self._positions: dict[int, int] = {}
        self._strategy_fills: list[Fill] = []
        self._last_trade_px: int | None = None
        self._arrival_mid2: int | None = None
        self._total_market_volume: int = 0
        self._tape: list[Fill] = []
        self._open_order_ids: dict[int, set[int]] = {}
        self._next_order_id: int = 100_000_000

    # ------------------------------------------------------------------ MarketView protocol

    @property
    def ts(self) -> int:
        """Current timestamp in nanoseconds."""
        return self._ts

    def best_bid(self) -> int | None:
        """Highest resting buy price."""
        return self.book.best_bid()

    def best_ask(self) -> int | None:
        """Lowest resting sell price."""
        return self.book.best_ask()

    def mid2(self) -> int | None:
        """Twice the mid-price (an integer in ticks)."""
        bid = self.book.best_bid()
        ask = self.book.best_ask()
        return bid + ask if bid is not None and ask is not None else None

    def last_trade_price(self) -> int | None:
        """Price of most recent trade."""
        return self._last_trade_px

    def reference_price(self) -> int:
        """Return reference price for anchoring orders."""
        m2 = self.mid2()
        if m2 is not None:
            return m2 // 2
        if self._last_trade_px is not None:
            return self._last_trade_px
        return self.opening_price

    def snapshot(self, depth: int) -> BookSnapshot:
        """Return current book depth snapshot."""
        return self.book.snapshot(depth)

    def position(self, agent_id: int) -> int:
        """Net position of an agent."""
        return self._positions.get(agent_id, 0)

    def open_orders(self, agent_id: int) -> tuple[int, ...]:
        """List of active open order IDs for an agent."""
        return tuple(self._open_order_ids.get(agent_id, set()))

    def fills_after(self, ts: int) -> tuple[Fill, ...]:
        """Trades with timestamp strictly greater than ts."""
        return tuple(f for f in self._tape if f.ts > ts)

    # ------------------------------------------------------------------ Replay execution

    def run(self) -> HistoricalRunResult:
        """Execute replay across all historical messages in the dataset."""
        if self.strategy is not None and self.parent is not None:
            self.strategy.reset(self.parent)

        events_count = 0
        next_wake_ts = self.parent.start_ns if self.parent is not None else 0

        msg_iter = iter_lobster_messages(self.message_path)
        book_iter: Iterator[LobsterBookSnapshot] | None = (
            iter_lobster_orderbook(self.book_path) if self.book_path else None
        )

        for msg in msg_iter:
            self._ts = msg.ts_ns
            events_count += 1

            # 1. Wake strategy at scheduled intervals prior to this event
            if (
                self.strategy is not None
                and self.parent is not None
                and self._ts >= self.parent.start_ns
            ):
                if self._arrival_mid2 is None:
                    m2 = self.mid2()
                    self._arrival_mid2 = m2 if m2 is not None else self.reference_price() * 2

                while next_wake_ts <= self._ts and next_wake_ts <= self.parent.end_ns:
                    self._wake_strategy(next_wake_ts)
                    next_wake_ts += self.wake_interval_ns

            # 2. Process historical message
            self._apply_lobster_event(msg)

            # 3. Optional orderbook snapshot verification
            if book_iter is not None:
                _ = next(book_iter)

        # Final wake at deadline if parent order remains incomplete
        if (
            self.strategy is not None
            and self.parent is not None
            and next_wake_ts <= self.parent.end_ns
        ):
            self._wake_strategy(self.parent.end_ns)

        # Calculate shortfall
        arrival_mid2 = self._arrival_mid2 or self.reference_price() * 2
        total_executed = sum(f.qty for f in self._strategy_fills)

        shortfall = None
        if self.parent is not None and total_executed == self.parent.qty:
            shortfall = implementation_shortfall_bps(
                self.parent, self._strategy_fills, arrival_mid2
            )

        return HistoricalRunResult(
            executed=total_executed,
            arrival_mid2=arrival_mid2,
            shortfall_bps=shortfall,
            fills=tuple(self._strategy_fills),
            events_replayed=events_count,
            total_market_volume=self._total_market_volume,
        )

    def _wake_strategy(self, wake_ts: int) -> None:
        """Invoke strategy step and process emitted actions."""
        if self.strategy is None or self.parent is None:
            return

        actions = self.strategy.step(wake_ts, self)
        for action in actions:
            if isinstance(action, Cancel):
                self.queue_tracker.cancel_order(action.order_id)
                self.book.cancel(action.order_id, wake_ts)
                self._open_order_ids.get(self.parent.agent_id, set()).discard(action.order_id)
            elif isinstance(action, Submit):
                order_id = self._next_order_id
                self._next_order_id += 1
                if action.order_type is OrderType.MARKET:
                    order = Order(
                        order_id=order_id,
                        agent_id=self.parent.agent_id,
                        side=action.side,
                        type=OrderType.MARKET,
                        price=None,
                        qty=action.qty,
                        ts=wake_ts,
                    )
                    fills = self.book.submit(order)
                    for fill in fills:
                        self._record_strategy_fill(fill)
                elif action.order_type is OrderType.LIMIT and action.price is not None:
                    # Calculate depth ahead at this price
                    snap = self.book.snapshot(depth=20)
                    levels = snap.bids if action.side is Side.BUY else snap.asks
                    qty_ahead = 0
                    for p, q in levels:
                        if p == action.price:
                            qty_ahead = q
                            break

                    self.queue_tracker.add_order(
                        order_id=order_id,
                        agent_id=self.parent.agent_id,
                        side=action.side,
                        price=action.price,
                        qty=action.qty,
                        qty_ahead=qty_ahead,
                        ts=wake_ts,
                    )
                    self._open_order_ids.setdefault(self.parent.agent_id, set()).add(order_id)
                    order = Order(
                        order_id=order_id,
                        agent_id=self.parent.agent_id,
                        side=action.side,
                        type=OrderType.LIMIT,
                        price=action.price,
                        qty=action.qty,
                        ts=wake_ts,
                    )
                    self.book.submit(order)

    def _apply_lobster_event(self, msg: LobsterMessage) -> None:
        """Apply a historical LOBSTER event to the book and queue tracker."""
        if msg.event_type is LobsterType.SUBMISSION:
            hist_order = Order(
                order_id=msg.order_id,
                agent_id=0,
                side=msg.side,
                type=OrderType.LIMIT,
                price=msg.price,
                qty=msg.qty,
                ts=msg.ts_ns,
            )
            self.book.submit(hist_order)

        elif msg.event_type in (LobsterType.CANCELLATION, LobsterType.DELETION):
            self.queue_tracker.on_cancel(side=msg.side, price=msg.price, cancel_qty=msg.qty)
            self.book.cancel(msg.order_id, msg.ts_ns)

        elif msg.event_type in (LobsterType.EXECUTION_VISIBLE, LobsterType.EXECUTION_HIDDEN):
            self._total_market_volume += msg.qty
            self._last_trade_px = msg.price

            # In LOBSTER: msg.side represents the resting order side that was executed
            resting_side = msg.side
            qfills = self.queue_tracker.on_trade(
                side=resting_side,
                price=msg.price,
                trade_qty=msg.qty,
                ts=msg.ts_ns,
            )
            for qf in qfills:
                f = Fill(
                    maker_order_id=qf.order_id,
                    taker_order_id=msg.order_id,
                    taker_side=Side.SELL if resting_side is Side.BUY else Side.BUY,
                    price=qf.price,
                    qty=qf.qty,
                    ts=qf.ts,
                )
                self._record_strategy_fill(f)

            # Match aggressive order against resting liquidity in book
            aggressor_side = Side.SELL if resting_side is Side.BUY else Side.BUY
            aggressor = Order(
                order_id=msg.order_id + 1_000_000_000,
                agent_id=0,
                side=aggressor_side,
                type=OrderType.MARKET,
                price=None,
                qty=msg.qty,
                ts=msg.ts_ns,
            )
            trade_fills = self.book.submit(aggressor)
            self._tape.extend(trade_fills)

    def _record_strategy_fill(self, fill: Fill) -> None:
        """Record an execution fill achieved by the strategy and update its net position."""
        if self.parent is None:
            return
        self._strategy_fills.append(fill)
        agent_id = self.parent.agent_id
        if self.parent.side is Side.BUY:
            self._positions[agent_id] = self._positions.get(agent_id, 0) + fill.qty
        else:
            self._positions[agent_id] = self._positions.get(agent_id, 0) - fill.qty
