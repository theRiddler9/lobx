"""FIFO queue-position tracking for resting limit orders in historical replay.

Under price-time priority, incoming market trades execute against resting limit
orders at the same price in strict FIFO sequence. A backtest that assumes immediate
fill upon touching the price exhibits massive optimistic bias.

``QueueTracker`` models the exact volume ahead of an order at price level P:
    - Trades at P first consume volume resting ahead (``qty_ahead``).
    - Only after ``qty_ahead == 0`` do subsequent trades at P execute against our order.
    - Cancellations at level P reduce queue priority proportionally or conservatively.

References:
    - Cont, R., Kukanov, S., & Stoikov, S. (2010). The Price Impact of Order Book Events.
    - Moallemi, C. C., & Yuan, M. (2017). A Model for Queue Position Valuation in LOBs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lobx.types import Side


class CancellationModel(Enum):
    """How cancellations at a price level affect resting queue position."""

    CONSERVATIVE = "conservative"  # Cancellations assumed to happen behind our order
    PROPORTIONAL = "proportional"  # Cancellations distributed proportionally across the queue
    OPTIMISTIC = "optimistic"  # Cancellations assumed ahead of our order


@dataclass(slots=True)
class QueueSlot:
    """Tracking state for one resting limit order inside the price queue."""

    order_id: int
    agent_id: int
    side: Side
    price: int
    qty: int
    qty_remaining: int
    qty_ahead: int
    ts_placed: int

    @property
    def is_filled(self) -> bool:
        """True if the entire order quantity has executed."""
        return self.qty_remaining == 0


@dataclass(frozen=True, slots=True)
class QueueFill:
    """An execution event generated when queue priority reaches our order."""

    order_id: int
    agent_id: int
    side: Side
    price: int
    qty: int
    ts: int


class QueueTracker:
    """Maintains queue priority for simulated agent orders placed in a historical book."""

    def __init__(
        self, cancellation_model: CancellationModel = CancellationModel.PROPORTIONAL
    ) -> None:
        """Initialize tracker with chosen cancellation assumption."""
        self.cancellation_model = cancellation_model
        # order_id -> QueueSlot
        self._slots: dict[int, QueueSlot] = {}
        # (side, price) -> list[QueueSlot] in FIFO order
        self._queues: dict[tuple[Side, int], list[QueueSlot]] = {}

    def add_order(
        self,
        order_id: int,
        agent_id: int,
        side: Side,
        price: int,
        qty: int,
        qty_ahead: int,
        ts: int,
    ) -> QueueSlot:
        """Register a new resting limit order with its current queue priority."""
        if order_id in self._slots:
            raise ValueError(f"order_id {order_id} already registered in queue tracker")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if qty_ahead < 0:
            raise ValueError(f"qty_ahead cannot be negative, got {qty_ahead}")

        slot = QueueSlot(
            order_id=order_id,
            agent_id=agent_id,
            side=side,
            price=price,
            qty=qty,
            qty_remaining=qty,
            qty_ahead=qty_ahead,
            ts_placed=ts,
        )
        self._slots[order_id] = slot
        self._queues.setdefault((side, price), []).append(slot)
        return slot

    def on_trade(self, side: Side, price: int, trade_qty: int, ts: int) -> list[QueueFill]:
        """Process a trade executing against resting orders at (side, price).

        Args:
            side: The resting side being traded against (e.g. Side.BUY for resting bid fills).
            price: Execution price level.
            trade_qty: Total shares executed in this trade.
            ts: Timestamp of the trade in nanoseconds.

        Returns:
            List of QueueFill executions achieved by our tracked orders.
        """
        key = (side, price)
        queue = self._queues.get(key)
        if not queue or trade_qty <= 0:
            return []

        fills: list[QueueFill] = []
        cumulative_consumed = 0

        # Process queued orders in FIFO sequence
        surviving: list[QueueSlot] = []
        for slot in queue:
            # Update qty_ahead based on trades that occurred earlier at this level
            if cumulative_consumed > 0:
                slot.qty_ahead = max(0, slot.qty_ahead - cumulative_consumed)

            rem_trade = trade_qty - cumulative_consumed
            if rem_trade <= 0:
                surviving.append(slot)
                continue

            # First, trade consumes volume resting ahead of this order
            if slot.qty_ahead > 0:
                consumed_ahead = min(slot.qty_ahead, rem_trade)
                slot.qty_ahead -= consumed_ahead
                cumulative_consumed += consumed_ahead
                rem_trade -= consumed_ahead

            # If volume ahead is exhausted, trade consumes this order
            if rem_trade > 0 and slot.qty_ahead == 0:
                fill_qty = min(slot.qty_remaining, rem_trade)
                slot.qty_remaining -= fill_qty
                cumulative_consumed += fill_qty
                rem_trade -= fill_qty
                fills.append(
                    QueueFill(
                        order_id=slot.order_id,
                        agent_id=slot.agent_id,
                        side=slot.side,
                        price=slot.price,
                        qty=fill_qty,
                        ts=ts,
                    )
                )

            if slot.qty_remaining > 0:
                surviving.append(slot)
            else:
                self._slots.pop(slot.order_id, None)

        if surviving:
            self._queues[key] = surviving
        else:
            self._queues.pop(key, None)

        return fills

    def on_cancel(
        self,
        side: Side,
        price: int,
        cancel_qty: int,
        total_depth_before: int | None = None,
    ) -> None:
        """Process a cancellation event at (side, price)."""
        key = (side, price)
        queue = self._queues.get(key)
        if not queue or cancel_qty <= 0:
            return

        for slot in queue:
            if slot.qty_ahead <= 0:
                continue

            if self.cancellation_model == CancellationModel.CONSERVATIVE:
                # Assume cancellations occurred behind our order; qty_ahead unchanged
                continue
            elif self.cancellation_model == CancellationModel.OPTIMISTIC:
                # Assume cancellations occurred ahead of us
                reduced = min(slot.qty_ahead, cancel_qty)
                slot.qty_ahead -= reduced
            elif self.cancellation_model == CancellationModel.PROPORTIONAL:
                # Proportional to our share of the queue ahead
                depth = (
                    total_depth_before
                    if total_depth_before and total_depth_before > 0
                    else slot.qty_ahead + slot.qty
                )
                ratio = min(1.0, max(0.0, slot.qty_ahead / depth))
                reduced = min(slot.qty_ahead, int(round(cancel_qty * ratio)))
                slot.qty_ahead -= reduced

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a tracked order and remove it from its price queue."""
        slot = self._slots.pop(order_id, None)
        if slot is None:
            return False

        key = (slot.side, slot.price)
        if key in self._queues:
            self._queues[key] = [s for s in self._queues[key] if s.order_id != order_id]
            if not self._queues[key]:
                self._queues.pop(key, None)
        return True

    def get_slot(self, order_id: int) -> QueueSlot | None:
        """Look up a tracked slot by order_id."""
        return self._slots.get(order_id)

    def active_orders(self) -> list[QueueSlot]:
        """Return all active resting orders."""
        return list(self._slots.values())
