"""One price level of the book: a FIFO queue of resting orders at a single price.

Price-time priority means that at a given price the order that arrived first trades first. An
``OrderedDict`` keyed by order id gives us exactly that: iteration order is insertion order (the
queue), and removing an order from the middle (a cancel) is O(1).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from lobx.types import Side


@dataclass(slots=True)
class RestingOrder:
    """A limit order sitting in the book.

    Unlike ``lobx.types.Order`` this is mutable: ``qty`` is the quantity still unfilled and shrinks
    as the order trades. It never leaves the engine; callers only see ``Fill`` objects.
    """

    order_id: int
    agent_id: int
    side: Side
    price: int
    qty: int
    ts: int


class PriceLevel:
    """All resting orders at one price, in arrival order."""

    __slots__ = ("_orders", "price", "total_qty")

    def __init__(self, price: int) -> None:
        """Create an empty level at ``price``."""
        self.price = price
        # Sum of remaining qty of every order in the level. We keep it up to date on every change
        # so that building a snapshot never has to walk the queue.
        self.total_qty = 0
        self._orders: OrderedDict[int, RestingOrder] = OrderedDict()

    def __len__(self) -> int:
        """Return the number of resting orders at this level."""
        return len(self._orders)

    def is_empty(self) -> bool:
        """Return ``True`` when no orders rest at this level."""
        return not self._orders

    def add(self, order: RestingOrder) -> None:
        """Append ``order`` to the back of the queue (it has the lowest time priority)."""
        self._orders[order.order_id] = order
        self.total_qty += order.qty

    def front(self) -> RestingOrder:
        """Return the order with the highest time priority without removing it."""
        # next(iter(...)) is O(1); popitem(last=False) would remove it, which we do not want yet.
        return next(iter(self._orders.values()))

    def fill_front(self, qty: int) -> RestingOrder:
        """Trade ``qty`` against the front order and return that order.

        If the front order is fully consumed it is removed from the queue, so after this call
        ``order.qty == 0`` tells the caller the order is gone.
        """
        order = self.front()
        order.qty -= qty
        self.total_qty -= qty
        if order.qty == 0:
            del self._orders[order.order_id]
        return order

    def remove(self, order_id: int) -> RestingOrder | None:
        """Remove an order from anywhere in the queue (used by cancel). ``None`` if absent."""
        order = self._orders.pop(order_id, None)
        if order is not None:
            self.total_qty -= order.qty
        return order
