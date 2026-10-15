"""The Python reference matching engine.

This is the readable implementation of the ``OrderBook`` protocol. It favours clarity over speed:
the C++ engine added in phase 3 must behave identically, and this file is what it is checked
against. The rules it follows are written down in ``docs/contracts/shared-contracts.md``.
"""

from __future__ import annotations

from lobx.engine.book_side import BookSide
from lobx.engine.price_level import RestingOrder
from lobx.types import Fill, Order, OrderType, Side


class PythonOrderBook:
    """A limit order book with price-time priority."""

    def __init__(self) -> None:
        """Create an empty book."""
        self._bids = BookSide(Side.BUY)
        self._asks = BookSide(Side.SELL)
        # Every resting order by id, so a cancel can find its price level in O(1).
        self._orders: dict[int, RestingOrder] = {}
        self._last_ts = 0

    def submit(self, order: Order) -> list[Fill]:
        """Apply ``order`` and return the fills it produced, in execution order.

        Work in progress: for now limit orders are only inserted. Matching is added next.
        """
        if order.type is not OrderType.LIMIT or order.price is None:
            raise NotImplementedError("market orders are added with matching")
        self._rest(order, order.price, order.qty)
        self._last_ts = order.ts
        return []

    def _side_for(self, side: Side) -> BookSide:
        """Return the book side an order of ``side`` rests on (buys rest as bids)."""
        return self._bids if side is Side.BUY else self._asks

    def _rest(self, order: Order, price: int, qty: int) -> None:
        """Place ``qty`` of ``order`` in the book at ``price`` behind earlier orders there."""
        resting = RestingOrder(order.order_id, order.agent_id, order.side, price, qty, order.ts)
        self._side_for(order.side).get_or_create(price).add(resting)
        self._orders[order.order_id] = resting
