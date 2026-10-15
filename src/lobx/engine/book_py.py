"""The Python reference matching engine.

This is the readable implementation of the ``OrderBook`` protocol. It favours clarity over speed:
the C++ engine added in phase 3 must behave identically, and this file is what it is checked
against. The rules it follows are written down in ``docs/contracts/shared-contracts.md``.
"""

from __future__ import annotations

from itertools import islice

from lobx.engine.book_side import BookSide
from lobx.engine.price_level import RestingOrder
from lobx.types import BookSnapshot, Fill, Order, OrderType, Side


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

        A limit order first trades against any resting orders it crosses; whatever is left rests
        in the book at its limit price. Market orders are added in a later commit.
        """
        if order.type is not OrderType.LIMIT or order.price is None:
            raise NotImplementedError("market orders are added in a later commit")
        fills, remaining = self._match(order, order.price)
        if remaining > 0:
            self._rest(order, order.price, remaining)
        self._last_ts = order.ts
        return fills

    def best_bid(self) -> int | None:
        """Return the highest resting buy price, or ``None`` if there are no bids."""
        return self._bids.best_price()

    def best_ask(self) -> int | None:
        """Return the lowest resting sell price, or ``None`` if there are no asks."""
        return self._asks.best_price()

    def snapshot(self, depth: int) -> BookSnapshot:
        """Return up to ``depth`` aggregated price levels per side, best level first."""
        if depth <= 0:
            raise ValueError(f"depth must be greater than 0, got {depth}")
        return BookSnapshot(
            ts=self._last_ts,
            bids=self._levels_to_tuples(self._bids, depth),
            asks=self._levels_to_tuples(self._asks, depth),
        )

    @staticmethod
    def _levels_to_tuples(side: BookSide, depth: int) -> tuple[tuple[int, int], ...]:
        """Turn the best ``depth`` levels of ``side`` into ``(price, total_qty)`` pairs."""
        best_levels = islice(side.levels_best_first(), depth)
        return tuple((level.price, level.total_qty) for level in best_levels)

    def _match(self, order: Order, limit_price: int | None) -> tuple[list[Fill], int]:
        """Trade ``order`` against the opposite side and return ``(fills, unfilled_qty)``.

        Walks the opposite side from its best price outwards. At each price the earliest resting
        order trades first (price-time priority), and every trade executes at the *resting*
        order's price. ``limit_price`` of ``None`` means "no price limit".
        """
        opposite = self._asks if order.side is Side.BUY else self._bids
        fills: list[Fill] = []
        remaining = order.qty
        while remaining > 0:
            level = opposite.best_level()
            if level is None or not self._crosses(order.side, limit_price, level.price):
                break
            traded = min(remaining, level.front().qty)
            maker = level.fill_front(traded)
            fills.append(
                Fill(
                    maker_order_id=maker.order_id,
                    taker_order_id=order.order_id,
                    taker_side=order.side,
                    price=level.price,
                    qty=traded,
                    ts=order.ts,
                )
            )
            remaining -= traded
            if maker.qty == 0:
                del self._orders[maker.order_id]
                opposite.remove_if_empty(level.price)
        return fills, remaining

    @staticmethod
    def _crosses(taker_side: Side, limit_price: int | None, level_price: int) -> bool:
        """Return ``True`` if an incoming order is willing to trade at ``level_price``."""
        if limit_price is None:
            return True
        if taker_side is Side.BUY:
            return level_price <= limit_price  # a buyer accepts any ask at or below its limit
        return level_price >= limit_price  # a seller accepts any bid at or above its limit

    def _side_for(self, side: Side) -> BookSide:
        """Return the book side an order of ``side`` rests on (buys rest as bids)."""
        return self._bids if side is Side.BUY else self._asks

    def _rest(self, order: Order, price: int, qty: int) -> None:
        """Place ``qty`` of ``order`` in the book at ``price`` behind earlier orders there."""
        resting = RestingOrder(order.order_id, order.agent_id, order.side, price, qty, order.ts)
        self._side_for(order.side).get_or_create(price).add(resting)
        self._orders[order.order_id] = resting
