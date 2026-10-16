"""The Python reference matching engine.

This is the readable implementation of the ``OrderBook`` protocol. It favours clarity over speed:
the C++ engine added in phase 3 must behave identically, and this file is what it is checked
against. The rules it follows are written down in ``docs/contracts/shared-contracts.md``.
"""

from __future__ import annotations

from itertools import islice

from lobx.engine.book_side import BookSide
from lobx.engine.price_level import RestingOrder
from lobx.errors import DuplicateOrderIdError, InvalidOrderError
from lobx.types import BookSnapshot, Fill, Order, OrderType, Side


def _is_int(value: object) -> bool:
    """Return ``True`` for real integers. ``bool`` is excluded even though it subclasses ``int``."""
    return isinstance(value, int) and not isinstance(value, bool)


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
        in the book at its limit price. A market order trades against whatever is available and
        its unfilled remainder is discarded: market orders never rest.

        Raises ``InvalidOrderError`` or ``DuplicateOrderIdError`` before touching the book if the
        order breaks a validation rule.
        """
        limit_price = self._validate(order)
        fills, remaining = self._match(order, limit_price)
        # Only an order with a price can rest. A market order has none, so its remainder vanishes.
        if remaining > 0 and limit_price is not None:
            self._rest(order, limit_price, remaining)
        self._last_ts = order.ts
        return fills

    def cancel(self, order_id: int, ts: int) -> bool:
        """Remove a resting order. Return ``True`` if it was resting, ``False`` otherwise.

        Unknown, already filled, and already cancelled orders all return ``False``; a cancel
        racing with a fill is normal in a market, so it is not an error.
        """
        self._check_ts(ts)
        self._last_ts = ts
        resting = self._orders.pop(order_id, None)
        if resting is None:
            return False
        side = self._side_for(resting.side)
        level = side.get(resting.price)
        if level is None or level.remove(order_id) is None:
            # Cannot happen unless the id index and the price levels have drifted apart.
            raise RuntimeError(f"order {order_id} is indexed but missing from its price level")
        side.remove_if_empty(resting.price)
        return True

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

    def _validate(self, order: Order) -> int | None:
        """Check every validation rule and return the limit price (``None`` for market orders).

        Everything is checked before the book changes, so a rejected order leaves no trace.
        """
        if not _is_int(order.qty) or order.qty <= 0:
            raise InvalidOrderError(f"qty must be an integer greater than 0, got {order.qty!r}")
        self._check_ts(order.ts)
        if order.type is OrderType.MARKET:
            if order.price is not None:
                raise InvalidOrderError(
                    f"a market order must not have a price, got {order.price!r}"
                )
        elif order.price is None or not _is_int(order.price) or order.price <= 0:
            raise InvalidOrderError(
                f"a limit order needs an integer price greater than 0, got {order.price!r}"
            )
        if order.order_id in self._orders:
            raise DuplicateOrderIdError(f"order id {order.order_id} is already resting in the book")
        return order.price

    def _check_ts(self, ts: int) -> None:
        """Reject timestamps that are not integers or that go backwards in time."""
        if not _is_int(ts) or ts < self._last_ts:
            raise InvalidOrderError(
                f"ts must be an integer not earlier than {self._last_ts} "
                f"(the previous call), got {ts!r}"
            )

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
