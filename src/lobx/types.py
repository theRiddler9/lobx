"""Shared data types that cross component boundaries.

These are plain frozen dataclasses with no behavior and no validation. Validation lives in the
engine (see ``docs/contracts/shared-contracts.md``) so that the Python and C++ engines can apply
exactly the same rules.

Conventions used everywhere:

* prices are integers in ticks (never floats),
* quantities are integers,
* time is an integer count of nanoseconds since simulation start.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    """Which side of the book an order is on."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """How an order behaves when it reaches the engine.

    A ``LIMIT`` order has a price and may rest in the book. A ``MARKET`` order has no price and
    never rests: whatever it cannot fill immediately is discarded.
    """

    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass(frozen=True, slots=True)
class Order:
    """An instruction submitted to the matching engine.

    ``price`` is ``None`` for market orders and an integer tick price for limit orders.
    """

    order_id: int
    agent_id: int
    side: Side
    type: OrderType
    price: int | None
    qty: int
    ts: int


@dataclass(frozen=True, slots=True)
class Fill:
    """One trade between a resting (maker) order and an incoming (taker) order.

    ``price`` is always the maker's price: trades execute at the resting order's price.
    ``taker_side`` is the side of the incoming order, i.e. who was the aggressor.
    """

    maker_order_id: int
    taker_order_id: int
    taker_side: Side
    price: int
    qty: int
    ts: int


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Aggregated book depth at one moment.

    Each level is ``(price, total_qty)``. ``bids`` run from the highest price down and ``asks``
    from the lowest price up, so index 0 on each side is the best level.
    """

    ts: int
    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
