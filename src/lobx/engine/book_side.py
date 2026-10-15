"""One side of the book (all bids, or all asks), organised as price levels keyed by price.

A dict gives O(1) access to a level by price, and a sorted list of prices lets us find and walk the
best levels. The list is kept ascending for both sides: the best bid is the *last* price, the best
ask is the *first*. Inserting a new price costs O(n) with ``bisect.insort``; that is fine for the
readable reference engine, and the C++ engine uses an ordered map instead.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections.abc import Iterator

from lobx.engine.price_level import PriceLevel
from lobx.types import Side


class BookSide:
    """All price levels for one side of the book."""

    def __init__(self, side: Side) -> None:
        """Create an empty side. ``side`` decides which end of the price list is "best"."""
        self.side = side
        self._levels: dict[int, PriceLevel] = {}
        self._prices: list[int] = []  # ascending, one entry per non-empty level

    def is_empty(self) -> bool:
        """Return ``True`` when there are no levels on this side."""
        return not self._prices

    def best_price(self) -> int | None:
        """Return the best price (highest bid or lowest ask), or ``None`` if the side is empty."""
        if not self._prices:
            return None
        return self._prices[-1] if self.side is Side.BUY else self._prices[0]

    def best_level(self) -> PriceLevel | None:
        """Return the level at the best price, or ``None`` if the side is empty."""
        price = self.best_price()
        return None if price is None else self._levels[price]

    def get(self, price: int) -> PriceLevel | None:
        """Return the level at ``price`` if it exists."""
        return self._levels.get(price)

    def get_or_create(self, price: int) -> PriceLevel:
        """Return the level at ``price``, creating an empty one if needed."""
        level = self._levels.get(price)
        if level is None:
            level = PriceLevel(price)
            self._levels[price] = level
            insort(self._prices, price)
        return level

    def remove_if_empty(self, price: int) -> None:
        """Drop the level at ``price`` once its last order is gone, so empty levels never linger."""
        level = self._levels.get(price)
        if level is not None and level.is_empty():
            del self._levels[price]
            del self._prices[bisect_left(self._prices, price)]

    def levels_best_first(self) -> Iterator[PriceLevel]:
        """Yield levels from the best price outwards (bids high to low, asks low to high)."""
        prices = reversed(self._prices) if self.side is Side.BUY else iter(self._prices)
        for price in prices:
            yield self._levels[price]
