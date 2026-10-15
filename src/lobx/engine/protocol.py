"""The ``OrderBook`` protocol that every matching engine implements.

Both the readable Python reference engine and the fast C++ engine satisfy this interface. The
rules each method must follow (matching, validation, snapshot layout) are written down in
``docs/contracts/shared-contracts.md``; this file only fixes the shape of the calls.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lobx.types import BookSnapshot, Fill, Order


@runtime_checkable
class OrderBook(Protocol):
    """A limit order book with price-time priority."""

    def submit(self, order: Order) -> list[Fill]:
        """Apply ``order`` and return the fills it produced, in execution order.

        Raises ``InvalidOrderError`` or ``DuplicateOrderIdError`` for bad input.
        """
        ...

    def cancel(self, order_id: int, ts: int) -> bool:
        """Remove a resting order. Return ``True`` if it was resting, ``False`` otherwise."""
        ...

    def snapshot(self, depth: int) -> BookSnapshot:
        """Return up to ``depth`` aggregated price levels per side, best level first."""
        ...

    def best_bid(self) -> int | None:
        """Return the highest resting buy price, or ``None`` if there are no bids."""
        ...

    def best_ask(self) -> int | None:
        """Return the lowest resting sell price, or ``None`` if there are no asks."""
        ...
