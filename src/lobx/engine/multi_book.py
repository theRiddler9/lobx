"""Multi-symbol order book router and container.

Manages multiple independent OrderBook instances keyed by ticker symbol, providing
centralized order routing, cross-book snapshots, and microstructure metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.protocol import OrderBook
from lobx.types import BookSnapshot, Fill, Order


class UnknownSymbolError(KeyError):
    """Raised when an operation targets an unregistered symbol."""


class MultiOrderBook:
    """Manages independent OrderBook instances keyed by ticker symbol."""

    def __init__(
        self,
        symbols: Sequence[str] = (),
        book_factory: Callable[[], OrderBook] | None = None,
    ) -> None:
        """Initialize MultiOrderBook with optional symbols and book factory."""
        self._factory: Callable[[], OrderBook] = book_factory or PythonOrderBook
        self._books: dict[str, OrderBook] = {}
        for s in symbols:
            self.add_symbol(s)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return all registered symbols."""
        return tuple(self._books.keys())

    def add_symbol(self, symbol: str) -> None:
        """Register a new symbol book if not already present."""
        if symbol not in self._books:
            self._books[symbol] = self._factory()

    def get_book(self, symbol: str) -> OrderBook:
        """Return the underlying OrderBook for the given symbol."""
        try:
            return self._books[symbol]
        except KeyError as err:
            raise UnknownSymbolError(f"Symbol '{symbol}' not found in MultiOrderBook") from err

    def submit(self, symbol: str, order: Order) -> list[Fill]:
        """Route order submission to the specific symbol book."""
        return self.get_book(symbol).submit(order)

    def cancel(self, symbol: str, order_id: int, ts: int) -> bool:
        """Route cancel request to the specific symbol book."""
        return self.get_book(symbol).cancel(order_id, ts)

    def snapshot(self, symbol: str, depth: int = 5) -> BookSnapshot:
        """Fetch book snapshot for a specific symbol."""
        return self.get_book(symbol).snapshot(depth)

    def best_bid(self, symbol: str) -> int | None:
        """Fetch best bid price for a specific symbol."""
        return self.get_book(symbol).best_bid()

    def best_ask(self, symbol: str) -> int | None:
        """Fetch best ask price for a specific symbol."""
        return self.get_book(symbol).best_ask()

    def mid_price(self, symbol: str) -> float | None:
        """Compute the current mid-price for the symbol, or None if book is one-sided or empty."""
        bid = self.best_bid(symbol)
        ask = self.best_ask(symbol)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def spread(self, symbol: str) -> int | None:
        """Compute the bid-ask spread in ticks, or None if book is one-sided or empty."""
        bid = self.best_bid(symbol)
        ask = self.best_ask(symbol)
        if bid is None or ask is None:
            return None
        return ask - bid

    def imbalance(self, symbol: str, depth: int = 1) -> float:
        """Compute order book depth imbalance at depth K: (Q_bid - Q_ask)/(Q_bid + Q_ask)."""
        snap = self.snapshot(symbol, depth=depth)
        bid_vol = sum(qty for _, qty in snap.bids[:depth])
        ask_vol = sum(qty for _, qty in snap.asks[:depth])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / float(total)

    def all_snapshots(self, depth: int = 5) -> dict[str, BookSnapshot]:
        """Fetch depth-K snapshots across all registered symbols."""
        return {sym: book.snapshot(depth) for sym, book in self._books.items()}
