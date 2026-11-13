"""Loader and Python wrapper for the C++ matching engine.

``NativeOrderBook`` implements the same ``OrderBook`` protocol as ``PythonOrderBook`` but the
matching runs in C++ (``cpp/``). The wrapper converts between the contract dataclasses and the
plain ints and tuples the extension speaks, so the rest of LOBX cannot tell the engines apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

from lobx.errors import NativeEngineUnavailableError
from lobx.types import BookSnapshot, Fill, Order, OrderType, Side

if TYPE_CHECKING:
    from lobx._native import NativeBook, NativeMultiBook

_native: ModuleType | None
_IMPORT_ERROR: ImportError | None
try:
    from lobx import _native  # the extension is compiled by `pip install -e .`

    _IMPORT_ERROR = None
except ImportError as _exc:
    _native = None
    _IMPORT_ERROR = _exc


def native_available() -> bool:
    """Return ``True`` if the compiled C++ engine can be imported."""
    return _native is not None


class NativeOrderBook:
    """The C++ order book behind the ``OrderBook`` protocol.

    Validation rules and error types match ``PythonOrderBook``. One difference: the extension
    only accepts real ints, so a wrongly typed field (a float quantity, say) raises ``TypeError``
    from pybind11 instead of ``InvalidOrderError``. Range errors (qty <= 0, price <= 0, time going
    backwards, duplicate ids) raise the same exceptions from both engines.
    """

    def __init__(self) -> None:
        """Create an empty book, or raise ``NativeEngineUnavailableError`` if it is not built."""
        if _native is None:
            raise NativeEngineUnavailableError(
                "the C++ engine is not built; run `pip install -e .` (needs a C++17 compiler and "
                f"CMake). Import error: {_IMPORT_ERROR}"
            )
        self._book: NativeBook = _native.NativeBook()

    _TYPE_MAP = {
        OrderType.LIMIT: 0,
        OrderType.MARKET: 1,
        OrderType.IOC: 2,
        OrderType.FOK: 3,
        OrderType.POST_ONLY: 4,
    }

    def submit(self, order: Order) -> list[Fill]:
        """Apply ``order`` and return the fills it produced, in execution order."""
        raw = self._book.submit(
            order.order_id,
            order.agent_id,
            0 if order.side is Side.BUY else 1,
            self._TYPE_MAP.get(order.type, 0),
            order.price,
            order.qty,
            order.ts,
        )
        return [
            Fill(maker, taker, Side.BUY if side == 0 else Side.SELL, price, qty, ts)
            for maker, taker, side, price, qty, ts in raw
        ]

    def submit_many(self, orders: list[Order]) -> list[Fill]:
        """Submit a batch of orders to the C++ engine without per-order GIL overhead.

        The GIL is released for the entire C++ matching loop via
        ``py::call_guard<py::gil_scoped_release>()`` in bindings.cpp.  Python
        only pays the boundary cost once per batch instead of once per order.

        Orders are processed in list order (identical to calling submit() one
        at a time).  Fills from all orders are concatenated and returned.

        This method is equivalent to::

            fills = []
            for o in orders:
                fills.extend(book.submit(o))

        but with substantially lower boundary overhead for large batches.
        """
        # Build the flat tuple list the C++ extension expects: each element is
        # (order_id, agent_id, side_int, type_int, price_or_None, qty, ts).
        # Side 0 = BUY, 1 = SELL (matches the C++ enum values in types.hpp).
        py_orders = [
            (
                o.order_id,
                o.agent_id,
                0 if o.side is Side.BUY else 1,
                self._TYPE_MAP.get(o.type, 0),
                o.price,
                o.qty,
                o.ts,
            )
            for o in orders
        ]
        # submit_many releases the GIL inside C++ and returns a list of fill
        # tuples: (maker_id, taker_id, taker_side_int, price, qty, ts).
        raw = self._book.submit_many(py_orders)
        return [
            Fill(maker, taker, Side.BUY if side == 0 else Side.SELL, price, qty, ts)
            for maker, taker, side, price, qty, ts in raw
        ]

    def cancel(self, order_id: int, ts: int) -> bool:
        """Remove a resting order; return ``True`` if it was resting."""
        return self._book.cancel(order_id, ts)

    def snapshot(self, depth: int) -> BookSnapshot:
        """Return up to ``depth`` aggregated levels per side, best first."""
        if depth <= 0:
            raise ValueError(f"depth must be greater than 0, got {depth}")
        ts, bids, asks = self._book.snapshot(depth)
        return BookSnapshot(ts=ts, bids=tuple(bids), asks=tuple(asks))

    def best_bid(self) -> int | None:
        """Return the highest resting bid price, or ``None`` if there are no bids."""
        return self._book.best_bid()

    def best_ask(self) -> int | None:
        """Return the lowest resting ask price, or ``None`` if there are no asks."""
        return self._book.best_ask()


class NativeMultiOrderBook:
    """The C++ multi-book router wrapped for python interoperability."""

    def __init__(self, symbols: Sequence[str] = ()) -> None:
        """Initialize NativeMultiOrderBook with optional symbols."""
        if _native is None:
            raise NativeEngineUnavailableError(
                "the C++ engine is not built; run `pip install -e .` (needs a C++17 compiler and "
                f"CMake). Import error: {_IMPORT_ERROR}"
            )
        self._mb: NativeMultiBook = _native.NativeMultiBook(list(symbols))

    _TYPE_MAP = {
        OrderType.LIMIT: 0,
        OrderType.MARKET: 1,
        OrderType.IOC: 2,
        OrderType.FOK: 3,
        OrderType.POST_ONLY: 4,
    }

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return all registered symbols."""
        return tuple(self._mb.symbols())

    def add_symbol(self, symbol: str) -> None:
        """Register a new symbol book."""
        self._mb.add_symbol(symbol)

    def has_symbol(self, symbol: str) -> bool:
        """Check if symbol is registered."""
        return self._mb.has_symbol(symbol)

    def submit(self, symbol: str, order: Order) -> list[Fill]:
        """Route order submission to specific symbol in C++ engine."""
        raw = self._mb.submit(
            symbol,
            order.order_id,
            order.agent_id,
            0 if order.side is Side.BUY else 1,
            self._TYPE_MAP.get(order.type, 0),
            order.price,
            order.qty,
            order.ts,
        )
        return [
            Fill(maker, taker, Side.BUY if side == 0 else Side.SELL, price, qty, ts)
            for maker, taker, side, price, qty, ts in raw
        ]

    def cancel(self, symbol: str, order_id: int, ts: int) -> bool:
        """Route cancel request to specific symbol in C++ engine."""
        return self._mb.cancel(symbol, order_id, ts)

    def snapshot(self, symbol: str, depth: int = 5) -> BookSnapshot:
        """Fetch depth-K snapshot for symbol."""
        if depth <= 0:
            raise ValueError(f"depth must be greater than 0, got {depth}")
        ts, bids, asks = self._mb.snapshot(symbol, depth)
        return BookSnapshot(ts=ts, bids=tuple(bids), asks=tuple(asks))

    def best_bid(self, symbol: str) -> int | None:
        """Fetch best bid price for symbol."""
        return self._mb.best_bid(symbol)

    def best_ask(self, symbol: str) -> int | None:
        """Fetch best ask price for symbol."""
        return self._mb.best_ask(symbol)

    def mid_price(self, symbol: str) -> float | None:
        """Fetch current mid price for symbol."""
        return self._mb.mid_price(symbol)

    def imbalance(self, symbol: str, depth: int = 1) -> float:
        """Compute depth-K imbalance for symbol."""
        return self._mb.imbalance(symbol, depth)

    def all_snapshots(self, depth: int = 5) -> dict[str, BookSnapshot]:
        """Fetch depth-K snapshots across all symbols."""
        return {sym: self.snapshot(sym, depth) for sym in self.symbols}
