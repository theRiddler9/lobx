"""Loader and Python wrapper for the C++ matching engine.

``NativeOrderBook`` implements the same ``OrderBook`` protocol as ``PythonOrderBook`` but the
matching runs in C++ (``cpp/``). The wrapper converts between the contract dataclasses and the
plain ints and tuples the extension speaks, so the rest of LOBX cannot tell the engines apart.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

from lobx.errors import NativeEngineUnavailableError
from lobx.types import BookSnapshot, Fill, Order, OrderType, Side

if TYPE_CHECKING:
    from lobx._native import NativeBook

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

    def submit(self, order: Order) -> list[Fill]:
        """Apply ``order`` and return the fills it produced, in execution order."""
        raw = self._book.submit(
            order.order_id,
            order.agent_id,
            0 if order.side is Side.BUY else 1,
            0 if order.type is OrderType.LIMIT else 1,
            order.price,
            order.qty,
            order.ts,
        )
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
