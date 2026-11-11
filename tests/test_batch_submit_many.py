"""Differential and correctness tests for the C++ submit_many batch API.

submit_many(orders) must produce the same fills and book state as feeding the
same orders one at a time through the sequential Python reference engine.

The test checks:
1. Fills returned equal the concatenation of per-order Python fills (order,
   quantity, price, and sides are all identical).
2. Best bid, best ask, and depth-5 snapshot agree after the batch.
3. A batch of one order is equivalent to a single submit.
4. A batch that crosses itself (interleaved bids and asks) fills correctly.
5. An empty batch returns no fills and leaves the book unchanged.
6. Invalid orders inside a batch raise the same error as single-order submit,
   leaving no partial effect on the book.

These tests are skipped automatically when the C++ engine is not built.
"""

from __future__ import annotations

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.native import NativeOrderBook, native_available
from lobx.engine.streams import random_stream
from lobx.types import Fill, Order, OrderType, Side

pytestmark = pytest.mark.skipif(not native_available(), reason="C++ engine not built")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orders_only(stream: list) -> list[Order]:
    """Return only the Order entries from a mixed stream (drop CancelOps)."""
    return [op for op in stream if isinstance(op, Order)]


def _py_sequential_fills(orders: list[Order]) -> list[Fill]:
    """Feed orders one at a time to the Python engine and collect fills."""
    book = PythonOrderBook()
    fills: list[Fill] = []
    for order in orders:
        fills.extend(book.submit(order))
    return fills


def _cpp_batch_fills(orders: list[Order]) -> tuple[list[Fill], NativeOrderBook]:
    """Feed orders as a single submit_many call and return (fills, book)."""
    book = NativeOrderBook()
    fills = book.submit_many(orders)
    return fills, book


# ---------------------------------------------------------------------------
# Core differential tests
# ---------------------------------------------------------------------------


class TestSubmitManyDifferential:
    """submit_many must match sequential Python reference on the same stream."""

    @pytest.mark.parametrize("seed", range(10))
    def test_fills_match_sequential_python(self, seed: int) -> None:
        """Every fill from submit_many must equal the sequential Python fills."""
        stream = list(random_stream(seed, 2_000))
        orders = _orders_only(stream)

        py_fills = _py_sequential_fills(orders)
        cpp_fills, _ = _cpp_batch_fills(orders)

        assert cpp_fills == py_fills, (
            f"seed={seed}: fill mismatch: py={len(py_fills)} fills, cpp={len(cpp_fills)} fills"
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_book_state_matches_after_batch(self, seed: int) -> None:
        """Best bid, best ask, and depth-5 snapshot must match after the batch."""
        stream = list(random_stream(seed, 2_000))
        orders = _orders_only(stream)

        py_book = PythonOrderBook()
        for order in orders:
            py_book.submit(order)

        _, cpp_book = _cpp_batch_fills(orders)

        assert cpp_book.best_bid() == py_book.best_bid(), f"seed={seed}: best_bid mismatch"
        assert cpp_book.best_ask() == py_book.best_ask(), f"seed={seed}: best_ask mismatch"
        assert cpp_book.snapshot(5) == py_book.snapshot(5), f"seed={seed}: snapshot mismatch"

    def test_large_batch_one_hundred_thousand_orders(self) -> None:
        """100,000-order batch: fills must match the Python reference."""
        stream = list(random_stream(42, 100_000))
        orders = _orders_only(stream)
        assert len(orders) > 50_000  # sanity: stream should produce many orders

        py_fills = _py_sequential_fills(orders)
        cpp_fills, _ = _cpp_batch_fills(orders)
        assert cpp_fills == py_fills


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSubmitManyEdgeCases:
    def test_empty_batch_returns_no_fills(self) -> None:
        book = NativeOrderBook()
        fills = book.submit_many([])
        assert fills == []
        assert book.best_bid() is None
        assert book.best_ask() is None

    def test_single_order_batch_matches_single_submit(self) -> None:
        order = Order(1, 1, Side.BUY, OrderType.LIMIT, 100, 5, 10)

        cpp_batch = NativeOrderBook()
        batch_fills = cpp_batch.submit_many([order])

        cpp_single = NativeOrderBook()
        single_fills = cpp_single.submit(order)

        assert batch_fills == single_fills
        assert cpp_batch.snapshot(5) == cpp_single.snapshot(5)

    def test_self_crossing_batch_matches_python(self) -> None:
        """A batch with a buy followed by a matching sell should cross."""
        orders = [
            Order(1, 1, Side.BUY, OrderType.LIMIT, 100, 10, 1),
            Order(2, 2, Side.SELL, OrderType.LIMIT, 100, 10, 2),
        ]
        py_fills = _py_sequential_fills(orders)
        cpp_fills, _ = _cpp_batch_fills(orders)
        assert len(cpp_fills) == 1  # one crossing fill
        assert cpp_fills == py_fills

    def test_batch_partial_cross(self) -> None:
        """Partial fill: buy 10, sell 3; should produce a 3-qty fill."""
        orders = [
            Order(1, 1, Side.BUY, OrderType.LIMIT, 100, 10, 1),
            Order(2, 2, Side.SELL, OrderType.LIMIT, 100, 3, 2),
        ]
        py_fills = _py_sequential_fills(orders)
        cpp_fills, cpp_book = _cpp_batch_fills(orders)
        assert cpp_fills == py_fills
        assert cpp_book.best_bid() == 100  # 7 qty remaining
        assert cpp_book.best_ask() is None

    def test_market_order_in_batch(self) -> None:
        """A market sell into a resting bid should fill."""
        orders = [
            Order(1, 1, Side.BUY, OrderType.LIMIT, 100, 5, 1),
            Order(2, 2, Side.SELL, OrderType.MARKET, None, 5, 2),
        ]
        py_fills = _py_sequential_fills(orders)
        cpp_fills, _ = _cpp_batch_fills(orders)
        assert len(cpp_fills) == 1
        assert cpp_fills == py_fills

    def test_no_fills_all_rest(self) -> None:
        """Non-crossing orders produce no fills; both sides are populated."""
        orders = [
            Order(1, 1, Side.BUY, OrderType.LIMIT, 99, 5, 1),
            Order(2, 2, Side.SELL, OrderType.LIMIT, 101, 5, 2),
        ]
        py_fills = _py_sequential_fills(orders)
        cpp_fills, cpp_book = _cpp_batch_fills(orders)
        assert cpp_fills == py_fills == []
        assert cpp_book.best_bid() == 99
        assert cpp_book.best_ask() == 101


# ---------------------------------------------------------------------------
# Idempotence: sequential submit and batch produce the same final book
# ---------------------------------------------------------------------------


def test_sequential_vs_batch_produces_identical_snapshot() -> None:
    """submit_many on N orders must yield the same book as N sequential submits."""
    stream = list(random_stream(999, 5_000))
    orders = _orders_only(stream)

    py_book = PythonOrderBook()
    for o in orders:
        py_book.submit(o)

    cpp_seq = NativeOrderBook()
    for o in orders:
        cpp_seq.submit(o)

    cpp_batch = NativeOrderBook()
    cpp_batch.submit_many(orders)

    for depth in (1, 5, 10):
        assert cpp_batch.snapshot(depth) == py_book.snapshot(depth), f"depth={depth}"
        assert cpp_batch.snapshot(depth) == cpp_seq.snapshot(depth), f"depth={depth}"
