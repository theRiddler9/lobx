"""Differential tests: the C++ engine must match the Python reference exactly.

Both engines get the same operations; every fill, cancel result, best price and snapshot must be
equal. The Python engine is the specification, so any difference is a bug in the C++ engine.
"""

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.native import NativeOrderBook, native_available
from lobx.engine.protocol import OrderBook
from lobx.engine.streams import CancelOp, random_stream
from lobx.errors import DuplicateOrderIdError, InvalidOrderError
from lobx.types import Order, OrderType, Side

pytestmark = pytest.mark.skipif(not native_available(), reason="C++ engine not built")


def _run_both(seed: int, count: int, check_every: int) -> int:
    """Feed one stream to both engines, comparing as we go. Return the number of fills."""
    py: OrderBook = PythonOrderBook()
    cpp: OrderBook = NativeOrderBook()
    total_fills = 0
    for i, op in enumerate(random_stream(seed, count)):
        if isinstance(op, CancelOp):
            assert py.cancel(op.order_id, op.ts) == cpp.cancel(op.order_id, op.ts), (seed, i, op)
        else:
            expected = py.submit(op)
            assert cpp.submit(op) == expected, (seed, i, op)
            total_fills += len(expected)
        if i % check_every == 0:
            assert py.best_bid() == cpp.best_bid(), (seed, i)
            assert py.best_ask() == cpp.best_ask(), (seed, i)
            assert py.snapshot(5) == cpp.snapshot(5), (seed, i)
    assert py.snapshot(1000) == cpp.snapshot(1000)
    return total_fills


@pytest.mark.parametrize("seed", range(20))
def test_many_short_streams(seed: int) -> None:
    assert _run_both(seed, 5_000, check_every=1) > 0


def test_one_million_random_orders() -> None:
    """The Phase 3 exit criterion: one million operations, identical results."""
    assert _run_both(seed=12345, count=1_000_000, check_every=1_000) > 100_000


def _mutate(order: Order, **changes: object) -> Order:
    fields = {
        "order_id": order.order_id,
        "agent_id": order.agent_id,
        "side": order.side,
        "type": order.type,
        "price": order.price,
        "qty": order.qty,
        "ts": order.ts,
    }
    fields.update(changes)
    return Order(**fields)  # type: ignore[arg-type]


def test_invalid_calls_raise_the_same_error_type_and_change_nothing() -> None:
    base = Order(1, 1, Side.BUY, OrderType.LIMIT, 100, 5, 10)
    bad_orders = [
        _mutate(base, order_id=2, qty=0),
        _mutate(base, order_id=2, qty=-1),
        _mutate(base, order_id=2, price=0),
        _mutate(base, order_id=2, price=-5),
        _mutate(base, order_id=2, price=None),
        _mutate(base, order_id=2, type=OrderType.MARKET),  # a market order with a price
        _mutate(base, order_id=2, ts=9),  # time going backwards
        _mutate(base, order_id=1, ts=11),  # duplicate id
        _mutate(base, order_id=1, ts=1),  # ts error wins over the duplicate id
        _mutate(base, order_id=1, qty=0, ts=1),  # qty error wins over both
    ]
    for factory in (PythonOrderBook, NativeOrderBook):
        book = factory()
        book.submit(base)
        outcomes = []
        for bad in bad_orders:
            with pytest.raises(InvalidOrderError) as info:
                book.submit(bad)
            outcomes.append(type(info.value))
        with pytest.raises(InvalidOrderError):
            book.cancel(1, 9)
        assert DuplicateOrderIdError in outcomes
        assert outcomes.count(DuplicateOrderIdError) == 1
        assert book.snapshot(5).ts == 10
        assert book.snapshot(5).bids == ((100, 5),)


def test_snapshot_depth_must_be_positive() -> None:
    for depth in (0, -1):
        with pytest.raises(ValueError, match="depth"):
            NativeOrderBook().snapshot(depth)
