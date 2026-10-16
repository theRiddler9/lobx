"""Property tests: random order streams must never break the book.

Hypothesis generates many random streams of limit orders, market orders and cancels. Two kinds of
check run after every single operation:

1. The engine must agree, fill for fill and level for level, with ``_NaiveBook``: a deliberately
   slow model that re-searches a flat list on every trade, so it is obviously correct.
2. The invariants in docs/contracts/shared-contracts.md must hold: the book is never crossed,
   levels are sorted and never empty, and a fill never exceeds its order or its limit price.

The prices are squeezed into 95..105 so that orders collide and cross often; with a wide range
most random orders would just rest and nothing interesting would be tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lobx.engine.book_py import PythonOrderBook
from lobx.types import Fill, Order, OrderType, Side

_ALL_LEVELS = 10_000

# One generated operation is (kind, side, number, qty). For "limit", number is the price. For
# "cancel", number only picks which earlier order id to cancel. "limit" appears twice so there are
# more resting orders than market orders eating them.
_operations = st.lists(
    st.tuples(
        st.sampled_from(["limit", "limit", "market", "cancel"]),
        st.sampled_from([Side.BUY, Side.SELL]),
        st.integers(min_value=95, max_value=105),
        st.integers(min_value=1, max_value=10),
    ),
    max_size=80,
)


def _build_steps(operations: list[tuple[str, Side, int, int]]) -> list[tuple[str, Any, int]]:
    """Turn raw operations into ``("submit", Order, ts)`` and ``("cancel", order_id, ts)`` steps.

    Order ids count up from 1 and ``ts`` is the step number, so time never goes backwards.
    """
    steps: list[tuple[str, Any, int]] = []
    next_id = 1
    for ts, (kind, side, number, qty) in enumerate(operations):
        if kind == "cancel":
            # Before any order exists, id 1 is unknown, which is a fine thing to cancel too.
            target = 1 + number % max(next_id - 1, 1)
            steps.append(("cancel", target, ts))
            continue
        order_type = OrderType.LIMIT if kind == "limit" else OrderType.MARKET
        price = number if order_type is OrderType.LIMIT else None
        steps.append(("submit", Order(next_id, next_id % 3, side, order_type, price, qty, ts), ts))
        next_id += 1
    return steps


@dataclass
class _Resting:
    order_id: int
    side: Side
    price: int
    qty: int
    seq: int  # arrival number: lower means earlier, which is what time priority means


class _NaiveBook:
    """The simplest correct matching engine: one flat list, searched from scratch every time."""

    def __init__(self) -> None:
        self.resting: list[_Resting] = []
        self._seq = 0

    def submit(self, order: Order) -> list[tuple[int, int, int, int]]:
        """Return fills as ``(maker_id, taker_id, price, qty)``."""
        fills: list[tuple[int, int, int, int]] = []
        remaining = order.qty
        # A buyer wants the lowest price first, a seller the highest. Negating the price for
        # sellers lets one ``min`` pick the best price and, on ties, the earliest arrival.
        sign = 1 if order.side is Side.BUY else -1
        while remaining > 0:
            candidates = [
                r for r in self.resting if r.side is not order.side and self._crosses(order, r)
            ]
            if not candidates:
                break
            best = min(candidates, key=lambda r: (sign * r.price, r.seq))
            traded = min(remaining, best.qty)
            best.qty -= traded
            remaining -= traded
            fills.append((best.order_id, order.order_id, best.price, traded))
            if best.qty == 0:
                self.resting.remove(best)
        if remaining > 0 and order.type is OrderType.LIMIT:
            assert order.price is not None
            self._seq += 1
            self.resting.append(
                _Resting(order.order_id, order.side, order.price, remaining, self._seq)
            )
        return fills

    def cancel(self, order_id: int) -> bool:
        for resting in self.resting:
            if resting.order_id == order_id:
                self.resting.remove(resting)
                return True
        return False

    def levels(self, side: Side) -> tuple[tuple[int, int], ...]:
        """Aggregate quantity per price, best price first."""
        totals: dict[int, int] = {}
        for r in self.resting:
            if r.side is side:
                totals[r.price] = totals.get(r.price, 0) + r.qty
        return tuple(sorted(totals.items(), reverse=side is Side.BUY))

    @staticmethod
    def _crosses(order: Order, resting: _Resting) -> bool:
        if order.type is OrderType.MARKET:
            return True
        assert order.price is not None
        if order.side is Side.BUY:
            return resting.price <= order.price
        return resting.price >= order.price


def _check_fills(order: Order, fills: list[Fill]) -> None:
    assert sum(f.qty for f in fills) <= order.qty
    for fill in fills:
        assert fill.qty > 0
        assert fill.taker_order_id == order.order_id
        assert fill.taker_side is order.side
        assert fill.ts == order.ts
        if order.type is OrderType.LIMIT:
            assert order.price is not None
            # A limit order never trades at a worse price than its limit.
            if order.side is Side.BUY:
                assert fill.price <= order.price
            else:
                assert fill.price >= order.price
    prices = [f.price for f in fills]
    # Fills walk the book outwards: a buyer meets ever higher asks, a seller ever lower bids.
    assert prices == sorted(prices, reverse=order.side is Side.SELL)


def _check_book(engine: PythonOrderBook, model: _NaiveBook) -> None:
    snapshot = engine.snapshot(_ALL_LEVELS)
    assert snapshot.bids == model.levels(Side.BUY)
    assert snapshot.asks == model.levels(Side.SELL)
    for levels in (snapshot.bids, snapshot.asks):
        assert all(qty > 0 for _, qty in levels)
    bid_prices = [price for price, _ in snapshot.bids]
    ask_prices = [price for price, _ in snapshot.asks]
    assert bid_prices == sorted(set(bid_prices), reverse=True)
    assert ask_prices == sorted(set(ask_prices))
    assert engine.best_bid() == (bid_prices[0] if bid_prices else None)
    assert engine.best_ask() == (ask_prices[0] if ask_prices else None)
    if bid_prices and ask_prices:
        assert bid_prices[0] < ask_prices[0], "the book is crossed"


@settings(max_examples=300, deadline=None)
@given(_operations)
def test_engine_matches_naive_model_and_keeps_invariants(
    operations: list[tuple[str, Side, int, int]],
) -> None:
    engine = PythonOrderBook()
    model = _NaiveBook()
    for kind, payload, ts in _build_steps(operations):
        if kind == "cancel":
            assert engine.cancel(payload, ts) == model.cancel(payload)
        else:
            fills = engine.submit(payload)
            assert [
                (f.maker_order_id, f.taker_order_id, f.price, f.qty) for f in fills
            ] == model.submit(payload)
            _check_fills(payload, fills)
        _check_book(engine, model)


@settings(max_examples=100, deadline=None)
@given(_operations)
def test_same_stream_gives_identical_results(
    operations: list[tuple[str, Side, int, int]],
) -> None:
    def run() -> list[object]:
        engine = PythonOrderBook()
        results: list[object] = []
        for kind, payload, ts in _build_steps(operations):
            results.append(
                engine.cancel(payload, ts) if kind == "cancel" else engine.submit(payload)
            )
        results.append(engine.snapshot(_ALL_LEVELS))
        return results

    assert run() == run()


@settings(max_examples=100, deadline=None)
@given(_operations)
def test_a_huge_market_order_empties_the_opposite_side(
    operations: list[tuple[str, Side, int, int]],
) -> None:
    engine = PythonOrderBook()
    steps = _build_steps(operations)
    for kind, payload, ts in steps:
        if kind == "cancel":
            engine.cancel(payload, ts)
        else:
            engine.submit(payload)
    sweep_ts = len(steps)
    engine.submit(Order(10**6, 0, Side.BUY, OrderType.MARKET, None, 10**6, sweep_ts))
    assert engine.best_ask() is None
    assert engine.snapshot(_ALL_LEVELS).asks == ()
