"""Tests for the resting-order ownership index."""

import pytest

from lobx.engine.order_tracker import OrderTracker
from lobx.errors import SimulationError
from lobx.types import Fill, Order, OrderType, Side


def _limit(order_id: int, agent: int = 1, qty: int = 10, side: Side = Side.BUY) -> Order:
    return Order(order_id, agent, side, OrderType.LIMIT, 100, qty, 0)


def test_unfilled_limit_order_is_tracked() -> None:
    tracker = OrderTracker()
    tracker.on_submit(_limit(1, agent=5, qty=10), [])
    assert tracker.agent_of(1) == 5
    assert tracker.remaining(1) == 10
    assert tracker.open_order_ids(5) == (1,)
    assert len(tracker) == 1


def test_market_order_is_never_tracked() -> None:
    tracker = OrderTracker()
    market = Order(1, 5, Side.BUY, OrderType.MARKET, None, 10, 0)
    tracker.on_submit(market, [])
    assert len(tracker) == 0


def test_partial_fill_reduces_the_maker_and_the_taker_remainder_rests() -> None:
    tracker = OrderTracker()
    tracker.on_submit(_limit(1, agent=1, qty=10, side=Side.SELL), [])
    taker = _limit(2, agent=2, qty=15)
    tracker.on_submit(taker, [Fill(1, 2, Side.BUY, 100, 4, 0)])
    assert tracker.remaining(1) == 6  # maker partly filled, still resting
    assert tracker.remaining(2) == 11  # 15 wanted, 4 traded, 11 now rests


def test_fully_filled_maker_is_forgotten() -> None:
    tracker = OrderTracker()
    tracker.on_submit(_limit(1, agent=1, qty=4, side=Side.SELL), [])
    tracker.on_submit(_limit(2, agent=2, qty=4), [Fill(1, 2, Side.BUY, 100, 4, 0)])
    assert tracker.agent_of(1) is None
    assert tracker.open_order_ids(1) == ()
    assert tracker.agent_of(2) is None  # the taker was filled completely too


def test_cancel_forgets_only_when_the_engine_cancelled_it() -> None:
    tracker = OrderTracker()
    tracker.on_submit(_limit(1), [])
    tracker.on_cancel(1, cancelled=False)
    assert tracker.agent_of(1) == 1
    tracker.on_cancel(1, cancelled=True)
    assert tracker.agent_of(1) is None


def test_open_orders_are_listed_oldest_first_per_agent() -> None:
    tracker = OrderTracker()
    for order_id, agent in [(3, 1), (1, 2), (2, 1)]:
        tracker.on_submit(_limit(order_id, agent=agent), [])
    assert tracker.open_order_ids(1) == (3, 2)
    assert tracker.open_order_ids(2) == (1,)
    assert tracker.open_order_ids(99) == ()


def test_fill_for_an_unknown_maker_is_an_error() -> None:
    with pytest.raises(SimulationError, match="not open"):
        OrderTracker().on_submit(_limit(2), [Fill(1, 2, Side.BUY, 100, 1, 0)])


def test_overfilling_a_maker_is_an_error() -> None:
    tracker = OrderTracker()
    tracker.on_submit(_limit(1, qty=3, side=Side.SELL), [])
    with pytest.raises(SimulationError, match="more than it had"):
        tracker.on_submit(_limit(2), [Fill(1, 2, Side.BUY, 100, 5, 0)])
