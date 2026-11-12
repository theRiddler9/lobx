import pytest

from lobx.sim.queue_position import CancellationModel, QueueTracker
from lobx.types import Side


def test_queue_tracker_fifo_priority() -> None:
    tracker = QueueTracker()

    # Place resting buy order: 200 shares at price 10000 with 500 shares ahead
    slot = tracker.add_order(
        order_id=1,
        agent_id=99,
        side=Side.BUY,
        price=10000,
        qty=200,
        qty_ahead=500,
        ts=1_000_000,
    )
    assert slot.qty_ahead == 500
    assert slot.qty_remaining == 200
    assert not slot.is_filled

    # 1. Trade of 300 shares executes at 10000
    fills1 = tracker.on_trade(side=Side.BUY, price=10000, trade_qty=300, ts=2_000_000)
    assert len(fills1) == 0  # Consumed ahead
    assert slot.qty_ahead == 200
    assert slot.qty_remaining == 200

    # 2. Trade of 150 shares executes
    fills2 = tracker.on_trade(side=Side.BUY, price=10000, trade_qty=150, ts=3_000_000)
    assert len(fills2) == 0
    assert slot.qty_ahead == 50

    # 3. Trade of 100 shares executes: 50 ahead consumed, 50 fills our order!
    fills3 = tracker.on_trade(side=Side.BUY, price=10000, trade_qty=100, ts=4_000_000)
    assert len(fills3) == 1
    f3 = fills3[0]
    assert f3.order_id == 1
    assert f3.qty == 50
    assert f3.price == 10000
    assert slot.qty_ahead == 0
    assert slot.qty_remaining == 150

    # 4. Trade of 200 shares executes: completes remaining 150!
    fills4 = tracker.on_trade(side=Side.BUY, price=10000, trade_qty=200, ts=5_000_000)
    assert len(fills4) == 1
    f4 = fills4[0]
    assert f4.order_id == 1
    assert f4.qty == 150
    assert slot.is_filled
    assert tracker.get_slot(1) is None
    assert len(tracker.active_orders()) == 0


def test_queue_tracker_multiple_orders_at_same_price() -> None:
    tracker = QueueTracker()

    # Order A at front of queue
    tracker.add_order(
        order_id=1, agent_id=10, side=Side.SELL, price=10020, qty=100, qty_ahead=0, ts=100
    )
    # Order B behind Order A
    tracker.add_order(
        order_id=2, agent_id=20, side=Side.SELL, price=10020, qty=200, qty_ahead=100, ts=200
    )

    # Trade of 150 arrives
    fills = tracker.on_trade(side=Side.SELL, price=10020, trade_qty=150, ts=300)
    assert len(fills) == 2

    # Order 1 filled completely (100)
    assert fills[0].order_id == 1
    assert fills[0].qty == 100

    # Order 2 filled partially (50)
    assert fills[1].order_id == 2
    assert fills[1].qty == 50

    # Check remaining
    slot_b = tracker.get_slot(2)
    assert slot_b is not None
    assert slot_b.qty_remaining == 150
    assert slot_b.qty_ahead == 0


def test_queue_tracker_cancellation_models() -> None:
    # 1. Conservative
    t_cons = QueueTracker(cancellation_model=CancellationModel.CONSERVATIVE)
    s_cons = t_cons.add_order(
        order_id=1, agent_id=1, side=Side.BUY, price=5000, qty=100, qty_ahead=400, ts=1
    )
    t_cons.on_cancel(side=Side.BUY, price=5000, cancel_qty=200, total_depth_before=1000)
    assert s_cons.qty_ahead == 400  # unchanged

    # 2. Optimistic
    t_opt = QueueTracker(cancellation_model=CancellationModel.OPTIMISTIC)
    s_opt = t_opt.add_order(
        order_id=2, agent_id=1, side=Side.BUY, price=5000, qty=100, qty_ahead=400, ts=1
    )
    t_opt.on_cancel(side=Side.BUY, price=5000, cancel_qty=150, total_depth_before=1000)
    assert s_opt.qty_ahead == 250  # 400 - 150

    # 3. Proportional
    t_prop = QueueTracker(cancellation_model=CancellationModel.PROPORTIONAL)
    s_prop = t_prop.add_order(
        order_id=3, agent_id=1, side=Side.BUY, price=5000, qty=100, qty_ahead=400, ts=1
    )
    # Depth before is 1000, qty_ahead is 400 (40% of queue)
    # Cancel of 200 -> reduces qty_ahead by ~80 (40% of 200)
    t_prop.on_cancel(side=Side.BUY, price=5000, cancel_qty=200, total_depth_before=1000)
    assert s_prop.qty_ahead == 320


def test_queue_tracker_cancel_order() -> None:
    tracker = QueueTracker()
    tracker.add_order(
        order_id=5, agent_id=1, side=Side.BUY, price=10000, qty=100, qty_ahead=50, ts=1
    )
    assert tracker.get_slot(5) is not None

    # Cancel existing
    assert tracker.cancel_order(5) is True
    assert tracker.get_slot(5) is None

    # Cancel nonexistent
    assert tracker.cancel_order(999) is False


def test_queue_tracker_validation() -> None:
    tracker = QueueTracker()
    tracker.add_order(order_id=1, agent_id=1, side=Side.BUY, price=100, qty=10, qty_ahead=0, ts=1)

    # Duplicate id
    with pytest.raises(ValueError, match="already registered"):
        tracker.add_order(
            order_id=1, agent_id=1, side=Side.BUY, price=100, qty=10, qty_ahead=0, ts=1
        )

    # Invalid qty
    with pytest.raises(ValueError, match="qty must be positive"):
        tracker.add_order(
            order_id=2, agent_id=1, side=Side.BUY, price=100, qty=0, qty_ahead=0, ts=1
        )

    # Negative qty_ahead
    with pytest.raises(ValueError, match="qty_ahead cannot be negative"):
        tracker.add_order(
            order_id=3, agent_id=1, side=Side.BUY, price=100, qty=10, qty_ahead=-5, ts=1
        )
