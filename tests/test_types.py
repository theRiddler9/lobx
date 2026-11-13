"""Tests for the shared contract types."""

import dataclasses

import pytest

from lobx.types import (
    BookSnapshot,
    CancelEvent,
    Fill,
    FillEvent,
    MultiBookSnapshot,
    Order,
    OrderType,
    Side,
    SubmitEvent,
    SymbolFill,
    SymbolOrder,
)


def test_order_is_frozen() -> None:
    order = Order(1, 7, Side.BUY, OrderType.LIMIT, 100, 5, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.qty = 6  # type: ignore[misc]


def test_market_order_has_no_price() -> None:
    order = Order(2, 7, Side.SELL, OrderType.MARKET, None, 5, 10)
    assert order.price is None


def test_fill_records_taker_side() -> None:
    fill = Fill(maker_order_id=1, taker_order_id=2, taker_side=Side.SELL, price=100, qty=3, ts=10)
    assert fill.taker_side is Side.SELL


def test_snapshot_levels_are_best_first() -> None:
    snap = BookSnapshot(ts=0, bids=((100, 5), (99, 2)), asks=((101, 4), (102, 1)))
    assert snap.bids[0][0] > snap.bids[1][0]
    assert snap.asks[0][0] < snap.asks[1][0]


def test_events_are_frozen_and_comparable() -> None:
    order = Order(1, 7, Side.BUY, OrderType.LIMIT, 100, 5, 0)
    assert SubmitEvent(order) == SubmitEvent(order)
    with pytest.raises(dataclasses.FrozenInstanceError):
        CancelEvent(1, 0, True).cancelled = False  # type: ignore[misc]


def test_fill_event_flags_self_trades() -> None:
    fill = Fill(1, 2, Side.BUY, 100, 3, 10)
    assert FillEvent(fill, self_trade=True).self_trade
    assert not FillEvent(fill, self_trade=False).self_trade


def test_symbol_order_and_fill_frozen() -> None:
    order = Order(1, 7, Side.BUY, OrderType.LIMIT, 100, 5, 0)
    sym_order = SymbolOrder(symbol="SPY", order=order)
    assert sym_order.symbol == "SPY"
    assert sym_order.order.price == 100
    with pytest.raises(dataclasses.FrozenInstanceError):
        sym_order.symbol = "QQQ"  # type: ignore[misc]

    fill = Fill(1, 2, Side.BUY, 100, 5, 0)
    sym_fill = SymbolFill(symbol="QQQ", fill=fill)
    assert sym_fill.symbol == "QQQ"
    assert sym_fill.fill.qty == 5


def test_multi_book_snapshot() -> None:
    snap_spy = BookSnapshot(ts=100, bids=((100, 10),), asks=((101, 10),))
    snap_qqq = BookSnapshot(ts=100, bids=((200, 5),), asks=((202, 5),))
    multi_snap = MultiBookSnapshot(ts=100, snapshots={"SPY": snap_spy, "QQQ": snap_qqq})
    assert multi_snap.ts == 100
    assert "SPY" in multi_snap.snapshots
    assert multi_snap.snapshots["QQQ"].bids[0][0] == 200
