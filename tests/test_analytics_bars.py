"""Tests for candlestick bar aggregation, OHLCV, volume delta, and CVD."""

from __future__ import annotations

import pytest

from lobx.analytics.bars import aggregate_bars
from lobx.types import Fill, Side


def test_aggregate_bars_basic() -> None:
    """Verify that trades inside a bucket aggregate into correct OHLCV values."""
    fills = [
        Fill(
            maker_order_id=1,
            taker_order_id=2,
            taker_side=Side.BUY,
            price=100,
            qty=10,
            ts=100_000_000,
        ),
        Fill(
            maker_order_id=3,
            taker_order_id=4,
            taker_side=Side.BUY,
            price=105,
            qty=20,
            ts=300_000_000,
        ),
        Fill(
            maker_order_id=5,
            taker_order_id=6,
            taker_side=Side.SELL,
            price=98,
            qty=5,
            ts=500_000_000,
        ),
        Fill(
            maker_order_id=7,
            taker_order_id=8,
            taker_side=Side.BUY,
            price=102,
            qty=15,
            ts=800_000_000,
        ),
    ]

    bars = aggregate_bars(fills, bar_duration_ns=1_000_000_000, base_epoch_s=1_000)
    assert len(bars) == 1
    b = bars[0]
    assert b.open == 100
    assert b.high == 105
    assert b.low == 98
    assert b.close == 102
    assert b.volume == 50
    assert b.buy_volume == 45
    assert b.sell_volume == 5
    assert b.volume_delta == 40
    assert b.cvd == 40
    assert b.trade_count == 4


def test_aggregate_bars_empty_gap_forwarding() -> None:
    """Verify that an empty interval carries forward previous close with zero volume."""
    fills = [
        Fill(
            maker_order_id=1,
            taker_order_id=2,
            taker_side=Side.BUY,
            price=100,
            qty=10,
            ts=500_000_000,
        ),
        # 1.0s - 2.0s is empty
        Fill(
            maker_order_id=3,
            taker_order_id=4,
            taker_side=Side.SELL,
            price=102,
            qty=10,
            ts=2_500_000_000,
        ),
    ]

    bars = aggregate_bars(fills, bar_duration_ns=1_000_000_000, base_epoch_s=1_000)
    assert len(bars) == 3

    # Bar 0 (0-1s)
    assert bars[0].close == 100
    assert bars[0].volume == 10

    # Bar 1 (1-2s): empty
    assert bars[1].open == 100
    assert bars[1].close == 100
    assert bars[1].volume == 0
    assert bars[1].cvd == bars[0].cvd

    # Bar 2 (2-3s)
    assert bars[2].close == 102
    assert bars[2].volume == 10


def test_aggregate_bars_invalid_duration() -> None:
    """Verify that non-positive duration raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        aggregate_bars([], bar_duration_ns=0)
