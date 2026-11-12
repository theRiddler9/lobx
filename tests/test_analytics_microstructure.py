"""Tests for microstructure metrics: Stoikov micro-price, VPIN, and spread decomposition."""

from __future__ import annotations

import pytest

from lobx.analytics.microstructure import (
    compute_stoikov_microprice,
    compute_vpin,
    decompose_executions,
)
from lobx.types import BookSnapshot, Fill, Side


def test_stoikov_microprice() -> None:
    """Verify that micro-price shifts towards the side with greater order book pressure."""
    # Balanced book: bid=100 (qty 10), ask=102 (qty 10) -> micro = 101.0
    snap_balanced = BookSnapshot(ts=1_000, bids=((100, 10),), asks=((102, 10),))
    res1 = compute_stoikov_microprice([snap_balanced])
    assert len(res1) == 1
    assert res1[0][1] == pytest.approx(101.0)

    # Buy-heavy book: bid=100 (qty 30), ask=102 (qty 10)
    # micro = (30 * 102 + 10 * 100) / 40 = (3060 + 1000) / 40 = 101.5
    snap_buy_heavy = BookSnapshot(ts=2_000, bids=((100, 30),), asks=((102, 10),))
    res2 = compute_stoikov_microprice([snap_buy_heavy])
    assert res2[0][1] == pytest.approx(101.5)


def test_vpin_calculation() -> None:
    """Verify that VPIN computes expected flow toxicity over volume buckets."""
    # 4 buckets of 50 shares each = 200 shares
    fills = [
        # Bucket 0: 50 buy shares -> imb = 50
        Fill(maker_order_id=1, taker_order_id=2, taker_side=Side.BUY, price=100, qty=50, ts=1_000),
        # Bucket 1: 50 buy shares -> imb = 50
        Fill(maker_order_id=3, taker_order_id=4, taker_side=Side.BUY, price=100, qty=50, ts=2_000),
        # Bucket 2: 50 buy shares -> imb = 50
        Fill(maker_order_id=5, taker_order_id=6, taker_side=Side.BUY, price=100, qty=50, ts=3_000),
        # Bucket 3: 50 buy shares -> imb = 50
        Fill(maker_order_id=7, taker_order_id=8, taker_side=Side.BUY, price=100, qty=50, ts=4_000),
    ]

    # With window_buckets = 2 and bucket_volume = 50, all buy flow -> VPIN = 1.0 (pure toxicity)
    vpin = compute_vpin(fills, bucket_volume=50, window_buckets=2)
    assert len(vpin) == 3
    for pt in vpin:
        assert pt.vpin == pytest.approx(1.0)


def test_decompose_executions() -> None:
    """Verify spread decomposition into effective half-spread, realized spread, and price impact."""
    snaps = [
        BookSnapshot(ts=1_000_000_000, bids=((10000, 10),), asks=((10004, 10),)),  # mid = 10002
        BookSnapshot(
            ts=6_000_000_000, bids=((10006, 10),), asks=((10010, 10),)
        ),  # mid = 10008 (+6 ticks)
    ]
    # Buy market order at 10004
    fill = Fill(
        maker_order_id=1,
        taker_order_id=2,
        taker_side=Side.BUY,
        price=10004,
        qty=10,
        ts=1_000_000_000,
    )

    decomps = decompose_executions([fill], snaps, horizon_ns=5_000_000_000)
    assert len(decomps) == 1
    d = decomps[0]
    # Effective half-spread: (10004 - 10002) / 10002 * 10000 = ~1.9996 bps
    assert d.effective_spread_bps == pytest.approx(1.9996, rel=1e-3)
    # Price impact: (10008 - 10002) / 10002 * 10000 = ~5.9988 bps
    assert d.price_impact_bps == pytest.approx(5.9988, rel=1e-3)
    # Realized spread: eff - impact
    assert d.realized_spread_bps == pytest.approx(d.effective_spread_bps - d.price_impact_bps)
