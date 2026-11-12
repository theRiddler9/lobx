"""Tests for depth heatmap generation and TradingView HTML rendering."""

from __future__ import annotations

from lobx.analytics.bars import CandleBar
from lobx.analytics.heatmap import build_depth_heatmap
from lobx.analytics.tradingview import render_tradingview_html
from lobx.types import BookSnapshot, Fill, Side


def test_build_depth_heatmap() -> None:
    """Verify that heatmap correctly samples levels and prices."""
    snaps = [
        BookSnapshot(ts=1_000_000_000, bids=((100, 5), (99, 10)), asks=((101, 8), (102, 12))),
        BookSnapshot(ts=2_000_000_000, bids=((100, 6), (99, 9)), asks=((101, 7), (102, 14))),
    ]
    heatmap = build_depth_heatmap(snaps, sample_stride=1, max_depth_per_side=2)
    assert len(heatmap) == 2
    assert heatmap[0].best_bid == 100
    assert heatmap[0].best_ask == 101
    assert len(heatmap[0].levels) == 4  # 2 bids + 2 asks


def test_render_tradingview_html() -> None:
    """Verify that TradingView HTML generation builds valid markup with embedded JSON data."""
    bars = [
        CandleBar(
            time=1_700_000_000,
            start_ns=0,
            end_ns=1_000_000_000,
            open=10000,
            high=10010,
            low=9990,
            close=10005,
            volume=50,
            buy_volume=30,
            sell_volume=20,
            volume_delta=10,
            cvd=10,
            trade_count=5,
            vwap=10002.5,
        )
    ]
    fills = [
        Fill(
            maker_order_id=1,
            taker_order_id=2,
            taker_side=Side.BUY,
            price=10005,
            qty=10,
            ts=500_000_000,
        )
    ]
    microprices = [(500_000_000, 10003.5)]

    html = render_tradingview_html(bars, fills=fills, microprices=microprices, title="Test Chart")
    assert "<!DOCTYPE html>" in html
    assert "LightweightCharts.createChart" in html
    assert "candlestickSeries" in html
    assert "Test Chart" in html
    assert "100.05" in html  # Tick scaled price
