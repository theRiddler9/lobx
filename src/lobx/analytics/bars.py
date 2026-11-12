"""Candlestick bar aggregation from raw limit order book execution tapes.

Aggregates discrete trades into uniform OHLCV bars with buy/sell volume delta
and Cumulative Volume Delta (CVD) compatible with TradingView Lightweight Charts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lobx.types import Fill, Side


@dataclass(frozen=True, slots=True)
class CandleBar:
    """An aggregated candlestick bar over a fixed time duration."""

    time: int  # Bar timestamp in seconds (UNIX or relative elapsed seconds)
    start_ns: int
    end_ns: int
    open: int  # Ticks
    high: int  # Ticks
    low: int  # Ticks
    close: int  # Ticks
    volume: int
    buy_volume: int
    sell_volume: int
    volume_delta: int
    cvd: int
    trade_count: int
    vwap: float

    def to_tradingview(self, tick_scale: float = 1.0) -> dict[str, Any]:
        """Convert bar to TradingView Lightweight Charts format."""
        return {
            "time": self.time,
            "open": self.open * tick_scale,
            "high": self.high * tick_scale,
            "low": self.low * tick_scale,
            "close": self.close * tick_scale,
            "volume": self.volume,
            "buyVolume": self.buy_volume,
            "sellVolume": self.sell_volume,
            "delta": self.volume_delta,
            "cvd": self.cvd,
            "vwap": self.vwap * tick_scale,
        }


def aggregate_bars(
    fills: Sequence[Fill],
    bar_duration_ns: int = 1_000_000_000,
    base_epoch_s: int = 1_700_000_000,
    opening_price: int = 10_000,
) -> list[CandleBar]:
    """Aggregate a sequence of trade fills into uniform time bars.

    Args:
        fills: Sequence of Fill objects in timestamp order.
        bar_duration_ns: Duration of each bar in nanoseconds (default 1s = 1e9 ns).
        base_epoch_s: Base UNIX epoch in seconds for charting timeline.
        opening_price: Fallback price in ticks for the first bar if no trades occur early.

    Returns:
        List of CandleBar objects covering the active trade duration.
    """
    if bar_duration_ns <= 0:
        raise ValueError("bar_duration_ns must be strictly positive")

    if not fills:
        return []

    min_ts = fills[0].ts
    max_ts = fills[-1].ts

    start_bucket = (min_ts // bar_duration_ns) * bar_duration_ns
    end_bucket = ((max_ts // bar_duration_ns) + 1) * bar_duration_ns

    # Group fills into bucket intervals
    bucket_map: dict[int, list[Fill]] = {}
    for f in fills:
        b_idx = (f.ts // bar_duration_ns) * bar_duration_ns
        bucket_map.setdefault(b_idx, []).append(f)

    bars: list[CandleBar] = []
    last_close = fills[0].price if fills else opening_price
    running_cvd = 0

    curr = start_bucket
    while curr < end_bucket:
        bucket_fills = bucket_map.get(curr, [])
        bar_time = base_epoch_s + int(curr // 1_000_000_000)

        if bucket_fills:
            o = bucket_fills[0].price
            c = bucket_fills[-1].price
            h = max(f.price for f in bucket_fills)
            low = min(f.price for f in bucket_fills)
            vol = sum(f.qty for f in bucket_fills)
            buy_vol = sum(f.qty for f in bucket_fills if f.taker_side is Side.BUY)
            sell_vol = sum(f.qty for f in bucket_fills if f.taker_side is Side.SELL)
            delta = buy_vol - sell_vol
            running_cvd += delta

            # VWAP = sum(price * qty) / sum(qty)
            dollar_vol = sum(f.price * f.qty for f in bucket_fills)
            vwap = dollar_vol / vol if vol > 0 else float(c)

            bar = CandleBar(
                time=bar_time,
                start_ns=curr,
                end_ns=curr + bar_duration_ns,
                open=o,
                high=h,
                low=low,
                close=c,
                volume=vol,
                buy_volume=buy_vol,
                sell_volume=sell_vol,
                volume_delta=delta,
                cvd=running_cvd,
                trade_count=len(bucket_fills),
                vwap=vwap,
            )
            last_close = c
        else:
            # Empty interval carries forward previous close
            bar = CandleBar(
                time=bar_time,
                start_ns=curr,
                end_ns=curr + bar_duration_ns,
                open=last_close,
                high=last_close,
                low=last_close,
                close=last_close,
                volume=0,
                buy_volume=0,
                sell_volume=0,
                volume_delta=0,
                cvd=running_cvd,
                trade_count=0,
                vwap=float(last_close),
            )

        bars.append(bar)
        curr += bar_duration_ns

    return bars
