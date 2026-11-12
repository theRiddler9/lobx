"""Order book depth heatmap generator for liquidity visualization and market study.

Extracts resting liquidity distribution across price levels and time slices,
enabling Bookmap-style depth heatmaps of liquidity walls, absorption, and cancellations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lobx.types import BookSnapshot


@dataclass(frozen=True, slots=True)
class HeatmapSnapshot:
    """Aggregated liquidity at one timestamp."""

    time: int  # Seconds
    ts_ns: int
    best_bid: int | None
    best_ask: int | None
    levels: list[dict[str, Any]]  # [{"price": p, "qty": q, "side": "buy"|"sell"}]


def build_depth_heatmap(
    snapshots: Sequence[BookSnapshot],
    sample_stride: int = 1,
    max_depth_per_side: int = 10,
    base_epoch_s: int = 1_700_000_000,
) -> list[HeatmapSnapshot]:
    """Extract a time-series of limit order depth distributions for heatmap visualization.

    Args:
        snapshots: Recorded BookSnapshot objects.
        sample_stride: Stride interval for downsampling high-frequency snapshots.
        max_depth_per_side: Maximum price levels included on each side of the book.
        base_epoch_s: Base timestamp in seconds for timeline visualization.

    Returns:
        List of HeatmapSnapshot objects.
    """
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least 1")

    heatmap_series: list[HeatmapSnapshot] = []

    for idx, snap in enumerate(snapshots):
        if idx % sample_stride != 0:
            continue

        bar_time = base_epoch_s + int(snap.ts // 1_000_000_000)
        best_b = snap.bids[0][0] if snap.bids else None
        best_a = snap.asks[0][0] if snap.asks else None

        level_entries: list[dict[str, Any]] = []
        for p, q in snap.bids[:max_depth_per_side]:
            level_entries.append({"price": p, "qty": q, "side": "buy"})
        for p, q in snap.asks[:max_depth_per_side]:
            level_entries.append({"price": p, "qty": q, "side": "sell"})

        heatmap_series.append(
            HeatmapSnapshot(
                time=bar_time,
                ts_ns=snap.ts,
                best_bid=best_b,
                best_ask=best_a,
                levels=level_entries,
            )
        )

    return heatmap_series
