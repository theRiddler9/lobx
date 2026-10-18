"""Estimate how traded volume is spread over time, from past runs (the input VWAP needs)."""

from __future__ import annotations

from collections.abc import Sequence


def estimate_volume_profile(
    runs: Sequence[Sequence[tuple[int, int]]], start_ns: int, end_ns: int, buckets: int
) -> tuple[float, ...]:
    """Return the average share of volume traded in each of ``buckets`` equal slices of a window.

    ``runs`` holds one list of ``(ts, qty)`` trades per past run. Each run's bucket volumes are
    turned into shares of that run's total *before* averaging, so a busy run does not outweigh a
    quiet one. Runs with no trades in the window are skipped. The result sums to 1.

    Only past data is used: the caller passes training seeds, never the seeds being evaluated.
    """
    if buckets <= 0 or end_ns <= start_ns:
        raise ValueError("need buckets > 0 and end_ns > start_ns")
    width = (end_ns - start_ns) / buckets
    shares = [0.0] * buckets
    used = 0
    for trades in runs:
        volume = [0] * buckets
        for ts, qty in trades:
            if start_ns <= ts < end_ns:
                volume[min(int((ts - start_ns) / width), buckets - 1)] += qty
        total = sum(volume)
        if total == 0:
            continue
        used += 1
        for i, v in enumerate(volume):
            shares[i] += v / total
    if used == 0:
        raise ValueError("no trades in the window: cannot estimate a volume profile")
    return tuple(s / used for s in shares)
