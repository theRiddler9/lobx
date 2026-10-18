"""Turn a run's book snapshots and trades into one row of numbers per snapshot.

The rule that matters most: **the features in row t may only use data with timestamp <= t.**
Using anything later is *lookahead* (data leakage): the model would look brilliant in testing and
be useless in a live market. ``tests/test_features_causal.py`` checks the rule by recomputing each
row from only the data available at that time.

Each feature is computed by a small function over whole arrays, so the file reads top to bottom as a
list of definitions. Missing information (an empty side of the book, not enough history yet) is
``NaN``, never a made-up number; the dataset builder drops those rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from lobx.types import BookSnapshot, Fill, Side

FloatArray = npt.NDArray[np.float64]

RETURN_WINDOWS = (1, 5, 25)  # in snapshots
FLOW_WINDOWS_NS = (1_000_000_000, 5_000_000_000)  # 1 s and 5 s

FEATURE_NAMES: tuple[str, ...] = (
    "spread",
    "imbalance_1",
    "imbalance_5",
    *(f"mid_return_{k}" for k in RETURN_WINDOWS),
    *(f"trade_flow_{w // 1_000_000_000}s" for w in FLOW_WINDOWS_NS),
)


def _best_prices(snapshots: Sequence[BookSnapshot]) -> tuple[FloatArray, FloatArray]:
    """Return the best bid and best ask price of every snapshot (``NaN`` for an empty side)."""
    bid = np.array([s.bids[0][0] if s.bids else np.nan for s in snapshots], dtype=np.float64)
    ask = np.array([s.asks[0][0] if s.asks else np.nan for s in snapshots], dtype=np.float64)
    return bid, ask


def spread(snapshots: Sequence[BookSnapshot]) -> FloatArray:
    """Return best ask minus best bid, in ticks. A wide spread means expensive, thin trading."""
    bid, ask = _best_prices(snapshots)
    return ask - bid


def _imbalance(snapshots: Sequence[BookSnapshot], depth: int) -> FloatArray:
    """Return (bid qty - ask qty) / (bid qty + ask qty) over the best ``depth`` levels.

    The value is in [-1, 1]. Near +1 there is far more resting buy interest than sell interest, so
    the next price move tends to be up: sellers run out of depth first. ``NaN`` if either side is
    empty (no such evidence exists).
    """
    out = np.full(len(snapshots), np.nan)
    for i, snap in enumerate(snapshots):
        bid_qty = sum(q for _, q in snap.bids[:depth])
        ask_qty = sum(q for _, q in snap.asks[:depth])
        if bid_qty > 0 and ask_qty > 0:
            out[i] = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    return out


def _mid_returns(snapshots: Sequence[BookSnapshot], steps: int) -> FloatArray:
    """Return how far the mid-price moved over the last ``steps`` snapshots, in ticks.

    Row t is ``(mid[t] - mid[t - steps])``; the first ``steps`` rows have no history and are NaN.
    Recent momentum (or reversal) is one of the oldest signals in short-term price prediction.
    """
    bid, ask = _best_prices(snapshots)
    mid = (bid + ask) / 2
    out = np.full(len(snapshots), np.nan)
    if len(snapshots) > steps:
        out[steps:] = mid[steps:] - mid[:-steps]
    return out


def _trade_flow(
    snapshots: Sequence[BookSnapshot], fills: Sequence[Fill], window_ns: int
) -> FloatArray:
    """Return (buy volume - sell volume) / total volume over the trailing ``window_ns``.

    A trade is *buyer-initiated* when the taker was buying (it lifted a resting ask). Heavy
    buyer-initiated flow says aggressive buyers are active. Row t counts trades with timestamp in
    ``(ts[t] - window_ns, ts[t]]``, so nothing after t is touched. With no trades in the window the
    value is 0.0: "no imbalance" is a true statement, not a missing one.
    """
    fill_ts = np.array([f.ts for f in fills], dtype=np.int64)
    if len(fill_ts) > 1 and np.any(np.diff(fill_ts) < 0):
        raise ValueError("fills must be sorted by timestamp")
    signed = np.array([f.qty if f.taker_side is Side.BUY else -f.qty for f in fills], dtype=float)
    size = np.array([f.qty for f in fills], dtype=float)
    # Running totals with a leading 0, so the sum over fills[lo:hi] is total[hi] - total[lo].
    signed_total = np.concatenate(([0.0], np.cumsum(signed)))
    size_total = np.concatenate(([0.0], np.cumsum(size)))
    now = np.array([s.ts for s in snapshots], dtype=np.int64)
    hi = np.searchsorted(fill_ts, now, side="right")  # fills with ts <= now
    lo = np.searchsorted(fill_ts, now - window_ns, side="right")  # fills with ts <= now - window
    volume = size_total[hi] - size_total[lo]
    net = signed_total[hi] - signed_total[lo]
    flow: FloatArray = np.divide(net, volume, out=np.zeros(len(now)), where=volume > 0)
    return flow


def extract_features(snapshots: Sequence[BookSnapshot], fills: Sequence[Fill]) -> FloatArray:
    """Return an array of shape ``(len(snapshots), len(FEATURE_NAMES))``; row t uses data <= t.

    ``snapshots`` must be in time order and ``fills`` sorted by timestamp.
    """
    columns = [
        spread(snapshots),
        _imbalance(snapshots, 1),
        _imbalance(snapshots, 5),
        *(_mid_returns(snapshots, k) for k in RETURN_WINDOWS),
        *(_trade_flow(snapshots, fills, w) for w in FLOW_WINDOWS_NS),
    ]
    return np.column_stack(columns) if snapshots else np.empty((0, len(FEATURE_NAMES)))
