"""Turn a run's book snapshots and trades into one row of numbers per snapshot.

The rule that matters most: the features in row t may only use data with timestamp <= t.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from lobx.types import BookSnapshot, Fill

FloatArray = npt.NDArray[np.float64]

FEATURE_NAMES: tuple[str, ...] = ("spread",)


def _best_prices(snapshots: Sequence[BookSnapshot]) -> tuple[FloatArray, FloatArray]:
    """Return the best bid and best ask price of every snapshot (NaN for an empty side)."""
    bid = np.array([s.bids[0][0] if s.bids else np.nan for s in snapshots], dtype=np.float64)
    ask = np.array([s.asks[0][0] if s.asks else np.nan for s in snapshots], dtype=np.float64)
    return bid, ask


def spread(snapshots: Sequence[BookSnapshot]) -> FloatArray:
    """Return best ask minus best bid, in ticks. A wide spread means expensive, thin trading."""
    bid, ask = _best_prices(snapshots)
    return ask - bid


def extract_features(snapshots: Sequence[BookSnapshot], fills: Sequence[Fill]) -> FloatArray:
    """Return an array of shape (len(snapshots), len(FEATURE_NAMES)); row t uses data <= t."""
    columns = [
        spread(snapshots),
    ]
    return np.column_stack(columns) if snapshots else np.empty((0, len(FEATURE_NAMES)))
