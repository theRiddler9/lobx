"""What the model is asked to predict: which way the mid-price moves over the next few snapshots.

A label looks *forward* (that is its job), a feature must never do so. Keeping them in separate
files, and building them separately, makes that difference hard to forget.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from lobx.types import BookSnapshot

DOWN, FLAT, UP = 0, 1, 2
CLASS_NAMES = ("down", "flat", "up")
INVALID = -1  # no label: the mid-price is unknown now or at the horizon


def make_labels(snapshots: Sequence[BookSnapshot], horizon: int) -> npt.NDArray[np.int8]:
    """Return, for each snapshot t, whether the mid moved down, stayed, or moved up by t + horizon.

    ``horizon`` is a number of snapshots (a time span, since snapshots are evenly spaced). The mid
    is compared exactly (best bid + best ask, an integer), so any change at all counts as a move.
    The last ``horizon`` snapshots have nothing to compare with, and a snapshot whose book was
    one-sided at t or t + horizon has no mid: all of these are ``INVALID``.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be greater than 0, got {horizon}")
    mid2 = np.array(
        [s.bids[0][0] + s.asks[0][0] if s.bids and s.asks else np.nan for s in snapshots],
        dtype=np.float64,
    )
    labels = np.full(len(snapshots), INVALID, dtype=np.int8)
    if len(snapshots) <= horizon:
        return labels
    change = mid2[horizon:] - mid2[:-horizon]  # NaN wherever either end is unknown
    known = ~np.isnan(change)
    labels[:-horizon][known] = (np.sign(change[known]) + 1).astype(np.int8)
    return labels
