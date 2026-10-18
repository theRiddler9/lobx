"""A strategy that follows a fixed schedule: how much of the parent order should be done by when.

TWAP and VWAP differ only in the *weights* they give the time slices, so both are built on this
class. The parent window is cut into equal-length slices; slice ``k`` has a weight ``w[k]``, and by
the end of slice ``k`` the strategy wants ``qty * (w[0] + ... + w[k]) / sum(w)`` shares done.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import accumulate

from lobx.sim.actions import Action, Submit
from lobx.sim.agents.base import MarketView
from lobx.strategies.base import ParentOrder


class ScheduledStrategy:
    """Send market orders so the cumulative executed quantity follows a weighted schedule."""

    def __init__(self, weights: Sequence[float]) -> None:
        """Create a schedule with one non-negative weight per time slice."""
        if not weights:
            raise ValueError("a schedule needs at least one slice")
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("slice weights must be non-negative and not all zero")
        self._weights = tuple(weights)
        self._parent: ParentOrder | None = None
        self._targets: tuple[int, ...] = ()

    @property
    def num_slices(self) -> int:
        """Return how many time slices the schedule has."""
        return len(self._weights)

    def reset(self, parent: ParentOrder) -> None:
        """Compute the cumulative target quantity at the end of each slice."""
        self._parent = parent
        total = sum(self._weights)
        # Targets are cumulative *rounded* quantities, and each child order is the difference of two
        # consecutive targets. Rounding the running total (rather than each slice) means rounding
        # errors never pile up, and the last target is exactly the parent quantity.
        targets = [round(parent.qty * done / total) for done in accumulate(self._weights)]
        targets[-1] = parent.qty
        self._targets = tuple(targets)

    def step(self, ts: int, view: MarketView) -> list[Action]:
        """Return a market order for whatever is behind schedule at time ``ts``."""
        parent = self._parent
        if parent is None:
            raise RuntimeError("call reset(parent) before step()")
        if ts < parent.start_ns:
            return []
        slice_ns = (parent.end_ns - parent.start_ns) // self.num_slices
        index = min((ts - parent.start_ns) // slice_ns, self.num_slices - 1)
        # Comparing with what has *actually* traded (not with what we sent) means a shortfall from
        # a thin book is carried into the next slice, and calling step twice in one slice is
        # harmless: the second call finds nothing behind schedule.
        behind = self._targets[index] - parent.executed(view)
        if behind <= 0:
            return []
        return [Submit.market(parent.side, behind)]
