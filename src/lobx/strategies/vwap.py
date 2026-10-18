"""VWAP: volume-weighted average price. Trade more when the market trades more.

Slice sizes follow an estimated volume profile, so the order is a roughly constant share of market
volume. In a market whose volume is flat over time the profile is flat and VWAP behaves like TWAP;
that is a property of the simulated market, not a bug (see ``docs/results.md``).
"""

from __future__ import annotations

from collections.abc import Sequence

from lobx.strategies.schedule import ScheduledStrategy


class VwapStrategy(ScheduledStrategy):
    """Execute the parent order in slices proportional to ``profile``."""

    def __init__(self, profile: Sequence[float]) -> None:
        """Create a VWAP schedule; ``profile[k]`` is the expected volume share of slice ``k``."""
        super().__init__(profile)
