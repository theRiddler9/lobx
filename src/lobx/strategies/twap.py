"""TWAP: time-weighted average price. Equal slices at equal intervals.

The simplest execution baseline. It ignores the market entirely, so its cost is the price of
trading a fixed, predictable amount at fixed times (Almgren and Chriss: spreading trades out lowers
market impact, but exposes the order to price drift for longer).
"""

from __future__ import annotations

from lobx.strategies.schedule import ScheduledStrategy


class TwapStrategy(ScheduledStrategy):
    """Execute the parent order in ``num_slices`` equal parts."""

    def __init__(self, num_slices: int) -> None:
        """Create a TWAP schedule with ``num_slices`` equal slices."""
        if num_slices <= 0:
            raise ValueError(f"num_slices must be greater than 0, got {num_slices}")
        super().__init__([1.0] * num_slices)
