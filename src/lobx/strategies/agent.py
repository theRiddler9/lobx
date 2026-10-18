"""Adapter that lets an ``ExecutionStrategy`` run inside the market as an ordinary ``Agent``."""

from __future__ import annotations

from numpy.random import Generator

from lobx.sim.actions import Action
from lobx.sim.agents.base import MarketView
from lobx.strategies.base import ExecutionStrategy, ParentOrder


class StrategyAgent:
    """Wake once per slice, between ``parent.start_ns`` and ``parent.end_ns``, and ask the strategy.

    The first wake also records the *arrival mid-price* (twice the mid, to stay an integer): the
    price the market showed just before the first child order. Implementation shortfall is measured
    against it.
    """

    def __init__(self, strategy: ExecutionStrategy, parent: ParentOrder, num_slices: int) -> None:
        """Wrap ``strategy`` to work ``parent`` in ``num_slices`` evenly spaced wakes."""
        if num_slices <= 0:
            raise ValueError(f"num_slices must be greater than 0, got {num_slices}")
        self.agent_id = parent.agent_id
        self.parent = parent
        self.arrival_mid2: int | None = None
        self._strategy = strategy
        self._interval = (parent.end_ns - parent.start_ns) // num_slices
        if self._interval <= 0:
            raise ValueError("the parent window is too short for that many slices")
        strategy.reset(parent)

    def first_wake(self, rng: Generator) -> int:
        """Return the start of the parent window."""
        return self.parent.start_ns

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Record the arrival price on the first wake, then return the strategy's child orders."""
        if self.arrival_mid2 is None:
            mid2 = view.mid2()
            self.arrival_mid2 = mid2 if mid2 is not None else 2 * view.reference_price()
        return self._strategy.step(ts, view)

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        """Return the next slice start, or ``None`` after the last slice."""
        following = ts + self._interval
        return following if following < self.parent.end_ns else None
