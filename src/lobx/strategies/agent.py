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

    def __init__(
        self,
        strategy: ExecutionStrategy,
        parent: ParentOrder,
        num_slices: int,
        wake_interval_ns: int | None = None,
        lookback_ns: int = 0,
    ) -> None:
        """Wrap ``strategy`` to work ``parent``.

        By default the agent wakes once per slice (``num_slices`` evenly spaced wakes). A strategy
        that must watch the market continuously passes ``wake_interval_ns`` to wake that often
        instead, starting ``lookback_ns`` *before* the window so it can build up history; it is
        trusted to send nothing before ``parent.start_ns``.
        """
        if num_slices <= 0:
            raise ValueError(f"num_slices must be greater than 0, got {num_slices}")
        if wake_interval_ns is not None and wake_interval_ns <= 0:
            raise ValueError(f"wake_interval_ns must be greater than 0, got {wake_interval_ns}")
        if lookback_ns < 0:
            raise ValueError(f"lookback_ns must not be negative, got {lookback_ns}")
        self.agent_id = parent.agent_id
        self.parent = parent
        self.arrival_mid2: int | None = None
        self._strategy = strategy
        self._lookback = lookback_ns if wake_interval_ns is not None else 0
        self._interval = (
            wake_interval_ns
            if wake_interval_ns is not None
            else (parent.end_ns - parent.start_ns) // num_slices
        )
        if self._interval <= 0:
            raise ValueError("the parent window is too short for that many slices")
        strategy.reset(parent)

    def first_wake(self, rng: Generator) -> int:
        """Return the start of the parent window (earlier by the lookback, if any)."""
        return max(0, self.parent.start_ns - self._lookback)

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Record the arrival price on the first wake, then return the strategy's child orders."""
        if self.arrival_mid2 is None and ts >= self.parent.start_ns:
            mid2 = view.mid2()
            self.arrival_mid2 = mid2 if mid2 is not None else 2 * view.reference_price()
        return self._strategy.step(ts, view)

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        """Return the next slice start, or ``None`` after the last slice."""
        following = ts + self._interval
        return following if following < self.parent.end_ns else None
