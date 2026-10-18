"""An agent that trades nothing and only writes down the book at regular times.

It lets data-collection code (the ML dataset) see a run the way any participant would: through
the ``MarketView``, at evenly spaced instants.
"""

from __future__ import annotations

from dataclasses import replace

from numpy.random import Generator

from lobx.sim.actions import Action
from lobx.sim.agents.base import MarketView
from lobx.types import BookSnapshot


class SnapshotRecorder:
    """Record ``depth`` levels of the book every ``interval_ns``, starting at ``interval_ns``."""

    def __init__(self, agent_id: int, interval_ns: int, depth: int) -> None:
        """Create a recorder; ``snapshots`` fills up as the market runs."""
        if interval_ns <= 0 or depth <= 0:
            raise ValueError("interval_ns and depth must be greater than 0")
        self.agent_id = agent_id
        self.snapshots: list[BookSnapshot] = []
        self._interval = interval_ns
        self._depth = depth

    def first_wake(self, rng: Generator) -> int:
        """Return the first recording time."""
        return self._interval

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Store the book as of ``ts`` and trade nothing.

        The engine stamps a snapshot with the time of the last order or cancel, which can be
        earlier than now. The snapshot is re-stamped with ``ts`` so that every one is exactly one
        interval after the previous (feature windows depend on that).
        """
        self.snapshots.append(replace(view.snapshot(self._depth), ts=ts))
        return []

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        """Return the next recording time."""
        return ts + self._interval
