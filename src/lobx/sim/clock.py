"""Simulated time and the queue that decides whose turn it is.

Nothing in the simulator reads the real clock. Time is an integer number of nanoseconds that only
moves when the market takes the next entry from the ``WakeQueue``. That makes a run a pure function
of its inputs: it takes the same simulated time whether the computer is fast or slow.
"""

from __future__ import annotations

import heapq

from lobx.errors import SimulationError

NS_PER_US = 1_000
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


class Clock:
    """The current simulation time, in nanoseconds. It can only move forward."""

    def __init__(self) -> None:
        """Start at time 0."""
        self._now = 0

    @property
    def now(self) -> int:
        """Return the current time in nanoseconds."""
        return self._now

    def advance_to(self, ts: int) -> None:
        """Move the clock to ``ts``. Staying at the same time is fine; going back is an error."""
        if ts < self._now:
            raise SimulationError(f"time cannot go backwards: now is {self._now}, asked for {ts}")
        self._now = ts


class WakeQueue:
    """Agents waiting for their turn, earliest first.

    Entries with the same time come out in the order they were pushed. That tie-break is what keeps
    a run deterministic: without it two agents due at the same instant could swap places from one
    run to the next.
    """

    def __init__(self) -> None:
        """Create an empty queue."""
        # Heap entries are (time, push counter, agent id). The counter is the tie-break.
        self._heap: list[tuple[int, int, int]] = []
        self._pushed = 0

    def push(self, ts: int, agent_id: int) -> None:
        """Schedule ``agent_id`` to wake at time ``ts``."""
        heapq.heappush(self._heap, (ts, self._pushed, agent_id))
        self._pushed += 1

    def pop(self) -> tuple[int, int]:
        """Remove and return ``(time, agent_id)`` for the earliest entry."""
        ts, _, agent_id = heapq.heappop(self._heap)
        return ts, agent_id

    def peek_ts(self) -> int | None:
        """Return the time of the earliest entry without removing it, or ``None`` if empty."""
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        """Return the number of waiting entries."""
        return len(self._heap)

    def __bool__(self) -> bool:
        """Return ``True`` if anything is waiting."""
        return bool(self._heap)
