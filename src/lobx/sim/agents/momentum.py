"""The momentum trader: buys what has been rising, sells what has been falling.

It compares a short moving average of the mid-price with a longer one. When the short average is
clearly above the long one, the price has been trending up and it buys at market; when clearly
below, it sells. This is the simplest trend-following rule. Momentum traders add *feedback* to the
market (their buying pushes the price up, which makes them buy more), which is one source of the
bursts of volatility real markets show.
"""

from __future__ import annotations

from collections import deque

from numpy.random import Generator

from lobx.sim.actions import Action, Submit
from lobx.sim.agents.base import MarketView
from lobx.sim.agents.timing import jittered_gap_ns
from lobx.types import Side


class MomentumTrader:
    """A trend follower that trades at market when a short average crosses a long average."""

    def __init__(
        self,
        agent_id: int,
        interval_ns: int,
        short_window: int = 3,
        long_window: int = 12,
        threshold_ticks: float = 1.0,
        qty: int = 5,
        max_position: int = 100,
    ) -> None:
        """Create a momentum trader.

        It looks at the mid-price every ``interval_ns`` (give or take some jitter), keeps the last
        ``long_window`` observations, and trades ``qty`` shares when the average of the last
        ``short_window`` differs from the average of all of them by more than ``threshold_ticks``.
        It never holds more than ``max_position`` shares long or short.
        """
        if interval_ns < 1 or qty < 1 or max_position < 1 or threshold_ticks < 0.0:
            raise ValueError("interval, qty and max_position must be >= 1 and threshold >= 0")
        if not 1 <= short_window < long_window:
            raise ValueError("windows must satisfy 1 <= short_window < long_window")
        self.agent_id = agent_id
        self.interval_ns = interval_ns
        self.short_window = short_window
        self.long_window = long_window
        # mid2 is in half-ticks, so a threshold in ticks is twice as big in mid2 units.
        self.threshold_mid2 = 2.0 * threshold_ticks
        self.qty = qty
        self.max_position = max_position
        self._history: deque[int] = deque(maxlen=long_window)

    def first_wake(self, rng: Generator) -> int:
        """Return the first turn: one jittered interval after time 0."""
        return jittered_gap_ns(rng, self.interval_ns)

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        """Return the next turn, one jittered interval after ``ts``. Never retires."""
        return ts + jittered_gap_ns(rng, self.interval_ns)

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Record the mid-price and trade if the short average has crossed the long one."""
        mid2 = view.mid2()
        if mid2 is None:
            return []  # one side of the book is empty: there is no price to follow
        self._history.append(mid2)
        if len(self._history) < self.long_window:
            return []  # not enough history yet
        values = list(self._history)
        short_average = sum(values[-self.short_window :]) / self.short_window
        long_average = sum(values) / len(values)
        signal = short_average - long_average
        position = view.position(self.agent_id)
        if signal > self.threshold_mid2 and position < self.max_position:
            return [Submit.market(Side.BUY, min(self.qty, self.max_position - position))]
        if signal < -self.threshold_mid2 and position > -self.max_position:
            return [Submit.market(Side.SELL, min(self.qty, self.max_position + position))]
        return []
