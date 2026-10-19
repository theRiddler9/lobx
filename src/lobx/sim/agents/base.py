"""The two protocols every agent is written against: ``Agent`` and ``MarketView``.

The market calls an agent when it is that agent's turn (``wake``). The agent looks at the market
through a read-only ``MarketView``, and answers with a list of actions. It never touches the engine
or other agents directly, which is what keeps a run a pure function of ``(config, seed)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.random import Generator

from lobx.sim.actions import Action
from lobx.types import BookSnapshot, Fill


class MarketView(Protocol):
    """What an agent is allowed to see. Everything here is read-only.

    Prices are integer ticks. ``mid2`` is *twice* the mid-price, so it stays an integer even when
    the best bid and ask are an odd number of ticks apart.
    """

    @property
    def ts(self) -> int:
        """Return the current simulation time in nanoseconds."""
        ...

    def best_bid(self) -> int | None:
        """Return the highest resting buy price, or ``None`` if there are no bids."""
        ...

    def best_ask(self) -> int | None:
        """Return the lowest resting sell price, or ``None`` if there are no asks."""
        ...

    def mid2(self) -> int | None:
        """Return best bid plus best ask (twice the mid-price), or ``None`` if a side is empty."""
        ...

    def last_trade_price(self) -> int | None:
        """Return the price of the most recent trade, or ``None`` before the first trade."""
        ...

    def reference_price(self) -> int:
        """Return the mid-price if both sides exist, else the last trade, else the opening price.

        It is always defined, so an agent has something to anchor its orders to even on an empty
        or one-sided book.
        """
        ...

    def snapshot(self, depth: int) -> BookSnapshot:
        """Return up to ``depth`` aggregated price levels per side, best first."""
        ...

    def position(self, agent_id: int) -> int:
        """Return how many shares ``agent_id`` is long (positive) or short (negative)."""
        ...

    def open_orders(self, agent_id: int) -> tuple[int, ...]:
        """Return the ids of ``agent_id``'s resting orders, oldest first."""
        ...

    def fills_after(self, ts: int) -> tuple[Fill, ...]:
        """Return every trade with timestamp strictly greater than ``ts``, in time order.

        A real participant sees the public trade tape; this is that tape. It only ever contains
        trades that have already happened.
        """
        ...


@runtime_checkable
class Agent(Protocol):
    """Something that trades. Subclassing is not required: any object with these members works.

    The market owns the clock. An agent only says *when it wants to be woken next*; it never
    waits or sleeps. All randomness must come from the ``rng`` the market passes in.
    """

    agent_id: int

    def first_wake(self, rng: Generator) -> int:
        """Return the absolute time (ns) of this agent's first turn."""
        ...

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Take a turn at time ``ts`` and return the actions to carry out, in order."""
        ...

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        """Return the absolute time of the next turn (after ``ts``), or ``None`` to stop."""
        ...
