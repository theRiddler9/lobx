"""The execution contract: a parent order to work, and the protocol every strategy implements.

A strategy never touches the book. Each time it is asked (``step``) it looks at the market through
a read-only ``MarketView`` and returns actions (child orders); the market carries them out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lobx.sim.actions import Action
from lobx.sim.agents.base import MarketView
from lobx.types import Side


@dataclass(frozen=True, slots=True)
class ParentOrder:
    """A large order to execute gradually between ``start_ns`` and ``end_ns``.

    ``agent_id`` is the id the strategy trades under. The strategy reads its own progress from
    ``view.position(agent_id)``, so it never has to be told about fills.
    """

    agent_id: int
    side: Side
    qty: int
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        """Reject a parent order that cannot be executed."""
        if self.qty <= 0:
            raise ValueError(f"parent qty must be greater than 0, got {self.qty}")
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError(
                f"need 0 <= start_ns < end_ns, got start_ns={self.start_ns}, end_ns={self.end_ns}"
            )

    def executed(self, view: MarketView) -> int:
        """Return how many shares of this parent order have traded so far.

        A seller's position goes negative as it sells and a buyer's goes positive, so the sign is
        flipped for sells to give a non-negative count.
        """
        position = view.position(self.agent_id)
        return -position if self.side is Side.SELL else position


class ExecutionStrategy(Protocol):
    """An algorithm that turns a parent order into child orders over time."""

    def reset(self, parent: ParentOrder) -> None:
        """Prepare to execute ``parent``. Called once, before the first ``step``."""
        ...

    def step(self, ts: int, view: MarketView) -> list[Action]:
        """Return the child orders to send at time ``ts``. May use only data at or before ``ts``."""
        ...
