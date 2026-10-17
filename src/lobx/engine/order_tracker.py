"""Who owns each resting order, and how much of it is left.

The engine itself does not care which agent an order belongs to. Everything built on top of it does:
the event log flags self-trades, the market gives each agent its own list of open orders, and
replay re-checks those flags. This small index answers "who owns order 17?" and "which orders does
agent 3 have open?" without asking the engine to remember more than it needs to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lobx.errors import SimulationError
from lobx.types import Fill, Order, OrderType


@dataclass(slots=True)
class _OpenOrder:
    agent_id: int
    remaining: int


class OrderTracker:
    """An index of the limit orders that are currently resting, built from submits and cancels."""

    def __init__(self) -> None:
        """Create an empty tracker."""
        self._open: dict[int, _OpenOrder] = {}
        # Per agent, a dict used as an ordered set: insertion order is arrival order, and removing
        # one order stays O(1).
        self._by_agent: dict[int, dict[int, None]] = {}

    def on_submit(self, order: Order, fills: Sequence[Fill]) -> None:
        """Record the effect of ``order`` and the ``fills`` it produced.

        Makers lose the filled quantity and are forgotten when nothing is left. Whatever part of a
        limit order did not trade is now resting; a market order never rests.
        """
        for fill in fills:
            maker = self._open.get(fill.maker_order_id)
            if maker is None:
                raise SimulationError(
                    f"fill refers to order {fill.maker_order_id}, which is not open"
                )
            maker.remaining -= fill.qty
            if maker.remaining < 0:
                raise SimulationError(
                    f"order {fill.maker_order_id} was filled for more than it had"
                )
            if maker.remaining == 0:
                self._forget(fill.maker_order_id)
        unfilled = order.qty - sum(fill.qty for fill in fills)
        if unfilled > 0 and order.type is OrderType.LIMIT:
            self._open[order.order_id] = _OpenOrder(order.agent_id, unfilled)
            self._by_agent.setdefault(order.agent_id, {})[order.order_id] = None

    def on_cancel(self, order_id: int, cancelled: bool) -> None:
        """Forget ``order_id`` if the engine reported that it really was cancelled."""
        if cancelled:
            self._forget(order_id)

    def agent_of(self, order_id: int) -> int | None:
        """Return the owner of a resting order, or ``None`` if it is not resting."""
        entry = self._open.get(order_id)
        return None if entry is None else entry.agent_id

    def remaining(self, order_id: int) -> int:
        """Return the unfilled quantity of a resting order (0 if it is not resting)."""
        entry = self._open.get(order_id)
        return 0 if entry is None else entry.remaining

    def open_order_ids(self, agent_id: int) -> tuple[int, ...]:
        """Return the ids of ``agent_id``'s resting orders, oldest first."""
        return tuple(self._by_agent.get(agent_id, ()))

    def __len__(self) -> int:
        """Return the number of resting orders across all agents."""
        return len(self._open)

    def _forget(self, order_id: int) -> None:
        entry = self._open.pop(order_id, None)
        if entry is None:
            return
        owned = self._by_agent[entry.agent_id]
        del owned[order_id]
        if not owned:
            del self._by_agent[entry.agent_id]
