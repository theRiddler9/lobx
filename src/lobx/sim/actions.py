"""What an agent can ask the market to do.

An action is a *request*, not an order. The agent says what it wants (side, price, quantity) and the
market stamps on the order id, the agent id and the timestamp. Agents therefore cannot forge an
id, claim to be another agent, or travel in time, and the numbering of orders is the market's alone
(which keeps replays reproducible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from lobx.types import OrderType, Side


@dataclass(frozen=True, slots=True)
class Submit:
    """Ask the market to submit a new order on the agent's behalf."""

    side: Side
    order_type: OrderType
    price: int | None
    qty: int

    @classmethod
    def limit(cls, side: Side, price: int, qty: int) -> Submit:
        """Build a limit order request: it may rest in the book at ``price``."""
        return cls(side, OrderType.LIMIT, price, qty)

    @classmethod
    def market(cls, side: Side, qty: int) -> Submit:
        """Build a market order request: it trades now at whatever price is available."""
        return cls(side, OrderType.MARKET, None, qty)


@dataclass(frozen=True, slots=True)
class Cancel:
    """Ask the market to cancel one of the agent's own resting orders."""

    order_id: int


Action: TypeAlias = Submit | Cancel
