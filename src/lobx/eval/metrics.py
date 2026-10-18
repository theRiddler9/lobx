"""The cost of an execution, in basis points."""

from __future__ import annotations

from collections.abc import Sequence

from lobx.strategies.base import ParentOrder
from lobx.types import Fill, Side

_BPS_PER_UNIT = 10_000


def implementation_shortfall_bps(
    parent: ParentOrder, fills: Sequence[Fill], arrival_mid2: int
) -> float:
    """Return how much worse the average execution price was than the arrival mid, in bps.

    Implementation shortfall compares what you got with what you would have got by trading the whole
    order instantly at the price you saw when you decided to trade (the *arrival mid-price*)::

        sell: (arrival_mid - average_price) / arrival_mid
        buy:  (average_price - arrival_mid) / arrival_mid

    Positive means the execution cost money; negative means it beat the arrival price. One basis
    point (bp) is 0.01%.

    ``arrival_mid2`` is *twice* the arrival mid so it stays an integer (the mid of a 2-tick spread
    can be half a tick); this differs from the draft contract's ``arrival_mid`` (see
    ``docs/integration.md``). ``fills`` must be exactly the strategy's own fills and must add up to
    the parent quantity: a partly executed order has an unmeasured opportunity cost, so it is an
    error here and callers report it separately.
    """
    if arrival_mid2 <= 0:
        raise ValueError(f"arrival_mid2 must be greater than 0, got {arrival_mid2}")
    executed = sum(fill.qty for fill in fills)
    if executed != parent.qty:
        raise ValueError(f"fills add up to {executed} shares but the parent order is {parent.qty}")
    notional = sum(fill.price * fill.qty for fill in fills)  # exact integer ticks x shares
    # average = notional / qty and mid = arrival_mid2 / 2, so average / mid = 2*notional /
    # (qty * arrival_mid2). Doing the division once, at the end, avoids accumulating float error.
    ratio = 2 * notional / (parent.qty * arrival_mid2)
    signed = 1 - ratio if parent.side is Side.SELL else ratio - 1
    return signed * _BPS_PER_UNIT
