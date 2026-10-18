"""Seeded random order streams for testing and benchmarking the engines.

Both engines are fed the same stream and must give the same answers. The stream is dense on
purpose: prices live in a narrow band so orders cross, partially fill, and get cancelled often.

Only ``random.Random.random()`` is used, never ``randint`` or ``choice``. ``random()`` is
guaranteed to give the same sequence for the same seed on every Python version, so a failing seed
can be reproduced anywhere.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from lobx.types import Order, OrderType, Side

MID_PRICE = 1_000
PRICE_BAND = 20  # limit prices fall in [MID_PRICE - PRICE_BAND, MID_PRICE + PRICE_BAND]
MAX_QTY = 20


@dataclass(frozen=True, slots=True)
class CancelOp:
    """A cancel request in a generated stream."""

    order_id: int
    ts: int


StreamOp = Order | CancelOp


def random_stream(seed: int, count: int) -> Iterator[StreamOp]:
    """Yield ``count`` valid operations: ~55% limit, ~15% market, ~30% cancel.

    Time never goes backwards and order ids are unique. A cancel targets an id issued earlier,
    which may already be filled or cancelled, so the ``False`` result is exercised too.
    """
    rng = random.Random(seed)
    ts = 0
    next_id = 1
    for _ in range(count):
        ts += int(rng.random() * 3)  # 0, 1 or 2 ns: equal timestamps are allowed
        kind = rng.random()
        if kind < 0.30 and next_id > 1:
            yield CancelOp(1 + int(rng.random() * (next_id - 1)), ts)
            continue
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        qty = 1 + int(rng.random() * MAX_QTY)
        agent = 1 + int(rng.random() * 5)
        if kind < 0.45:
            yield Order(next_id, agent, side, OrderType.MARKET, None, qty, ts)
        else:
            price = MID_PRICE - PRICE_BAND + int(rng.random() * (2 * PRICE_BAND + 1))
            yield Order(next_id, agent, side, OrderType.LIMIT, price, qty, ts)
        next_id += 1
