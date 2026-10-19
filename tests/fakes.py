"""Test doubles shared by the agent tests."""

from __future__ import annotations

from lobx.types import BookSnapshot, Fill


class FakeView:
    """A hand-set ``MarketView`` so an agent can be tested without running a market."""

    def __init__(
        self,
        bid: int | None = None,
        ask: int | None = None,
        last_trade: int | None = None,
        opening_price: int = 10_000,
        ts: int = 0,
    ) -> None:
        self.bid = bid
        self.ask = ask
        self.last_trade = last_trade
        self.opening_price = opening_price
        self._ts = ts
        self.positions: dict[int, int] = {}
        self.open: dict[int, tuple[int, ...]] = {}
        self.tape: list[Fill] = []

    @property
    def ts(self) -> int:
        return self._ts

    def best_bid(self) -> int | None:
        return self.bid

    def best_ask(self) -> int | None:
        return self.ask

    def mid2(self) -> int | None:
        if self.bid is None or self.ask is None:
            return None
        return self.bid + self.ask

    def last_trade_price(self) -> int | None:
        return self.last_trade

    def reference_price(self) -> int:
        mid2 = self.mid2()
        if mid2 is not None:
            return mid2 // 2
        return self.last_trade if self.last_trade is not None else self.opening_price

    def snapshot(self, depth: int) -> BookSnapshot:
        bids = () if self.bid is None else ((self.bid, 1),)
        asks = () if self.ask is None else ((self.ask, 1),)
        return BookSnapshot(self._ts, bids, asks)

    def position(self, agent_id: int) -> int:
        return self.positions.get(agent_id, 0)

    def open_orders(self, agent_id: int) -> tuple[int, ...]:
        return self.open.get(agent_id, ())

    def fills_after(self, ts: int) -> tuple[Fill, ...]:
        return tuple(f for f in self.tape if f.ts > ts)
