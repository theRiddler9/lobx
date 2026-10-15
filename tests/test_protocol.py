"""Tests that the OrderBook protocol is usable for structural checks."""

from lobx.engine.protocol import OrderBook
from lobx.types import BookSnapshot, Fill, Order


class _Stub:
    def submit(self, order: Order) -> list[Fill]:
        return []

    def cancel(self, order_id: int, ts: int) -> bool:
        return False

    def snapshot(self, depth: int) -> BookSnapshot:
        return BookSnapshot(ts=0, bids=(), asks=())

    def best_bid(self) -> int | None:
        return None

    def best_ask(self) -> int | None:
        return None


def test_stub_satisfies_protocol() -> None:
    assert isinstance(_Stub(), OrderBook)


def test_object_missing_methods_does_not_satisfy_protocol() -> None:
    assert not isinstance(object(), OrderBook)
