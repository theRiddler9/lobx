import pytest

from fakes import FakeView
from lobx.strategies.base import ParentOrder
from lobx.types import Side


def _parent(**changes: int) -> ParentOrder:
    fields = {"agent_id": 9, "qty": 100, "start_ns": 0, "end_ns": 10}
    fields.update(changes)
    return ParentOrder(side=Side.SELL, **fields)


def test_rejects_non_positive_qty() -> None:
    with pytest.raises(ValueError, match="qty"):
        _parent(qty=0)


@pytest.mark.parametrize("start,end", [(-1, 10), (5, 5), (6, 5)])
def test_rejects_bad_window(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="start_ns"):
        _parent(start_ns=start, end_ns=end)


def test_executed_is_positive_for_both_sides() -> None:
    sell = _parent()
    buy = ParentOrder(agent_id=9, side=Side.BUY, qty=100, start_ns=0, end_ns=10)
    view = FakeView()
    view.positions[9] = -30
    assert sell.executed(view) == 30
    view.positions[9] = 30
    assert buy.executed(view) == 30
