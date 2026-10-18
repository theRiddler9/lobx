import pytest

from lobx.eval.metrics import implementation_shortfall_bps
from lobx.strategies.base import ParentOrder
from lobx.types import Fill, Side


def _fills(taker_side: Side, *legs: tuple[int, int]) -> list[Fill]:
    return [Fill(1, 2, taker_side, price, qty, 0) for price, qty in legs]


def test_sell_below_arrival_costs_money() -> None:
    # Arrival mid 100.0. Sold 60 @ 99 and 40 @ 98: average 98.6, so (100 - 98.6) / 100 = 1.4% = 140.
    parent = ParentOrder(2, Side.SELL, 100, 0, 10)
    fills = _fills(Side.SELL, (99, 60), (98, 40))
    assert implementation_shortfall_bps(parent, fills, arrival_mid2=200) == pytest.approx(140.0)


def test_buy_above_arrival_costs_money() -> None:
    # Arrival mid 50.5 (mid2 = 101). Bought 10 @ 51: (51 - 50.5) / 50.5 = 0.990099% = 99.0099 bps.
    parent = ParentOrder(2, Side.BUY, 10, 0, 10)
    assert implementation_shortfall_bps(parent, _fills(Side.BUY, (51, 10)), 101) == pytest.approx(
        99.00990099
    )


def test_beating_the_arrival_price_is_negative() -> None:
    parent = ParentOrder(2, Side.SELL, 10, 0, 10)
    assert implementation_shortfall_bps(
        parent, _fills(Side.SELL, (101, 10)), arrival_mid2=200
    ) == pytest.approx(-100.0)


def test_executing_exactly_at_the_mid_costs_nothing() -> None:
    parent = ParentOrder(2, Side.BUY, 4, 0, 10)
    fills = _fills(Side.BUY, (100, 2), (100, 2))
    assert implementation_shortfall_bps(parent, fills, arrival_mid2=200) == 0.0


def test_sell_and_buy_are_mirror_images() -> None:
    sell = ParentOrder(2, Side.SELL, 10, 0, 10)
    buy = ParentOrder(2, Side.BUY, 10, 0, 10)
    sold = implementation_shortfall_bps(sell, _fills(Side.SELL, (99, 10)), 200)
    bought = implementation_shortfall_bps(buy, _fills(Side.BUY, (101, 10)), 200)
    assert sold == pytest.approx(100.0)
    assert bought == pytest.approx(100.0)


def test_incomplete_execution_is_an_error() -> None:
    parent = ParentOrder(2, Side.SELL, 100, 0, 10)
    with pytest.raises(ValueError, match="add up to 90"):
        implementation_shortfall_bps(parent, _fills(Side.SELL, (99, 90)), 200)


def test_rejects_non_positive_arrival_mid() -> None:
    parent = ParentOrder(2, Side.SELL, 1, 0, 10)
    with pytest.raises(ValueError, match="arrival_mid2"):
        implementation_shortfall_bps(parent, _fills(Side.SELL, (99, 1)), 0)
