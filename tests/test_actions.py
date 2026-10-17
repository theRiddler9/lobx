"""Tests for the action types agents return."""

import dataclasses

import pytest

from lobx.sim.actions import Cancel, Submit
from lobx.types import OrderType, Side


def test_limit_request_carries_a_price() -> None:
    action = Submit.limit(Side.BUY, 10_000, 5)
    assert (action.side, action.order_type, action.price, action.qty) == (
        Side.BUY,
        OrderType.LIMIT,
        10_000,
        5,
    )


def test_market_request_has_no_price() -> None:
    action = Submit.market(Side.SELL, 3)
    assert (action.order_type, action.price) == (OrderType.MARKET, None)


def test_actions_are_frozen_and_comparable() -> None:
    assert Submit.limit(Side.BUY, 1, 1) == Submit.limit(Side.BUY, 1, 1)
    assert Cancel(4) == Cancel(4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Cancel(4).order_id = 5  # type: ignore[misc]
