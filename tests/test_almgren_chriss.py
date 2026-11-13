"""Tests for AlmgrenChrissStrategy execution schedules and market behavior."""

import pytest

from lobx.sim.actions import Submit
from lobx.sim.clock import NS_PER_S
from lobx.strategies.almgren_chriss import AlmgrenChrissStrategy
from lobx.strategies.base import ParentOrder
from lobx.types import Side


class MockMarketView:
    def __init__(self, position: int = 0) -> None:
        self._pos = position

    def position(self, agent_id: int) -> int:
        return self._pos


def test_almgren_chriss_zero_risk_aversion_matches_twap() -> None:
    strat = AlmgrenChrissStrategy(num_slices=5, risk_aversion=0.0)
    parent = ParentOrder(
        agent_id=1,
        side=Side.SELL,
        qty=1000,
        start_ns=0,
        end_ns=50 * NS_PER_S,
    )
    strat.reset(parent)

    # Each slice is 200
    assert strat.target_at(0) == 200
    assert strat.target_at(10 * NS_PER_S) == 400
    assert strat.target_at(20 * NS_PER_S) == 600
    assert strat.target_at(30 * NS_PER_S) == 800
    assert strat.target_at(40 * NS_PER_S) == 1000


def test_almgren_chriss_risk_aversion_front_loads_targets() -> None:
    strat = AlmgrenChrissStrategy(num_slices=5, risk_aversion=1e-3, volatility=2.0)
    parent = ParentOrder(
        agent_id=1,
        side=Side.SELL,
        qty=1000,
        start_ns=0,
        end_ns=50 * NS_PER_S,
    )
    strat.reset(parent)

    # First slice target should be strictly higher than TWAP (200)
    target_1 = strat.target_at(0)
    assert target_1 > 200

    # End target is still 100% of parent quantity
    assert strat.target_at(50 * NS_PER_S) == 1000


def test_almgren_chriss_step_generates_child_market_orders() -> None:
    strat = AlmgrenChrissStrategy(num_slices=4, risk_aversion=1e-4)
    parent = ParentOrder(
        agent_id=2,
        side=Side.BUY,
        qty=400,
        start_ns=10 * NS_PER_S,
        end_ns=50 * NS_PER_S,
    )
    strat.reset(parent)

    view = MockMarketView(position=0)

    # Before start_ns: no actions
    assert len(strat.step(5 * NS_PER_S, view)) == 0

    # At start_ns: emits child order for slice 1
    actions = strat.step(10 * NS_PER_S, view)
    assert len(actions) == 1
    assert isinstance(actions[0], Submit)
    assert actions[0].side is Side.BUY
    assert actions[0].qty > 0


def test_almgren_chriss_invalid_slices_raises() -> None:
    with pytest.raises(ValueError, match="num_slices must be greater than 0"):
        AlmgrenChrissStrategy(num_slices=0)
