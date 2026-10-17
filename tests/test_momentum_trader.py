"""Tests for the momentum trader, run against a hand-set market view."""

import pytest
from numpy.random import default_rng

from fakes import FakeView
from lobx.sim.actions import Action, Submit
from lobx.sim.agents.base import Agent
from lobx.sim.agents.momentum import MomentumTrader
from lobx.types import Side

_ME = 9
_RNG = default_rng(0)


def _trader(**overrides: object) -> MomentumTrader:
    params: dict[str, object] = {"short_window": 2, "long_window": 6, "threshold_ticks": 1.0}
    params.update(overrides)
    return MomentumTrader(_ME, interval_ns=1_000_000, **params)  # type: ignore[arg-type]


def _feed(trader: MomentumTrader, mids: list[int], position: int = 0) -> list[Action]:
    """Show the trader a series of mid-prices (in ticks) and return the last wake's actions."""
    view = FakeView()
    view.positions[_ME] = position
    actions: list[Action] = []
    for mid in mids:
        view.bid, view.ask = mid - 1, mid + 1  # mid2 == 2 * mid
        actions = trader.wake(0, view, _RNG)
    return actions


def test_it_is_an_agent() -> None:
    assert isinstance(_trader(), Agent)


def test_wake_times_are_in_the_future() -> None:
    rng = default_rng(1)
    trader = _trader()
    assert trader.first_wake(rng) >= 1
    nxt = trader.next_wake(100, rng)
    assert nxt is not None and nxt > 100


def test_it_does_nothing_until_it_has_a_full_window() -> None:
    for mids in ([100], [100, 101], [100, 101, 102, 103, 104]):  # fewer than long_window (6)
        assert _feed(_trader(), mids) == []


def test_rising_prices_trigger_a_market_buy() -> None:
    actions = _feed(_trader(qty=5), [100, 100, 100, 100, 104, 108])
    assert actions == [Submit.market(Side.BUY, 5)]


def test_falling_prices_trigger_a_market_sell() -> None:
    actions = _feed(_trader(qty=5), [108, 108, 108, 108, 104, 100])
    assert actions == [Submit.market(Side.SELL, 5)]


def test_flat_prices_do_nothing() -> None:
    assert _feed(_trader(), [100] * 10) == []


def test_a_small_move_below_the_threshold_does_nothing() -> None:
    # short avg 100.5 vs long avg 100.17: a 0.33 tick gap, below the 1 tick threshold
    assert _feed(_trader(threshold_ticks=1.0), [100, 100, 100, 100, 100, 101]) == []


def test_it_will_not_buy_at_the_long_limit() -> None:
    rising = [100, 100, 100, 100, 104, 108]
    assert _feed(_trader(max_position=50), rising, position=50) == []


def test_it_will_not_sell_at_the_short_limit() -> None:
    falling = [108, 108, 108, 108, 104, 100]
    assert _feed(_trader(max_position=50), falling, position=-50) == []


def test_order_size_is_trimmed_so_the_limit_is_not_exceeded() -> None:
    rising = [100, 100, 100, 100, 104, 108]
    actions = _feed(_trader(qty=5, max_position=50), rising, position=47)
    assert actions == [Submit.market(Side.BUY, 3)]
    falling = [108, 108, 108, 108, 104, 100]
    actions = _feed(_trader(qty=5, max_position=50), falling, position=-48)
    assert actions == [Submit.market(Side.SELL, 2)]


def test_no_mid_price_means_no_action_and_no_history() -> None:
    trader = _trader()
    view = FakeView(bid=None, ask=100)
    assert trader.wake(0, view, _RNG) == []
    assert len(trader._history) == 0  # internal state, checked on purpose


@pytest.mark.parametrize(
    "bad",
    [
        {"short_window": 6, "long_window": 6},
        {"short_window": 0},
        {"qty": 0},
        {"max_position": 0},
        {"threshold_ticks": -1.0},
    ],
)
def test_bad_parameters_are_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _trader(**bad)
