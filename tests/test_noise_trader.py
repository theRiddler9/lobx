"""Tests for the noise trader, run against a hand-set market view."""

import pytest
from numpy.random import default_rng

from fakes import FakeView
from lobx.sim.actions import Cancel, Submit
from lobx.sim.agents.base import Agent
from lobx.sim.agents.noise import NoiseTrader
from lobx.types import OrderType, Side

_ME = 5


def _trader(**overrides: object) -> NoiseTrader:
    return NoiseTrader(_ME, mean_interval_ns=1_000_000, **overrides)  # type: ignore[arg-type]


def _many_wakes(trader: NoiseTrader, n: int = 4_000) -> list[Submit]:
    rng = default_rng(11)
    view = FakeView(bid=9_999, ask=10_001)  # reference price 10_000
    submits: list[Submit] = []
    for _ in range(n):
        submits.extend(a for a in trader.wake(0, view, rng) if isinstance(a, Submit))
    return submits


def test_it_is_an_agent() -> None:
    assert isinstance(_trader(), Agent)


def test_wake_times_move_forward() -> None:
    rng = default_rng(1)
    trader = _trader()
    ts = trader.first_wake(rng)
    assert ts >= 1
    for _ in range(100):
        nxt = trader.next_wake(ts, rng)
        assert nxt is not None and nxt > ts
        ts = nxt


def test_each_wake_places_exactly_one_order_below_the_cap() -> None:
    rng = default_rng(2)
    view = FakeView(bid=9_999, ask=10_001)
    for _ in range(200):
        actions = _trader().wake(0, view, rng)
        assert len(actions) == 1 and isinstance(actions[0], Submit)


def test_limit_orders_rest_on_the_passive_side_of_the_reference() -> None:
    for action in _many_wakes(_trader()):
        if action.order_type is OrderType.LIMIT:
            assert action.price is not None
            if action.side is Side.BUY:
                assert action.price < 10_000
            else:
                assert action.price > 10_000


def test_sizes_are_within_bounds_and_market_orders_are_heavy_tailed() -> None:
    submits = _many_wakes(_trader())
    limits = [s for s in submits if s.order_type is OrderType.LIMIT]
    markets = [s for s in submits if s.order_type is OrderType.MARKET]
    assert all(1 <= s.qty <= 10 for s in limits)
    assert all(1 <= s.qty <= 100 for s in markets)
    assert all(s.price is None for s in markets)
    sizes = sorted(s.qty for s in markets)
    median, biggest = sizes[len(sizes) // 2], sizes[-1]
    assert median <= 5 and biggest >= 40  # most are small, a few are huge


def test_share_of_market_orders_matches_p_market() -> None:
    submits = _many_wakes(_trader(p_market=0.3), n=6_000)
    share = sum(s.order_type is OrderType.MARKET for s in submits) / len(submits)
    assert abs(share - 0.3) < 0.03


def test_both_sides_are_used_about_equally() -> None:
    submits = _many_wakes(_trader(), n=6_000)
    buys = sum(s.side is Side.BUY for s in submits) / len(submits)
    assert abs(buys - 0.5) < 0.03


def test_oldest_order_is_cancelled_when_at_the_cap() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    view.open[_ME] = (11, 12, 13)
    actions = _trader(max_open_orders=3).wake(0, view, default_rng(3))
    assert actions[0] == Cancel(11)
    assert isinstance(actions[1], Submit) and len(actions) == 2


def test_no_cancel_below_the_cap() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    view.open[_ME] = (11, 12)
    actions = _trader(max_open_orders=3).wake(0, view, default_rng(3))
    assert not any(isinstance(a, Cancel) for a in actions)


def test_on_an_empty_book_it_anchors_to_the_opening_price() -> None:
    view = FakeView(opening_price=5_000)
    rng = default_rng(4)
    trader = _trader(p_market=0.0)
    prices = [
        a.price for _ in range(200) for a in trader.wake(0, view, rng) if isinstance(a, Submit)
    ]
    assert all(p is not None and 4_900 < p < 5_100 for p in prices)


def test_prices_never_drop_below_one_tick() -> None:
    view = FakeView(bid=1, ask=2, opening_price=1)
    rng = default_rng(5)
    trader = _trader(p_market=0.0, mean_offset_ticks=50.0)
    for _ in range(500):
        for action in trader.wake(0, view, rng):
            assert isinstance(action, Submit) and action.price is not None and action.price >= 1


def test_same_stream_gives_the_same_orders() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    first = [_trader().wake(0, view, default_rng(8)) for _ in range(1)]
    second = [_trader().wake(0, view, default_rng(8)) for _ in range(1)]
    assert first == second


@pytest.mark.parametrize(
    "bad",
    [
        {"p_market": 1.5},
        {"p_market": -0.1},
        {"mean_offset_ticks": 0.5},
        {"max_limit_qty": 0},
        {"max_open_orders": 0},
        {"pareto_alpha": 0.0},
    ],
)
def test_bad_parameters_are_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _trader(**bad)
