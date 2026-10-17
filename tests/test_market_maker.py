"""Tests for the market maker, run against a hand-set market view."""

import pytest
from numpy.random import default_rng

from fakes import FakeView
from lobx.sim.actions import Action, Cancel, Submit
from lobx.sim.agents.base import Agent
from lobx.sim.agents.market_maker import MarketMaker
from lobx.types import Side

_ME = 1


def _maker(**overrides: object) -> MarketMaker:
    return MarketMaker(_ME, interval_ns=1_000_000, **overrides)  # type: ignore[arg-type]


def _quotes(actions: list[Action], side: Side) -> list[int]:
    return [
        a.price for a in actions if isinstance(a, Submit) and a.side is side and a.price is not None
    ]


def test_it_is_an_agent() -> None:
    assert isinstance(_maker(), Agent)


def test_first_wake_is_time_zero_so_the_book_is_seeded_first() -> None:
    assert _maker().first_wake(default_rng(0)) == 0


def test_next_wake_is_in_the_future_and_near_the_interval() -> None:
    rng = default_rng(1)
    gaps = [_maker().next_wake(0, rng) for _ in range(500)]
    assert all(g is not None and 750_000 <= g <= 1_250_000 for g in gaps)


def test_flat_inventory_gives_symmetric_quotes_around_the_reference() -> None:
    view = FakeView(bid=9_999, ask=10_001)  # reference price 10_000
    actions = _maker(half_spread_ticks=2, levels=3, level_gap_ticks=2).wake(0, view, default_rng(0))
    assert _quotes(actions, Side.BUY) == [9_998, 9_996, 9_994]
    assert _quotes(actions, Side.SELL) == [10_002, 10_004, 10_006]


def test_quote_size_is_applied_to_every_quote() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    actions = _maker(quote_qty=7).wake(0, view, default_rng(0))
    assert {a.qty for a in actions if isinstance(a, Submit)} == {7}


def test_old_quotes_are_cancelled_before_new_ones_are_posted() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    view.open[_ME] = (4, 5, 6)
    actions = _maker().wake(0, view, default_rng(0))
    assert actions[:3] == [Cancel(4), Cancel(5), Cancel(6)]
    assert all(isinstance(a, Submit) for a in actions[3:])


def test_long_inventory_shifts_both_quotes_down() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    view.positions[_ME] = 100  # 100 shares * 0.02 = 2 ticks down
    actions = _maker(skew_ticks_per_share=0.02, levels=1).wake(0, view, default_rng(0))
    assert _quotes(actions, Side.BUY) == [9_996]
    assert _quotes(actions, Side.SELL) == [10_000]


def test_short_inventory_shifts_both_quotes_up() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    view.positions[_ME] = -100
    actions = _maker(skew_ticks_per_share=0.02, levels=1).wake(0, view, default_rng(0))
    assert _quotes(actions, Side.BUY) == [10_000]
    assert _quotes(actions, Side.SELL) == [10_004]


def test_at_the_long_limit_it_only_sells() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    view.positions[_ME] = 500
    actions = _maker(max_position=500).wake(0, view, default_rng(0))
    assert _quotes(actions, Side.BUY) == []
    assert len(_quotes(actions, Side.SELL)) == 3


def test_at_the_short_limit_it_only_buys() -> None:
    view = FakeView(bid=9_999, ask=10_001)
    view.positions[_ME] = -500
    actions = _maker(max_position=500).wake(0, view, default_rng(0))
    assert _quotes(actions, Side.SELL) == []
    assert len(_quotes(actions, Side.BUY)) == 3


def test_bid_is_always_below_ask_even_at_extreme_skew() -> None:
    view = FakeView(bid=2, ask=3, opening_price=2)
    view.positions[_ME] = -499  # skew pushes the reservation price far up
    for position in (-499, 0, 499):
        view.positions[_ME] = position
        actions = _maker(skew_ticks_per_share=5.0, levels=1).wake(0, view, default_rng(0))
        bids, asks = _quotes(actions, Side.BUY), _quotes(actions, Side.SELL)
        assert all(b >= 1 for b in bids) and all(a >= 1 for a in asks)
        if bids and asks:
            assert bids[0] < asks[0]


def test_empty_book_quotes_around_the_opening_price() -> None:
    view = FakeView(opening_price=7_000)
    actions = _maker(half_spread_ticks=2, levels=1).wake(0, view, default_rng(0))
    assert _quotes(actions, Side.BUY) == [6_998]
    assert _quotes(actions, Side.SELL) == [7_002]


@pytest.mark.parametrize(
    "bad",
    [
        {"half_spread_ticks": 0},
        {"levels": 0},
        {"level_gap_ticks": 0},
        {"quote_qty": 0},
        {"max_position": 0},
        {"skew_ticks_per_share": -0.1},
    ],
)
def test_bad_parameters_are_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _maker(**bad)
