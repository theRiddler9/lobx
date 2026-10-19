"""Tests of the market loop's mechanics, using small scripted agents and one short real run."""

from __future__ import annotations

import pytest
from numpy.random import Generator

from lobx.engine.book_py import PythonOrderBook
from lobx.errors import SimulationError
from lobx.sim.actions import Action, Cancel, Submit
from lobx.sim.agents.base import Agent, MarketView
from lobx.sim.market import Market, RunResult
from lobx.sim.population import MarketConfig
from lobx.types import CancelEvent, FillEvent, Side, SubmitEvent

_GAP = 1_000  # ns between a scripted agent's turns


class Scripted:
    """An agent that plays back a fixed list of action lists, one list per turn."""

    def __init__(self, agent_id: int, script: list[list[Action]], first: int = _GAP) -> None:
        self.agent_id = agent_id
        self._script = list(script)
        self._first = first
        self.seen_references: list[int] = []

    def first_wake(self, rng: Generator) -> int:
        return self._first

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        self.seen_references.append(view.reference_price())
        return self._script.pop(0) if self._script else []

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        return ts + _GAP if self._script else None


def _market(agents: list[Agent], duration_ns: int = 1_000_000) -> Market:
    return Market(MarketConfig(duration_ns=duration_ns, opening_price=100), agents=agents)


def test_scripted_trade_updates_positions_last_price_and_ids() -> None:
    seller = Scripted(1, [[Submit.limit(Side.SELL, 100, 5)]], first=1_000)
    buyer = Scripted(2, [[Submit.market(Side.BUY, 3)]], first=2_000)
    market = _market([seller, buyer])
    result = market.run()
    assert market.position(1) == -3
    assert market.position(2) == 3
    assert market.last_trade_price() == 100
    assert market.open_orders(1) == (1,)  # order ids are stamped by the market, starting at 1
    assert result.trades == 1
    assert result.self_trades == 0
    assert result.final_snapshot.asks == ((100, 2),)


def test_positions_always_sum_to_zero() -> None:
    market = Market(MarketConfig(seed=4, duration_ns=15_000_000_000, num_noise_traders=30))
    result = market.run()
    agent_ids = result.roles.keys()
    assert sum(market.position(a) for a in agent_ids) == 0
    assert result.trades > 0


def test_order_ids_are_unique_increasing_and_timestamps_never_go_back() -> None:
    result = Market(MarketConfig(seed=2, duration_ns=10_000_000_000, num_noise_traders=20)).run()
    submit_ids = [e.order.order_id for e in result.log if isinstance(e, SubmitEvent)]
    assert submit_ids == sorted(set(submit_ids))
    stamps: list[int] = []
    for event in result.log:
        if isinstance(event, SubmitEvent):
            stamps.append(event.order.ts)
        elif isinstance(event, CancelEvent):
            stamps.append(event.ts)
        else:
            stamps.append(event.fill.ts)
    assert stamps == sorted(stamps)


def test_resting_quantity_in_the_book_matches_the_trackers_books() -> None:
    market = Market(MarketConfig(seed=7, duration_ns=15_000_000_000, num_noise_traders=30))
    result = market.run()
    tracker = market.book.tracker
    tracked = sum(
        tracker.remaining(order_id)
        for agent_id in result.roles
        for order_id in tracker.open_order_ids(agent_id)
    )
    full = market.snapshot(1_000_000)
    in_book = sum(q for _, q in full.bids) + sum(q for _, q in full.asks)
    assert tracked == in_book


def test_self_trade_is_counted_and_flagged_in_the_log() -> None:
    both = Scripted(1, [[Submit.limit(Side.SELL, 100, 2), Submit.market(Side.BUY, 2)]])
    market = _market([both])
    result = market.run()
    assert result.self_trades == 1
    fills = [e for e in result.log if isinstance(e, FillEvent)]
    assert [f.self_trade for f in fills] == [True]
    assert market.position(1) == 0  # bought and sold the same shares to itself


def test_cancelling_another_agents_order_is_an_error() -> None:
    owner = Scripted(1, [[Submit.limit(Side.SELL, 100, 5)]], first=1_000)
    thief = Scripted(2, [[Cancel(1)]], first=2_000)
    with pytest.raises(SimulationError, match="belongs to agent 1"):
        _market([owner, thief]).run()


def test_cancelling_your_own_order_works() -> None:
    agent = Scripted(1, [[Submit.limit(Side.SELL, 100, 5)], [Cancel(1)]])
    market = _market([agent])
    result = market.run()
    assert market.open_orders(1) == ()
    assert result.final_snapshot.asks == ()
    cancels = [e for e in result.log if isinstance(e, CancelEvent)]
    assert [(c.order_id, c.cancelled) for c in cancels] == [(1, True)]


def test_cancelling_a_gone_order_is_logged_as_not_cancelled() -> None:
    agent = Scripted(1, [[Cancel(999)]])
    result = _market([agent]).run()
    cancels = [e for e in result.log if isinstance(e, CancelEvent)]
    assert [(c.order_id, c.cancelled) for c in cancels] == [(999, False)]


def test_an_order_the_engine_rejects_is_reported_with_the_agent_id() -> None:
    bad = Scripted(5, [[Submit.limit(Side.BUY, 0, 1)]])
    with pytest.raises(SimulationError, match="agent 5 sent an order the engine rejected"):
        _market([bad]).run()


def test_waking_again_at_the_same_instant_is_an_error() -> None:
    class Stuck(Scripted):
        def next_wake(self, ts: int, rng: Generator) -> int | None:
            return ts

    with pytest.raises(SimulationError, match="not after the current time"):
        _market([Stuck(1, [])]).run()


def test_first_wake_before_time_zero_is_an_error() -> None:
    with pytest.raises(SimulationError, match="before 0"):
        _market([Scripted(1, [], first=-1)]).run()


def test_duplicate_agent_ids_are_rejected() -> None:
    with pytest.raises(SimulationError, match="share the id 1"):
        _market([Scripted(1, []), Scripted(1, [])])


def test_a_market_can_only_run_once() -> None:
    market = _market([Scripted(1, [])])
    market.run()
    with pytest.raises(SimulationError, match="only be run once"):
        market.run()


def test_reference_price_follows_mid_then_last_trade_then_opening_price() -> None:
    # The probe looks at the market at 1000, 2000, 3000 and 4000 ns. In between:
    #   1500: a maker posts a bid at 90 and an ask at 120.
    #   2500: a taker buys the whole ask, leaving only the bid.
    probe = Scripted(9, [[], [], [], []])
    maker = Scripted(1, [[Submit.limit(Side.BUY, 90, 5), Submit.limit(Side.SELL, 120, 5)]], 1_500)
    taker = Scripted(2, [[Submit.market(Side.BUY, 5)]], 2_500)
    _market([probe, maker, taker]).run()
    assert probe.seen_references == [
        100,  # empty book, nothing traded yet: the opening price
        105,  # both sides present: the mid of 90 and 120
        120,  # one-sided book: the last trade
        120,
    ]


def test_samples_are_taken_at_every_interval_up_to_the_duration() -> None:
    config = MarketConfig(duration_ns=5_000, sample_interval_ns=1_000, opening_price=100)
    result = Market(
        config, agents=[Scripted(1, [[Submit.limit(Side.BUY, 99, 1)]], first=1_500)]
    ).run()
    assert [s.ts for s in result.samples] == [1_000, 2_000, 3_000, 4_000, 5_000]
    assert result.samples[0].best_bid is None  # before the order arrived at 1500
    assert result.samples[1].best_bid == 99
    assert result.samples[1].spread is None and result.samples[1].mid2 is None


def test_sample_spread_and_mid2() -> None:
    config = MarketConfig(duration_ns=3_000, sample_interval_ns=1_000, opening_price=100)
    quoter = Scripted(1, [[Submit.limit(Side.BUY, 99, 1), Submit.limit(Side.SELL, 102, 1)]], 500)
    sample = Market(config, agents=[quoter]).run().samples[0]
    assert sample.spread == 3
    assert sample.mid2 == 201


def test_a_different_engine_can_be_plugged_in() -> None:
    engine = PythonOrderBook()
    agent = Scripted(1, [[Submit.limit(Side.BUY, 50, 1)]])
    Market(MarketConfig(duration_ns=10_000, opening_price=100), [agent], book=engine).run()
    assert engine.best_bid() == 50


def test_result_reports_roles_and_peak_positions() -> None:
    result: RunResult = Market(
        MarketConfig(seed=1, duration_ns=8_000_000_000, num_noise_traders=10)
    ).run()
    assert set(result.roles.values()) == {"MarketMaker", "NoiseTrader", "MomentumTrader"}
    assert set(result.peak_abs_position) == set(result.roles)  # idle agents are listed too
    assert all(peak >= 0 for peak in result.peak_abs_position.values())
    assert result.log.seed == 1
    assert result.log.config == result.config.as_dict()


def test_fills_after_is_the_public_trade_tape() -> None:
    from lobx.sim.clock import NS_PER_S

    config = MarketConfig(seed=2, duration_ns=30 * NS_PER_S)
    market = Market(config)
    result = market.run()
    all_fills = [e.fill for e in result.log if isinstance(e, FillEvent)]
    assert list(market.fills_after(-1)) == all_fills
    midpoint = all_fills[len(all_fills) // 2].ts
    assert list(market.fills_after(midpoint)) == [f for f in all_fills if f.ts > midpoint]
    assert market.fills_after(all_fills[-1].ts) == ()
