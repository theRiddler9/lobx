import numpy as np

from fakes import FakeView
from lobx.sim.actions import Submit
from lobx.strategies.agent import StrategyAgent
from lobx.strategies.base import ParentOrder
from lobx.strategies.twap import TwapStrategy
from lobx.types import Side

RNG = np.random.default_rng(0)


def _agent() -> StrategyAgent:
    parent = ParentOrder(5, Side.SELL, 100, start_ns=1_000, end_ns=1_400)
    return StrategyAgent(TwapStrategy(4), parent, num_slices=4)


def test_wakes_once_per_slice_and_stops_at_the_deadline() -> None:
    agent = _agent()
    times = [agent.first_wake(RNG)]
    while (following := agent.next_wake(times[-1], RNG)) is not None:
        times.append(following)
    assert times == [1_000, 1_100, 1_200, 1_300]


def test_records_arrival_mid_on_first_wake_only() -> None:
    agent = _agent()
    agent.wake(1_000, FakeView(bid=99, ask=103), RNG)
    assert agent.arrival_mid2 == 202
    agent.wake(1_100, FakeView(bid=50, ask=60), RNG)
    assert agent.arrival_mid2 == 202  # unchanged


def test_arrival_mid_falls_back_to_reference_price_on_an_empty_book() -> None:
    agent = _agent()
    agent.wake(1_000, FakeView(opening_price=10_000), RNG)
    assert agent.arrival_mid2 == 20_000


def test_wake_returns_the_strategy_actions() -> None:
    (action,) = _agent().wake(1_000, FakeView(bid=99, ask=101), RNG)
    assert action == Submit.market(Side.SELL, 25)


def test_a_watching_strategy_wakes_often_and_starts_before_the_window() -> None:
    parent = ParentOrder(5, Side.SELL, 100, start_ns=1_000, end_ns=1_100)
    agent = StrategyAgent(TwapStrategy(2), parent, 2, wake_interval_ns=20, lookback_ns=40)
    times = [agent.first_wake(RNG)]
    while (following := agent.next_wake(times[-1], RNG)) is not None:
        times.append(following)
    assert times == [960, 980, 1_000, 1_020, 1_040, 1_060, 1_080]


def test_arrival_mid_waits_for_the_window_even_when_waking_early() -> None:
    parent = ParentOrder(5, Side.SELL, 100, start_ns=1_000, end_ns=1_100)
    agent = StrategyAgent(TwapStrategy(2), parent, 2, wake_interval_ns=20, lookback_ns=40)
    agent.wake(960, FakeView(bid=10, ask=14), RNG)
    assert agent.arrival_mid2 is None
    agent.wake(1_000, FakeView(bid=99, ask=103), RNG)
    assert agent.arrival_mid2 == 202


def test_the_lookback_cannot_start_before_time_zero() -> None:
    parent = ParentOrder(5, Side.SELL, 100, start_ns=30, end_ns=130)
    agent = StrategyAgent(TwapStrategy(2), parent, 2, wake_interval_ns=10, lookback_ns=500)
    assert agent.first_wake(RNG) == 0
