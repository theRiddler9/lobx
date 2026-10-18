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
