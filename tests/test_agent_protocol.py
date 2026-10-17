"""Tests that the Agent protocol accepts what it should and rejects what it should not."""

from numpy.random import Generator

from lobx.sim.actions import Action
from lobx.sim.agents.base import Agent, MarketView


class _Idle:
    """The smallest valid agent: wakes once and does nothing."""

    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id

    def first_wake(self, rng: Generator) -> int:
        return 10

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        return []

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        return None


class _NoNextWake:
    agent_id = 1

    def first_wake(self, rng: Generator) -> int:
        return 0

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        return []


def test_a_class_with_all_members_is_an_agent() -> None:
    assert isinstance(_Idle(3), Agent)


def test_a_class_missing_a_member_is_not_an_agent() -> None:
    assert not isinstance(_NoNextWake(), Agent)
