"""Tests for the per-agent seeded random streams."""

import pytest

from lobx.sim.rng import AgentStreams


def _draws(streams: AgentStreams, agent_id: int, n: int = 5) -> list[int]:
    return [int(x) for x in streams.for_agent(agent_id).integers(0, 1_000_000, size=n)]


def test_same_seed_and_agent_give_the_same_numbers() -> None:
    assert _draws(AgentStreams(7), 3) == _draws(AgentStreams(7), 3)


def test_different_agents_get_different_numbers() -> None:
    streams = AgentStreams(7)
    assert _draws(streams, 1) != _draws(streams, 2)


def test_different_seeds_give_different_numbers() -> None:
    assert _draws(AgentStreams(1), 3) != _draws(AgentStreams(2), 3)


def test_an_agents_numbers_do_not_depend_on_other_agents() -> None:
    alone = _draws(AgentStreams(7), 3)
    crowded = AgentStreams(7)
    for other in (9, 1, 5):
        _draws(crowded, other)  # creating and using other streams first must not matter
    assert _draws(crowded, 3) == alone


def test_repeated_calls_continue_the_same_stream() -> None:
    streams = AgentStreams(7)
    first = _draws(streams, 3)
    second = _draws(streams, 3)
    assert first != second  # the stream advanced
    fresh = AgentStreams(7)
    assert _draws(fresh, 3, n=10) == first + second


def test_golden_values_for_seed_0() -> None:
    # Pinned on purpose: if a numpy upgrade changes the stream, every seeded result in the
    # project changes with it, and this is the test that says so.
    assert _draws(AgentStreams(0), 0, n=3) == _GOLDEN


@pytest.mark.parametrize("seed", [-1, 1.5, True, "0"])
def test_bad_seed_is_rejected(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        AgentStreams(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("agent_id", [-1, 1.5, True])
def test_bad_agent_id_is_rejected(agent_id: object) -> None:
    with pytest.raises(ValueError, match="agent_id"):
        AgentStreams(0).for_agent(agent_id)  # type: ignore[arg-type]


# Recorded from numpy 2.4.6 (the pinned version), not computed by hand.
_GOLDEN = [802237, 942937, 5483]
