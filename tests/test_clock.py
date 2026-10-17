"""Tests for the simulation clock and the wake queue."""

import pytest

from lobx.errors import SimulationError
from lobx.sim.clock import NS_PER_MS, NS_PER_S, NS_PER_US, Clock, WakeQueue


def test_unit_constants() -> None:
    assert (NS_PER_US, NS_PER_MS, NS_PER_S) == (1_000, 1_000_000, 1_000_000_000)


def test_clock_starts_at_zero_and_moves_forward() -> None:
    clock = Clock()
    assert clock.now == 0
    clock.advance_to(50)
    clock.advance_to(50)  # standing still is allowed
    assert clock.now == 50


def test_clock_refuses_to_go_back() -> None:
    clock = Clock()
    clock.advance_to(50)
    with pytest.raises(SimulationError, match="backwards"):
        clock.advance_to(49)
    assert clock.now == 50


def test_queue_pops_earliest_first() -> None:
    queue = WakeQueue()
    for ts, agent in [(30, 1), (10, 2), (20, 3)]:
        queue.push(ts, agent)
    assert [queue.pop() for _ in range(3)] == [(10, 2), (20, 3), (30, 1)]


def test_equal_times_come_out_in_push_order() -> None:
    queue = WakeQueue()
    for agent in [5, 3, 9, 1]:
        queue.push(100, agent)
    assert [queue.pop()[1] for _ in range(4)] == [5, 3, 9, 1]


def test_peek_len_and_truthiness() -> None:
    queue = WakeQueue()
    assert not queue and len(queue) == 0 and queue.peek_ts() is None
    queue.push(7, 1)
    queue.push(3, 2)
    assert queue and len(queue) == 2 and queue.peek_ts() == 3
    queue.pop()
    assert queue.peek_ts() == 7
