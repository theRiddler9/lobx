"""Helpers for turning a random draw into "when should this agent wake next"."""

from __future__ import annotations

from numpy.random import Generator


def exponential_gap_ns(rng: Generator, mean_ns: int) -> int:
    """Return a random gap whose average is ``mean_ns``, at least 1 ns.

    Gaps drawn from an exponential distribution make arrivals a *Poisson process*: events come at
    a steady average rate but at unpredictable moments, with no memory of the previous one. That
    is the standard model for orders arriving from many independent traders. The floor of 1 ns
    guarantees time always moves forward.
    """
    return max(1, int(rng.exponential(mean_ns)))


def jittered_gap_ns(rng: Generator, interval_ns: int, jitter_fraction: float = 0.25) -> int:
    """Return ``interval_ns`` plus or minus up to ``jitter_fraction`` of it, at least 1 ns.

    Used by agents that act on a rough schedule. The jitter stops every such agent from waking at
    exactly the same instants, which would make the market unrealistically lock-stepped.
    """
    spread = int(interval_ns * jitter_fraction)
    return max(1, interval_ns + int(rng.integers(-spread, spread + 1)))
