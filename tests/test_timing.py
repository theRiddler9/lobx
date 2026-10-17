"""Tests for the wake-time helpers."""

from numpy.random import default_rng

from lobx.sim.agents.timing import exponential_gap_ns, jittered_gap_ns


def test_exponential_gaps_average_close_to_the_mean_and_are_positive() -> None:
    rng = default_rng(1)
    gaps = [exponential_gap_ns(rng, 1_000_000) for _ in range(20_000)]
    assert min(gaps) >= 1
    assert abs(sum(gaps) / len(gaps) - 1_000_000) < 30_000  # within 3 %


def test_exponential_gap_is_at_least_one_even_for_a_tiny_mean() -> None:
    rng = default_rng(1)
    assert all(exponential_gap_ns(rng, 1) >= 1 for _ in range(1_000))


def test_jittered_gaps_stay_within_the_band() -> None:
    rng = default_rng(2)
    gaps = [jittered_gap_ns(rng, 1_000, 0.25) for _ in range(5_000)]
    assert min(gaps) >= 750 and max(gaps) <= 1_250
    assert len(set(gaps)) > 50  # it really varies


def test_jitter_of_zero_gives_the_exact_interval() -> None:
    rng = default_rng(3)
    assert {jittered_gap_ns(rng, 500, 0.0) for _ in range(100)} == {500}
