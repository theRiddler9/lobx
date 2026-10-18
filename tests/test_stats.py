import math

import pytest

from lobx.eval.stats import summarize, t_critical_95


@pytest.mark.parametrize(
    "df,expected",
    [(1, 12.706), (4, 2.776), (5, 2.571), (10, 2.228), (30, 2.042), (100, 1.984), (199, 1.972)],
)
def test_t_critical_matches_published_tables(df: int, expected: float) -> None:
    assert t_critical_95(df) == pytest.approx(expected, abs=0.004)


def test_summary_of_a_hand_computed_sample() -> None:
    # Values 2, 4, 4, 4, 5, 5, 7, 9: mean 5, squared deviations sum to 32, sample variance 32/7.
    s = summarize([2, 4, 4, 4, 5, 5, 7, 9])
    assert s.n == 8 and s.mean == 5.0
    assert s.std == pytest.approx(math.sqrt(32 / 7))
    half = 2.365 * s.std / math.sqrt(8)  # t(0.975, df=7) = 2.365
    assert s.ci_low == pytest.approx(5 - half, abs=0.01)
    assert s.ci_high == pytest.approx(5 + half, abs=0.01)


def test_identical_values_have_zero_spread() -> None:
    s = summarize([3.0, 3.0, 3.0])
    assert s.std == 0.0 and s.ci_low == s.ci_high == 3.0


def test_the_interval_narrows_as_runs_are_added() -> None:
    few = summarize([1.0, 3.0] * 5)
    many = summarize([1.0, 3.0] * 500)
    assert (many.ci_high - many.ci_low) < (few.ci_high - few.ci_low)


def test_needs_two_values() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        summarize([1.0])
