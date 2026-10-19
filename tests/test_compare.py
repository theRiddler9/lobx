from collections.abc import Sequence

import pytest

from lobx.eval.compare import comparison_table, paired_comparison, verdict
from lobx.eval.harness import SeedResult


def runs(values: Sequence[float | None]) -> list[SeedResult]:
    return [SeedResult(seed, 100, 20_000, v) for seed, v in enumerate(values)]


def test_differences_are_taken_seed_by_seed() -> None:
    c = paired_comparison("a", runs([10, 20, 30, 40]), "b", runs([12, 25, 33, 46]))
    assert c.used == 4 and c.excluded == 0
    assert c.difference.mean == pytest.approx(-4.0)  # -2, -5, -3, -6
    assert c.a_cheaper == 4


def test_seeds_incomplete_for_either_strategy_are_excluded_from_both() -> None:
    c = paired_comparison("a", runs([10, None, 30, 40]), "b", runs([12, 25, None, 46]))
    assert c.used == 2 and c.excluded == 2
    assert c.difference.mean == pytest.approx(-4.0)  # seeds 0 and 3 only


def test_pairing_removes_shared_luck() -> None:
    # A huge seed-to-seed swing that hits both strategies equally; a is always 1 bp cheaper.
    swing = [10.0, 500.0, 40.0, 900.0, 5.0]
    c = paired_comparison("a", runs([x - 1 for x in swing]), "b", runs(swing))
    assert c.difference.mean == pytest.approx(-1.0)
    assert c.difference.std == pytest.approx(0.0)


def test_different_seed_sets_are_rejected() -> None:
    with pytest.raises(ValueError, match="same seeds"):
        paired_comparison("a", runs([1, 2]), "b", runs([1, 2, 3]))


def test_table_gives_a_verdict_from_the_interval() -> None:
    results = {
        "twap": runs([10, 20, 30, 40, 50, 60]),
        "cheap": runs([8, 17, 27, 36, 47, 55]),  # always cheaper by 2-5
        "noisy": runs([12, 18, 33, 38, 52, 58]),  # sometimes cheaper, sometimes dearer
    }
    table = comparison_table(results)
    rows = {line.split("|")[1].strip(): line for line in table.splitlines()[2:]}
    assert "cheap cheaper" in rows["cheap - twap"]
    assert "no clear difference" in rows["noisy - twap"]
    assert set(rows) == {"cheap - twap", "noisy - twap", "noisy - cheap"}


def test_verdict_names_the_cheaper_strategy_or_admits_a_tie() -> None:
    a = runs([8, 17, 27, 36, 47, 55])
    b = runs([10, 20, 30, 40, 50, 60])
    assert verdict(paired_comparison("a", a, "b", b)) == "a cheaper"
    assert verdict(paired_comparison("b", b, "a", a)) == "a cheaper"  # order does not matter
    noisy = runs([12, 18, 33, 38, 52, 58])
    assert verdict(paired_comparison("n", noisy, "b", b)) == "no clear difference"
