from pathlib import Path

from lobx.eval.harness import SeedResult
from lobx.eval.report import plot_results, results_table, summarize_results


def _results(values: list[float | None]) -> list[SeedResult]:
    return [SeedResult(seed, 100, 20_000, v) for seed, v in enumerate(values)]


def test_incomplete_runs_are_counted_not_dropped() -> None:
    summary, incomplete = summarize_results(_results([10.0, 20.0, None, 30.0]))
    assert summary.n == 3 and summary.mean == 20.0 and incomplete == 1


def test_table_has_one_row_per_strategy() -> None:
    table = results_table({"twap": _results([10.0, 20.0]), "vwap": _results([1.0, 3.0, 5.0])})
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, two strategies
    assert lines[2].startswith("| twap | 2 | 0 | 15.00 |")
    assert lines[3].startswith("| vwap | 3 | 0 | 3.00 | 2.00 |")


def test_plot_writes_a_png(tmp_path: Path) -> None:
    out = tmp_path / "plot.png"
    plot_results({"twap": _results([10.0, 20.0, 30.0])}, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
