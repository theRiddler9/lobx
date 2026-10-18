from pathlib import Path

import pytest

from lobx.cli import main
from lobx.eval.harness import EvalConfig
from lobx.eval.runner import evaluate
from lobx.eval.seeds import TEST_SEEDS
from lobx.sim.clock import NS_PER_S

SHORT = EvalConfig(parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6)


def test_all_strategies_run_on_the_same_test_seeds() -> None:
    results = evaluate(["twap", "vwap"], 3, SHORT)
    assert list(results) == ["twap", "vwap"]
    for runs in results.values():
        assert [r.seed for r in runs] == [TEST_SEEDS.start + i for i in range(3)]
        assert all(r.shortfall_bps is not None for r in runs)


def test_unknown_strategy_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown strategy 'magic'"):
        evaluate(["magic"], 3, SHORT)


def test_needs_at_least_two_seeds() -> None:
    with pytest.raises(ValueError, match="2 seeds"):
        evaluate(["twap"], 1, SHORT)


def test_cli_eval_prints_a_table_and_writes_a_plot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plot = tmp_path / "eval.png"
    assert main(["eval", "--strategies", "twap", "--seeds", "3", "--plot", str(plot)]) == 0
    out = capsys.readouterr().out
    assert "| twap | 3 | 0 |" in out
    assert plot.exists()


def test_cli_eval_reports_an_unknown_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--strategies", "nope", "--seeds", "3"]) == 1
    assert "unknown strategy" in capsys.readouterr().err
