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


def _tiny_model() -> object:
    import numpy as np

    from lobx.ml.dataset import Dataset
    from lobx.ml.model import NaiveModel

    model = NaiveModel()
    y = np.array([0, 1, 2], dtype=np.int64)
    model.fit(Dataset(np.zeros((3, 1)), y, np.zeros(3, dtype=np.int64)))
    return model


def test_passive_and_adaptive_run_and_complete() -> None:
    config = EvalConfig(
        parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6
    )
    results = evaluate(["passive", "adaptive"], 2, config, model=_tiny_model(), threshold=0.5)  # type: ignore[arg-type]
    for runs in results.values():
        assert all(r.executed == config.parent_qty for r in runs)


def test_adaptive_without_a_model_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="needs a fitted model"):
        evaluate(["adaptive"], 2, SHORT)


def test_obtain_model_loads_an_existing_file(tmp_path: Path) -> None:
    from lobx.eval.runner import obtain_model
    from lobx.ml.store import save_model

    path = tmp_path / "m.json"
    save_model(_tiny_model(), path)  # type: ignore[arg-type]
    assert obtain_model(path) is not None
