from pathlib import Path

import pytest

from lobx.cli import main
from lobx.ml.store import load_model


def test_train_prints_a_table_and_saves_loadable_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["train", "--train-seeds", "6", "--validation-seeds", "2", "--test-seeds", "2"]
    assert main([*args, "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for name in ("naive", "logistic", "lightgbm"):
        assert f"| {name} |" in out
        assert load_model(tmp_path / f"{name}.json") is not None
