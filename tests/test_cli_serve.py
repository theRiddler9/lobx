import sys
from pathlib import Path

import pytest

from lobx.cli import main


def test_serve_starts_uvicorn_with_the_given_address(monkeypatch: pytest.MonkeyPatch) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    assert main(["serve", "--host", "0.0.0.0", "--port", "9123", "--model", "m.json"]) == 0
    assert calls == [{"host": "0.0.0.0", "port": 9123}]


def test_serve_explains_how_to_install_the_extra_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # makes "import uvicorn" fail
    assert main(["serve", "--model", str(tmp_path / "m.json")]) == 1
    assert "pip install -e '.[service]'" in capsys.readouterr().err
