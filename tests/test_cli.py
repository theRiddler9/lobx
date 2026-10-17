import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from lobx.cli import main


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_writes_an_event_log_and_prints_its_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "a.eventlog"
    assert main(["run", "--seed", "3", "--duration", "5", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert out.exists()
    assert out.read_text().startswith('{"type":"meta"')
    assert f"digest: {_digest(out)}" in text
    assert "seed 3, 5 simulated seconds" in text
    assert f"log: {out}" in text
    assert "final book: best bid" in text


def test_run_defaults_to_a_file_named_after_the_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--seed", "7", "--duration", "2"]) == 0
    assert (tmp_path / "run-seed-7.eventlog").exists()


def test_same_seed_writes_identical_files(tmp_path: Path) -> None:
    first, second = tmp_path / "1.eventlog", tmp_path / "2.eventlog"
    main(["run", "--seed", "5", "--duration", "5", "--out", str(first)])
    main(["run", "--seed", "5", "--duration", "5", "--out", str(second)])
    assert first.read_bytes() == second.read_bytes()


def test_different_seeds_write_different_files(tmp_path: Path) -> None:
    first, second = tmp_path / "1.eventlog", tmp_path / "2.eventlog"
    main(["run", "--seed", "5", "--duration", "5", "--out", str(first)])
    main(["run", "--seed", "6", "--duration", "5", "--out", str(second)])
    assert first.read_bytes() != second.read_bytes()


def test_negative_seed_is_reported_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--seed", "-1", "--out", str(tmp_path / "x.eventlog")]) == 1
    assert "seed must be a non-negative integer" in capsys.readouterr().err


def test_unwritable_output_path_is_reported_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_dir = tmp_path / "nope" / "x.eventlog"
    assert main(["run", "--duration", "1", "--out", str(missing_dir)]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_unknown_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as info:
        main(["frobnicate"])
    assert info.value.code == 2


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2


def test_module_entry_point_works_as_documented(tmp_path: Path) -> None:
    out = tmp_path / "m.eventlog"
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "lobx.cli",
            "run",
            "--seed",
            "0",
            "--duration",
            "2",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert out.exists()
    assert f"digest: {_digest(out)}" in done.stdout
