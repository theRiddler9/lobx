import re
import subprocess
import sys
from pathlib import Path

import pytest

from lobx.cli import main


def _run(tmp_path: Path, seed: int = 0, duration: int = 10) -> Path:
    out = tmp_path / f"seed{seed}.eventlog"
    assert main(["run", "--seed", str(seed), "--duration", str(duration), "--out", str(out)]) == 0
    return out


def _field(text: str, key: str) -> str:
    """Return the value on the line that starts with ``key:`` in the CLI's output."""
    match = re.search(rf"^{key}: (.*)$", text, re.MULTILINE)
    assert match is not None, f"no '{key}:' line in {text!r}"
    return match.group(1)


def test_replay_accepts_a_log_written_by_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _run(tmp_path)
    capsys.readouterr()
    assert main(["replay", str(log)]) == 0
    text = capsys.readouterr().out
    assert "replayed" in text and "events" in text
    assert "every fill, self-trade flag and cancel result matched the log" in text


def test_replay_reports_the_same_final_book_and_digest_as_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _run(tmp_path, seed=4, duration=15)
    run_text = capsys.readouterr().out
    main(["replay", str(log)])
    replay_text = capsys.readouterr().out
    assert _field(replay_text, "final book") == _field(run_text, "final book")
    assert _field(replay_text, "digest") == _field(run_text, "digest")


def test_replay_event_count_matches_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = _run(tmp_path, seed=2)
    run_events = int(_field(capsys.readouterr().out, "events").split()[0])
    main(["replay", str(log)])
    assert f"replayed {run_events} events" in capsys.readouterr().out


def test_replay_detects_a_log_with_a_missing_fill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _run(tmp_path)
    lines = log.read_text().splitlines()
    first_fill = next(i for i, line in enumerate(lines) if '"type":"fill"' in line)
    del lines[first_fill]
    log.write_text("\n".join(lines) + "\n")
    capsys.readouterr()
    assert main(["replay", str(log)]) == 1
    assert "error:" in capsys.readouterr().err


def test_replay_detects_a_doctored_price(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _run(tmp_path)
    lines = log.read_text().splitlines()
    first_fill = next(i for i, line in enumerate(lines) if '"type":"fill"' in line)
    lines[first_fill] = re.sub(
        r'"price":(\d+)', lambda m: f'"price":{int(m.group(1)) + 1}', lines[first_fill]
    )
    log.write_text("\n".join(lines) + "\n")
    capsys.readouterr()
    assert main(["replay", str(log)]) == 1
    assert "fill differs" in capsys.readouterr().err


def test_replay_of_a_missing_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["replay", str(tmp_path / "absent.eventlog")]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_replay_of_a_garbled_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.eventlog"
    bad.write_text("this is not a log\n")
    assert main(["replay", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_replay_works_through_the_module_entry_point(tmp_path: Path) -> None:
    log = _run(tmp_path, duration=3)
    done = subprocess.run(
        [sys.executable, "-m", "lobx.cli", "replay", str(log)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "matched the log" in done.stdout
