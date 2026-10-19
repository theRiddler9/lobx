"""The whole pipeline through the command line, the way the golden demo runs it."""

from pathlib import Path

import pytest

from lobx.cli import main


def test_run_then_replay_reproduces_the_final_book(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "demo.eventlog"
    assert main(["run", "--seed", "0", "--duration", "20", "--out", str(log)]) == 0
    ran = capsys.readouterr().out
    assert main(["replay", str(log)]) == 0
    replayed = capsys.readouterr().out
    final_book = next(line for line in ran.splitlines() if line.startswith("final book"))
    assert final_book in replayed  # the replay ends in exactly the book the run ended in


def test_eval_gives_a_results_table_and_a_paired_table_and_is_reproducible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = ["eval", "--strategies", "twap,passive", "--seeds", "3", "--plot"]
    assert main([*command, str(tmp_path / "a.png")]) == 0
    first = capsys.readouterr().out
    assert main([*command, str(tmp_path / "b.png")]) == 0
    second = capsys.readouterr().out

    assert "| strategy | seeds | incomplete | mean (bps) | std (bps) | 95% CI" in first
    assert "| twap |" in first and "| passive |" in first
    assert "| passive - twap |" in first  # the paired comparison
    assert (tmp_path / "a.png").stat().st_size > 1_000
    # A run is a pure function of (config, seed): the same command prints the same numbers.
    assert first.replace("a.png", "x") == second.replace("b.png", "x")
