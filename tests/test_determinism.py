"""A run is a pure function of (config, seed): same inputs, byte-identical event log."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lobx.engine.event_log import EventLog
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import MarketConfig


def _lines(seed: int, seconds: int) -> list[str]:
    config = MarketConfig(seed=seed, duration_ns=seconds * NS_PER_S)
    return list(Market(config).run().log.lines())


def test_same_seed_gives_identical_event_logs() -> None:
    assert _lines(seed=0, seconds=30) == _lines(seed=0, seconds=30)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_identical_for_several_seeds(seed: int) -> None:
    assert _lines(seed, 20) == _lines(seed, 20)


def test_different_seeds_give_different_logs() -> None:
    assert _lines(seed=0, seconds=20) != _lines(seed=1, seconds=20)


def test_the_run_is_not_trivially_empty() -> None:
    # Guards the tests above against passing because both runs produced nothing.
    assert len(_lines(seed=0, seconds=20)) > 500  # a 20 second run has about 950 events


def test_a_longer_run_extends_a_shorter_one_without_changing_the_past() -> None:
    """Duration only decides when to stop. If it leaked into behaviour, this would fail."""
    short = _lines(seed=5, seconds=10)[1:]  # [1:] drops the header, which records the duration
    long = _lines(seed=5, seconds=20)[1:]
    assert len(long) > len(short)
    assert long[: len(short)] == short


def test_written_file_survives_a_read_back_unchanged(tmp_path: Path) -> None:
    result = Market(MarketConfig(seed=9, duration_ns=15 * NS_PER_S)).run()
    path = tmp_path / "run.eventlog"
    result.log.write(path)
    assert list(EventLog.read(path).lines()) == list(result.log.lines())


def test_two_separate_processes_write_byte_identical_files(tmp_path: Path) -> None:
    """Catches hidden dependence on hash randomisation, e.g. iterating a set of agents."""
    files = []
    for hash_seed in ("1", "2"):
        out = tmp_path / f"hash{hash_seed}.eventlog"
        done = subprocess.run(
            [sys.executable, "-m", "lobx.cli", "run", "--seed", "11", "--duration", "10"]
            + ["--out", str(out)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        assert done.returncode == 0, done.stderr
        files.append(out.read_bytes())
    assert files[0] == files[1]
