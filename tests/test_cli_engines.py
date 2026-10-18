"""The CLI's two engines must produce byte-identical event logs for the same seed."""

import hashlib
from pathlib import Path

import pytest

from lobx.cli import main
from lobx.engine.native import native_available


@pytest.mark.skipif(not native_available(), reason="C++ engine not built")
def test_python_and_native_engines_write_identical_logs(tmp_path: Path) -> None:
    digests = []
    for engine in ("python", "native"):
        out = tmp_path / f"{engine}.eventlog"
        assert (
            main(["run", "--seed", "3", "--duration", "30", "--engine", engine, "--out", str(out)])
            == 0
        )
        digests.append(hashlib.sha256(out.read_bytes()).hexdigest())
    assert digests[0] == digests[1]
