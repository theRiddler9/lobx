"""Shared fixtures. The ``engine`` fixture runs a test once per engine implementation."""

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.native import NativeOrderBook, native_available
from lobx.engine.protocol import OrderBook
from lobx.eval.harness import EvalConfig
from lobx.sim.clock import NS_PER_S

# A short scenario (80 simulated seconds) keeps service tests fast.
SHORT_CONFIG = EvalConfig(
    parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6
)

_NATIVE_MISSING = pytest.mark.skipif(
    not native_available(), reason="C++ engine not built (pip install -e .)"
)


@pytest.fixture(
    params=[
        pytest.param(PythonOrderBook, id="python"),
        pytest.param(NativeOrderBook, id="native", marks=_NATIVE_MISSING),
    ]
)
def engine(request: pytest.FixtureRequest) -> Callable[[], OrderBook]:
    """A factory that builds an empty book of the engine under test."""
    factory: Callable[[], OrderBook] = request.param
    return factory


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    """Run the real API on a free local port, in a thread, and give back its address."""
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from lobx.service.app import create_app

    app = create_app(SHORT_CONFIG, PythonOrderBook, Path("does-not-exist.json"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
