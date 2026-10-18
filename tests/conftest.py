"""Shared fixtures. The ``engine`` fixture runs a test once per engine implementation."""

from collections.abc import Callable

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.native import NativeOrderBook, native_available
from lobx.engine.protocol import OrderBook

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
