"""Smoke test: the package imports and exposes a version string."""

import lobx


def test_version_is_a_string() -> None:
    assert isinstance(lobx.__version__, str)
    assert lobx.__version__ != ""
