"""LOBX: a limit-order-book exchange simulator with an ML-driven execution agent."""

from __future__ import annotations

import contextlib
import os
import sys

if sys.platform == "win32":
    for _p in (r"C:\msys64\ucrt64\bin", r"C:\msys64\mingw64\bin"):
        if os.path.isdir(_p):
            with contextlib.suppress(OSError):
                os.add_dll_directory(_p)

__version__ = "0.1.0"
