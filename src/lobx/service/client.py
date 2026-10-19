"""A small client for the LOBX API, used by the Streamlit dashboard.

It speaks plain HTTP with the standard library and WebSocket with the ``websockets`` package, so
the dashboard needs nothing else. Keeping it here (not in ``dashboard/app.py``) means it can be
tested against a real running server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any

from websockets.sync.client import connect

from lobx.errors import LobxError

JSON = dict[str, Any]
REQUEST_TIMEOUT_S = 30.0


class ApiError(LobxError):
    """The API answered with an error status, or could not be reached."""


def depth_rows(message: Mapping[str, Any]) -> list[dict[str, int | None]]:
    """Turn a ``snapshot`` message into chart rows, one per price, sorted by price.

    Each row is ``{"price": p, "bids": qty or None, "asks": qty or None}``. A bid and an ask never
    share a price in a book that is not crossed, so a row has at most one quantity.
    """
    rows: dict[int, dict[str, int | None]] = {}
    for side in ("bids", "asks"):
        for price, qty in message[side]:
            rows.setdefault(price, {"price": price, "bids": None, "asks": None})[side] = qty
    return [rows[price] for price in sorted(rows)]


def spread_of(message: Mapping[str, Any]) -> int | None:
    """Return best ask minus best bid in ticks, or ``None`` if a side is empty."""
    if not message["bids"] or not message["asks"]:
        return None
    return int(message["asks"][0][0] - message["bids"][0][0])


class ApiClient:
    """Talks to one LOBX server, for example ``ApiClient("http://localhost:8000")``."""

    def __init__(self, base_url: str) -> None:
        """Remember the server address (a trailing slash is ignored)."""
        self.base_url = base_url.rstrip("/")

    def health(self) -> JSON:
        """Return ``GET /health``."""
        return self._call("GET", "/health")

    def start_run(self, request: Mapping[str, Any]) -> JSON:
        """Start a run (``POST /runs``) and return its info."""
        return self._call("POST", "/runs", request)

    def get_run(self, run_id: str) -> JSON:
        """Return ``GET /runs/{id}``."""
        return self._call("GET", f"/runs/{run_id}")

    def start_comparison(self, request: Mapping[str, Any]) -> JSON:
        """Start a comparison (``POST /comparisons``) and return its info."""
        return self._call("POST", "/comparisons", request)

    def get_comparison(self, comparison_id: str) -> JSON:
        """Return ``GET /comparisons/{id}``."""
        return self._call("GET", f"/comparisons/{comparison_id}")

    def stream(self, run_id: str, delay_ms: int = 50) -> Iterator[JSON]:
        """Yield the run's WebSocket messages: every snapshot, then the final ``done`` message."""
        url = self.base_url.replace("http", "ws", 1) + f"/runs/{run_id}/stream?delay_ms={delay_ms}"
        try:
            with connect(url, open_timeout=REQUEST_TIMEOUT_S) as socket:
                for raw in socket:
                    message: JSON = json.loads(raw)
                    yield message
                    if message["type"] == "done":
                        return
        except OSError as error:
            raise ApiError(f"cannot stream {url}: {error}") from error

    def _call(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> JSON:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                result: JSON = json.load(response)
                return result
        except urllib.error.HTTPError as error:
            raise ApiError(f"{method} {path}: {error.code} {_detail(error)}") from error
        except urllib.error.URLError as error:
            raise ApiError(f"cannot reach {self.base_url}: {error.reason}") from error


def _detail(error: urllib.error.HTTPError) -> str:
    """Pull the server's ``detail`` message out of an error response, if it has one."""
    try:
        return str(json.load(error).get("detail", error.reason))
    except (ValueError, AttributeError):
        return str(error.reason)
