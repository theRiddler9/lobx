"""The append-only event log: the single record of what happened in a run.

A log is a header (seed and config, so a run can be reproduced) followed by events in the order the
engine handled them. It can be written to and read from a ``.eventlog`` text file; the format is
specified in docs/contracts/shared-contracts.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from lobx.engine.event_codec import dumps, event_from_line, event_to_line, parse_object
from lobx.errors import EventLogFormatError
from lobx.types import Event, Fill, FillEvent

FORMAT_VERSION = 1


class EventLog:
    """An ordered, append-only list of events plus the header that describes the run."""

    def __init__(self, seed: int = 0, config: Mapping[str, Any] | None = None) -> None:
        """Create an empty log. ``seed`` and ``config`` are stored in the header only."""
        self.seed = seed
        self.config: dict[str, Any] = dict(config) if config is not None else {}
        self._events: list[Event] = []

    def append(self, event: Event) -> None:
        """Add an event at the end. There is deliberately no way to edit or remove one."""
        self._events.append(event)

    def __len__(self) -> int:
        """Return the number of events (the header is not counted)."""
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        """Iterate over the events in order."""
        return iter(self._events)

    def __getitem__(self, index: int) -> Event:
        """Return the event at ``index``."""
        return self._events[index]

    def fills(self) -> list[Fill]:
        """Return every trade in the log, in execution order."""
        return [event.fill for event in self._events if isinstance(event, FillEvent)]

    def header_line(self) -> str:
        """Return the header as one JSON line."""
        return dumps(
            {"type": "meta", "format": FORMAT_VERSION, "seed": self.seed, "config": self.config}
        )

    def lines(self) -> Iterator[str]:
        """Yield the header line, then one line per event (no trailing newlines)."""
        yield self.header_line()
        for event in self._events:
            yield event_to_line(event)

    def write(self, path: str | Path) -> None:
        """Write the log to ``path`` with unix newlines, so files match across platforms."""
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for line in self.lines():
                handle.write(line)
                handle.write("\n")

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> EventLog:
        """Rebuild a log from its lines. Errors name the 1-based line number."""
        log: EventLog | None = None
        for number, raw in enumerate(lines, start=1):
            line = raw.rstrip("\r\n")
            try:
                if log is None:
                    log = cls._from_header(line)
                else:
                    log.append(event_from_line(line))
            except EventLogFormatError as exc:
                raise EventLogFormatError(f"line {number}: {exc}") from exc
        if log is None:
            raise EventLogFormatError("the log is empty: it needs a header line")
        return log

    @classmethod
    def read(cls, path: str | Path) -> EventLog:
        """Read a log written by :meth:`write`."""
        with open(path, encoding="utf-8", newline="") as handle:
            return cls.from_lines(handle)

    @classmethod
    def _from_header(cls, line: str) -> EventLog:
        data = parse_object(line)
        if data.get("type") != "meta":
            raise EventLogFormatError("the first line must be the meta header")
        if data.get("format") != FORMAT_VERSION:
            raise EventLogFormatError(
                f"unsupported log format {data.get('format')!r}, expected {FORMAT_VERSION}"
            )
        seed, config = data.get("seed"), data.get("config")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EventLogFormatError(f"header seed must be an integer, got {seed!r}")
        if not isinstance(config, dict):
            raise EventLogFormatError("header config must be a JSON object")
        return cls(seed=seed, config=config)
