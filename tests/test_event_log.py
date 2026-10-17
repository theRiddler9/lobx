"""Tests for the append-only event log and its file format."""

from pathlib import Path

import pytest

from lobx.engine.event_log import EventLog
from lobx.errors import EventLogFormatError
from lobx.types import CancelEvent, Fill, FillEvent, Order, OrderType, Side, SubmitEvent

_ORDER = Order(1, 2, Side.BUY, OrderType.LIMIT, 100, 5, 10)
_FILL = Fill(1, 2, Side.SELL, 100, 3, 20)


def _sample_log() -> EventLog:
    log = EventLog(seed=42, config={"horizon_ns": 1000, "name": "demo"})
    log.append(SubmitEvent(_ORDER))
    log.append(FillEvent(_FILL, self_trade=False))
    log.append(CancelEvent(1, 30, cancelled=True))
    return log


def test_new_log_is_empty_and_keeps_its_header() -> None:
    log = EventLog(seed=7, config={"a": 1})
    assert len(log) == 0
    assert (log.seed, log.config) == (7, {"a": 1})


def test_events_keep_their_order_and_can_be_indexed() -> None:
    log = _sample_log()
    assert len(log) == 3
    assert [type(e).__name__ for e in log] == ["SubmitEvent", "FillEvent", "CancelEvent"]
    assert log[1] == FillEvent(_FILL, self_trade=False)


def test_fills_returns_only_trades() -> None:
    assert _sample_log().fills() == [_FILL]


def test_header_is_the_first_line() -> None:
    first = next(_sample_log().lines())
    assert first == (
        '{"type":"meta","format":1,"seed":42,"config":{"horizon_ns":1000,"name":"demo"}}'
    )


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "run.eventlog"
    original = _sample_log()
    original.write(path)
    loaded = EventLog.read(path)
    assert (loaded.seed, loaded.config) == (original.seed, original.config)
    assert list(loaded) == list(original)


def test_file_has_unix_newlines_and_one_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "run.eventlog"
    _sample_log().write(path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 4  # header + 3 events


def test_same_log_gives_identical_bytes(tmp_path: Path) -> None:
    _sample_log().write(tmp_path / "a.eventlog")
    _sample_log().write(tmp_path / "b.eventlog")
    assert (tmp_path / "a.eventlog").read_bytes() == (tmp_path / "b.eventlog").read_bytes()


def test_empty_input_is_rejected() -> None:
    with pytest.raises(EventLogFormatError, match="empty"):
        EventLog.from_lines([])


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ('{"type":"submit"}', "first line must be the meta header"),
        ('{"type":"meta","format":2,"seed":0,"config":{}}', "unsupported log format"),
        ('{"type":"meta","format":1,"seed":"x","config":{}}', "seed must be an integer"),
        ('{"type":"meta","format":1,"seed":0,"config":[]}', "config must be a JSON object"),
    ],
)
def test_bad_headers_are_rejected(header: str, message: str) -> None:
    with pytest.raises(EventLogFormatError, match=message):
        EventLog.from_lines([header])


def test_error_names_the_line_number() -> None:
    lines = [*_sample_log().lines()][:2] + ['{"type":"explode"}']
    with pytest.raises(EventLogFormatError, match=r"line 3: unknown event type"):
        EventLog.from_lines(lines)
