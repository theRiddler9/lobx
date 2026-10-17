"""Turn events into JSON lines and back. The file format is specified in the shared contracts.

One event is one line of compact JSON. The key order is fixed by the order the keys are written in
below, and ``json.dumps`` keeps insertion order, so the same run always produces the same bytes.
That is what lets a test compare two runs with ``==`` on the files.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, TypeVar

from lobx.errors import EventLogFormatError
from lobx.types import CancelEvent, Event, Fill, FillEvent, Order, OrderType, Side, SubmitEvent

_E = TypeVar("_E", bound=Enum)
_MISSING = object()


def dumps(obj: dict[str, Any]) -> str:
    """Serialise ``obj`` as one compact JSON line (no spaces, no trailing newline)."""
    return json.dumps(obj, separators=(",", ":"))


def parse_object(line: str) -> dict[str, Any]:
    """Parse one JSON line and require it to be an object."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise EventLogFormatError(f"not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise EventLogFormatError(f"expected a JSON object, got {type(data).__name__}")
    return data


def event_to_line(event: Event) -> str:
    """Serialise one event. Enums are written by name, a market order's price as ``null``."""
    if isinstance(event, SubmitEvent):
        order = event.order
        return dumps(
            {
                "type": "submit",
                "order_id": order.order_id,
                "agent_id": order.agent_id,
                "side": order.side.name,
                "order_type": order.type.name,
                "price": order.price,
                "qty": order.qty,
                "ts": order.ts,
            }
        )
    if isinstance(event, CancelEvent):
        return dumps(
            {
                "type": "cancel",
                "order_id": event.order_id,
                "ts": event.ts,
                "cancelled": event.cancelled,
            }
        )
    fill = event.fill
    return dumps(
        {
            "type": "fill",
            "maker_order_id": fill.maker_order_id,
            "taker_order_id": fill.taker_order_id,
            "taker_side": fill.taker_side.name,
            "price": fill.price,
            "qty": fill.qty,
            "ts": fill.ts,
            "self_trade": event.self_trade,
        }
    )


def event_from_line(line: str) -> Event:
    """Parse one event line. Raises ``EventLogFormatError`` if anything is off."""
    data = parse_object(line)
    kind = data.get("type")
    if kind == "submit":
        return SubmitEvent(
            Order(
                order_id=_int(data, "order_id"),
                agent_id=_int(data, "agent_id"),
                side=_enum(data, "side", Side),
                type=_enum(data, "order_type", OrderType),
                price=_optional_int(data, "price"),
                qty=_int(data, "qty"),
                ts=_int(data, "ts"),
            )
        )
    if kind == "cancel":
        return CancelEvent(
            order_id=_int(data, "order_id"),
            ts=_int(data, "ts"),
            cancelled=_bool(data, "cancelled"),
        )
    if kind == "fill":
        return FillEvent(
            fill=Fill(
                maker_order_id=_int(data, "maker_order_id"),
                taker_order_id=_int(data, "taker_order_id"),
                taker_side=_enum(data, "taker_side", Side),
                price=_int(data, "price"),
                qty=_int(data, "qty"),
                ts=_int(data, "ts"),
            ),
            self_trade=_bool(data, "self_trade"),
        )
    raise EventLogFormatError(f"unknown event type {kind!r}")


def _get(data: dict[str, Any], key: str) -> Any:
    value = data.get(key, _MISSING)
    if value is _MISSING:
        raise EventLogFormatError(f"missing field {key!r}")
    return value


def _int(data: dict[str, Any], key: str) -> int:
    value = _get(data, key)
    # bool is a subclass of int in Python, but a JSON true/false is never a valid number here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise EventLogFormatError(f"field {key!r} must be an integer, got {value!r}")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    return None if _get(data, key) is None else _int(data, key)


def _bool(data: dict[str, Any], key: str) -> bool:
    value = _get(data, key)
    if not isinstance(value, bool):
        raise EventLogFormatError(f"field {key!r} must be true or false, got {value!r}")
    return value


def _enum(data: dict[str, Any], key: str, enum_type: type[_E]) -> _E:
    value = _get(data, key)
    if not isinstance(value, str) or value not in enum_type.__members__:
        names = ", ".join(enum_type.__members__)
        raise EventLogFormatError(f"field {key!r} must be one of {names}, got {value!r}")
    return enum_type[value]
