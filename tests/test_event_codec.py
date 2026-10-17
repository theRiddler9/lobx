"""Tests for turning events into JSON lines and back."""

import pytest

from lobx.engine.event_codec import event_from_line, event_to_line
from lobx.errors import EventLogFormatError
from lobx.types import CancelEvent, Event, Fill, FillEvent, Order, OrderType, Side, SubmitEvent

_LIMIT = Order(7, 3, Side.BUY, OrderType.LIMIT, 10_050, 5, 1_000)
_MARKET = Order(8, 4, Side.SELL, OrderType.MARKET, None, 2, 2_000)
_FILL = Fill(7, 8, Side.SELL, 10_050, 2, 2_000)

_EVENTS: list[Event] = [
    SubmitEvent(_LIMIT),
    SubmitEvent(_MARKET),
    FillEvent(_FILL, self_trade=False),
    FillEvent(_FILL, self_trade=True),
    CancelEvent(7, 3_000, cancelled=True),
    CancelEvent(99, 4_000, cancelled=False),
]


@pytest.mark.parametrize("event", _EVENTS)
def test_round_trip(event: Event) -> None:
    assert event_from_line(event_to_line(event)) == event


def test_submit_line_is_exactly_this() -> None:
    # The exact bytes matter: two runs are compared by comparing their files.
    assert event_to_line(SubmitEvent(_LIMIT)) == (
        '{"type":"submit","order_id":7,"agent_id":3,"side":"BUY","order_type":"LIMIT",'
        '"price":10050,"qty":5,"ts":1000}'
    )


def test_market_order_price_is_null() -> None:
    assert '"price":null' in event_to_line(SubmitEvent(_MARKET))


def test_fill_and_cancel_lines_are_exactly_this() -> None:
    assert event_to_line(FillEvent(_FILL, self_trade=True)) == (
        '{"type":"fill","maker_order_id":7,"taker_order_id":8,"taker_side":"SELL",'
        '"price":10050,"qty":2,"ts":2000,"self_trade":true}'
    )
    assert event_to_line(CancelEvent(7, 3_000, cancelled=False)) == (
        '{"type":"cancel","order_id":7,"ts":3000,"cancelled":false}'
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"type":"explode"}', "unknown event type"),
        ("{}", "unknown event type"),
        ('{"type":"cancel","ts":1,"cancelled":true}', "missing field 'order_id'"),
        ('{"type":"cancel","order_id":1.5,"ts":1,"cancelled":true}', "must be an integer"),
        ('{"type":"cancel","order_id":true,"ts":1,"cancelled":true}', "must be an integer"),
        ('{"type":"cancel","order_id":1,"ts":1,"cancelled":"yes"}', "true or false"),
        (
            '{"type":"submit","order_id":1,"agent_id":1,"side":"HOLD","order_type":"LIMIT",'
            '"price":1,"qty":1,"ts":0}',
            "must be one of BUY, SELL",
        ),
        (
            '{"type":"submit","order_id":1,"agent_id":1,"side":"BUY","order_type":"LIMIT",'
            '"price":100.5,"qty":1,"ts":0}',
            "must be an integer",
        ),
    ],
)
def test_bad_lines_are_rejected_with_a_clear_message(line: str, message: str) -> None:
    with pytest.raises(EventLogFormatError, match=message):
        event_from_line(line)
