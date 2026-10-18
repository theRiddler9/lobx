"""Run the hand-computed scenarios in tests/fixtures/book_scenarios.json against the engine.

The expected values were worked out on paper, not copied from the engine's output, so these tests
check the engine against the rules and not against itself. Both engines run the same file.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lobx.engine.protocol import OrderBook
from lobx.types import Fill, Order, OrderType, Side

_FIXTURE = Path(__file__).parent / "fixtures" / "book_scenarios.json"
_SCENARIOS: list[dict[str, Any]] = json.loads(_FIXTURE.read_text())["scenarios"]


def _side(name: str) -> Side:
    # The fixture writes sides in lowercase ("buy"); the enum members are BUY and SELL.
    return Side[name.upper()]


def _expected_fills(step: dict[str, Any]) -> list[Fill]:
    return [
        Fill(maker, taker, _side(taker_side), price, qty, step["ts"])
        for maker, taker, taker_side, price, qty in step["fills"]
    ]


def _order_from(step: dict[str, Any]) -> Order:
    order_type = OrderType.LIMIT if step["op"] == "limit" else OrderType.MARKET
    return Order(
        order_id=step["id"],
        agent_id=step["agent"],
        side=_side(step["side"]),
        type=order_type,
        price=step.get("price"),
        qty=step["qty"],
        ts=step["ts"],
    )


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["name"] for s in _SCENARIOS])
def test_scenario(scenario: dict[str, Any], engine: Callable[[], OrderBook]) -> None:
    book = engine()
    for step in scenario["steps"]:
        if step["op"] == "cancel":
            assert book.cancel(step["id"], step["ts"]) is step["result"]
        else:
            assert book.submit(_order_from(step)) == _expected_fills(step), step

    final = scenario["final"]
    snapshot = book.snapshot(final["depth"])
    assert snapshot.ts == final["ts"]
    assert book.best_bid() == final["best_bid"]
    assert book.best_ask() == final["best_ask"]
    assert snapshot.bids == tuple(tuple(level) for level in final["bids"])
    assert snapshot.asks == tuple(tuple(level) for level in final["asks"])
