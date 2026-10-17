"""Replaying the log of a real simulated run must rebuild the final book exactly."""

from pathlib import Path

import pytest

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.event_log import EventLog
from lobx.engine.replay import replay
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import MarketConfig

_EVERYTHING = 1_000_000  # deeper than any book here, so a snapshot shows every resting order


def _run(seed: int, seconds: int = 30) -> tuple[Market, EventLog]:
    market = Market(MarketConfig(seed=seed, duration_ns=seconds * NS_PER_S))
    return market, market.run().log


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_replay_reproduces_the_final_book_exactly(seed: int) -> None:
    market, log = _run(seed)
    rebuilt = replay(log)
    assert rebuilt.snapshot(_EVERYTHING) == market.snapshot(_EVERYTHING)
    assert rebuilt.best_bid() == market.best_bid()
    assert rebuilt.best_ask() == market.best_ask()


def test_the_final_book_is_not_trivially_empty() -> None:
    market, _ = _run(0)
    full = market.snapshot(_EVERYTHING)
    assert len(full.bids) > 3 and len(full.asks) > 3


def test_replay_works_after_a_round_trip_through_a_file(tmp_path: Path) -> None:
    market, log = _run(6)
    path = tmp_path / "run.eventlog"
    log.write(path)
    rebuilt = replay(EventLog.read(path))
    assert rebuilt.snapshot(_EVERYTHING) == market.snapshot(_EVERYTHING)


def test_replay_visits_every_event_in_order() -> None:
    _, log = _run(3, seconds=10)
    seen = []
    replay(log, on_event=lambda event, book: seen.append(event))
    assert seen == list(log)


def test_replay_into_an_explicit_engine_fills_that_engine() -> None:
    market, log = _run(4, seconds=10)
    engine = PythonOrderBook()
    returned = replay(log, book=engine)
    assert returned is engine
    assert engine.snapshot(_EVERYTHING) == market.snapshot(_EVERYTHING)
