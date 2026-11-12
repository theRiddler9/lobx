from pathlib import Path

from lobx.data.lobster import (
    LobsterType,
    generate_lobster_sample,
    iter_lobster_messages,
    iter_lobster_orderbook,
)
from lobx.engine.book_py import PythonOrderBook
from lobx.sim.replay_market import ReplayMarket


def test_lobster_event_stream_reconstructs_valid_book(tmp_path: Path) -> None:
    msg_path = tmp_path / "recon_msg.csv"
    book_path = tmp_path / "recon_book.csv"

    generate_lobster_sample(
        msg_path,
        book_path,
        num_events=50,
        start_time_s=34200.0,
        seed_price=10_000,
    )

    market = ReplayMarket(message_path=msg_path, book=PythonOrderBook())
    result = market.run()

    assert result.events_replayed == 50
    # Final book should have positive spread
    best_bid = market.best_bid()
    best_ask = market.best_ask()
    assert best_bid is not None
    assert best_ask is not None
    assert best_ask > best_bid

    # Verify depth snapshot is populated
    snap = market.snapshot(depth=5)
    assert len(snap.bids) > 0
    assert len(snap.asks) > 0


def test_lobster_paired_file_iteration(tmp_path: Path) -> None:
    msg_path = tmp_path / "iter_msg.csv"
    book_path = tmp_path / "iter_book.csv"

    generate_lobster_sample(msg_path, book_path, num_events=20)

    msgs = list(iter_lobster_messages(msg_path))
    books = list(iter_lobster_orderbook(book_path, depth=5))

    assert len(msgs) == 20
    assert len(books) == 20

    # Ensure message types match expected enums
    types = {m.event_type for m in msgs}
    assert LobsterType.SUBMISSION in types
