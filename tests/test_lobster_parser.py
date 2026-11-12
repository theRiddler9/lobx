from pathlib import Path

import pytest

from lobx.data.lobster import (
    LobsterType,
    generate_lobster_sample,
    iter_lobster_messages,
    iter_lobster_orderbook,
    parse_lobster_time_ns,
)
from lobx.types import Side


def test_parse_lobster_time_ns_exact() -> None:
    # 34200 seconds = 9.5 hours
    # .017459617 = 17,459,617 nanoseconds
    ns = parse_lobster_time_ns("34200.017459617")
    expected = 34_200 * 1_000_000_000 + 17_459_617
    assert ns == expected

    # Edge cases
    assert parse_lobster_time_ns("100") == 100 * 1_000_000_000
    assert parse_lobster_time_ns("0.000000001") == 1
    assert parse_lobster_time_ns("0.5") == 500_000_000

    with pytest.raises(ValueError, match="empty"):
        parse_lobster_time_ns("")


def test_iter_lobster_messages(tmp_path: Path) -> None:
    msg_file = tmp_path / "sample_message.csv"
    lines = [
        "34200.010000000,1,1001,500,10000,1\n",
        "34200.020000000,1,1002,300,10010,-1\n",
        "34200.030000000,2,1001,200,10000,1\n",
        "34200.040000000,3,1002,300,10010,-1\n",
        "34200.050000000,4,1001,100,10000,1\n",
    ]
    msg_file.write_text("".join(lines), encoding="utf-8")

    messages = list(iter_lobster_messages(msg_file))
    assert len(messages) == 5

    m0 = messages[0]
    assert m0.event_type == LobsterType.SUBMISSION
    assert m0.order_id == 1001
    assert m0.qty == 500
    assert m0.price == 10000
    assert m0.side == Side.BUY

    m1 = messages[1]
    assert m1.event_type == LobsterType.SUBMISSION
    assert m1.order_id == 1002
    assert m1.qty == 300
    assert m1.price == 10010
    assert m1.side == Side.SELL

    m2 = messages[2]
    assert m2.event_type == LobsterType.CANCELLATION
    assert m2.qty == 200

    m3 = messages[3]
    assert m3.event_type == LobsterType.DELETION

    m4 = messages[4]
    assert m4.event_type == LobsterType.EXECUTION_VISIBLE


def test_iter_lobster_messages_with_base_time(tmp_path: Path) -> None:
    msg_file = tmp_path / "sample_msg.csv"
    msg_file.write_text("34200.100000000,1,10,100,1000,1\n", encoding="utf-8")

    base = 34_200 * 1_000_000_000
    messages = list(iter_lobster_messages(msg_file, base_time_ns=base))
    assert len(messages) == 1
    assert messages[0].ts_ns == 100_000_000


def test_iter_lobster_orderbook(tmp_path: Path) -> None:
    book_file = tmp_path / "sample_book.csv"
    # Format: AskP1, AskQ1, BidP1, BidQ1, AskP2, AskQ2, BidP2, BidQ2, ...
    row1 = "10020,500,10000,400,10030,600,9990,700\n"
    book_file.write_text(row1, encoding="utf-8")

    snaps = list(iter_lobster_orderbook(book_file, depth=2))
    assert len(snaps) == 1
    s = snaps[0]
    assert s.asks == ((10020, 500), (10030, 600))
    assert s.bids == ((10000, 400), (9990, 700))


def test_lobster_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_lobster_messages("nonexistent_lobster.csv"))
    with pytest.raises(FileNotFoundError):
        list(iter_lobster_orderbook("nonexistent_lobster_book.csv"))


def test_invalid_lobster_format(tmp_path: Path) -> None:
    bad_msg = tmp_path / "bad_msg.csv"
    bad_msg.write_text("34200,1,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 6 fields"):
        list(iter_lobster_messages(bad_msg))

    bad_book = tmp_path / "bad_book.csv"
    bad_book.write_text("1000,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least 20 fields"):
        list(iter_lobster_orderbook(bad_book, depth=5))


def test_generate_lobster_sample(tmp_path: Path) -> None:
    msg_path = tmp_path / "test_msgs.csv"
    book_path = tmp_path / "test_book.csv"

    generate_lobster_sample(msg_path, book_path, num_events=30)
    assert msg_path.exists()
    assert book_path.exists()

    msgs = list(iter_lobster_messages(msg_path))
    books = list(iter_lobster_orderbook(book_path, depth=5))

    assert len(msgs) == 30
    assert len(books) == 30
    assert len(books[0].asks) == 5
    assert len(books[0].bids) == 5
