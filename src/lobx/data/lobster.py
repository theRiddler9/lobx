"""LOBSTER (Limit Order Book System - The Efficient Reconstructor) data parser.

Ingests NASDAQ historical order book event streams and depth snapshots according
to the LOBSTER benchmark data specifications.

LOBSTER format reference:
- Message file columns:
    Time (sec since midnight), Type, OrderID, Size, Price, Direction
    - Type 1: Submission of a new limit order
    - Type 2: Cancellation (Partial deletion of a limit order)
    - Type 3: Deletion (Total cancellation of a limit order)
    - Type 4: Execution of a visible limit order
    - Type 5: Execution of a hidden limit order
    - Type 6: Cross trade
    - Type 7: Trading halt indicator
    - Direction 1: Buy, -1: Sell
- Orderbook file columns (depth K):
    Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1, ... up to depth K
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from lobx.types import Side


class LobsterType(IntEnum):
    """LOBSTER event types."""

    SUBMISSION = 1
    CANCELLATION = 2
    DELETION = 3
    EXECUTION_VISIBLE = 4
    EXECUTION_HIDDEN = 5
    CROSS_TRADE = 6
    HALT = 7


def parse_lobster_time_ns(time_str: str) -> int:
    """Parse decimal seconds string into exact integer nanoseconds without float drift.

    Example:
        '34200.017459617' -> 34,200 * 1,000,000,000 + 17,459,617 ns.
    """
    time_str = time_str.strip()
    if not time_str:
        raise ValueError("time_str cannot be empty")
    parts = time_str.split(".")
    seconds = int(parts[0])
    if len(parts) == 1:
        return seconds * 1_000_000_000
    frac_str = parts[1].ljust(9, "0")[:9]
    return seconds * 1_000_000_000 + int(frac_str)


@dataclass(frozen=True, slots=True)
class LobsterMessage:
    """A single parsed LOBSTER historical order book event."""

    ts_ns: int
    event_type: LobsterType
    order_id: int
    qty: int
    price: int
    side: Side


@dataclass(frozen=True, slots=True)
class LobsterBookSnapshot:
    """A single parsed LOBSTER order book depth snapshot."""

    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]


def iter_lobster_messages(
    file_path: Path | str, base_time_ns: int | None = None
) -> Iterator[LobsterMessage]:
    """Stream parsed LOBSTER messages line-by-line from a CSV file.

    Args:
        file_path: Path to LOBSTER message file.
        base_time_ns: Optional epoch offset to subtract, producing relative timestamps.

    Yields:
        LobsterMessage events.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"LOBSTER message file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 6:
                raise ValueError(
                    f"Invalid LOBSTER message at line {line_num}: "
                    f"expected 6 fields, got {len(fields)}"
                )

            ts_raw = parse_lobster_time_ns(fields[0])
            ts_ns = ts_raw - base_time_ns if base_time_ns is not None else ts_raw
            event_type = LobsterType(int(fields[1]))
            order_id = int(fields[2])
            qty = int(fields[3])
            price = int(fields[4])
            direction = int(fields[5])

            side = Side.BUY if direction == 1 else Side.SELL

            yield LobsterMessage(
                ts_ns=ts_ns,
                event_type=event_type,
                order_id=order_id,
                qty=qty,
                price=price,
                side=side,
            )


def iter_lobster_orderbook(file_path: Path | str, depth: int = 5) -> Iterator[LobsterBookSnapshot]:
    """Stream parsed LOBSTER orderbook depth snapshots line-by-line from a CSV file.

    Args:
        file_path: Path to LOBSTER orderbook file.
        depth: Depth level K (each level contains 4 numbers: Ask P, Ask Q, Bid P, Bid Q).

    Yields:
        LobsterBookSnapshot objects.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"LOBSTER orderbook file not found: {path}")
    if depth < 1:
        raise ValueError("depth must be at least 1")

    expected_fields = depth * 4

    with path.open("r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < expected_fields:
                raise ValueError(
                    f"Invalid LOBSTER orderbook line {line_num}: "
                    f"expected at least {expected_fields} fields"
                )

            asks: list[tuple[int, int]] = []
            bids: list[tuple[int, int]] = []

            for i in range(depth):
                base_idx = i * 4
                ask_px = int(fields[base_idx])
                ask_qty = int(fields[base_idx + 1])
                bid_px = int(fields[base_idx + 2])
                bid_qty = int(fields[base_idx + 3])

                if ask_px > 0 and ask_qty > 0:
                    asks.append((ask_px, ask_qty))
                if bid_px > 0 and bid_qty > 0:
                    bids.append((bid_px, bid_qty))

            yield LobsterBookSnapshot(bids=tuple(bids), asks=tuple(asks))


def generate_lobster_sample(
    message_path: Path | str,
    book_path: Path | str,
    num_events: int = 50,
    start_time_s: float = 34200.0,
    seed_price: int = 10_000,
) -> None:
    """Generate paired valid synthetic LOBSTER message and orderbook CSV files for testing."""
    msg_p = Path(message_path)
    book_p = Path(book_path)
    msg_p.parent.mkdir(parents=True, exist_ok=True)
    book_p.parent.mkdir(parents=True, exist_ok=True)

    curr_time = start_time_s
    best_bid = seed_price - 10
    best_ask = seed_price + 10
    order_id_counter = 1000

    msg_lines = []
    book_lines = []

    for i in range(num_events):
        curr_time += 0.05
        order_id_counter += 1
        side_dir = 1 if i % 2 == 0 else -1

        if i % 5 == 0:
            # Cancellation
            ev_type = LobsterType.DELETION
            qty = 100
            price = best_bid if side_dir == 1 else best_ask
            target_id = order_id_counter - 2
        elif i % 7 == 0:
            # Execution
            ev_type = LobsterType.EXECUTION_VISIBLE
            qty = 50
            price = best_ask if side_dir == 1 else best_bid
            target_id = order_id_counter - 3
        else:
            # Submission
            ev_type = LobsterType.SUBMISSION
            qty = 100
            price = best_bid if side_dir == 1 else best_ask
            target_id = order_id_counter

        # Format: Time, Type, OrderID, Size, Price, Direction
        msg_line = f"{curr_time:.9f},{int(ev_type)},{target_id},{qty},{price},{side_dir}\n"
        msg_lines.append(msg_line)

        # 5-level Orderbook line: AskP1, AskQ1, BidP1, BidQ1, ...
        levels = []
        for d in range(5):
            a_p = best_ask + d * 10
            a_q = 100 * (d + 1)
            b_p = best_bid - d * 10
            b_q = 100 * (d + 1)
            levels.extend([str(a_p), str(a_q), str(b_p), str(b_q)])
        book_lines.append(",".join(levels) + "\n")

    msg_p.write_text("".join(msg_lines), encoding="utf-8")
    book_p.write_text("".join(book_lines), encoding="utf-8")
