"""Command line entry point: ``python -m lobx.cli <command>``.

Commands:

* ``run``: simulate a market and write its event log to a file.
* ``replay``: read an event log back, replay it into a fresh engine, and verify it matches.

The CLI is deliberately thin. It parses arguments, calls library code, and prints a short summary;
nothing about how a market behaves lives here.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.event_log import EventLog
from lobx.engine.native import NativeOrderBook
from lobx.engine.protocol import OrderBook
from lobx.engine.replay import replay
from lobx.errors import LobxError
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import MarketConfig
from lobx.types import FillEvent


def _file_digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes: equal runs give equal digests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _book_line(best_bid: int | None, best_ask: int | None) -> str:
    bid = "none" if best_bid is None else str(best_bid)
    ask = "none" if best_ask is None else str(best_ask)
    return f"best bid {bid}, best ask {ask}"


def _cmd_run(args: argparse.Namespace) -> int:
    config = MarketConfig(seed=args.seed, duration_ns=round(args.duration * NS_PER_S))
    out = args.out if args.out is not None else Path(f"run-seed-{args.seed}.eventlog")
    # Both engines give byte-identical logs (tests/test_cli_engines.py), so this is a speed choice.
    book: OrderBook = NativeOrderBook() if args.engine == "native" else PythonOrderBook()
    result = Market(config, book=book).run()
    result.log.write(out)
    snapshot = result.final_snapshot
    best_bid = snapshot.bids[0][0] if snapshot.bids else None
    best_ask = snapshot.asks[0][0] if snapshot.asks else None
    print(f"seed {config.seed}, {args.duration:g} simulated seconds")
    print(f"events: {len(result.log)} (trades: {result.trades}, self-trades: {result.self_trades})")
    print(f"log: {out}")
    print(f"digest: {_file_digest(out)}")
    print(f"final book: {_book_line(best_bid, best_ask)}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    log = EventLog.read(args.log)
    # replay() raises ReplayMismatchError, caught in main(), if the engine disagrees with the log.
    book = replay(log)
    trades = sum(1 for event in log if isinstance(event, FillEvent))
    print(f"replayed {len(log)} events ({trades} trades) from {args.log}")
    print("every fill, self-trade flag and cancel result matched the log")
    print(f"digest: {_file_digest(args.log)}")
    print(f"final book: {_book_line(book.best_bid(), book.best_ask())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``lobx`` command."""
    parser = argparse.ArgumentParser(prog="lobx", description="LOBX market simulator tools.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="simulate a market and write its event log")
    run.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    run.add_argument(
        "--engine",
        choices=("python", "native"),
        default="python",
        help="matching engine: the Python reference or the C++ one (default: python)",
    )
    run.add_argument(
        "--duration", type=float, default=600.0, help="simulated seconds (default: 600)"
    )
    run.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the event log (default: run-seed-<seed>.eventlog)",
    )
    run.set_defaults(handler=_cmd_run)

    replay_parser = commands.add_parser(
        "replay", help="replay an event log into a fresh engine and verify it"
    )
    replay_parser.add_argument("log", type=Path, help="an .eventlog file written by 'run'")
    replay_parser.set_defaults(handler=_cmd_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line tool and return the process exit code (0 on success)."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (LobxError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
