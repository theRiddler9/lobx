"""Command line entry point: ``python -m lobx.cli <command>``.

Commands:

* ``run``: simulate a market and write its event log to a file.
* ``replay``: read an event log back, replay it into a fresh engine, and verify it matches.
* ``eval``, ``train``, ``tune``: the Phase 4-6 evaluation and model tools.
* ``serve``: start the HTTP API (needs the ``service`` extra).

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
from lobx.eval.compare import comparison_table
from lobx.eval.harness import default_engine
from lobx.eval.report import plot_results, results_table
from lobx.eval.runner import (
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    STRATEGY_NAMES,
    evaluate,
    obtain_model,
)
from lobx.eval.seeds import TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS
from lobx.eval.tuning import tune_threshold, tuning_table
from lobx.ml.store import load_model, save_model
from lobx.ml.train import evaluation_table, score_models, train_models
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


def _cmd_eval(args: argparse.Namespace) -> int:
    names = [name.strip() for name in args.strategies.split(",") if name.strip()]
    model = obtain_model(args.model, default_engine()) if "adaptive" in names else None
    results = evaluate(names, args.seeds, model=model, threshold=args.threshold)
    first = TEST_SEEDS.start
    print(f"test seeds {first}..{first + args.seeds - 1} (simulated market; lower is better)")
    print(results_table(results))
    if len(results) > 1:
        print("\npaired by seed (a - b; negative means a is cheaper):")
        print(comparison_table(results))
    if args.plot is not None:
        plot_results(results, args.plot)
        print(f"plot: {args.plot}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    engine = default_engine()
    models = train_models(
        TRAIN_SEEDS[: args.train_seeds], VALIDATION_SEEDS[: args.validation_seeds], engine=engine
    )
    test_seeds = range(TEST_SEEDS.start, TEST_SEEDS.start + args.test_seeds)
    results = score_models(models, test_seeds, engine=engine)
    print(f"trained on {args.train_seeds} seeds, early-stopped on {args.validation_seeds} seeds,")
    print(f"scored on test seeds {test_seeds.start}..{test_seeds.stop - 1} (simulated market)")
    print(evaluation_table(results))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, model in models.by_name().items():
            save_model(model, args.out / f"{name}.json")
        print(f"models saved in {args.out}/")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    seeds = VALIDATION_SEEDS[: args.seeds]
    rows = tune_threshold(model, thresholds, seeds, engine=default_engine())
    print(f"validation seeds {seeds.start}..{seeds.stop - 1}; adaptive minus TWAP, paired by seed")
    print(tuning_table(rows))
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


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # optional: only the service needs it

        from lobx.service.app import create_app
    except ImportError as error:
        raise LobxError(
            f"the service needs its extra packages ({error}); install them with: "
            "pip install -e '.[service]'"
        ) from error
    print(f"serving on http://{args.host}:{args.port} (docs at /docs); Ctrl+C to stop")
    uvicorn.run(create_app(model_path=args.model), host=args.host, port=args.port)
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

    train_parser = commands.add_parser(
        "train", help="train the order-flow models and score them on held-out seeds"
    )
    train_parser.add_argument("--train-seeds", type=int, default=100, help="default: 100")
    train_parser.add_argument("--validation-seeds", type=int, default=100, help="default: 100")
    train_parser.add_argument("--test-seeds", type=int, default=100, help="default: 100")
    train_parser.add_argument(
        "--out", type=Path, default=None, help="directory to save the fitted models in (JSON)"
    )
    train_parser.set_defaults(handler=_cmd_train)

    tune_parser = commands.add_parser(
        "tune", help="pick the adaptive threshold on validation seeds (never test seeds)"
    )
    tune_parser.add_argument("--model", type=Path, default=Path("models/lightgbm.json"))
    tune_parser.add_argument("--thresholds", default="0.3,0.5,0.7,0.9,1.0")
    tune_parser.add_argument("--seeds", type=int, default=40, help="validation seeds (default: 40)")
    tune_parser.set_defaults(handler=_cmd_tune)

    evaluate_parser = commands.add_parser(
        "eval", help="compare execution strategies on held-out seeds"
    )
    evaluate_parser.add_argument(
        "--strategies",
        default=",".join(STRATEGY_NAMES),
        help=f"comma-separated names from: {', '.join(STRATEGY_NAMES)} (default: all)",
    )
    evaluate_parser.add_argument(
        "--seeds", type=int, default=200, help="number of test seeds (default: 200)"
    )
    evaluate_parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="LightGBM model (trained if absent)"
    )
    evaluate_parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="adaptive threshold"
    )
    evaluate_parser.add_argument(
        "--plot", type=Path, default=None, help="also write a PNG of the results here"
    )
    evaluate_parser.set_defaults(handler=_cmd_eval)

    serve_parser = commands.add_parser("serve", help="start the HTTP API and WebSocket stream")
    serve_parser.add_argument("--host", default="127.0.0.1", help="interface (default: localhost)")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    serve_parser.set_defaults(handler=_cmd_serve)
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
