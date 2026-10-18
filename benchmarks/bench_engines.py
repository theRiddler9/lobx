"""Benchmark the Python and C++ engines on the same random stream.

Usage: python benchmarks/bench_engines.py [--ops 200000] [--seed 1]

Three variants are timed:
  python          PythonOrderBook
  native-wrapped  NativeOrderBook (builds Order/Fill dataclasses, as the simulator would use it)
  native-raw      the extension called with plain ints (isolates the C++ engine plus pybind11)
Throughput is operations per second over the whole stream. Latency is measured per operation with
perf_counter_ns, which itself costs roughly 50-100 ns per call, so treat the p99 as an upper bound.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.native import NativeOrderBook, native_available
from lobx.engine.streams import CancelOp, StreamOp, random_stream
from lobx.types import Order, OrderType, Side


def _time_ops(
    apply: Callable[[StreamOp], object], ops: list[StreamOp]
) -> tuple[float, float, float]:
    """Return (ops/s, median latency ns, p99 latency ns) for applying every op in order."""
    latencies: list[int] = []
    clock = time.perf_counter_ns
    start = clock()
    for op in ops:
        t0 = clock()
        apply(op)
        latencies.append(clock() - t0)
    elapsed_s = (clock() - start) / 1e9
    latencies.sort()
    return len(ops) / elapsed_s, statistics.median(latencies), latencies[int(len(latencies) * 0.99)]


def _python_apply() -> Callable[[StreamOp], object]:
    book = PythonOrderBook()
    return lambda op: (
        book.cancel(op.order_id, op.ts) if isinstance(op, CancelOp) else book.submit(op)
    )


def _wrapped_apply() -> Callable[[StreamOp], object]:
    book = NativeOrderBook()
    return lambda op: (
        book.cancel(op.order_id, op.ts) if isinstance(op, CancelOp) else book.submit(op)
    )


def _raw_apply() -> Callable[[StreamOp], object]:
    from lobx import _native

    book = _native.NativeBook()

    def apply(op: StreamOp) -> object:
        if isinstance(op, CancelOp):
            return book.cancel(op.order_id, op.ts)
        assert isinstance(op, Order)
        return book.submit(
            op.order_id,
            op.agent_id,
            0 if op.side is Side.BUY else 1,
            0 if op.type is OrderType.LIMIT else 1,
            op.price,
            op.qty,
            op.ts,
        )

    return apply


def main() -> None:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--ops", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not native_available():
        raise SystemExit("error: C++ engine not built; run `pip install -e .`")

    ops = list(random_stream(args.seed, args.ops))
    print(f"machine: {platform.platform()}, {platform.processor() or platform.machine()}")
    print(f"python: {platform.python_version()}, ops: {args.ops}, seed: {args.seed}\n")
    print("| engine | ops/s | median ns | p99 ns | speedup vs python |")
    print("|---|---:|---:|---:|---:|")
    baseline = 0.0
    for name, make in (
        ("python", _python_apply),
        ("native-wrapped", _wrapped_apply),
        ("native-raw", _raw_apply),
    ):
        rate, median, p99 = _time_ops(make(), ops)
        baseline = baseline or rate
        print(f"| {name} | {rate:,.0f} | {median:,.0f} | {p99:,.0f} | {rate / baseline:.1f}x |")


if __name__ == "__main__":
    main()
