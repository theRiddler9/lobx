"""Benchmark: sequential per-order submit versus batch submit_many.

Measures three calling modes on the same stream of orders:
  python-seq     PythonOrderBook, one submit() call per order
  native-seq     NativeOrderBook.submit(), one call per order
  native-batch   NativeOrderBook.submit_many(), one call for the whole batch

For each mode:  throughput (orders/s), median latency (ns), p99 latency (ns).

Native-batch latency is not meaningful per-order (the whole batch is one
call), so its latency column shows the per-order average across the full run.

Usage:
  python benchmarks/bench_submit_many.py [--ops N] [--batch-size K] [--seed S]

Defaults: 200,000 orders, batch size 500, seed 1.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.native import NativeOrderBook, native_available
from lobx.engine.streams import random_stream
from lobx.types import Order


def _orders_only(seed: int, count: int) -> list[Order]:
    from lobx.engine.streams import CancelOp

    return [op for op in random_stream(seed, count) if not isinstance(op, CancelOp)]


def bench_python_seq(orders: list[Order]) -> tuple[float, float, float]:
    """Python reference engine, one submit() per order."""
    book = PythonOrderBook()
    latencies: list[int] = []
    t_start = time.perf_counter_ns()
    for o in orders:
        t0 = time.perf_counter_ns()
        book.submit(o)
        latencies.append(time.perf_counter_ns() - t0)
    elapsed = (time.perf_counter_ns() - t_start) / 1e9
    latencies.sort()
    return (
        len(orders) / elapsed,
        statistics.median(latencies),
        latencies[int(len(latencies) * 0.99)],
    )


def bench_native_seq(orders: list[Order]) -> tuple[float, float, float]:
    """C++ engine via NativeOrderBook, one submit() per order (per-order GIL crossing)."""
    book = NativeOrderBook()
    latencies: list[int] = []
    t_start = time.perf_counter_ns()
    for o in orders:
        t0 = time.perf_counter_ns()
        book.submit(o)
        latencies.append(time.perf_counter_ns() - t0)
    elapsed = (time.perf_counter_ns() - t_start) / 1e9
    latencies.sort()
    return (
        len(orders) / elapsed,
        statistics.median(latencies),
        latencies[int(len(latencies) * 0.99)],
    )


def bench_native_batch(orders: list[Order], batch_size: int) -> tuple[float, float, float]:
    """C++ engine via submit_many(), GIL released for each batch.

    The batch is split into chunks of ``batch_size`` to reflect realistic usage
    (the simulator would batch one tick's orders, not the whole run at once).
    Returns throughput and per-order average latency (not a meaningful p99).
    """
    book = NativeOrderBook()
    total_orders = len(orders)
    per_order_ns: list[float] = []  # average ns per order inside each chunk
    t_start = time.perf_counter_ns()
    for i in range(0, total_orders, batch_size):
        chunk = orders[i : i + batch_size]
        t0 = time.perf_counter_ns()
        book.submit_many(chunk)
        elapsed_chunk = time.perf_counter_ns() - t0
        # distribute latency evenly across the chunk
        per_order_ns.extend([elapsed_chunk / len(chunk)] * len(chunk))
    elapsed_total = (time.perf_counter_ns() - t_start) / 1e9
    per_order_ns.sort()
    return (
        total_orders / elapsed_total,
        statistics.median(per_order_ns),
        per_order_ns[int(len(per_order_ns) * 0.99)],
    )


def main() -> None:
    """Run benchmarks and print a results table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=int, default=200_000, help="total orders (default 200,000)")
    parser.add_argument(
        "--batch-size", type=int, default=500, help="orders per submit_many chunk (default 500)"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not native_available():
        raise SystemExit("error: C++ engine not built; run `pip install -e .`")

    orders = _orders_only(args.seed, args.ops)
    print(f"machine: {platform.platform()}, {platform.processor() or platform.machine()}")
    print(
        f"python: {platform.python_version()}, "
        f"orders: {len(orders):,}, seed: {args.seed}, batch_size: {args.batch_size}\n"
    )
    print(
        "| mode | ops/s | median ns/order | p99 ns/order | speedup vs python-seq |"
    )
    print("|---|---:|---:|---:|---:|")

    baseline = 0.0
    for name, bench in [
        ("python-seq", lambda: bench_python_seq(orders)),
        ("native-seq", lambda: bench_native_seq(orders)),
        (
            f"native-batch (size={args.batch_size})",
            lambda: bench_native_batch(orders, args.batch_size),
        ),
    ]:
        rate, median_ns, p99_ns = bench()
        if baseline == 0.0:
            baseline = rate
        print(
            f"| {name} | {rate:,.0f} | {median_ns:,.0f} | {p99_ns:,.0f} | {rate / baseline:.1f}x |"
        )

    print(
        "\nNote: batch latency is per-order average within each chunk, not per-call."
    )
    print("Re-run on a quiet machine; cache effects and GC can shift numbers by ±20%.")


if __name__ == "__main__":
    main()
