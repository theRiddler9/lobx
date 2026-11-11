"""Verification that multi-threaded parallel execution across seeds is deterministic.

Simulations are pure functions: (config, seed) -> event_log.
Running independent seeds concurrently via a thread pool must yield byte-identical
event logs to running them sequentially.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import MarketConfig


def _run_seed_log(seed: int, seconds: int = 10) -> list[str]:
    """Execute a single simulation run and return its event log lines."""
    config = MarketConfig(seed=seed, duration_ns=seconds * NS_PER_S)
    result = Market(config).run()
    return list(result.log.lines())


def test_parallel_seeds_produce_identical_event_logs_to_serial() -> None:
    seeds = [10, 11, 12, 13, 14, 15]

    # 1. Sequential ground truth
    serial_logs = [_run_seed_log(s) for s in seeds]

    # 2. Concurrent execution with 4 worker threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel_logs = list(executor.map(_run_seed_log, seeds))

    # 3. Assert exact byte parity for each seed
    for i, s in enumerate(seeds):
        assert parallel_logs[i] == serial_logs[i], (
            f"Seed {s} event log differed between serial and parallel runs!"
        )
        assert len(parallel_logs[i]) > 100, f"Seed {s} produced trivial log"
