# LOBX

Limit-order-book exchange simulator with an ML-driven trade execution agent, written in Python and C++17.

LOBX simulates an equity exchange with a price-time priority matching engine, populates it with simulated trading agents (market makers, noise traders, momentum traders), and tests whether an ML-adaptive trade execution agent can execute large parent orders (e.g. 50,000 shares over 30 simulated minutes) at lower implementation shortfall (in basis points) than classical TWAP and VWAP benchmarks.

---

## Architecture & Pipeline

```text
Agents (noise, market maker, momentum, execution agent)
  -> Orders (integer ticks, integer ns)
  -> Matching Engine (Python reference OR C++ via pybind11)
  -> Event Log (append-only, deterministic replay)
  -> Book Snapshots
  -> Feature Pipeline
  -> ML Model (order-flow predictor)
  -> Execution Strategy (TWAP / VWAP / ML-adaptive)
  -> Evaluation Harness (multi-seed, bps shortfall, confidence intervals)
```

### Core Design Principles

- **Integer Arithmetic**: Prices are integers in ticks; time is an integer in nanoseconds. Quantities are integer share counts. No floating-point rounding issues in book state or matching.
- **Bit-Level Determinism**: A simulation run is a pure function of `(config, seed)`. Two runs with the same parameters produce byte-identical event logs and exact SHA-256 digests across different runs and environments.
- **Dual Engine Architecture**: A clean, readable Python reference engine and a high-performance C++17 engine (via `pybind11`), verified by differential property tests against identical order streams.
- **Append-Only Event Log & Exact Replay**: Every accepted order, cancel result, and fill is recorded. A fresh engine replaying the inputs must reproduce every fill, trade price, and self-trade flag identically.
- **No-Lookahead Guarantee**: Strategies and feature pipelines are tested against lookahead (data leakage) via poison injection tests that modify future frames and verify decisions remain invariant.

---

## Project Status

| Phase | Description | Status | Deliverables |
|---|---|---|---|
| **Phase 0** | Scaffold, docs, contracts | **Complete** (`phase-0-scaffold`) | Types, errors, protocols, CI, architecture docs |
| **Phase 1** | Python reference engine | **Complete** (`phase-1-engine`) | FIFO price levels, book sides, crossing matching, cancels, snapshots, property tests |
| **Phase 2** | Event log, replay & simulator | **Complete** (`phase-2-sim`) | Nanosecond clock, priority wake queue, agent streams, noise/MM/momentum agents, CLI, determinism & realism tests |
| **Phase 3** | C++ engine & differential testing | **Complete** (`phase-3-cpp`) | High-performance C++17 engine, pybind11 bindings, O(1) cancels, 1M-op differential tests, benchmarks |
| **Phase 4** | Baselines & evaluation harness | **Complete** (`phase-4-eval`) | TWAP, VWAP, implementation shortfall metric, 200+ test seed evaluation, confidence intervals, lookahead poison test, results plot |
| **Phase 5** | ML Level 1: Order-flow predictor | *Next* | Feature pipeline, short-horizon mid-price model, LightGBM / logreg |
| **Phase 6** | ML Level 2: Adaptive execution | *Planned* | Model-driven child order placement, deadline enforcement |
| **Phase 7** | Service layer & dashboard | *Planned* | FastAPI service, WebSocket order book stream, Streamlit dashboard |
| **Phase 8** | Hardening & release | *Planned* | End-to-end integration, performance profiling, documentation |

---

## Installation & Setup

Prerequisites: Python 3.11+, a C++17 compiler (GCC, Clang, or MSVC), CMake 3.20+, and [`uv`](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone https://github.com/theRiddler9/lobx.git
cd lobx

# Install dependencies and build C++ native extension
uv sync
```

---

## Benchmarks & Engine Performance

LOBX implements the matching engine twice:
1. **Python reference engine** (`lobx.engine.book_py.OrderBookPy`): Clean reference implementation prioritizing readability and explicit invariants.
2. **C++17 engine** (`cpp/include/lobx/order_book.hpp`, exposed via pybind11 as `NativeOrderBook`): High-performance matching engine utilizing `std::map` for sorted price levels and `std::unordered_map` for $O(1)$ order cancellation without scanning FIFO queues.

### Standalone C++ Engine Throughput

Tested standalone without Python interpreter overhead (`cmake --build build && ./lobx_bench 1000000`):
- **Throughput**: **5,220,442 orders/s**
- **Median Latency**: **124 ns**
- **p99 Latency**: **463 ns**

### Python vs Native Comparison (Called from Python)

Run the benchmark suite:
```bash
uv run python benchmarks/bench_engines.py --ops 1000000
```

| Engine | Throughput (ops/s) | Median Latency | p99 Latency | Relative Speedup |
|---|---:|---:|---:|---:|
| **Python Reference** | 335,980 ops/s | 2,153 ns | 11,321 ns | 1.0x |
| **Native (Wrapped Dataclasses)** | 724,793 ops/s | 841 ns | 5,101 ns | 2.2x |
| **Native (Raw pybind11 ints)** | 1,072,217 ops/s | 710 ns | 2,414 ns | 3.2x |

*(Measured on Linux x86_64 2-vCPU Xeon @ 2.10GHz. See [`docs/results.md`](docs/results.md) for full benchmark configuration).*

---

## Execution Baselines (TWAP & VWAP)

Phase 4 evaluates classical benchmark execution algorithms inside the simulated market across 200 held-out test seeds (`1000..1199`). The task sells 3,000 shares (approx. 20% of market volume) across 30 time slices over 5 simulated minutes.

Cost is measured in **implementation shortfall (bps)** relative to the arrival mid-price (lower is better):

```bash
uv run lobx eval --strategies twap,vwap --seeds 200 --plot docs/figures/baselines.png
```

| Strategy | Completed Seeds | Incomplete | Mean Shortfall (bps) | Std Dev (bps) | 95% CI for the Mean (bps) |
|---|---:|---:|---:|---:|---|
| **TWAP** | 198 | 2 | 56.54 | 41.14 | [50.77, 62.31] |
| **VWAP** | 197 | 3 | 57.81 | 42.22 | [51.88, 63.74] |

*(See [`docs/results.md`](docs/results.md) for analysis and [`docs/figures/baselines.png`](docs/figures/baselines.png) for the error-bar plot).*

---

## Running Tests & Quality Gates

The test suite contains 436 unit, property, differential, and integration tests:

```bash
# Run all tests
uv run pytest

# Run lookahead (data leakage) poison test
uv run pytest tests/test_leakage.py -v

# Run C++ vs Python differential tests (1,000,000 random operations)
uv run pytest tests/test_native_differential.py -v

# Run market realism tests (spread positivity, fat tails / kurtosis, price drift)
uv run pytest tests/test_sim_realism.py -v

# Run determinism tests (identical event logs across processes and seeds)
uv run pytest tests/test_determinism.py -v

# Run replay verification tests
uv run pytest tests/test_replay_of_runs.py -v

# Run code formatting, linter, and type checks
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

---

## Command Line Interface (CLI)

LOBX provides a command-line interface for running simulations, replaying event logs, and evaluating execution algorithms.

### 1. Run a Simulation

Simulate a market and write an append-only `.eventlog`:

```bash
# Run with C++ engine (default)
uv run lobx run --engine native --seed 42 --duration 60 --out run-seed-42.eventlog

# Or run with Python reference engine
uv run lobx run --engine python --seed 42 --duration 60 --out run-seed-42.eventlog
```

**Output:**
```text
seed 42, 60 simulated seconds
events: 3348 (trades: 1124, self-trades: 4)
log: run-seed-42.eventlog
digest: 6632f059bc438e8cb28dbdd8b17b8f94d9bbfad4db0b49cb37d6eec08db82570
final book: best bid 9988, best ask 9992
```

### 2. Replay & Verify an Event Log

Replay the recorded submits and cancels into a fresh engine and check that every trade, price, and self-trade flag matches:

```bash
uv run lobx replay run-seed-42.eventlog
```

**Output:**
```text
replayed 3348 events (1124 trades) from run-seed-42.eventlog
every fill, self-trade flag and cancel result matched the log
digest: 6632f059bc438e8cb28dbdd8b17b8f94d9bbfad4db0b49cb37d6eec08db82570
final book: best bid 9988, best ask 9992
```

### 3. Evaluate Execution Strategies

Compare execution strategies across test seeds and generate a summary table and plot:

```bash
uv run lobx eval --strategies twap,vwap --seeds 200 --plot docs/figures/baselines.png
```

---

## Documentation

- [`AGENTS.md`](AGENTS.md): Full development guide, rules, and commit history for all phases.
- [`docs/plan.md`](docs/plan.md): High-level checklist and integration status.
- [`docs/results.md`](docs/results.md): Benchmark results, latency profiling, baseline shortfall tables.
- [`docs/contracts/shared-contracts.md`](docs/contracts/shared-contracts.md): Core protocols (`OrderBook`, `Agent`, `ExecutionStrategy`, `MarketView`, `Action`) and event log format.
- [`docs/integration.md`](docs/integration.md): Cross-component decisions and architectural log.
- [`docs/learn/phase-1.md`](docs/learn/phase-1.md): Phase 1: order book mechanics, FIFO queues, matching rules, and property tests.
- [`docs/learn/phase-2.md`](docs/learn/phase-2.md): Phase 2: discrete-event simulation, Poisson processes, inventory skew, and determinism.
- [`docs/learn/phase-3.md`](docs/learn/phase-3.md): Phase 3: C++ order book architecture, O(1) cancellations, differential testing against 1M orders, pybind11 bindings.
- [`docs/learn/phase-4.md`](docs/learn/phase-4.md): Phase 4: execution algorithms, implementation shortfall metric, multi-seed evaluation, lookahead poison test.
