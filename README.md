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

---

## Project Status

| Phase | Description | Status | Deliverables |
|---|---|---|---|
| **Phase 0** | Scaffold, docs, contracts | **Complete** (`phase-0-scaffold`) | Types, errors, protocols, CI, architecture docs |
| **Phase 1** | Python reference engine | **Complete** (`phase-1-engine`) | FIFO price levels, book sides, crossing matching, cancels, snapshots, property tests |
| **Phase 2** | Event log, replay & simulator | **Complete** (`phase-2-sim`) | Nanosecond clock, priority wake queue, agent streams, noise/MM/momentum agents, CLI, determinism & realism tests |
| **Phase 3** | C++ engine & differential testing | *Next* | High-performance C++17 engine, pybind11 bindings, benchmarks |
| **Phase 4** | Baselines & evaluation harness | *Planned* | TWAP, VWAP, implementation shortfall metric, seed splits, leakage tests |
| **Phase 5** | ML Level 1: Order-flow predictor | *Planned* | Feature pipeline, short-horizon mid-price model, LightGBM / logreg |
| **Phase 6** | ML Level 2: Adaptive execution | *Planned* | Model-driven child order placement, deadline enforcement |
| **Phase 7** | Service layer & dashboard | *Planned* | FastAPI service, WebSocket order book stream, Streamlit dashboard |
| **Phase 8** | Hardening & release | *Planned* | End-to-end integration, performance profiling, documentation |

---

## Installation & Setup

Prerequisites: Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone https://github.com/theRiddler9/lobx.git
cd lobx

# Install dependencies and sync environment
uv sync
```

---

## Running Tests & Quality Gates

The test suite contains 340 unit, property, and integration tests:

```bash
# Run all tests
uv run pytest

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

LOBX provides a command-line interface for running simulations and verifying replay integrity.

### 1. Run a Simulation

Simulate a market and write an append-only `.eventlog`:

```bash
uv run lobx run --seed 42 --duration 60 --out run-seed-42.eventlog
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

---

## Documentation

- [`AGENTS.md`](AGENTS.md): Full development guide, rules, and commit history for all phases.
- [`docs/plan.md`](docs/plan.md): High-level checklist and integration status.
- [`docs/contracts/shared-contracts.md`](docs/contracts/shared-contracts.md): Core protocols (`OrderBook`, `Agent`, `MarketView`, `Action`) and event log format.
- [`docs/integration.md`](docs/integration.md): Cross-component decisions and architectural log.
- [`docs/learn/phase-1.md`](docs/learn/phase-1.md): Phase 1 educational guide: order book mechanics, FIFO queues, matching rules, and property tests.
- [`docs/learn/phase-2.md`](docs/learn/phase-2.md): Phase 2 educational guide: discrete-event simulation, Poisson processes, inventory skew, and determinism.
