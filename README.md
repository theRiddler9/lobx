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
- **No-Lookahead Guarantee**: Strategies and feature pipelines are tested against lookahead (data leakage) via truncation and poison injection tests that verify decisions and features remain strictly causal.
- **Strict Execution Invariants**: Execution strategies guarantee 100% parent order completion by the deadline with zero overfill risk under any adverse market condition.

---

## Project Status

| Phase | Description | Status | Deliverables |
|---|---|---|---|
| **Phase 0** | Scaffold, docs, contracts | **Complete** (`phase-0-scaffold`) | Types, errors, protocols, CI, architecture docs |
| **Phase 1** | Python reference engine | **Complete** (`phase-1-engine`) | FIFO price levels, book sides, crossing matching, cancels, snapshots, property tests |
| **Phase 2** | Event log, replay & simulator | **Complete** (`phase-2-sim`) | Nanosecond clock, priority wake queue, agent streams, noise/MM/momentum agents, CLI, determinism & realism tests |
| **Phase 3** | C++ engine & differential testing | **Complete** (`phase-3-cpp`) | High-performance C++17 engine, pybind11 bindings, O(1) cancels, 1M-op differential tests, benchmarks |
| **Phase 4** | Baselines & evaluation harness | **Complete** (`phase-4-eval`) | TWAP, VWAP, implementation shortfall metric, 200+ test seed evaluation, confidence intervals, lookahead poison test, results plot |
| **Phase 5** | ML Level 1: Order-flow predictor | **Complete** (`phase-5-ml`) | Feature pipeline (spread, depth 1/5 imbalance, returns, trade-flow), labels, Naive/Logistic/LightGBM, 134k test rows evaluation, calibration |
| **Phase 6** | ML Level 2: Adaptive execution | **Complete** (`phase-6-adaptive`) | Model-driven child order placement, passive limit order placement at touch, aggressive fallback, deadline completion & non-overfill guarantees, paired comparison vs TWAP/VWAP, ablation study |
| **Phase 7** | Service layer & dashboard | **Complete** (`phase-7-service`) | FastAPI REST API, WebSocket live order book stream, Streamlit dashboard, Docker containerization & compose orchestration |
| **Phase 8** | Hardening & release | *Next* | End-to-end integration, performance profiling, documentation |

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

## Order-Flow Prediction (ML Level 1)

Phase 5 extracts causal microstructural features from book snapshots and trade flow (spread, depth 1 and 5 imbalance, windowed mid-returns, and trade-flow imbalance) to predict short-horizon mid-price movement (down, flat, up) over the next 1 second (5 snapshots of 200 ms).

Models are trained on seeds `0..99`, early-stopped on validation seeds `100..199`, and scored on **133,938 held-out rows across test seeds `1000..1099`**:

```bash
uv run lobx train --train-seeds 100 --validation-seeds 100 --test-seeds 100 --out models
```

| Model | Rows | Accuracy | Log Loss | Directional Acc. | ECE Up | ECE Down |
|---|---:|---:|---:|---:|---:|---:|
| **Naive (majority class)** | 133,938 | 0.4510 | 1.0660 | 0.5558 | 0.0064 | 0.0101 |
| **Multinomial Logistic** | 133,938 | 0.5345 | 0.9589 | 0.7000 | 0.0088 | 0.0559 |
| **LightGBM (GBDT)** | 133,938 | 0.5992 | 0.8645 | 0.7410 | 0.0047 | 0.0049 |

- **Beating the Baseline**: LightGBM achieves 59.9% multiclass accuracy and 74.1% directional accuracy on non-flat moves, beating the 45.1% naive baseline by 14.8 percentage points.
- **Calibration**: Expected Calibration Error (ECE) under 0.005 shows predicted probabilities match real observed frequencies.
- **Strict Causality**: Verified by truncation and poison tests proving features in row $t$ only read data at or before $t$.

*(See [`docs/results.md`](docs/results.md) and [`docs/learn/phase-5.md`](docs/learn/phase-5.md) for detailed analysis).*

---

## Adaptive Execution & Strategy Comparison (ML Level 2)

Phase 6 implements `AdaptiveStrategy`, an execution algorithm that combines:
1. **Passive Limit Orders**: Resting at the inside touch to earn the half-spread rather than paying it.
2. **Order-Flow Model Timing**: Proactively crossing the spread with aggressive market orders when the LightGBM model forecasts an imminent adverse price move ($P(\text{adverse}) - P(\text{favourable}) > \theta$).
3. **Execution Invariants**: Overdue catch-up and deadline margins guaranteeing 100% completion of the parent order by the deadline with zero overfill.

### Results on 200 Held-Out Test Seeds (`1000..1199`)

Selling 3,000 shares across 30 slices (5 minutes). Shortfall in basis points relative to the arrival mid (lower is better):

```bash
uv run lobx eval --strategies twap,vwap,passive,adaptive --compare --seeds 200 --plot docs/figures/comparison.png
```

#### Strategy Performance Table

| Strategy | Completed Seeds | Incomplete | Mean Shortfall (bps) | Std Dev (bps) | 95% CI for the Mean (bps) |
|---|---:|---:|---:|---:|---|
| **TWAP** | 198 | 2 | 56.54 | 41.14 | [50.77, 62.31] |
| **VWAP** | 197 | 3 | 57.81 | 42.22 | [51.88, 63.74] |
| **Passive (ablation, no model)** | 200 | 0 | 50.14 | 41.05 | [44.41, 55.87] |
| **Adaptive (LightGBM)** | 200 | 0 | **45.69** | **40.50** | **[40.04, 51.34]** |

#### Paired Head-to-Head Comparison

Pairing removes market drift noise across random seeds to reveal statistically significant differences:

| Pair | Valid Seeds | Mean Difference (bps) | Std Dev (bps) | 95% CI for Difference | Win Rate | Verdict |
|---|---:|---:|---:|---|---:|---|
| **adaptive vs twap** | 198 | **-10.97 bps** | 9.07 | **[-12.24, -9.70]** | **91.4%** (181/198) | adaptive cheaper |
| **adaptive vs vwap** | 197 | **-12.27 bps** | 10.63 | **[-13.76, -10.77]** | **90.9%** (179/197) | adaptive cheaper |
| **adaptive vs passive** | 200 | **-4.45 bps** | 3.51 | **[-4.94, -3.96]** | **96.0%** (192/200) | adaptive cheaper |

- **Ablation Insight**: Passive placement saves **6.4 bps** over TWAP by earning the spread. The LightGBM model contributes an additional **4.5 bps** of savings by anticipating adverse moves and taking liquidity before the price moves away.
- **Reliability**: Adaptive execution finished on 200/200 seeds (TWAP and VWAP left unfilled orders on thin books).

*(See [`docs/results.md`](docs/results.md), [`docs/learn/phase-6.md`](docs/learn/phase-6.md), and [`docs/figures/comparison.png`](docs/figures/comparison.png)).*

---

## Service Layer & Interactive Dashboard (Phase 7)

Phase 7 wraps the exchange simulator, execution strategies, and ML models into an asynchronous web service and an interactive web dashboard:

1. **FastAPI Service (`src/lobx/service/`)**:
   - Asynchronous job execution model (`HTTP 202 Accepted`) for simulation runs and multi-strategy comparisons.
   - Pydantic v2 request/response schemas with boundary validation.
   - WebSocket streaming endpoint (`WS /runs/{id}/stream`) providing paced real-time depth book replay.
   - Interactive OpenAPI documentation available at `http://localhost:8000/docs`.

2. **Streamlit Dashboard (`dashboard/app.py`)**:
   - **Live Order Book Tab**: Submit parent order execution runs and watch the resting order book depth chart update live via WebSocket.
   - **Strategy Comparison Tab**: Run head-to-head comparisons across random seeds, displaying shortfall distributions, 95% confidence intervals, and paired difference verdicts.

3. **Docker Orchestration (`compose.yaml`)**:
   - Single multi-stage container image with C++ engine compilation, OpenMP runtime, and pre-trained LightGBM models.
   - `docker compose up` spins up the REST API on port `8000` and Streamlit UI on port `8501`.

### Running the Service & Dashboard

```bash
# Start the FastAPI service (Terminal 1)
uv run lobx serve --host 127.0.0.1 --port 8000

# Start the Streamlit dashboard (Terminal 2)
uv run streamlit run dashboard/app.py

# Or launch both via Docker Compose
docker compose up
```

---

## Running Tests & Quality Gates

The test suite contains 591 unit, property, differential, causal, and service integration tests:

```bash
# Run all tests
uv run pytest

# Run service API, client, WebSocket, and dashboard tests
uv run pytest tests/test_service_app.py tests/test_service_client.py tests/test_service_execution.py tests/test_dashboard.py -v

# Run adaptive execution guarantees (completion by deadline, zero overfill)
uv run pytest tests/test_adaptive_guarantees.py -v

# Run feature causality (non-lookahead) tests
uv run pytest tests/test_features_causal.py -v

# Run strategy lookahead poison tests
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

LOBX provides a unified command-line interface for running simulations, replaying event logs, evaluating execution algorithms, and training order-flow prediction models.

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

Replay recorded submits and cancels into a fresh engine and verify exact matching:

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

### 3. Evaluate Execution Strategies & Paired Comparison

Compare execution strategies across test seeds and perform a paired statistical comparison:

```bash
uv run lobx eval --strategies twap,vwap,passive,adaptive --compare --seeds 200 --plot docs/figures/comparison.png
```

### 4. Train Order-Flow Predictors

Train models to forecast short-horizon price moves and evaluate on test seeds:

```bash
uv run lobx train --train-seeds 100 --validation-seeds 100 --test-seeds 100 --out models
```

### 5. Launch the Service API

Start the FastAPI application and WebSocket server:

```bash
uv run lobx serve --host 127.0.0.1 --port 8000
```

---

## Documentation

- [`AGENTS.md`](AGENTS.md): Full development guide, rules, and commit history for all phases.
- [`docs/plan.md`](docs/plan.md): High-level checklist and integration status.
- [`docs/results.md`](docs/results.md): Benchmark results, latency profiling, baseline shortfall, ML prediction scores, and paired comparisons.
- [`docs/contracts/shared-contracts.md`](docs/contracts/shared-contracts.md): Core protocols (`OrderBook`, `Agent`, `ExecutionStrategy`, `MarketView`, `Action`) and event log format.
- [`docs/integration.md`](docs/integration.md): Cross-component decisions and architectural log.
- [`docs/learn/phase-1.md`](docs/learn/phase-1.md): Phase 1: order book mechanics, FIFO queues, matching rules, and property tests.
- [`docs/learn/phase-2.md`](docs/learn/phase-2.md): Phase 2: discrete-event simulation, Poisson processes, inventory skew, and determinism.
- [`docs/learn/phase-3.md`](docs/learn/phase-3.md): Phase 3: C++ order book architecture, O(1) cancellations, differential testing against 1M orders, pybind11 bindings.
- [`docs/learn/phase-4.md`](docs/learn/phase-4.md): Phase 4: execution algorithms, implementation shortfall metric, multi-seed evaluation, lookahead poison test.
- [`docs/learn/phase-5.md`](docs/learn/phase-5.md): Phase 5: microstructural features, labels, causality verification, LightGBM, calibration.
- [`docs/learn/phase-6.md`](docs/learn/phase-6.md): Phase 6: adaptive order placement, deadline guarantees, paired comparisons, and ablation analysis.
- [`docs/learn/phase-7.md`](docs/learn/phase-7.md): Phase 7: asynchronous job store, FastAPI HTTP 202 patterns, paced WebSocket order book streaming, Streamlit dashboard, and Docker setup.
