# Historical L2/L3 Market Data Replay & Queue-Position Execution Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest real institutional LOBSTER/NASDAQ order book data and replay historical limit order books tick-by-tick with FIFO queue-position modeling and execution strategy evaluation.

**Architecture:** A streaming data ingestion pipeline parses LOBSTER message and orderbook CSVs into integer ticks and nanoseconds. A queue-position tracker maintains realistic FIFO fill priority for strategy child orders resting inside the historical liquidity queue. An event-driven replay market coordinates historical order playback with execution strategy waking and shortfall measurement.

**Tech Stack:** Python 3.11+, LOBX native/Python matching engines, Pytest, Hypothesis.

**Spec:** `docs/plans/historical_replay.md`

---

## Global Constraints

- **Prices in integer ticks**: LOBSTER prices (cents * 100) are converted to integer ticks without floating point drift.
- **Time in nanoseconds**: Historical timestamps are converted to integer nanoseconds from session start.
- **Queue Priority**: Limit order fills must rigorously respect queue position (`qty_ahead`).
- **No external heavy dependencies**: Pure Python standard library parsing + existing LOBX engine.
- Every commit must pass `ruff check`, `ruff format --check`, `mypy`, and `pytest`.

---

## Tasks

- [x] **Task 1: LOBSTER L2/L3 Message and Orderbook Parser**
  - Create `src/lobx/data/lobster.py` with `LobsterMessage`, `LobsterType`, `LobsterBookSnapshot`, and `iter_lobster_messages`.
  - Implement synthetic fixture generator `create_sample_lobster_files` in `tests/fixtures/lobster.py`.
  - Write unit tests in `tests/test_lobster_parser.py`.
  - Commit: `feat(data): add LOBSTER L2/L3 message and orderbook parser with stream iteration`

- [x] **Task 2: FIFO Queue-Position Tracker**
  - Create `src/lobx/sim/queue_position.py` with `QueueTracker`, `QueueSlot`, and `QueueFill`.
  - Implement fill-through logic: trades at price level $P$ decrement `qty_ahead` before filling the agent's resting order.
  - Write unit and property tests in `tests/test_queue_position.py`.
  - Commit: `feat(sim): add FIFO queue-position tracker for realistic resting limit orders`

- [x] **Task 3: Replay Market and Strategy Execution Runner**
  - Create `src/lobx/sim/replay_market.py` with `ReplayMarket` and `HistoricalRunResult`.
  - Implement exact order book reconstruction differential test against LOBSTER orderbook files.
  - Implement strategy child order execution against historical stream with queue tracking.
  - Write tests in `tests/test_replay_market.py` and `tests/test_lobster_reconstruction.py`.
  - Commit: `feat(sim): add ReplayMarket for tick-by-tick historical book replay and strategy execution`

- [x] **Task 4: CLI Integration**
  - Add `replay-lobster` sub-command to `src/lobx/cli.py`.
  - Allow specifying `--message`, `--book`, `--strategy`, `--shares`, and `--side`.
  - Write tests in `tests/test_cli_lobster.py`.
  - Commit: `feat(cli): add replay-lobster command to CLI with shortfall reporting`

- [x] **Task 5: Documentation & Learning Notes**
  - Create `docs/learn/phase-historical-replay.md` with mathematical formulation of queue priority and 3 practice exercises.
  - Update `docs/plan.md`.
  - Commit: `docs(learn): add phase historical replay notes, queue-position math, and exercises`
