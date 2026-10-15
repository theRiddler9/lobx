# LOBX

## Mission

Build **LOBX**, a limit-order-book exchange simulator with an ML-driven trade execution agent, written in Python and C++17.

LOBX simulates a stock exchange with a real matching engine, populates it with simulated trading agents, and tests whether an ML-driven agent can execute a large parent order (for example, sell 50,000 shares over 30 simulated minutes) at lower cost than the standard TWAP and VWAP algorithms. Cost is measured as implementation shortfall in basis points.

The matching engine is implemented twice: a readable Python reference and a fast C++ version exposed through pybind11. The two are differential-tested against each other. The ML agent and the evaluation harness run on top of either engine.

## Non-Goals

Do not reduce the project to:

- a backtest of a price-prediction model on downloaded stock data (the focus is execution against a simulated order book)
- a wrapper around `abides-jpmc-public`, `ABIDES`, `backtrader`, `zipline`, `vectorbt`, or any package that already implements the order book or the simulator for us
- a live or paper trading system; no broker, exchange, or market-data API is called
- a notebook-only project; all core logic lives in tested modules, and notebooks may only plot or explore
- a project that claims real-market profitability. All results are simulated and must be described that way.

Allowed libraries handle numerical work, ML training, serving, and plotting. The matching engine, the simulator, the agents, the execution strategies, the feature pipeline, and the evaluation harness are written in this repository.

A human will study this code later, line by line, and practice by re-implementing parts of it. Every non-trivial piece must be readable and explainable.

`abides-jpmc-public` is archived and read-only. It is a design reference only (core event simulator, market layer, gym wrapper, noise/value/momentum/market-maker agent mix). Never copy-paste from it.

## Core Pipeline

```
Agents (noise, market maker, momentum, execution agent)
  -> Orders (integer prices in ticks, integer timestamps in ns)
  -> Matching Engine (Python reference OR C++ via pybind11)
  -> Event Log (append-only, deterministic replay)
  -> Book Snapshots
  -> Feature Pipeline
  -> ML Model (order-flow predictor)
  -> Execution Strategy (TWAP / VWAP / ML-adaptive)
  -> Evaluation Harness (many seeds, bps cost, confidence intervals)

Service layer (stretch of the main path):
  Simulation run -> FastAPI + WebSocket -> Streamlit dashboard
```

The engine -> event log -> features -> strategy -> evaluation architecture is fixed. Do not merge stages or skip one.

## Shared Source of Truth

Read these before implementation. If a file does not exist yet, create it in Phase 0.

- `docs/plan.md`: overall architecture, phase checklist, and integration status
- `docs/contracts/shared-contracts.md`: cross-module types, protocols, and events
- `docs/plans/engine.md`: matching engine plan (Python and C++)
- `docs/plans/simulator.md`: agents, arrival processes, and replay plan
- `docs/plans/strategies.md`: TWAP, VWAP, and ML-adaptive execution plan
- `docs/plans/ml.md`: features, labels, models, and evaluation of the models
- `docs/plans/evaluation.md`: harness, metrics, statistics, leakage tests
- `docs/plans/service.md`: FastAPI, WebSocket, dashboard, Docker, CI
- `docs/integration.md`: merge and integration checkpoints
- `docs/runbooks/demo.md`: golden demo definition
- `docs/results.md`: every reported number, with the command that produced it
- `docs/learn/phase-N.md`: plain-language notes written after each phase, ending with three practice exercises

External references to follow when implementing behavior:

- Limit order book mechanics: price-time priority, order types, cancels
- Almgren and Chriss, "Optimal Execution of Portfolio Transactions" (why TWAP exists, impact trade-off)
- Avellaneda and Stoikov, "High-frequency trading in a limit order book" (market-maker idea)
- ABIDES papers and repository, for agent mix and design ideas, never copy-paste
- `pybind11`, `scikit-learn`, `LightGBM`, `Gymnasium`, `FastAPI` docs

## Tech Stack

| Concern | Tool |
|---|---|
| Reference engine, simulator, agents, ML, evaluation | Python 3.11+ |
| Fast engine | C++17 |
| Python/C++ bindings | `pybind11` with `scikit-build-core` and CMake |
| Numerics and data | `numpy`, `pandas` |
| ML | `scikit-learn`, `lightgbm` |
| RL environment (stretch only) | `gymnasium`, `stable-baselines3` |
| Service | `fastapi`, `uvicorn`, `pydantic` |
| Dashboard | `streamlit` |
| Plots | `matplotlib` |
| Python tests | `pytest`, `hypothesis` |
| C++ tests and benchmarks | `Catch2` or `doctest`, plus a simple timing harness (Google Benchmark optional) |
| Lint, format, types | `ruff`, `mypy`, `clang-format`, `clang-tidy` |
| Packaging and containers | `pyproject.toml`, `Dockerfile`, GitHub Actions |

Rules:

- Pin versions in `pyproject.toml` and commit a lock file.
- Ask before adding any dependency not listed here.
- No dependency that already implements an order book or matching engine.

## Package Boundaries

Any contributor may work anywhere in the repository. Keep changes coherent with these component boundaries, but do not treat them as exclusive ownership.

- **Engine** (`src/lobx/engine/`, `cpp/`): order types, the order book, matching, cancels, and the event log. `book_py.py` is the readable reference. `cpp/` is the fast implementation. Both implement the same `OrderBook` protocol. Contains no agent or ML logic.
- **Simulator** (`src/lobx/sim/`): clock, seeded random number streams, arrival processes, and background agents (noise, market maker, momentum). Drives an engine, never reaches into its internals.
- **Strategies** (`src/lobx/strategies/`): `ExecutionStrategy` implementations. TWAP, VWAP, ML-adaptive. They read book snapshots and emit child orders; they never mutate the book directly.
- **ML** (`src/lobx/ml/`): feature extraction, label creation, model training and loading, model evaluation. Reads event logs and snapshots, never touches the engine.
- **Evaluation** (`src/lobx/eval/`): runs strategies across seeds, computes implementation shortfall and statistics, writes result tables and plots. Contains the leakage tests.
- **Service** (`src/lobx/service/`, `dashboard/`): FastAPI app, WebSocket book stream, Streamlit dashboard. Wires existing components, adds no new logic.
- **CLI** (`src/lobx/cli.py`): small commands to run a simulation, an evaluation, or a replay. Stays small.

## Repository Layout

```
lobx/
├── AGENTS.md
├── pyproject.toml
├── CMakeLists.txt
├── README.md
├── Dockerfile
├── .gitignore
├── .clang-format
├── .github/workflows/ci.yml
├── docs/
│   ├── plan.md
│   ├── integration.md
│   ├── results.md
│   ├── contracts/shared-contracts.md
│   ├── plans/{engine,simulator,strategies,ml,evaluation,service}.md
│   ├── runbooks/demo.md
│   └── learn/phase-N.md
├── cpp/
│   ├── include/lobx/{types.hpp,order_book.hpp}
│   ├── src/{order_book.cpp,bindings.cpp}
│   └── tests/
├── src/lobx/
│   ├── __init__.py
│   ├── cli.py
│   ├── types.py              # Side, OrderType, Order, Fill, Event (the contract types)
│   ├── engine/
│   │   ├── protocol.py       # the OrderBook protocol
│   │   ├── book_py.py        # Python reference engine
│   │   ├── event_log.py
│   │   └── native.py         # loads the C++ engine, falls back with a clear error
│   ├── sim/
│   │   ├── clock.py
│   │   ├── rng.py
│   │   ├── market.py
│   │   └── agents/{base.py,noise.py,market_maker.py,momentum.py}
│   ├── strategies/{base.py,twap.py,vwap.py,adaptive.py}
│   ├── ml/{features.py,labels.py,train.py,model.py}
│   ├── eval/{harness.py,metrics.py,stats.py,leakage.py,report.py}
│   └── service/{app.py,stream.py}
├── dashboard/app.py
├── tests/
│   ├── fixtures/             # captured order streams and expected books
│   └── test_*.py
└── benchmarks/
```

## Shared Contracts

These are the starting contracts. Write them in `docs/contracts/shared-contracts.md` and as code stubs first. Any change to a contract goes through its own `contract/*` branch and merges before dependent work continues.

Global rules for all contracts:

- Prices are **integers in ticks**. Never use floating point for prices or quantities.
- Time is an **integer in nanoseconds** since simulation start.
- Every random draw comes from a seeded stream passed in explicitly. No global `random` or `np.random` calls.

```python
# types.py — shared types
class Side(Enum): BUY; SELL
class OrderType(Enum): LIMIT; MARKET
@dataclass(frozen=True)
class Order:
    order_id: int; agent_id: int; side: Side; type: OrderType
    price: int | None; qty: int; ts: int
@dataclass(frozen=True)
class Fill:
    maker_order_id: int; taker_order_id: int
    price: int; qty: int; ts: int
@dataclass(frozen=True)
class BookSnapshot:
    ts: int
    bids: tuple[tuple[int, int], ...]   # (price, total_qty), best first
    asks: tuple[tuple[int, int], ...]

# engine/protocol.py — the contract both engines implement
class OrderBook(Protocol):
    def submit(self, order: Order) -> list[Fill]: ...
    def cancel(self, order_id: int, ts: int) -> bool: ...
    def snapshot(self, depth: int) -> BookSnapshot: ...
    def best_bid(self) -> int | None: ...
    def best_ask(self) -> int | None: ...

# sim/agents/base.py — agent contract
class Agent(Protocol):
    agent_id: int
    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]: ...
    # Action = Submit(order) | Cancel(order_id)

# strategies/base.py — execution contract
class ExecutionStrategy(Protocol):
    def reset(self, parent: ParentOrder) -> None: ...
    def step(self, ts: int, view: MarketView) -> list[Action]: ...

# ml/features.py — feature contract
def extract_features(snapshots: Sequence[BookSnapshot], fills: Sequence[Fill]) -> np.ndarray: ...
# Rule: features at time t may only use data with timestamp <= t.

# eval/metrics.py — metric contract
def implementation_shortfall_bps(parent: ParentOrder, fills: Sequence[Fill], arrival_mid: int) -> float: ...
```

## Data Flow Boundary

```
Simulator wakes agents in timestamp order (single thread, deterministic)
  -> Agents return Actions
  -> Market applies Actions to the OrderBook
  -> OrderBook returns Fills and appends Events to the Event Log
  -> Market publishes BookSnapshots to the strategies and the feature pipeline
  -> Strategy returns child Actions
```

Rules:

- A run is a pure function of `(config, seed)`. Same inputs must give byte-identical event logs.
- The engine is the only component that mutates the book.
- The Python and C++ engines must produce identical fills and snapshots for any order stream. This is enforced by a differential test.
- Strategies and features never look at data from the future. The evaluation harness proves this with an injected-lookahead test.
- Train/test splits are by seed ranges, with the test seeds never used for tuning.

## v1 Feature Scope

Must work in v1:

- limit and market orders, cancel, price-time priority, partial fills
- self-trade is allowed but logged, no hidden orders, no iceberg orders
- an event log that can replay any run exactly from the log alone
- background agents: noise traders (Poisson arrivals), a market maker quoting both sides with inventory skew, a momentum trader
- baselines: TWAP and VWAP execution
- features from the book: spread, order-book imbalance at depth 1 and 5, mid-price returns over several windows, recent trade-flow imbalance
- a supervised model that predicts the short-horizon mid-price move
- an adaptive execution strategy that uses the model to choose between passive limit orders and aggressive market orders
- evaluation over 200 or more seeds with mean implementation shortfall, standard deviation, and 95% confidence intervals
- differential tests (Python vs C++) and a leakage test
- a FastAPI endpoint to submit a parent order, a WebSocket book stream, a Streamlit dashboard, a Dockerfile

Stretch (only after the golden demo passes): a Gymnasium environment with PPO training, an ABIDES-Gym comparison in a separate virtual environment, multiple symbols, queue-position modeling, fee and rebate models.

## Build Phases

Work through the phases in order. Each phase ends with a tag, a `docs/learn/phase-N.md`, and passing checks. Do not start a phase until the previous one meets its exit criteria. A phase with fewer than 8 commits is under-committed.

### Phase 0: Scaffold, docs, contracts

Exit: the repo installs, CI is green, all docs and contract stubs exist.

Commits:

1. `chore: initialize python project with pyproject.toml`
2. `chore: add .gitignore, ruff, mypy, and pytest settings`
3. `docs: add AGENTS.md`
4. `docs: add plan.md with phase checklist`
5. `docs(contracts): add shared-contracts.md`
6. `docs(plans): add per-component plan files`
7. `feat(types): add Side, OrderType, Order, Fill, and BookSnapshot`
8. `feat(engine): add OrderBook protocol stub`
9. `chore(ci): add GitHub Actions for ruff, mypy, and pytest`
10. `docs(runbook): add golden demo definition`

### Phase 1: Python reference engine

Exit: the Python order book passes unit tests and property tests; the book never crosses and quantities never go negative.

Commits:

1. `feat(engine): add price level with FIFO queue`
2. `feat(engine): add book sides keyed by price`
3. `feat(engine): add limit order insertion`
4. `feat(engine): add best bid, best ask, and snapshot`
5. `feat(engine): add matching for crossing limit orders`
6. `feat(engine): add partial fill handling`
7. `feat(engine): add market order matching`
8. `feat(engine): add cancel by order id`
9. `feat(engine): reject invalid orders with clear errors`
10. `test(engine): add unit tests for each order scenario`
11. `test(engine): add hypothesis property tests for book invariants`
12. `docs(learn): add phase 1 notes`

### Phase 2: Event log, replay, and simulator

Exit: a seeded run produces the same event log every time; replaying the log reproduces the final book exactly; the simulated market shows a positive spread and fat-tailed returns.

Commits:

1. `feat(engine): add Event types and append-only event log`
2. `feat(engine): log every submit, cancel, and fill`
3. `feat(engine): add replay from event log`
4. `feat(sim): add seeded random streams per agent`
5. `feat(sim): add clock and event queue`
6. `feat(sim): add Agent base protocol and MarketView`
7. `feat(sim): add noise trader with Poisson arrivals`
8. `feat(sim): add market maker with inventory skew`
9. `feat(sim): add momentum trader`
10. `feat(sim): add Market that applies agent actions to the engine`
11. `feat(cli): add run command that writes an event log`
12. `test(sim): verify same seed gives identical event logs`
13. `test(sim): verify replay reproduces the final book`
14. `test(sim): check spread and return statistics are realistic`
15. `docs(learn): add phase 2 notes`

### Phase 3: C++ engine and differential testing

Exit: the C++ engine, called from Python, matches the Python engine on one million random orders and is measurably faster.

Commits:

1. `chore(cpp): add CMakeLists and scikit-build-core setup`
2. `feat(cpp): add shared types header with integer ticks`
3. `feat(cpp): add price level and order storage`
4. `feat(cpp): add book sides with ordered map`
5. `feat(cpp): add limit order insertion and matching`
6. `feat(cpp): add market orders and partial fills`
7. `feat(cpp): add O(1) cancel with an order id index`
8. `feat(cpp): add snapshot and best bid/ask`
9. `feat(bindings): expose the C++ engine with pybind11`
10. `feat(engine): add native loader with a clear fallback error`
11. `test(cpp): add C++ unit tests`
12. `test(engine): add differential test, Python vs C++, on random streams`
13. `perf(benchmarks): add engine throughput and p99 latency benchmark`
14. `docs(results): record the benchmark table and the machine it ran on`
15. `docs(learn): add phase 3 notes`

### Phase 4: Baselines and evaluation harness

Exit: TWAP and VWAP run end to end over 200+ seeds and produce a results table with confidence intervals; the lookahead test catches an injected bug.

Commits:

1. `feat(strategies): add ParentOrder and ExecutionStrategy protocol`
2. `feat(strategies): add TWAP`
3. `feat(strategies): add volume profile estimation`
4. `feat(strategies): add VWAP`
5. `feat(eval): add implementation shortfall metric`
6. `feat(eval): add run-one-seed function`
7. `feat(eval): add multi-seed runner with a train/test seed split`
8. `feat(eval): add mean, standard deviation, and confidence intervals`
9. `feat(eval): add results table and plot output`
10. `test(eval): verify the metric on hand-computed examples`
11. `test(eval): add an injected-lookahead test that must be caught`
12. `docs(results): record baseline results with the exact command`
13. `docs(learn): add phase 4 notes`

### Phase 5: ML level 1, the order-flow predictor

Exit: the model beats a naive baseline on held-out seeds; features pass the no-future-data check.

Commits:

1. `chore(deps): add scikit-learn and lightgbm`
2. `feat(ml): add spread and mid-price features`
3. `feat(ml): add order-book imbalance features`
4. `feat(ml): add windowed return and trade-flow features`
5. `feat(ml): add label creation for short-horizon mid-price move`
6. `feat(ml): add dataset builder from event logs`
7. `feat(ml): add logistic regression baseline`
8. `feat(ml): add LightGBM model with seed-based train/validation split`
9. `feat(ml): add model save and load`
10. `test(ml): verify features never use data after time t`
11. `test(ml): add model evaluation with accuracy and calibration`
12. `docs(results): record model results and the naive baseline`
13. `docs(learn): add phase 5 notes`

### Phase 6: ML level 2, adaptive execution

Exit: the adaptive strategy is evaluated against TWAP and VWAP on held-out seeds, and the result is reported honestly whether it wins or not.

Commits:

1. `feat(strategies): add AdaptiveStrategy skeleton using the model`
2. `feat(strategies): add passive limit order placement`
3. `feat(strategies): add aggressive market order fallback`
4. `feat(strategies): add the decision rule from model output`
5. `feat(strategies): add remaining-quantity and deadline handling`
6. `feat(eval): add strategy comparison runner`
7. `feat(eval): add paired comparison against TWAP and VWAP`
8. `test(strategies): verify the parent order always completes by the deadline`
9. `test(strategies): verify no overfill beyond parent quantity`
10. `docs(results): record the comparison table and the failure analysis`
11. `docs(learn): add phase 6 notes`

### Phase 7: Service layer

Exit: `docker compose up` shows a live book and a strategy comparison in the browser.

Commits:

1. `chore(deps): add fastapi, uvicorn, pydantic, and streamlit`
2. `feat(service): add FastAPI app with a health endpoint`
3. `feat(service): add endpoint to submit a parent order and start a run`
4. `feat(service): add endpoint to fetch results`
5. `feat(service): add WebSocket book stream`
6. `feat(dashboard): add live book view`
7. `feat(dashboard): add strategy comparison view`
8. `chore(docker): add Dockerfile and compose file`
9. `test(service): add API and WebSocket tests`
10. `docs(learn): add phase 7 notes`

### Phase 8: Hardening and release

Exit: the golden demo passes from a clean clone and the project is presentable.

Commits:

1. `test: add a full end-to-end test from run to results table`
2. `test(cpp): run C++ tests under sanitizers`
3. `perf: profile the engine and record a flame graph`
4. `fix: address issues found by sanitizers and clang-tidy`
5. `chore(ci): add a Linux and macOS build matrix with C++ build`
6. `docs: write README with install steps, results table, and limitations`
7. `docs: add architecture diagram`
8. `docs: write the limitations and "what failed" sections`
9. `chore: add LICENSE`
10. `docs(learn): add final notes`

### Optional Phase 9: RL (stretch)

Only after Phase 8 is tagged. Gymnasium environment, PPO with stable-baselines3, training curves, an honest comparison against the Phase 6 strategy. Same rules: at least 8 small commits and a learn doc.

## Time Budget and Cut Line

The project targets roughly three weeks. If time runs short, cut in this order and keep the rest intact:

1. Skip Optional Phase 9.
2. Skip the Streamlit dashboard and WebSocket in Phase 7 (keep the FastAPI endpoints and Docker).
3. Skip LightGBM and keep logistic regression only in Phase 5.
4. Never cut: Phases 1, 2, 3 (engine, replay, C++ with differential test), Phase 4 (evaluation), and a simple version of Phase 6.

## Git Rules

- Work on separate feature branches. Never commit directly to `main`.
- Make small logical commits.
- Merge shared contracts before dependent work.
- Pull/rebase shared changes before integration.
- Avoid destructive Git operations.

### Branches

- Naming: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `test/<topic>`, `chore/<topic>`, `refactor/<topic>`, `perf/<topic>`, `contract/<topic>`.
- One branch per feature, for example `feat/engine-matching`, `feat/cpp-cancel`, `feat/ml-features`.
- Branches are short-lived. Merge and delete them once finished.
- Contract changes get their own `contract/<topic>` branch and merge to `main` before any dependent branch continues.

### Commits

- Use Conventional Commits: `type(scope): summary`. Scope is the module (`engine`, `cpp`, `bindings`, `sim`, `strategies`, `ml`, `eval`, `service`, `dashboard`, `cli`, `types`, `ci`, `docs`, `docker`).
- Summary is imperative, lowercase, under 72 characters.
- Add a body when the reason is not obvious, especially for market-mechanics behavior (state the rule, for example "price-time priority: earlier order at the same price fills first").
- Commit frequently. Make at least one commit for each of:
  - a new file
  - a new public type, class, or protocol
  - a new order type or matching rule
  - a new agent or strategy
  - a new feature or label in the ML pipeline
  - a new metric
  - a new test file
  - a docs update
- Keep a commit under roughly 150 changed lines and one concern.
- Do not mix refactoring with feature work. Do not mix formatting changes with logic changes.
- Every commit must pass `ruff check`, `mypy`, and `pytest`. Do not leave broken intermediate commits.
- Commit tests with their feature or immediately after, as `test(scope): ...`.
- Never commit `build/`, `__pycache__/`, `.venv/`, `.env`, trained model binaries larger than 5 MB, editor settings, or machine-specific paths.

### Integration

```
git switch main && git pull --ff-only
git switch -c feat/engine-matching

# work in small steps, commit after each logical change

git fetch origin
git rebase origin/main
ruff check . && ruff format --check . && mypy src && pytest

git push -u origin feat/engine-matching
# open a PR; merge with a merge commit (no squash, history is the progress log)
```

- Before opening a PR, rebase on the latest `main`.
- Merge with merge commits. Do not squash; the commit trail is part of the deliverable.
- Tag each finished phase: `git tag phase-1-engine`, `phase-2-sim`, `phase-3-cpp`, `phase-4-eval`, `phase-5-ml`, `phase-6-adaptive`, `phase-7-service`, `phase-8-release`.

### Destructive operations

Do not run these without explicit human approval:

- `git push --force` (on your own feature branch after a rebase, `--force-with-lease` is the only acceptable form)
- `git reset --hard`
- `git clean -fd`
- `git checkout .` or `git restore .` on uncommitted work
- `git branch -D` on an unmerged branch
- `git rebase` or any history rewrite on `main` or on a branch others use
- `git commit --amend` on a commit that is already pushed

If unsure, create a backup branch first: `git branch backup/<name>`.

## Coding Standards

- `ruff check`, `ruff format --check`, and `mypy --strict` on `src/` must pass. C++ must pass `clang-format --dry-run` and compile with `-Wall -Wextra -Werror`.
- No bare `except`. No silent failures. Invalid input raises a specific exception with a clear message.
- Prices and quantities are integers. No floats in the engine, ever.
- No global random state. Every random draw uses a seeded generator passed in explicitly.
- No hidden mutable globals. Components receive their dependencies as arguments.
- In C++: no raw owning pointers, use RAII and standard containers; no undefined behavior; every non-obvious choice (container, iterator invalidation, complexity) gets a comment.
- Use the `logging` module, not `print`, except in explicit CLI output.
- Keep modules under about 400 lines; split when they grow.
- Public items get docstrings. Market-mechanics code gets comments explaining why the rule exists.
- Engine logic is tested with fixtures in `tests/fixtures/` plus property tests; the two engines are tested against each other.
- Every reported number in the README or in `docs/results.md` must come from a command that can be re-run, and that command is written next to the number.

Platforms: Linux and macOS are first-class. Windows works through WSL.

Performance targets:

- the C++ engine is at least 10x faster than the Python engine on the benchmark stream, and the actual measured ratio is recorded
- the C++ engine reports throughput in orders per second and p99 latency per order
- a 200-seed evaluation of one strategy finishes in a reasonable time on a laptop (target: under 10 minutes with the C++ engine)

## Honesty Rules

- Results are from a simulated market with simple agents. Never claim real-market profitability or "alpha".
- Report mean, standard deviation, and 95% confidence intervals, never a single best run.
- Test seeds are never used for training, tuning, or choosing thresholds.
- If the ML strategy does not beat TWAP or VWAP, record that in `docs/results.md` with an analysis of why. A clear negative result is acceptable; a hidden one is not.
- Every project README has a "Limitations" section and a "What failed" section.

## Agent Working Agreement

1. Identify the current phase from `docs/plan.md` and `git tag`. Do not skip phases or combine them.
2. Read the relevant docs and contracts before writing code.
3. State the plan first: list the commits you intend to make for this task.
4. Implement the smallest slice, verify it, commit it, and repeat.
5. After each task run `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest` (plus the C++ tests from Phase 3 onward), then show `git status` and `git log --oneline -n 15`.
6. Tick the phase checklist in `docs/plan.md` as part of the relevant commit.
7. At the end of every phase write `docs/learn/phase-N.md` in plain language: what the component does, the key concepts (limit order book, price-time priority, implementation shortfall, differential testing, data leakage, and so on), what to read next, and where in the code to look. End each learn doc with three practice exercises the human can do alone, such as "re-implement cancel without looking at the code" or "add a new order type and write its tests".
8. If blocked, record the blocker in `docs/integration.md` and ask. Do not silently work around a contract.
9. Ask before adding a dependency outside the tech stack table.
10. When market-mechanics behavior is unclear, choose the standard exchange behavior (price-time priority, trades execute at the resting order's price) and note it in a comment.
11. Explain as you go: when you write a non-trivial function, add a short comment on why it is written that way, since the human will study it line by line later.

## Golden Demo

The full definition lives in `docs/runbooks/demo.md`. The project is demo-ready when all of this works from a clean clone:

1. `pip install -e .` builds the C++ engine and installs the package without errors.
2. `pytest` passes, including the Python vs C++ differential test over one million random orders.
3. `python -m lobx.cli run --seed 0` writes an event log, and `python -m lobx.cli replay <log>` reproduces the final book exactly.
4. The benchmark script prints orders per second and p99 latency for both engines, and the C++ engine is at least 10x faster.
5. `python -m lobx.cli eval --strategies twap,vwap,adaptive --seeds 200` prints a results table with mean shortfall in bps and 95% confidence intervals.
6. The leakage test fails when a lookahead bug is injected and passes on the real pipeline.
7. `docker compose up` serves the API and the dashboard, which shows a live book and a strategy comparison.
8. The README has the results table, an architecture diagram, a Limitations section, and a "What failed" section.

## Definition of Done

- All eight phases are tagged and merged to `main`.
- CI is green on Linux and macOS and builds the C++ engine.
- The golden demo passes.
- `docs/results.md` contains every reported number with its command.
- `README.md` explains what LOBX is, how to build and run it, how to reproduce the results, and its limitations.
- Every phase has a `docs/learn/phase-N.md` with three practice exercises.
- The Git history shows small, logical commits for every stage, file, and feature.
