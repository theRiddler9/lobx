# Integration log

Merge checkpoints, blockers, and decisions that affect more than one component. If a contract is
blocking, record it here and ask before working around it.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-16 | `Fill` gets a `taker_side` field (extends the first draft in `AGENTS.md`). | Phase 5 trade-flow features need the aggressor side without reading the event log. |
| 2026-10-16 | Exceptions live in `src/lobx/errors.py`. | Shared by the Python and C++ engines (the bindings translate to the same classes). |
| 2026-10-16 | Packaging starts on `setuptools`; it moves to `scikit-build-core` in Phase 3. | There is no C++ code to build until Phase 3. |
| 2026-10-16 | `ruff format` skips `*.md`. | `AGENTS.md` holds illustrative pseudo-code that must stay as written. |
| 2026-10-16 | Timestamps may repeat but never go backwards across `submit` and `cancel`; an id may be reused once its order is gone. | Catches simulator clock bugs early; keeps the engine check cheap (only resting ids are tracked). The C++ engine must match. |
| 2026-10-17 | `Submit` carries `side, order_type, price, qty` instead of a full `Order` (changes the first draft in `AGENTS.md`). | The market stamps order id, agent id and ts, so agents cannot forge them and order numbering is reproducible. |
| 2026-10-17 | `Agent` has `first_wake` and `next_wake` besides `wake`. | The market owns the clock; agents only say when they want to be woken. |
| 2026-10-17 | `MarketView.mid2()` returns twice the mid-price. | Keeps the mid an integer when the spread is an odd number of ticks. |
| 2026-10-17 | numpy is pinned to an exact version. | `Generator` streams are not guaranteed identical across numpy versions; golden-seed tests flag a change. |
| 2026-10-17 | Default market is tuned lively: market maker quotes 3 shares x 2 levels, noise traders 30% market orders with size scale 5. | The first defaults froze the price (0.04 ticks per second). About 1 tick per second is needed for execution and prediction experiments. |
| 2026-10-18 | The C++ engine has no test framework: plain `CHECK` macro tests run by CTest. | The build environment has no network for Catch2/doctest; fewer dependencies. |
| 2026-10-18 | `NativeOrderBook` raises `TypeError` (pybind11) for wrongly typed fields; range errors match the Python engine's types. | Type checks in the wrapper would slow every call; the differential test covers range-error parity. |
| 2026-10-18 | Measured speedup from Python is 2-3x, not 10x; the engine alone is about 15x. | Boundary overhead dominates. Recorded in `docs/results.md`; a batch API is a possible follow-up. |
| 2026-10-18 | `ParentOrder` is `(agent_id, side, qty, start_ns, end_ns)`; strategies read progress from `view.position(agent_id)`. | `ExecutionStrategy.step` only receives a view, so the id lets it find its own position without fill callbacks. |
| 2026-10-18 | `implementation_shortfall_bps` takes `arrival_mid2` (twice the mid), not `arrival_mid` (changes the draft contract). | The mid of a one-tick-odd spread is half a tick; doubling it keeps an integer, like `MarketView.mid2()`. |
| 2026-10-18 | The default parent order is 3,000 shares over 5 simulated minutes, not 50,000 over 30 minutes. | The simulated market trades about 3,000 shares a minute; the order is scaled to the market (about 20% of volume). |
| 2026-10-18 | Runs where the parent order does not complete are counted and excluded, not valued with an opportunity cost. | Simple and visible in the table; Phase 6 must make strategies finish and compare on seeds complete for both. |
| 2026-10-18 | The lookahead check is a poison test on recorded snapshots. | It shows decisions at t are unchanged when only data after t changes; it needs no live-market hooks. |
| 2026-10-18 | Snapshots for ML are taken by a `SnapshotRecorder` agent every 200 ms and re-stamped with the wake time. | The engine stamps a snapshot with the last order time; features need evenly spaced times. |
| 2026-10-18 | The model predicts down/flat/up of the mid over 1 s (5 snapshots); LightGBM and logistic are saved as JSON, `models/` is git-ignored. | Readable files that cannot run code; models are regenerated with `lobx train` (about 2 minutes). |
| 2026-10-18 | Feature `trade_flow_*` needs a recent-trades feed that `MarketView` does not expose. | Open for Phase 6: either add recent trades to `MarketView` (a contract change on its own branch) or drop the flow features for the live strategy. |
| 2026-10-19 | `MarketView.fills_after(ts)` added (own `contract/market-view-fills` branch, merged first). | Resolves the Phase 5 open item: the adaptive strategy sees the public trade tape, so it can compute the trade-flow features the model was trained on. |
| 2026-10-19 | The adaptive strategy wakes every 200 ms from 6 s before the window; baselines still wake once per slice. | Features need evenly spaced snapshots on the training grid; keeping the baselines unchanged keeps Phase 4 numbers reproducible. |
| 2026-10-19 | `DEFAULT_THRESHOLD = 0.9`, chosen on validation seeds 100..139. | Cheapest of {0.3, 0.5, 0.7, 0.9, 1.0}; a tie with 1.0, at which the model almost never fires (see `docs/results.md`). |
| 2026-10-19 | A `passive` strategy (model off) is part of the comparison. | Ablation: shows the 37 bps gain over TWAP comes from resting orders, not from the model. |
| 2026-10-19 | `websockets` (runtime) and `httpx` (dev) added although they are outside the AGENTS.md table. | uvicorn cannot serve WebSocket without `websockets`; FastAPI's `TestClient` needs `httpx`. Asked first; the user approved. |
| 2026-10-19 | The service stack is an optional extra: `pip install -e ".[service]"`. | The engine, simulator and evaluation install without a web stack. Service tests skip themselves when the extra is missing. |
| 2026-10-19 | `run_detailed` and an `observer` hook added to the eval harness; `verdict` split out of `comparison_table`. | The service needs a run's fills, a recording of the book, and the same verdict wording; both are behaviour-preserving (tested). |
| 2026-10-19 | The WebSocket streams a *recorded* run, paced by `delay_ms`, not a market computed at wall-clock speed. | A run simulates in about a second; pacing the replay makes the book watchable. The data are exactly the run's snapshots (one per simulated second, 10 levels). |
| 2026-10-19 | The service never trains a model; adaptive without `models/lightgbm.json` is a 409. | Training takes minutes; a request should not silently block on it. The Docker image trains at build time. |
| 2026-10-19 | The Docker image was not built in the sandbox (Docker Hub unreachable). | The same steps were replayed in a clean venv (install, train, serve, dashboard health). First real build still to be done on a machine with network access. |
| 2026-10-19 | `LOBX_SANITIZE` CMake option, a `lobx_stress` ctest, `.clang-tidy`, a CI sanitizer job. | Phase 8 hardening; sanitizers found nothing, clang-tidy led to small clean-ups (see `docs/results.md`). |
| 2026-10-19 | LICENSE is MIT with the holder "LOBX contributors". | AGENTS.md requires a LICENSE but names none; the holder should be replaced with the owner's name. |
| 2026-10-19 | Optional Phase 9 (RL) not started. | AGENTS.md says only after Phase 8; it is on the cut line. |

## Blockers

None.

## Merge checkpoints

- Contracts merge to `main` before dependent work starts.
- Rebase on `origin/main` before opening a PR; merge with merge commits, no squash.
