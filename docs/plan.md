# LOBX plan

This file is the single place to see where the project stands. `AGENTS.md` holds the rules and the
full commit lists per phase; this file holds the checklist and the integration status.

## Architecture in one picture

```
Agents (noise, market maker, momentum, execution agent)
  -> Orders (integer ticks, integer ns)
  -> Matching Engine (Python reference OR C++ via pybind11)
  -> Event Log (append-only, deterministic replay)
  -> Book Snapshots
  -> Feature Pipeline
  -> ML Model
  -> Execution Strategy (TWAP / VWAP / ML-adaptive)
  -> Evaluation Harness (many seeds, bps cost, confidence intervals)
```

The stage order is fixed. See `docs/contracts/shared-contracts.md` for the types that cross stages.

## Phase checklist

Tick a box in the same commit that finishes the phase work. A phase is done only when its exit
criteria hold, its `docs/learn/phase-N.md` exists, and the tag is created.

- [x] Phase 0: scaffold, docs, contracts (tag `phase-0-scaffold`)
- [x] Phase 1: Python reference engine (tag `phase-1-engine`)
- [x] Phase 2: event log, replay, simulator (tag `phase-2-sim`)
- [x] Phase 3: C++ engine and differential testing (tag `phase-3-cpp`)
- [x] Phase 4: baselines and evaluation harness (tag `phase-4-eval`)
- [ ] Phase 5: ML level 1, order-flow predictor (tag `phase-5-ml`)
- [ ] Phase 6: ML level 2, adaptive execution (tag `phase-6-adaptive`)
- [ ] Phase 7: service layer (tag `phase-7-service`)
- [ ] Phase 8: hardening and release (tag `phase-8-release`)
- [ ] Optional Phase 9: RL (stretch, only after Phase 8)

## Cut line (if time runs short)

1. Skip Optional Phase 9.
2. Skip the Streamlit dashboard and WebSocket in Phase 7.
3. Keep logistic regression only in Phase 5.
4. Never cut Phases 1 to 4 or a simple version of Phase 6.

## Integration status

Record blockers and cross-component decisions in `docs/integration.md`.

| Component | Status | Notes |
|---|---|---|
| Types and protocols | stubs | Phase 0 |
| Engine (Python) | done | Phase 1: limit and market orders, cancel, validation, 80 tests incl. property tests |
| Event log and replay | done | Phase 2: text log, LoggedBook, replay with self-trade check |
| Simulator | done | Phase 2: Market, 3 agent types, CLI run and replay, realism tests |
| Engine (C++) | done, differential-tested against Python | Phase 3 |
| Strategies and evaluation | TWAP, VWAP, harness, stats, leakage test done; adaptive in Phase 6 | Phase 4 |
| ML | not started | Phase 5 |
| Service | not started | Phase 7 |
