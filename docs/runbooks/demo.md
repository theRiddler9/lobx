# Golden demo

The project is demo-ready when every step below works from a clean clone. Commands marked
"(planned)" become real in the phase shown in brackets; keep this file in sync as they land.

## Setup

```
git clone <repo-url> lobx && cd lobx
uv sync
```

(Once the C++ engine exists in Phase 3, `pip install -e .` also builds it.)

## Steps

1. **Install.** `pip install -e .` builds the C++ engine and installs the package without errors.
   [Phase 3]
2. **Tests.** `pytest` passes, including the Python vs C++ differential test over one million random
   orders. [Phase 3]
3. **Run and replay.** `python -m lobx.cli run --seed 0` writes an event log, and
   `python -m lobx.cli replay <log>` reproduces the final book exactly. [Phase 2]
4. **Benchmark.** The benchmark script prints orders per second and p99 latency for both engines;
   the C++ engine is at least 10x faster, and the measured ratio is recorded in `docs/results.md`.
   [Phase 3]
5. **Evaluate.** `python -m lobx.cli eval --strategies twap,vwap,adaptive --seeds 200` prints a
   results table with mean shortfall in bps and 95% confidence intervals. [Phase 4 and 6]
6. **Leakage test.** The leakage test fails when a lookahead bug is injected and passes on the real
   pipeline. [Phase 4]
7. **Service.** `docker compose up` serves the API (http://localhost:8000/docs) and the dashboard
   (http://localhost:8501), which shows a live book and a strategy comparison. Without Docker:
   `pip install -e ".[service]"`, then `python -m lobx.cli serve` and, in a second terminal,
   `streamlit run dashboard/app.py`. [Phase 7]
8. **README.** It has the results table, an architecture diagram, a Limitations section, and a
   "What failed" section. [Phase 8]

## What to check at each step

Every number shown in the demo must also appear in `docs/results.md` with the command that produced
it. If a step fails, record it in `docs/integration.md` before changing the plan.
