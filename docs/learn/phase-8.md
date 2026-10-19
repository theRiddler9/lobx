# Phase 8 and the whole project: final notes

## What this phase did

Phase 8 added no features. It checked the work and made it presentable: an end-to-end test, the
C++ engine under sanitizers, a static-analysis pass, a profile, a CI matrix for Linux and macOS, and
the README with its Limitations and "What failed" sections.

## Key concepts

- **Sanitizers.** AddressSanitizer and UndefinedBehaviorSanitizer instrument the C++ at compile
  time to catch memory errors and undefined behaviour while the tests run. A clean result is
  evidence, not proof: it covers only what the tests exercise.
- **Static analysis (clang-tidy).** Reads the code without running it and flags risky patterns.
  Most findings are style; the skill is deciding which are real, and writing down why a check is
  turned off (`.clang-tidy`).
- **Profiling.** Measure before optimising. The flame graph showed the matching engine, the part
  that was carefully optimised in C++, is about 5% of an evaluation run; the model call is about
  59%. Optimising the wrong thing would have changed nothing.
- **Honest reporting.** The project's most important result is a negative one: the ML model does
  not help execution; limit orders do. The README says so up front, and says what was not
  verified (the Docker image, the macOS build).
- **Reproducibility.** Every number has its command next to it, and a run is a pure function of
  `(config, seed)`, so anyone can re-run it.

## Where to look

- `tests/test_end_to_end.py`, `CMakeLists.txt` (`LOBX_SANITIZE`), `.clang-tidy`
- `benchmarks/profile_eval.py`, `docs/figures/flamegraph.svg`
- `docs/results.md` (all numbers), `docs/architecture.md`, `README.md`

## The whole project in one paragraph

Agents send orders to a matching engine (Python and C++, tested against each other); every event is
logged so a run replays exactly; a feature pipeline and a model forecast the next price move; TWAP,
VWAP and an adaptive strategy sell a parent order; an evaluation harness compares them over many
seeds with confidence intervals; a service and dashboard make it visible.

## What to read next

Almgren and Chriss (optimal execution), Avellaneda and Stoikov (market making), then the ABIDES
papers for a much larger agent-based simulator. For the engineering side: "Systems Performance"
(Gregg) for profiling.

## Practice exercises

1. Re-implement `cancel` in the Python engine from scratch without looking at the code, then run
   `pytest tests/test_book_cancel.py` and the differential test against your version.
2. Add a batch API to the C++ binding (`submit_many` taking a list of orders) and measure the
   speedup from Python with `benchmarks/bench_engines.py`. Does it reach 10x?
3. Replace the one-row-at-a-time model call in the adaptive strategy with a cheaper rule (predict
   every fifth wake, or compute features incrementally). Re-run the profile, then the 200-seed
   evaluation, and check whether the cost changed.
