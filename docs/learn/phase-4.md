# Phase 4: baselines and the evaluation harness

## What this phase built

Two textbook execution algorithms (TWAP and VWAP), a metric that prices an execution
(implementation shortfall), a harness that runs a strategy inside the simulated market over many
seeds, summary statistics with confidence intervals, and a test that proves the harness can catch a
strategy that peeks at the future.

## Key concepts

- **Parent and child orders.** A parent order ("sell 3,000 shares in 5 minutes") is too big to
  send at once: it would eat through the book and move the price against you (market impact). So a
  strategy sends small child orders over time. Almgren and Chriss formalize the trade-off: trade
  fast and pay impact; trade slowly and pay price risk.
- **TWAP** sends equal slices at equal times. **VWAP** sends slices proportional to the expected
  volume in each time bucket. Both are `ScheduledStrategy`: only the weights differ. In this
  simulated market volume is flat over time, so the estimated profile is flat and VWAP is almost
  TWAP. That is a fact about the simulator (no intraday seasonality), not a bug.
- **Cumulative targets.** The strategy compares what it *wants done by now* with what has *actually
  traded* (read from its position), so a thin-book shortfall carries into the next slice.
- **Implementation shortfall.** Cost versus trading everything instantly at the arrival mid-price,
  in basis points (1 bp = 0.01%). Positive = cost. See `eval/metrics.py` and its hand-computed
  tests.
- **Paired seeds.** Every strategy runs on the same seeds, so they face the same starting market.
- **Train / validation / test seeds.** VWAP's profile is learned on training seeds; results are
  reported on test seeds that were never used for anything else.
- **Mean, standard deviation, confidence interval.** The standard deviation says how much one run
  varies; the 95% confidence interval says how precisely 200 runs pin down the *mean*.
- **Lookahead (data leakage).** Using information from after the decision time. The poison test
  replaces the future with garbage; honest strategies cannot notice, cheaters can.

## Where to look

- `src/lobx/strategies/schedule.py`, `twap.py`, `vwap.py`, `volume_profile.py`, `agent.py`
- `src/lobx/eval/metrics.py`, `harness.py`, `stats.py`, `report.py`, `leakage.py`, `runner.py`
- `tests/test_metrics.py`, `tests/test_leakage.py`, `tests/test_schedule_strategies.py`

## What to read next

Almgren and Chriss, "Optimal Execution of Portfolio Transactions"; then think about what an
adaptive strategy could exploit that a fixed schedule cannot (Phase 5 and 6).

## Practice exercises

1. Re-derive the shortfall formula for a buy order by hand, then write a test with your own numbers
   and check it against `implementation_shortfall_bps`.
2. Add a "front-loaded" strategy (weights decreasing over time) as a `ScheduledStrategy`. Predict
   whether it beats TWAP in this market, then run `lobx eval` to see if you were right.
3. Write a second buggy strategy that leaks the future in a *different* way (for example it reads
   the final frame) and confirm `check_no_lookahead` catches it.
