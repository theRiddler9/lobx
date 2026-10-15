# Evaluation plan

Owner area: `src/lobx/eval/`.

## Goal

Compare execution strategies fairly across many seeds and report uncertainty honestly.

## Metric

Implementation shortfall in basis points: how much worse the average execution price is than the
arrival mid price, signed so that a positive number is a cost. Hand-computed examples back the
tests.

## Harness

- Run one seed: build the market, run background agents, run the strategy, collect fills.
- Run many seeds (200 or more): train seeds, validation seeds, test seeds are disjoint ranges.
- Statistics: mean, standard deviation, 95% confidence interval; paired comparison against TWAP and
  VWAP on the same seeds.

## Leakage test

An intentionally broken strategy that peeks at the future must be caught by the harness. The test
fails if the lookahead is not detected.

## Output

A results table and plots, each number tied to the command that produced it in `docs/results.md`.
