# ML plan

Owner area: `src/lobx/ml/`.

## Goal

Predict the short-horizon mid-price move from the order book, well enough to improve execution.

## Features (all computed from data at time t or earlier)

- spread and mid price
- order-book imbalance at depth 1 and depth 5
- mid-price returns over several windows
- recent trade-flow imbalance (uses `Fill.taker_side`)

## Label

Direction of the mid-price move over a fixed horizon (up, down, flat).

## Models

1. Naive baseline (majority class).
2. Logistic regression.
3. LightGBM.

## Splits

Train, validation, and test sets are separated by **seed ranges**, never by shuffling rows from the
same run. Test seeds are never used for tuning.

## Honest reporting

Accuracy and calibration on held-out seeds, compared with the naive baseline. If the model does not
beat the baseline, that goes in `docs/results.md` with an analysis.
