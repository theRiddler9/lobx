# Phase 6: adaptive execution (ML level 2)

## What this phase built

An execution strategy that follows a TWAP schedule but uses the Phase 5 model to choose, moment by
moment, between a patient limit order and an immediate market order. Plus the tools to judge it
fairly: a paired comparison, threshold tuning on validation seeds, and an ablation with the model
switched off.

## Key concepts

- **Passive versus aggressive orders.** A *limit order* rests in the book and may fill later at
  your price, earning the spread; it may also never fill. A *market order* trades now at the
  current price, paying the spread and moving the price. Execution is the art of choosing.
- **Safety nets.** A strategy that waits must still finish: overdue shares are sold at once, and
  everything left is sold shortly before the deadline. Tests check the order always completes and
  is never overfilled (cancel the resting order *before* sending the market order).
- **The decision rule.** For a seller, signal = P(down) - P(up). Above a threshold, trade now.
- **Tuning on validation seeds.** The threshold is a setting. It is chosen on validation seeds so
  the test seeds stay untouched, and the tool refuses test seeds.
- **Paired comparison.** All strategies run on the same seeds, so per-seed differences cancel luck
  shared by both. Much sharper than comparing two separate averages.
- **Ablation.** Switch off the part you are claiming helps (the model) and see if the result
  survives. Here it did not: the model-free version was just as good. That is a valid, useful
  result, and the write-up says so plainly.
- **Negative and confounded results.** "Adaptive beats TWAP by 37 bps" is true and misleading.
  The honest sentence is "passivity beats TWAP; the model adds nothing".

## Where to look

- `src/lobx/strategies/adaptive.py` (the whole strategy, in small private methods)
- `src/lobx/eval/compare.py`, `tuning.py`, `runner.py`
- `tests/test_adaptive.py` (rules, one by one), `tests/test_adaptive_guarantees.py` (whole markets)
- `docs/results.md` (the failure analysis)

## What to read next

Almgren and Chriss again (market impact versus risk), then literature on limit-order fill
probability and adverse selection (why passive orders are harder in real markets).

## Practice exercises

1. Change the catch-up rule so a share is overdue after two slices instead of one. Predict the
   effect on cost and on completion, run `lobx tune`, and compare with what you predicted.
2. Use the model differently: instead of deciding whether to cross the spread, make the passive
   order bigger when the forecast is favourable and smaller when it is adverse. Does the ablation
   (model off) still match?
3. Write a test that builds an adversarial book (almost no depth on our side) and shows the
   deadline rule still completes the order by retrying every 200 ms in the last 5 seconds.
