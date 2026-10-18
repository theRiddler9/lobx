# Phase 5: the order-flow predictor (ML level 1)

## What this phase built

A pipeline that turns a simulated run into a table of numbers, trains three models to predict the
next second's mid-price move (down, flat, up), and scores them on seeds they never saw.

## Key concepts

- **Features** are what the model is allowed to look at: spread, imbalance (buy versus sell depth
  in the book), recent mid-price change, and recent trade flow (were aggressive buyers or sellers
  more active?). Each is a function of data at time t or earlier.
- **Labels** look forward on purpose: "which way did the mid move over the next second?" Features
  and labels live in separate files so the difference is hard to blur.
- **Data leakage** is a feature that quietly uses the future. It makes a model look brilliant in
  testing and useless in production. The tests check the rule two ways (recompute from only past
  data; replace the future with garbage) and prove the checks work on a deliberately leaky feature.
- **Splitting by seed, not by row.** Neighbouring rows of one run are nearly identical; shuffling
  rows into train and test would let the model memorize runs. Train, validation and test are
  disjoint sets of seeds.
- **Baselines.** A model is only good if it beats something simple. The naive model predicts the
  most common outcome. Always compare with it.
- **Logistic regression** (a weighted sum turned into probabilities) versus **gradient-boosted
  trees** (many small trees, each fixing the last one's mistakes): the trees can learn
  interactions the straight-line model cannot.
- **Accuracy, log loss, calibration.** Accuracy can mislead; log loss rewards honest
  probabilities; calibration asks whether "70%" means 70%.
- **Early stopping** halts training when a validation set stops improving, to avoid overfitting.

## Where to look

- `src/lobx/ml/features.py`, `labels.py`, `dataset.py`
- `src/lobx/ml/model.py`, `logistic.py`, `gbm.py`, `store.py`, `train.py`, `evaluate.py`
- `tests/test_features_causal.py` (the leakage checks), `tests/test_ml_train.py`

## What to read next

The scikit-learn guide on calibration, and the LightGBM parameters page. Then think about how a
predictor could change *how* you trade (Phase 6).

## Practice exercises

1. Add a new feature (for example, total depth at the best bid and ask). Add it to
   `FEATURE_NAMES`, write a unit test with a hand-computed example, make sure the causality tests
   still pass, and see whether LightGBM's accuracy changes.
2. Change the horizon from 5 to 10 snapshots and predict, before running, whether accuracy goes up
   or down and why (look at the class frequencies). Then check with `lobx train`.
3. Write a leaky feature of your own that is harder to spot (for example one that uses the *last*
   snapshot of the whole run to normalize), and confirm the truncation test catches it.
