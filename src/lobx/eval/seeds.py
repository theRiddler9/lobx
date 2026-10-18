"""Seed ranges. Disjoint on purpose: what is learned on one range is judged on another.

* ``TRAIN_SEEDS``: fit anything that learns from data (the VWAP volume profile, ML models).
* ``VALIDATION_SEEDS``: tune settings and pick thresholds.
* ``TEST_SEEDS``: the final report only. Never used for training, tuning, or choosing a threshold.
"""

TRAIN_SEEDS = range(0, 100)
VALIDATION_SEEDS = range(100, 200)
TEST_SEEDS = range(1_000, 1_200)

assert not (set(TRAIN_SEEDS) | set(VALIDATION_SEEDS)) & set(TEST_SEEDS)
assert not set(TRAIN_SEEDS) & set(VALIDATION_SEEDS)
