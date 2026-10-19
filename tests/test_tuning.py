import numpy as np
import pytest

from lobx.eval.harness import EvalConfig
from lobx.eval.tuning import tune_threshold, tuning_table
from lobx.ml.dataset import Dataset
from lobx.ml.model import NaiveModel
from lobx.sim.clock import NS_PER_S

SHORT = EvalConfig(parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6)


def _model() -> NaiveModel:
    model = NaiveModel()
    model.fit(
        Dataset(np.zeros((3, 1)), np.array([0, 1, 2], dtype=np.int64), np.zeros(3, dtype=np.int64))
    )
    return model


def test_test_seeds_are_refused() -> None:
    with pytest.raises(ValueError, match="test seeds"):
        tune_threshold(_model(), [0.5], [1000, 1001], SHORT)


def test_one_row_per_threshold_with_a_paired_difference() -> None:
    rows = tune_threshold(_model(), [0.5, 2.0], range(100, 104), SHORT)
    assert [r.threshold for r in rows] == [0.5, 2.0]
    assert all(r.difference_vs_twap.n == 4 for r in rows)
    table = tuning_table(rows)
    assert len(table.splitlines()) == 4
