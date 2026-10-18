import numpy as np
import pytest

from lobx.ml.dataset import Dataset
from lobx.ml.evaluate import (
    accuracy,
    calibration_table,
    directional_accuracy,
    evaluate,
    expected_calibration_error,
    log_loss,
)
from lobx.ml.model import NaiveModel

Y = np.array([0, 1, 2, 2], dtype=np.int64)
PROBA = np.array(
    [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1]]
)  # the last row is wrong


def test_accuracy_counts_rows_whose_top_class_is_right() -> None:
    assert accuracy(PROBA, Y) == 0.75


def test_log_loss_matches_a_hand_computation() -> None:
    expected = -(np.log(0.7) + np.log(0.5) + np.log(0.8) + np.log(0.1)) / 4
    assert log_loss(PROBA, Y) == pytest.approx(expected)


def test_log_loss_survives_a_zero_probability() -> None:
    assert np.isfinite(log_loss(np.array([[1.0, 0.0, 0.0]]), np.array([1])))


def test_directional_accuracy_ignores_flat_rows() -> None:
    # Moved rows: 0 (down, p_up<p_down ok), 2 (up, ok), 3 (up, but p_up<p_down: wrong) -> 2/3.
    assert directional_accuracy(PROBA, Y) == pytest.approx(2 / 3)


def test_directional_accuracy_with_no_moves_is_nan() -> None:
    assert np.isnan(directional_accuracy(PROBA[:1], np.array([1])))


def test_calibration_of_perfectly_calibrated_predictions_is_zero() -> None:
    # 100 rows predicted 0.25 where exactly 25 happened; 100 rows predicted 0.75 with 75 happening.
    predicted = np.array([0.25] * 100 + [0.75] * 100)
    happened = np.array([True] * 25 + [False] * 75 + [True] * 75 + [False] * 25)
    table = calibration_table(predicted, happened, bins=4)
    assert [b.count for b in table] == [100, 100]
    assert expected_calibration_error(table) == pytest.approx(0.0)


def test_overconfident_predictions_have_a_calibration_gap() -> None:
    predicted = np.array([0.9] * 100)
    happened = np.array([True] * 50 + [False] * 50)
    assert expected_calibration_error(calibration_table(predicted, happened)) == pytest.approx(0.4)


def test_probability_one_lands_in_the_last_bin() -> None:
    table = calibration_table(np.array([1.0]), np.array([True]), bins=10)
    assert table[0].high == 1.0 and table[0].count == 1


def test_evaluate_scores_the_naive_model_at_its_class_frequency() -> None:
    train = Dataset(
        np.zeros((4, 1)), np.array([1, 1, 1, 0], dtype=np.int64), np.zeros(4, dtype=np.int64)
    )
    model = NaiveModel()
    model.fit(train)
    test = Dataset(
        np.zeros((4, 1)), np.array([1, 1, 0, 2], dtype=np.int64), np.zeros(4, dtype=np.int64)
    )
    result = evaluate(model, test)
    assert result.accuracy == 0.5 and result.rows == 4
