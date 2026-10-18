import numpy as np
import pytest

from lobx.ml.labels import DOWN, FLAT, INVALID, UP, make_labels
from lobx.types import BookSnapshot


def _snaps(mids_bid: list[int | None]) -> list[BookSnapshot]:
    """A book per entry: bid b and ask b + 2; ``None`` gives a book with no bids."""
    return [
        BookSnapshot(i, ((b, 1),) if b is not None else (), ((100 if b is None else b + 2, 1),))
        for i, b in enumerate(mids_bid)
    ]


def test_labels_compare_the_mid_now_with_the_mid_at_the_horizon() -> None:
    # Bids 100, 101, 101, 99 (asks +2): mids move +1, 0, -2 per step.
    labels = make_labels(_snaps([100, 101, 101, 99]), horizon=1)
    assert labels.tolist() == [UP, FLAT, DOWN, INVALID]


def test_horizon_two_skips_a_step() -> None:
    labels = make_labels(_snaps([100, 101, 101, 99]), horizon=2)
    assert labels.tolist() == [UP, DOWN, INVALID, INVALID]


def test_the_last_horizon_rows_are_invalid() -> None:
    labels = make_labels(_snaps([100] * 6), horizon=3)
    assert labels.tolist() == [FLAT, FLAT, FLAT, INVALID, INVALID, INVALID]


def test_a_missing_mid_at_either_end_makes_the_label_invalid() -> None:
    labels = make_labels(_snaps([100, None, 102]), horizon=1)
    assert labels.tolist() == [INVALID, INVALID, INVALID]


def test_too_few_snapshots() -> None:
    assert make_labels(_snaps([100]), horizon=1).tolist() == [INVALID]
    assert make_labels([], horizon=1).shape == (0,)


def test_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        make_labels(_snaps([100, 101]), horizon=0)


def test_labels_are_a_small_integer_array() -> None:
    assert make_labels(_snaps([100, 101]), horizon=1).dtype == np.int8
