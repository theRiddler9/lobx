import pytest

from fakes import FakeView
from lobx.strategies.base import ParentOrder
from lobx.strategies.volume_profile import estimate_volume_profile
from lobx.strategies.vwap import VwapStrategy
from lobx.types import Side


def test_profile_is_the_average_share_per_bucket() -> None:
    # Window [0, 100) in 2 buckets. Run A: 30 | 10 -> shares .75/.25. Run B: 5 | 5 -> .5/.5.
    run_a = [(10, 30), (60, 10)]
    run_b = [(0, 5), (99, 5)]
    assert estimate_volume_profile([run_a, run_b], 0, 100, 2) == pytest.approx((0.625, 0.375))


def test_trades_outside_the_window_are_ignored() -> None:
    run = [(-1, 100), (10, 4), (100, 100)]  # the window is half-open: ts=100 is outside
    assert estimate_volume_profile([run], 0, 100, 1) == (1.0,)


def test_busy_runs_do_not_outweigh_quiet_ones() -> None:
    busy = [(10, 1_000_000)]  # all in bucket 0
    quiet = [(60, 1)]  # all in bucket 1
    assert estimate_volume_profile([busy, quiet], 0, 100, 2) == pytest.approx((0.5, 0.5))


def test_empty_history_is_an_error() -> None:
    with pytest.raises(ValueError, match="no trades"):
        estimate_volume_profile([[], [(500, 1)]], 0, 100, 2)


def test_vwap_follows_the_profile() -> None:
    parent = ParentOrder(1, Side.SELL, 100, start_ns=0, end_ns=300)
    strategy = VwapStrategy([0.2, 0.5, 0.3])
    strategy.reset(parent)
    view = FakeView()
    sent = []
    for ts in (0, 100, 200):
        (action,) = strategy.step(ts, view)
        sent.append(action.qty)  # type: ignore[union-attr]
        view.positions[1] = view.positions.get(1, 0) - action.qty  # type: ignore[union-attr]
    assert sent == [20, 50, 30]
