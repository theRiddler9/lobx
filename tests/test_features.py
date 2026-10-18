import numpy as np
import pytest

from lobx.ml.features import (
    FEATURE_NAMES,
    _mid_returns,
    _trade_flow,
    extract_features,
    spread,
)
from lobx.types import BookSnapshot, Fill, Side


def snap(ts: int, bids: list[tuple[int, int]], asks: list[tuple[int, int]]) -> BookSnapshot:
    return BookSnapshot(ts, tuple(bids), tuple(asks))


def test_spread_is_ask_minus_bid_in_ticks() -> None:
    snaps = [snap(0, [(99, 5)], [(102, 4)]), snap(1, [(100, 1)], [(101, 1)])]
    assert spread(snaps).tolist() == [3.0, 1.0]


def test_spread_is_nan_when_a_side_is_empty() -> None:
    snaps = [snap(0, [], [(102, 4)]), snap(1, [(99, 5)], [])]
    assert np.isnan(spread(snaps)).all()


def test_feature_matrix_has_one_named_column_per_feature() -> None:
    matrix = extract_features([snap(0, [(99, 5)], [(102, 4)])], [])
    assert matrix.shape == (1, len(FEATURE_NAMES))


def test_empty_input_gives_an_empty_matrix() -> None:
    assert extract_features([], []).shape == (0, len(FEATURE_NAMES))


@pytest.mark.parametrize("n", [1, 3])
def test_rows_match_snapshots(n: int) -> None:
    assert extract_features([snap(i, [(99, 5)], [(101, 4)]) for i in range(n)], []).shape[0] == n


def test_imbalance_at_depth_one_uses_only_the_best_level() -> None:
    from lobx.ml.features import _imbalance

    s = snap(0, [(100, 9), (99, 100)], [(101, 1), (102, 100)])
    assert _imbalance([s], 1)[0] == pytest.approx((9 - 1) / (9 + 1))


def test_imbalance_at_depth_five_sums_the_levels() -> None:
    from lobx.ml.features import _imbalance

    s = snap(0, [(100, 3), (99, 3)], [(101, 1), (102, 1), (103, 2)])
    assert _imbalance([s], 5)[0] == pytest.approx((6 - 4) / (6 + 4))


def test_imbalance_is_nan_for_a_one_sided_book() -> None:
    from lobx.ml.features import _imbalance

    assert np.isnan(_imbalance([snap(0, [(100, 3)], [])], 1)[0])


def test_mid_return_is_the_change_over_the_window() -> None:
    # Mids: 100, 101, 103, 102 -> over 2 steps: NaN, NaN, +3, +1.
    snaps = [snap(i, [(b, 1)], [(b + 2, 1)]) for i, b in enumerate([99, 100, 102, 101])]
    out = _mid_returns(snaps, 2)
    assert np.isnan(out[:2]).all()
    assert out[2:].tolist() == [3.0, 1.0]


def _fill(ts: int, qty: int, side: Side) -> Fill:
    return Fill(1, 2, side, 100, qty, ts)


def test_trade_flow_nets_buyer_and_seller_initiated_volume_in_the_window() -> None:
    fills = [_fill(10, 6, Side.BUY), _fill(20, 2, Side.SELL), _fill(100, 4, Side.SELL)]
    snaps = [snap(t, [(99, 1)], [(101, 1)]) for t in (5, 25, 105)]
    # Window 20 ns. t=5: no trades -> 0. t=25: trades at 10 (outside, 10 <= 5? no: 10 > 25-20=5, in)
    # and 20 -> (6 - 2) / 8 = 0.5. t=105: only the trade at 100 -> -1.
    out = _trade_flow(snaps, fills, 20)
    assert out.tolist() == [0.0, 0.5, -1.0]


def test_trade_flow_window_is_open_at_the_start_and_closed_at_the_end() -> None:
    fills = [_fill(10, 1, Side.BUY)]
    # At now=30 with window 20 the window is (10, 30]: the trade at exactly 10 is excluded.
    # At now=10 the window is (-10, 10]: the trade at exactly 10 is included.
    out = _trade_flow([snap(30, [(99, 1)], [(101, 1)]), snap(10, [(99, 1)], [(101, 1)])], fills, 20)
    assert out.tolist() == [0.0, 1.0]


def test_unsorted_fills_are_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        _trade_flow(
            [snap(0, [(99, 1)], [(101, 1)])], [_fill(5, 1, Side.BUY), _fill(1, 1, Side.BUY)], 10
        )
