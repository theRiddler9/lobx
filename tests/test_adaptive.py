import numpy as np
import pytest

from fakes import FakeView
from lobx.ml.features import FEATURE_NAMES
from lobx.ml.model import NUM_CLASSES, FloatArray
from lobx.sim.actions import Cancel, Submit
from lobx.strategies.adaptive import HISTORY, AdaptiveStrategy
from lobx.strategies.base import ParentOrder
from lobx.types import Fill, Side

AGENT = 9
START, END = 100_000_000_000, 160_000_000_000
STEP = 200_000_000


class ConstantModel:
    """A stand-in model that always returns the same probabilities and records its inputs."""

    def __init__(self, down: float, flat: float, up: float) -> None:
        self.proba = np.array([down, flat, up])
        self.seen: list[FloatArray] = []

    def fit(self, train: object, validation: object = None) -> None:
        raise AssertionError("not trained here")

    def predict_proba(self, X: FloatArray) -> FloatArray:
        self.seen.append(X.copy())
        return np.tile(self.proba, (len(X), 1))


def make(
    side: Side = Side.SELL, model: ConstantModel | None = None, threshold: float = 0.1
) -> tuple[AdaptiveStrategy, ConstantModel]:
    model = model or ConstantModel(0.2, 0.5, 0.3)
    strategy = AdaptiveStrategy(model, num_slices=6, threshold=threshold)
    strategy.reset(ParentOrder(AGENT, side, 600, START, END))
    return strategy, model


def warm_up(strategy: AdaptiveStrategy, view: FakeView, steps: int = HISTORY) -> int:
    """Feed ``steps`` observations ending just before START; return the next timestamp."""
    ts = START - steps * STEP
    for _ in range(steps):
        view.bid, view.ask = 99, 101
        strategy.observe(ts, view)
        ts += STEP
    return ts


def test_no_probabilities_until_enough_history() -> None:
    strategy, _ = make()
    view = FakeView(bid=99, ask=101)
    for i in range(HISTORY - 1):
        strategy.observe(i * STEP, view)
    assert strategy.probabilities((HISTORY - 1) * STEP, view) is None


def test_probabilities_come_from_the_model_on_the_latest_feature_row() -> None:
    strategy, model = make()
    view = FakeView(bid=99, ask=101)
    ts = warm_up(strategy, view) - STEP
    assert strategy.probabilities(ts, view) == pytest.approx((0.2, 0.5, 0.3))
    assert model.seen[-1].shape == (1, len(FEATURE_NAMES))


def test_a_one_sided_book_gives_no_probabilities() -> None:
    strategy, _ = make()
    view = FakeView(bid=99, ask=None)
    for i in range(HISTORY):
        strategy.observe(i * STEP, view)
    assert strategy.probabilities(HISTORY * STEP, view) is None


def test_only_the_recent_tape_is_requested() -> None:
    strategy, model = make()
    view = FakeView(bid=99, ask=101)
    ts = warm_up(strategy, view) - STEP
    view.tape = [Fill(1, 2, Side.BUY, 100, 5, ts - 1_000_000)]
    strategy.probabilities(ts, view)
    flow_1s = FEATURE_NAMES.index("trade_flow_1s")
    assert model.seen[-1][0, flow_1s] == 1.0  # one buyer-initiated trade just happened


def test_the_adverse_signal_depends_on_the_side() -> None:
    sell, _ = make(Side.SELL)
    buy, _ = make(Side.BUY)
    probabilities = (0.5, 0.2, 0.3)
    assert sell.adverse_signal(probabilities) == pytest.approx(0.2)  # a fall hurts a seller
    assert buy.adverse_signal(probabilities) == pytest.approx(-0.2)  # a fall helps a buyer


def test_step_before_reset_is_an_error() -> None:
    strategy = AdaptiveStrategy(ConstantModel(0, 1, 0), 6, 0.1)
    with pytest.raises(RuntimeError, match="reset"):
        strategy.step(0, FakeView())


def test_step_keeps_only_the_snapshots_the_features_need() -> None:
    strategy, _ = make()
    view = FakeView(bid=99, ask=101)
    for i in range(HISTORY * 3):
        assert strategy.step(i * STEP, view) == []
    assert NUM_CLASSES == 3
    assert len(strategy._history) == HISTORY


def started(side: Side = Side.SELL) -> tuple[AdaptiveStrategy, FakeView, int]:
    strategy, model = make(side)
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    return strategy, view, START


def test_nothing_is_sent_before_the_parent_window_starts() -> None:
    strategy, _ = make()
    view = FakeView(bid=99, ask=101)
    assert strategy.step(START - STEP, view) == []


def test_a_seller_rests_a_limit_order_at_the_ask_for_what_is_owed() -> None:
    strategy, view, ts = started()
    assert strategy.step(ts, view) == [Submit.limit(Side.SELL, 101, 100)]  # slice 0 owes 100


def test_a_buyer_rests_at_the_bid() -> None:
    strategy, view, ts = started(Side.BUY)
    assert strategy.step(ts, view) == [Submit.limit(Side.BUY, 99, 100)]


def test_a_resting_order_at_the_touch_is_left_alone() -> None:
    strategy, view, ts = started()
    strategy.step(ts, view)
    view.open[AGENT] = (1,)
    assert strategy.step(ts + STEP, view) == []


def test_the_order_is_replaced_when_the_ask_falls_below_it() -> None:
    strategy, view, ts = started()
    strategy.step(ts, view)
    view.open[AGENT] = (7,)
    view.ask = 100  # someone undercut us: our 101 is no longer at the touch
    assert strategy.step(ts + STEP, view) == [Cancel(7), Submit.limit(Side.SELL, 100, 100)]


def test_a_partly_filled_order_keeps_its_place_while_it_still_covers_what_is_owed() -> None:
    strategy, view, ts = started()
    strategy.step(ts, view)
    view.open[AGENT] = (7,)
    view.positions[AGENT] = -40  # 40 of the 100 filled; 60 still rest, and 60 are still owed
    assert strategy.step(ts + STEP, view) == []


def test_the_order_is_topped_up_when_the_schedule_moves_ahead_of_it() -> None:
    strategy, view, ts = started()
    strategy.step(ts, view)
    view.open[AGENT] = (7,)
    view.positions[AGENT] = -100  # slice 0 is done, but the old order has nothing left to give
    next_slice = START + 10_000_000_000  # slice 1: target 200, so 100 are owed and none rest
    assert strategy.step(next_slice, view) == [Cancel(7), Submit.limit(Side.SELL, 101, 100)]


def test_a_filled_order_is_forgotten_and_nothing_is_posted_when_on_schedule() -> None:
    strategy, view, ts = started()
    strategy.step(ts, view)
    view.positions[AGENT] = -100  # filled completely: no open orders remain
    assert strategy.step(ts + STEP, view) == []


def test_aggressive_actions_cancel_first_then_cross_the_spread() -> None:
    strategy, view, ts = started()
    strategy.step(ts, view)  # rests an order
    view.open[AGENT] = (7, 8)
    actions = strategy._aggressive_actions(view, 60)
    assert actions == [Cancel(7), Cancel(8), Submit.market(Side.SELL, 60)]
    assert strategy._resting is None


def test_aggressive_actions_with_nothing_owed_only_cancel() -> None:
    strategy, view, _ = started()
    view.open[AGENT] = (7,)
    assert strategy._aggressive_actions(view, 0) == [Cancel(7)]


def test_an_adverse_forecast_makes_a_seller_trade_aggressively() -> None:
    # P(down) 0.7 vs P(up) 0.1: signal 0.6 > threshold 0.1 -> market order for what is owed.
    strategy, _ = make(model=ConstantModel(0.7, 0.2, 0.1))
    view = FakeView(bid=99, ask=101)
    ts = warm_up(strategy, view)
    assert strategy.step(START, view) == [Submit.market(Side.SELL, 100)]
    assert ts == START


def test_a_favourable_forecast_makes_a_seller_wait_passively() -> None:
    strategy, _ = make(model=ConstantModel(0.1, 0.2, 0.7))
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    assert strategy.step(START, view) == [Submit.limit(Side.SELL, 101, 100)]


def test_the_same_forecast_means_the_opposite_for_a_buyer() -> None:
    strategy, _ = make(Side.BUY, ConstantModel(0.7, 0.2, 0.1))  # a fall: good for a buyer
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    assert strategy.step(START, view) == [Submit.limit(Side.BUY, 99, 100)]


def test_a_signal_at_or_below_the_threshold_does_not_trigger_aggression() -> None:
    strategy, _ = make(model=ConstantModel(0.38, 0.3, 0.32), threshold=0.1)  # signal 0.06
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    (action,) = strategy.step(START, view)
    assert isinstance(action, Submit) and action.price == 101


def test_with_nothing_owed_an_adverse_forecast_sends_nothing() -> None:
    strategy, _ = make(model=ConstantModel(0.7, 0.2, 0.1))
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    view.positions[AGENT] = -100  # slice 0 is already done
    assert strategy.step(START + STEP, view) == []


def test_without_history_the_strategy_defaults_to_passive() -> None:
    strategy, _ = make(model=ConstantModel(0.9, 0.05, 0.05))
    view = FakeView(bid=99, ask=101)
    (action,) = strategy.step(START, view)  # only one observation so far: no forecast possible
    assert isinstance(action, Submit) and action.price == 101


def test_everything_left_is_crossed_inside_the_deadline_margin() -> None:
    strategy, _ = make(model=ConstantModel(0.1, 0.2, 0.7), threshold=0.5)  # model says: wait
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    view.positions[AGENT] = -450
    view.open[AGENT] = (7,)
    late = END - 5_000_000_000
    assert strategy.step(late, view) == [Cancel(7), Submit.market(Side.SELL, 150)]


def test_before_the_margin_a_favourable_forecast_is_still_respected() -> None:
    strategy, _ = make(model=ConstantModel(0.1, 0.2, 0.7))
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    view.positions[AGENT] = -500  # on schedule: nothing overdue
    (action,) = strategy.step(END - 5_000_000_000 - STEP, view)
    assert isinstance(action, Submit) and action.price is not None  # still a limit order


def test_shares_overdue_by_a_full_slice_are_traded_at_once() -> None:
    strategy, _ = make(model=ConstantModel(0.1, 0.2, 0.7))  # favourable: would wait
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    # In slice 1 (target 200) with only 50 done: slice 0's 100 was due a slice ago and is overdue.
    view.positions[AGENT] = -50
    view.open[AGENT] = (7,)
    assert strategy.step(START + 10_000_000_000, view) == [Cancel(7), Submit.market(Side.SELL, 150)]


def test_nothing_overdue_in_the_first_slice() -> None:
    strategy, _ = make(model=ConstantModel(0.1, 0.2, 0.7))
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    (action,) = strategy.step(START + 5 * STEP, view)
    assert isinstance(action, Submit) and action.price == 101


def test_a_finished_parent_order_only_cancels_leftovers() -> None:
    strategy, _ = make()
    view = FakeView(bid=99, ask=101)
    warm_up(strategy, view)
    view.positions[AGENT] = -600
    view.open[AGENT] = (7,)
    assert strategy.step(START + 5 * STEP, view) == [Cancel(7)]


def test_without_a_model_the_strategy_is_purely_passive_until_a_safety_net_applies() -> None:
    strategy = AdaptiveStrategy(None, num_slices=6, threshold=0.0)
    strategy.reset(ParentOrder(AGENT, Side.SELL, 600, START, END))
    view = FakeView(bid=99, ask=101)
    assert strategy.probabilities(START, view) is None
    assert strategy.step(START, view) == [Submit.limit(Side.SELL, 101, 100)]
    assert len(strategy._history) == 0  # no model, so nothing is recorded
