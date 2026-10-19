import pytest

from fakes import FakeView
from lobx.sim.actions import Submit
from lobx.strategies.base import ParentOrder
from lobx.strategies.schedule import ScheduledStrategy
from lobx.strategies.twap import TwapStrategy
from lobx.types import OrderType, Side

AGENT = 7


def _run(strategy: ScheduledStrategy, qty: int, slices: int, fill_ratio: float = 1.0) -> list[int]:
    """Drive a strategy over its slices on a fake market; return the child quantities sent."""
    parent = ParentOrder(AGENT, Side.SELL, qty, start_ns=1_000, end_ns=1_000 + slices * 100)
    strategy.reset(parent)
    view = FakeView()
    sent = []
    for k in range(slices):
        actions = strategy.step(1_000 + k * 100, view)
        child = sum(a.qty for a in actions if isinstance(a, Submit))
        assert all(isinstance(a, Submit) and a.order_type is OrderType.MARKET for a in actions)
        sent.append(child)
        view.positions[AGENT] = view.positions.get(AGENT, 0) - int(child * fill_ratio)
    return sent


def test_twap_splits_evenly() -> None:
    assert _run(TwapStrategy(4), qty=100, slices=4) == [25, 25, 25, 25]


def test_twap_distributes_the_remainder_and_totals_exactly() -> None:
    sent = _run(TwapStrategy(3), qty=100, slices=3)
    assert sum(sent) == 100
    assert max(sent) - min(sent) <= 1


def test_weights_shape_the_schedule() -> None:
    # Weights 1:3 -> a quarter of the order in the first slice, the rest in the second.
    assert _run(ScheduledStrategy([1, 3]), qty=100, slices=2) == [25, 75]


def test_zero_weight_slice_sends_nothing() -> None:
    assert _run(ScheduledStrategy([1, 0, 1]), qty=10, slices=3) == [5, 0, 5]


def test_shortfall_from_a_thin_book_is_carried_forward() -> None:
    # Only half of each order trades; the strategy must keep catching up, never over-send.
    parent = ParentOrder(AGENT, Side.SELL, 100, start_ns=0, end_ns=400)
    strategy = TwapStrategy(4)
    strategy.reset(parent)
    view = FakeView()
    view.positions[AGENT] = -10  # only 10 of the first 25 traded
    (action,) = strategy.step(100, view)  # slice 1: target 50, executed 10
    assert isinstance(action, Submit) and action.qty == 40


def test_calling_step_twice_in_a_slice_is_harmless() -> None:
    parent = ParentOrder(AGENT, Side.BUY, 100, start_ns=0, end_ns=400)
    strategy = TwapStrategy(4)
    strategy.reset(parent)
    view = FakeView()
    assert len(strategy.step(0, view)) == 1
    view.positions[AGENT] = 25
    assert strategy.step(50, view) == []


def test_nothing_before_start_and_everything_after_the_deadline() -> None:
    parent = ParentOrder(AGENT, Side.SELL, 90, start_ns=1_000, end_ns=1_300)
    strategy = TwapStrategy(3)
    strategy.reset(parent)
    view = FakeView()
    assert strategy.step(999, view) == []
    (late,) = strategy.step(5_000, view)  # long past the deadline: finish the whole order
    assert isinstance(late, Submit) and late.qty == 90


def test_never_overfills() -> None:
    parent = ParentOrder(AGENT, Side.SELL, 100, start_ns=0, end_ns=200)
    strategy = TwapStrategy(2)
    strategy.reset(parent)
    view = FakeView()
    view.positions[AGENT] = -100
    assert strategy.step(150, view) == []


def test_step_before_reset_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        TwapStrategy(2).step(0, FakeView())


@pytest.mark.parametrize("weights", [[], [0, 0], [1, -1]])
def test_rejects_bad_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        ScheduledStrategy(weights)


def test_twap_rejects_zero_slices() -> None:
    with pytest.raises(ValueError, match="num_slices"):
        TwapStrategy(0)


def test_target_at_gives_the_cumulative_goal_of_the_current_slice() -> None:
    strategy = TwapStrategy(4)
    strategy.reset(ParentOrder(AGENT, Side.SELL, 100, start_ns=1_000, end_ns=1_400))
    assert [strategy.target_at(ts) for ts in (999, 1_000, 1_099, 1_100, 1_399, 9_999)] == [
        0,
        25,
        25,
        50,
        100,
        100,
    ]
