"""The adaptive strategy's promises, checked in whole simulated markets.

* The parent order always completes by the deadline, whatever the model says.
* It never trades more than the parent quantity, at any moment.
"""

from collections.abc import Callable

import numpy as np
import pytest

from lobx.eval.harness import EvalConfig
from lobx.eval.leakage import Recorder, Recording, check_no_lookahead
from lobx.ml.model import Classifier, FloatArray
from lobx.sim.actions import Submit
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import build_background_agents
from lobx.strategies.adaptive import AdaptiveStrategy
from lobx.strategies.agent import StrategyAgent
from lobx.types import Side

CONFIG = EvalConfig(parent_qty=400, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6)
MARGIN = 3 * NS_PER_S


class FixedModel:
    """Always forecasts the same thing."""

    def __init__(self, down: float, flat: float, up: float) -> None:
        self.row = np.array([down, flat, up])

    def fit(self, train: object, validation: object = None) -> None:
        raise AssertionError

    def predict_proba(self, X: FloatArray) -> FloatArray:
        return np.tile(self.row, (len(X), 1))


class RandomModel:
    """Forecasts random probabilities: the worst case for a decision rule."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def fit(self, train: object, validation: object = None) -> None:
        raise AssertionError

    def predict_proba(self, X: FloatArray) -> FloatArray:
        raw = self.rng.random((len(X), 3))
        res: FloatArray = raw / raw.sum(axis=1, keepdims=True)
        return res


def _run(strategy: AdaptiveStrategy, seed: int, side: Side = Side.SELL) -> tuple[int, int]:
    """Run one market; return (shares executed, peak absolute position) of the strategy."""
    market_config = CONFIG.market_config(seed)
    agents = build_background_agents(market_config)
    parent = CONFIG.parent(len(agents))
    if side is not parent.side:
        from dataclasses import replace

        parent = replace(parent, side=side)
    agent = StrategyAgent(
        strategy, parent, CONFIG.num_slices, CONFIG.watch_interval_ns, CONFIG.watch_lookback_ns
    )
    market = Market(market_config, [*agents, agent])
    result = market.run()
    executed = abs(market.position(parent.agent_id))
    return executed, result.peak_abs_position[parent.agent_id]


MODELS: dict[str, Callable[[], Classifier | None]] = {
    "always-adverse": lambda: FixedModel(0.9, 0.05, 0.05),
    "always-favourable": lambda: FixedModel(0.05, 0.05, 0.9),
    "random": lambda: RandomModel(7),
    "no-model": lambda: None,
}


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("side", [Side.SELL, Side.BUY])
def test_the_parent_order_completes_and_is_never_overfilled(name: str, side: Side) -> None:
    for seed in range(1000, 1006):
        strategy = AdaptiveStrategy(
            MODELS[name](), CONFIG.num_slices, threshold=0.3, deadline_margin_ns=MARGIN
        )
        executed, peak = _run(strategy, seed, side)
        assert executed == CONFIG.parent_qty, (name, side, seed, executed)
        assert peak <= CONFIG.parent_qty, (name, side, seed, peak)


def test_never_overfills_when_already_completed() -> None:
    """A strategy whose parent quantity is fully filled emits no further submit actions."""
    from fakes import FakeView

    parent = CONFIG.parent(99)
    strategy = AdaptiveStrategy(None, CONFIG.num_slices, threshold=0.3)
    strategy.reset(parent)
    view = FakeView(bid=99, ask=101)
    view.positions[99] = -parent.qty  # already completely filled for a SELL
    actions = strategy.step(parent.start_ns + 100, view)
    assert not any(isinstance(a, Submit) for a in actions)


def test_no_resting_order_is_left_behind_at_the_end() -> None:
    market_config = CONFIG.market_config(1003)
    agents = build_background_agents(market_config)
    parent = CONFIG.parent(len(agents))
    strategy = AdaptiveStrategy(None, CONFIG.num_slices, 0.3, deadline_margin_ns=MARGIN)
    agent = StrategyAgent(
        strategy, parent, CONFIG.num_slices, CONFIG.watch_interval_ns, CONFIG.watch_lookback_ns
    )
    market = Market(market_config, [*agents, agent])
    market.run()
    assert market.open_orders(parent.agent_id) == ()


def test_the_adaptive_strategy_has_no_lookahead() -> None:
    market_config = CONFIG.market_config(1)
    agents = build_background_agents(market_config)
    start = CONFIG.start_ns - CONFIG.watch_lookback_ns
    recorder = Recorder(len(agents), start, CONFIG.watch_interval_ns)
    Market(market_config, [*agents, recorder]).run()
    frames = tuple(f for f in recorder.frames if f.ts <= CONFIG.start_ns + 20 * NS_PER_S)
    parent = CONFIG.parent(len(agents) + 1)

    def make(_: Recording) -> AdaptiveStrategy:
        return AdaptiveStrategy(RandomModelSeeded(), CONFIG.num_slices, 0.2)

    check_no_lookahead(make, Recording(frames), parent)


class RandomModelSeeded:
    """Deterministic in the *features*, so the same history always gives the same forecast."""

    def fit(self, train: object, validation: object = None) -> None:
        raise AssertionError

    def predict_proba(self, X: FloatArray) -> FloatArray:
        score = 1 / (1 + np.exp(-X[:, 1]))  # a function of imbalance_1 only
        return np.column_stack([1 - score, np.full(len(X), 0.0), score])
