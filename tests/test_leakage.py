import pytest

from lobx.eval.harness import EvalConfig
from lobx.eval.leakage import (
    Frame,
    LookaheadError,
    Recorder,
    Recording,
    check_no_lookahead,
)
from lobx.sim.actions import Action, Submit
from lobx.sim.agents.base import MarketView
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import build_background_agents
from lobx.strategies.base import ExecutionStrategy, ParentOrder
from lobx.strategies.twap import TwapStrategy
from lobx.strategies.vwap import VwapStrategy
from lobx.types import Side

CONFIG = EvalConfig(parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6)


def _record(seed: int = 1) -> tuple[Recording, ParentOrder]:
    market_config = CONFIG.market_config(seed)
    agents = build_background_agents(market_config)
    interval = CONFIG.window_ns // CONFIG.num_slices
    recorder = Recorder(len(agents), CONFIG.start_ns, interval)
    Market(market_config, [*agents, recorder]).run()
    assert len(recorder.frames) >= CONFIG.num_slices
    return Recording(tuple(recorder.frames)), CONFIG.parent(agent_id=len(agents) + 1)


class _PeekingSeller:
    """BUGGY ON PURPOSE: sells more when the *next* frame's bid is lower. That is lookahead."""

    def __init__(self, recording: Recording) -> None:
        self._recording = recording

    def reset(self, parent: ParentOrder) -> None:
        self._parent = parent

    def step(self, ts: int, view: MarketView) -> list[Action]:
        frames = self._recording.frames
        i = next(i for i, f in enumerate(frames) if f.ts == ts)
        if i + 1 >= len(frames):
            return []
        future_bid = frames[i + 1].snapshot.bids[0][0]
        now_bid = view.best_bid()
        assert now_bid is not None
        return [Submit.market(Side.SELL, 10 if future_bid < now_bid else 1)]


def test_twap_and_vwap_have_no_lookahead() -> None:
    recording, parent = _record()

    def twap(_: Recording) -> ExecutionStrategy:
        return TwapStrategy(CONFIG.num_slices)

    def vwap(_: Recording) -> ExecutionStrategy:
        return VwapStrategy([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])

    check_no_lookahead(twap, recording, parent)
    check_no_lookahead(vwap, recording, parent)


def test_an_injected_lookahead_bug_is_caught() -> None:
    recording, parent = _record()
    with pytest.raises(LookaheadError, match="future data"):
        check_no_lookahead(_PeekingSeller, recording, parent)


def test_the_poison_really_changes_the_future_and_only_the_future() -> None:
    recording, _ = _record()
    poisoned = recording.poisoned_after(2)
    assert poisoned.frames[:3] == recording.frames[:3]
    assert poisoned.frames[3] != recording.frames[3]
    assert isinstance(poisoned.frames[3], Frame)
