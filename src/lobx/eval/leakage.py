"""Lookahead (data leakage) check: a strategy at time t must not depend on the future.

Method (a *poison test*). Record a market run as a list of frames, one per decision time. Run the
strategy once on the real recording. Then, for each time ``k``, run it again on a recording whose
frames *after* ``k`` have been replaced by obviously wrong ones (prices shifted by 1,000 ticks).
An honest strategy cannot tell, so its decisions up to and including ``k`` must be identical. If
they change, the decision at some time depended on data from after that time: lookahead.

The strategy factory receives the recording. Honest strategies ignore it; a buggy one that holds
on to the recording and reads ahead is exactly what this test exists to catch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from numpy.random import Generator

from lobx.errors import LobxError
from lobx.sim.actions import Action, Submit
from lobx.sim.agents.base import MarketView
from lobx.strategies.base import ExecutionStrategy, ParentOrder
from lobx.types import BookSnapshot, OrderType, Side

_POISON_TICKS = 1_000
_DEPTH = 5


class LookaheadError(LobxError):
    """A strategy's decision changed when only the future was changed."""


@dataclass(frozen=True, slots=True)
class Frame:
    """The market as one decision time saw it."""

    ts: int
    snapshot: BookSnapshot


@dataclass(frozen=True, slots=True)
class Recording:
    """A market run captured at regular decision times."""

    frames: tuple[Frame, ...]

    def poisoned_after(self, k: int) -> Recording:
        """Return a copy whose frames after index ``k`` show prices shifted by 1,000 ticks."""

        def shifted(frame: Frame) -> Frame:
            snap = frame.snapshot
            return Frame(
                frame.ts,
                BookSnapshot(
                    snap.ts,
                    tuple((p + _POISON_TICKS, q) for p, q in snap.bids),
                    tuple((p + _POISON_TICKS, q) for p, q in snap.asks),
                ),
            )

        return Recording(self.frames[: k + 1] + tuple(shifted(f) for f in self.frames[k + 1 :]))


class Recorder:
    """An agent that does nothing but write down the book every ``interval_ns``."""

    def __init__(self, agent_id: int, start_ns: int, interval_ns: int) -> None:
        """Record from ``start_ns`` onwards."""
        self.agent_id = agent_id
        self.frames: list[Frame] = []
        self._start = start_ns
        self._interval = interval_ns

    def first_wake(self, rng: Generator) -> int:
        """Return the first recording time."""
        return self._start

    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]:
        """Store the current book; trade nothing."""
        self.frames.append(Frame(ts, view.snapshot(_DEPTH)))
        return []

    def next_wake(self, ts: int, rng: Generator) -> int | None:
        """Return the next recording time."""
        return ts + self._interval


class _RecordedView:
    """A ``MarketView`` built from one frame. Only what a strategy needs is implemented."""

    def __init__(self, frame: Frame, agent_id: int, position: int) -> None:
        self._frame = frame
        self._agent_id = agent_id
        self._position = position

    @property
    def ts(self) -> int:
        return self._frame.ts

    def best_bid(self) -> int | None:
        bids = self._frame.snapshot.bids
        return bids[0][0] if bids else None

    def best_ask(self) -> int | None:
        asks = self._frame.snapshot.asks
        return asks[0][0] if asks else None

    def mid2(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else bid + ask

    def last_trade_price(self) -> int | None:
        return None

    def reference_price(self) -> int:
        mid2 = self.mid2()
        if mid2 is None:
            raise LookaheadError("the recording has a one-sided book; pick another seed")
        return mid2 // 2

    def snapshot(self, depth: int) -> BookSnapshot:
        return self._frame.snapshot

    def position(self, agent_id: int) -> int:
        return self._position if agent_id == self._agent_id else 0

    def open_orders(self, agent_id: int) -> tuple[int, ...]:
        return ()


def _decisions(
    make_strategy: Callable[[Recording], ExecutionStrategy],
    recording: Recording,
    parent: ParentOrder,
) -> list[list[Action]]:
    """Run the strategy over a recording; return its actions at every frame."""
    strategy = make_strategy(recording)
    strategy.reset(parent)
    position = 0
    decisions: list[list[Action]] = []
    for frame in recording.frames:
        actions = strategy.step(frame.ts, _RecordedView(frame, parent.agent_id, position))
        decisions.append(actions)
        for action in actions:  # assume every market order fills completely
            if isinstance(action, Submit) and action.order_type is OrderType.MARKET:
                position += action.qty if action.side is Side.BUY else -action.qty
    return decisions


def check_no_lookahead(
    make_strategy: Callable[[Recording], ExecutionStrategy],
    recording: Recording,
    parent: ParentOrder,
) -> None:
    """Raise ``LookaheadError`` if any decision depends on frames after the one it was made at."""
    frames: Sequence[Frame] = recording.frames
    baseline = _decisions(make_strategy, recording, parent)
    for k in range(len(frames) - 1):
        poisoned = _decisions(make_strategy, recording.poisoned_after(k), parent)
        if poisoned[: k + 1] != baseline[: k + 1]:
            raise LookaheadError(
                f"decisions up to frame {k} (ts={frames[k].ts}) changed when only the "
                f"frames after it were altered: the strategy used future data"
            )
