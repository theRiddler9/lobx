"""Build training data (features and labels) by simulating runs.

Rows from one run are strongly correlated with each other (neighbouring snapshots share almost the
same book), so train, validation and test sets are made from **different seeds**, never by
shuffling rows. Every row remembers its seed so that this can be checked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.protocol import OrderBook
from lobx.ml.features import FEATURE_NAMES, extract_features
from lobx.ml.labels import INVALID, make_labels
from lobx.sim.agents.recorder import SnapshotRecorder
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import MarketConfig, build_background_agents
from lobx.types import BookSnapshot, Fill, FillEvent

EngineFactory = Callable[[], OrderBook]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """How runs are sampled: one snapshot per ``interval_ns``, label horizon in snapshots."""

    duration_ns: int = 300 * NS_PER_S
    interval_ns: int = 200_000_000  # 200 ms
    horizon: int = 5  # so labels look 1 second ahead
    warmup_ns: int = 30 * NS_PER_S  # rows before this are dropped: the book is still filling
    depth: int = 5


@dataclass(frozen=True, slots=True)
class Dataset:
    """Feature rows ``X``, class labels ``y``, and the seed each row came from."""

    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    seeds: npt.NDArray[np.int64]

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.y)


def simulate_run(
    seed: int, config: DataConfig, engine: EngineFactory | None = None
) -> tuple[list[BookSnapshot], list[Fill]]:
    """Run the background market for ``seed``; return its snapshots and all of its trades."""
    market_config = MarketConfig(seed=seed, duration_ns=config.duration_ns)
    agents = build_background_agents(market_config)
    recorder = SnapshotRecorder(len(agents), config.interval_ns, config.depth)
    book = (engine or PythonOrderBook)()
    result = Market(market_config, [*agents, recorder], book=book).run()
    fills = [event.fill for event in result.log if isinstance(event, FillEvent)]
    return recorder.snapshots, fills


def build_run_dataset(
    seed: int, config: DataConfig, engine: EngineFactory | None = None
) -> Dataset:
    """Return the usable rows of one run: complete features, a known label, past warm-up."""
    snapshots, fills = simulate_run(seed, config, engine)
    X = extract_features(snapshots, fills)
    y = make_labels(snapshots, config.horizon)
    ts = np.array([s.ts for s in snapshots])
    keep = ~np.isnan(X).any(axis=1) & (y != INVALID) & (ts >= config.warmup_ns)
    return Dataset(
        X[keep], y[keep].astype(np.int64), np.full(int(keep.sum()), seed, dtype=np.int64)
    )


def build_dataset(
    seeds: Iterable[int], config: DataConfig | None = None, engine: EngineFactory | None = None
) -> Dataset:
    """Concatenate the datasets of every seed, in order."""
    config = config or DataConfig()
    parts = [build_run_dataset(seed, config, engine) for seed in seeds]
    if not parts:
        return Dataset(
            np.empty((0, len(FEATURE_NAMES))),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return Dataset(
        np.concatenate([p.X for p in parts]),
        np.concatenate([p.y for p in parts]),
        np.concatenate([p.seeds for p in parts]),
    )
