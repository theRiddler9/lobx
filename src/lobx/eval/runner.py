"""Evaluate named strategies on the held-out test seeds."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from lobx.eval.harness import (
    EngineFactory,
    EvalConfig,
    SeedResult,
    StrategyFactory,
    background_trades,
    run_many,
)
from lobx.eval.seeds import TEST_SEEDS, TRAIN_SEEDS
from lobx.strategies.twap import TwapStrategy
from lobx.strategies.volume_profile import estimate_volume_profile
from lobx.strategies.vwap import VwapStrategy

logger = logging.getLogger(__name__)

STRATEGY_NAMES = ("twap", "vwap")


def build_factory(name: str, config: EvalConfig, engine: EngineFactory | None) -> StrategyFactory:
    """Return a factory that makes a fresh strategy called ``name``.

    VWAP's volume profile is estimated here, on the *training* seeds only.
    """
    if name == "twap":
        return lambda: TwapStrategy(config.num_slices)
    if name == "vwap":
        history = [background_trades(seed, config, engine) for seed in TRAIN_SEEDS]
        profile = estimate_volume_profile(
            history, config.start_ns, config.start_ns + config.window_ns, config.num_slices
        )
        return lambda: VwapStrategy(profile)
    raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(STRATEGY_NAMES)}")


def evaluate(
    names: Sequence[str],
    num_seeds: int,
    config: EvalConfig | None = None,
    engine: EngineFactory | None = None,
) -> dict[str, list[SeedResult]]:
    """Run every strategy in ``names`` on the same ``num_seeds`` test seeds."""
    if num_seeds < 2:
        raise ValueError("need at least 2 seeds to measure spread")
    config = config or EvalConfig()
    seeds = range(TEST_SEEDS.start, TEST_SEEDS.start + num_seeds)
    results: dict[str, list[SeedResult]] = {}
    for name in names:
        logger.info("running %s on %d seeds", name, num_seeds)
        results[name] = run_many(build_factory(name, config, engine), seeds, config, engine)
    return results
