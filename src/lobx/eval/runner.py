"""Evaluate named strategies on the held-out test seeds."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from lobx.eval.harness import (
    EngineFactory,
    EvalConfig,
    SeedResult,
    StrategyFactory,
    background_trades,
    run_many,
)
from lobx.eval.seeds import TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS
from lobx.ml.model import Classifier
from lobx.ml.store import load_model, save_model
from lobx.ml.train import train_models
from lobx.strategies.adaptive import AdaptiveStrategy
from lobx.strategies.twap import TwapStrategy
from lobx.strategies.volume_profile import estimate_volume_profile
from lobx.strategies.vwap import VwapStrategy

logger = logging.getLogger(__name__)

# "passive" is the adaptive strategy with the model switched off (an ablation, not a competitor).
STRATEGY_NAMES = ("twap", "vwap", "passive", "adaptive")
# Strategies that wake every 200 ms instead of once per slice.
WATCHING = frozenset({"passive", "adaptive"})
DEFAULT_MODEL_PATH = Path("models/lightgbm.json")
# Chosen on validation seeds 100..139 with `python -m lobx.cli tune` (see docs/results.md): the
# cheapest threshold tried. Never chosen on test seeds.
DEFAULT_THRESHOLD = 0.9


def obtain_model(path: Path, engine: EngineFactory | None = None) -> Classifier:
    """Load the LightGBM model at ``path``, or train one (about 2 minutes) and save it there."""
    if path.exists():
        return load_model(path)
    logger.info("no model at %s: training one on seeds 0..99", path)
    model = train_models(TRAIN_SEEDS, VALIDATION_SEEDS, engine=engine).lightgbm
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, path)
    return model


def build_factory(
    name: str,
    config: EvalConfig,
    engine: EngineFactory | None,
    model: Classifier | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> StrategyFactory:
    """Return a factory that makes a fresh strategy called ``name``.

    VWAP's volume profile is estimated here, on the *training* seeds only. ``adaptive`` needs a
    fitted ``model``.
    """
    if name == "twap":
        return lambda: TwapStrategy(config.num_slices)
    if name == "vwap":
        history = [background_trades(seed, config, engine) for seed in TRAIN_SEEDS]
        profile = estimate_volume_profile(
            history, config.start_ns, config.start_ns + config.window_ns, config.num_slices
        )
        return lambda: VwapStrategy(profile)
    if name == "passive":
        return lambda: AdaptiveStrategy(None, config.num_slices, threshold)
    if name == "adaptive":
        if model is None:
            raise ValueError("the adaptive strategy needs a fitted model")
        return lambda: AdaptiveStrategy(model, config.num_slices, threshold)
    raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(STRATEGY_NAMES)}")


def evaluate(
    names: Sequence[str],
    num_seeds: int,
    config: EvalConfig | None = None,
    engine: EngineFactory | None = None,
    model: Classifier | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, list[SeedResult]]:
    """Run every strategy in ``names`` on the same ``num_seeds`` test seeds."""
    if num_seeds < 2:
        raise ValueError("need at least 2 seeds to measure spread")
    config = config or EvalConfig()
    seeds = range(TEST_SEEDS.start, TEST_SEEDS.start + num_seeds)
    results: dict[str, list[SeedResult]] = {}
    for name in names:
        logger.info("running %s on %d seeds", name, num_seeds)
        factory = build_factory(name, config, engine, model, threshold)
        results[name] = run_many(factory, seeds, config, engine, watching=name in WATCHING)
    return results
