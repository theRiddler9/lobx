"""Choose the adaptive strategy's threshold on validation seeds (never on test seeds)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lobx.eval.harness import EngineFactory, EvalConfig, run_many
from lobx.eval.seeds import TEST_SEEDS
from lobx.eval.stats import Summary, summarize
from lobx.ml.model import Classifier
from lobx.strategies.adaptive import AdaptiveStrategy
from lobx.strategies.twap import TwapStrategy


@dataclass(frozen=True, slots=True)
class TuningRow:
    """How one threshold did on the tuning seeds, as a paired difference from TWAP."""

    threshold: float
    adaptive_mean_bps: float
    difference_vs_twap: Summary  # adaptive minus TWAP per seed; negative means adaptive is cheaper


def tune_threshold(
    model: Classifier,
    thresholds: Sequence[float],
    seeds: Sequence[int],
    config: EvalConfig | None = None,
    engine: EngineFactory | None = None,
) -> list[TuningRow]:
    """Run TWAP and the adaptive strategy at each threshold on ``seeds`` and compare them.

    ``seeds`` must be validation seeds: tuning on the test seeds would make the final numbers
    optimistic, so any test seed is refused.
    """
    config = config or EvalConfig()
    if any(seed in TEST_SEEDS for seed in seeds):
        raise ValueError("test seeds must never be used for tuning")
    twap = run_many(lambda: TwapStrategy(config.num_slices), seeds, config, engine)
    rows = []
    for threshold in thresholds:
        adaptive = run_many(
            lambda t=threshold: AdaptiveStrategy(model, config.num_slices, t),  # type: ignore[misc]
            seeds,
            config,
            engine,
            watching=True,
        )
        pairs = [
            (a.shortfall_bps, b.shortfall_bps)
            for a, b in zip(adaptive, twap, strict=True)
            if a.shortfall_bps is not None and b.shortfall_bps is not None
        ]
        differences = [a - b for a, b in pairs]
        mean_adaptive = sum(a for a, _ in pairs) / len(pairs)
        rows.append(TuningRow(threshold, mean_adaptive, summarize(differences)))
    return rows


def tuning_table(rows: Sequence[TuningRow]) -> str:
    """Return a markdown table of the tuning results."""
    lines = [
        "| threshold | adaptive mean (bps) | adaptive - TWAP, mean (bps) | 95% CI (bps) |",
        "|---:|---:|---:|---|",
    ]
    for r in rows:
        d = r.difference_vs_twap
        lines.append(
            f"| {r.threshold:.2f} | {r.adaptive_mean_bps:.2f} | {d.mean:.2f} "
            f"| [{d.ci_low:.2f}, {d.ci_high:.2f}] |"
        )
    return "\n".join(lines)
