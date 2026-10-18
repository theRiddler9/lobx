"""Train every model on one set of seeds and score them on another."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lobx.ml.dataset import DataConfig, EngineFactory, build_dataset
from lobx.ml.evaluate import Evaluation, evaluate
from lobx.ml.gbm import LightGBMModel
from lobx.ml.labels import CLASS_NAMES
from lobx.ml.logistic import LogisticModel
from lobx.ml.model import Classifier, NaiveModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainedModels:
    """The three fitted models."""

    naive: NaiveModel
    logistic: LogisticModel
    lightgbm: LightGBMModel

    def by_name(self) -> dict[str, Classifier]:
        """Return the models keyed by name, baseline first."""
        return {"naive": self.naive, "logistic": self.logistic, "lightgbm": self.lightgbm}


def train_models(
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    config: DataConfig | None = None,
    engine: EngineFactory | None = None,
) -> TrainedModels:
    """Fit the naive, logistic and LightGBM models.

    ``train_seeds`` fit the models; ``validation_seeds`` stop LightGBM's training early. The two
    sets must not overlap, and neither may contain the test seeds used later for reporting.
    """
    overlap = set(train_seeds) & set(validation_seeds)
    if overlap:
        raise ValueError(f"train and validation seeds overlap: {sorted(overlap)[:5]}")
    config = config or DataConfig()
    logger.info("building training data from %d seeds", len(train_seeds))
    train = build_dataset(train_seeds, config, engine)
    logger.info("building validation data from %d seeds", len(validation_seeds))
    validation = build_dataset(validation_seeds, config, engine)
    models = TrainedModels(NaiveModel(), LogisticModel(), LightGBMModel())
    for name, model in models.by_name().items():
        logger.info("fitting %s on %d rows", name, len(train))
        model.fit(train, validation)
    return models


def evaluation_table(results: Mapping[str, Evaluation]) -> str:
    """Return a markdown table of model results (accuracy higher is better, log loss lower)."""
    lines = [
        "| model | rows | accuracy | log loss | direction acc. | ECE up | ECE down |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r.rows} | {r.accuracy:.4f} | {r.log_loss:.4f} "
            f"| {r.directional_accuracy:.4f} | {r.ece_up:.4f} | {r.ece_down:.4f} |"
        )
    return "\n".join(lines)


def score_models(
    models: TrainedModels,
    test_seeds: Sequence[int],
    config: DataConfig | None = None,
    engine: EngineFactory | None = None,
) -> dict[str, Evaluation]:
    """Evaluate every model on ``test_seeds`` (seeds never used to fit or stop anything)."""
    test = build_dataset(test_seeds, config or DataConfig(), engine)
    logger.info(
        "scoring on %d rows from %d seeds; classes: %s", len(test), len(test_seeds), CLASS_NAMES
    )
    return {name: evaluate(model, test) for name, model in models.by_name().items()}
