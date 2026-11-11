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
from lobx.ml.mlp_classifier import MLPClassifier
from lobx.ml.model import Classifier, NaiveModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainedModels:
    """The fitted models."""

    naive: NaiveModel
    logistic: LogisticModel
    lightgbm: LightGBMModel
    mlp: MLPClassifier | None = None

    def by_name(self) -> dict[str, Classifier]:
        """Return the models keyed by name, baseline first."""
        models: dict[str, Classifier] = {
            "naive": self.naive,
            "logistic": self.logistic,
            "lightgbm": self.lightgbm,
        }
        if self.mlp is not None:
            models["mlp"] = self.mlp
        return models


def train_models(
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    config: DataConfig | None = None,
    engine: EngineFactory | None = None,
    include_mlp: bool = True,
) -> TrainedModels:
    """Fit the naive, logistic, LightGBM, and optional MLP models.

    ``train_seeds`` fit the models; ``validation_seeds`` stop LightGBM's and MLP's training early.
    The two sets must not overlap, and neither may contain the test seeds used later for reporting.
    """
    overlap = set(train_seeds) & set(validation_seeds)
    if overlap:
        raise ValueError(f"train and validation seeds overlap: {sorted(overlap)[:5]}")
    config = config or DataConfig()
    logger.info("building training data from %d seeds", len(train_seeds))
    train = build_dataset(train_seeds, config, engine)
    logger.info("building validation data from %d seeds", len(validation_seeds))
    validation = build_dataset(validation_seeds, config, engine)
    mlp_model = MLPClassifier(hidden_dim=32, epochs=40, patience=8) if include_mlp else None
    models = TrainedModels(NaiveModel(), LogisticModel(), LightGBMModel(), mlp=mlp_model)
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
