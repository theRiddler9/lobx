import pytest

from lobx.ml.dataset import DataConfig
from lobx.ml.train import evaluation_table, score_models, train_models
from lobx.sim.clock import NS_PER_S

SMALL = DataConfig(duration_ns=90 * NS_PER_S, warmup_ns=10 * NS_PER_S)


def test_overlapping_train_and_validation_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        train_models([1, 2, 3], [3, 4], SMALL)


def test_models_beat_the_naive_baseline_on_unseen_seeds() -> None:
    models = train_models(range(0, 12), range(100, 104), SMALL)
    results = score_models(models, range(1000, 1004), SMALL)
    naive = results["naive"]
    for name in ("logistic", "lightgbm"):
        assert results[name].log_loss < naive.log_loss, name
        assert results[name].accuracy > naive.accuracy, name
        # The baseline always guesses the commoner direction, so it already scores above 50%.
        assert results[name].directional_accuracy > naive.directional_accuracy, name


def test_table_lists_every_model() -> None:
    models = train_models(range(0, 6), range(100, 102), SMALL)
    table = evaluation_table(score_models(models, range(1000, 1002), SMALL))
    assert [line.split("|")[1].strip() for line in table.splitlines()[2:]] == [
        "naive",
        "logistic",
        "lightgbm",
    ]
