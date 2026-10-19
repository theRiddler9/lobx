from dataclasses import replace

import pytest

from lobx.engine.native import NativeOrderBook, native_available
from lobx.eval.harness import (
    EvalConfig,
    background_trades,
    run_many,
    run_one_seed,
)
from lobx.eval.seeds import TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS
from lobx.sim.clock import NS_PER_S
from lobx.strategies.twap import TwapStrategy
from lobx.types import Side

# A short window keeps these tests fast; the defaults are exercised by the eval CLI tests.
SHORT = EvalConfig(parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6)


def _twap() -> TwapStrategy:
    return TwapStrategy(SHORT.num_slices)


def test_a_run_executes_the_whole_parent_order() -> None:
    result = run_one_seed(_twap, seed=1, config=SHORT)
    assert result.executed == SHORT.parent_qty
    assert result.shortfall_bps is not None


def test_same_seed_gives_the_same_result() -> None:
    assert run_one_seed(_twap, 2, SHORT) == run_one_seed(_twap, 2, SHORT)


def test_different_seeds_give_different_results() -> None:
    results = {r.shortfall_bps for r in run_many(_twap, range(5), SHORT)}
    assert len(results) > 1


def test_strategies_on_one_seed_see_the_same_arrival_price() -> None:
    # The arrival price is read before the strategy trades, so it must not depend on the strategy.
    slow = run_one_seed(_twap, 3, SHORT)
    fast = run_one_seed(lambda: TwapStrategy(2), 3, replace(SHORT, num_slices=2))
    assert slow.arrival_mid2 == fast.arrival_mid2


def test_buying_also_works() -> None:
    buy = replace(SHORT, side=Side.BUY)
    assert run_one_seed(_twap, 1, buy).executed == buy.parent_qty


@pytest.mark.skipif(not native_available(), reason="C++ engine not built")
def test_both_engines_give_the_same_result() -> None:
    from lobx.engine.book_py import PythonOrderBook

    assert run_one_seed(_twap, 4, SHORT, PythonOrderBook) == run_one_seed(
        _twap, 4, SHORT, NativeOrderBook
    )


def test_background_trades_are_inside_the_run() -> None:
    trades = background_trades(1, SHORT)
    assert trades
    assert all(0 <= ts <= SHORT.start_ns + SHORT.window_ns and qty > 0 for ts, qty in trades)


def test_seed_ranges_are_disjoint() -> None:
    train, validation, test = set(TRAIN_SEEDS), set(VALIDATION_SEEDS), set(TEST_SEEDS)
    assert not (train & validation) and not (train & test) and not (validation & test)
    assert len(TEST_SEEDS) >= 200


def test_a_watching_strategy_runs_and_sees_the_same_arrival_price() -> None:
    import numpy as np

    from lobx.ml.dataset import Dataset
    from lobx.ml.model import NaiveModel
    from lobx.strategies.adaptive import AdaptiveStrategy

    model = NaiveModel()
    model.fit(
        Dataset(np.zeros((3, 1)), np.array([0, 1, 2], dtype=np.int64), np.zeros(3, dtype=np.int64))
    )
    watched = run_one_seed(
        lambda: AdaptiveStrategy(
            model, SHORT.num_slices, threshold=0.5, deadline_margin_ns=3 * NS_PER_S
        ),
        seed=3,
        config=SHORT,
        watching=True,
    )
    plain = run_one_seed(_twap, 3, SHORT)
    assert watched.arrival_mid2 == plain.arrival_mid2
    assert watched.executed == SHORT.parent_qty
