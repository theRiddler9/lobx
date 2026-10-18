import numpy as np

from lobx.ml.dataset import DataConfig, build_dataset, build_run_dataset, simulate_run
from lobx.ml.features import FEATURE_NAMES
from lobx.sim.clock import NS_PER_S

SMALL = DataConfig(duration_ns=60 * NS_PER_S, warmup_ns=10 * NS_PER_S)


def test_snapshots_are_evenly_spaced_and_fills_are_time_ordered() -> None:
    snapshots, fills = simulate_run(1, SMALL)
    gaps = {b.ts - a.ts for a, b in zip(snapshots, snapshots[1:], strict=False)}
    assert gaps == {SMALL.interval_ns}
    assert [f.ts for f in fills] == sorted(f.ts for f in fills)


def test_rows_are_complete_and_labelled() -> None:
    data = build_run_dataset(1, SMALL)
    assert len(data) > 100
    assert data.X.shape == (len(data), len(FEATURE_NAMES))
    assert not np.isnan(data.X).any()
    assert set(data.y.tolist()) <= {0, 1, 2}
    assert (data.seeds == 1).all()


def test_warmup_rows_are_dropped() -> None:
    full = build_run_dataset(1, DataConfig(duration_ns=60 * NS_PER_S, warmup_ns=0))
    warmed = build_run_dataset(1, SMALL)
    assert len(warmed) < len(full)


def test_same_seed_gives_the_same_data() -> None:
    a, b = build_run_dataset(3, SMALL), build_run_dataset(3, SMALL)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)


def test_multi_seed_dataset_keeps_each_row_seed() -> None:
    data = build_dataset([1, 2], SMALL)
    assert set(data.seeds.tolist()) == {1, 2}
    assert len(data) == len(build_run_dataset(1, SMALL)) + len(build_run_dataset(2, SMALL))


def test_no_seeds_gives_an_empty_dataset() -> None:
    assert len(build_dataset([], SMALL)) == 0
