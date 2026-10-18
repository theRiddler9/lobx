"""Features at time t may only use data with timestamp <= t (the rule in AGENTS.md).

Two independent checks:

1. *Truncation*: recompute row t from only the data available at t; it must equal row t computed
   from the whole run. A feature that peeks ahead would differ.
2. *Poison*: change everything after t to garbage; row t must not change.

A deliberately leaky feature is also run through both checks to prove they would notice.
"""

import numpy as np
import pytest

from lobx.ml.dataset import DataConfig, simulate_run
from lobx.ml.features import extract_features
from lobx.sim.clock import NS_PER_S
from lobx.types import BookSnapshot, Fill

CONFIG = DataConfig(duration_ns=60 * NS_PER_S)
PROBES = [0, 1, 7, 26, 80, 150, 250]


@pytest.fixture(scope="module")
def run() -> tuple[list[BookSnapshot], list[Fill]]:
    snapshots, fills = simulate_run(5, CONFIG)
    assert len(snapshots) > max(PROBES) + 1 and len(fills) > 50
    return snapshots, fills


def _known_at(
    snapshots: list[BookSnapshot], fills: list[Fill], t: int
) -> tuple[list[BookSnapshot], list[Fill]]:
    """Everything an observer at snapshot t could have seen."""
    now = snapshots[t].ts
    return snapshots[: t + 1], [f for f in fills if f.ts <= now]


def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def test_rows_computed_from_truncated_data_match_the_full_computation(
    run: tuple[list[BookSnapshot], list[Fill]],
) -> None:
    snapshots, fills = run
    full = extract_features(snapshots, fills)
    for t in PROBES:
        past_snaps, past_fills = _known_at(snapshots, fills, t)
        assert _equal(extract_features(past_snaps, past_fills)[t], full[t]), t


def _poison(
    snapshots: list[BookSnapshot], fills: list[Fill], t: int
) -> tuple[list[BookSnapshot], list[Fill]]:
    now = snapshots[t].ts
    bad_snaps = snapshots[: t + 1] + [
        BookSnapshot(s.ts, ((1, 999),), ((50_000, 999),)) for s in snapshots[t + 1 :]
    ]
    bad_fills = [f for f in fills if f.ts <= now] + [
        Fill(f.maker_order_id, f.taker_order_id, f.taker_side, 1, 10_000, f.ts)
        for f in fills
        if f.ts > now
    ]
    return bad_snaps, bad_fills


def test_rows_do_not_change_when_the_future_is_replaced_by_garbage(
    run: tuple[list[BookSnapshot], list[Fill]],
) -> None:
    snapshots, fills = run
    full = extract_features(snapshots, fills)
    for t in PROBES:
        bad_snaps, bad_fills = _poison(snapshots, fills, t)
        assert _equal(extract_features(bad_snaps, bad_fills)[t], full[t]), t


def _leaky_feature(snapshots: list[BookSnapshot]) -> np.ndarray:
    """BUGGY ON PURPOSE: 'the best bid one step from now'."""
    bids = np.array([s.bids[0][0] if s.bids else np.nan for s in snapshots], dtype=float)
    return np.append(bids[1:], np.nan)


def test_both_checks_would_catch_a_leaky_feature(
    run: tuple[list[BookSnapshot], list[Fill]],
) -> None:
    snapshots, fills = run
    full = _leaky_feature(snapshots)
    caught_by_truncation = caught_by_poison = 0
    for t in PROBES:
        past_snaps, _ = _known_at(snapshots, fills, t)
        caught_by_truncation += not _equal(_leaky_feature(past_snaps)[t : t + 1], full[t : t + 1])
        bad_snaps, _ = _poison(snapshots, fills, t)
        caught_by_poison += not _equal(_leaky_feature(bad_snaps)[t : t + 1], full[t : t + 1])
    assert caught_by_truncation == len(PROBES)
    assert caught_by_poison == len(PROBES)
