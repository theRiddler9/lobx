"""The simulated market must look like a market: the Phase 2 exit criteria.

* the spread is positive on average,
* mid-price returns have fat tails (excess kurtosis above 0, the value for a normal distribution),
* a market maker's inventory stays bounded.

Two more checks guard against the market silently degenerating: the price must actually move (an
early default barely did), and it must not explode or let the book grow without limit.

The numbers are properties of a *simulation* of simple agents. They show the market is lively
enough to run execution experiments on; they say nothing about real markets.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import pytest

from lobx.sim.agents.market_maker import MarketMaker
from lobx.sim.agents.noise import NoiseTrader
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market, RunResult
from lobx.sim.population import MarketConfig

_SEEDS = (0, 1, 2, 3)
_SECONDS = 300


def excess_kurtosis(values: list[float]) -> float:
    """Fourth standardised moment minus 3, so a normal distribution scores about 0."""
    mean = statistics.fmean(values)
    variance = statistics.fmean([(v - mean) ** 2 for v in values])
    return statistics.fmean([(v - mean) ** 4 for v in values]) / variance**2 - 3


@dataclass
class _Runs:
    results: list[RunResult]
    resting_orders: list[int]

    @property
    def spreads(self) -> list[int]:
        return [s.spread for r in self.results for s in r.samples if s.spread is not None]

    @property
    def mid_changes(self) -> list[float]:
        """Change of the mid-price (in ticks) between consecutive one-second samples, pooled."""
        changes: list[float] = []
        for result in self.results:
            mids = [s.mid2 / 2 for s in result.samples if s.mid2 is not None]
            changes += [after - before for before, after in zip(mids, mids[1:], strict=False)]
        return changes


@pytest.fixture(scope="module")
def runs() -> _Runs:
    results, resting = [], []
    for seed in _SEEDS:
        market = Market(MarketConfig(seed=seed, duration_ns=_SECONDS * NS_PER_S))
        results.append(market.run())
        resting.append(len(market.book.tracker))
    return _Runs(results, resting)


def test_spread_is_always_positive_and_positive_on_average(runs: _Runs) -> None:
    assert min(runs.spreads) >= 1
    assert 1.0 < statistics.fmean(runs.spreads) < 5.0


def test_the_book_has_both_sides_almost_all_the_time(runs: _Runs) -> None:
    samples = [s for r in runs.results for s in r.samples]
    two_sided = sum(1 for s in samples if s.spread is not None)
    assert two_sided / len(samples) > 0.95


def test_mid_price_returns_have_fat_tails(runs: _Runs) -> None:
    changes = runs.mid_changes
    assert len(changes) > 1_000
    assert excess_kurtosis(changes) > 3.0


def test_the_price_actually_moves(runs: _Runs) -> None:
    # A frozen market (the first default had a standard deviation of 0.04 ticks per second) would
    # make every execution strategy look the same and give a model nothing to predict.
    assert 0.3 < statistics.pstdev(runs.mid_changes) < 5.0


def test_the_price_stays_near_where_it_opened(runs: _Runs) -> None:
    for result in runs.results:
        mids = [s.mid2 / 2 for s in result.samples if s.mid2 is not None]
        assert max(abs(m - result.config.opening_price) for m in mids) < 300


def test_trading_happens_and_self_trades_are_rare(runs: _Runs) -> None:
    trades = sum(r.trades for r in runs.results)
    self_trades = sum(r.self_trades for r in runs.results)
    assert trades > 5_000
    assert self_trades / trades < 0.05


def test_the_number_of_resting_orders_stays_bounded(runs: _Runs) -> None:
    # Noise traders cap their own open orders, so the book must not keep growing run after run.
    assert max(runs.resting_orders) < 2_000


def test_default_market_makers_stay_well_inside_their_position_limit(runs: _Runs) -> None:
    for result in runs.results:
        peaks = [
            peak for i, peak in result.peak_abs_position.items() if result.roles[i] == "MarketMaker"
        ]
        assert peaks
        assert max(peaks) < 500


@pytest.mark.parametrize("seed", _SEEDS)
def test_a_market_maker_never_exceeds_limit_plus_one_round_of_quotes(seed: int) -> None:
    """The formal guarantee: it stops quoting a side at the limit, so it can overshoot by at most
    what one set of quotes on that side could fill (levels * quote_qty)."""
    limit, levels, quote_qty = 20, 2, 3
    maker = MarketMaker(
        0, NS_PER_S, half_spread_ticks=1, levels=levels, quote_qty=quote_qty, max_position=limit
    )
    noise = [
        NoiseTrader(i, 5 * NS_PER_S, p_market=0.3, market_qty_scale=5.0) for i in range(1, 101)
    ]
    config = MarketConfig(seed=seed, duration_ns=_SECONDS * NS_PER_S, num_market_makers=0)
    peak = Market(config, agents=[maker, *noise]).run().peak_abs_position[0]
    assert peak <= limit + levels * quote_qty
    assert peak >= 15, "the limit was never approached, so this test proved nothing"
