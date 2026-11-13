"""The JSON shapes of the HTTP API, as pydantic models.

pydantic validates every request (a bad field gives a 422 response with a clear message before any
simulation starts) and turns every response into JSON. Prices are integer ticks and times are
integer nanoseconds, exactly as in the rest of LOBX; only derived statistics are floats.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, Field

from lobx.eval.runner import STRATEGY_NAMES
from lobx.eval.seeds import TEST_SEEDS

# The same four names the CLI accepts; the literal type makes FastAPI reject anything else.
StrategyName = Literal["twap", "vwap", "passive", "adaptive", "almgren-chriss"]
JobStatus = Literal["queued", "running", "done", "failed"]

# Guard: the names above must stay in step with the evaluation runner.
assert set(get_args(StrategyName)) == set(STRATEGY_NAMES)

MAX_COMPARISON_SEEDS = 200


class RunRequest(BaseModel):
    """Start one simulated run: sell (or buy) a parent order with one strategy."""

    strategy: StrategyName
    seed: int = Field(default=TEST_SEEDS.start, ge=0, description="Seeds fix the whole market.")
    parent_qty: int = Field(default=3_000, ge=1, le=20_000, description="Shares to execute.")
    side: Literal["sell", "buy"] = "sell"


class FillOut(BaseModel):
    """One trade the strategy took part in."""

    ts_ns: int
    price: int
    qty: int


class RunResult(BaseModel):
    """What a finished run cost."""

    executed: int
    # None when the parent order did not complete, so no shortfall can be measured.
    shortfall_bps: float | None
    arrival_mid: float  # the mid-price when the order arrived, in ticks
    fills: list[FillOut]


class RunInfo(BaseModel):
    """The state of one run: what was asked, how far it got, and the result once done."""

    id: str
    status: JobStatus
    request: RunRequest
    snapshots: int  # how many book snapshots have been recorded so far
    result: RunResult | None = None
    error: str | None = None


class ComparisonRequest(BaseModel):
    """Compare strategies on the same held-out test seeds."""

    strategies: list[StrategyName] = Field(min_length=1, max_length=5)
    seeds: int = Field(default=10, ge=2, le=MAX_COMPARISON_SEEDS)


class StrategyRow(BaseModel):
    """Mean shortfall of one strategy over its completed seeds, with a 95% CI for the mean."""

    name: str
    seeds: int
    incomplete: int
    mean_bps: float
    std_bps: float
    ci_low_bps: float
    ci_high_bps: float


class PairRow(BaseModel):
    """``a`` minus ``b`` per seed (negative: a is cheaper), over seeds where both completed."""

    a: str
    b: str
    seeds_used: int
    mean_bps: float
    ci_low_bps: float
    ci_high_bps: float
    a_cheaper_fraction: float
    verdict: str


class ComparisonInfo(BaseModel):
    """The state of one comparison, with its tables once done."""

    id: str
    status: JobStatus
    request: ComparisonRequest
    rows: list[StrategyRow] = []
    pairs: list[PairRow] = []
    error: str | None = None


class HealthInfo(BaseModel):
    """What the service is running on."""

    status: Literal["ok"] = "ok"
    engine: Literal["native", "python"]
    model_available: bool


class CandleBarOut(BaseModel):
    """Aggregated OHLCV and volume delta candle bar."""

    time: int
    open: int
    high: int
    low: int
    close: int
    volume: int
    buy_volume: int
    sell_volume: int
    volume_delta: int
    cvd: int
    trade_count: int
    vwap: float


class ExecutionDecompositionOut(BaseModel):
    """Microstructural decomposition of an execution fill."""

    fill_ts: int
    fill_price: int
    qty: int
    arrival_mid: float
    future_mid: float
    effective_spread_bps: float
    realized_spread_bps: float
    price_impact_bps: float


class MicropricePointOut(BaseModel):
    """Stoikov analytical micro-price point."""

    time: int
    microprice: float


class VpinPointOut(BaseModel):
    """VPIN toxicity calculation point."""

    ts: int
    vpin: float
    bucket_index: int


class MicrostructureOut(BaseModel):
    """Microstructure analytics: flow toxicity, spread decomposition, and microprice series."""

    vpin_series: list[VpinPointOut]
    mean_vpin: float
    spread_decomposition: list[ExecutionDecompositionOut]
    mean_effective_spread_bps: float
    mean_realized_spread_bps: float
    mean_price_impact_bps: float
    microprices: list[MicropricePointOut]


class HeatmapLevelOut(BaseModel):
    """Depth level item in heatmap."""

    price: int
    qty: int
    side: str


class HeatmapSnapshotOut(BaseModel):
    """Depth-of-market liquidity snapshot."""

    time: int
    ts_ns: int
    best_bid: int | None
    best_ask: int | None
    levels: list[HeatmapLevelOut]
