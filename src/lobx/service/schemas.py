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
StrategyName = Literal["twap", "vwap", "passive", "adaptive"]
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

    strategies: list[StrategyName] = Field(min_length=1, max_length=4)
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
