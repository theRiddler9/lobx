"""The FastAPI application.

``create_app`` builds an app from explicit settings and keeps all of its state (job store, worker
pool) in local variables captured by the route functions, so two apps never share anything and
tests can build one with a short configuration. Serve it with
``uvicorn --factory lobx.service.app:create_app`` (or ``python -m lobx.cli serve``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.responses import HTMLResponse

from lobx.analytics.bars import aggregate_bars
from lobx.analytics.heatmap import build_depth_heatmap
from lobx.analytics.microstructure import (
    compute_stoikov_microprice,
    compute_vpin,
    decompose_executions,
)
from lobx.analytics.tradingview import render_tradingview_html
from lobx.engine.native import NativeOrderBook
from lobx.eval.harness import EngineFactory, EvalConfig
from lobx.eval.runner import DEFAULT_MODEL_PATH
from lobx.service.execution import ExecutionService
from lobx.service.jobs import JobStore
from lobx.service.schemas import (
    CandleBarOut,
    ComparisonInfo,
    ComparisonRequest,
    ExecutionDecompositionOut,
    HealthInfo,
    HeatmapLevelOut,
    HeatmapSnapshotOut,
    MicropricePointOut,
    MicrostructureOut,
    RunInfo,
    RunRequest,
    VpinPointOut,
)
from lobx.service.stream import stream_run
from lobx.sim.clock import NS_PER_S


def create_app(
    config: EvalConfig | None = None,
    engine: EngineFactory | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    max_workers: int = 2,
) -> FastAPI:
    """Build the app.

    ``config`` is the scenario every run uses (default: the Phase 4 one, 3,000 shares over five
    minutes); a request may change only the quantity and side. ``engine`` defaults to the C++ engine
    when it is built. At most ``max_workers`` simulations run at once; the rest wait in a queue.
    """
    execution = ExecutionService(config or EvalConfig(), engine, model_path)
    store = JobStore()
    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lobx-job")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # On shutdown, drop queued jobs; a job already running finishes in its own thread.
        pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="LOBX", description="Simulated limit-order-book exchange", lifespan=lifespan
    )

    @app.get("/health")
    def health() -> HealthInfo:
        """Report that the service is up, which engine it uses, and whether a model is ready."""
        return HealthInfo(
            engine="native" if execution.engine is NativeOrderBook else "python",
            model_available=execution.model_available(),
        )

    @app.post("/runs", status_code=202)
    def start_run(request: RunRequest) -> RunInfo:
        """Start a run and return at once; poll ``GET /runs/{id}`` for the result.

        202 Accepted is the honest answer: the simulation has been queued, not finished.
        """
        if request.strategy == "adaptive" and not execution.model_available():
            # Training takes minutes, so the service refuses instead of silently training.
            raise HTTPException(
                status_code=409,
                detail=f"no trained model at {model_path}; run `python -m lobx.cli train` first",
            )
        job = store.add_run(request)
        accepted = job.info()
        pool.submit(execution.run, job)
        return accepted

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> RunInfo:
        """Return a run's status and, once it is done, its cost and fills."""
        job = store.run(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        return job.info()

    @app.get("/runs/{run_id}/bars")
    def get_bars(
        run_id: str, duration_s: int = Query(default=5, ge=1, le=60)
    ) -> list[CandleBarOut]:
        """Aggregate trade fills into OHLCV candlestick bars with cumulative volume delta."""
        job = store.run(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"run {run_id} is still {job.status}")
        fills = job.market_fills or job.strategy_fills
        opening = int(round(job.result.arrival_mid)) if job.result else 10_000
        bars = aggregate_bars(fills, bar_duration_ns=duration_s * NS_PER_S, opening_price=opening)
        return [
            CandleBarOut(
                time=b.time,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                buy_volume=b.buy_volume,
                sell_volume=b.sell_volume,
                volume_delta=b.volume_delta,
                cvd=b.cvd,
                trade_count=b.trade_count,
                vwap=b.vwap,
            )
            for b in bars
        ]

    @app.get("/runs/{run_id}/microstructure")
    def get_microstructure(run_id: str) -> MicrostructureOut:
        """Calculate Stoikov analytical micro-price, VPIN toxicity, and execution decomposition."""
        job = store.run(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"run {run_id} is still {job.status}")
        snapshots = job.recorder.snapshots if job.recorder else []
        fills = job.market_fills or job.strategy_fills
        microprices = compute_stoikov_microprice(snapshots)
        vpin_points = compute_vpin(fills)
        decomp_points = decompose_executions(job.strategy_fills, snapshots)
        mean_vpin = sum(p.vpin for p in vpin_points) / len(vpin_points) if vpin_points else 0.0
        mean_eff = (
            sum(d.effective_spread_bps for d in decomp_points) / len(decomp_points)
            if decomp_points
            else 0.0
        )
        mean_realized = (
            sum(d.realized_spread_bps for d in decomp_points) / len(decomp_points)
            if decomp_points
            else 0.0
        )
        mean_impact = (
            sum(d.price_impact_bps for d in decomp_points) / len(decomp_points)
            if decomp_points
            else 0.0
        )
        return MicrostructureOut(
            vpin_series=[
                VpinPointOut(ts=p.ts, vpin=p.vpin, bucket_index=p.bucket_index) for p in vpin_points
            ],
            mean_vpin=mean_vpin,
            spread_decomposition=[
                ExecutionDecompositionOut(
                    fill_ts=d.fill_ts,
                    fill_price=d.fill_price,
                    qty=d.qty,
                    arrival_mid=d.arrival_mid,
                    future_mid=d.future_mid,
                    effective_spread_bps=d.effective_spread_bps,
                    realized_spread_bps=d.realized_spread_bps,
                    price_impact_bps=d.price_impact_bps,
                )
                for d in decomp_points
            ],
            mean_effective_spread_bps=mean_eff,
            mean_realized_spread_bps=mean_realized,
            mean_price_impact_bps=mean_impact,
            microprices=[MicropricePointOut(time=pt[0], microprice=pt[1]) for pt in microprices],
        )

    @app.get("/runs/{run_id}/chart", response_class=HTMLResponse)
    def get_chart(run_id: str, duration_s: int = Query(default=5, ge=1, le=60)) -> HTMLResponse:
        """Render responsive TradingView Lightweight Chart with execution markers."""
        job = store.run(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"run {run_id} is still {job.status}")
        fills = job.market_fills or job.strategy_fills
        opening = int(round(job.result.arrival_mid)) if job.result else 10_000
        bars = aggregate_bars(fills, bar_duration_ns=duration_s * NS_PER_S, opening_price=opening)
        snapshots = job.recorder.snapshots if job.recorder else []
        microprices = compute_stoikov_microprice(snapshots)
        title = f"LOBX Execution & Market Study: Run {run_id} ({job.request.strategy.upper()})"
        html = render_tradingview_html(bars, job.strategy_fills, microprices, title=title)
        return HTMLResponse(content=html)

    @app.get("/runs/{run_id}/heatmap")
    def get_heatmap(
        run_id: str,
        sample_stride: int = Query(default=1, ge=1, le=100),
        max_depth: int = Query(default=5, ge=1, le=10),
    ) -> list[HeatmapSnapshotOut]:
        """Generate Bookmap-style depth-of-market liquidity heatmap."""
        job = store.run(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"run {run_id} is still {job.status}")
        snapshots = job.recorder.snapshots if job.recorder else []
        heatmap = build_depth_heatmap(
            snapshots, sample_stride=sample_stride, max_depth_per_side=max_depth
        )
        return [
            HeatmapSnapshotOut(
                time=h.time,
                ts_ns=h.ts_ns,
                best_bid=h.best_bid,
                best_ask=h.best_ask,
                levels=[
                    HeatmapLevelOut(
                        price=lvl["price"],
                        qty=lvl["qty"],
                        side=lvl["side"],
                    )
                    for lvl in h.levels
                ],
            )
            for h in heatmap
        ]

    @app.websocket("/runs/{run_id}/stream")
    async def stream(
        websocket: WebSocket, run_id: str, delay_ms: int = Query(default=50, ge=0, le=1_000)
    ) -> None:
        """Stream the run's book, one snapshot every ``delay_ms`` milliseconds, then its outcome."""
        job = store.run(run_id)
        if job is None:
            await websocket.close(code=4404)  # no such run; closing before accept rejects the call
            return
        await stream_run(websocket, job, delay_ms / 1_000)

    @app.post("/comparisons", status_code=202)
    def start_comparison(request: ComparisonRequest) -> ComparisonInfo:
        """Compare strategies on the first ``seeds`` held-out test seeds (queued like a run).

        The same code as ``python -m lobx.cli eval`` computes the tables, so the numbers agree
        with the CLI for the same strategies and seeds.
        """
        if "adaptive" in request.strategies and not execution.model_available():
            raise HTTPException(
                status_code=409,
                detail=f"no trained model at {model_path}; run `python -m lobx.cli train` first",
            )
        job = store.add_comparison(request)
        accepted = job.info()
        pool.submit(execution.compare, job)
        return accepted

    @app.get("/comparisons/{comparison_id}")
    def get_comparison(comparison_id: str) -> ComparisonInfo:
        """Return a comparison's status and, once done, its per-strategy and paired tables."""
        job = store.comparison(comparison_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no comparison {comparison_id!r}")
        return job.info()

    return app
