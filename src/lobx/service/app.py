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

from lobx.engine.native import NativeOrderBook
from lobx.eval.harness import EngineFactory, EvalConfig
from lobx.eval.runner import DEFAULT_MODEL_PATH
from lobx.service.execution import ExecutionService
from lobx.service.jobs import JobStore
from lobx.service.schemas import ComparisonInfo, ComparisonRequest, HealthInfo, RunInfo, RunRequest
from lobx.service.stream import stream_run


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
        pool.submit(execution.run, job)
        return job.info()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> RunInfo:
        """Return a run's status and, once it is done, its cost and fills."""
        job = store.run(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        return job.info()

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
        pool.submit(execution.compare, job)
        return job.info()

    @app.get("/comparisons/{comparison_id}")
    def get_comparison(comparison_id: str) -> ComparisonInfo:
        """Return a comparison's status and, once done, its per-strategy and paired tables."""
        job = store.comparison(comparison_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no comparison {comparison_id!r}")
        return job.info()

    return app
