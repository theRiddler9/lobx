"""Run jobs by calling the evaluation harness; this is the only place the service touches it."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from pathlib import Path

from lobx.eval.compare import paired_comparison, verdict
from lobx.eval.harness import (
    EngineFactory,
    EvalConfig,
    StrategyFactory,
    default_engine,
    run_detailed,
    run_many,
)
from lobx.eval.report import summarize_results
from lobx.eval.runner import WATCHING, build_factory
from lobx.eval.seeds import TEST_SEEDS
from lobx.ml.model import Classifier
from lobx.ml.store import load_model
from lobx.service.jobs import ComparisonJob, RunJob
from lobx.service.schemas import FillOut, PairRow, RunResult, StrategyRow
from lobx.sim.agents.recorder import SnapshotRecorder
from lobx.sim.clock import NS_PER_S
from lobx.types import Side

logger = logging.getLogger(__name__)

# The book is recorded once per simulated second, ten levels deep: enough to draw a live book.
SNAPSHOT_INTERVAL_NS = NS_PER_S
SNAPSHOT_DEPTH = 10


class ExecutionService:
    """Builds strategies (once each) and runs jobs on them.

    VWAP learns its volume profile from 100 simulated markets, which takes a while; adaptive
    needs the trained model. Both are built the first time they are asked for and then reused.
    """

    def __init__(self, config: EvalConfig, engine: EngineFactory | None, model_path: Path) -> None:
        """Remember the settings; nothing is built until a job needs it."""
        self.config = config
        self.engine = engine or default_engine()
        self.model_path = model_path
        self._lock = threading.Lock()  # one thread builds a factory or loads the model at a time
        self._model: Classifier | None = None
        self._factories: dict[str, StrategyFactory] = {}

    def model_available(self) -> bool:
        """Return True if a trained model file exists (the service never trains one itself)."""
        return self._model is not None or self.model_path.exists()

    def _factory(self, name: str) -> StrategyFactory:
        with self._lock:
            if name not in self._factories:
                if name == "adaptive" and self._model is None:
                    self._model = load_model(self.model_path)
                self._factories[name] = build_factory(name, self.config, self.engine, self._model)
            return self._factories[name]

    def run(self, job: RunJob) -> None:
        """Execute a run job, recording its book, and store the outcome on the job."""
        request = job.request
        try:
            job.status = "running"
            config = replace(
                self.config,
                parent_qty=request.parent_qty,
                side=Side.SELL if request.side == "sell" else Side.BUY,
            )

            def make_recorder(agent_id: int) -> SnapshotRecorder:
                job.recorder = SnapshotRecorder(agent_id, SNAPSHOT_INTERVAL_NS, SNAPSHOT_DEPTH)
                return job.recorder

            detail = run_detailed(
                self._factory(request.strategy),
                request.seed,
                config,
                self.engine,
                watching=request.strategy in WATCHING,
                observer=make_recorder,
            )
            outcome = detail.result
            job.result = RunResult(
                executed=outcome.executed,
                shortfall_bps=outcome.shortfall_bps,
                arrival_mid=outcome.arrival_mid2 / 2,
                fills=[FillOut(ts_ns=f.ts, price=f.price, qty=f.qty) for f in detail.fills],
            )
            job.status = "done"
        except Exception as error:  # a worker thread must report its failure, not die silently
            logger.exception("run %s failed", job.id)
            job.error = f"{type(error).__name__}: {error}"
            job.status = "failed"

    def compare(self, job: ComparisonJob) -> None:
        """Execute a comparison job on the first ``seeds`` test seeds and store its tables."""
        request = job.request
        try:
            job.status = "running"
            seeds = range(TEST_SEEDS.start, TEST_SEEDS.start + request.seeds)
            results = {
                name: run_many(
                    self._factory(name), seeds, self.config, self.engine, name in WATCHING
                )
                for name in dict.fromkeys(request.strategies)  # drop repeats, keep the order
            }
            rows = []
            for name, runs in results.items():
                summary, incomplete = summarize_results(runs)
                rows.append(
                    StrategyRow(
                        name=name,
                        seeds=summary.n,
                        incomplete=incomplete,
                        mean_bps=summary.mean,
                        std_bps=summary.std,
                        ci_low_bps=summary.ci_low,
                        ci_high_bps=summary.ci_high,
                    )
                )
            pairs = []
            names = list(results)
            for i, b in enumerate(names):
                for a in names[i + 1 :]:
                    c = paired_comparison(a, results[a], b, results[b])
                    d = c.difference
                    pairs.append(
                        PairRow(
                            a=a,
                            b=b,
                            seeds_used=c.used,
                            mean_bps=d.mean,
                            ci_low_bps=d.ci_low,
                            ci_high_bps=d.ci_high,
                            a_cheaper_fraction=c.a_cheaper / c.used,
                            verdict=verdict(c),
                        )
                    )
            job.rows, job.pairs = rows, pairs
            job.status = "done"
        except Exception as error:  # a worker thread must report its failure, not die silently
            logger.exception("comparison %s failed", job.id)
            job.error = f"{type(error).__name__}: {error}"
            job.status = "failed"
