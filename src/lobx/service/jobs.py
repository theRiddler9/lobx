"""Jobs: runs and comparisons that were requested, are being computed, or are finished.

A simulation takes seconds, so the API answers a request at once with an id and computes in a
worker thread; the client polls for the result (or watches the WebSocket). The job objects are the
shared state between the worker thread (which writes them) and the request handlers (which read
them). That is safe here because each field is assigned as a whole (assignment is atomic in
CPython), the worker only moves a job forward (queued -> running -> done or failed), and
``snapshots`` is a list that only ever grows.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from lobx.service.schemas import (
    ComparisonInfo,
    ComparisonRequest,
    JobStatus,
    PairRow,
    RunInfo,
    RunRequest,
    RunResult,
    StrategyRow,
)
from lobx.sim.agents.recorder import SnapshotRecorder


@dataclass(slots=True)
class RunJob:
    """One run's request, progress and outcome."""

    id: str
    request: RunRequest
    status: JobStatus = "queued"
    result: RunResult | None = None
    error: str | None = None
    # Set by the worker when the market is built; its ``snapshots`` list grows while the run goes.
    recorder: SnapshotRecorder | None = None

    def info(self) -> RunInfo:
        """Return the API view of this job."""
        recorded = self.recorder.snapshots if self.recorder is not None else []
        return RunInfo(
            id=self.id,
            status=self.status,
            request=self.request,
            snapshots=len(recorded),
            result=self.result,
            error=self.error,
        )


@dataclass(slots=True)
class ComparisonJob:
    """One comparison's request, progress and tables."""

    id: str
    request: ComparisonRequest
    status: JobStatus = "queued"
    rows: list[StrategyRow] = field(default_factory=list)
    pairs: list[PairRow] = field(default_factory=list)
    error: str | None = None

    def info(self) -> ComparisonInfo:
        """Return the API view of this job."""
        return ComparisonInfo(
            id=self.id,
            status=self.status,
            request=self.request,
            rows=self.rows,
            pairs=self.pairs,
            error=self.error,
        )


class JobStore:
    """Hands out ids and remembers jobs, in memory, for the life of the process."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._lock = threading.Lock()  # protects the counter and both dicts
        self._count = 0
        self._runs: dict[str, RunJob] = {}
        self._comparisons: dict[str, ComparisonJob] = {}

    def add_run(self, request: RunRequest) -> RunJob:
        """Register a new run (ids are ``run-1``, ``run-2``, ...)."""
        with self._lock:
            self._count += 1
            job = RunJob(f"run-{self._count}", request)
            self._runs[job.id] = job
            return job

    def add_comparison(self, request: ComparisonRequest) -> ComparisonJob:
        """Register a new comparison (ids are ``cmp-1``, ``cmp-2``, ...)."""
        with self._lock:
            self._count += 1
            job = ComparisonJob(f"cmp-{self._count}", request)
            self._comparisons[job.id] = job
            return job

    def run(self, run_id: str) -> RunJob | None:
        """Return the run with this id, or ``None``."""
        with self._lock:
            return self._runs.get(run_id)

    def comparison(self, comparison_id: str) -> ComparisonJob | None:
        """Return the comparison with this id, or ``None``."""
        with self._lock:
            return self._comparisons.get(comparison_id)
