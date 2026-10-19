from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from lobx.eval.harness import EvalConfig, run_one_seed  # noqa: E402
from lobx.eval.runner import build_factory  # noqa: E402
from lobx.service.execution import ExecutionService  # noqa: E402
from lobx.service.jobs import JobStore  # noqa: E402
from lobx.service.schemas import ComparisonRequest, RunRequest  # noqa: E402
from lobx.sim.clock import NS_PER_S  # noqa: E402

SHORT = EvalConfig(parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6)
NO_MODEL = Path("does-not-exist.json")


def service() -> ExecutionService:
    return ExecutionService(SHORT, engine=None, model_path=NO_MODEL)


def test_a_run_job_matches_running_the_harness_directly() -> None:
    job = JobStore().add_run(RunRequest(strategy="twap", seed=7, parent_qty=300))
    assert job.info().status == "queued"
    service().run(job)
    assert job.info().status == "done" and job.error is None
    assert job.result is not None
    direct = run_one_seed(build_factory("twap", SHORT, None), 7, SHORT)
    assert job.result.shortfall_bps == direct.shortfall_bps
    assert job.result.executed == 300 == sum(f.qty for f in job.result.fills)


def test_a_run_records_one_snapshot_per_simulated_second() -> None:
    job = JobStore().add_run(RunRequest(strategy="passive", seed=8, parent_qty=300))
    service().run(job)
    assert job.recorder is not None
    assert job.info().snapshots == 80  # the market ends when the window does: 20 s + 60 s


def test_the_request_overrides_quantity_and_side() -> None:
    job = JobStore().add_run(RunRequest(strategy="twap", seed=9, parent_qty=120, side="buy"))
    service().run(job)
    assert job.result is not None and job.result.executed == 120


def test_a_failing_job_is_marked_failed_with_the_reason() -> None:
    job = JobStore().add_run(RunRequest(strategy="adaptive", seed=1))  # no model file
    service().run(job)
    assert job.status == "failed" and job.result is None
    assert job.error is not None and "FileNotFoundError" in job.error


def test_a_comparison_summarises_each_strategy_and_each_pair() -> None:
    job = JobStore().add_comparison(ComparisonRequest(strategies=["twap", "passive"], seeds=4))
    service().compare(job)
    assert job.status == "done", job.error
    assert [row.name for row in job.rows] == ["twap", "passive"]
    assert all(row.seeds + row.incomplete == 4 for row in job.rows)
    assert [(p.a, p.b) for p in job.pairs] == [("passive", "twap")]
    pair = job.pairs[0]
    assert pair.ci_low_bps <= pair.mean_bps <= pair.ci_high_bps
    assert pair.verdict in {"passive cheaper", "twap cheaper", "no clear difference"}


def test_ids_are_unique_and_lookups_work() -> None:
    store = JobStore()
    run = store.add_run(RunRequest(strategy="twap"))
    cmp = store.add_comparison(ComparisonRequest(strategies=["twap"]))
    assert run.id != cmp.id
    assert store.run(run.id) is run and store.comparison(cmp.id) is cmp
    assert store.run(cmp.id) is None and store.run("nope") is None
