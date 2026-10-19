import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from lobx.engine.book_py import PythonOrderBook  # noqa: E402
from lobx.eval.harness import EvalConfig  # noqa: E402
from lobx.service.app import create_app  # noqa: E402
from lobx.sim.clock import NS_PER_S  # noqa: E402

SHORT = EvalConfig(parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6)


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(SHORT, PythonOrderBook, Path("does-not-exist.json"))
    with TestClient(app) as test_client:  # "with" runs the app's startup and shutdown
        yield test_client


def wait_until_finished(
    client: TestClient, run_id: str, timeout_s: float = 60.0
) -> dict[str, object]:
    """Poll a run until it is done or failed (the simulation runs in a worker thread)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body: dict[str, object] = client.get(f"/runs/{run_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"{run_id} did not finish within {timeout_s} s")


def test_health_reports_the_engine_and_the_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "engine": "python", "model_available": False}


def test_a_submitted_run_is_accepted_then_completes(client: TestClient) -> None:
    response = client.post("/runs", json={"strategy": "twap", "seed": 5, "parent_qty": 300})
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["id"].startswith("run-") and accepted["status"] in ("queued", "running")
    done = wait_until_finished(client, accepted["id"])
    assert done["status"] == "done" and done["error"] is None
    result = done["result"]
    assert isinstance(result, dict) and result["executed"] == 300
    assert sum(fill["qty"] for fill in result["fills"]) == 300


def test_the_same_request_gives_the_same_cost(client: TestClient) -> None:
    body = {"strategy": "passive", "seed": 6, "parent_qty": 200}
    ids = [client.post("/runs", json=body).json()["id"] for _ in range(2)]
    costs = [wait_until_finished(client, i)["result"]["shortfall_bps"] for i in ids]  # type: ignore[index]
    assert costs[0] == costs[1]


@pytest.mark.parametrize(
    "bad",
    [{"strategy": "magic"}, {"strategy": "twap", "parent_qty": 0}, {"seed": 1}, {}],
)
def test_invalid_requests_are_rejected_before_any_work(
    client: TestClient, bad: dict[str, object]
) -> None:
    assert client.post("/runs", json=bad).status_code == 422
    assert client.get("/runs/run-1").status_code == 404  # nothing was queued


def test_an_unknown_run_is_a_404(client: TestClient) -> None:
    response = client.get("/runs/run-999")
    assert response.status_code == 404 and "run-999" in response.json()["detail"]


def test_adaptive_without_a_model_is_refused_with_a_hint(client: TestClient) -> None:
    response = client.post("/runs", json={"strategy": "adaptive"})
    assert response.status_code == 409 and "train" in response.json()["detail"]


def wait_for_comparison(client: TestClient, comparison_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        body: dict[str, object] = client.get(f"/comparisons/{comparison_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("comparison did not finish")


def test_a_comparison_returns_strategy_rows_and_paired_rows(client: TestClient) -> None:
    response = client.post("/comparisons", json={"strategies": ["twap", "passive"], "seeds": 3})
    assert response.status_code == 202
    done = wait_for_comparison(client, response.json()["id"])
    assert done["status"] == "done", done["error"]
    rows, pairs = done["rows"], done["pairs"]
    assert isinstance(rows, list) and [r["name"] for r in rows] == ["twap", "passive"]
    assert isinstance(pairs, list) and (pairs[0]["a"], pairs[0]["b"]) == ("passive", "twap")


def test_comparison_requests_are_validated_and_unknown_ids_are_404(client: TestClient) -> None:
    assert client.post("/comparisons", json={"strategies": [], "seeds": 3}).status_code == 422
    assert client.post("/comparisons", json={"strategies": ["twap"], "seeds": 1}).status_code == 422
    assert client.post("/comparisons", json={"strategies": ["adaptive"]}).status_code == 409
    assert client.get("/comparisons/cmp-42").status_code == 404


def test_the_websocket_streams_every_snapshot_then_the_outcome(client: TestClient) -> None:
    run_id = client.post("/runs", json={"strategy": "twap", "seed": 5, "parent_qty": 300}).json()[
        "id"
    ]
    with client.websocket_connect(f"/runs/{run_id}/stream?delay_ms=0") as ws:
        messages = []
        while True:
            message = ws.receive_json()
            messages.append(message)
            if message["type"] == "done":
                break
    snapshots, done = messages[:-1], messages[-1]
    assert [m["index"] for m in snapshots] == list(range(80))  # 20 s + 60 s, one per second
    assert all(m["type"] == "snapshot" for m in snapshots)
    assert snapshots[-1]["ts_ns"] == 80 * NS_PER_S
    first = snapshots[-1]
    assert first["bids"][0][0] < first["asks"][0][0]  # the book never crosses
    assert done["run"]["status"] == "done" and done["run"]["result"]["executed"] == 300


def test_the_websocket_rejects_an_unknown_run(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as stopped,
        client.websocket_connect("/runs/run-9/stream"),
    ):
        pass
    assert stopped.value.code == 4404
