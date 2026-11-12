import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from lobx.engine.book_py import PythonOrderBook  # noqa: E402
from lobx.eval.harness import EvalConfig  # noqa: E402
from lobx.service.app import create_app  # noqa: E402
from lobx.sim.clock import NS_PER_S  # noqa: E402

SHORT = EvalConfig(parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6)


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(SHORT, PythonOrderBook, Path("does-not-exist.json"))
    with TestClient(app) as test_client:
        yield test_client


def wait_until_finished(
    client: TestClient, run_id: str, timeout_s: float = 60.0
) -> dict[str, object]:
    """Poll a run until it is done or failed."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body: dict[str, object] = client.get(f"/runs/{run_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"{run_id} did not finish within {timeout_s} s")


def test_service_bars_endpoint(client: TestClient) -> None:
    # 404 on missing run
    resp_404 = client.get("/runs/nonexistent/bars")
    assert resp_404.status_code == 404

    # Submit and wait
    start_resp = client.post("/runs", json={"strategy": "twap", "seed": 7, "parent_qty": 300})
    run_id = start_resp.json()["id"]
    wait_until_finished(client, run_id)

    # Bars endpoint
    bars_resp = client.get(f"/runs/{run_id}/bars?duration_s=5")
    assert bars_resp.status_code == 200
    bars = bars_resp.json()
    assert isinstance(bars, list)
    assert len(bars) > 0
    first = bars[0]
    assert "open" in first
    assert "close" in first
    assert "volume" in first
    assert "cvd" in first
    assert "trade_count" in first
    assert "vwap" in first


def test_service_microstructure_endpoint(client: TestClient) -> None:
    resp_404 = client.get("/runs/nonexistent/microstructure")
    assert resp_404.status_code == 404

    start_resp = client.post("/runs", json={"strategy": "twap", "seed": 8, "parent_qty": 300})
    run_id = start_resp.json()["id"]
    wait_until_finished(client, run_id)

    micro_resp = client.get(f"/runs/{run_id}/microstructure")
    assert micro_resp.status_code == 200
    data = micro_resp.json()
    assert "vpin_series" in data
    assert "mean_vpin" in data
    assert "spread_decomposition" in data
    assert "mean_effective_spread_bps" in data
    assert "microprices" in data
    assert len(data["microprices"]) > 0


def test_service_chart_endpoint(client: TestClient) -> None:
    resp_404 = client.get("/runs/nonexistent/chart")
    assert resp_404.status_code == 404

    start_resp = client.post("/runs", json={"strategy": "twap", "seed": 9, "parent_qty": 300})
    run_id = start_resp.json()["id"]
    wait_until_finished(client, run_id)

    chart_resp = client.get(f"/runs/{run_id}/chart?duration_s=5")
    assert chart_resp.status_code == 200
    html = chart_resp.text
    assert "text/html" in chart_resp.headers["content-type"]
    assert "LightweightCharts" in html
    assert "addCandlestickSeries" in html


def test_service_heatmap_endpoint(client: TestClient) -> None:
    resp_404 = client.get("/runs/nonexistent/heatmap")
    assert resp_404.status_code == 404

    start_resp = client.post("/runs", json={"strategy": "twap", "seed": 10, "parent_qty": 300})
    run_id = start_resp.json()["id"]
    wait_until_finished(client, run_id)

    heat_resp = client.get(f"/runs/{run_id}/heatmap?sample_stride=2&max_depth=5")
    assert heat_resp.status_code == 200
    heatmap = heat_resp.json()
    assert isinstance(heatmap, list)
    assert len(heatmap) > 0
    first = heatmap[0]
    assert "time" in first
    assert "levels" in first
    assert isinstance(first["levels"], list)
