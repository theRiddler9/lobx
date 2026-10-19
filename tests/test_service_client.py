import time

import pytest

pytest.importorskip("websockets")

from lobx.service.client import ApiClient, ApiError, depth_rows, spread_of  # noqa: E402


@pytest.fixture
def api(server_url: str) -> ApiClient:
    return ApiClient(server_url + "/")


def test_depth_rows_put_bids_and_asks_on_one_price_axis() -> None:
    message = {"bids": [[100, 5], [99, 7]], "asks": [[102, 3], [103, 4]]}
    assert depth_rows(message) == [
        {"price": 99, "bids": 7, "asks": None},
        {"price": 100, "bids": 5, "asks": None},
        {"price": 102, "bids": None, "asks": 3},
        {"price": 103, "bids": None, "asks": 4},
    ]
    assert spread_of(message) == 2
    assert spread_of({"bids": [], "asks": [[102, 3]]}) is None


def test_the_client_runs_a_whole_run_over_real_http_and_websocket(api: ApiClient) -> None:
    assert api.health()["status"] == "ok"
    run = api.start_run({"strategy": "twap", "seed": 5, "parent_qty": 300})
    messages = list(api.stream(run["id"], delay_ms=0))
    assert [m["type"] for m in messages] == ["snapshot"] * 80 + ["done"]
    done = messages[-1]["run"]
    assert done["status"] == "done"
    assert api.get_run(run["id"])["result"] == done["result"]


def test_the_client_runs_a_comparison(api: ApiClient) -> None:
    started = api.start_comparison({"strategies": ["twap", "passive"], "seeds": 3})
    deadline = time.monotonic() + 120
    info = api.get_comparison(started["id"])
    while info["status"] not in ("done", "failed"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
        info = api.get_comparison(started["id"])
    assert info["status"] == "done" and len(info["rows"]) == 2


def test_api_errors_carry_the_servers_message(api: ApiClient) -> None:
    with pytest.raises(ApiError, match="404.*run-77"):
        api.get_run("run-77")
    with pytest.raises(ApiError, match="409.*train"):
        api.start_run({"strategy": "adaptive"})


def test_an_unreachable_server_is_an_api_error() -> None:
    with pytest.raises(ApiError, match="cannot reach"):
        ApiClient("http://127.0.0.1:1").health()
