from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).parent.parent / "dashboard" / "app.py")


def test_the_page_explains_how_to_start_the_api_when_it_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOBX_API_URL", "http://127.0.0.1:1")
    page = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not page.exception
    assert any("lobx.cli serve" in error.value for error in page.error)


def test_starting_a_run_draws_the_book_and_shows_the_cost(
    monkeypatch: pytest.MonkeyPatch, server_url: str
) -> None:
    monkeypatch.setenv("LOBX_API_URL", server_url)
    page = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not page.exception
    page.slider[0].set_value(0)  # no pause between snapshots
    page.button[0].click().run()
    assert not page.exception and not page.error
    labels = {metric.label for metric in page.metric}
    assert {"Simulated time", "Spread (ticks)", "Implementation shortfall"} <= labels
    assert page.dataframe  # the fills table


def test_a_comparison_shows_both_tables(monkeypatch: pytest.MonkeyPatch, server_url: str) -> None:
    monkeypatch.setenv("LOBX_API_URL", server_url)
    page = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    page.slider[1].set_value(3)  # three seeds: enough for an interval, quick to run
    page.button[1].click().run()
    assert not page.exception and not page.error
    assert len(page.dataframe) == 2  # per-strategy table and paired table
    assert "95% CI (bps)" in page.dataframe[0].value.columns
