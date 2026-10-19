"""The LOBX dashboard (Streamlit): watch a simulated book live and compare strategies.

Run it with ``streamlit run dashboard/app.py``. It adds no trading logic: every number comes from
the API (``python -m lobx.cli serve``), which calls the same code as the command line tools.
Streamlit re-runs this whole script from top to bottom on every click, which is why the work is
done in small functions called at the bottom.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import streamlit as st

from lobx.service.client import ApiClient, ApiError, depth_rows, spread_of
from lobx.sim.clock import NS_PER_S

STRATEGIES = ["twap", "vwap", "passive", "adaptive"]
BID_COLOUR, ASK_COLOUR = "#1f77b4", "#ff7f0e"  # blue and orange: distinguishable without red/green


def live_book_tab(client: ApiClient) -> None:
    """Start a run and draw its book as the snapshots arrive over the WebSocket."""
    with st.form("run"):
        left, right = st.columns(2)
        strategy = left.selectbox("Strategy", STRATEGIES)
        side = right.radio("Side", ["sell", "buy"], horizontal=True)
        seed = left.number_input("Seed", min_value=0, value=1000, step=1)
        quantity = right.number_input(
            "Shares", min_value=1, max_value=20_000, value=3_000, step=100
        )
        delay_ms = st.slider("Milliseconds between snapshots", 0, 500, 50)
        submitted = st.form_submit_button("Start run")
    if not submitted:
        st.info("Choose a strategy and press *Start run* to watch the book.")
        return

    try:
        run = client.start_run(
            {"strategy": strategy, "seed": seed, "parent_qty": quantity, "side": side}
        )
        clock, bid, ask, spread = st.columns(4)
        chart = st.empty()
        for message in client.stream(run["id"], delay_ms):
            if message["type"] == "snapshot":
                clock.metric("Simulated time", f"{message['ts_ns'] / NS_PER_S:.0f} s")
                bid.metric("Best bid (ticks)", message["bids"][0][0] if message["bids"] else "-")
                ask.metric("Best ask (ticks)", message["asks"][0][0] if message["asks"] else "-")
                gap = spread_of(message)
                spread.metric("Spread (ticks)", "-" if gap is None else gap)
                frame = pd.DataFrame(depth_rows(message)).set_index("price")
                chart.bar_chart(frame, color=[BID_COLOUR, ASK_COLOUR], y_label="shares resting")
            else:
                show_run_result(message["run"])
    except ApiError as error:
        st.error(str(error))


def show_run_result(run: dict[str, Any]) -> None:
    """Show what a finished run cost."""
    if run["status"] == "failed":
        st.error(run["error"])
        return
    result = run["result"]
    cost = result["shortfall_bps"]
    one, two = st.columns(2)
    one.metric("Implementation shortfall", "incomplete" if cost is None else f"{cost:.2f} bps")
    two.metric("Shares executed", result["executed"])
    st.caption("Positive shortfall = worse than the price when the order arrived. Simulated.")
    st.dataframe(pd.DataFrame(result["fills"]), width="stretch")


def with_interval(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a "95% CI (bps)" text column built from the interval's two ends."""
    frame = frame.copy()
    frame["95% CI (bps)"] = [
        f"[{low:.2f}, {high:.2f}]"
        for low, high in zip(frame["ci_low_bps"], frame["ci_high_bps"], strict=True)
    ]
    return frame


def comparison_tab(client: ApiClient) -> None:
    """Compare strategies on held-out test seeds, with confidence intervals and paired verdicts."""
    with st.form("comparison"):
        chosen = st.multiselect("Strategies", STRATEGIES, default=["twap", "passive"])
        seeds = st.slider("Test seeds", 2, 50, 10)
        submitted = st.form_submit_button("Run comparison")
    if not submitted:
        st.info(
            "Each strategy runs on the same seeds, so differences are paired. More seeds give "
            "narrower intervals; the published 200-seed results are in docs/results.md."
        )
        return
    if not chosen:
        st.error("Pick at least one strategy.")
        return

    try:
        started = client.start_comparison({"strategies": chosen, "seeds": seeds})
        with st.spinner(f"Simulating {len(chosen)} strategies on {seeds} seeds..."):
            info = client.get_comparison(started["id"])
            while info["status"] not in ("done", "failed"):
                time.sleep(0.5)
                info = client.get_comparison(started["id"])
    except ApiError as error:
        st.error(str(error))
        return
    if info["status"] == "failed":
        st.error(info["error"])
        return

    rows = pd.DataFrame(info["rows"]).set_index("name")
    rows = with_interval(rows)
    st.subheader("Implementation shortfall (bps, lower is better)")
    st.dataframe(
        rows[["seeds", "incomplete", "mean_bps", "std_bps", "95% CI (bps)"]], width="stretch"
    )
    st.bar_chart(rows["mean_bps"], y_label="mean shortfall (bps)")
    if info["pairs"]:
        st.subheader("Paired differences (a - b; negative means a is cheaper)")
        pairs = pd.DataFrame(info["pairs"])
        pairs = with_interval(pairs)
        st.dataframe(
            pairs[["a", "b", "seeds_used", "mean_bps", "95% CI (bps)", "verdict"]], width="stretch"
        )
    st.caption("Simulated market, simple agents. A confidence interval that contains 0 is a tie.")


def main() -> None:
    """Draw the page."""
    st.set_page_config(page_title="LOBX", layout="wide")
    st.title("LOBX: simulated limit-order-book exchange")
    st.warning(
        "Everything here comes from a **simulated** market with simple agents. It says nothing "
        "about real markets and is not a trading system."
    )
    url = st.sidebar.text_input(
        "API address", os.environ.get("LOBX_API_URL", "http://localhost:8000")
    )
    client = ApiClient(url)
    try:
        health = client.health()
    except ApiError as error:
        st.error(f"{error}. Start the API with: python -m lobx.cli serve")
        return
    st.sidebar.write(f"engine: **{health['engine']}**")
    st.sidebar.write(f"model: **{'ready' if health['model_available'] else 'not trained'}**")
    live, comparison = st.tabs(["Live book", "Strategy comparison"])
    with live:
        live_book_tab(client)
    with comparison:
        comparison_tab(client)


main()
