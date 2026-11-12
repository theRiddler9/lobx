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
import streamlit.components.v1 as components

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


def market_study_tab(client: ApiClient) -> None:
    """Analyze order flow dynamics, microstructure, and view interactive TradingView charts."""
    with st.form("market_study_form"):
        left, right = st.columns(2)
        strategy = left.selectbox("Execution Strategy", STRATEGIES, key="ms_strat")
        side = right.radio("Side", ["sell", "buy"], horizontal=True, key="ms_side")
        seed = left.number_input("Market Seed", min_value=0, value=1000, step=1, key="ms_seed")
        quantity = right.number_input(
            "Order Size (shares)",
            min_value=1,
            max_value=20_000,
            value=3_000,
            step=100,
            key="ms_qty",
        )
        bar_duration = st.slider("Candle Bar Duration (seconds)", 1, 30, 5, key="ms_bar_dur")
        submitted = st.form_submit_button("Run & Analyze Market")

    if not submitted:
        st.info(
            "Select parameters and click *Run & Analyze Market* to generate institutional "
            "microstructure analytics and interactive TradingView charts."
        )
        return

    try:
        run = client.start_run(
            {"strategy": strategy, "seed": seed, "parent_qty": quantity, "side": side}
        )
        with st.spinner("Simulating market and calculating microstructure metrics..."):
            info = client.get_run(run["id"])
            while info["status"] not in ("done", "failed"):
                time.sleep(0.5)
                info = client.get_run(run["id"])

        if info["status"] == "failed":
            st.error(f"Run failed: {info['error']}")
            return

        result = info["result"]
        chart_html = client.get_chart_html(run["id"], duration_s=bar_duration)
        micro = client.get_microstructure(run["id"])
        bars = client.get_bars(run["id"], duration_s=bar_duration)

        # 1. Headline Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Implementation Shortfall",
            f"{result['shortfall_bps']:.2f} bps"
            if result["shortfall_bps"] is not None
            else "Incomplete",
        )
        vpin = float(micro.get("mean_vpin", 0.0))
        c2.metric(
            "VPIN Flow Toxicity",
            f"{vpin:.3f}",
            delta="High Toxicity" if vpin > 0.4 else "Normal Flow",
            delta_color="inverse" if vpin > 0.4 else "normal",
        )
        final_cvd = bars[-1]["cvd"] if bars else 0
        c3.metric("Cumulative Volume Delta", f"{final_cvd:+,d} shares")
        c4.metric(
            "Avg Price Impact",
            f"{float(micro.get('mean_price_impact_bps', 0.0)):.2f} bps",
        )

        # 2. Interactive TradingView Chart
        st.subheader("Interactive TradingView Execution Tape & Flow Chart")
        st.caption(
            "Candlestick OHLCV + Bottom Pane Cumulative Volume Delta (CVD) + "
            "Stoikov Micro-price (dotted blue) + Execution Markers (green/red arrows)"
        )
        components.html(chart_html, height=620)

        # 3. Microstructural Execution Decomposition
        st.subheader("Microstructural Spread Decomposition")
        decomp_rows = micro.get("spread_decomposition", [])
        if decomp_rows:
            df_decomp = pd.DataFrame(decomp_rows)
            st.dataframe(
                df_decomp[
                    [
                        "fill_ts",
                        "fill_price",
                        "qty",
                        "arrival_mid",
                        "effective_spread_bps",
                        "realized_spread_bps",
                        "price_impact_bps",
                    ]
                ],
                width="stretch",
            )
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "Mean Effective Spread",
                f"{float(micro.get('mean_effective_spread_bps', 0.0)):.2f} bps",
            )
            col_b.metric(
                "Mean Realized Spread",
                f"{float(micro.get('mean_realized_spread_bps', 0.0)):.2f} bps",
            )
            col_c.metric(
                "Mean Adverse Selection",
                f"{float(micro.get('mean_price_impact_bps', 0.0)):.2f} bps",
            )

    except ApiError as error:
        st.error(str(error))


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
    live, comparison, market_study = st.tabs(
        ["Live book", "Strategy comparison", "Market Study (TradingView)"]
    )
    with live:
        live_book_tab(client)
    with comparison:
        comparison_tab(client)
    with market_study:
        market_study_tab(client)


main()
