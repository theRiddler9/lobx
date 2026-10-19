"""The LOBX dashboard (Streamlit): watch a simulated book live and compare strategies.

Run it with ``streamlit run dashboard/app.py``. It adds no trading logic: every number comes from
the API (``python -m lobx.cli serve``), which calls the same code as the command line tools.
Streamlit re-runs this whole script from top to bottom on every click, which is why the work is
done in small functions called at the bottom.
"""

from __future__ import annotations

import os
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
    st.subheader("Live order book")
    live_book_tab(client)


main()

