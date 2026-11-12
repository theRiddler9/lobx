"""TradingView Lightweight Charts HTML/JS generator for LOBX execution analysis.

Generates an interactive, responsive multi-pane chart featuring:
1. Candlestick OHLCV price series.
2. Volume Delta and Cumulative Volume Delta (CVD) series.
3. Execution fill markers (Buy/Sell arrows showing child order fills).
4. Stoikov Micro-Price overlay line.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from lobx.analytics.bars import CandleBar
from lobx.types import Fill, Side


def render_tradingview_html(
    bars: Sequence[CandleBar],
    fills: Sequence[Fill] | None = None,
    microprices: Sequence[tuple[int, float]] | None = None,
    title: str = "LOBX Interactive Order Flow & Execution Study",
    height: int = 600,
) -> str:
    """Generate self-contained HTML page embedding TradingView Lightweight Charts.

    Args:
        bars: CandleBar sequence generated from trade events.
        fills: Optional sequence of parent strategy fills to display as markers.
        microprices: Optional Stoikov micro-price points (ts_ns, micro_px).
        title: Title banner displayed above the chart.
        height: Height of the chart container in pixels.

    Returns:
        HTML string ready to render in browser or Streamlit component.
    """
    candle_data: list[dict[str, Any]] = [b.to_tradingview(tick_scale=0.01) for b in bars]

    # Cumulative Volume Delta (CVD) line
    cvd_data = [{"time": b.time, "value": b.cvd} for b in bars]

    # Execution markers for strategy fills
    markers: list[dict[str, Any]] = []
    if fills and bars:
        base_epoch = bars[0].time - int(bars[0].start_ns // 1_000_000_000)
        for f in fills:
            fill_time = base_epoch + int(f.ts // 1_000_000_000)
            is_buy = f.taker_side is Side.BUY
            markers.append(
                {
                    "time": fill_time,
                    "position": "belowBar" if is_buy else "aboveBar",
                    "color": "#26a69a" if is_buy else "#ef5350",
                    "shape": "arrowUp" if is_buy else "arrowDown",
                    "text": f"{'BUY' if is_buy else 'SELL'} {f.qty}@{f.price / 100.0:.2f}",
                }
            )

    # Sort markers chronologically (required by Lightweight Charts)
    markers.sort(key=lambda m: m["time"])

    # Stoikov micro-price line
    micro_data: list[dict[str, Any]] = []
    if microprices and bars:
        base_epoch = bars[0].time - int(bars[0].start_ns // 1_000_000_000)
        # Sample microprices to avoid browser rendering slowdown
        stride = max(1, len(microprices) // 500)
        for idx in range(0, len(microprices), stride):
            ts_ns, px = microprices[idx]
            micro_data.append(
                {
                    "time": base_epoch + int(ts_ns // 1_000_000_000),
                    "value": px * 0.01,
                }
            )
        micro_data.sort(key=lambda d: d["time"])
        # Deduplicate consecutive timestamps
        dedup_micro: list[dict[str, Any]] = []
        for pt in micro_data:
            if not dedup_micro or dedup_micro[-1]["time"] != pt["time"]:
                dedup_micro.append(pt)
        micro_data = dedup_micro

    candle_json = json.dumps(candle_data)
    cvd_json = json.dumps(cvd_data)
    markers_json = json.dumps(markers)
    micro_json = json.dumps(micro_data)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 10px;
            background-color: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding: 0 4px;
        }}
        .title {{
            font-size: 16px;
            font-weight: 600;
            color: #e0e3eb;
        }}
        .badges {{
            font-size: 12px;
            display: flex;
            gap: 12px;
        }}
        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            background: #1e222d;
            border: 1px solid #2a2e39;
        }}
        .badge.buy {{ color: #26a69a; }}
        .badge.sell {{ color: #ef5350; }}
        .badge.micro {{ color: #2962ff; }}
        #chart-container {{
            width: 100%;
            height: {height}px;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #2a2e39;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">{title}</div>
        <div class="badges">
            <span class="badge buy">● Buy Executions</span>
            <span class="badge sell">● Sell Executions</span>
            <span class="badge micro">― Stoikov Micro-Price</span>
        </div>
    </div>
    <div id="chart-container"></div>

    <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart-container'), {{
            layout: {{
                background: {{ type: 'solid', color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            rightPriceScale: {{
                borderColor: '#2a2e39',
            }},
            timeScale: {{
                borderColor: '#2a2e39',
                timeVisible: true,
                secondsVisible: true,
            }},
        }});

        // 1. Candlestick Price Series
        const candlestickSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        }});
        const candleData = {candle_json};
        candlestickSeries.setData(candleData);

        // 2. Child Order Execution Markers
        const markers = {markers_json};
        if (markers.length > 0) {{
            candlestickSeries.setMarkers(markers);
        }}

        // 3. Stoikov Micro-Price Line
        const microData = {micro_json};
        if (microData.length > 0) {{
            const microSeries = chart.addLineSeries({{
                color: '#2962ff',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                priceLineVisible: false,
                lastValueVisible: false,
            }});
            microSeries.setData(microData);
        }}

        // 4. Cumulative Volume Delta (CVD) Overlay
        const cvdSeries = chart.addLineSeries({{
            color: '#ff9800',
            lineWidth: 1,
            priceScaleId: 'cvd',
            priceLineVisible: false,
        }});
        chart.priceScale('cvd').applyOptions({{
            scaleMargins: {{ top: 0.8, bottom: 0.02 }},
        }});
        cvdSeries.setData({cvd_json});

        chart.timeScale().fitContent();

        // Responsive resize
        window.addEventListener('resize', () => {{
            chart.applyOptions({{ width: document.getElementById('chart-container').clientWidth }});
        }});
    </script>
</body>
</html>
"""
