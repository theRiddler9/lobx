# Phase Visualizer: TradingView Charts, Cumulative Volume Delta, and Market Microstructure

## 1. Motivation

In institutional electronic trading, an execution algorithm cannot be assessed solely from a summary statistic such as implementation shortfall in basis points. Algorithmic traders, quantitative researchers, and market operators must study **how** execution unfolds against market dynamics:

1. **Order Flow Imbalance and Aggression**: Did child orders cross the spread during buyer-initiated momentum runs or adverse order flow?
2. **Cumulative Volume Delta (CVD)**: Was net institutional volume accumulating or distributing, and did our execution trade with or against the prevailing tape?
3. **Stoikov Analytical Micro-price**: Did the resting depth imbalance predict mid-price shifts before our quotes filled or missed?
4. **Flow Toxicity (VPIN)**: Was toxic informed trading present in the book, causing adverse selection?

This module introduces institutional market study capabilities to LOBX, pairing real-time matching engine event logs with standalone **TradingView Lightweight Charts (v4.2.0)** and quantitative microstructure analytics.

---

## 2. Core Components

### 2.1 Candlestick Aggregation & Cumulative Volume Delta (`lobx.analytics.bars`)

Limit order book fills are discrete execution events $(p_i, q_i, t_i, \text{taker\_side}_i)$. The `aggregate_bars` function buckets trades into uniform time slices (e.g. 1-second, 5-second, 15-second bars) while computing:

- **OHLCV**: Open, High, Low, Close, Volume.
- **Buy / Sell Volume**: Partitioned by taker side ($\text{taker\_side} = \text{BUY}$ vs $\text{SELL}$).
- **Bar Volume Delta**: $\Delta V = V_{\text{buy}} - V_{\text{sell}}$.
- **Cumulative Volume Delta (CVD)**: Running accumulation of volume delta across the entire trading session:
  $$\text{CVD}_k = \sum_{j=1}^k (V_{\text{buy}, j} - V_{\text{sell}, j})$$
- **VWAP**: Volume-Weighted Average Price within the bar.

### 2.2 Stoikov Analytical Micro-Price (`lobx.analytics.microstructure`)

Mid-price $\frac{P_a + P_b}{2}$ assigns equal weight to best bid and best ask regardless of liquidity. Stoikov's analytical micro-price incorporates resting depth:

$$P_{\text{micro}} = \frac{Q_b P_a + Q_a P_b}{Q_b + Q_a}$$

When resting bid volume $Q_b$ significantly exceeds ask volume $Q_a$, $P_{\text{micro}}$ shifts toward $P_a$, anticipating an upward price move before quotes change.

### 2.3 Volume-Synchronized Probability of Toxicity (VPIN)

Following Easley, López de Prado, and O'Hara (2012), trade volume is divided into equal volume buckets of size $V$. For each bucket $\tau$:

$$V_\tau^{\text{buy}} = \sum_{i \in \tau, \text{taker}=\text{BUY}} q_i, \quad V_\tau^{\text{sell}} = \sum_{i \in \tau, \text{taker}=\text{SELL}} q_i$$

The VPIN metric over a rolling window of $N$ buckets is:

$$\text{VPIN} = \frac{\sum_{\tau=1}^N |V_\tau^{\text{buy}} - V_\tau^{\text{sell}}|}{N \times V}$$

When one-sided informed trading dominates, $\text{VPIN} \to 1.0$. When order flow is balanced and uninformative, $\text{VPIN} \to 0.0$.

### 2.4 Execution Spread Decomposition

Every fill of the execution strategy is decomposed into:
1. **Effective Half-Spread**: The immediate liquidity cost paid relative to mid-price at fill arrival:
   $$S_{\text{eff}} = \text{side\_sign} \times \frac{P_{\text{fill}} - P_{\text{mid}}(t)}{P_{\text{mid}}(t)} \times 10{,}000 \text{ bps}$$
2. **Price Impact (Adverse Selection)**: The subsequent price move over horizon $\Delta t$:
   $$\text{Impact} = \text{side\_sign} \times \frac{P_{\text{mid}}(t + \Delta t) - P_{\text{mid}}(t)}{P_{\text{mid}}(t)} \times 10{,}000 \text{ bps}$$
3. **Realized Half-Spread**: The residual revenue or loss after allowing for adverse selection:
   $$S_{\text{realized}} = S_{\text{eff}} - \text{Impact}$$

---

## 3. TradingView Lightweight Charts Architecture

Rather than adding heavy server-side Python plotting libraries, LOBX uses the standalone production build of **TradingView Lightweight Charts (v4.2.0)** delivered via browser script.

### Layout
- **Top Pane**: Candlestick series (green/red candles), Stoikov micro-price dotted line overlay, and execution markers (green upward arrows for parent buys, red downward arrows for parent sells).
- **Bottom Pane**: Cumulative Volume Delta (CVD) area/line series synchronized with the candlestick timeline.
- **Crosshair Synchronization**: Time and price crosshairs are synchronized between panes with millisecond precision.

---

## 4. API & Dashboard Integration

### FastAPI Endpoints
- `GET /runs/{run_id}/bars`: Returns OHLCV + CVD bar array.
- `GET /runs/{run_id}/microstructure`: Returns Stoikov micro-price points, rolling VPIN series, and execution spread decomposition.
- `GET /runs/{run_id}/chart`: Directly renders the standalone TradingView Lightweight Charts HTML.
- `GET /runs/{run_id}/heatmap`: Returns depth-of-market liquidity distribution for Bookmap-style analysis.

### Streamlit Dashboard
The dashboard (`dashboard/app.py`) features a dedicated **Market Study (TradingView)** tab allowing quantitative traders to:
1. Configure simulation parameters, market seed, order size, and strategy.
2. View real-time headline metrics: Implementation Shortfall, VPIN Flow Toxicity score, and net CVD.
3. Interact with the responsive TradingView chart, zooming and panning across simulated execution history.
4. Inspect the execution fill decomposition table (Effective Spread vs Realized Spread vs Adverse Selection).

---

## 5. Practice Exercises

1. **Multi-Horizon Adverse Selection**: Extend `decompose_executions` to calculate price impact over multiple time horizons ($\Delta t \in \{1\text{s}, 5\text{s}, 30\text{s}, 60\text{s}\}$). Observe how adverse selection decays or accumulates for passive versus aggressive strategies.
2. **Dynamic Order Routing via VPIN**: Implement an adaptive strategy rule that monitors rolling VPIN: when VPIN exceeds 0.40 (elevated toxicity), pause passive limit quotes and wait for flow toxicity to normalize before resuming execution.
3. **Custom Volume Profiles on TradingView**: Extend the TradingView HTML generator (`lobx.analytics.tradingview`) to add a horizontal Volume Profile indicator on the right axis showing total volume traded at each price level.
