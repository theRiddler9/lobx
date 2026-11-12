# Phase Historical Replay: LOBSTER Ingestion, Queue-Position Modeling, and Historical Replay

## 1. Motivation

A fundamental limitation of traditional backtesting engines is the **fill-on-touch fallacy**: assuming that as soon as market mid-price or best bid/ask touches your limit order's price, your order is executed. 

In real electronic exchanges operating under price-time priority (FIFO):
1. When your order arrives at price $P$, it sits at the back of the queue behind existing resting liquidity ($\text{qty\_ahead}$).
2. Incoming trades at price $P$ execute first against orders ahead of yours.
3. Your order only fills if aggregate traded volume at price $P$ exceeds $\text{qty\_ahead}$.
4. If the market reverses before the queue ahead is depleted, your order experiences **adverse selection**: you missed the trade, yet the market is moving against you.

To conduct realistic quantitative research, LOBX provides **historical L2/L3 order book replay** using the institutional standard **LOBSTER** (Limit Order Book System - The Efficient Reconstructor) benchmark protocol, coupled with a strict **FIFO queue-position execution model**.

---

## 2. LOBSTER Protocol Specification

LOBSTER parses NASDAQ historical order book event streams into paired CSV files:

### Message File Format
Columns: `Time, Type, OrderID, Size, Price, Direction`

- **Time**: Decimal seconds since midnight (e.g. `34200.017459617` = 09:30:00.017459617). Converted in LOBX to exact integer nanoseconds without floating-point error.
- **Type**:
  - `1`: Submission of a new limit order
  - `2`: Cancellation (partial deletion)
  - `3`: Deletion (total cancellation)
  - `4`: Execution of a visible limit order
  - `5`: Execution of a hidden limit order
  - `6`: Cross trade
  - `7`: Trading halt status
- **OrderID**: Unique order reference identifier
- **Size**: Number of shares
- **Price**: Integer price in dollar cents $\times 100$ (e.g., $10{,}000 = \$1.0000$)
- **Direction**: `1` for Buy, `-1` for Sell

### Orderbook File Format
Depth-$K$ snapshot at every event:
`AskP1, AskQ1, BidP1, BidQ1, AskP2, AskQ2, BidP2, BidQ2, ...`

---

## 3. FIFO Queue-Position Modeling

The `QueueTracker` module (`lobx.sim.queue_position`) maintains exact queue priority for strategy child orders:

### Trade Fill Condition
Let an order of size $Q$ be placed at price $P$ when resting volume ahead is $V_{\text{ahead}}$.
- Initial state:
  $$\text{qty\_ahead} = V_{\text{ahead}}, \quad \text{qty\_remaining} = Q$$
- When a trade of volume $T$ occurs at price $P$:
  - If $T \le \text{qty\_ahead}$:
    $$\text{qty\_ahead} \leftarrow \text{qty\_ahead} - T$$
    $$\text{fill\_qty} = 0$$
  - If $T > \text{qty\_ahead}$:
    $$\text{consumed} = T - \text{qty\_ahead}$$
    $$\text{fill\_qty} = \min(\text{qty\_remaining}, \text{consumed})$$
    $$\text{qty\_ahead} = 0$$
    $$\text{qty\_remaining} \leftarrow \text{qty\_remaining} - \text{fill\_qty}$$

### Cancellation Handling
When resting liquidity cancels at price $P$ with volume $C$:
- **Conservative**: Assumes cancellations happen behind our order ($\text{qty\_ahead}$ unchanged).
- **Optimistic**: Assumes cancellations happen ahead of our order ($\text{qty\_ahead} \leftarrow \max(0, \text{qty\_ahead} - C)$).
- **Proportional**: Decrements $\text{qty\_ahead}$ in proportion to our share of the price level queue:
  $$\Delta \text{ahead} = C \times \frac{\text{qty\_ahead}}{\text{Total Depth}}$$

---

## 4. Historical Strategy Evaluation

`ReplayMarket` (`lobx.sim.replay_market`) streams historical LOBSTER messages through LOBX's C++ or Python matching engines while waking execution strategies (e.g. TWAP):
1. **Arrival Mid-Price**: Recorded at the timestamp when the parent execution begins.
2. **Strategy Action Execution**:
   - Market orders execute immediately against the reconstructed book.
   - Limit orders query book depth and register with `QueueTracker`.
3. **Implementation Shortfall**: Computed in basis points upon order completion:
   $$\text{IS}_{\text{bps}} = \pm \frac{P_{\text{avg}} - P_{\text{arrival}}}{P_{\text{arrival}}} \times 10{,}000$$

### CLI Usage
```bash
python -m lobx.cli replay-lobster --message data/AAPL_message.csv --strategy twap --shares 300 --engine native
```

---

## 5. Practice Exercises

1. **Adverse Selection in Historical Fill Rates**: Run a passive quoting strategy with `conservative` vs `optimistic` cancellation models across a high-volatility historical sample. Measure how the fill rate and implementation shortfall diverge.
2. **Dynamic Queue Cancel-and-Replace**: Implement a strategy that monitors its `qty_ahead`. If the order book shifts and the order remains behind more than 2,000 shares for longer than 3 seconds, cancel the order and re-quote at the revised best bid/ask.
3. **L3 Hidden Order Detection**: Use LOBSTER Type 5 events (execution of hidden limit orders) to estimate the proportion of hidden iceberg liquidity at inside prices and study whether aggressive orders suffer higher slippage when hidden liquidity is present.
