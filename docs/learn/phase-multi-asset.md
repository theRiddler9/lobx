# Learn: Multi-Asset Matching & Cross-Asset Statistical Arbitrage

## 1. Overview and Core Microstructure Principles

In real financial markets, trading does not take place in isolated single-symbol silos. Instruments are deeply connected through economic fundamentals, index arbitrage, and multi-venue liquidity.

This phase implements:
1. **Multi-Book Matching Engine**: Pure C++17 `MultiBookEngine` with pybind11 bindings and Python reference router managing concurrent independent order books keyed by symbol.
2. **Lead-Lag Price Discovery Dynamics**: Modeling information transmission where liquid benchmark symbols (e.g. `SPY`, `LEAD`) reflect order flow imbalance before correlated follower symbols (e.g. `QQQ`, `LAG`).
3. **Cross-Arbitrage Execution**: High-speed agents sniping stale quotes on lagging books when order book imbalance signals impending drift.
4. **Pairs Execution Strategy**: Two-legged institutional parent order execution with dynamic leg risk throttling to guarantee portfolio delta neutrality.

---

## 2. Mathematical Formulations

### 2.1 Cross-Asset Order Flow Imbalance (OFI)
Order book imbalance measures the buying vs selling pressure at the top of the book:
$$I_t = \frac{Q_{b, t} - Q_{a, t}}{Q_{b, t} + Q_{a, t}} \in [-1, 1]$$

Under cross-impact microstructure models (e.g. Cont & Kukanov, Bouchaud et al.), the expected short-horizon price drift of correlated follower asset $B$ at lag $\tau$ is conditioned on the leader's imbalance:
$$\mathbb{E}[P_{B, t+\tau} - P_{B, t}] = \beta \cdot I_{A, t}$$

### 2.2 Pairs Execution Shortfall & Leg Risk
When executing a pairs trade (Long $Q_A$ of asset $A$, Short $Q_B$ of asset $B$), execution shortfall in basis points is:
$$\text{Shortfall}_{\text{pairs}} = \frac{1}{2} \left( \frac{\bar{P}_A - P_{A, 0}}{P_{A, 0}} + \frac{P_{B, 0} - \bar{P}_B}{P_{B, 0}} \right) \times 10{,}000$$

The unhedged inventory risk at time $t$ is:
$$\Delta_{\text{unhedged}}(t) = |Q_{A, \text{filled}}(t) - Q_{B, \text{filled}}(t)|$$

If $\Delta_{\text{unhedged}}(t) > \Delta_{\max}$, the strategy throttles the faster leg and aggressively fills the lagging leg to eliminate directional exposure.

---

## 3. Practice Exercises

### Exercise 1: Multi-Symbol Order Routing
**Prompt**: Instantiate `MultiOrderBook` with symbols `["SPY", "QQQ"]`. Submit a limit buy order of 100 shares at 5,000 ticks on `SPY`, and a limit sell order of 50 shares at 10,000 ticks on `QQQ`. Verify the best bid on `SPY` and the best ask on `QQQ`.
```python
from lobx.engine.multi_book import MultiOrderBook
from lobx.types import Order, OrderType, Side

mb = MultiOrderBook(["SPY", "QQQ"])
mb.submit("SPY", Order(1, 10, Side.BUY, OrderType.LIMIT, 5000, 100, 0))
mb.submit("QQQ", Order(2, 10, Side.SELL, OrderType.LIMIT, 10000, 50, 0))

assert mb.best_bid("SPY") == 5000
assert mb.best_ask("QQQ") == 10000
```

### Exercise 2: Detecting Cross-Asset Imbalance
**Prompt**: Write a function that inspects a `CrossMarketView` and returns `+1` (Bullish) if `LEAD` imbalance $> 0.35$, `-1` (Bearish) if `LEAD` imbalance $< -0.35$, and `0` otherwise.
```python
def classify_lead_signal(view, symbol="LEAD", threshold=0.35) -> int:
    imb = view.imbalance(symbol, depth=1)
    if imb > threshold:
        return 1
    elif imb < -threshold:
        return -1
    return 0
```

### Exercise 3: Throttling Pairs Execution
**Prompt**: Given a target of 500 shares on Leg A and 500 shares on Leg B, with a maximum allowed unhedged discrepancy of 50 shares: if Leg A has filled 300 shares and Leg B has filled 200 shares, should Leg A slice orders be emitted?
**Answer**: No. Leg A is ahead by 100 shares ($300 - 200 = 100 > \Delta_{\max} = 50$). Slicing on Leg A is temporarily paused while Leg B aggressively catches up.
