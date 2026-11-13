# Plan: Multi-Asset Matching & Cross-Asset Statistical Arbitrage

## 1. Motivation

Institutional algorithmic trading often operates across baskets of correlated assets (e.g. index ETFs like SPY and QQQ, or equity pairs like KO and PEP). In electronic markets, price discovery does not occur simultaneously across all venues; highly liquid benchmark assets absorb order flow and fundamental news first, creating short-lived lead-lag cross-correlation patterns and temporary quote mispricings in correlated lagging assets.

## 2. Core Architecture

```
                  ┌───────────────────────────────┐
                  │       CrossMarket Engine      │
                  │  (MultiOrderBook Router: C++) │
                  └──────┬─────────────────┬──────┘
                         │                 │
              Order Flow │                 │ Order Flow
                         ▼                 ▼
                  ┌────────────┐     ┌────────────┐
                  │ Symbol: A  │     │ Symbol: B  │
                  │   (LEAD)   │     │   (LAG)    │
                  └──────┬─────┘     └─────▲──────┘
                         │                 │
                Imbalance│                 │ Cross-Arbitrage
                Detection│                 │ Orders
                         ▼                 │
                  ┌────────────────────────┴──────┐
                  │    LeadLagArbitrageAgent      │
                  │   (or PairsExecutionStrategy) │
                  └───────────────────────────────┘
```

## 3. Mathematical Foundations

### 3.1 Cross-Asset Order Flow Imbalance (OFI)
Let $I_{A, t}$ be the normalized order book depth imbalance of the leading asset $A$:
$$I_{A, t} = \frac{Q_{A, \text{bid}} - Q_{A, \text{ask}}}{Q_{A, \text{bid}} + Q_{A, \text{ask}}}$$

When $|I_{A, t}| > \theta$, cross-impact models predict price change in follower asset $B$ over latency horizon $\tau$:
$$\mathbb{E}[\Delta P_{B, t+\tau}] = \beta \cdot I_{A, t}$$

### 3.2 Dynamic Pairs Hedging & Leg Risk Control
For a pairs parent order (Long $A$, Short $B$), the net unhedged leg exposure is:
$$\Delta_{\text{unhedged}} = Q_{A, \text{filled}} - Q_{B, \text{filled}}$$

Dynamic throttling enforces $|\Delta_{\text{unhedged}}| \le \Delta_{\max}$. If Leg $A$ runs ahead of Leg $B$, execution on $A$ pauses until $B$ catches up.

## 4. Components

- `MultiOrderBook` / `NativeMultiOrderBook`: Fast symbol routing in Python and C++ (`cpp/include/lobx/multi_book.hpp`).
- `CrossMarket`: Multi-symbol discrete-event simulator with initial liquidity seeding and cross-impact dynamics.
- `LeadLagArbitrageAgent`: Microstructure agent capturing cross-asset lead-lag price discovery.
- `PairsExecutionStrategy`: Dynamic 2-legged parent order execution strategy with delta-neutral tracking.
- CLI command: `lobx multi-sim --symbols LEAD,LAG --strategy pairs`.
