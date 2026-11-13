# Plan: Almgren-Chriss Optimal Execution & Market Impact Modeling

## 1. Motivation

Standard naive execution algorithms (such as TWAP) spread slices evenly across time, assuming market impact is the only friction. However, real institutional orders face a fundamental trade-off:
1. **Trading too slowly** minimizes instantaneous market impact, but leaves the order exposed to directional price drift and market volatility for longer.
2. **Trading too quickly** eliminates volatility risk, but pays severe market impact costs by aggressively depleting order book depth.

The **Almgren & Chriss (2000)** framework provides the mathematically optimal execution trajectory on the efficient frontier of expected shortfall versus capture variance.

## 2. Mathematical Formulation

### 2.1 Price Dynamics & Impact Decomposition
Let $S_k$ denote asset mid-price at step $k$:
$$S_k = S_{k-1} + \sigma \tau^{1/2} \xi_k - \tau \gamma v_k$$
$$\tilde{S}_k = S_{k-1} - \eta v_k - \frac{1}{2} \tau \gamma v_k$$
where:
- $\gamma$: permanent price impact coefficient (shifts equilibrium price)
- $\eta$: temporary price impact coefficient (liquidity depletion penalty)
- $v_k = n_k / \tau$: trade velocity
- $\sigma$: asset volatility

### 2.2 Optimal Inventory Trajectory
Given risk aversion $\lambda \ge 0$, the optimal discrete trajectory is:
$$\kappa \approx \sqrt{\frac{\lambda \sigma^2}{\eta}}$$
$$x_j = X \cdot \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)}$$
$$n_j = x_{j-1} - x_j$$

## 3. Architecture & Components

- `AlmgrenChrissTrajectory` (`src/lobx/analytics/impact.py`): Closed-form solver for inventory schedules and theoretical E-V frontiers.
- `AlmgrenChrissStrategy` (`src/lobx/strategies/almgren_chriss.py`): Execution strategy emitting child market orders according to the optimal schedule.
- Multi-seed Evaluation Harness: Evaluates `almgren-chriss` across held-out test seeds against TWAP and VWAP.
- CLI Integration: `lobx eval --strategies twap,almgren-chriss --risk-aversion 1e-4`.
