# Learn: Almgren-Chriss Optimal Execution & Market Impact Modeling

## 1. Overview and Core Microstructure Principles

When an institutional trader needs to sell 100,000 shares, dumping the entire order into the order book at once would crash the market, causing catastrophic implementation shortfall. 

The **Almgren-Chriss (2000)** framework formalizes the fundamental tension of algorithmic execution:
- **Market Impact Penalty**: Pushing volume into the book quickly consumes available liquidity and widens the spread.
- **Timing / Volatility Risk**: Holding inventory over time exposes the position to unhedged variance ($\sigma^2 \tau x_k^2$).

By tuning a risk-aversion parameter $\lambda$:
- $\lambda \to 0$: The trader is risk-neutral, caring only about minimizing expected cost $\to$ trajectory converges to linear **TWAP**.
- $\lambda > 0$: The trader penalizes variance $\to$ execution is **front-loaded**, trading larger slices early to rapidly decrease inventory exposure.

---

## 2. Mathematical Formulations

### 2.1 The Objective Function
Minimize expected utility of execution:
$$U(x) = \mathbb{E}[x] + \lambda \mathbb{V}[x]$$
where:
$$\mathbb{E}[x] = \frac{1}{2} \gamma X^2 + \eta \sum_{k=1}^N \frac{n_k^2}{\tau}$$
$$\mathbb{V}[x] = \sigma^2 \sum_{k=1}^N \tau x_k^2$$

### 2.2 Closed-Form Solution
Defining the urgency parameter $\kappa$:
$$\kappa = \frac{1}{\tau} \operatorname{arccosh}\left( 1 + \frac{\lambda \sigma^2 \tau^2}{2 \eta} \right)$$

The optimal inventory $x_j$ at time $t_j = j \tau$ is:
$$x_j = X \cdot \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)}$$

The trading slice $n_j = x_{j-1} - x_j$ is:
$$n_j = 2 X \sinh\left(\frac{1}{2} \kappa \tau\right) \frac{\cosh(\kappa (T - t_{j-1/2}))}{\sinh(\kappa T)}$$

---

## 3. Practice Exercises

### Exercise 1: Risk-Neutral Equivalence
**Prompt**: Prove that when risk aversion $\lambda = 0$, the Almgren-Chriss optimal trajectory reduces to pure TWAP ($x_j = X(1 - j/N)$).
**Solution**:
When $\lambda = 0$, $\kappa \to 0$. Using the Taylor expansion $\sinh(z) = z + O(z^3)$:
$$x_j = \lim_{\kappa \to 0} X \frac{\kappa (T - t_j)}{\kappa T} = X \left(1 - \frac{t_j}{T}\right) = X \left(1 - \frac{j}{N}\right)$$
which is the uniform linear TWAP trajectory.

### Exercise 2: Calculating Urgency $\kappa$
**Prompt**: Suppose $\lambda = 10^{-4}$, $\sigma = 2.0$, $\eta = 0.01$, and $\tau = 10.0$ seconds. Compute the urgency parameter $\kappa$.
```python
import math

lam, sigma, eta, tau = 1e-4, 2.0, 0.01, 10.0
arg = 1.0 + 0.5 * (lam * (sigma**2) * (tau**2)) / eta  # 1.0 + 0.5 * 0.04 / 0.01 = 3.0
kappa_tau = math.acosh(arg)
kappa = kappa_tau / tau
print(f"kappa: {kappa:.4f}")
```
Output: $\kappa \approx 0.1763\ \text{s}^{-1}$.

### Exercise 3: Evaluating Trade-Off on the Efficient Frontier
**Prompt**: If a risk-averse trader chooses $\lambda = 10^{-3}$ rather than $\lambda = 0$, what happens to their expected shortfall and variance of capture?
**Answer**: Expected shortfall $\mathbb{E}[x]$ increases because larger slices are traded aggressively early, incurring higher temporary market impact $\eta \sum \frac{n_k^2}{\tau}$. In return, variance $\mathbb{V}[x]$ decreases significantly because inventory $x_k$ drops much faster, reducing exposure to price volatility.
