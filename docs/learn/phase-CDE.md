# Phase C-D-E: Neural Networks, Adverse Selection, and RL Execution

> *Plain-language notes written after implementing the Neural Network, Quote Shading, and Reinforcement Learning execution pipeline.*

In this unified phase, we build three foundational pillars of modern algorithmic execution:
1. **Machine Learning Foundations (Phase C)**: A multi-layer perceptron written entirely from scratch in NumPy with mathematical gradient checking, followed by temporal sequence modeling (causal 1D-CNN and LSTM).
2. **Microstructural Adverse Selection & Quote Shading (Phase D)**: An informed trader agent that aggressively exploits stale limit orders, and an adaptive execution strategy that shades quotes and sizes up dynamically based on model forecasts.
3. **Reinforcement Learning for Execution (Phase E)**: A Gymnasium-compatible `ExecutionEnv` driven by the engine, optimized via Proximal Policy Optimization (PPO) Actor-Critic with Generalized Advantage Estimation (GAE).

---

## 1. Phase C: Neural Networks from Scratch

### The Forward Pass & Numerical Stability
Our two-layer MLP maps input features $X \in \mathbb{R}^{N \times D}$ to 3 class probabilities:
$$\text{Down} (0), \quad \text{Flat} (1), \quad \text{Up} (2)$$

The forward equations are:
$$z_1 = X W_1 + b_1$$
$$a_1 = \max(0, z_1) \quad (\text{ReLU activation})$$
$$z_2 = a_1 W_2 + b_2 \quad (\text{Logits})$$

To compute probabilities, we apply Softmax. In naive implementations, $\exp(z_2)$ easily overflows when logits exceed 700 (causing `NaN`). To guarantee numerical stability, we subtract the maximum logit per row:
$$p_k = \frac{\exp(z_{2, k} - \max_j z_{2, j})}{\sum_m \exp(z_{2, m} - \max_j z_{2, j})}$$

Subtracting a constant from all logits leaves the Softmax probabilities mathematically unchanged ($e^{z_k - c} / \sum e^{z_m - c} = e^{-c} e^{z_k} / (e^{-c} \sum e^{z_m}) = p_k$), but prevents floating-point overflow.

### The Backward Pass (Analytical Chain Rule)
For Categorical Cross-Entropy loss with true labels $y \in \{0, 1, 2\}$:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log(p_{i, y_i})$$

The gradient with respect to the output logits $z_2$ has an exceptionally elegant form:
$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{1}{N} (P - Y_{\text{one-hot}})$$

By applying the chain rule backwards through the network:
$$\frac{\partial \mathcal{L}}{\partial W_2} = a_1^T \left( \frac{\partial \mathcal{L}}{\partial z_2} \right), \quad \frac{\partial \mathcal{L}}{\partial b_2} = \sum_{\text{rows}} \frac{\partial \mathcal{L}}{\partial z_2}$$
$$\frac{\partial \mathcal{L}}{\partial a_1} = \left( \frac{\partial \mathcal{L}}{\partial z_2} \right) W_2^T$$
$$\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial a_1} \odot \mathbb{I}(z_1 > 0) \quad (\text{ReLU derivative})$$
$$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \left( \frac{\partial \mathcal{L}}{\partial z_1} \right), \quad \frac{\partial \mathcal{L}}{\partial b_1} = \sum_{\text{rows}} \frac{\partial \mathcal{L}}{\partial z_1}$$

### Numerical Gradient Checking
To prove mathematical correctness, `tests/test_numpy_mlp.py` computes two-sided finite-difference numerical gradients:
$$\frac{\partial \mathcal{L}}{\partial \theta_j} \approx \frac{\mathcal{L}(\theta_j + \epsilon) - \mathcal{L}(\theta_j - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-6}$$
The relative error between analytical and numerical gradients satisfies:
$$\frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|_2}{\|\nabla_{\text{analytical}}\|_2 + \|\nabla_{\text{numerical}}\|_2 + 10^{-15}} < 10^{-6}$$

### Temporal Sequence Modeling: Causal 1D-CNN
Snapshots have sequential structure. `lobx.ml.sequence_models` constructs sliding windows of length $W=10$ snapshots:
$$X_{\text{seq}}[t] = [x_{t-W+1}, \, x_{t-W+2}, \, \dots, \, x_t] \in \mathbb{R}^{W \times F}$$
A temporal 1D-CNN applies $K=3$ convolutional filters over time with ReLU and global temporal max-pooling. Crucially, sliding windows only look at past and current snapshots; zero lookahead data leakage occurs.

---

## 2. Phase D: Adverse Selection and Model-Driven Quote Shading

### Why the Original ML Strategy Failed
In Phase 6, our earlier model predicted price moves with 59.9% accuracy, but the adaptive strategy cost **19.45 bps** vs **19.25 bps** with the model switched off.
Why?
1. **Spread & Impact Penalty**: Crossing the spread with a market order cost 2–4 ticks. A 1-tick price prediction was smaller than the cost paid to act on it!
2. **Lack of Adverse Selection**: The simulator background agents consisted only of noise traders, market makers, and momentum traders. None of them had forward private information, so resting orders at the touch were never picked off!

### Introducing the Informed Trader Agent
`lobx.sim.agents.informed.InformedTrader` introduces microstructural adverse selection (Glosten-Milgrom 1985):
- The informed trader evaluates directional price drift and shallow book pressure.
- When an imminent downward shift is detected, it aggressively hits resting bids with market sell orders.
- When an upward shift is detected, it lifts resting asks with market buy orders.
- Resting liquidity providers now face real toxic flow: if they sit passively at the touch, they are filled immediately before the price falls!

### The Solution: Quote Shading and Dynamic Sizing
Rather than paying spread by crossing with market orders, `ModelPassiveStrategy` uses the model for **liquidity defense**:
1. **Quote Shading**:
   When $P(\text{adverse}) - P(\text{favorable}) > \theta_{\text{shade}}$:
   The market regime is toxic. The strategy retreats its limit quote 1–2 ticks away from the touch (e.g. higher ask for sellers, lower bid for buyers), dodging the aggressive informed seller.
2. **Dynamic Sizing**:
   When $P(\text{favorable}) - P(\text{adverse}) > \theta_{\text{shade}}$:
   The market is calm or favorable. The strategy sizes up its resting limit order by $1.5\times$ to capture the spread on more volume safely.
3. **Guaranteed Completion**:
   If behind schedule by more than a slice or approaching the deadline margin ($T_{\text{deadline}} - 5\text{s}$), the strategy aggressively sweeps remaining volume to guarantee 100% parent order completion.

---

## 3. Phase E: Reinforcement Learning for Execution (ExecutionEnv + PPO)

### The Markov Decision Process (MDP)
Optimal execution is framed as a discrete-time MDP:
- **State Space** $s_t \in \mathbb{R}^5$:
  1. Remaining quantity ratio: $q_t / Q_0 \in [0, 1]$
  2. Remaining time ratio: $(T - t) / T \in [0, 1]$
  3. Bid-ask spread: $a_t - b_t$
  4. Top-level order book imbalance: $(v_b - v_a) / (v_b + v_a) \in [-1, 1]$
  5. Short-term mid-price momentum: $m_t - m_{t-1}$
- **Action Space** $a_t \in \{0, 1, 2\}$:
  - `0`: Passive limit order at the touch
  - `1`: Aggressive market order (immediate fill)
  - `2`: Hold / shade (wait for favorable quote conditions)
- **Reward Function**:
  $$R_t = -\text{shortfall\_bps} \times 0.01$$
  Penalizes implementation shortfall relative to the arrival mid-price.

### Proximal Policy Optimization (PPO) with GAE
`lobx.rl.ppo.PPOAgent` implements Actor-Critic optimization:
1. **Generalized Advantage Estimation (GAE)**:
   $$\delta_t = R_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$
   $$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
   $\gamma=0.99$ balances long-term execution horizon, while $\lambda=0.95$ smoothly interpolates between bias and variance.
2. **Clipped Surrogate Objective**:
   $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
   where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ and $\epsilon = 0.2$. Clipping stops excessively large policy updates, ensuring stable convergence.

---

## 4. Practice Exercises

Try these three exercises to deepen your understanding:

### Exercise 1: Implement Learning Rate Warmup & Cosine Annealing
Modify `NumPyMLP.fit` in `src/lobx/ml/numpy_mlp.py` to support a learning rate scheduler:
- First $E_{\text{warmup}} = 5$ epochs: linear warmup from $\eta_0 / 10$ to $\eta_0$.
- Remaining epochs: cosine annealing $\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_0 - \eta_{\text{min}})\left(1 + \cos\left(\frac{t \pi}{E}\right)\right)$.
*Verify*: Measure training stability and loss convergence compared to constant learning rate on `tests/test_numpy_mlp.py`.

### Exercise 2: Inventory-Dependent Quote Shading (Avellaneda-Stoikov)
In `ModelPassiveStrategy` (`src/lobx/strategies/model_passive.py`), integrate an inventory risk penalty:
$$r(s, q) = s - q \gamma \sigma^2 (T - t)$$
When remaining inventory $q_t$ is high and time is running out, shade quotes less or even cross into the spread, but when inventory is low, shade more aggressively.
*Verify*: Run paired evaluation on 50 seeds and check if inventory penalty reduces deadline sweep penalties.

### Exercise 3: Policy Entropy Regularization in PPO
In `src/lobx/rl/ppo.py`, add an entropy bonus to the policy objective:
$$L^{\text{total}}(\theta) = L^{\text{CLIP}}(\theta) + c_e \mathcal{H}(\pi_\theta(s))$$
where $\mathcal{H}(p) = -\sum_k p_k \log p_k$.
*Verify*: Show that with $c_e = 0.01$, the agent avoids early policy collapse into 100% aggressive or 100% passive actions during early training episodes.
