# Results

Every number here was produced by the command written next to it. All results come from a
**simulated** market; nothing here says anything about real markets.

## Phase 3: engine benchmark

Machine: Linux 6.18 x86_64 (cloud container, 2 vCPU), Intel Xeon @ 2.10 GHz, g++ 13.3 `-O2`
(CMake Release), Python 3.11.17. Numbers vary by machine; re-run before quoting them elsewhere.

### Pure C++ engine (no Python in the loop)

Command: `cmake -S . -B build/cpp -DLOBX_BUILD_TESTS=ON && cmake --build build/cpp && ./build/cpp/lobx_bench 1000000`

| metric | value |
|---|---:|
| throughput | 5,220,442 orders/s |
| median latency | 124 ns |
| p99 latency | 463 ns |

### Python reference vs C++ engine, called from Python

Command: `python benchmarks/bench_engines.py --ops 1000000`

| engine | ops/s | median ns | p99 ns | speedup vs python |
|---|---:|---:|---:|---:|
| python | 335,980 | 2,153 | 11,321 | 1.0x |
| native-wrapped | 724,793 | 841 | 5,101 | 2.2x |
| native-raw | 1,072,217 | 710 | 2,414 | 3.2x |

`native-wrapped` is `NativeOrderBook` (builds `Order` and `Fill` dataclasses, what the simulator
uses). `native-raw` calls the extension with plain ints. Latency is timed per operation with a
clock that costs about 50-100 ns per call, so the p99 values are upper bounds.

### Reading the numbers (the 10x target)

The AGENTS.md target is a C++ engine at least 10x faster than the Python engine.

- **Engine against engine: met.** The C++ engine does 5.22M orders/s; the Python reference does
  0.34M orders/s on a stream of the same shape: about **15x**. The two benchmarks use different
  random generators (an LCG in C++, `random.Random` in Python), so this ratio is approximate.
- **Called from Python: not met.** Through pybind11 the speedup is only 2.2x (wrapped) to 3.2x
  (raw). The matching itself is fast; what remains is the cost of crossing the Python/C++
  boundary and converting arguments and fills on every call.
- Consequence: the simulator, which makes one call per order, gains little from the native engine.
  A batch API (many orders per call) would remove most of that overhead; it is not built yet.

## Differential test

Command: `pytest tests/test_native_differential.py`

One million random operations plus 20 short streams: every fill, cancel result, best price and
snapshot is identical between the engines. Mutation check: changing `<=` to `<` in the C++
crossing rule makes the test fail at once.

## Phase 4: baseline execution strategies

Setup: sell 3,000 shares (about 20% of the market's volume in the window) between simulated
t=60 s and t=360 s, in 30 slices of 100 shares, as market orders. Test seeds 1000..1199. VWAP's
volume profile is estimated on training seeds 0..99. C++ engine. Everything is from a **simulated**
market with simple agents; it says nothing about real markets.

Command: `python -m lobx.cli eval --strategies twap,vwap --seeds 200 --plot docs/figures/baselines.png`
(about 1.6 minutes on the machine described above, including the VWAP training runs)

| strategy | seeds | incomplete | mean (bps) | std (bps) | 95% CI for the mean (bps) |
|---|---:|---:|---:|---:|---|
| twap | 198 | 2 | 56.54 | 41.14 | [50.77, 62.31] |
| vwap | 197 | 3 | 57.81 | 42.22 | [51.88, 63.74] |

Plot: `docs/figures/baselines.png`.

How to read it:

- **TWAP and VWAP are indistinguishable here.** The confidence intervals overlap almost entirely
  and the estimated volume profile is flat (every bucket holds about 3.3% of volume, which is
  1/30), so VWAP sends almost the same slices as TWAP. That is a property of the simulator, whose
  agents trade at constant rates with no intraday volume pattern. It is not evidence that VWAP is
  not useful in real markets. A formal paired comparison comes in Phase 6.
- **The spread between seeds is large (std about 41 bps)** compared with the difference between
  strategies. Most of a single run's cost comes from where the price drifts during the five
  minutes, not from the algorithm.
- **Incomplete runs.** In 2 (TWAP) and 3 (VWAP) of 200 seeds the order did not finish: the bid
  side of the book was too thin when a slice ran, a market order's unfilled remainder is
  discarded, and the final slice could not catch up. These runs have no measurable shortfall and
  are excluded from the mean, which probably flatters both baselines slightly (thin books are the
  expensive ones). Phase 6 must handle this explicitly: finish by the deadline, and compare
  strategies only on seeds where both completed.

## Lookahead test

Command: `pytest tests/test_leakage.py`

TWAP and VWAP pass the poison test; a deliberately peeking strategy raises `LookaheadError`. The
test covers decisions made from recorded book snapshots. It cannot prove the absence of every
possible leak, only that a strategy's decisions up to time t do not change when later data does.

## Phase 5: order-flow predictor

The model predicts which way the mid-price moves over the next 1 second (5 snapshots of 200 ms):
down, flat, or up. Features: spread, order-book imbalance at depth 1 and 5, mid-price change over
1, 5 and 25 snapshots, and trade-flow imbalance over 1 s and 5 s. Rows come from 300-second runs
of the **simulated** market, after a 30-second warm-up.

Seeds: trained on 0..99; LightGBM early-stopped on 100..199; **scored on test seeds 1000..1099**
(133,938 rows), which were never used to fit or tune anything. The 1-second horizon was picked by a
quick look at validation seeds only (2, 5 and 10 snapshots were tried; the test seeds were not
touched).

Command: `python -m lobx.cli train --out models` (about 2 minutes, C++ engine)

| model | rows | accuracy | log loss | direction acc. | ECE up | ECE down |
|---|---:|---:|---:|---:|---:|---:|
| naive | 133938 | 0.4510 | 1.0660 | 0.5558 | 0.0064 | 0.0101 |
| logistic | 133938 | 0.5345 | 0.9589 | 0.7000 | 0.0088 | 0.0559 |
| lightgbm | 133938 | 0.5992 | 0.8645 | 0.7410 | 0.0047 | 0.0049 |

- *naive* always predicts the training class frequencies. *direction acc.* is accuracy on rows where
  the mid really moved (up versus down). The naive model scores 55.6% because down moves are
  commoner than up moves in this market (a downward drift), so even guessing "down" wins.
  *ECE* is the expected calibration error of the up and down probabilities (0 = perfectly
  calibrated; "70%" really happens 70% of the time).
- **Both models beat the naive baseline** on held-out seeds, LightGBM clearly (accuracy 0.599 against
  0.451; log loss 0.865 against 1.066) and with good calibration (ECE under 0.005). Logistic
  regression gets much of the directional skill but is poorly calibrated for "down".
- **Caveats.** Rows within one run are strongly correlated, so 134k rows is not 134k independent
  samples; the real sample size is closer to the 100 seeds. No confidence intervals are reported
  for these metrics yet. The simulated market is made of simple agents (a market maker quoting
  around a reference price, noise traders, a momentum trader), so predictability here may be
  easier than in a real market. This result says the pipeline finds signal in the simulator. It
  does not say the signal is profitable: that is what Phase 6 tests.

## Feature leakage test

Command: `pytest tests/test_features_causal.py`

Each feature row recomputed from only the data available at time t equals the row computed from
the whole run, and replacing everything after t with garbage does not change it. A deliberately
leaky feature fails both checks.

## Phase 6: adaptive execution versus TWAP and VWAP

Same setup as Phase 4 (sell 3,000 shares between t=60 s and t=360 s; test seeds 1000..1199; C++
engine). All results are from a **simulated** market with simple agents.

The strategies:

- **twap**, **vwap**: as in Phase 4 (market orders, one slice every 10 s).
- **adaptive**: follows the TWAP pace, wakes every 200 ms, and uses the LightGBM model's forecast
  of the next second. If P(price falls) minus P(price rises) is above a threshold it sells what is
  owed at once with a market order; otherwise it rests a limit order at the best ask. Safety nets:
  anything overdue by a full slice is sold at once, and everything left is sold in the last 5 s.
  Threshold 0.9, chosen on validation seeds (below).
- **passive**: the *same strategy with the model switched off* (no forecasts, so only resting
  orders plus the safety nets). It is an ablation, added to show how much of the result comes from
  the model and how much from passivity.

Command: `python -m lobx.cli eval --strategies twap,vwap,passive,adaptive --seeds 200 --plot docs/figures/comparison.png`
(6.2 minutes on the machine described above; model from `python -m lobx.cli train --out models`)

| strategy | seeds | incomplete | mean (bps) | std (bps) | 95% CI for the mean (bps) |
|---|---:|---:|---:|---:|---|
| twap | 198 | 2 | 56.54 | 41.14 | [50.77, 62.31] |
| vwap | 197 | 3 | 57.81 | 42.22 | [51.88, 63.74] |
| passive | 200 | 0 | 19.25 | 19.54 | [16.53, 21.98] |
| adaptive | 200 | 0 | 19.45 | 19.77 | [16.69, 22.20] |

Paired by seed (a - b in bps; negative means a is cheaper; only seeds where both completed):

| comparison | seeds used | a - b, mean (bps) | 95% CI (bps) | a cheaper on | verdict |
|---|---:|---:|---|---:|---|
| vwap - twap | 197 | 1.41 | [0.71, 2.10] | 26% of seeds | twap cheaper |
| passive - twap | 198 | -37.25 | [-41.44, -33.06] | 99% of seeds | passive cheaper |
| adaptive - twap | 198 | -37.06 | [-41.26, -32.85] | 99% of seeds | adaptive cheaper |
| passive - vwap | 197 | -38.53 | [-42.86, -34.21] | 99% of seeds | passive cheaper |
| adaptive - vwap | 197 | -38.34 | [-42.68, -34.00] | 99% of seeds | adaptive cheaper |
| adaptive - passive | 200 | 0.19 | [-0.06, 0.45] | 0% of seeds | no clear difference |

Plot: `docs/figures/comparison.png`.

### The honest reading

1. **The adaptive strategy beats TWAP and VWAP by about 37 bps on 99% of seeds, and it always
   completes** (200 of 200; the baselines left 2 and 3 seeds unfinished).
2. **But the model is not why.** The passive ablation, with no model at all, is just as cheap
   (difference +0.19 bps, 95% CI [-0.06, 0.45]). The gain comes from *resting limit orders instead
   of crossing the spread*, not from the forecast. Calling this an "ML win" would be wrong.
3. **Using the model made things worse, not better.** Tuning on validation seeds 100..139
   (`python -m lobx.cli tune`; adaptive minus TWAP, paired):

   | threshold | adaptive mean (bps) | adaptive - TWAP, mean (bps) | 95% CI (bps) |
   |---:|---:|---:|---|
   | 0.30 | 56.46 | -0.92 | [-5.69, 3.84] |
   | 0.50 | 49.83 | -7.55 | [-12.58, -2.53] |
   | 0.70 | 37.62 | -19.77 | [-25.99, -13.54] |
   | 0.90 | 20.61 | -36.77 | [-47.07, -26.48] |
   | 1.00 | 20.61 | -36.77 | [-47.07, -26.48] |

   The lower the threshold (the more often the forecast triggers a market order), the higher the
   cost. The cheapest thresholds, 0.9 and 1.0, are those at which the model almost never fires.
   Selecting the threshold on validation seeds therefore selected "ignore the model", which is why
   adaptive and passive are indistinguishable on the test seeds.

### Why the model does not help (analysis; the mechanisms below were not tested separately)

- **The forecast is small compared with the cost of acting on it.** The model predicts a mid-price
  move over one second; such moves are about one tick. Crossing the spread to dodge a one-tick
  fall costs about a tick of spread plus the price impact of the order, so even a correct forecast
  barely pays.
- **A market sell moves the price against the seller in this simulator.** TWAP's 57 bps is mostly
  impact and its aftermath: each market sell walks the book, and the momentum agents in the market
  follow the falling price. Resting orders avoid both effects. This is a property of the simulated
  agents, so the size of the 37 bps gap should not be expected to carry over to a real market.
- **Passive orders have easy fills here.** Noise traders send random market orders (30% of their
  orders), which keep hitting resting orders. The engine models queue position (price-time
  priority), but there is no adverse selection from informed traders, no latency, and no fees or
  rebates, all of which make passive trading harder in practice.
- **The deadline rules cap the risk.** Passivity can delay a share by at most one slice, and
  everything left is sold by t = 355 s. Those safety nets, not the model, guarantee completion.

### What this does and does not show

It shows that, in this simulator, a limit-order strategy with deadline safety nets is much cheaper
than TWAP or VWAP market-order slicing, and it shows how to test that fairly: paired seeds, held-out
test seeds, a no-model ablation. It does **not** show that the model has any execution value, and
it says nothing about real markets. A fair next experiment would be a model whose forecast is large
enough to matter (a longer horizon, or a larger predicted move), or using the forecast to decide
*how much* to post passively rather than whether to cross the spread.

### Other observations

- VWAP is about 1.4 bps *more* expensive than TWAP (CI [0.71, 2.10]): small but not zero. The
  simulated volume is flat, so the estimated profile is noise around 1/30 per slice and only adds
  variation to the slice sizes.
- Phase 4's baseline numbers are reproduced exactly (same seeds, same code path).
