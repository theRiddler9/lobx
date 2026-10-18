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
