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
