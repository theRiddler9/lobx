# Phase 3: the C++ engine and differential testing

## What this phase built

A second matching engine in C++17 (`cpp/`) that behaves exactly like the Python one, exposed to
Python with pybind11 as `lobx.engine.native.NativeOrderBook`, plus the tests that prove they agree.

## Key concepts

- **Same contract, two implementations.** Both engines implement the `OrderBook` protocol. The
  simulator does not care which one it gets (`lobx run --engine native`).
- **Data structures.** Each side is a `std::map` from price to a `Level`; a level holds a
  `std::list` (the FIFO queue) and its total quantity. `std::map` keeps prices sorted, so the best
  price is `begin()`. An `unordered_map` from order id to (level, list iterator) makes cancel O(1)
  for the order itself: no searching the queue. A level is erased when its last order leaves.
- **Price-time priority.** Best price first; at one price the oldest order first; trades happen at
  the resting order's price.
- **Differential testing.** Feed the same random stream to a slow, obviously-correct engine and a
  fast one; any disagreement is a bug in the fast one. The Python engine is the specification.
  Mutation checking (break the C++ on purpose, confirm the test fails) shows the test has teeth.
- **Why the speedup is smaller than expected.** Crossing the Python/C++ boundary on every order
  costs more than the matching. See `docs/results.md`.
- **Build.** `pip install -e .` runs scikit-build-core, which runs CMake, which compiles the
  extension `lobx/_native`. `native.py` imports it and raises a clear error if it is missing.

## Where to look

- `cpp/include/lobx/order_book.hpp` (the data structures, with comments on why)
- `cpp/src/order_book.cpp` (`validate`, `match`, `rest`, `cancel`)
- `cpp/src/bindings.cpp` and `src/lobx/engine/native.py` (the boundary)
- `src/lobx/engine/streams.py`, `tests/test_native_differential.py`
- `cpp/tests/test_order_book.cpp`, `benchmarks/bench_engines.py`, `cpp/bench/`

## What to read next

Almgren and Chriss on optimal execution (Phase 4), and the pybind11 docs on exception translation.

## Practice exercises

1. Re-implement `cancel` in C++ without looking at the code, then run the C++ tests and the
   differential test against it.
2. Add an "immediate-or-cancel" limit order type to both engines (trade what crosses, discard the
   rest instead of resting). Write the scenario fixtures first, then make both engines pass.
3. Add a batch method `submit_many` to the bindings that takes arrays of orders, and measure how
   much of the Python/C++ boundary cost it removes.
