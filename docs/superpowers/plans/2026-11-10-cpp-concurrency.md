# Asynchronous Concurrent C++ Exchange Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, high-performance concurrent exchange architecture in C++17 using threads (`std::thread`), atomics (`std::atomic`), and mutexes (`std::mutex`, `std::shared_mutex`, `std::condition_variable`), including a thread-safe generic queue, a concurrent order book, an asynchronous exchange gateway with an atomic sequencer and background worker thread, an egress market data event stream, and comprehensive unit tests and throughput benchmarks.

**Architecture:** Multiple participant threads submit order and cancel requests to an asynchronous `ExchangeGateway`. The gateway assigns monotonic sequence numbers using `std::atomic<uint64_t>` and pushes them to a thread-safe `ConcurrentQueue`. A dedicated background matching worker thread pulls sequenced requests, applies them to a `ConcurrentOrderBook` (protecting matching logic and maintaining atomic trade and volume counters), publishes execution events to an egress subscriber queue, and signals sequence barriers. The components are exposed to Python via pybind11 and validated through native C++ multi-threaded tests, benchmarks, and Python unit tests.

**Tech Stack:** C++17 (`std::thread`, `std::atomic`, `std::mutex`, `std::shared_mutex`, `std::condition_variable`), CMake 3.18+, pybind11, Python 3.11+.

**Spec:** Requirement to add intensive, readable, and explainable C++ functionality demonstrating real-world exchange concurrency with threads, atomics, and mutexes on a dedicated branch (`feat/cpp-concurrency`).

## Global Constraints

- Prices are strictly integer ticks (`Price` = `std::int64_t`). No floats for prices or quantities.
- Timestamps are integer nanoseconds (`Timestamp` = `std::uint64_t`).
- Monotonically increasing sequence IDs use `std::atomic<uint64_t>`.
- All shared state is strictly protected by RAII locks (`std::lock_guard`, `std::unique_lock`, `std::shared_lock`) or atomic operations; no data races.
- Code must follow C++17 standards and existing project conventions (no external test framework for C++ unit tests; macro-based assertions with clean return codes).
- pybind11 bindings must release the Python GIL (`py::gil_scoped_release`) when waiting on concurrent C++ queues or barriers to allow true cross-language concurrency.

## Review Focus

1. **Deadlock & Order of Locking**: Gateway shutdown vs worker thread join (condition variable notification must unblock worker before `join()`).
2. **Atomic Memory Orderings**: Ensure correct acquire/release or sequential consistency semantics for sequence numbers and lifecycle flags (`closed_`, `running_`).
3. **Queue Bounded Backpressure**: Handling full queues when capacity is bounded (non-blocking `try_push` vs blocking `push` with timeout).
4. **Exception Safety in Worker Thread**: If an invalid order or exception is thrown during matching, the worker thread must catch it, record the error event, and continue processing rather than terminating the exchange thread.
5. **Clean Resource Deallocation**: Graceful shutdown in destructors if `stop()` was not called explicitly, ensuring zero dangling threads.

---

### Task 1: Thread-Safe Concurrent Queue (`ConcurrentQueue<T>`)

**Files:**
- Create: `cpp/include/lobx/concurrent_queue.hpp`
- Create: `cpp/tests/test_concurrent_queue.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `lobx::ConcurrentQueue<T>` with `push(T)`, `try_push(T)`, `pop()`, `try_pop()`, `wait_pop_for(duration)`, `close()`, `size()`, `empty()`, `is_closed()`, `total_pushed()`, `total_popped()`.

- [ ] **Step 1: Write the failing C++ test `cpp/tests/test_concurrent_queue.cpp`**

```cpp
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "lobx/concurrent_queue.hpp"

namespace {
int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { ++g_failures; std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; } } while (0)

void test_basic_queue_operations() {
  lobx::ConcurrentQueue<int> q(10);
  CHECK(q.empty());
  CHECK(q.size() == 0);
  CHECK(q.push(42));
  CHECK(q.size() == 1);
  CHECK(!q.empty());
  auto item = q.pop();
  CHECK(item.has_value() && *item == 42);
  CHECK(q.empty());
}

void test_multi_producer_multi_consumer() {
  lobx::ConcurrentQueue<int> q(100);
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kItemsPerProducer = 1000;
  std::atomic<int> sum_consumed{0};
  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  for (int i = 0; i < kConsumers; ++i) {
    consumers.emplace_back([&]() {
      while (auto val = q.pop()) {
        sum_consumed.fetch_add(*val, std::memory_order_relaxed);
      }
    });
  }

  for (int i = 0; i < kProducers; ++i) {
    producers.emplace_back([&]() {
      for (int j = 1; j <= kItemsPerProducer; ++j) {
        q.push(1);
      }
    });
  }

  for (auto& t : producers) t.join();
  q.close();
  for (auto& t : consumers) t.join();

  CHECK(sum_consumed.load() == kProducers * kItemsPerProducer);
  CHECK(q.total_pushed() == static_cast<std::size_t>(kProducers * kItemsPerProducer));
}
} // namespace

int main() {
  test_basic_queue_operations();
  test_multi_producer_multi_consumer();
  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "All concurrent queue tests passed\n";
  return 0;
}
```

- [ ] **Step 2: Add test target to `CMakeLists.txt` and verify it fails**

Modify `CMakeLists.txt` to find `Threads` and build `test_concurrent_queue`:
```cmake
find_package(Threads REQUIRED)
target_link_libraries(lobx_engine PUBLIC Threads::Threads)
```
Run: `cmake -S . -B build/cpp -DLOBX_BUILD_TESTS=ON && cmake --build build/cpp`
Expected: Build failure due to missing `lobx/concurrent_queue.hpp`.

- [ ] **Step 3: Implement `lobx::ConcurrentQueue<T>` in `cpp/include/lobx/concurrent_queue.hpp`**

Implement generic thread-safe queue:
- `std::queue<T> queue_`
- `mutable std::mutex mutex_`
- `std::condition_variable not_empty_`
- `std::condition_variable not_full_`
- `std::atomic<bool> closed_{false}`
- `std::atomic<std::size_t> total_pushed_{0}`
- `std::atomic<std::size_t> total_popped_{0}`
- `std::size_t capacity_` (0 = unbounded)
- Full push/pop/timeout/close lifecycle with RAII locking.

- [ ] **Step 4: Build and run test to verify it passes**

Run: `cmake --build build/cpp && ./build/cpp/test_concurrent_queue`
Expected: PASS with "All concurrent queue tests passed".

- [ ] **Step 5: Commit**

```bash
git add cpp/include/lobx/concurrent_queue.hpp cpp/tests/test_concurrent_queue.cpp CMakeLists.txt
git commit -m "feat(cpp): add thread-safe ConcurrentQueue with atomics and mutexes"
```

---

### Task 2: Thread-Safe Concurrent Order Book (`ConcurrentOrderBook`)

**Files:**
- Create: `cpp/include/lobx/concurrent_order_book.hpp`
- Create: `cpp/src/concurrent_order_book.cpp`
- Create: `cpp/tests/test_concurrent_order_book.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `lobx::OrderBook`, `lobx::Order`, `lobx::Fill`, `lobx::Snapshot`
- Produces: `lobx::ConcurrentOrderBook` with thread-safe `submit()`, `cancel()`, `snapshot()`, `best_bid()`, `best_ask()`, `resting_orders()`, and atomic performance statistics (`total_submits()`, `total_cancels()`, `total_fills()`, `total_traded_volume()`).

- [ ] **Step 1: Write the failing test `cpp/tests/test_concurrent_order_book.cpp`**

Test concurrent order insertion, cancellations, and snapshot reads across multiple worker threads:
- 4 concurrent writer threads submitting crossing orders.
- 2 concurrent reader threads taking snapshots.
- Verify atomic counters match actual operations and book invariants hold without data races.

- [ ] **Step 2: Add test target to `CMakeLists.txt` and verify it fails**

Run: `cmake --build build/cpp`
Expected: Build failure because `concurrent_order_book.hpp` does not exist yet.

- [ ] **Step 3: Implement `ConcurrentOrderBook`**

Create header `cpp/include/lobx/concurrent_order_book.hpp` and implementation `cpp/src/concurrent_order_book.cpp`:
- Encapsulates `lobx::OrderBook book_`.
- `mutable std::shared_mutex rw_mutex_` for reader-writer concurrency (exclusive lock for `submit`/`cancel`, shared lock for `snapshot`/`best_bid`/`best_ask`).
- Atomic metric counters:
  - `std::atomic<uint64_t> total_submits_{0}`
  - `std::atomic<uint64_t> total_cancels_{0}`
  - `std::atomic<uint64_t> total_fills_{0}`
  - `std::atomic<uint64_t> total_traded_volume_{0}`

- [ ] **Step 4: Build and run test to verify it passes**

Run: `cmake --build build/cpp && ./build/cpp/test_concurrent_order_book`
Expected: PASS with "All concurrent order book tests passed".

- [ ] **Step 5: Commit**

```bash
git add cpp/include/lobx/concurrent_order_book.hpp cpp/src/concurrent_order_book.cpp cpp/tests/test_concurrent_order_book.cpp CMakeLists.txt
git commit -m "feat(cpp): add thread-safe ConcurrentOrderBook with shared mutex and atomic metrics"
```

---

### Task 3: Asynchronous Exchange Gateway (`ExchangeGateway`)

**Files:**
- Create: `cpp/include/lobx/gateway.hpp`
- Create: `cpp/src/gateway.cpp`
- Create: `cpp/tests/test_gateway.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `lobx::ConcurrentQueue`, `lobx::ConcurrentOrderBook`, `lobx::Order`, `lobx::Fill`
- Produces: `lobx::ExchangeGateway` with:
  - `submit_order(Side, OrderType, Price, Qty, Timestamp, AgentId) -> uint64_t seq_id`
  - `cancel_order(OrderId, Timestamp, AgentId) -> uint64_t seq_id`
  - `wait_until_processed(uint64_t seq_id, timeout_ms)` (condition variable barrier)
  - `poll_event()` / `wait_event()` for market data egress
  - `start()`, `stop()`, `is_running()`
  - Atomic stats: `orders_received()`, `requests_processed()`, `trades_published()`

- [ ] **Step 1: Write failing test `cpp/tests/test_gateway.cpp`**

Write test covering:
- Starting gateway.
- Multiple client threads submitting hundreds of orders.
- Synchronization barrier `wait_until_processed()`.
- Verifying egress event stream receives all execution and cancel reports.
- Graceful shutdown and queue drainage.

- [ ] **Step 2: Add test target to `CMakeLists.txt` and verify it fails**

Run: `cmake --build build/cpp`
Expected: Build failure due to missing `gateway.hpp`.

- [ ] **Step 3: Implement `ExchangeGateway`**

Create `cpp/include/lobx/gateway.hpp` and `cpp/src/gateway.cpp`:
- `std::atomic<uint64_t> next_seq_{1}` for sequencing ingress requests.
- `std::atomic<uint64_t> last_processed_seq_{0}`.
- `ConcurrentQueue<SequencedRequest> ingress_queue_`.
- `ConcurrentQueue<GatewayEvent> egress_queue_`.
- `std::thread worker_thread_`.
- `std::condition_variable seq_cv_` and `std::mutex seq_mutex_`.
- Background worker execution loop matching orders against `ConcurrentOrderBook`, recording fill/cancel events, updating `last_processed_seq_`, and notifying barrier waiters.

- [ ] **Step 4: Build and run test to verify it passes**

Run: `cmake --build build/cpp && ./build/cpp/test_gateway`
Expected: PASS with "All gateway tests passed".

- [ ] **Step 5: Commit**

```bash
git add cpp/include/lobx/gateway.hpp cpp/src/gateway.cpp cpp/tests/test_gateway.cpp CMakeLists.txt
git commit -m "feat(cpp): add asynchronous ExchangeGateway with background worker thread and atomic sequencer"
```

---

### Task 4: High-Throughput Multi-Threaded Benchmark (`bench_concurrent_gateway`)

**Files:**
- Create: `cpp/bench/bench_concurrent_gateway.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `lobx::ExchangeGateway`
- Produces: Native benchmark executable measuring orders/second across 1, 2, 4, 8 concurrent producer threads with barrier completion timing.

- [ ] **Step 1: Implement `cpp/bench/bench_concurrent_gateway.cpp`**

Implement benchmarking harness:
- Configurable order count (e.g. 1,000,000 orders) and thread count (e.g. 1 to 8 threads).
- Measure ingress queue rate (producers submitting) and engine matching rate (worker processing).
- Print detailed throughput (orders/sec) and latency statistics.

- [ ] **Step 2: Add benchmark target to `CMakeLists.txt`**

```cmake
add_executable(lobx_concurrent_bench cpp/bench/bench_concurrent_gateway.cpp)
target_link_libraries(lobx_concurrent_bench PRIVATE lobx_engine)
```

- [ ] **Step 3: Build and run benchmark**

Run: `cmake --build build/cpp && ./build/cpp/lobx_concurrent_bench 100000 4`
Expected: Benchmark completes, printing ingress rate and matching worker throughput.

- [ ] **Step 4: Commit**

```bash
git add cpp/bench/bench_concurrent_gateway.cpp CMakeLists.txt
git commit -m "perf(benchmarks): add multi-threaded concurrent gateway benchmark"
```

---

### Task 5: Python pybind11 Bindings & Integration Tests

**Files:**
- Modify: `cpp/src/bindings.cpp`
- Modify: `src/lobx/_native.pyi`
- Create: `tests/test_concurrent_gateway_py.py`

**Interfaces:**
- Exposes: `_native.ConcurrentBook`, `_native.ExchangeGateway`, `_native.GatewayEventType`, `_native.GatewayEvent` to Python.

- [ ] **Step 1: Write failing Python integration test `tests/test_concurrent_gateway_py.py`**

Test multi-threaded usage from Python using `concurrent.futures.ThreadPoolExecutor`:
- Multiple Python threads submitting orders via `gateway.submit_order()`.
- Waiting on sequence barrier with `gateway.wait_until_processed()`.
- Polling egress events and checking book snapshot.

- [ ] **Step 2: Extend pybind11 bindings in `cpp/src/bindings.cpp`**

Bind:
- `ConcurrentOrderBook`
- `ExchangeGateway` (with `py::gil_scoped_release` during blocking calls like `wait_until_processed` and `wait_event`)
- `GatewayEvent` and event types.

- [ ] **Step 3: Update stub file `src/lobx/_native.pyi`**

Add type annotations for `ConcurrentBook`, `ExchangeGateway`, `GatewayEvent`.

- [ ] **Step 4: Run test suite**

Run tests and linters:
`pytest tests/test_concurrent_gateway_py.py`
`ruff check .`
`mypy src`
Expected: All tests pass and typechecks clean.

- [ ] **Step 5: Commit**

```bash
git add cpp/src/bindings.cpp src/lobx/_native.pyi tests/test_concurrent_gateway_py.py
git commit -m "feat(bindings): expose ConcurrentOrderBook and ExchangeGateway to Python"
```

---

### Task 6: Documentation and Educational Guide

**Files:**
- Create: `docs/learn/concurrent-exchange.md`
- Modify: `docs/architecture.md`
- Modify: `README.md`

**Interfaces:**
- Educational guide explaining:
  - Why and how threads, atomics, and mutexes are used in financial exchanges.
  - Reader-writer locks (`std::shared_mutex`) vs simple mutexes (`std::mutex`).
  - Condition variables and spurious wakeup prevention (`cv.wait(lock, predicate)`).
  - Memory orderings (`memory_order_relaxed`, `memory_order_acquire`, `memory_order_release`, `memory_order_seq_cst`).
  - Sequence numbers, barriers, and backpressure.
  - Three practice exercises for the reader.

- [ ] **Step 1: Write `docs/learn/concurrent-exchange.md`**
- [ ] **Step 2: Update `docs/architecture.md` and `README.md`**
- [ ] **Step 3: Commit**

```bash
git add docs/learn/concurrent-exchange.md docs/architecture.md README.md
git commit -m "docs: add comprehensive educational guide on C++ concurrency in LOBX"
```
