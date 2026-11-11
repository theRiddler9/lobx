# Phase B: C++ Concurrency — Locks, Lock-Free Rings, and Thread Pools

> *Plain-language notes written after Phase B. Read these after exploring the code in `cpp/include/lobx/`.*

In high-performance trading systems, concurrency exists at two distinct levels:
1. **Microsecond Order Ingress & Matching**: Multiple clients or market makers communicating with a single order book.
2. **Batch Simulation & Multi-Seed Evaluation**: Simulating hundreds of independent trading sessions across multicore CPUs.

Phase B systematically explores four concurrency patterns, their trade-offs, and how to verify their correctness under ThreadSanitizer.

---

## 1. Reader-Writer Locking: `std::shared_mutex`

### The Problem
In an exchange matching engine, read queries (such as querying `best_bid()`, `best_ask()`, or extracting depth snapshots for market makers) occur far more frequently than order executions or cancellations.

If we use a traditional `std::mutex`, every reader serializes behind every other reader. If 16 market data subscribers query the book at the same time, 15 of them stall waiting for the lock, causing severe lock contention and cache invalidation.

### The Solution
`lobx::ConcurrentOrderBook` uses `std::shared_mutex`:
- **Readers** acquire a **shared lock** via `std::shared_lock<std::shared_mutex>`:
  Multiple threads can hold a shared lock simultaneously. No reader blocks any other reader.
- **Writers** (`submit`, `cancel`) acquire an **exclusive lock** via `std::unique_lock<std::shared_mutex>`:
  Only one writer can execute at a time, and all readers are locked out until matching is complete.

### Monotonic Timestamps in Concurrent Markets
When multiple client threads submit orders concurrently, thread scheduling jitter can cause an order stamped at $t=1002$ to arrive before an order stamped at $t=1000$. The underlying deterministic engine requires non-decreasing timestamps. `ConcurrentOrderBook` clamps the effective timestamp:
```cpp
Timestamp ts = std::max(order.ts, last_ts_);
last_ts_ = ts;
```
This guarantees monotonic progression without throwing unexpected backwards-time exceptions to racing clients.

---

## 2. Mutex + Condition Variable Queue: `ExchangeGateway`

### The Pattern
When multiple clients connect to an exchange, an ingress gateway serializes incoming requests before feeding them to the matching engine:
- Producer threads push `SequencedRequest` objects into `lobx::ConcurrentQueue`.
- The background matching thread pops requests and executes them against the order book.
- When no orders are pending, the worker thread sleeps on `std::condition_variable::wait()`.

### Why Condition Variables?
Without a condition variable, the engine thread would have to busy-spin in an infinite `while (true)` loop checking `queue.empty()`. Busy-spinning consumes 100% of a CPU core, generating heat and starving other processes. A condition variable yields the CPU to the OS scheduler when idle.

### The Trade-off: OS Latency Overhead
While robust and simple, condition variable wakeups require an operating system context switch (via `futex` on Linux or `WaitOnAddress` on Windows). Each wakeup incurs a **1,000–3,000 ns latency penalty**, making it unsuitable for sub-microsecond trading hot paths.

---

## 3. Lock-Free SPSC Ring Buffer: `SpscGateway`

### The Concept
When an institutional client has a direct market access (DMA) line, there is only **one producer thread** (the network socket reader) and **one consumer thread** (the matching engine).

In a Single-Producer Single-Consumer (SPSC) topology, **all mutexes and condition variables can be eliminated**.

`lobx::SpscRingBuffer` implements a circular array of power-of-two capacity ($2^k$) using atomic head and tail pointers:
- Capacity mask: `index & (Capacity - 1)` (bitwise AND is significantly faster than modulo `%`).

### Cacheline Alignment (`alignas(64)`) & False Sharing
On modern x86 and ARM processors, memory is fetched from RAM into L1/L2 caches in **64-byte cache lines**.

If `head_` and `tail_` resided in adjacent memory, they would share the same 64-byte cache line:
```
[ head_ (8 bytes) | tail_ (8 bytes) | ... unused ... ] <-- 64-byte Cache Line
      ▲                   ▲
Core 0 writes       Core 1 writes
```
Every time Core 0 updates `head_`, Core 1's entire L1 cache line is invalidated, forcing Core 1 to reload from L3 cache! This is called **false sharing**.

By declaring both pointers with `alignas(64)`:
```cpp
alignas(64) std::atomic<uint64_t> head_{0};
alignas(64) std::atomic<uint64_t> tail_{0};
```
they sit in separate physical cache lines, completely eliminating cache-line bouncing.

### Explicit Memory Ordering: Acquire vs. Release
Why don't we use default `std::memory_order_seq_cst`? Because sequential consistency inserts full memory fences (`MFENCE`), which stall CPU execution pipelines.

Instead, we use precise acquire-release semantics:
- **Producer writes payload, then stores head with `std::memory_order_release`**:
  Guarantees that the written order payload is visible in main memory *before* the consumer sees the updated `head_` index.
- **Consumer reads head with `std::memory_order_acquire`**:
  Guarantees that subsequent reads of the order payload will observe the data written by the producer.
- **Thread reading its own pointer uses `std::memory_order_relaxed`**:
  A single core always observes its own prior writes in program order; no synchronization fence is required.

### Empirical Benchmark: Mutex vs. SPSC
Command: `./build/test/lobx_concurrent_bench 100000 4`

| Gateway Mode | Implementation | Throughput | Average Latency |
|---|---|---:|---:|
| **ExchangeGateway** | Mutex + Condition Variable Queue | 1,334,525 orders/s | 749.33 ns/order |
| **SpscGateway** | Lock-Free SPSC Ring Buffer | **8,657,259 orders/s** | **115.51 ns/order** |

> **Result**: The lock-free SPSC gateway achieves **6.5× higher throughput** and reduces latency from 749 ns down to **115 ns per order** (a 6.5× latency reduction).

---

## 4. Multi-Seed Parallel Simulations: `ThreadPool`

### Embarrassingly Parallel Determinism
In LOBX, simulation is a pure mathematical function:
$$\text{Simulate}(\text{config}, \text{seed}) \to \text{EventLog}$$

There is zero global mutable state between distinct seeds. Therefore, multi-seed evaluation (e.g. running 200 seeds for strategy evaluation) can be executed concurrently across all available CPU cores using `lobx::ThreadPool`.

### Determinism Verification Test
In `cpp/tests/test_thread_pool.cpp` and `tests/test_parallel_determinism.py`, we run seeds concurrently across 4 worker threads and compare against single-threaded serial runs:
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    parallel_logs = list(executor.map(run_seed, seeds))

assert parallel_logs == serial_logs  # Exact byte-level parity!
```
Every fill, order ID, cancel event, and digest is 100% identical.

---

## 5. Sanitizer Verification: ThreadSanitizer (TSan)

Race conditions in concurrent C++ code are notorious for being non-deterministic—they reproduce once in a million runs and vanish under debuggers (Heisenbugs).

To formally prove the absence of data races, we add **ThreadSanitizer (`-fsanitize=thread`)** to CMake and CI:
```bash
cmake -S . -B build/tsan -DLOBX_BUILD_TESTS=ON -DLOBX_TSAN=ON
cmake --build build/tsan
ctest --test-dir build/tsan -R "concurrent|gateway|spsc|thread_pool" --output-on-failure
```
TSan instruments all memory accesses and intercepts synchronization primitives, mathematically verifying that no two threads access the same memory location without proper happens-before synchronization.

---

## Three Practice Exercises

**Exercise 1 — Observe False Sharing:**
Create a copy of `spsc_ring_buffer.hpp` without the `alignas(64)` decorators on `head_` and `tail_`. Run `test_spsc_multithreaded_stress` across 5 million items on both versions. Measure the elapsed time. How much does false sharing degrade throughput on your processor?

**Exercise 2 — Bounded Gateway Drop Policy:**
In `SpscGateway`, `submit_order()` currently returns `false` when the ring buffer is full. Modify the gateway to support an institutional `DropOldest` policy where, if full, the gateway pops the oldest unserviced request, marks it as `Rejected`, and enqueues the fresh order. How does this affect latency under queue saturation?

**Exercise 3 — Work-Stealing Pool:**
The current `ThreadPool` uses a single shared queue guarded by a mutex. For fine-grained tasks, this central lock can become a bottleneck. Research the *Chase-Lev work-stealing deque*. Explain why assigning each worker thread its own local deque reduces lock contention, and how work-stealing balances uneven seed workloads.
