# Concurrency in Electronic Exchanges: Threads, Atomics, and Mutexes

This guide walks through how modern multi-threaded electronic trading systems are designed in C++17, using the concurrent exchange pipeline added in `cpp/include/lobx/`.

---

## 1. Why Concurrency Matters in an Exchange

In a real financial exchange (e.g., NASDAQ, CME, LMAX):
* **Thousands of market participants** (market makers, hedge funds, algorithmic brokers) send orders simultaneously over network connections.
* **The order book must remain strictly deterministic**: price-time priority requires orders at the same price level to match in the exact order they arrive.
* If multiple threads modified an order book's linked lists or maps without synchronization, the pointers would corrupt immediately (a **data race** causing crashes or memory corruption).
* Conversely, if every participant blocked on a single global lock just to submit an order, the exchange's throughput would collapse.

To solve this, exchanges decouple **order ingress** from **order matching** using an **asynchronous gateway and sequencer pattern**.

---

## 2. The Core Concurrency Primitives

### A. Mutual Exclusion: `std::mutex` and `std::shared_mutex`

A **mutex** (mutual exclusion) ensures only one thread executes a critical section at any instant.

```cpp
std::mutex mutex_;
{
  std::lock_guard<std::mutex> lock(mutex_); // RAII lock: automatically releases on scope exit
  // critical section
}
```

In `cpp/include/lobx/concurrent_order_book.hpp`, we use `std::shared_mutex` for **Reader-Writer Locking**:
* **Writers** (`submit`, `cancel`): Need exclusive access because they modify resting orders. They acquire `std::unique_lock<std::shared_mutex>`.
* **Readers** (`snapshot`, `best_bid`, `best_ask`): Do not modify anything. Multiple threads can read the book in parallel without blocking each other by acquiring `std::shared_lock<std::shared_mutex>`.

```cpp
// Writer: exclusive lock
std::vector<Fill> ConcurrentOrderBook::submit(const Order& order) {
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  return book_.submit(order);
}

// Reader: shared lock (parallel reads allowed)
Snapshot ConcurrentOrderBook::snapshot(std::size_t depth) const {
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  return book_.snapshot(depth);
}
```

---

### B. Condition Variables: `std::condition_variable`

Threads often need to wait for a condition to become true (e.g., a queue has items, or an order has finished processing) without burning CPU cycles in a spin loop.

A `std::condition_variable` puts the thread to sleep in the operating system scheduler until another thread wakes it up.

#### Preventing Spurious Wakeups
Operating systems can occasionally wake waiting threads even if no signal was sent (spurious wakeups). We always pass a **predicate lambda** to `wait()`:

```cpp
// In ConcurrentQueue<T>::pop():
std::unique_lock<std::mutex> lock(mutex_);
not_empty_.wait(lock, [this] { 
  return closed_.load(std::memory_order_relaxed) || !queue_.empty(); 
});
// When wait returns, the lock is held AND the condition is guaranteed to be true!
```

When a new item is added:
```cpp
queue_.push(std::move(item));
lock.unlock(); // Release lock before notifying to minimize thread contention
not_empty_.notify_one(); // Wake up one sleeping consumer
```

---

### C. Atomics: `std::atomic<T>` and Memory Ordering

Atomic operations modify a memory location without a lock, implemented directly in CPU hardware instructions (e.g., `LOCK XADD` on x86-64).

#### Monotonic Sequence IDs (The Sequencer)
In `ExchangeGateway`, incoming orders from different threads receive an atomic sequence number:
```cpp
std::atomic<uint64_t> next_seq_{1};

// Thread-safe, lock-free sequence assignment:
uint64_t seq = next_seq_.fetch_add(1, std::memory_order_relaxed);
```

#### Memory Orderings:
* `std::memory_order_relaxed`: Guarantees atomicity of the variable itself, but places no synchronization order on surrounding memory. Perfect for independent metrics (`total_submits`, `total_fills`).
* `std::memory_order_release` / `std::memory_order_acquire`: Creates a happens-before relationship between threads without the cost of a full memory fence.
  * In `ExchangeGateway`: The worker thread stores `last_processed_seq_.store(seq, std::memory_order_release)`.
  * The waiting client thread loads `last_processed_seq_.load(std::memory_order_acquire)`. All changes made before the release are guaranteed visible to the client after the acquire!

---

## 3. The Asynchronous Exchange Architecture

```
[Client Thread 1] ──┐
[Client Thread 2] ──┼──► submit_order() ──► Atomic Sequencer (next_seq_++)
[Client Thread 3] ──┘                             │
                                                  ▼
                                      [Concurrent Ingress Queue]
                                                  │
                                                  ▼ (Worker pulls sequenced requests)
                                       [Matching Worker Thread]
                                                  │
                                                  ├──► Matches against ConcurrentOrderBook
                                                  ├──► Updates last_processed_seq_ & notifies CV
                                                  ▼
                                      [Concurrent Egress Queue] ──► Fills & Market Data
```

1. **Ingress**: Multiple client threads call `gateway.submit_order(...)`. Each order gets an atomic `seq_id` and is pushed into `ingress_queue_`.
2. **Matching Worker**: A dedicated `std::thread` runs `run_worker()`. It pops orders in strict arrival sequence and matches them against `ConcurrentOrderBook`.
3. **Synchronization Barrier**: Client threads can optionally call `gateway.wait_until_processed(seq_id)`. The thread sleeps on `seq_cv_` until the matching engine has processed all orders up to that sequence.
4. **Egress**: As trades execute, fills and cancel confirmations are pushed into `egress_queue_` for consumption by market data subscriber threads.
5. **Non-Blocking Resiliency**: In `run_worker()`, egress events use `try_push()` so a slow downstream consumer can never stall the exchange matching engine.

---

## 4. Advanced High-Frequency C++ Engine Patterns

The repository also includes four low-latency C++ components designed for maximum performance:

### A. Vectorized Feature Extraction Engine (`cpp/include/lobx/features.hpp`)
* **Why**: Flamegraphs showed that Python feature extraction took **26%** of total evaluation runtime.
* **How**: Precomputes prefix sums for trades to calculate trade-flow imbalance in $O(\log M)$ using binary search (`std::upper_bound`). Vectorized snapshot depth traversal calculates spread, imbalance at depth 1 and 5, and rolling mid returns in a single contiguous memory pass.

### B. Zero-Allocation Slab Memory Pool (`cpp/include/lobx/memory_pool.hpp` & `slab_order_book.hpp`)
* **Why**: Standard `std::list` performs dynamic heap allocation (`new` / `malloc`) on every order insert and cancel, leading to memory fragmentation and non-deterministic latency spikes.
* **How**: `FixedPool<T, Capacity>` pre-allocates an intrusive freelist buffer. `SlabOrderBook` uses intrusive doubly-linked list pointers (`prev`, `next`) inside each resting order node. **Zero dynamic heap allocations** occur in the critical matching path.

### C. Lock-Free SPSC Ring Buffer with Cache-Line Padding (`cpp/include/lobx/spsc_ring_buffer.hpp`)
* **Why**: Mutexes cause OS thread context switches costing 1,000–3,000 ns.
* **How**: A lock-free single-producer single-consumer circular queue using `std::atomic<uint64_t>` with `memory_order_acquire` and `memory_order_release`.
* **Eliminating False Sharing**: Both `head_` and `tail_` are decorated with `alignas(64)` to place them on separate CPU L1/L2 cache lines, preventing cache invalidation across cores. Bitwise masking (`head & (Capacity - 1)`) executes in 1 clock cycle.

### D. Vectorized Batch Submission (`submit_many`)
* **Why**: Crossing the Python/C++ pybind11 boundary order-by-order introduces function call overhead.
* **How**: `OrderBook::submit_many` and `ConcurrentOrderBook::submit_many` process vectors of orders in one call. In `ConcurrentOrderBook`, the mutex is acquired once for the entire batch.

---

## 5. Benchmark Performance

Run the multi-threaded benchmark:
```bash
./build/cpp/lobx_concurrent_bench 200000 4
```

On a standard multi-core CPU:
* **Ingress throughput**: Over **5,000,000 orders/sec** across 4 concurrent client threads.
* **Worker matching throughput**: Over **1,400,000 orders/sec** end-to-end.
* **Average latency**: **~680–740 ns/order**.

---

## 6. Practice Exercises

1. **Implement Order Rejection on Duplicate ID in the Gateway**:
   Catch duplicate order IDs in `ExchangeGateway::run_worker()` and verify that a `GatewayEventType::Rejected` event is emitted on the egress queue without crashing the worker thread.
2. **Add a Ring-Buffer Subscriber**:
   Replace `egress_queue_` with a fixed-size lock-free `SpscRingBuffer` where consumer threads can read market data without mutual exclusion.
3. **Compare Slab Allocator vs. Standard Engine Latency**:
   Write a microbenchmark comparing `OrderBook` and `SlabOrderBook` on 1,000,000 orders to measure the allocation overhead of `std::list`.
4. **Batch Feature Extraction**:
   Use `extract_features` from Python via `_native.extract_features` and compare its runtime against `src/lobx/ml/features.py`.
