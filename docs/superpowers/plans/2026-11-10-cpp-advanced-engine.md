# Advanced C++ Engine Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 high-impact C++ architectural enhancements in order: (1) Vectorized Feature Extraction Engine, (2) Zero-Allocation Memory Pool (Slab Allocator) for Orders, (3) Lock-Free Cache-Aligned SPSC Ring Buffer, and (4) Vectorized Batch Ingress API (`submit_many`).

**Architecture:** Each phase introduces a focused, high-performance C++ subsystem adhering to modern C++17, compact code (minimal lines, maximum efficiency), strict mathematical parity with Python contracts, and zero heap allocation in critical matching paths.

**Tech Stack:** C++17, pybind11, CMake 3.18+, GCC 15.2 (Ninja), Python 3.11+.

**Spec:** User instruction to implement all 4 C++ improvement areas phase-wise with maximized efficiency and compact, clean code.

## Global Constraints
- Prices are integer ticks (`int64_t`); timestamps are nanoseconds (`int64_t`).
- Strict RAII memory safety and cache-line awareness (`alignas(64)`).
- Zero heap allocation during hot matching paths in the memory pool.
- Full test coverage with macro-based assertions (`CHECK`) and CTest integration.

---

### Phase 1: High-Performance C++ Feature Extraction Engine

#### Task 1: C++ Feature Engine (`features.hpp`, `features.cpp`)
**Files:**
- Create: `cpp/include/lobx/features.hpp`
- Create: `cpp/src/features.cpp`
- Create: `cpp/tests/test_features.cpp`
- Modify: `CMakeLists.txt`
- Modify: `cpp/src/bindings.cpp`
- Modify: `src/lobx/_native.pyi`

**Interfaces:**
- Produces: `lobx::extract_features(const std::vector<Snapshot>& snapshots, const std::vector<Fill>& fills, ...)` -> flat 2D row-major array matching Python's feature schema (`spread`, `imbalance_1`, `imbalance_5`, `mid_return_1`, `mid_return_5`, `mid_return_25`, `trade_flow_1s`, `trade_flow_5s`).

- [ ] **Step 1: Write failing C++ test `cpp/tests/test_features.cpp`**
- [ ] **Step 2: Add target to `CMakeLists.txt` and verify build fails**
- [ ] **Step 3: Implement `cpp/include/lobx/features.hpp` and `cpp/src/features.cpp`**
- [ ] **Step 4: Build and run test to verify it passes**
- [ ] **Step 5: Bind to Python in `bindings.cpp` and update `_native.pyi`**
- [ ] **Step 6: Commit Phase 1**

---

### Phase 2: Zero-Allocation Memory Pool (Slab Allocator) for Orders

#### Task 2: Slab Allocator and Intrusive Order Queue (`memory_pool.hpp`, `slab_order_book.hpp`)
**Files:**
- Create: `cpp/include/lobx/memory_pool.hpp`
- Create: `cpp/include/lobx/slab_order_book.hpp`
- Create: `cpp/src/slab_order_book.cpp`
- Create: `cpp/tests/test_slab_order_book.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:
  - `lobx::FixedPool<T, Capacity>`: Pre-allocated buffer with $O(1)$ allocate/deallocate via intrusive freelist.
  - `lobx::SlabOrderBook<MaxOrders>`: Matching engine that uses pre-allocated pool nodes instead of `std::list` heap allocations.

- [ ] **Step 1: Write failing C++ test `cpp/tests/test_slab_order_book.cpp`**
- [ ] **Step 2: Add target to `CMakeLists.txt` and verify build fails**
- [ ] **Step 3: Implement `memory_pool.hpp`, `slab_order_book.hpp`, and `slab_order_book.cpp`**
- [ ] **Step 4: Build and run test to verify zero heap allocation and matching parity**
- [ ] **Step 5: Commit Phase 2**

---

### Phase 3: Lock-Free Cache-Aligned SPSC Ring Buffer

#### Task 3: Lock-Free Ring Buffer (`spsc_ring_buffer.hpp`)
**Files:**
- Create: `cpp/include/lobx/spsc_ring_buffer.hpp`
- Create: `cpp/tests/test_spsc_ring_buffer.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `lobx::SpscRingBuffer<T, Capacity>` with:
  - `alignas(64)` on head and tail atomic cursors to prevent CPU false sharing.
  - Power-of-two capacity for bitwise indexing (`head & mask`).
  - Lock-free `push(T)` and `pop()` using acquire/release memory orders.

- [ ] **Step 1: Write failing C++ test `cpp/tests/test_spsc_ring_buffer.cpp`**
- [ ] **Step 2: Add target to `CMakeLists.txt` and verify build fails**
- [ ] **Step 3: Implement `cpp/include/lobx/spsc_ring_buffer.hpp`**
- [ ] **Step 4: Build and run test verifying multi-threaded correctness and ultra-low latency**
- [ ] **Step 5: Commit Phase 3**

---

### Phase 4: Vectorized Batch Ingress API (`submit_many`)

#### Task 4: Vectorized Batch Submission (`submit_many`)
**Files:**
- Modify: `cpp/include/lobx/order_book.hpp`
- Modify: `cpp/src/order_book.cpp`
- Modify: `cpp/src/bindings.cpp`
- Modify: `src/lobx/_native.pyi`
- Create: `cpp/bench/bench_batch_ingress.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:
  - `OrderBook::submit_many(const std::vector<Order>& orders) -> std::vector<Fill>`
  - pybind11 `submit_many` accepting contiguous order batches with a single GIL-release crossing.

- [ ] **Step 1: Add `submit_many` to `OrderBook` and test in `test_order_book.cpp`**
- [ ] **Step 2: Implement batch matching in `order_book.cpp`**
- [ ] **Step 3: Expose batch API in `bindings.cpp`**
- [ ] **Step 4: Create and run `bench_batch_ingress.cpp` measuring pybind11 speedup**
- [ ] **Step 5: Commit Phase 4**
