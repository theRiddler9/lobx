# Advanced Exchange Expansion Implementation Plan: Institutional Orders, C++ Simulator, Binary ITCH Log, and RL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 high-value exchange subsystems in order: (1) Advanced Institutional Order Types (IOC, FOK, Post-Only), (2) Ultra-Fast C++ Simulation Engine & Agents (`CppMarket`), (3) Binary Event Journaling & Zero-Copy Mmap Replay, and (4) Reinforcement Learning Execution Environment.

**Architecture:** Each phase adds production-grade exchange functionality in compact, modern C++17 and Python, maximizing efficiency, maintaining strict invariants, and providing comprehensive test coverage.

**Tech Stack:** C++17, pybind11, CMake 3.18+, GCC 15.2 (Ninja), Python 3.11+.

---

### Phase 1: Advanced Institutional Order Types (IOC, FOK, Post-Only)

#### Task 1: Institutional Orders in C++ Engine
**Files:**
- Modify: `cpp/include/lobx/types.hpp`
- Modify: `cpp/include/lobx/order_book.hpp`
- Modify: `cpp/src/order_book.cpp`
- Create: `cpp/tests/test_order_types.cpp`
- Modify: `CMakeLists.txt`
- Modify: `cpp/src/bindings.cpp`
- Modify: `src/lobx/_native.pyi`

**Interfaces:**
- Produces: `OrderType::IOC`, `OrderType::FOK`, `OrderType::PostOnly` with exact execution semantics:
  - `IOC`: Matches available crossing depth, cancels unfilled remainder immediately without resting.
  - `FOK`: All-or-nothing: matches full quantity or aborts completely with 0 fills.
  - `PostOnly`: Only rests at limit price; rejected with 0 fills if it would cross and take liquidity.

- [ ] **Step 1: Write failing test `cpp/tests/test_order_types.cpp`**
- [ ] **Step 2: Add test to `CMakeLists.txt` and verify build fails**
- [ ] **Step 3: Implement IOC, FOK, and PostOnly in `cpp/src/order_book.cpp`**
- [ ] **Step 4: Build and run test to verify all order types pass**
- [ ] **Step 5: Bind new enum values in `cpp/src/bindings.cpp` and update `_native.pyi`**
- [ ] **Step 6: Commit Phase 1**

---

### Phase 2: Ultra-Fast C++ Simulation Engine & Agents (`CppMarket`)

#### Task 2: C++ Discrete-Event Market and Agent Population
**Files:**
- Create: `cpp/include/lobx/sim_engine.hpp`
- Create: `cpp/src/sim_engine.cpp`
- Create: `cpp/tests/test_cpp_sim.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `CppMarket` with native agents (`CppNoiseTrader`, `CppMarketMaker`, `CppMomentumTrader`):
  - Simulates 300+ seconds in single-digit milliseconds.
  - Deterministic priority queue event scheduling.

- [ ] **Step 1: Write failing test `cpp/tests/test_cpp_sim.cpp`**
- [ ] **Step 2: Add test to `CMakeLists.txt` and verify build fails**
- [ ] **Step 3: Implement `sim_engine.hpp` and `sim_engine.cpp`**
- [ ] **Step 4: Build and run test to verify deterministic simulation**
- [ ] **Step 5: Commit Phase 2**

---

### Phase 3: Binary Event Journaling & Zero-Copy Mmap Replay

#### Task 3: Compact Binary ITCH Logger and Fast Replay
**Files:**
- Create: `cpp/include/lobx/binary_log.hpp`
- Create: `cpp/src/binary_log.cpp`
- Create: `cpp/tests/test_binary_log.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `BinaryEventLogger` and `BinaryReplay`:
  - 24-byte packed binary records for submit, cancel, and fill.
  - Fast replay iterating over memory buffers with zero parsing allocations.

- [ ] **Step 1: Write failing test `cpp/tests/test_binary_log.cpp`**
- [ ] **Step 2: Add test to `CMakeLists.txt` and verify build fails**
- [ ] **Step 3: Implement `binary_log.hpp` and `binary_log.cpp`**
- [ ] **Step 4: Build and run test to verify exact binary serialization and replay**
- [ ] **Step 5: Commit Phase 3**

---

### Phase 4: Reinforcement Learning (RL) Execution Environment

#### Task 4: Gymnasium Execution Environment
**Files:**
- Create: `src/lobx/rl/__init__.py`
- Create: `src/lobx/rl/environment.py`
- Create: `tests/test_rl_env.py`

**Interfaces:**
- Produces: `ExecutionEnv`:
  - Standard `reset()` and `step(action)` API.
  - State: depth levels, position, time remaining, spread.
  - Reward: Implementation shortfall minimization with deadline completion penalty.

- [ ] **Step 1: Write failing Python test `tests/test_rl_env.py`**
- [ ] **Step 2: Implement `src/lobx/rl/environment.py`**
- [ ] **Step 3: Run test to verify step transitions and shortfall rewards**
- [ ] **Step 4: Commit Phase 4**
