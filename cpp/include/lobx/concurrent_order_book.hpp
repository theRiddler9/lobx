// A thread-safe, reader-writer locked order book wrapper using std::shared_mutex
// and lock-free atomic telemetry counters.
//
// CONCURRENCY DESIGN RATIONALE:
// In limit order book architectures, query operations (such as sampling book depth,
// checking best bid/ask, and streaming level snapshots to market makers or feature pipelines)
// occur at much higher frequencies than order insertions and cancellations.
//
// 1. std::shared_mutex vs std::mutex:
//    Using a traditional std::mutex would force all concurrent reader threads (e.g. 16 market
//    data subscribers) to serialize behind each other, causing massive lock contention and cache
//    invalidations. std::shared_mutex enables multiple readers to concurrently hold shared locks
//    (std::shared_lock) without contention, while writers acquire an exclusive lock
//    (std::unique_lock) guaranteeing complete data isolation during matching and book mutation.
//
// 2. Memory Ordering on Telemetry Counters:
//    Counters such as total_submits_, total_cancels_, and total_fills_ are updated with
//    std::memory_order_relaxed. Why relaxed? Because these counters track statistical metrics
//    independently. They do not guard any shared memory or establish synchronization barriers
//    between threads (unlike flags or pointers), so relaxed ordering provides the highest
//    possible instruction-level throughput without unnecessary CPU store buffers or memory fences.
#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <shared_mutex>
#include <vector>

#include "lobx/order_book.hpp"
#include "lobx/types.hpp"

namespace lobx {

class ConcurrentOrderBook {
 public:
  ConcurrentOrderBook() = default;
  ~ConcurrentOrderBook() = default;

  ConcurrentOrderBook(const ConcurrentOrderBook&) = delete;
  ConcurrentOrderBook& operator=(const ConcurrentOrderBook&) = delete;

  // Mutating operations: acquire exclusive lock (std::unique_lock) so no reader
  // or other writer observes intermediate book states or partial fills.
  std::vector<Fill> submit(const Order& order);
  std::vector<Fill> submit_many(const std::vector<Order>& orders);
  bool cancel(OrderId id, Timestamp ts);

  // Read-only queries: acquire shared lock (std::shared_lock) allowing concurrent reads.
  [[nodiscard]] Snapshot snapshot(std::size_t depth) const;
  [[nodiscard]] std::optional<Price> best_bid() const;
  [[nodiscard]] std::optional<Price> best_ask() const;
  [[nodiscard]] std::size_t resting_orders() const;

  // Lock-free performance counters.
  // std::memory_order_relaxed is sufficient here because these counters are purely diagnostic
  // and do not synchronize or order other memory accesses.
  [[nodiscard]] uint64_t total_submits() const noexcept {
    return total_submits_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t total_cancels() const noexcept {
    return total_cancels_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t total_fills() const noexcept {
    return total_fills_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t total_traded_volume() const noexcept {
    return total_traded_volume_.load(std::memory_order_relaxed);
  }

 private:
  OrderBook book_;
  mutable std::shared_mutex rw_mutex_;
  Timestamp last_ts_{0};
  std::atomic<uint64_t> total_submits_{0};
  std::atomic<uint64_t> total_cancels_{0};
  std::atomic<uint64_t> total_fills_{0};
  std::atomic<uint64_t> total_traded_volume_{0};
};

}  // namespace lobx
