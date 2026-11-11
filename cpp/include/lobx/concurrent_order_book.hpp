// A thread-safe order book wrapper using shared_mutex and atomic metrics.
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

  // Exclusive lock for mutating operations
  std::vector<Fill> submit(const Order& order);
  std::vector<Fill> submit_many(const std::vector<Order>& orders);
  bool cancel(OrderId id, Timestamp ts);

  // Shared lock for concurrent read-only queries
  [[nodiscard]] Snapshot snapshot(std::size_t depth) const;
  [[nodiscard]] std::optional<Price> best_bid() const;
  [[nodiscard]] std::optional<Price> best_ask() const;
  [[nodiscard]] std::size_t resting_orders() const;

  // Atomic performance counters (lock-free)
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
