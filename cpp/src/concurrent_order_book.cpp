#include "lobx/concurrent_order_book.hpp"

#include <algorithm>
#include <mutex>

namespace lobx {

std::vector<Fill> ConcurrentOrderBook::submit(const Order& order) {
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  Timestamp ts = std::max(order.ts, last_ts_);
  last_ts_ = ts;
  Order adj = order;
  adj.ts = ts;

  total_submits_.fetch_add(1, std::memory_order_relaxed);
  std::vector<Fill> fills = book_.submit(adj);
  if (!fills.empty()) {
    total_fills_.fetch_add(fills.size(), std::memory_order_relaxed);
    uint64_t vol = 0;
    for (const auto& f : fills) vol += f.qty;
    total_traded_volume_.fetch_add(vol, std::memory_order_relaxed);
  }
  return fills;
}

bool ConcurrentOrderBook::cancel(OrderId id, Timestamp ts) {
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  Timestamp effective_ts = std::max(ts, last_ts_);
  last_ts_ = effective_ts;
  bool ok = book_.cancel(id, effective_ts);
  if (ok) total_cancels_.fetch_add(1, std::memory_order_relaxed);
  return ok;
}

Snapshot ConcurrentOrderBook::snapshot(std::size_t depth) const {
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  return book_.snapshot(depth);
}

std::optional<Price> ConcurrentOrderBook::best_bid() const {
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  return book_.best_bid();
}

std::optional<Price> ConcurrentOrderBook::best_ask() const {
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  return book_.best_ask();
}

std::size_t ConcurrentOrderBook::resting_orders() const {
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  return book_.resting_orders();
}

}  // namespace lobx
