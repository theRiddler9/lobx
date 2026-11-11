#pragma once

#include <queue>
#include <random>
#include <vector>

#include "lobx/order_book.hpp"

namespace lobx {

struct SimConfig {
  std::uint64_t seed = 42;
  Timestamp duration_ns = 60'000'000'000ULL;
  Price initial_price = 10'000;
  Qty trade_size = 100;
  int num_noise_agents = 5;
  double noise_lambda = 2.0;  // arrivals per second
  Price mm_half_spread = 2;
  Price mm_skew_per_qty = 1;
  double volume_u_shape = 0.0;
};

class CppMarket {
 public:
  explicit CppMarket(SimConfig config);

  void run();
  bool step();

  const OrderBook& book() const { return book_; }
  Timestamp current_time() const { return current_ts_; }
  const std::vector<Fill>& trades() const { return trades_; }
  std::size_t total_orders_submitted() const { return total_orders_; }

 private:
  struct WakeEvent {
    Timestamp ts;
    int type;  // 0: noise, 1: mm, 2: momentum
    int id;
    bool operator>(const WakeEvent& o) const { return ts > o.ts; }
  };

  SimConfig cfg_;
  Timestamp current_ts_ = 0;
  OrderId next_order_id_ = 1;
  std::size_t total_orders_ = 0;
  OrderBook book_;
  std::vector<Fill> trades_;
  std::priority_queue<WakeEvent, std::vector<WakeEvent>, std::greater<WakeEvent>> queue_;
  std::mt19937_64 rng_;

  Qty mm_inv_ = 0;
  OrderId mm_bid_ = 0, mm_ask_ = 0;
  Price last_mid_;

  void handle_noise(int id);
  void handle_mm(int id);
  void handle_momentum(int id);
  Price current_mid() const;
  void schedule_next(int type, int id, double lambda_sec);
};

}  // namespace lobx
