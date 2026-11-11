#include "lobx/sim_engine.hpp"

#include <algorithm>

namespace lobx {

CppMarket::CppMarket(SimConfig cfg) : cfg_(cfg), rng_(cfg.seed), last_mid_(cfg.initial_price) {
  // Seed initial agent wakes
  queue_.push({1'000'000, 1, 100});  // Market Maker at 1ms
  for (int i = 0; i < cfg_.num_noise_agents; ++i) {
    schedule_next(0, 10 + i, cfg_.noise_lambda);
  }
  schedule_next(2, 200, 1.0);  // Momentum agent at 1 wake/sec
}

void CppMarket::schedule_next(int type, int id, double lambda_sec) {
  double eff_lambda = lambda_sec;
  if (type == 0 && cfg_.volume_u_shape > 0.0 && cfg_.duration_ns > 0) {
    double t_mid = cfg_.duration_ns / 2.0;
    double u = (static_cast<double>(current_ts_) - t_mid) / t_mid;
    double scale = 1.0 + cfg_.volume_u_shape * (u * u);
    eff_lambda *= scale;
  }
  std::exponential_distribution<double> dist(eff_lambda);
  auto dt_ns = static_cast<Timestamp>(dist(rng_) * 1e9);
  queue_.push({current_ts_ + std::max<Timestamp>(100'000, dt_ns), type, id});
}

Price CppMarket::current_mid() const {
  auto bb = book_.best_bid(), ba = book_.best_ask();
  if (bb && ba) return (*bb + *ba) / 2;
  return bb ? *bb : (ba ? *ba : last_mid_);
}

void CppMarket::handle_noise(int id) {
  std::uniform_int_distribution<int> side_dist(0, 1), type_dist(0, 9), offset(-3, 3);
  Side side = side_dist(rng_) == 0 ? Side::Buy : Side::Sell;
  OrderType otype = type_dist(rng_) < 7 ? OrderType::Limit : OrderType::Market;
  Price px = current_mid() + offset(rng_);
  if (px <= 0) px = 1;

  Order o{next_order_id_++,
          id,
          side,
          otype,
          otype == OrderType::Market ? std::nullopt : std::make_optional(px),
          cfg_.trade_size,
          current_ts_};
  auto fills = book_.submit(o);
  total_orders_++;
  trades_.insert(trades_.end(), fills.begin(), fills.end());
  schedule_next(0, id, cfg_.noise_lambda);
}

void CppMarket::handle_mm(int id) {
  if (mm_bid_ > 0) book_.cancel(mm_bid_, current_ts_);
  if (mm_ask_ > 0) book_.cancel(mm_ask_, current_ts_);

  Price mid = current_mid();
  Price reservation = mid - (mm_inv_ / cfg_.trade_size) * cfg_.mm_skew_per_qty;
  Price bid_px = std::max<Price>(1, reservation - cfg_.mm_half_spread);
  Price ask_px = std::max<Price>(bid_px + 1, reservation + cfg_.mm_half_spread);

  mm_bid_ = next_order_id_++;
  auto f1 = book_.submit(
      {mm_bid_, id, Side::Buy, OrderType::Limit, bid_px, cfg_.trade_size, current_ts_});
  mm_ask_ = next_order_id_++;
  auto f2 = book_.submit(
      {mm_ask_, id, Side::Sell, OrderType::Limit, ask_px, cfg_.trade_size, current_ts_});
  total_orders_ += 2;

  for (const auto& f : f1) mm_inv_ += f.qty;
  for (const auto& f : f2) mm_inv_ -= f.qty;
  trades_.insert(trades_.end(), f1.begin(), f1.end());
  trades_.insert(trades_.end(), f2.begin(), f2.end());
  schedule_next(1, id, 10.0);  // MM wakes 10 times a second
}

void CppMarket::handle_momentum(int id) {
  Price mid = current_mid();
  if (mid != last_mid_) {
    Side side = mid > last_mid_ ? Side::Buy : Side::Sell;
    Order o{next_order_id_++, id,         side, OrderType::Market, std::nullopt,
            cfg_.trade_size,  current_ts_};
    auto fills = book_.submit(o);
    total_orders_++;
    trades_.insert(trades_.end(), fills.begin(), fills.end());
  }
  last_mid_ = mid;
  schedule_next(2, id, 1.0);
}

bool CppMarket::step() {
  if (queue_.empty()) return false;
  WakeEvent ev = queue_.top();
  queue_.pop();
  current_ts_ = ev.ts;

  if (ev.type == 0)
    handle_noise(ev.id);
  else if (ev.type == 1)
    handle_mm(ev.id);
  else if (ev.type == 2)
    handle_momentum(ev.id);
  return true;
}

void CppMarket::run() {
  while (current_ts_ < cfg_.duration_ns && step()) {
  }
}

}  // namespace lobx
