// Benchmark comparing single-order submission vs batch submission (submit_many).
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "lobx/order_book.hpp"

namespace {

using namespace lobx;

void run_comparison(std::size_t total_orders, std::size_t batch_size) {
  std::cout << "Benchmarking single-order vs batch ingress (" << total_orders
            << " orders, batch size " << batch_size << ")...\n";

  std::vector<Order> orders;
  orders.reserve(total_orders);
  for (std::size_t i = 0; i < total_orders; ++i) {
    OrderId id = static_cast<OrderId>(i + 1);
    Side side = (i % 2 == 0) ? Side::Buy : Side::Sell;
    Price px = 10000 + ((i % 50) - 25);
    orders.push_back(Order{id, 1, side, OrderType::Limit, px, 10, static_cast<Timestamp>(id * 10)});
  }

  // 1. Single order loop
  OrderBook book1;
  auto t1 = std::chrono::steady_clock::now();
  for (const auto& o : orders) {
    book1.submit(o);
  }
  auto t2 = std::chrono::steady_clock::now();
  double single_sec = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e6;
  double single_rate = total_orders / single_sec;

  // 2. Batch submission in chunks
  OrderBook book2;
  auto t3 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < total_orders; i += batch_size) {
    std::size_t end_idx = std::min(i + batch_size, total_orders);
    std::vector<Order> chunk(orders.begin() + i, orders.begin() + end_idx);
    book2.submit_many(chunk);
  }
  auto t4 = std::chrono::steady_clock::now();
  double batch_sec = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1e6;
  double batch_rate = total_orders / batch_sec;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  Single-order:  " << single_sec << " s (" << single_rate << " orders/s)\n"
            << "  Batch API:     " << batch_sec << " s (" << batch_rate << " orders/s)\n"
            << "  Speedup:       " << (batch_rate / single_rate) << "x\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::size_t total = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 200000;
  std::size_t batch = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 500;
  run_comparison(total, batch);
  return 0;
}
