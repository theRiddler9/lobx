// Multi-threaded throughput benchmark for ExchangeGateway.
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "lobx/gateway.hpp"

namespace {

using namespace lobx;

void run_benchmark(std::size_t total_orders, int num_threads) {
  std::cout << "Benchmarking ExchangeGateway with " << total_orders << " orders across "
            << num_threads << " producer threads...\n";

  ExchangeGateway gateway;
  gateway.start();

  const std::size_t orders_per_thread = total_orders / num_threads;
  const std::size_t actual_total = orders_per_thread * num_threads;

  std::vector<std::thread> producers;
  producers.reserve(num_threads);

  auto start_time = std::chrono::steady_clock::now();

  for (int t = 0; t < num_threads; ++t) {
    producers.emplace_back([&, t]() {
      for (std::size_t i = 0; i < orders_per_thread; ++i) {
        OrderId id = static_cast<OrderId>(t * 100000000LL + i + 1);
        Side side = (i % 2 == 0) ? Side::Buy : Side::Sell;
        Price px = 10000 + ((i % 100) - 50);
        Order o{id, t + 1, side, OrderType::Limit, px, 10, static_cast<Timestamp>(id)};
        gateway.submit_order(o);
      }
    });
  }

  for (auto& p : producers) p.join();
  auto producers_done = std::chrono::steady_clock::now();

  // Wait until all submitted orders are matched
  gateway.wait_until_processed(actual_total, std::chrono::milliseconds(30000));
  auto all_matched = std::chrono::steady_clock::now();

  gateway.stop();

  auto ingress_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(producers_done - start_time).count();
  auto total_us =
      std::chrono::duration_cast<std::chrono::microseconds>(all_matched - start_time).count();

  double total_sec = total_us / 1e6;
  double throughput = actual_total / total_sec;
  double avg_ns = (total_us * 1000.0) / actual_total;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Completed in " << total_sec << " s\n"
            << "  Ingress rate:      " << (actual_total / (ingress_ms / 1e6)) << " orders/s\n"
            << "  Total throughput:  " << throughput << " orders/s\n"
            << "  Average latency:   " << avg_ns << " ns/order\n"
            << "  Trades published:  " << gateway.total_trades_published() << "\n"
            << "  Traded volume:     " << gateway.book().total_traded_volume() << " shares\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::size_t total_orders = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 500000;
  int num_threads = (argc > 2) ? std::atoi(argv[2]) : 4;
  if (num_threads < 1) num_threads = 1;
  run_benchmark(total_orders, num_threads);
  return 0;
}
