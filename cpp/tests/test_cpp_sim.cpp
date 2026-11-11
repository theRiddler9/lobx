#include <cassert>
#include <chrono>
#include <iostream>

#include "lobx/sim_engine.hpp"

using namespace lobx;

void test_sim_run_and_determinism() {
  SimConfig cfg;
  cfg.seed = 12345;
  cfg.duration_ns = 5'000'000'000ULL;  // 5 seconds

  // Run 1
  CppMarket sim1(cfg);
  sim1.run();
  assert(sim1.total_orders_submitted() > 50);
  assert(sim1.current_time() >= cfg.duration_ns);

  // Run 2 (same seed)
  CppMarket sim2(cfg);
  sim2.run();

  // Determinism check
  assert(sim1.total_orders_submitted() == sim2.total_orders_submitted());
  assert(sim1.trades().size() == sim2.trades().size());
  assert(sim1.book().best_bid() == sim2.book().best_bid());
  assert(sim1.book().best_ask() == sim2.book().best_ask());

  for (std::size_t i = 0; i < sim1.trades().size(); ++i) {
    assert(sim1.trades()[i].price == sim2.trades()[i].price);
    assert(sim1.trades()[i].qty == sim2.trades()[i].qty);
    assert(sim1.trades()[i].ts == sim2.trades()[i].ts);
  }
  std::cout << "test_sim_run_and_determinism passed (" << sim1.total_orders_submitted()
            << " orders, " << sim1.trades().size() << " trades)\n";
}

void test_sim_performance() {
  SimConfig cfg;
  cfg.seed = 999;
  cfg.duration_ns = 30'000'000'000ULL;  // 30 seconds

  auto t0 = std::chrono::steady_clock::now();
  CppMarket sim(cfg);
  sim.run();
  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
          .count();

  std::cout << "Simulated 30s (" << sim.total_orders_submitted() << " orders) in " << elapsed_ms
            << " ms\n";
  assert(elapsed_ms < 500);  // Must be ultra-fast (expected < 15ms)
  std::cout << "test_sim_performance passed\n";
}

int main() {
  test_sim_run_and_determinism();
  test_sim_performance();
  std::cout << "All CppMarket sim tests passed!\n";
  return 0;
}
