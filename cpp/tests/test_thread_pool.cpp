// Unit tests for ThreadPool and multi-seed parallel simulation determinism.
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "lobx/sim_engine.hpp"
#include "lobx/thread_pool.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      ++g_failures;                                                              \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
    }                                                                            \
  } while (0)

using namespace lobx;

void test_basic_thread_pool_tasks() {
  ThreadPool pool(4);
  CHECK(pool.num_workers() == 4);

  std::vector<std::future<int>> results;
  for (int i = 0; i < 20; ++i) {
    results.push_back(pool.enqueue([i]() { return i * i; }));
  }

  for (int i = 0; i < 20; ++i) {
    CHECK(results[i].get() == i * i);
  }
}

struct SeedResult {
  uint64_t seed{0};
  std::size_t total_orders{0};
  std::size_t total_trades{0};
  std::optional<Price> best_bid;
  std::optional<Price> best_ask;
  std::vector<Fill> trades;

  bool operator==(const SeedResult& o) const {
    if (seed != o.seed || total_orders != o.total_orders || total_trades != o.total_trades ||
        best_bid != o.best_bid || best_ask != o.best_ask || trades.size() != o.trades.size()) {
      return false;
    }
    for (std::size_t i = 0; i < trades.size(); ++i) {
      if (trades[i].maker_id != o.trades[i].maker_id ||
          trades[i].taker_id != o.trades[i].taker_id || trades[i].price != o.trades[i].price ||
          trades[i].qty != o.trades[i].qty || trades[i].ts != o.trades[i].ts) {
        return false;
      }
    }
    return true;
  }
};

SeedResult run_simulation_seed(uint64_t seed) {
  SimConfig cfg;
  cfg.seed = seed;
  cfg.duration_ns = 5'000'000'000ULL;  // 5 simulated seconds
  CppMarket market(cfg);
  market.run();

  SeedResult res;
  res.seed = seed;
  res.total_orders = market.total_orders_submitted();
  res.total_trades = market.trades().size();
  res.best_bid = market.book().best_bid();
  res.best_ask = market.book().best_ask();
  res.trades = market.trades();
  return res;
}

void test_parallel_seeds_determinism() {
  constexpr int kNumSeeds = 10;
  std::vector<uint64_t> seeds;
  for (int i = 0; i < kNumSeeds; ++i) {
    seeds.push_back(100 + i);
  }

  // 1. Run seeds serially (ground truth)
  std::vector<SeedResult> serial_results;
  serial_results.reserve(kNumSeeds);
  for (uint64_t s : seeds) {
    serial_results.push_back(run_simulation_seed(s));
  }

  // 2. Run seeds in parallel across 4 worker threads
  ThreadPool pool(4);
  std::vector<std::future<SeedResult>> parallel_futures;
  parallel_futures.reserve(kNumSeeds);
  for (uint64_t s : seeds) {
    parallel_futures.push_back(pool.enqueue([s]() { return run_simulation_seed(s); }));
  }

  std::vector<SeedResult> parallel_results;
  parallel_results.reserve(kNumSeeds);
  for (auto& fut : parallel_futures) {
    parallel_results.push_back(fut.get());
  }

  // 3. Verify exact byte-level determinism between serial and parallel runs
  for (std::size_t i = 0; i < kNumSeeds; ++i) {
    CHECK(parallel_results[i] == serial_results[i]);
    CHECK(parallel_results[i].total_orders > 0);
    CHECK(parallel_results[i].total_trades > 0);
  }
}

}  // namespace

int main() {
  test_basic_thread_pool_tasks();
  test_parallel_seeds_determinism();

  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "all ThreadPool and parallel simulation determinism tests passed\n";
  return 0;
}
