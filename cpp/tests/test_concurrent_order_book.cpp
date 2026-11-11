// Unit tests for ConcurrentOrderBook.
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "lobx/concurrent_order_book.hpp"

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

void test_basic_concurrent_book() {
  ConcurrentOrderBook book;
  CHECK(!book.best_bid().has_value());
  CHECK(!book.best_ask().has_value());
  CHECK(book.resting_orders() == 0);

  Order buy1{1, 10, Side::Buy, OrderType::Limit, 100, 50, 1000};
  auto fills1 = book.submit(buy1);
  CHECK(fills1.empty());
  CHECK(book.best_bid() == 100);
  CHECK(book.resting_orders() == 1);
  CHECK(book.total_submits() == 1);

  // Crossing sell order
  Order sell1{2, 20, Side::Sell, OrderType::Limit, 100, 20, 2000};
  auto fills2 = book.submit(sell1);
  CHECK(fills2.size() == 1);
  CHECK(fills2[0].qty == 20);
  CHECK(fills2[0].price == 100);
  CHECK(book.total_fills() == 1);
  CHECK(book.total_traded_volume() == 20);

  // Cancel remaining 30 shares
  CHECK(book.cancel(1, 3000));
  CHECK(book.resting_orders() == 0);
  CHECK(book.total_cancels() == 1);
}

void test_multithreaded_read_write_stress() {
  ConcurrentOrderBook book;
  constexpr int kWriters = 4;
  constexpr int kReaders = 4;
  constexpr int kOrdersPerWriter = 2500;
  std::atomic<bool> stop_readers{false};
  std::vector<std::thread> writers;
  std::vector<std::thread> readers;

  for (int r = 0; r < kReaders; ++r) {
    readers.emplace_back([&]() {
      while (!stop_readers.load(std::memory_order_relaxed)) {
        auto snap = book.snapshot(5);
        if (!snap.bids.empty() && !snap.asks.empty()) {
          CHECK(snap.bids.front().first <
                snap.asks.front().first);  // Invariant: book is never crossed
        }
      }
    });
  }

  std::atomic<Timestamp> global_ts{10000};
  for (int w = 0; w < kWriters; ++w) {
    writers.emplace_back([&, w]() {
      for (int i = 0; i < kOrdersPerWriter; ++i) {
        OrderId id = static_cast<OrderId>(w * 100000 + i + 1);
        Price bid_px = 10000 - (i % 20) * 5;
        Timestamp ts1 = global_ts.fetch_add(10, std::memory_order_relaxed);
        Order buy{id, w + 1, Side::Buy, OrderType::Limit, bid_px, 10, ts1};
        book.submit(buy);

        if (i % 5 == 0) {
          Timestamp ts2 = global_ts.fetch_add(10, std::memory_order_relaxed);
          book.cancel(id, ts2);
        }
      }
    });
  }

  for (auto& t : writers) t.join();
  stop_readers.store(true, std::memory_order_relaxed);
  for (auto& t : readers) t.join();

  CHECK(book.total_submits() == kWriters * kOrdersPerWriter);
  CHECK(book.total_cancels() > 0);
}

}  // namespace

int main() {
  test_basic_concurrent_book();
  test_multithreaded_read_write_stress();
  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "all concurrent order book tests passed\n";
  return 0;
}
