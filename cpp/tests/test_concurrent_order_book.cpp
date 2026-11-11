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
  std::atomic<uint64_t> total_reader_checks{0};
  std::vector<std::thread> writers;
  std::vector<std::thread> readers;

  // Reader threads: concurrently query the order book and verify critical invariants
  for (int r = 0; r < kReaders; ++r) {
    readers.emplace_back([&]() {
      while (!stop_readers.load(std::memory_order_relaxed)) {
        auto snap = book.snapshot(10);
        auto b_bid = book.best_bid();
        auto b_ask = book.best_ask();

        // Invariant 1: Book is NEVER crossed (best bid strictly less than best ask)
        if (!snap.bids.empty() && !snap.asks.empty()) {
          CHECK(snap.bids.front().first < snap.asks.front().first);
        }

        // Invariant 2: Bids strictly monotonic descending by price, quantities positive
        for (std::size_t i = 0; i < snap.bids.size(); ++i) {
          CHECK(snap.bids[i].second > 0);
          if (i > 0) {
            CHECK(snap.bids[i - 1].first > snap.bids[i].first);
          }
        }

        // Invariant 3: Asks strictly monotonic ascending by price, quantities positive
        for (std::size_t i = 0; i < snap.asks.size(); ++i) {
          CHECK(snap.asks[i].second > 0);
          if (i > 0) {
            CHECK(snap.asks[i - 1].first < snap.asks[i].first);
          }
        }

        // Invariant 4: Best bid and best ask agree with top of book
        if (b_bid && !snap.bids.empty()) {
          // Note: b_bid could be newer than snapshot, but neither can be crossed
          if (b_ask) {
            CHECK(*b_bid < *b_ask);
          }
        }

        total_reader_checks.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Writer threads: interleave bids, asks, crossing orders, and cancels
  std::atomic<Timestamp> global_ts{10000};
  for (int w = 0; w < kWriters; ++w) {
    writers.emplace_back([&, w]() {
      for (int i = 0; i < kOrdersPerWriter; ++i) {
        OrderId id = static_cast<OrderId>(w * 100000 + i + 1);
        Side side = (i % 2 == 0) ? Side::Buy : Side::Sell;
        Price px = (side == Side::Buy) ? (9990 - (i % 15) * 2) : (10010 + (i % 15) * 2);

        // Occasionally cross the spread to trigger real matching and partial fills
        if (i % 50 == 0) {
          px = (side == Side::Buy) ? 10020 : 9980;
        }

        Timestamp ts1 = global_ts.fetch_add(10, std::memory_order_relaxed);
        Order o{id, w + 1, side, OrderType::Limit, px, 10, ts1};
        book.submit(o);

        if (i % 7 == 0) {
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
  CHECK(total_reader_checks.load() > 100);
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
