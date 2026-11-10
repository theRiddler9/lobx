#include "lobx/binary_log.hpp"
#include "lobx/order_book.hpp"

#include <cassert>
#include <chrono>
#include <iostream>

using namespace lobx;

void test_binary_logging_and_replay() {
  BinaryEventLogger logger;
  OrderBook book1;

  // 1. Submit resting buy
  Order o1{101, 1, Side::Buy, OrderType::Limit, 100, 50, 1000};
  logger.log_submit(o1);
  book1.submit(o1);

  // 2. Submit resting sell
  Order o2{102, 2, Side::Sell, OrderType::Limit, 105, 30, 2000};
  logger.log_submit(o2);
  book1.submit(o2);

  // 3. Submit partial cross
  Order o3{103, 3, Side::Buy, OrderType::Limit, 105, 10, 3000};
  logger.log_submit(o3);
  book1.submit(o3);

  // 4. Cancel
  logger.log_cancel(101, 4000);
  book1.cancel(101, 4000);

  // Replay binary events onto a fresh book
  OrderBook book2;
  BinaryReplayer replayer;
  std::size_t replayed = replayer.replay(logger.data(), logger.size(), book2);

  assert(replayed == logger.event_count());
  assert(book1.best_bid() == book2.best_bid());
  assert(book1.best_ask() == book2.best_ask());

  auto snap1 = book1.snapshot(5);
  auto snap2 = book2.snapshot(5);
  assert(snap1.bids == snap2.bids);
  assert(snap1.asks == snap2.asks);

  std::cout << "test_binary_logging_and_replay passed! (Replayed " << replayed << " events)\n";
}

void test_binary_log_performance() {
  BinaryEventLogger logger;
  const int N = 100'000;

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) {
    Order o{i + 1, 1, Side::Buy, OrderType::Limit, 100 + (i % 50), 10, i * 1000LL};
    logger.log_submit(o);
  }
  auto write_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
          .count();

  OrderBook replay_book;
  BinaryReplayer replayer;
  auto t1 = std::chrono::steady_clock::now();
  std::size_t count = replayer.replay(logger.data(), logger.size(), replay_book);
  auto replay_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1)
          .count();

  std::cout << "Binary ITCH: Logged " << N << " events in " << write_ms << " ms, Replayed in "
            << replay_ms << " ms\n";
  if (count != N || write_ms >= 500 || replay_ms >= 500) {
    std::cerr << "Performance check failed\n";
    std::abort();
  }
  std::cout << "test_binary_log_performance passed!\n";
}

int main() {
  test_binary_logging_and_replay();
  test_binary_log_performance();
  std::cout << "All Binary ITCH Log tests passed!\n";
  return 0;
}
