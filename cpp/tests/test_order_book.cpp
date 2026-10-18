// Unit tests for the C++ engine. No framework (the build has no network access): each test is a
// function, CHECK records a failure instead of aborting, and main() returns non-zero on failure.
// CHECK is a macro rather than assert() because assert() is compiled out in Release builds.
#include <cstdlib>
#include <iostream>
#include <string>

#include "lobx/order_book.hpp"

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

Order limit(OrderId id, Side side, Price price, Qty qty, Timestamp ts) {
  return Order{id, 1, side, OrderType::Limit, price, qty, ts};
}

Order market(OrderId id, Side side, Qty qty, Timestamp ts) {
  return Order{id, 1, side, OrderType::Market, std::nullopt, qty, ts};
}

template <class Fn>
bool throws_invalid(Fn fn) {
  try {
    fn();
  } catch (const InvalidOrderError&) {
    return true;
  }
  return false;
}

void test_empty_book() {
  OrderBook book;
  CHECK(!book.best_bid().has_value());
  CHECK(!book.best_ask().has_value());
  const Snapshot snap = book.snapshot(5);
  CHECK(snap.ts == 0 && snap.bids.empty() && snap.asks.empty());
}

void test_resting_limit_orders_aggregate_by_level() {
  OrderBook book;
  book.submit(limit(1, Side::Buy, 100, 5, 1));
  book.submit(limit(2, Side::Buy, 100, 3, 2));
  book.submit(limit(3, Side::Buy, 99, 4, 3));
  book.submit(limit(4, Side::Sell, 102, 7, 4));
  CHECK(*book.best_bid() == 100);
  CHECK(*book.best_ask() == 102);
  const Snapshot snap = book.snapshot(5);
  CHECK(snap.ts == 4);
  CHECK((snap.bids == std::vector<PriceLevel>{{100, 8}, {99, 4}}));
  CHECK((snap.asks == std::vector<PriceLevel>{{102, 7}}));
  CHECK(book.snapshot(1).bids.size() == 1);
}

void test_trade_happens_at_resting_price() {
  OrderBook book;
  book.submit(limit(1, Side::Sell, 101, 5, 1));
  const auto fills = book.submit(limit(2, Side::Buy, 105, 5, 2));
  CHECK(fills.size() == 1);
  CHECK(fills[0].price == 101);  // the maker's price, not the taker's 105
  CHECK(fills[0].maker_id == 1 && fills[0].taker_id == 2);
  CHECK(fills[0].taker_side == Side::Buy && fills[0].ts == 2);
  CHECK(!book.best_ask().has_value() && !book.best_bid().has_value());
}

void test_price_time_priority() {
  OrderBook book;
  book.submit(limit(1, Side::Sell, 101, 2, 1));
  book.submit(limit(2, Side::Sell, 101, 2, 2));
  book.submit(limit(3, Side::Sell, 100, 2, 3));
  const auto fills = book.submit(market(4, Side::Buy, 5, 4));
  // Best price first (order 3), then the older order at 101 (order 1), then order 2.
  CHECK(fills.size() == 3);
  CHECK(fills[0].maker_id == 3 && fills[0].price == 100 && fills[0].qty == 2);
  CHECK(fills[1].maker_id == 1 && fills[1].price == 101 && fills[1].qty == 2);
  CHECK(fills[2].maker_id == 2 && fills[2].price == 101 && fills[2].qty == 1);
  CHECK((book.snapshot(5).asks == std::vector<PriceLevel>{{101, 1}}));
}

void test_partial_fill_rests_remainder() {
  OrderBook book;
  book.submit(limit(1, Side::Sell, 100, 3, 1));
  const auto fills = book.submit(limit(2, Side::Buy, 100, 10, 2));
  CHECK(fills.size() == 1 && fills[0].qty == 3);
  CHECK(!book.best_ask().has_value());
  CHECK((book.snapshot(5).bids == std::vector<PriceLevel>{{100, 7}}));
}

void test_partial_fill_of_resting_order_keeps_its_place() {
  OrderBook book;
  book.submit(limit(1, Side::Sell, 100, 5, 1));
  book.submit(limit(2, Side::Sell, 100, 5, 2));
  book.submit(market(3, Side::Buy, 2, 3));  // takes 2 of order 1
  const auto fills = book.submit(market(4, Side::Buy, 4, 4));
  CHECK(fills.size() == 2);
  CHECK(fills[0].maker_id == 1 && fills[0].qty == 3);  // still first in the queue
  CHECK(fills[1].maker_id == 2 && fills[1].qty == 1);
}

void test_market_order_remainder_is_discarded() {
  OrderBook book;
  book.submit(limit(1, Side::Sell, 100, 2, 1));
  const auto fills = book.submit(market(2, Side::Buy, 10, 2));
  CHECK(fills.size() == 1 && fills[0].qty == 2);
  CHECK(!book.best_bid().has_value());  // nothing rests
  CHECK(book.resting_orders() == 0);
  CHECK(book.submit(market(3, Side::Sell, 4, 3)).empty());  // empty book: no fills, no error
}

void test_limit_order_does_not_cross_its_own_limit() {
  OrderBook book;
  book.submit(limit(1, Side::Sell, 105, 5, 1));
  const auto fills = book.submit(limit(2, Side::Buy, 104, 5, 2));
  CHECK(fills.empty());
  CHECK(*book.best_bid() == 104 && *book.best_ask() == 105);
}

void test_cancel() {
  OrderBook book;
  book.submit(limit(1, Side::Buy, 100, 5, 1));
  book.submit(limit(2, Side::Buy, 100, 3, 2));
  book.submit(limit(3, Side::Buy, 100, 4, 3));
  CHECK(book.cancel(2, 4));  // from the middle of the queue
  CHECK((book.snapshot(5).bids == std::vector<PriceLevel>{{100, 9}}));
  CHECK(!book.cancel(2, 5));   // already gone
  CHECK(!book.cancel(99, 6));  // never existed
  CHECK(book.snapshot(5).ts == 6);
  const auto fills = book.submit(market(4, Side::Sell, 20, 7));
  CHECK(fills.size() == 2 && fills[0].maker_id == 1 && fills[1].maker_id == 3);
}

void test_cancel_last_order_removes_level() {
  OrderBook book;
  book.submit(limit(1, Side::Buy, 100, 5, 1));
  book.submit(limit(2, Side::Buy, 99, 5, 2));
  CHECK(book.cancel(1, 3));
  CHECK(*book.best_bid() == 99);
  CHECK(book.cancel(2, 4));
  CHECK(!book.best_bid().has_value());
  CHECK(book.resting_orders() == 0);
}

void test_filled_order_cannot_be_cancelled() {
  OrderBook book;
  book.submit(limit(1, Side::Sell, 100, 5, 1));
  book.submit(market(2, Side::Buy, 5, 2));
  CHECK(!book.cancel(1, 3));
}

void test_self_trade_is_allowed() {
  OrderBook book;
  book.submit(limit(1, Side::Sell, 100, 5, 1));  // same agent id (1) on both orders
  const auto fills = book.submit(limit(2, Side::Buy, 100, 5, 2));
  CHECK(fills.size() == 1);
}

void test_rejects_invalid_orders_and_changes_nothing() {
  OrderBook book;
  book.submit(limit(1, Side::Buy, 100, 5, 10));
  CHECK(throws_invalid([&] { book.submit(limit(2, Side::Buy, 100, 0, 11)); }));
  CHECK(throws_invalid([&] { book.submit(limit(2, Side::Buy, 100, -3, 11)); }));
  CHECK(throws_invalid([&] { book.submit(limit(2, Side::Buy, 0, 5, 11)); }));
  CHECK(throws_invalid([&] { book.submit(limit(2, Side::Buy, -4, 5, 11)); }));
  CHECK(throws_invalid([&] { book.submit(limit(2, Side::Buy, 100, 5, 9)); }));  // ts went back
  CHECK(throws_invalid(
      [&] { book.submit(Order{2, 1, Side::Buy, OrderType::Limit, std::nullopt, 5, 11}); }));
  CHECK(throws_invalid(
      [&] { book.submit(Order{2, 1, Side::Buy, OrderType::Market, Price{100}, 5, 11}); }));
  CHECK(throws_invalid([&] { book.cancel(1, 9); }));  // cancel obeys the clock too
  CHECK(book.snapshot(5).ts == 10);                   // a rejected call does not advance time
  CHECK((book.snapshot(5).bids == std::vector<PriceLevel>{{100, 5}}));
  CHECK(book.resting_orders() == 1);
}

void test_duplicate_id_rejected() {
  OrderBook book;
  book.submit(limit(1, Side::Buy, 100, 5, 1));
  bool duplicate = false;
  try {
    book.submit(limit(1, Side::Sell, 120, 5, 2));
  } catch (const DuplicateOrderIdError&) {
    duplicate = true;
  }
  CHECK(duplicate);
  // Once the order is gone its id can be used again (the Python engine only tracks resting ids).
  CHECK(book.cancel(1, 3));
  book.submit(limit(1, Side::Sell, 120, 5, 4));
  CHECK(*book.best_ask() == 120);
}

void test_validation_order_matches_python() {
  OrderBook book;
  book.submit(limit(1, Side::Buy, 100, 5, 10));
  // qty is checked before ts: both are wrong, the message must be about qty.
  try {
    book.submit(limit(2, Side::Buy, 100, 0, 1));
    CHECK(false);
  } catch (const InvalidOrderError& e) {
    CHECK(std::string(e.what()).find("qty") != std::string::npos);
  }
  // ts is checked before a duplicate id: it is a plain InvalidOrderError, not the subclass.
  try {
    book.submit(limit(1, Side::Buy, 100, 5, 1));
    CHECK(false);
  } catch (const DuplicateOrderIdError&) {
    CHECK(false);
  } catch (const InvalidOrderError&) {
  }
}

void test_sweeping_many_levels_cleans_up() {
  OrderBook book;
  for (int i = 0; i < 10; ++i) book.submit(limit(i + 1, Side::Sell, 100 + i, 1, i + 1));
  const auto fills = book.submit(limit(100, Side::Buy, 104, 20, 11));
  CHECK(fills.size() == 5);
  CHECK(*book.best_ask() == 105);
  CHECK(*book.best_bid() == 104);  // the unfilled 15 rests
  CHECK((book.snapshot(1).bids == std::vector<PriceLevel>{{104, 15}}));
  CHECK(book.resting_orders() == 6);
}

}  // namespace

int main() {
  test_empty_book();
  test_resting_limit_orders_aggregate_by_level();
  test_trade_happens_at_resting_price();
  test_price_time_priority();
  test_partial_fill_rests_remainder();
  test_partial_fill_of_resting_order_keeps_its_place();
  test_market_order_remainder_is_discarded();
  test_limit_order_does_not_cross_its_own_limit();
  test_cancel();
  test_cancel_last_order_removes_level();
  test_filled_order_cannot_be_cancelled();
  test_self_trade_is_allowed();
  test_rejects_invalid_orders_and_changes_nothing();
  test_duplicate_id_rejected();
  test_validation_order_matches_python();
  test_sweeping_many_levels_cleans_up();
  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "all C++ engine tests passed\n";
  return EXIT_SUCCESS;
}
