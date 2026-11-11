#include <cassert>
#include <iostream>

#include "lobx/order_book.hpp"

using namespace lobx;

void test_ioc() {
  OrderBook book;
  // Maker rests 100 qty @ 100
  book.submit(Order{1, 1, Side::Buy, OrderType::Limit, 100, 100, 1000});

  // Taker IOC for 150 @ 100: fills 100, remaining 50 is canceled
  auto fills = book.submit(Order{2, 2, Side::Sell, OrderType::IOC, 100, 150, 2000});
  assert(fills.size() == 1);
  assert(fills[0].qty == 100);
  assert(fills[0].price == 100);
  assert(!book.best_bid().has_value());
  assert(!book.best_ask().has_value());
  std::cout << "test_ioc passed\n";
}

void test_fok() {
  OrderBook book;
  book.submit(Order{1, 1, Side::Sell, OrderType::Limit, 101, 50, 1000});
  book.submit(Order{2, 1, Side::Sell, OrderType::Limit, 102, 50, 2000});

  // FOK for 150 @ 102: cannot fill entirely, should kill without fills
  auto killed_fills = book.submit(Order{3, 2, Side::Buy, OrderType::FOK, 102, 150, 3000});
  assert(killed_fills.empty());
  assert(book.best_ask() == 101);
  assert(book.snapshot(2).asks[0].second == 50);

  // FOK for 100 @ 102: can fill entirely
  auto fills = book.submit(Order{4, 2, Side::Buy, OrderType::FOK, 102, 100, 4000});
  assert(fills.size() == 2);
  assert(fills[0].qty == 50 && fills[0].price == 101);
  assert(fills[1].qty == 50 && fills[1].price == 102);
  assert(!book.best_ask().has_value());
  std::cout << "test_fok passed\n";
}

void test_post_only() {
  OrderBook book;
  book.submit(Order{1, 1, Side::Sell, OrderType::Limit, 100, 50, 1000});

  // PostOnly that would cross best ask @ 100: rejected
  auto rej_fills = book.submit(Order{2, 2, Side::Buy, OrderType::PostOnly, 100, 20, 2000});
  assert(rej_fills.empty());
  assert(!book.best_bid().has_value());
  assert(book.best_ask() == 100);

  // PostOnly passive @ 99: rests successfully
  auto ok_fills = book.submit(Order{3, 2, Side::Buy, OrderType::PostOnly, 99, 20, 3000});
  assert(ok_fills.empty());
  assert(book.best_bid() == 99);
  assert(book.best_ask() == 100);
  std::cout << "test_post_only passed\n";
}

int main() {
  test_ioc();
  test_fok();
  test_post_only();
  std::cout << "All institutional order tests passed!\n";
  return 0;
}
