// Unit tests for FixedPool and SlabOrderBook.
#include <iostream>
#include <vector>

#include "lobx/memory_pool.hpp"
#include "lobx/slab_order_book.hpp"

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

void test_fixed_pool_operations() {
  struct Item {
    int id;
    double val;
  };

  FixedPool<Item, 4> pool;
  CHECK(pool.capacity() == 4);
  CHECK(pool.active_count() == 0);

  Item* a = pool.allocate(Item{1, 10.0});
  Item* b = pool.allocate(Item{2, 20.0});
  Item* c = pool.allocate(Item{3, 30.0});
  Item* d = pool.allocate(Item{4, 40.0});
  CHECK(pool.active_count() == 4);
  CHECK(pool.allocate(Item{5, 50.0}) == nullptr);  // pool full

  CHECK(a->id == 1 && b->id == 2 && c->id == 3 && d->id == 4);

  pool.deallocate(b);
  CHECK(pool.active_count() == 3);

  // Can allocate again into the freed slot
  Item* e = pool.allocate(Item{5, 50.0});
  CHECK(e != nullptr);
  CHECK(e->id == 5);
  CHECK(pool.active_count() == 4);

  pool.deallocate(a);
  pool.deallocate(c);
  pool.deallocate(d);
  pool.deallocate(e);
  CHECK(pool.active_count() == 0);
}

void test_slab_order_book() {
  SlabOrderBook<100> book;
  CHECK(book.pool_active_count() == 0);

  // Submit resting buy orders
  Order b1{1, 10, Side::Buy, OrderType::Limit, 100, 50, 1000};
  Order b2{2, 10, Side::Buy, OrderType::Limit, 100, 30, 2000};
  book.submit(b1);
  book.submit(b2);
  CHECK(book.resting_orders() == 2);
  CHECK(book.pool_active_count() == 2);
  CHECK(book.best_bid() == 100);

  // Cross with sell order of 60 shares
  Order s1{3, 20, Side::Sell, OrderType::Limit, 100, 60, 3000};
  auto fills = book.submit(s1);
  CHECK(fills.size() == 2);
  CHECK(fills[0].maker_id == 1 && fills[0].qty == 50);  // b1 fully filled and deallocated
  CHECK(fills[1].maker_id == 2 && fills[1].qty == 10);  // b2 partially filled

  // b1 is gone, b2 has 20 remaining
  CHECK(book.resting_orders() == 1);
  CHECK(book.pool_active_count() == 1);

  // Cancel b2
  CHECK(book.cancel(2, 4000));
  CHECK(book.resting_orders() == 0);
  CHECK(book.pool_active_count() == 0);  // Zero memory leaks in pool!
}

}  // namespace

int main() {
  test_fixed_pool_operations();
  test_slab_order_book();
  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "all slab order book tests passed\n";
  return 0;
}
