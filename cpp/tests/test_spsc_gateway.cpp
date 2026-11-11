// Unit tests for lock-free SpscGateway.
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "lobx/spsc_gateway.hpp"

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

void test_spsc_gateway_basic_lifecycle() {
  SpscGateway gw;
  CHECK(!gw.is_running());
  gw.start();
  CHECK(gw.is_running());

  // 1. Submit resting limit buy
  Order buy1{1, 1, Side::Buy, OrderType::Limit, 100, 10, 1000};
  CHECK(gw.submit_order(buy1));
  CHECK(gw.wait_until_processed(1));
  CHECK(gw.book().best_bid() == 100);

  // 2. Submit crossing limit sell
  Order sell1{2, 2, Side::Sell, OrderType::Limit, 100, 4, 2000};
  CHECK(gw.submit_order(sell1));
  CHECK(gw.wait_until_processed(2));

  // Verify execution event published to egress ring buffer
  auto ev = gw.poll_event();
  CHECK(ev.has_value());
  CHECK(ev->type == GatewayEventType::Fill);
  CHECK(ev->price == 100);
  CHECK(ev->qty == 4);
  CHECK(gw.total_trades() == 1);

  // Remaining quantity on resting bid should be 6
  auto snap = gw.book().snapshot(5);
  CHECK(snap.bids.size() == 1);
  CHECK(snap.bids[0].first == 100 && snap.bids[0].second == 6);

  // 3. Cancel resting order
  CHECK(gw.cancel_order(1, 3000, 1));
  CHECK(gw.wait_until_processed(3));
  auto cancel_ev = gw.poll_event();
  CHECK(cancel_ev.has_value());
  CHECK(cancel_ev->type == GatewayEventType::Cancelled);
  CHECK(gw.book().snapshot(5).bids.empty());

  gw.stop();
  CHECK(!gw.is_running());
}

void test_spsc_gateway_throughput_stress() {
  SpscGateway gw;
  gw.start();

  constexpr int kTotalOrders = 50000;
  for (int i = 0; i < kTotalOrders; ++i) {
    OrderId id = static_cast<OrderId>(i + 1);
    Side side = (i % 2 == 0) ? Side::Buy : Side::Sell;
    Price px = (side == Side::Buy) ? (10000 - (i % 20) * 5) : (10010 + (i % 20) * 5);
    Order o{id, 1, side, OrderType::Limit, px, 10, static_cast<Timestamp>(id * 10)};

    while (!gw.submit_order(o)) {
      std::this_thread::yield();
    }
  }

  CHECK(gw.wait_until_processed(kTotalOrders));
  CHECK(gw.total_processed() == kTotalOrders);

  gw.stop();
}

}  // namespace

int main() {
  test_spsc_gateway_basic_lifecycle();
  test_spsc_gateway_throughput_stress();

  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "all SpscGateway tests passed\n";
  return 0;
}
