// Unit tests for ExchangeGateway.
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "lobx/gateway.hpp"

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

void test_gateway_basic_lifecycle() {
  ExchangeGateway gateway;
  CHECK(!gateway.is_running());
  gateway.start();
  CHECK(gateway.is_running());

  // Submit resting buy
  Order buy{101, 1, Side::Buy, OrderType::Limit, 100, 50, 1000};
  uint64_t seq1 = gateway.submit_order(buy);
  CHECK(seq1 == 1);
  CHECK(gateway.wait_until_processed(seq1));

  // Submit crossing sell
  Order sell{102, 2, Side::Sell, OrderType::Limit, 100, 30, 2000};
  uint64_t seq2 = gateway.submit_order(sell);
  CHECK(seq2 == 2);
  CHECK(gateway.wait_until_processed(seq2));

  // Verify book state
  auto snap = gateway.book().snapshot(5);
  CHECK(snap.bids.size() == 1);
  CHECK(snap.bids[0].first == 100 && snap.bids[0].second == 20);

  // Read egress events
  auto ev1 = gateway.poll_event();
  CHECK(ev1.has_value());
  CHECK(ev1->type == GatewayEventType::Fill);
  CHECK(ev1->qty == 30);
  CHECK(ev1->price == 100);

  // Cancel remaining
  uint64_t seq3 = gateway.cancel_order(101, 3000, 1);
  CHECK(gateway.wait_until_processed(seq3));
  auto snap2 = gateway.book().snapshot(5);
  CHECK(snap2.bids.empty());

  gateway.stop();
  CHECK(!gateway.is_running());
}

void test_gateway_multithreaded_clients() {
  ExchangeGateway gateway(1024, 1024);
  gateway.start();

  constexpr int kClients = 4;
  constexpr int kOrdersPerClient = 1000;
  std::vector<std::thread> clients;
  std::atomic<uint64_t> max_seq{0};

  for (int c = 0; c < kClients; ++c) {
    clients.emplace_back([&, c]() {
      for (int i = 0; i < kOrdersPerClient; ++i) {
        OrderId id = static_cast<OrderId>(c * 100000 + i + 1);
        Order o{id,
                c + 1,
                (i % 2 == 0) ? Side::Buy : Side::Sell,
                OrderType::Limit,
                10000 + (i % 10) * 2,
                10,
                static_cast<Timestamp>(id * 10)};
        uint64_t seq = gateway.submit_order(o);
        uint64_t curr = max_seq.load(std::memory_order_relaxed);
        while (seq > curr && !max_seq.compare_exchange_weak(curr, seq, std::memory_order_relaxed)) {
        }
      }
    });
  }

  for (auto& t : clients) t.join();

  // Wait for all orders to process
  CHECK(gateway.wait_until_processed(max_seq.load()));
  CHECK(gateway.total_requests_processed() == kClients * kOrdersPerClient);

  gateway.stop();
}

}  // namespace

int main() {
  test_gateway_basic_lifecycle();
  test_gateway_multithreaded_clients();
  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "all gateway tests passed\n";
  return 0;
}
