// Unit tests for ConcurrentQueue.
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "lobx/concurrent_queue.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      ++g_failures;                                                              \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
    }                                                                            \
  } while (0)

void test_basic_queue_operations() {
  lobx::ConcurrentQueue<int> q(2);
  CHECK(q.empty());
  CHECK(q.size() == 0);
  CHECK(q.try_push(10));
  CHECK(q.try_push(20));
  CHECK(!q.try_push(30));  // full
  CHECK(q.size() == 2);

  auto item1 = q.try_pop();
  CHECK(item1.has_value() && *item1 == 10);
  auto item2 = q.pop();
  CHECK(item2.has_value() && *item2 == 20);
  CHECK(q.empty());
  CHECK(q.total_pushed() == 2);
  CHECK(q.total_popped() == 2);
}

void test_timeout_and_close() {
  lobx::ConcurrentQueue<int> q(5);
  auto start = std::chrono::steady_clock::now();
  auto val = q.wait_pop_for(std::chrono::milliseconds(20));
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  CHECK(!val.has_value());
  CHECK(elapsed.count() >= 15);

  std::thread consumer([&q]() {
    auto res = q.pop();
    CHECK(!res.has_value());  // woke up due to close
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  q.close();
  consumer.join();
  CHECK(q.is_closed());
}

void test_multi_producer_multi_consumer() {
  lobx::ConcurrentQueue<int> q(128);
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kItemsPerProducer = 10000;
  std::atomic<long long> sum_consumed{0};
  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  for (int i = 0; i < kConsumers; ++i) {
    consumers.emplace_back([&]() {
      while (auto val = q.pop()) {
        sum_consumed.fetch_add(*val, std::memory_order_relaxed);
      }
    });
  }

  for (int i = 0; i < kProducers; ++i) {
    producers.emplace_back([&]() {
      for (int j = 1; j <= kItemsPerProducer; ++j) {
        q.push(j);
      }
    });
  }

  for (auto& t : producers) t.join();
  q.close();
  for (auto& t : consumers) t.join();

  long long expected_per_producer = static_cast<long long>(kItemsPerProducer) * (kItemsPerProducer + 1) / 2;
  CHECK(sum_consumed.load() == kProducers * expected_per_producer);
  CHECK(q.total_pushed() == static_cast<std::size_t>(kProducers * kItemsPerProducer));
  CHECK(q.total_popped() == static_cast<std::size_t>(kProducers * kItemsPerProducer));
}

}  // namespace

int main() {
  test_basic_queue_operations();
  test_timeout_and_close();
  test_multi_producer_multi_consumer();
  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "all concurrent queue tests passed\n";
  return 0;
}
