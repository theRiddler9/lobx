// Unit tests for Lock-Free Cache-Aligned SpscRingBuffer.
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "lobx/spsc_ring_buffer.hpp"

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

void test_spsc_basic_operations() {
  SpscRingBuffer<int, 4> rb;
  CHECK(rb.empty());
  CHECK(rb.size() == 0);
  CHECK(rb.capacity() == 4);

  CHECK(rb.try_push(10));
  CHECK(rb.try_push(20));
  CHECK(rb.try_push(30));
  CHECK(rb.try_push(40));
  CHECK(!rb.try_push(50));  // full!
  CHECK(rb.size() == 4);

  auto val1 = rb.try_pop();
  CHECK(val1.has_value() && *val1 == 10);
  auto val2 = rb.try_pop();
  CHECK(val2.has_value() && *val2 == 20);
  CHECK(rb.size() == 2);

  CHECK(rb.try_push(50));
  CHECK(rb.try_push(60));
  CHECK(rb.size() == 4);

  auto val3 = rb.try_pop();
  CHECK(val3.has_value() && *val3 == 30);
}

void test_spsc_multithreaded_stress() {
  constexpr std::size_t kCapacity = 1024;
  constexpr int kTotalItems = 1000000;
  SpscRingBuffer<int, kCapacity> rb;

  std::atomic<bool> producer_done{false};
  long long consumer_sum = 0;

  // Single consumer thread
  std::thread consumer([&]() {
    while (!producer_done.load(std::memory_order_relaxed) || !rb.empty()) {
      if (auto val = rb.try_pop()) {
        consumer_sum += *val;
      }
    }
  });

  // Single producer thread (main thread)
  for (int i = 1; i <= kTotalItems; ++i) {
    while (!rb.try_push(i)) {
      // Busy wait / yield
      std::this_thread::yield();
    }
  }
  producer_done.store(true, std::memory_order_release);
  consumer.join();

  long long expected_sum = static_cast<long long>(kTotalItems) * (kTotalItems + 1) / 2;
  CHECK(consumer_sum == expected_sum);
}

}  // namespace

int main() {
  test_spsc_basic_operations();
  test_spsc_multithreaded_stress();
  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "all lock-free spsc ring buffer tests passed\n";
  return 0;
}
