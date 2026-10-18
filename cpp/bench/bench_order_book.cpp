// Pure C++ throughput and latency benchmark (no Python in the loop).
// Uses the same shape of stream as src/lobx/engine/streams.py: prices in a narrow band so orders
// cross often. The generator is a small LCG so the stream is identical on every machine.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "lobx/order_book.hpp"

namespace {

struct Lcg {
  std::uint64_t state;
  // Returns a value in [0, n). The high bits of an LCG are the well-distributed ones.
  std::uint64_t next(std::uint64_t n) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (state >> 33) % n;
  }
};

struct Op {
  bool is_cancel;
  lobx::Order order;        // used when !is_cancel
  lobx::OrderId cancel_id;  // used when is_cancel
  lobx::Timestamp ts;
};

std::vector<Op> make_stream(std::size_t count) {
  Lcg rng{1};
  std::vector<Op> ops;
  ops.reserve(count);
  lobx::Timestamp ts = 0;
  lobx::OrderId next_id = 1;
  for (std::size_t i = 0; i < count; ++i) {
    ts += static_cast<lobx::Timestamp>(rng.next(3));
    const std::uint64_t kind = rng.next(100);
    if (kind < 30 && next_id > 1) {
      ops.push_back({true, {}, 1 + static_cast<lobx::OrderId>(rng.next(next_id - 1)), ts});
      continue;
    }
    const lobx::Side side = rng.next(2) == 0 ? lobx::Side::Buy : lobx::Side::Sell;
    const lobx::Qty qty = 1 + static_cast<lobx::Qty>(rng.next(20));
    lobx::Order order{next_id++, 1, side, lobx::OrderType::Market, std::nullopt, qty, ts};
    if (kind >= 45) {
      order.type = lobx::OrderType::Limit;
      order.price = 980 + static_cast<lobx::Price>(rng.next(41));
    }
    ops.push_back({false, order, 0, ts});
  }
  return ops;
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
  const std::vector<Op> ops = make_stream(count);
  lobx::OrderBook book;
  std::vector<std::int64_t> latencies;
  latencies.reserve(count);
  std::size_t fills = 0;

  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  for (const Op& op : ops) {
    const auto t0 = clock::now();
    if (op.is_cancel) {
      book.cancel(op.cancel_id, op.ts);
    } else {
      fills += book.submit(op.order).size();
    }
    latencies.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
  }
  const double seconds = std::chrono::duration<double>(clock::now() - start).count();

  std::sort(latencies.begin(), latencies.end());
  std::printf("ops=%zu fills=%zu\n", count, fills);
  std::printf("throughput: %.0f ops/s\n", static_cast<double>(count) / seconds);
  std::printf("median latency: %lld ns\n", static_cast<long long>(latencies[count / 2]));
  std::printf("p99 latency: %lld ns\n", static_cast<long long>(latencies[count * 99 / 100]));
  return 0;
}
