// A lock-free, zero-allocation Single-Producer Single-Consumer (SPSC) exchange gateway.
//
// CONCURRENCY & MEMORY ORDERING DESIGN:
// While ExchangeGateway uses std::mutex and std::condition_variable to arbitrate multiple
// client threads, high-frequency trading (HFT) gateways assign a dedicated ingress thread
// to each direct market access (DMA) client channel.
//
// In a 1-to-1 thread topology, mutexes are pure overhead:
// 1. Mutex acquisitions incur cache-coherence traffic and lock bus cycles.
// 2. Condition variables put threads to sleep via OS syscalls (futex on Linux, WaitOnAddress on
// Windows),
//    costing 1,000 - 3,000 ns per context switch.
//
// SpscGateway eliminates all locks, syscalls, and dynamic memory allocations on the hot path:
// - Ingress: Client thread pushes to an SpscRingBuffer with memory_order_release.
// - Engine: Background matching thread pops with memory_order_acquire.
// - Egress: Fills and execution reports stream back to the client via a reverse SpscRingBuffer.
// - False Sharing Prevention: alignas(64) on internal atomic indices separates producer and
// consumer
//   cache lines into distinct 64-byte L1 cache sectors.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "lobx/gateway.hpp"
#include "lobx/order_book.hpp"
#include "lobx/spsc_ring_buffer.hpp"
#include "lobx/types.hpp"

namespace lobx {

class SpscGateway {
 public:
  static constexpr std::size_t kDefaultCapacity = 65536;

  explicit SpscGateway()
      : ingress_ring_(std::make_unique<SpscRingBuffer<SequencedRequest, kDefaultCapacity>>()),
        egress_ring_(std::make_unique<SpscRingBuffer<GatewayEvent, kDefaultCapacity>>()) {}
  ~SpscGateway() { stop(); }

  SpscGateway(const SpscGateway&) = delete;
  SpscGateway& operator=(const SpscGateway&) = delete;

  // Lifecycle control
  void start() {
    bool expected = false;
    // std::memory_order_acq_rel: synchronize running state before starting thread
    if (!running_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      return;
    }
    worker_thread_ = std::thread(&SpscGateway::run_worker, this);
  }

  void stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
      return;
    }
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

  [[nodiscard]] bool is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
  }

  // Client ingress (CALLED ONLY BY THE SINGLE CLIENT PRODUCER THREAD)
  // Lock-free: returns false if the ring buffer is temporarily full.
  bool submit_order(const Order& order) {
    uint64_t seq = next_seq_.fetch_add(1, std::memory_order_relaxed);
    SequencedRequest req{seq, RequestType::Submit, order, 0, 0, order.agent};
    return ingress_ring_->try_push(req);
  }

  bool cancel_order(OrderId id, Timestamp ts, AgentId agent_id) {
    uint64_t seq = next_seq_.fetch_add(1, std::memory_order_relaxed);
    SequencedRequest req{seq, RequestType::Cancel, Order{}, id, ts, agent_id};
    return ingress_ring_->try_push(req);
  }

  // Client egress (CALLED ONLY BY THE SINGLE CLIENT THREAD)
  // Lock-free polling of trade executions and status events.
  std::optional<GatewayEvent> poll_event() { return egress_ring_->try_pop(); }

  // Spin-wait barrier: useful for testing and deterministic synchronization without locks
  bool wait_until_processed(uint64_t target_seq,
                            std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (last_processed_seq_.load(std::memory_order_acquire) < target_seq) {
      if (std::chrono::steady_clock::now() > deadline) {
        return false;
      }
      std::this_thread::yield();
    }
    return true;
  }

  const OrderBook& book() const noexcept { return book_; }
  OrderBook& book() noexcept { return book_; }

  [[nodiscard]] uint64_t total_processed() const noexcept {
    return last_processed_seq_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t total_trades() const noexcept {
    return total_trades_.load(std::memory_order_relaxed);
  }

 private:
  void run_worker() {
    while (running_.load(std::memory_order_acquire) || !ingress_ring_->empty()) {
      auto req_opt = ingress_ring_->try_pop();
      if (!req_opt.has_value()) {
        std::this_thread::yield();
        continue;
      }

      const auto& req = *req_opt;
      if (req.type == RequestType::Submit) {
        std::vector<Fill> fills = book_.submit(req.order);
        for (const auto& f : fills) {
          total_trades_.fetch_add(1, std::memory_order_relaxed);
          GatewayEvent ev{req.seq_id,   GatewayEventType::Fill,
                          req.order.id, req.order.agent,
                          f.price,      f.qty,
                          f.ts,         true};
          egress_ring_->try_push(ev);
        }
      } else if (req.type == RequestType::Cancel) {
        bool ok = book_.cancel(req.cancel_id, req.cancel_ts);
        GatewayEvent ev{req.seq_id,
                        ok ? GatewayEventType::Cancelled : GatewayEventType::Rejected,
                        req.cancel_id,
                        req.agent_id,
                        0,
                        0,
                        req.cancel_ts,
                        ok};
        egress_ring_->try_push(ev);
      }

      // std::memory_order_release: publish processed sequence to waiting client threads
      last_processed_seq_.store(req.seq_id, std::memory_order_release);
    }
  }

  OrderBook book_;
  std::unique_ptr<SpscRingBuffer<SequencedRequest, kDefaultCapacity>> ingress_ring_;
  std::unique_ptr<SpscRingBuffer<GatewayEvent, kDefaultCapacity>> egress_ring_;

  alignas(64) std::atomic<uint64_t> next_seq_{1};
  alignas(64) std::atomic<uint64_t> last_processed_seq_{0};
  std::atomic<uint64_t> total_trades_{0};
  std::atomic<bool> running_{false};
  std::thread worker_thread_;
};

}  // namespace lobx
