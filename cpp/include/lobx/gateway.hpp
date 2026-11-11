// An asynchronous exchange gateway featuring an atomic sequencer, background matching worker,
// thread-safe ingress/egress queues, and sequence synchronization barriers.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "lobx/concurrent_order_book.hpp"
#include "lobx/concurrent_queue.hpp"
#include "lobx/types.hpp"

namespace lobx {

enum class RequestType : uint8_t { Submit, Cancel };

struct SequencedRequest {
  uint64_t seq_id{0};
  RequestType type{RequestType::Submit};
  Order order{0, 0, Side::Buy, OrderType::Limit, 0, 0, 0};
  OrderId cancel_id{0};
  Timestamp cancel_ts{0};
  AgentId agent_id{0};
};

enum class GatewayEventType : uint8_t { Fill, Cancelled, Rejected };

struct GatewayEvent {
  uint64_t seq_id{0};
  GatewayEventType type{GatewayEventType::Fill};
  OrderId order_id{0};
  AgentId agent_id{0};
  Price price{0};
  Qty qty{0};
  Timestamp ts{0};
  bool success{true};
};

class ExchangeGateway {
 public:
  explicit ExchangeGateway(std::size_t ingress_capacity = 0, std::size_t egress_capacity = 0);
  ~ExchangeGateway();

  ExchangeGateway(const ExchangeGateway&) = delete;
  ExchangeGateway& operator=(const ExchangeGateway&) = delete;

  // Gateway lifecycle
  void start();
  void stop();
  [[nodiscard]] bool is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
  }

  // Client ingress (thread-safe, callable concurrently from any number of client threads)
  uint64_t submit_order(const Order& order);
  uint64_t cancel_order(OrderId id, Timestamp ts, AgentId agent_id);

  // Synchronization barrier: blocks until all requests up to seq_id are matched by the worker
  // thread.
  bool wait_until_processed(uint64_t seq_id,
                            std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

  // Client egress: subscribe to execution/cancel events
  std::optional<GatewayEvent> poll_event() { return egress_queue_.try_pop(); }
  std::optional<GatewayEvent> wait_event() { return egress_queue_.pop(); }

  // Access the underlying thread-safe order book
  ConcurrentOrderBook& book() noexcept { return book_; }
  const ConcurrentOrderBook& book() const noexcept { return book_; }

  // Lock-free performance statistics
  [[nodiscard]] uint64_t total_requests_received() const noexcept {
    return next_seq_.load(std::memory_order_relaxed) - 1;
  }
  [[nodiscard]] uint64_t total_requests_processed() const noexcept {
    return last_processed_seq_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t total_trades_published() const noexcept {
    return total_trades_published_.load(std::memory_order_relaxed);
  }

 private:
  void run_worker();

  ConcurrentOrderBook book_;
  ConcurrentQueue<SequencedRequest> ingress_queue_;
  ConcurrentQueue<GatewayEvent> egress_queue_;

  std::atomic<uint64_t> next_seq_{1};
  std::atomic<uint64_t> last_processed_seq_{0};
  std::atomic<uint64_t> total_trades_published_{0};

  std::atomic<bool> running_{false};
  std::thread worker_thread_;

  std::mutex seq_mutex_;
  std::condition_variable seq_cv_;
};

}  // namespace lobx
