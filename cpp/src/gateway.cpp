#include "lobx/gateway.hpp"

#include <exception>

namespace lobx {

ExchangeGateway::ExchangeGateway(std::size_t ingress_capacity, std::size_t egress_capacity)
    : ingress_queue_(ingress_capacity), egress_queue_(egress_capacity) {}

ExchangeGateway::~ExchangeGateway() { stop(); }

void ExchangeGateway::start() {
  if (running_.exchange(true, std::memory_order_acq_rel)) return;
  worker_thread_ = std::thread(&ExchangeGateway::run_worker, this);
}

void ExchangeGateway::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  ingress_queue_.close();
  if (worker_thread_.joinable()) worker_thread_.join();
  egress_queue_.close();
  seq_cv_.notify_all();
}

uint64_t ExchangeGateway::submit_order(const Order& order) {
  uint64_t seq = next_seq_.fetch_add(1, std::memory_order_relaxed);
  ingress_queue_.push(SequencedRequest{seq, RequestType::Submit, order, 0, 0, order.agent});
  return seq;
}

uint64_t ExchangeGateway::cancel_order(OrderId id, Timestamp ts, AgentId agent_id) {
  uint64_t seq = next_seq_.fetch_add(1, std::memory_order_relaxed);
  ingress_queue_.push(SequencedRequest{seq, RequestType::Cancel, Order{}, id, ts, agent_id});
  return seq;
}

bool ExchangeGateway::wait_until_processed(uint64_t seq_id, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(seq_mutex_);
  return seq_cv_.wait_for(lock, timeout, [this, seq_id] {
    return last_processed_seq_.load(std::memory_order_acquire) >= seq_id ||
           !running_.load(std::memory_order_acquire);
  });
}

void ExchangeGateway::run_worker() {
  while (auto opt = ingress_queue_.pop()) {
    const auto& req = *opt;
    if (req.type == RequestType::Submit) {
      try {
        auto fills = book_.submit(req.order);
        for (const auto& f : fills) {
          if (egress_queue_.try_push(GatewayEvent{req.seq_id, GatewayEventType::Fill, req.order.id,
                                                  req.agent_id, f.price, f.qty, f.ts, true})) {
            total_trades_published_.fetch_add(1, std::memory_order_relaxed);
          }
        }
      } catch (const std::exception&) {
        egress_queue_.try_push(GatewayEvent{req.seq_id, GatewayEventType::Rejected, req.order.id,
                                            req.agent_id, 0, 0, req.order.ts, false});
      }
    } else if (req.type == RequestType::Cancel) {
      bool ok = book_.cancel(req.cancel_id, req.cancel_ts);
      egress_queue_.try_push(GatewayEvent{req.seq_id, GatewayEventType::Cancelled, req.cancel_id,
                                          req.agent_id, 0, 0, req.cancel_ts, ok});
    }

    uint64_t curr = last_processed_seq_.load(std::memory_order_relaxed);
    while (req.seq_id > curr &&
           !last_processed_seq_.compare_exchange_weak(curr, req.seq_id, std::memory_order_release,
                                                      std::memory_order_relaxed)) {
    }
    {
      std::lock_guard<std::mutex> lock(seq_mutex_);
    }
    seq_cv_.notify_all();
  }
}

}  // namespace lobx
