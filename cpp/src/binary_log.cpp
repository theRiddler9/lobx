#include "lobx/binary_log.hpp"

#include <cstring>

namespace lobx {

void BinaryEventLogger::append(const BinaryEvent& ev) {
  const auto* ptr = reinterpret_cast<const std::uint8_t*>(&ev);
  buffer_.insert(buffer_.end(), ptr, ptr + sizeof(BinaryEvent));
}

void BinaryEventLogger::log_submit(const Order& o) {
  BinaryEvent ev{};
  ev.type = 'A';
  ev.side = static_cast<std::uint8_t>(o.side == Side::Buy ? 0 : 1);
  ev.order_type = static_cast<std::uint8_t>(o.type);
  ev.order_id = o.id;
  ev.agent_id = o.agent;
  ev.price = o.price.value_or(0);
  ev.qty = o.qty;
  ev.ts = o.ts;
  append(ev);
}

void BinaryEventLogger::log_cancel(OrderId id, Timestamp ts) {
  BinaryEvent ev{};
  ev.type = 'C';
  ev.order_id = id;
  ev.ts = ts;
  append(ev);
}

void BinaryEventLogger::log_fill(const Fill& f) {
  BinaryEvent ev{};
  ev.type = 'F';
  ev.side = static_cast<std::uint8_t>(f.taker_side == Side::Buy ? 0 : 1);
  ev.order_id = f.maker_id;
  ev.agent_id = f.taker_id;
  ev.price = f.price;
  ev.qty = f.qty;
  ev.ts = f.ts;
  append(ev);
}

std::size_t BinaryReplayer::replay(const std::uint8_t* data, std::size_t size, OrderBook& book) {
  const std::size_t count = size / sizeof(BinaryEvent);
  for (std::size_t i = 0; i < count; ++i) {
    BinaryEvent ev{};
    std::memcpy(&ev, data + i * sizeof(BinaryEvent), sizeof(BinaryEvent));
    if (ev.type == 'A') {
      const OrderId id = ev.order_id;
      const AgentId agent = ev.agent_id;
      const Side side = (ev.side == 0) ? Side::Buy : Side::Sell;
      const OrderType type = static_cast<OrderType>(ev.order_type);
      const Price px = ev.price;
      const Qty qty = ev.qty;
      const Timestamp ts = ev.ts;
      book.submit(
          Order{id, agent, side, type, px > 0 ? std::make_optional(px) : std::nullopt, qty, ts});
    } else if (ev.type == 'C') {
      book.cancel(ev.order_id, ev.ts);
    }
  }
  return count;
}

}  // namespace lobx
