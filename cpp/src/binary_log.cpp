#include "lobx/binary_log.hpp"

namespace lobx {

void BinaryEventLogger::append(const BinaryEvent& ev) {
  const auto* ptr = reinterpret_cast<const std::uint8_t*>(&ev);
  buffer_.insert(buffer_.end(), ptr, ptr + sizeof(BinaryEvent));
}

void BinaryEventLogger::log_submit(const Order& o) {
  append(BinaryEvent{
      'A', static_cast<std::uint8_t>(o.side == Side::Buy ? 0 : 1),
      static_cast<std::uint8_t>(o.type), 0, o.id, o.agent,
      o.price.value_or(0), o.qty, o.ts});
}

void BinaryEventLogger::log_cancel(OrderId id, Timestamp ts) {
  append(BinaryEvent{'C', 0, 0, 0, id, 0, 0, 0, ts});
}

void BinaryEventLogger::log_fill(const Fill& f) {
  append(BinaryEvent{
      'F', static_cast<std::uint8_t>(f.taker_side == Side::Buy ? 0 : 1),
      0, 0, f.maker_id, f.taker_id, f.price, f.qty, f.ts});
}

std::size_t BinaryReplayer::replay(const std::uint8_t* data, std::size_t size, OrderBook& book) {
  const std::size_t count = size / sizeof(BinaryEvent);
  const auto* events = reinterpret_cast<const BinaryEvent*>(data);
  for (std::size_t i = 0; i < count; ++i) {
    const BinaryEvent& ev = events[i];
    if (ev.type == 'A') {
      book.submit(Order{
          ev.order_id, ev.agent_id,
          ev.side == 0 ? Side::Buy : Side::Sell,
          static_cast<OrderType>(ev.order_type),
          ev.price > 0 ? std::make_optional(ev.price) : std::nullopt,
          ev.qty, ev.ts});
    } else if (ev.type == 'C') {
      book.cancel(ev.order_id, ev.ts);
    }
  }
  return count;
}

}  // namespace lobx
