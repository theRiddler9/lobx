#pragma once

#include <cstdint>
#include <vector>

#include "lobx/order_book.hpp"

namespace lobx {

#pragma pack(push, 1)
struct BinaryEvent {
  std::uint8_t type = 0;        // 'A' = Add/Submit, 'C' = Cancel, 'F' = Fill
  std::uint8_t side = 0;        // 0 = Buy, 1 = Sell
  std::uint8_t order_type = 0;  // OrderType (0: Limit, 1: Market, etc.)
  std::uint8_t pad[5] = {0};    // Pad to 8-byte alignment for 64-bit members
  std::int64_t order_id = 0;
  std::int64_t agent_id = 0;
  std::int64_t price = 0;
  std::int64_t qty = 0;
  std::int64_t ts = 0;
};
#pragma pack(pop)

static_assert(sizeof(BinaryEvent) == 48, "BinaryEvent must be 48 bytes naturally aligned");

class BinaryEventLogger {
 public:
  void log_submit(const Order& o);
  void log_cancel(OrderId id, Timestamp ts);
  void log_fill(const Fill& f);

  const std::uint8_t* data() const { return buffer_.data(); }
  std::size_t size() const { return buffer_.size(); }
  std::size_t event_count() const { return buffer_.size() / sizeof(BinaryEvent); }
  void clear() { buffer_.clear(); }

 private:
  std::vector<std::uint8_t> buffer_;
  void append(const BinaryEvent& ev);
};

class BinaryReplayer {
 public:
  std::size_t replay(const std::uint8_t* data, std::size_t size, OrderBook& book);
};

}  // namespace lobx
