#include "lobx/order_book.hpp"

#include <algorithm>
#include <iterator>
#include <string>

namespace lobx {

namespace {

bool crosses(const Order& order, Price level_price) {
  if (order.type == OrderType::Market) return true;
  const Price limit = *order.price;
  return order.side == Side::Buy ? level_price <= limit : level_price >= limit;
}

}  // namespace

void OrderBook::check_ts(Timestamp ts) const {
  if (ts < last_ts_) {
    throw InvalidOrderError("ts must not be earlier than " + std::to_string(last_ts_) +
                            " (the previous call), got " + std::to_string(ts));
  }
}

void OrderBook::validate(const Order& order) const {
  if (order.qty <= 0) {
    throw InvalidOrderError("qty must be an integer greater than 0, got " +
                            std::to_string(order.qty));
  }
  check_ts(order.ts);
  if (order.type == OrderType::Market) {
    if (order.price.has_value()) {
      throw InvalidOrderError("a market order must not have a price, got " +
                              std::to_string(*order.price));
    }
  } else if (!order.price.has_value() || *order.price <= 0) {
    throw InvalidOrderError("a limit order needs an integer price greater than 0");
  }
  if (index_.count(order.id) != 0) {
    throw DuplicateOrderIdError("order id " + std::to_string(order.id) +
                                " is already resting in the book");
  }
}

}  // namespace lobx
