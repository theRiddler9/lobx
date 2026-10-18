#include "lobx/order_book.hpp"

#include <algorithm>
#include <iterator>
#include <string>

namespace lobx {

namespace {

// True if an incoming order is willing to trade at `level_price`. A buyer accepts any ask at or
// below its limit; a seller accepts any bid at or above its limit; a market order accepts any.
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

// The checks run in the same order as the Python engine, because a call that breaks two rules
// must raise the same kind of error from both engines.
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

template <class Map>
Qty OrderBook::match(Map& opposite, const Order& order, std::vector<Fill>& fills) {
  Qty remaining = order.qty;
  // Walk the opposite side from its best price outwards. At each price the oldest order trades
  // first (price-time priority) and every trade happens at the *resting* order's price.
  while (remaining > 0 && !opposite.empty()) {
    auto level_it = opposite.begin();
    Level& level = level_it->second;
    if (!crosses(order, level.price)) break;
    Resting& maker = level.orders.front();
    const Qty traded = std::min(remaining, maker.qty);
    fills.push_back(Fill{maker.id, order.id, order.side, level.price, traded, order.ts});
    maker.qty -= traded;
    level.total -= traded;
    remaining -= traded;
    if (maker.qty == 0) {
      index_.erase(maker.id);
      level.orders.pop_front();
      // An empty level must not linger, or best_bid/best_ask would report a price nobody offers.
      if (level.orders.empty()) opposite.erase(level_it);
    }
  }
  return remaining;
}

void OrderBook::rest(const Order& order, Qty qty) {
  const Price price = *order.price;
  auto place = [&](auto& side_map) {
    auto [it, inserted] = side_map.try_emplace(price);
    if (inserted) it->second.price = price;
    Level& level = it->second;
    level.orders.push_back(Resting{order.id, qty});
    level.total += qty;
    index_[order.id] = Locator{order.side, &level, std::prev(level.orders.end())};
  };
  if (order.side == Side::Buy) {
    place(bids_);
  } else {
    place(asks_);
  }
}

std::vector<Fill> OrderBook::submit(const Order& order) {
  validate(order);
  std::vector<Fill> fills;
  const Qty remaining =
      order.side == Side::Buy ? match(asks_, order, fills) : match(bids_, order, fills);
  // Only an order with a price can rest. A market order has none, so its remainder vanishes.
  if (remaining > 0 && order.type == OrderType::Limit) rest(order, remaining);
  last_ts_ = order.ts;
  return fills;
}

bool OrderBook::cancel(OrderId id, Timestamp ts) {
  check_ts(ts);
  last_ts_ = ts;
  const auto found = index_.find(id);
  if (found == index_.end()) return false;
  const Locator where = found->second;
  Level& level = *where.level;
  const Price price = level.price;  // read before the level can be erased below
  level.total -= where.it->qty;
  level.orders.erase(where.it);
  index_.erase(found);
  if (level.orders.empty()) {
    if (where.side == Side::Buy) {
      bids_.erase(price);
    } else {
      asks_.erase(price);
    }
  }
  return true;
}

Snapshot OrderBook::snapshot(std::size_t depth) const {
  Snapshot snap{last_ts_, {}, {}};
  for (auto it = bids_.begin(); it != bids_.end() && snap.bids.size() < depth; ++it) {
    snap.bids.emplace_back(it->first, it->second.total);
  }
  for (auto it = asks_.begin(); it != asks_.end() && snap.asks.size() < depth; ++it) {
    snap.asks.emplace_back(it->first, it->second.total);
  }
  return snap;
}

std::optional<Price> OrderBook::best_bid() const {
  if (bids_.empty()) return std::nullopt;
  return bids_.begin()->first;
}

std::optional<Price> OrderBook::best_ask() const {
  if (asks_.empty()) return std::nullopt;
  return asks_.begin()->first;
}

}  // namespace lobx
