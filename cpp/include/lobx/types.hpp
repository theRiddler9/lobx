// Shared types for the C++ matching engine.
//
// These mirror src/lobx/types.py. Prices are integer ticks and times are integer nanoseconds:
// no floating point anywhere in the engine, so the C++ and Python engines can be compared exactly.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lobx {

using OrderId = std::int64_t;
using AgentId = std::int64_t;
using Price = std::int64_t;  // ticks
using Qty = std::int64_t;
using Timestamp = std::int64_t;  // nanoseconds since simulation start

enum class Side : int { Buy = 0, Sell = 1 };
enum class OrderType : int { Limit = 0, Market = 1 };

// An instruction to the engine. `price` is empty for market orders.
struct Order {
  OrderId id;
  AgentId agent;
  Side side;
  OrderType type;
  std::optional<Price> price;
  Qty qty;
  Timestamp ts;
};

// One trade between a resting (maker) order and an incoming (taker) order.
// `price` is always the maker's price.
struct Fill {
  OrderId maker_id;
  OrderId taker_id;
  Side taker_side;
  Price price;
  Qty qty;
  Timestamp ts;
};

using PriceLevel = std::pair<Price, Qty>;  // (price, total quantity at that price)

// Aggregated depth, best level first on each side.
struct Snapshot {
  Timestamp ts;
  std::vector<PriceLevel> bids;
  std::vector<PriceLevel> asks;
};

// Thrown for an order or call that breaks a validation rule (see docs/contracts).
struct InvalidOrderError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

// Thrown when an order id is already resting in the book.
struct DuplicateOrderIdError : InvalidOrderError {
  using InvalidOrderError::InvalidOrderError;
};

}  // namespace lobx
