// The fast matching engine. It must behave exactly like the Python reference engine
// (src/lobx/engine/book_py.py); the differential test in tests/ enforces that.
#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "lobx/types.hpp"

namespace lobx {

class OrderBook {
 public:
  // Apply `order` and return the fills it produced, in execution order.
  // Throws InvalidOrderError or DuplicateOrderIdError before touching the book.
  std::vector<Fill> submit(const Order& order);

  // Remove a resting order. Returns true if it was resting, false otherwise.
  bool cancel(OrderId id, Timestamp ts);

  // Up to `depth` aggregated levels per side, best first.
  [[nodiscard]] Snapshot snapshot(std::size_t depth) const;

  [[nodiscard]] std::optional<Price> best_bid() const;
  [[nodiscard]] std::optional<Price> best_ask() const;

  // Number of orders currently resting.
  [[nodiscard]] std::size_t resting_orders() const { return index_.size(); }

 private:
  // One order waiting in the book. Its price is the price of the level that holds it.
  struct Resting {
    OrderId id;
    Qty qty;
  };

  // All resting orders at one price, oldest first (price-time priority), plus their total.
  // std::list keeps iterators valid when other orders are added or removed, which is what lets
  // cancel remove an order from the middle of the queue in O(1).
  struct Level {
    Price price = 0;
    Qty total = 0;
    std::list<Resting> orders;
  };

  // std::map nodes never move, so a Level* stays valid until that level is erased (and a level is
  // only erased once it is empty, i.e. when nothing can point at it any more).
  using BidMap = std::map<Price, Level, std::greater<>>;  // best (highest) bid first
  using AskMap = std::map<Price, Level>;                  // best (lowest) ask first

  // Where an order lives, so cancel never has to search.
  struct Locator {
    Side side;
    Level* level;
    std::list<Resting>::iterator it;
  };

  void validate(const Order& order) const;
  void check_ts(Timestamp ts) const;

  // Trade `order` against `opposite` and return the quantity left unfilled.
  template <class Map>
  Qty match(Map& opposite, const Order& order, std::vector<Fill>& fills);

  // Place `qty` of `order` in the book at `price`, behind the earlier orders at that price.
  void rest(const Order& order, Price price, Qty qty);

  BidMap bids_;
  AskMap asks_;
  std::unordered_map<OrderId, Locator> index_;
  Timestamp last_ts_ = 0;
};

}  // namespace lobx
