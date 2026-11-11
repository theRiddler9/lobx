// Order book powered by a zero-allocation slab memory pool and intrusive doubly-linked queues.
#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "lobx/memory_pool.hpp"
#include "lobx/types.hpp"

namespace lobx {

struct SlabNode {
  OrderId id{0};
  Qty qty{0};
  Price price{0};
  Side side{Side::Buy};
  SlabNode* prev{nullptr};
  SlabNode* next{nullptr};
};

struct SlabLevel {
  Price price{0};
  Qty total_qty{0};
  SlabNode* head{nullptr};
  SlabNode* tail{nullptr};

  void append(SlabNode* node) noexcept {
    node->prev = tail;
    node->next = nullptr;
    if (tail)
      tail->next = node;
    else
      head = node;
    tail = node;
    total_qty += node->qty;
  }

  void remove(SlabNode* node) noexcept {
    if (node->prev)
      node->prev->next = node->next;
    else
      head = node->next;
    if (node->next)
      node->next->prev = node->prev;
    else
      tail = node->prev;
    total_qty -= node->qty;
  }
};

template <std::size_t MaxOrders = 100000>
class SlabOrderBook {
 public:
  using BidMap = std::map<Price, SlabLevel, std::greater<>>;
  using AskMap = std::map<Price, SlabLevel>;

  std::vector<Fill> submit(const Order& order) {
    if (order.qty <= 0) throw InvalidOrderError("qty must be > 0");
    std::vector<Fill> fills;
    Qty rem = (order.side == Side::Buy) ? match(asks_, order, fills) : match(bids_, order, fills);
    if (rem > 0 && order.type == OrderType::Limit && order.price.has_value()) {
      rest(order, *order.price, rem);
    }
    last_ts_ = std::max(last_ts_, order.ts);
    return fills;
  }

  bool cancel(OrderId id, Timestamp ts) {
    last_ts_ = std::max(last_ts_, ts);
    auto it = index_.find(id);
    if (it == index_.end()) return false;
    auto [lvl, node] = it->second;
    lvl->remove(node);
    if (lvl->total_qty == 0) {
      if (node->side == Side::Buy)
        bids_.erase(node->price);
      else
        asks_.erase(node->price);
    }
    pool_.deallocate(node);
    index_.erase(it);
    return true;
  }

  [[nodiscard]] Snapshot snapshot(std::size_t depth) const {
    Snapshot s{last_ts_, {}, {}};
    s.bids.reserve(depth);
    s.asks.reserve(depth);
    for (const auto& [px, lvl] : bids_) {
      if (s.bids.size() >= depth) break;
      s.bids.emplace_back(px, lvl.total_qty);
    }
    for (const auto& [px, lvl] : asks_) {
      if (s.asks.size() >= depth) break;
      s.asks.emplace_back(px, lvl.total_qty);
    }
    return s;
  }

  [[nodiscard]] std::optional<Price> best_bid() const {
    return bids_.empty() ? std::nullopt : std::optional<Price>(bids_.begin()->first);
  }
  [[nodiscard]] std::optional<Price> best_ask() const {
    return asks_.empty() ? std::nullopt : std::optional<Price>(asks_.begin()->first);
  }
  [[nodiscard]] std::size_t resting_orders() const noexcept { return index_.size(); }
  [[nodiscard]] std::size_t pool_active_count() const noexcept { return pool_.active_count(); }

 private:
  template <typename Map>
  Qty match(Map& opposite, const Order& order, std::vector<Fill>& fills) {
    Qty rem = order.qty;
    auto it = opposite.begin();
    while (it != opposite.end() && rem > 0) {
      if (order.type == OrderType::Limit && order.price.has_value()) {
        Price limit = *order.price;
        if (order.side == Side::Buy ? it->first > limit : it->first < limit) break;
      }
      SlabLevel& lvl = it->second;
      while (lvl.head && rem > 0) {
        SlabNode* resting = lvl.head;
        Qty trade_q = std::min(rem, resting->qty);
        fills.push_back(Fill{resting->id, order.id, order.side, resting->price, trade_q, order.ts});
        rem -= trade_q;
        resting->qty -= trade_q;
        lvl.total_qty -= trade_q;
        if (resting->qty == 0) {
          lvl.remove(resting);
          index_.erase(resting->id);
          pool_.deallocate(resting);
        }
      }
      it = (lvl.total_qty == 0) ? opposite.erase(it) : std::next(it);
    }
    return rem;
  }

  void rest(const Order& order, Price price, Qty qty) {
    if (index_.find(order.id) != index_.end()) throw DuplicateOrderIdError("duplicate order id");
    SlabNode* node = pool_.allocate(SlabNode{order.id, qty, price, order.side, nullptr, nullptr});
    if (!node) throw std::bad_alloc();
    if (order.side == Side::Buy) {
      auto& lvl = bids_[price];
      lvl.price = price;
      lvl.append(node);
      index_[order.id] = {&lvl, node};
    } else {
      auto& lvl = asks_[price];
      lvl.price = price;
      lvl.append(node);
      index_[order.id] = {&lvl, node};
    }
  }

  FixedPool<SlabNode, MaxOrders> pool_;
  BidMap bids_;
  AskMap asks_;
  std::unordered_map<OrderId, std::pair<SlabLevel*, SlabNode*>> index_;
  Timestamp last_ts_{0};
};

}  // namespace lobx
