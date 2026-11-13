#include "lobx/multi_book.hpp"

#include <numeric>
#include <stdexcept>

namespace lobx {

MultiBookEngine::MultiBookEngine(const std::vector<std::string>& symbols) {
  for (const auto& sym : symbols) {
    add_symbol(sym);
  }
}

void MultiBookEngine::add_symbol(const std::string& symbol) {
  if (books_.find(symbol) == books_.end()) {
    books_.emplace(symbol, OrderBook{});
  }
}

bool MultiBookEngine::has_symbol(const std::string& symbol) const {
  return books_.find(symbol) != books_.end();
}

std::vector<std::string> MultiBookEngine::symbols() const {
  std::vector<std::string> syms;
  syms.reserve(books_.size());
  for (const auto& [sym, _] : books_) {
    syms.push_back(sym);
  }
  return syms;
}

OrderBook& MultiBookEngine::get_book(const std::string& symbol) {
  auto it = books_.find(symbol);
  if (it == books_.end()) {
    throw std::out_of_range("Symbol not found: " + symbol);
  }
  return it->second;
}

const OrderBook& MultiBookEngine::get_book(const std::string& symbol) const {
  auto it = books_.find(symbol);
  if (it == books_.end()) {
    throw std::out_of_range("Symbol not found: " + symbol);
  }
  return it->second;
}

std::vector<Fill> MultiBookEngine::submit(const std::string& symbol, const Order& order) {
  return get_book(symbol).submit(order);
}

std::vector<Fill> MultiBookEngine::submit_many(const std::string& symbol,
                                               const std::vector<Order>& orders) {
  return get_book(symbol).submit_many(orders);
}

bool MultiBookEngine::cancel(const std::string& symbol, OrderId id, Timestamp ts) {
  return get_book(symbol).cancel(id, ts);
}

Snapshot MultiBookEngine::snapshot(const std::string& symbol, std::size_t depth) const {
  return get_book(symbol).snapshot(depth);
}

std::optional<Price> MultiBookEngine::best_bid(const std::string& symbol) const {
  return get_book(symbol).best_bid();
}

std::optional<Price> MultiBookEngine::best_ask(const std::string& symbol) const {
  return get_book(symbol).best_ask();
}

std::optional<double> MultiBookEngine::mid_price(const std::string& symbol) const {
  const auto bid = best_bid(symbol);
  const auto ask = best_ask(symbol);
  if (!bid.has_value() || !ask.has_value()) {
    return std::nullopt;
  }
  return static_cast<double>(*bid + *ask) / 2.0;
}

double MultiBookEngine::imbalance(const std::string& symbol, std::size_t depth) const {
  const Snapshot s = snapshot(symbol, depth);
  Qty bid_vol = 0;
  for (std::size_t i = 0; i < depth && i < s.bids.size(); ++i) {
    bid_vol += s.bids[i].second;
  }
  Qty ask_vol = 0;
  for (std::size_t i = 0; i < depth && i < s.asks.size(); ++i) {
    ask_vol += s.asks[i].second;
  }
  const Qty total = bid_vol + ask_vol;
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(bid_vol - ask_vol) / static_cast<double>(total);
}

std::size_t MultiBookEngine::resting_orders(const std::string& symbol) const {
  return get_book(symbol).resting_orders();
}

}  // namespace lobx
