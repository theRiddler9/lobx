#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "lobx/order_book.hpp"
#include "lobx/types.hpp"

namespace lobx {

class MultiBookEngine {
 public:
  MultiBookEngine() = default;
  explicit MultiBookEngine(const std::vector<std::string>& symbols);

  void add_symbol(const std::string& symbol);
  [[nodiscard]] bool has_symbol(const std::string& symbol) const;
  [[nodiscard]] std::vector<std::string> symbols() const;

  std::vector<Fill> submit(const std::string& symbol, const Order& order);
  std::vector<Fill> submit_many(const std::string& symbol, const std::vector<Order>& orders);
  bool cancel(const std::string& symbol, OrderId id, Timestamp ts);

  [[nodiscard]] Snapshot snapshot(const std::string& symbol, std::size_t depth) const;
  [[nodiscard]] std::optional<Price> best_bid(const std::string& symbol) const;
  [[nodiscard]] std::optional<Price> best_ask(const std::string& symbol) const;
  [[nodiscard]] std::optional<double> mid_price(const std::string& symbol) const;
  [[nodiscard]] double imbalance(const std::string& symbol, std::size_t depth) const;
  [[nodiscard]] std::size_t resting_orders(const std::string& symbol) const;

 private:
  std::unordered_map<std::string, OrderBook> books_;
  OrderBook& get_book(const std::string& symbol);
  const OrderBook& get_book(const std::string& symbol) const;
};

}  // namespace lobx
