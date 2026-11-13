#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "lobx/multi_book.hpp"

namespace {

void check(bool condition, const char* msg) {
  if (!condition) {
    std::cerr << "Check failed: " << msg << "\n";
    std::abort();
  }
}

void test_symbol_management() {
  lobx::MultiBookEngine mb({"SPY", "QQQ"});
  check(mb.has_symbol("SPY"), "has SPY");
  check(mb.has_symbol("QQQ"), "has QQQ");
  check(!mb.has_symbol("AAPL"), "not has AAPL");

  mb.add_symbol("AAPL");
  check(mb.has_symbol("AAPL"), "has AAPL after add");

  const auto syms = mb.symbols();
  check(syms.size() == 3, "symbols size 3");
}

void test_unknown_symbol_throws() {
  lobx::MultiBookEngine mb({"SPY"});
  bool caught = false;
  try {
    const lobx::Order order{1, 10, lobx::Side::Buy, lobx::OrderType::Limit, 10000, 50, 100};
    mb.submit("MSFT", order);
  } catch (const std::out_of_range&) {
    caught = true;
  }
  check(caught, "caught out_of_range for unknown symbol");
}

void test_multi_book_trading_and_metrics() {
  lobx::MultiBookEngine mb({"LEAD", "LAG"});

  // Insert bid on LEAD
  const lobx::Order bid_lead{1, 10, lobx::Side::Buy, lobx::OrderType::Limit, 5000, 100, 10};
  const auto fills1 = mb.submit("LEAD", bid_lead);
  check(fills1.empty(), "fills empty");
  check(mb.best_bid("LEAD").value_or(0) == 5000, "best bid LEAD is 5000");
  check(!mb.best_bid("LAG").has_value(), "LAG has no bids");

  // Insert ask on LEAD
  const lobx::Order ask_lead{2, 11, lobx::Side::Sell, lobx::OrderType::Limit, 5004, 300, 20};
  mb.submit("LEAD", ask_lead);
  check(mb.best_ask("LEAD").value_or(0) == 5004, "best ask LEAD is 5004");
  check(mb.mid_price("LEAD").value_or(0.0) == 5002.0, "mid price LEAD is 5002");

  // Check imbalance: (100 - 300) / 400 = -0.5
  const double imb = mb.imbalance("LEAD", 1);
  check(std::abs(imb - (-0.5)) < 1e-6, "imbalance is -0.5");

  // Insert matching order on LEAD
  const lobx::Order cross_lead{3, 12, lobx::Side::Buy, lobx::OrderType::Limit, 5004, 150, 30};
  const auto match_fills = mb.submit("LEAD", cross_lead);
  check(match_fills.size() == 1, "match fills size 1");
  check(match_fills[0].qty == 150, "match fill qty 150");
  check(match_fills[0].price == 5004, "match fill price 5004");

  // Cancel remaining ask
  check(mb.cancel("LEAD", 2, 40), "cancel succeeded");
  check(!mb.best_ask("LEAD").has_value(), "no remaining ask");
}

}  // namespace

int main() {
  test_symbol_management();
  test_unknown_symbol_throws();
  test_multi_book_trading_and_metrics();
  std::cout << "All MultiBookEngine C++ tests passed.\n";
  return 0;
}
