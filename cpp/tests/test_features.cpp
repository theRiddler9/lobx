// Unit tests for C++ feature extraction engine.
#include <cmath>
#include <iostream>
#include <vector>

#include "lobx/features.hpp"
#include "lobx/types.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      ++g_failures;                                                              \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
    }                                                                            \
  } while (0)

using namespace lobx;

void test_feature_extraction() {
  std::vector<Snapshot> snapshots;
  // Snapshots spaced by 200 ms (200,000,000 ns)
  for (int i = 0; i < 30; ++i) {
    Timestamp ts = (i + 1) * 200000000LL;
    Price bid_px = 10000 + i;
    Price ask_px = bid_px + 2;  // spread = 2 ticks
    std::vector<PriceLevel> bids = {{bid_px, 30}, {bid_px - 1, 20}};
    std::vector<PriceLevel> asks = {{ask_px, 10}, {ask_px + 1, 40}};
    snapshots.push_back(Snapshot{ts, bids, asks});
  }

  // Fills: one buyer-initiated trade of 50 shares at ts = 500,000,000 ns
  std::vector<Fill> fills = {
      Fill{1, 2, Side::Buy, 10005, 50, 500000000LL},
  };

  FeatureMatrix mat = extract_features(snapshots, fills);
  CHECK(mat.rows == 30);
  CHECK(mat.cols == 8);

  // Check row 0
  CHECK(mat.at(0, FeatureCol::Spread) == 2.0);
  // Imbalance depth 1: (30 - 10) / (30 + 10) = 20 / 40 = 0.5
  CHECK(std::abs(mat.at(0, FeatureCol::Imbalance1) - 0.5) < 1e-6);
  // Imbalance depth 5: sum bids = 50, sum asks = 50 -> 0.0
  CHECK(std::abs(mat.at(0, FeatureCol::Imbalance5) - 0.0) < 1e-6);
  // Returns at row 0 are NaN
  CHECK(std::isnan(mat.at(0, FeatureCol::MidReturn1)));
  CHECK(std::isnan(mat.at(0, FeatureCol::MidReturn5)));
  CHECK(std::isnan(mat.at(0, FeatureCol::MidReturn25)));

  // Check row 1 return (mid[1] - mid[0])
  // mid[0] = (10000 + 10002) / 2 = 10001
  // mid[1] = (10001 + 10003) / 2 = 10002
  // return = 1.0 tick
  CHECK(std::abs(mat.at(1, FeatureCol::MidReturn1) - 1.0) < 1e-6);

  // Check row 25 return (mid[25] - mid[0] = 25.0 ticks)
  CHECK(std::abs(mat.at(25, FeatureCol::MidReturn25) - 25.0) < 1e-6);

  // Trade flow: snapshot at index 2 is ts = 600,000,000 ns.
  // 1s window is (600ms - 1000ms, 600ms] = (0, 600ms].
  // The fill at 500ms is in this window (buyer-initiated 50 shares).
  // Flow = +50 / 50 = 1.0
  CHECK(std::abs(mat.at(2, FeatureCol::TradeFlow1s) - 1.0) < 1e-6);
}

}  // namespace

int main() {
  test_feature_extraction();
  if (g_failures > 0) {
    std::cerr << g_failures << " tests failed\n";
    return 1;
  }
  std::cout << "all C++ feature tests passed\n";
  return 0;
}
