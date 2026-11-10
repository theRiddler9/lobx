#include "lobx/features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lobx {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t k1s = 1000000000LL;
constexpr int64_t k5s = 5000000000LL;

double calc_imbalance(const Snapshot& s, std::size_t depth) noexcept {
  if (s.bids.empty() || s.asks.empty()) return NaN;
  double bq = 0, aq = 0;
  for (std::size_t i = 0; i < std::min(depth, s.bids.size()); ++i) bq += s.bids[i].second;
  for (std::size_t i = 0; i < std::min(depth, s.asks.size()); ++i) aq += s.asks[i].second;
  return (bq + aq > 0) ? (bq - aq) / (bq + aq) : NaN;
}

}  // namespace

FeatureMatrix extract_features(const std::vector<Snapshot>& snapshots,
                               const std::vector<Fill>& fills) {
  const std::size_t n = snapshots.size();
  FeatureMatrix mat{n, FeatureCol::Count, std::vector<double>(n * FeatureCol::Count, NaN)};
  if (n == 0) return mat;

  // Prefix sums for fills: O(1) trailing trade-flow calculation
  const std::size_t m = fills.size();
  std::vector<Timestamp> fill_ts(m);
  std::vector<double> pref_signed(m + 1, 0.0);
  std::vector<double> pref_vol(m + 1, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    fill_ts[i] = fills[i].ts;
    double signed_q = (fills[i].taker_side == Side::Buy) ? fills[i].qty : -fills[i].qty;
    pref_signed[i + 1] = pref_signed[i] + signed_q;
    pref_vol[i + 1] = pref_vol[i] + fills[i].qty;
  }

  std::vector<double> mids(n, NaN);

  for (std::size_t i = 0; i < n; ++i) {
    const auto& s = snapshots[i];
    double* row = &mat.data[i * FeatureCol::Count];

    if (!s.bids.empty() && !s.asks.empty()) {
      row[FeatureCol::Spread] = static_cast<double>(s.asks[0].first - s.bids[0].first);
      mids[i] = (s.bids[0].first + s.asks[0].first) / 2.0;
    }

    row[FeatureCol::Imbalance1] = calc_imbalance(s, 1);
    row[FeatureCol::Imbalance5] = calc_imbalance(s, 5);

    // Mid-price returns
    if (i >= 1 && !std::isnan(mids[i]) && !std::isnan(mids[i - 1])) row[FeatureCol::MidReturn1] = mids[i] - mids[i - 1];
    if (i >= 5 && !std::isnan(mids[i]) && !std::isnan(mids[i - 5])) row[FeatureCol::MidReturn5] = mids[i] - mids[i - 5];
    if (i >= 25 && !std::isnan(mids[i]) && !std::isnan(mids[i - 25])) row[FeatureCol::MidReturn25] = mids[i] - mids[i - 25];

    // Trailing trade-flow imbalance via binary search on prefix sums
    auto calc_flow = [&](int64_t win_ns) -> double {
      if (m == 0) return 0.0;
      auto hi_it = std::upper_bound(fill_ts.begin(), fill_ts.end(), s.ts);
      auto lo_it = std::upper_bound(fill_ts.begin(), fill_ts.end(), s.ts - win_ns);
      std::size_t hi = std::distance(fill_ts.begin(), hi_it);
      std::size_t lo = std::distance(fill_ts.begin(), lo_it);
      double vol = pref_vol[hi] - pref_vol[lo];
      return (vol > 0) ? (pref_signed[hi] - pref_signed[lo]) / vol : 0.0;
    };

    row[FeatureCol::TradeFlow1s] = calc_flow(k1s);
    row[FeatureCol::TradeFlow5s] = calc_flow(k5s);
  }

  return mat;
}

}  // namespace lobx
