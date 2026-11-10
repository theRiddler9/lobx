// High-performance vectorized C++ feature extraction engine.
#pragma once

#include <cstddef>
#include <vector>

#include "lobx/types.hpp"

namespace lobx {

namespace FeatureCol {
inline constexpr int Spread = 0;
inline constexpr int Imbalance1 = 1;
inline constexpr int Imbalance5 = 2;
inline constexpr int MidReturn1 = 3;
inline constexpr int MidReturn5 = 4;
inline constexpr int MidReturn25 = 5;
inline constexpr int TradeFlow1s = 6;
inline constexpr int TradeFlow5s = 7;
inline constexpr int Count = 8;
}  // namespace FeatureCol

struct FeatureMatrix {
  std::size_t rows{0};
  std::size_t cols{FeatureCol::Count};
  std::vector<double> data;  // flat row-major: data[r * cols + c]

  [[nodiscard]] double at(std::size_t r, int c) const noexcept {
    return data[r * cols + c];
  }
};

FeatureMatrix extract_features(const std::vector<Snapshot>& snapshots,
                               const std::vector<Fill>& fills);

}  // namespace lobx
