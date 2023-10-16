// Copyright 2023 Sirui Lu
#include "helper.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace quant {
QuantResultWithMinimum
QuantizeBlockWithMinimum(int max_quant_level,
                         std::span<const float> block_input,
                         std::span<uint8_t> quants) {
  float block_minimum = 0;
  float block_maximum = std::numeric_limits<float>::min();
  for (float value : block_input) {
    if (value < block_minimum) {
      block_minimum = value;
    }
    if (value > block_maximum) {
      block_maximum = value;
    }
  }
  // Special case: all x are the same.
  if (block_maximum == block_minimum) {
    std::fill(quants.begin(), quants.end(), 0);
    // the min is negated.
    return QuantResultWithMinimum{.scale = 0.F,
                                  .negated_minimum = -block_minimum};
  }

  float reciprocal_scale =
      static_cast<float>(max_quant_level) / (block_maximum - block_minimum);
  // scale = (max - min) / nmax
  float scale = 1 / reciprocal_scale;
  for (int i = 0; i < block_input.size(); ++i) {
    int quant = NearestInt(reciprocal_scale * (block_input[i] - block_minimum));
    quants[i] = std::max(0, std::min(max_quant_level, quant));
  }
  return QuantResultWithMinimum{.scale = scale,
                                .negated_minimum = -block_minimum};

  // Omitted the iterative improvement in the original code.
}
} // namespace quant
