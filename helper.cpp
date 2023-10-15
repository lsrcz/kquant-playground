// Copyright 2023 Sirui Lu
#include "helper.h"
#include "common.h"
#include <algorithm>
#include <cmath>

namespace quant {
QuantResultWithMinimum
QuantizeBlockWithMinimum(int num_of_elements_in_block, int max_quant_level,
                         const float *QUANT_RESTRICT block_input,
                         /* output */ uint8_t *QUANT_RESTRICT quants) {
  float block_minimum = block_input[0];
  float block_maximum = block_input[0];
  for (int i = 1; i < num_of_elements_in_block; ++i) {
    if (block_input[i] < block_minimum) {
      block_minimum = block_input[i];
    }
    if (block_input[i] > block_maximum) {
      block_maximum = block_input[i];
    }
  }
  // minimum could be negative or 0.
  if (block_minimum > 0)
    block_minimum = 0;
  // Special case: all x are the same.
  if (block_maximum == block_minimum) {
    for (int i = 0; i < num_of_elements_in_block; ++i) {
      quants[i] = 0;
    }
    // the min is negated.
    return QuantResultWithMinimum{.scale = 0.f,
                                  .negated_minimum = -block_minimum};
  }

  float iscale = max_quant_level / (block_maximum - block_minimum);
  // scale = (max - min) / nmax
  float scale = 1 / iscale;
  for (int i = 0; i < num_of_elements_in_block; ++i) {
    int l = NearestInt(iscale * (block_input[i] - block_minimum));
    quants[i] = std::max(0, std::min(max_quant_level, l));
  }
  return QuantResultWithMinimum{.scale = scale,
                                .negated_minimum = -block_minimum};

  // Omitted the iterative improvement in the original code.
}
} // namespace quant
