// Copyright 2023 Sirui Lu
#ifndef HELPER_H_
#define HELPER_H_

#include "common.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

namespace quant {

// Copied from llama.cpp
QUANT_ALWAYS_INLINE int NearestInt(float value) {
  assert(value <= 4194303.f);
  float res = value + 12582912.F;
  uint32_t int_res = 0;
  memcpy(&int_res, &res, sizeof(int));
  return static_cast<int>((int_res & static_cast<uint32_t>(0x007fffff)) -
                          0x00400000);
}

struct QuantResultWithMinimum {
  float scale;
  float negated_minimum;
};

QuantResultWithMinimum
QuantizeBlockWithMinimum(int max_quant_level,
                         std::span<const float> block_input,
                         // const float *QUANT_RESTRICT block_input,
                         // int num_of_elements_in_block,
                         std::span<uint8_t> quants);
// /* output */ uint8_t *QUANT_RESTRICT quants);

} // namespace quant

#endif // HELPER_H_
