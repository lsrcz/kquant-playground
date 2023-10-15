// Copyright 2023 Sirui Lu
#ifndef HELPER_H_
#define HELPER_H_

#include "common.h"
#include <cassert>
#include <cstdint>
#include <cstring>

namespace quant {

// Copied from llama.cpp
QUANT_ALWAYS_INLINE int NearestInt(float value) {
  assert(value <= 4194303.f);
  float res = value + 12582912.f;
  int i;
  memcpy(&i, &res, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

struct QuantResultWithMinimum {
  float scale;
  float negated_minimum;
};

QuantResultWithMinimum
QuantizeBlockWithMinimum(int num_of_elements_in_block, int max_quant_level,
                         const float *QUANT_RESTRICT block_input,
                         /* output */ uint8_t *QUANT_RESTRICT quants);

} // namespace quant

#endif // HELPER_H_
