// Copyright 2023 Sirui Lu

#include "quant.h"
#include "helper.h"
#include <algorithm>
#include <cassert>

namespace quant {
template <>
void Quantize<8>(const float *QUANT_RESTRICT input, size_t num_elements,
                 QuantBlock<8> *QUANT_RESTRICT output) {
  assert(num_elements % kSuperBlockSize == 0);
  const int num_super_blocks = num_elements / kSuperBlockSize;

  for (int i = 0; i < num_super_blocks; ++i) {

    float value_with_max_abs = 0;
    float max_abs_value = 0;
    for (int j = 0; j < kSuperBlockSize; ++j) {
      float abs_value = std::abs(input[j]);
      if (abs_value > max_abs_value) {
        max_abs_value = abs_value;
        value_with_max_abs = input[j];
      }
    }
    if (!max_abs_value) {
      output[i].scale = 0;
      memset(output[i].quants, 0, kSuperBlockSize);
      input += kSuperBlockSize;
      continue;
    }
    const float reciprocal_scale = -128.f / value_with_max_abs;
    for (int j = 0; j < kSuperBlockSize; ++j) {
      output[i].quants[j] =
          std::min(127, NearestInt(reciprocal_scale * input[j]));
    }
    for (int j = 0; j < kSuperBlockSize / 16; ++j) {
      int sum = 0;
      for (int k = 0; k < 16; ++k) {
        sum += output[i].quants[j * 16 + k];
      }
      output[i].block_sum_of_quants[j] = sum;
    }
    output[i].scale = 1 / reciprocal_scale;
    input += kSuperBlockSize;
  }
}
template <>
void Dequantize(const QuantBlock<8> *QUANT_RESTRICT input, size_t num_elements,
                float *QUANT_RESTRICT output) {
  assert(num_elements % kSuperBlockSize == 0);
  const int num_super_blocks = num_elements / kSuperBlockSize;

  for (int i = 0; i < num_super_blocks; ++i) {
    for (int j = 0; j < kSuperBlockSize; ++j) {
      output[j] = input[i].scale * input[i].quants[j];
    }
  }
}
} // namespace quant
