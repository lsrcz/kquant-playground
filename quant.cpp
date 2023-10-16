// Copyright 2023 Sirui Lu

#include "quant.h"
#include "helper.h"
#include <algorithm>
#include <cassert>

namespace quant {
template <>
void Quantize<8>(std::span<const float> input,
                 std::span<QuantBlock<8>> output) {
  const size_t num_elements = input.size();
  assert(num_elements % kSuperBlockSize == 0);
  const size_t num_super_blocks = output.size();
  assert(num_super_blocks == num_elements / kSuperBlockSize);

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
    if (max_abs_value == 0) {
      output[i].scale = 0;
      memset(output[i].quants.data(), 0, kSuperBlockSize);
      input = input.subspan(kSuperBlockSize);
      continue;
    }
    const float reciprocal_scale = -128.F / value_with_max_abs;
    for (int j = 0; j < kSuperBlockSize; ++j) {
      output[i].quants[j] = static_cast<int8_t>(
          std::min(127, NearestInt(reciprocal_scale * input[j])));
    }
    for (int j = 0; j < kSuperBlockSize / 16; ++j) {
      int16_t sum = 0;
      for (int k = 0; k < 16; ++k) {
        sum += static_cast<int16_t>(output[i].quants[j * 16 + k]);
      }
      output[i].block_sum_of_quants[j] = sum;
    }
    output[i].scale = 1 / reciprocal_scale;
    input = input.subspan(kSuperBlockSize);
  }
}

template <>
void Dequantize<8>(std::span<const QuantBlock<8>> input,
                   std::span<float> output) {
  const size_t num_super_blocks = input.size();
  assert(output.size() == num_super_blocks * kSuperBlockSize);
  const size_t num_elements = output.size();

  for (int i = 0; i < num_super_blocks; ++i) {
    for (int j = 0; j < kSuperBlockSize; ++j) {
      output[i * kSuperBlockSize + j] =
          input[i].scale * static_cast<float>(input[i].quants[j]);
    }
  }
}
} // namespace quant
