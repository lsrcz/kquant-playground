// Copyright 2023 Sirui Lu
#ifndef QUANT_H_
#define QUANT_H_

#include "helper.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace quant {
constexpr size_t kSuperBlockSize = 256;

enum class CPUFeature {
  kNone = 0,
  kAVX2 = 1,
};

#ifdef __AVX2__
constexpr CPUFeature kDefaultCPUFeature = CPUFeature::kAVX2;
#else
constexpr CPUFeature kDefaultCPUFeature = CPUFeature::kNone;
#endif

template <int kQuantBits, CPUFeature kFeature> struct QuantBlock;

template <int kQuantBits, CPUFeature kFeature = kDefaultCPUFeature>
void Quantize(std::span<const float> input,
              std::span<QuantBlock<kQuantBits, kFeature>> output) = delete;
template <int kQuantBits, CPUFeature kFeature = kDefaultCPUFeature>
void Dequantize(std::span<const QuantBlock<kQuantBits, kFeature>> input,
                std::span<float> output) = delete;

template <int kQuantBits, CPUFeature kFeature = kDefaultCPUFeature>
float DotProduct(std::span<QuantBlock<kQuantBits, kFeature>> weights,
                 std::span<QuantBlock<8, kFeature>> input) = delete;

// 8 bit quantization is special so we include the specialization here.
template <CPUFeature kFeature> struct QuantBlock<8, kFeature> {
  float scale; // delta
  std::array<int8_t, kSuperBlockSize> quants;
  std::array<int16_t, kSuperBlockSize / 16> block_sum_of_quants;
};

template <int kQuantBits, CPUFeature kFeature>
  requires(kQuantBits == 8)
void Quantize(std::span<const float> input,
              std::span<QuantBlock<8, kFeature>> output) {
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

template <int kQuantBits, CPUFeature kFeature>
  requires(kQuantBits == 8)
void Dequantize(std::span<const QuantBlock<8, kFeature>> input,
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

#endif // QUANT_H_
