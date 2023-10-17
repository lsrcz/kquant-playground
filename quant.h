// Copyright 2023 Sirui Lu
#ifndef QUANT_H_
#define QUANT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace quant {
constexpr size_t kSuperBlockSize = 256;

template <int kQuantBits> struct QuantBlock;

template <int kQuantBits>
void Quantize(std::span<const float> input,
              std::span<QuantBlock<kQuantBits>> output);
template <int kQuantBits>
void Dequantize(std::span<const QuantBlock<kQuantBits>> input,
                std::span<float> output);

enum class CPUFeature {
  kNone = 0,
  kAVX2 = 1,
};

#ifdef __AVX2__
constexpr CPUFeature kDefaultCPUFeature = CPUFeature::kAVX2;
#else
constexpr CPUFeature kDefaultCPUFeature = CPUFeature::kNone;
#endif

template <int kQuantBits, CPUFeature kFeature = kDefaultCPUFeature>
float DotProduct(std::span<QuantBlock<kQuantBits>> weights,
                 std::span<QuantBlock<8>> input);

// 8 bit quantization is special so we include the specialization here.
template <> struct QuantBlock<8> {
  float scale; // delta
  std::array<int8_t, kSuperBlockSize> quants;
  std::array<int16_t, kSuperBlockSize / 16> block_sum_of_quants;
};

template <>
void Quantize<8>(std::span<const float> input, std::span<QuantBlock<8>> output);
template <>
void Dequantize(std::span<const QuantBlock<8>> input, std::span<float> output);

} // namespace quant

#endif // QUANT_H_
