// Copyright 2023 Sirui Lu
#ifndef QUANT_H_
#define QUANT_H_

#include "type.h"
#include <cstddef>
#include <cstdint>

namespace quant {
constexpr size_t kSuperBlockSize = 256;

template <int kQuantBits> struct QuantBlock;

template <int kQuantBits>
void Quantize(const float *QUANT_RESTRICT input, size_t num_elements,
              QuantBlock<kQuantBits> *QUANT_RESTRICT output);
template <int kQuantBits>
void Dequantize(const QuantBlock<kQuantBits> *QUANT_RESTRICT input,
                size_t num_elements, float *QUANT_RESTRICT output);
template <int kQuantBits>
float DotProduct(const QuantBlock<kQuantBits> *QUANT_RESTRICT weights,
                 const QuantBlock<8> *QUANT_RESTRICT input,
                 size_t num_elements);

// 8 bit quantization is special so we include the specialization here.
template <> struct QuantBlock<8> {
  float scale; // delta
  int8_t quants[kSuperBlockSize];
  int16_t block_sum_of_quants[kSuperBlockSize / 16];
};

template <>
void Quantize<8>(const float *QUANT_RESTRICT input, size_t num_elements,
                 QuantBlock<8> *QUANT_RESTRICT output);
template <>
void Dequantize(const QuantBlock<8> *QUANT_RESTRICT input, size_t num_elements,
                float *QUANT_RESTRICT output);

} // namespace quant

#endif // QUANT_H_
