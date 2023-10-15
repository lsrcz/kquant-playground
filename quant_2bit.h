// Copyright 2023 Sirui Lu
//
// The memory layout notation in this file follows the convention:
//   <a, b, c, d> means that a is placed at the higher address, and d is placed
//   at the lower address, assuming lower-endian machine like x86.
#ifndef QUANT_2BIT_H_
#define QUANT_2BIT_H_

#include "quant.h"

namespace quant {
template <> struct QuantBlock<2> {
  // 16 blocks per super block, quantized with 4 bits.
  // Lower 4 bits are for scales, higher 4 bits are for negated minimums.
  //
  // Memory layout:
  //   scales_and_negated_minimums[i] is
  //     <negated_minimums[i], scales[i]>
  uint8_t scales_and_negated_minimums[kSuperBlockSize / 16];
  // 256 quants, packed into 512 bits (64 bytes).
  //
  // Memory layout:
  //   For the first 32 bytes:
  //     quants[i] packs 4 quants:
  //       < element[i + 96],
  //         element[i + 64],
  //         element[i + 32],
  //         element[i]
  //       >
  //   For the second 32 bytes:
  //     quants[i] packs 4 quants:
  //       < element[i + 128 + 96],
  //         element[i + 128 + 64],
  //         element[i + 128 + 32],
  //         element[i + 128]
  //       >
  //
  // element [0:16]    -> quants[0:16],  bit [0:2]
  // element [16:32]   -> quants[16:32], bit [0:2]
  // element [32:48]   -> quants[0:16],  bit [2:4]
  // element [48:64]   -> quants[16:32], bit [2:4]
  // ...
  // element [128:144] -> quants[32:48], bit [0:2]
  // element [144:160] -> quants[48:64], bit [0:2]
  // element [160:176] -> quants[32:48], bit [2:4]
  // element [176:192] -> quants[48:64], bit [2:4]
  // ...
  uint8_t quants[kSuperBlockSize / 4];
  // Super block scale for scales
  fp16_t super_block_scale_for_scale;
  // Super block scale for minimums
  fp16_t super_block_scale_for_minimum;
};

template <>
void Quantize<2>(const float *QUANT_RESTRICT input, size_t num_elements,
                 QuantBlock<2> *QUANT_RESTRICT output);
template <>
void Dequantize<2>(const QuantBlock<2> *QUANT_RESTRICT input,
                   size_t num_elements, float *QUANT_RESTRICT output);
template <>
float DotProduct<2>(const QuantBlock<2> *QUANT_RESTRICT weights,
                    const QuantBlock<8> *QUANT_RESTRICT input,
                    size_t num_elements);

} // namespace quant

#endif // QUANT_2BIT_H_
