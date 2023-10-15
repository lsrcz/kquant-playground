// Copyright 2023 Sirui Lu
#include "quant_2bit.h"
#include "common.h"
#include "helper.h"
#include "quant.h"
#include "type.h"
#include <algorithm>
#include <cassert>
#include <immintrin.h>

namespace quant {

template <>
void Quantize<2>(const float *QUANT_RESTRICT input, size_t num_elements,
                 QuantBlock<2> *QUANT_RESTRICT output) {
  assert(num_elements % kSuperBlockSize == 0);
  const int num_super_blocks = num_elements / kSuperBlockSize;

  const float kQuant4BitMaxQuantLevel = 15.f;

  for (int i = 0; i < num_super_blocks; i++) {
    uint8_t quants[kSuperBlockSize];
    // Minimums and scales for the blocks in the current super block.
    float negated_minimums[kSuperBlockSize / 16];
    float scales[kSuperBlockSize / 16];
    // As we are deducting the minimum, scales are always positive.
    float max_scale = 0;
    float max_negated_minimum = 0;
    for (int j = 0; j < kSuperBlockSize / 16; ++j) {
      // Quantize the current block.
      auto [current_block_scale, current_block_negated_minimum] =
          QuantizeBlockWithMinimum(16, 3, input + 16 * j, quants + 16 * j);
      scales[j] = current_block_scale;
      negated_minimums[j] = current_block_negated_minimum;
      if (current_block_scale > max_scale) {
        max_scale = current_block_scale;
      }
      if (current_block_negated_minimum > max_negated_minimum) {
        max_negated_minimum = current_block_negated_minimum;
      }
    }

    // The scales and minimums are furthered quantized to 4 bits.
    if (max_scale > 0) {
      float reciprocal_super_block_scale = kQuant4BitMaxQuantLevel / max_scale;
      for (int j = 0; j < kSuperBlockSize / 16; ++j) {
        int quantized_scale =
            NearestInt(reciprocal_super_block_scale * scales[j]);
        output[i].scales_and_negated_minimums[j] = quantized_scale;
      }
      output[i].super_block_scale_for_scale =
          Fp32ToFp16(max_scale / kQuant4BitMaxQuantLevel);
    } else {
      for (int j = 0; j < kSuperBlockSize / 16; ++j) {
        output[i].scales_and_negated_minimums[j] = 0;
      }
      output[i].super_block_scale_for_scale = Fp32ToFp16(0.f);
    }
    if (max_negated_minimum > 0) {
      float reciprocal_super_block_negated_minimum_scale =
          kQuant4BitMaxQuantLevel / max_negated_minimum;
      for (int j = 0; j < kSuperBlockSize / 16; ++j) {
        int quantized_negated_minimum = NearestInt(
            reciprocal_super_block_negated_minimum_scale * negated_minimums[j]);
        output[i].scales_and_negated_minimums[j] |=
            (quantized_negated_minimum << 4);
      }
      output[i].super_block_scale_for_minimum =
          Fp32ToFp16(max_negated_minimum / kQuant4BitMaxQuantLevel);
    } else {
      output[i].super_block_scale_for_minimum = Fp32ToFp16(0.f);
    }
    for (int j = 0; j < kSuperBlockSize / 16; ++j) {
      const float reconstructed_scale =
          Fp16ToFp32(output[i].super_block_scale_for_scale) *
          (output[i].scales_and_negated_minimums[j] & 0xF);
      if (!reconstructed_scale)
        continue;
      const float reconstructed_negated_minimum =
          Fp16ToFp32(output[i].super_block_scale_for_minimum) *
          (output[i].scales_and_negated_minimums[j] >> 4);
      for (int k = 0; k < 16; ++k) {
        int quant =
            NearestInt((input[16 * j + k] + reconstructed_negated_minimum) /
                       reconstructed_scale);
        quants[16 * j + k] = std::max(0, std::min(3, quant));
      }
    }

    for (int j = 0; j < kSuperBlockSize; j += 128) {
      for (int k = 0; k < 32; ++k) {
        output[i].quants[j / 4 + k] =
            quants[j + k] | (quants[j + k + 32] << 2) |
            (quants[j + k + 64] << 4) | (quants[j + k + 96] << 6);
      }
    }
    input += kSuperBlockSize;
  }
}

template <>
void Dequantize<2>(const QuantBlock<2> *QUANT_RESTRICT input,
                   size_t num_elements, float *QUANT_RESTRICT output) {
  assert(num_elements % kSuperBlockSize == 0);
  const int num_super_blocks = num_elements / kSuperBlockSize;

  for (int i = 0; i < num_super_blocks; ++i) {
    const float scale_for_scale =
        Fp16ToFp32(input[i].super_block_scale_for_scale);
    const float scale_for_minimum =
        Fp16ToFp32(input[i].super_block_scale_for_minimum);
    const uint8_t *quants = input[i].quants;

    for (int j = 0; j < 2; ++j) {
      QUANT_PRAGMA_UNROLL
      for (int k = 0; k < 4; ++k) {
        QUANT_PRAGMA_UNROLL
        for (int l = 0; l < 2; ++l) {
          int block_id = j * 8 + k * 2 + l;

          int quants_addr_offset = j * 32 + l * 16;
          int quants_shift = k * 2;

          uint8_t scale_and_negated_minimum =
              input[i].scales_and_negated_minimums[block_id];
          float scale = scale_for_scale * (scale_and_negated_minimum & 0xF);
          float negated_minimum =
              scale_for_minimum * (scale_and_negated_minimum >> 4);
          for (int m = 0; m < 16; ++m) {
            *output++ =
                scale *
                    ((quants[quants_addr_offset + m] >> quants_shift) & 0b11) -
                negated_minimum;
          }
        }
      }
    }
  }
}

#ifdef __AVX2__
// For each 16-bit block with block_id
//
// result is
//   ( weights.super_block_scale_for_minimum *
//     weights.minimum[block_id] *
//     input.block_sum[block_id] ) +
//   ( undefined )
float DotProduct2BitAVX2(const QuantBlock<2> *QUANT_RESTRICT weights,
                         const QuantBlock<8> *QUANT_RESTRICT input,
                         size_t num_elements) {
  assert(num_elements % kSuperBlockSize == 0);
  const int num_super_blocks = num_elements / kSuperBlockSize;

  // [32 x i8], filled with 0b00000011
  const __m256i kQuantMask = _mm256_set1_epi8(3);
  // [16 x i8], filled with 0b00001111
  const __m128i kScaleAndMinimumMask = _mm_set1_epi8(0xF);

  __m256 accumulator = _mm256_setzero_ps();

  for (int i = 0; i < num_super_blocks; ++i) {

    // Step 1: we multiply the scales for inputs and weights.
    const float multiplied_super_block_scale_for_scale =
        input[i].scale * Fp16ToFp32(weights[i].super_block_scale_for_scale);
    const float negated_multiplied_super_block_scale_for_minimum =
        -input[i].scale * Fp16ToFp32(weights[i].super_block_scale_for_minimum);

    // Step 2: Load the scales and negated minimums for the weights.
    const __m128i weights_scales_and_negated_minimums =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(
            weights[i].scales_and_negated_minimums));
    // [16 x i8], lower-bits for each 8-bit element is the scale
    const __m128i weights_scales = _mm_and_si128(
        weights_scales_and_negated_minimums, kScaleAndMinimumMask);
    // [16 x i8], lower-bits for each 8-bit element is the negated minimum
    const __m128i weights_negated_minimums =
        _mm_and_si128(_mm_srli_epi16(weights_scales_and_negated_minimums, 4),
                      kScaleAndMinimumMask);

    // Step 3: accumulate the minimums to the accumulator. We need to multiply
    // the minimums with the block sums.
    const __m256i block_minimum_offset =
        _mm256_add_epi16(_mm256_cvtepi8_epi16(weights_negated_minimums),
                         _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                             input[i].block_sum_of_quants)));
    accumulator = _mm256_fmadd_ps(
        _mm256_broadcast_ss(&negated_multiplied_super_block_scale_for_minimum),
        _mm256_cvtepi32_ps(block_minimum_offset), accumulator);

    // [16 x i16], lower 4 bits for each 16-bit element is the scale
    const __m256i weights_scales_epi16 = _mm256_cvtepi8_epi16(weights_scales);
    // [8 x i16], [7:0] is scale[7] to scale[0]
    const __m128i weights_scales_epi16_low =
        _mm256_extracti128_si256(weights_scales_epi16, 0);
    // [8 x i16], [7:0] is scale[15] to scale[8]
    const __m128i weights_scales_epi16_high =
        _mm256_extracti128_si256(weights_scales_epi16, 1);
    // weights_scales_packed[0] is [16 x i16], element i is scale[i % 8]
    // weights_scales_packed[1] is [16 x i16], element i is scale[i % 8 + 8]
    //
    // For the first __m256i element, the layout is
    //
    // < scale[7], scale[6], scale[5], scale[4], scale[3], scale[2], scale[1],
    //   scale[0], scale[7], scale[6], scale[5], scale[4], scale[3], scale[2],
    //   scale[1], scale[0] >
    //
    // when interpreted as 8-bit integers,
    //
    // < 0, scale[7], 0, scale[6], 0, scale[5], 0, scale[4], 0, scale[3], 0,
    //   scale[2], 0, scale[1], 0, scale[0], ... >
    const __m256i weights_scales_packed[2] = {
        Pack128Into256(weights_scales_epi16_low, weights_scales_epi16_low),
        Pack128Into256(weights_scales_epi16_high, weights_scales_epi16_high)};

    const uint8_t *QUANT_RESTRICT weights_quants = weights[i].quants;
    const int8_t *QUANT_RESTRICT input_quants = input[i].quants;

    __m256i sum = _mm256_setzero_si256();
    for (int j = 0; j < 2; ++j) {
      // [128 x i2]
      // weight <96, 64, 32, 0, 97, 65, 33, 1, 98, 66, 34, 2, ...>
      const __m256i weight_bits = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(weights_quants + j * 32));

      // [32 x i8]
      // weight <31, 30, 29, 28, 27, 26, 25, 24, ..., 0>
      const __m256i weight_0 = _mm256_and_si256(weight_bits, kQuantMask);
      // weight <63, 62, 61, 60, 59, 58, 57, 56, ..., 32>
      const __m256i weight_1 =
          _mm256_and_si256(_mm256_srli_epi16(weight_bits, 2), kQuantMask);
      // weight <95, 94, 93, 92, 91, 90, 89, 88, ..., 64>
      const __m256i weight_2 =
          _mm256_and_si256(_mm256_srli_epi16(weight_bits, 4), kQuantMask);
      // weight <127, 126, 125, 124, 123, 122, 121, 120, ..., 96>
      const __m256i weight_3 =
          _mm256_and_si256(_mm256_srli_epi16(weight_bits, 6), kQuantMask);

      const __m256i input_0 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(input_quants + j * 128));
      const __m256i input_1 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(input_quants + j * 128 + 32));
      const __m256i input_2 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(input_quants + j * 128 + 64));
      const __m256i input_3 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(input_quants + j * 128 + 96));

      // [16 x i16]
      // < weight[0] * input[0] + weight[1] * input[1],
      //   weight[2] * input[2] + weight[3] * input[3],
      //   ...
      //   weight[30] * input[30] + weight[31] * input[31]
      // >
      __m256i p0 = _mm256_maddubs_epi16(weight_0, input_0);
      __m256i p1 = _mm256_maddubs_epi16(weight_1, input_1);
      __m256i p2 = _mm256_maddubs_epi16(weight_2, input_2);
      __m256i p3 = _mm256_maddubs_epi16(weight_3, input_3);

      static constexpr uint8_t kShuffleMap[128] = {
          0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
          2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
          4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
          6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
          8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
          10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
          12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
          14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
      };

      // Quantized products, times the block scales from the weights,
      // adjacent pairs horizontally summed.
      // [8 x i32]
      //
      // < sum_{i=0..3}(weight[i] * input[i]) * scale[0],
      //   sum_{i=4..7}(weight[i] * input[i]) * scale[0],
      //   sum_{i=8..11}(weight[i] * input[i]) * scale[0],
      //   sum_{i=12..15}(weight[i] * input[i]) * scale[0]
      //   sum_{i=16..19}(weight[i] * input[i]) * scale[1],
      //   sum_{i=20..23}(weight[i] * input[i]) * scale[1],
      //   sum_{i=24..27}(weight[i] * input[i]) * scale[1],
      //   sum_{i=28..31}(weight[i] * input[i]) * scale[1]
      // >
      p0 = _mm256_madd_epi16(
          _mm256_shuffle_epi8(
              weights_scales_packed[j],
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i *>(kShuffleMap))),
          p0);
      p1 = _mm256_madd_epi16(
          _mm256_shuffle_epi8(
              weights_scales_packed[j],
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i *>(kShuffleMap) + 1)),
          p1);
      p2 = _mm256_madd_epi16(
          _mm256_shuffle_epi8(
              weights_scales_packed[j],
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i *>(kShuffleMap) + 2)),
          p2);
      p3 = _mm256_madd_epi16(
          _mm256_shuffle_epi8(
              weights_scales_packed[j],
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i *>(kShuffleMap) + 3)),
          p3);

      p0 = _mm256_add_epi32(p0, p1);
      p2 = _mm256_add_epi32(p2, p3);

      sum = _mm256_add_epi32(sum, _mm256_add_epi32(p0, p2));
    }

    accumulator = _mm256_fmadd_ps(
        _mm256_broadcast_ss(&multiplied_super_block_scale_for_scale),
        _mm256_cvtepi32_ps(sum), accumulator);
  }

  return HSum8Floats(accumulator);
}
#endif

template <>
float DotProduct<2>(const QuantBlock<2> *QUANT_RESTRICT weights,
                    const QuantBlock<8> *QUANT_RESTRICT input,
                    size_t num_elements) {
#ifdef __AVX2__
  return DotProduct2BitAVX2(weights, input, num_elements);
#else
  static_assert(false, "Not implemented.");
#endif
}

} // namespace quant
