// Copyright 2023 Sirui Lu
#ifndef TYPE_H_
#define TYPE_H_

#include "common.h"
#include <cstdint>
#include <immintrin.h>

namespace quant {
using fp16_t = uint16_t;

QUANT_ALWAYS_INLINE float Fp16ToFp32(fp16_t fp16) { return _cvtsh_ss(fp16); }

QUANT_ALWAYS_INLINE fp16_t Fp32ToFp16(float fp32) { return _cvtss_sh(fp32, 0); }

QUANT_ALWAYS_INLINE __m256i Pack128Into256(__m128i high, __m128i low) {
  return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
}

QUANT_ALWAYS_INLINE float HSum8Floats(__m256 packed_fp16) {
  // x = [a, b, c, d, e, f, g, h]
  // [255:128] of x
  // [a, b, c, d]
  __m128 res = _mm256_extractf128_ps(packed_fp16, 1);
  // [a + e, b + f, c + g, d + h]
  res = _mm_add_ps(res, _mm256_castps256_ps128(packed_fp16));
  // [..., ..., a + c + e + g, b + d + f + h]
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  // [..., ..., ..., a + b + c + d + e + f + g + h]
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}
} // namespace quant

#endif // TYPE_H_
