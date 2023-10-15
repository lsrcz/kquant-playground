// Copyright 2023 Sirui Lu
#ifndef COMMON_H_
#define COMMON_H_

#define QUANT_ALWAYS_INLINE inline __attribute__((always_inline))
#define QUANT_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define QUANT_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define QUANT_NOINLINE __attribute__((noinline))
#define QUANT_RESTRICT __restrict__
#define QUANT_PRAGMA_UNROLL _Pragma("unroll")

#endif // COMMON_H_
