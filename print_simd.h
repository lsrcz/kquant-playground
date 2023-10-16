// Copyright 2023 Sirui Lu
#ifndef PRINT_SIMD_H_
#define PRINT_SIMD_H_
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <immintrin.h>
#include <string>

#ifdef __SSE2__
inline std::string format_m128_hex_u8(__m128i xmm) {
  std::array<uint8_t, 16> vec{};
  memcpy(vec.data(), &xmm, sizeof(vec));
  return std::format(
      "[16 x u8]: [{:x} {:x} {:x} {:x} | {:x} {:x} {:x} {:x} | {:x} {:x} {:x} "
      "{:x} | {:x} {:x} {:x} {:x}]",
      vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8],
      vec[9], vec[10], vec[11], vec[12], vec[13], vec[14], vec[15]);
}

inline std::string print_m128_hex_u16(__m128i xmm) {
  std::array<uint16_t, 8> vec{};
  memcpy(vec.data(), &xmm, sizeof(vec));
  return std::format("[8 x u16]: [{:x} {:x} {:x} {:x} | {:x} {:x} {:x} {:x}]",
                     vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6],
                     vec[7]);
}

inline std::string print_m128_hex_u32(__m128i xmm) {
  std::array<uint32_t, 4> vec{};
  memcpy(vec.data(), &xmm, sizeof(vec));
  return std::format("[4 x u32]: [{:x} {:x} {:x} {:x}]\n", vec[0], vec[1],
                     vec[2], vec[3]);
}

inline std::string print_m128_hex_u64(__m128i xmm) {
  std::array<uint64_t, 2> vec{};
  memcpy(vec.data(), &xmm, sizeof(vec));
  return std::format("[2 x u64]: [{:x} {:x}]\n", vec[0], vec[1]);
}
#endif

#ifdef __AVX__
inline std::string print_m256_hex_u8(__m256i ymm) {
  std::array<uint8_t, 32> vec{};
  memcpy(vec.data(), &ymm, sizeof(vec));
  return std::format(
      "[32 x u8]: [{:x} {:x} {:x} {:x} | {:x} {:x} {:x} {:x} | {:x} {:x} {:x} "
      "{:x} | {:x} {:x} {:x} {:x} | {:x} {:x} {:x} {:x} | {:x} {:x} {:x} {:x} "
      "| {:x} {:x} {:x} {:x} | {:x} {:x} {:x} {:x}]",
      vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8],
      vec[9], vec[10], vec[11], vec[12], vec[13], vec[14], vec[15], vec[16],
      vec[17], vec[18], vec[19], vec[20], vec[21], vec[22], vec[23], vec[24],
      vec[25], vec[26], vec[27], vec[28], vec[29], vec[30], vec[31]);
}

inline std::string print_m256_hex_u16(__m256i ymm) {
  std::array<uint16_t, 16> vec{};
  memcpy(vec.data(), &ymm, sizeof(vec));
  return std::format(
      "[16 x u16]: [{:x} {:x} {:x} {:x} | {:x} {:x} {:x} {:x} | {:x} {:x} {:x} "
      "{:x} | {:x} {:x} {:x} {:x}]",
      vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8],
      vec[9], vec[10], vec[11], vec[12], vec[13], vec[14], vec[15]);
}

inline std::string print_m256_hex_u32(__m256i ymm) {
  std::array<uint32_t, 8> vec{};
  memcpy(vec.data(), &ymm, sizeof(vec));
  return std::format("[8 x u32]: [{:x} {:x} {:x} {:x} | {:x} {:x} {:x} {:x}]",
                     vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6],
                     vec[7]);
}

inline std::string print_m256_hex_u64(__m256i ymm) {
  std::array<uint64_t, 4> vec{};
  memcpy(vec.data(), &ymm, sizeof(vec));
  return std::format("[4 x u64]: [{:x} {:x} {:x} {:x}]\n", vec[0], vec[1],
                     vec[2], vec[3]);
}
#endif

#endif // PRINT_SIMD_H_
