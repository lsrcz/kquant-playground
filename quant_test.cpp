// Copyright 2023 Sirui Lu
#include "quant.h"
#include "quant_2bit.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <random>
#include <vector>

template <int kQuantBits_, quant::CPUFeature kFeature_, size_t kSuperBlockSize_,
          int kNumBlock_>
struct QuantDotTestType {
  static constexpr int kQuantBits = kQuantBits_;
  static constexpr quant::CPUFeature kFeature = kFeature_;
  static constexpr size_t kSuperBlockSize = kSuperBlockSize_;
  static constexpr int kNumBlock = kNumBlock_;
};

template <typename QuantDotTestType> class QuantDotTest : public testing::Test {
public:
  void SetUp() override {
    std::default_random_engine eng{};
    std::uniform_real_distribution<> dis(-100, 100);
    auto gen = [&]() { return dis(eng); };
    std::generate(std::begin(weight), std::end(weight), gen);
    std::generate(std::begin(input), std::end(input), gen);

    quant::Quantize<kQuantBits, kFeature, kSuperBlockSize>(
        std::span{weight}, std::span{quant_weight});

    quant::Quantize<8, kFeature, kSuperBlockSize>(std::span{input},
                                                  std::span{quant_input});
    quant::Dequantize<kQuantBits, kFeature, kSuperBlockSize>(
        std::span{quant_weight}, std::span{dequantized_weight});
    quant::Dequantize<8, kFeature, kSuperBlockSize>(
        std::span{quant_input}, std::span{dequantized_input});

    expected = 0;
    for (int i = 0; i < kNumElements; ++i) {
      expected += dequantized_weight[i] * dequantized_input[i];
    }
  }
  static constexpr int kQuantBits = QuantDotTestType::kQuantBits;
  static constexpr quant::CPUFeature kFeature = QuantDotTestType::kFeature;
  static constexpr size_t kSuperBlockSize = QuantDotTestType::kSuperBlockSize;
  static constexpr int kNumSuperBlock = QuantDotTestType::kNumBlock;

  static constexpr int kNumElements = kNumSuperBlock * kSuperBlockSize;
  std::array<float, kNumElements> weight{};
  std::array<float, kNumElements> input{};
  std::array<quant::QuantBlock<kQuantBits, kFeature, kSuperBlockSize>,
             kNumSuperBlock>
      quant_weight{};
  std::array<quant::QuantBlock<8, kFeature, kSuperBlockSize>, kNumSuperBlock>
      quant_input{};
  std::array<float, kNumElements> dequantized_weight{};
  std::array<float, kNumElements> dequantized_input{};
  float expected{};
};

using QuantDotTestTypes =
    testing::Types<QuantDotTestType<2, quant::CPUFeature::kNone, 256, 8>
#ifdef __AVX2__
                   ,
                   QuantDotTestType<2, quant::CPUFeature::kAVX2, 256, 8>
#endif
                   >;

TYPED_TEST_SUITE(QuantDotTest, QuantDotTestTypes);

TYPED_TEST(QuantDotTest, Dot) {
  float actual = quant::DotProduct<TypeParam::kQuantBits, TypeParam::kFeature,
                                   TypeParam::kSuperBlockSize>(
      this->quant_weight, this->quant_input);
  EXPECT_THAT(actual, testing::FloatEq(this->expected));
}
