cmake_minimum_required(VERSION 3.12)
project("kquant-playground" C CXX)

set(CMAKE_CXX_STANDARD 20)

option(USE_FMA "Use FMA" ON)
option(USE_AVX2 "Use AVX2" ON)
option(USE_F16C "Use F16C" ON)

if (USE_F16C)
  add_compile_options(-mf16c)
endif()
if (USE_FMA)
  add_compile_options(-mfma)
endif()
if (USE_AVX2)
  add_compile_options(-mavx2)
endif()

add_library(quant OBJECT
            quant_2bit.cpp
            helper.cpp
            quant.cpp
            )
