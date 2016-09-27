# simd types

This document is an overview of the types (and their respective namespaces)
provided by the simd header. Note, there is also the `simd::simd_type` template
alias for defining types other than those found in this list. Additionally,
there are copies of each type in an

contents
--------

* [types in the `simd::common` inline namespace](#inline_namespace_common)
* [x86/x86_64 MMX types in the `simd::mmx` namespace](#mmx)
* [x86/x86_64 SSE types in the `simd::sse` namespace](#sse)
* [x86/x86_64 SSE2 types in the `simd::sse2` namespace](#sse2)
* [x86/x86_64 SSE3 types in the `simd::sse3` namespace](#sse3)
* [x86/x86_64 SSSE3 types in the `simd::ssse3` namespace](#ssse3)
* [x86/x86_64 SSE4, SSE4.1, and SSE4.2 types in the `simd::sse4`,
`simd::sse4_1` and `simd::sse4_2` namespaces](#sse4)
* [x86/x86_64 AVX types in the `simd::avx` namespace](#avx)
* [x86/x86_64 AVX2 types in the `simd::avx2` namespace](#avx2)
* [x86/x86_64 AVX512 types in the `simd::avx512` namespace](#avx512)
* [ARM and ARM64 NEON types in the `simd::neon` namespace](#neon)

## <a name="inline_namespace_common"></a>inline namespace `simd::common`

The inline namespace `simd::common` includes all possible combinations of
lane-counts and bit-sizes for arithmetic types (8-bit, 16-bit, 32-bit, 64-bit,
and 128-bit signed and unsigned integers, 32-bit, 64-bit, and 80-bit or 128-bit
(x87 vs quad-precision floats) floating points, and 32-bit, 64-bit, and 80-bit
or 128-bit (x87 vs quad-precision floats) complex types) that combine to totals
of 64-bit, 128-bit, 256-bit, and 512-bit data types. It is important to note
that, while such types exist, not all are directly representable by any
particular target SIMD technology. If you use such a type, Clang and GCC will
synthesize instructions for the appropriate operations.

## <a name="mmx"></a>namespace `simd::mmx`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

## <a name="sse"></a>namespace `simd::sse`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 4 32-bit lanes
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

## <a name="sse2"></a>namespace `simd::sse2`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 16 8-bit lanes
* `bool8x16_t` -- 8-bit booleans
* `int8x16_t` -- signed 8-bit integers
* `uint8x16_t` -- unsigned 8-bit integers

#### 8 16-bit lanes
* `bool16x8_t` -- 16-bit booleans
* `int16x8_t` -- signed 16-bit integers
* `uint16x8_t` -- unsigned 16-bit integers

#### 4 32-bit lanes
* `bool32x4_t` -- 32-bit booleans
* `int32x4_t` -- signed 32-bit integers
* `uint32x4_t` -- unsigned 32-bit integers
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

#### 2 64-bit lanes
* `bool64x2_t` -- 64-bit booleans
* `int64x2_t` -- signed 64-bit integers
* `uint64x2_t` -- unsigned 64-bit integers
* `float64x2_t` -- 64-bit floating point values
* `complex_float64x2_t` -- 64-bit complex number values

#### 1 128-bit lane
* `bool128x1_t` -- 128-bit booleans
* `int128x1_t` -- signed 128-bit integers
* `uint128x1_t` -- unsigned 128-bit integers

## <a name="sse3"></a>namespace `simd::sse3`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 16 8-bit lanes
* `bool8x16_t` -- 8-bit booleans
* `int8x16_t` -- signed 8-bit integers
* `uint8x16_t` -- unsigned 8-bit integers

#### 8 16-bit lanes
* `bool16x8_t` -- 16-bit booleans
* `int16x8_t` -- signed 16-bit integers
* `uint16x8_t` -- unsigned 16-bit integers

#### 4 32-bit lanes
* `bool32x4_t` -- 32-bit booleans
* `int32x4_t` -- signed 32-bit integers
* `uint32x4_t` -- unsigned 32-bit integers
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

#### 2 64-bit lanes
* `bool64x2_t` -- 64-bit booleans
* `int64x2_t` -- signed 64-bit integers
* `uint64x2_t` -- unsigned 64-bit integers
* `float64x2_t` -- 64-bit floating point values
* `complex_float64x2_t` -- 64-bit complex number values

#### 1 128-bit lane
* `bool128x1_t` -- 128-bit booleans
* `int128x1_t` -- signed 128-bit integers
* `uint128x1_t` -- unsigned 128-bit integers

## <a name="ssse3"></a>namespace `simd::ssse3`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 16 8-bit lanes
* `bool8x16_t` -- 8-bit booleans
* `int8x16_t` -- signed 8-bit integers
* `uint8x16_t` -- unsigned 8-bit integers

#### 8 16-bit lanes
* `bool16x8_t` -- 16-bit booleans
* `int16x8_t` -- signed 16-bit integers
* `uint16x8_t` -- unsigned 16-bit integers

#### 4 32-bit lanes
* `bool32x4_t` -- 32-bit booleans
* `int32x4_t` -- signed 32-bit integers
* `uint32x4_t` -- unsigned 32-bit integers
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

#### 2 64-bit lanes
* `bool64x2_t` -- 64-bit booleans
* `int64x2_t` -- signed 64-bit integers
* `uint64x2_t` -- unsigned 64-bit integers
* `float64x2_t` -- 64-bit floating point values
* `complex_float64x2_t` -- 64-bit complex number values

#### 1 128-bit lane
* `bool128x1_t` -- 128-bit booleans
* `int128x1_t` -- signed 128-bit integers
* `uint128x1_t` -- unsigned 128-bit integers

## <a name="sse4"></a>namespace `simd::sse4`/`simd::sse4_1`/`simd::sse4_2`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 16 8-bit lanes
* `bool8x16_t` -- 8-bit booleans
* `int8x16_t` -- signed 8-bit integers
* `uint8x16_t` -- unsigned 8-bit integers

#### 8 16-bit lanes
* `bool16x8_t` -- 16-bit booleans
* `int16x8_t` -- signed 16-bit integers
* `uint16x8_t` -- unsigned 16-bit integers

#### 4 32-bit lanes
* `bool32x4_t` -- 32-bit booleans
* `int32x4_t` -- signed 32-bit integers
* `uint32x4_t` -- unsigned 32-bit integers
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

#### 2 64-bit lanes
* `bool64x2_t` -- 64-bit booleans
* `int64x2_t` -- signed 64-bit integers
* `uint64x2_t` -- unsigned 64-bit integers
* `float64x2_t` -- 64-bit floating point values
* `complex_float64x2_t` -- 64-bit complex number values

#### 1 128-bit lane
* `bool128x1_t` -- 128-bit booleans
* `int128x1_t` -- signed 128-bit integers
* `uint128x1_t` -- unsigned 128-bit integers

## <a name="avx"></a>namespace `simd::avx`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 16 8-bit lanes
* `bool8x16_t` -- 8-bit booleans
* `int8x16_t` -- signed 8-bit integers
* `uint8x16_t` -- unsigned 8-bit integers

#### 8 16-bit lanes
* `bool16x8_t` -- 16-bit booleans
* `int16x8_t` -- signed 16-bit integers
* `uint16x8_t` -- unsigned 16-bit integers

#### 4 32-bit lanes
* `bool32x4_t` -- 32-bit booleans
* `int32x4_t` -- signed 32-bit integers
* `uint32x4_t` -- unsigned 32-bit integers
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

#### 2 64-bit lanes
* `bool64x2_t` -- 64-bit booleans
* `int64x2_t` -- signed 64-bit integers
* `uint64x2_t` -- unsigned 64-bit integers
* `float64x2_t` -- 64-bit floating point values
* `complex_float64x2_t` -- 64-bit complex number values

#### 1 128-bit lane
* `bool128x1_t` -- 128-bit booleans
* `int128x1_t` -- signed 128-bit integers
* `uint128x1_t` -- unsigned 128-bit integers

#### 8 32-bit lanes
* `float32x8_t` -- 32-bit floating point values
* `complex_float32x8_t` -- 32-bit complex number values

#### 4 64-bit lanes
* `float64x4_t` -- 64-bit floating point values
* `complex_float64x4_t` -- 64-bit complex number values

## <a name="avx2"></a>namespace `simd::avx2`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 16 8-bit lanes
* `bool8x16_t` -- 8-bit booleans
* `int8x16_t` -- signed 8-bit integers
* `uint8x16_t` -- unsigned 8-bit integers

#### 8 16-bit lanes
* `bool16x8_t` -- 16-bit booleans
* `int16x8_t` -- signed 16-bit integers
* `uint16x8_t` -- unsigned 16-bit integers

#### 4 32-bit lanes
* `bool32x4_t` -- 32-bit booleans
* `int32x4_t` -- signed 32-bit integers
* `uint32x4_t` -- unsigned 32-bit integers
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

#### 2 64-bit lanes
* `bool64x2_t` -- 64-bit booleans
* `int64x2_t` -- signed 64-bit integers
* `uint64x2_t` -- unsigned 64-bit integers
* `float64x2_t` -- 64-bit floating point values
* `complex_float64x2_t` -- 64-bit complex number values

#### 1 128-bit lane
* `bool128x1_t` -- 128-bit booleans
* `int128x1_t` -- signed 128-bit integers
* `uint128x1_t` -- unsigned 128-bit integers

#### 8 32-bit lanes
* `bool32x8_t` -- 32-bit booleans
* `int32x8_t` -- signed 32-bit integers
* `uint32x8_t` -- unsigned 32-bit integers
* `float32x8_t` -- 32-bit floating point values
* `complex_float32x8_t` -- 32-bit complex number values

#### 4 64-bit lanes
* `bool64x4_t` -- 64-bit booleans
* `int64x4_t` -- signed 64-bit integers
* `uint64x4_t` -- unsigned 64-bit integers
* `float64x4_t` -- 64-bit floating point values
* `complex_float64x4_t` -- 64-bit complex number values

## <a name="avx512"></a>namespace `simd::avx512`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 16 8-bit lanes
* `bool8x16_t` -- 8-bit booleans
* `int8x16_t` -- signed 8-bit integers
* `uint8x16_t` -- unsigned 8-bit integers

#### 8 16-bit lanes
* `bool16x8_t` -- 16-bit booleans
* `int16x8_t` -- signed 16-bit integers
* `uint16x8_t` -- unsigned 16-bit integers

#### 4 32-bit lanes
* `bool32x4_t` -- 32-bit booleans
* `int32x4_t` -- signed 32-bit integers
* `uint32x4_t` -- unsigned 32-bit integers
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

#### 2 64-bit lanes
* `bool64x2_t` -- 64-bit booleans
* `int64x2_t` -- signed 64-bit integers
* `uint64x2_t` -- unsigned 64-bit integers
* `float64x2_t` -- 64-bit floating point values
* `complex_float64x2_t` -- 64-bit complex number values

#### 1 128-bit lane
* `bool128x1_t` -- 128-bit booleans
* `int128x1_t` -- signed 128-bit integers
* `uint128x1_t` -- unsigned 128-bit integers

#### 8 32-bit lanes
* `bool32x8_t` -- 32-bit booleans
* `int32x8_t` -- signed 32-bit integers
* `uint32x8_t` -- unsigned 32-bit integers
* `float32x8_t` -- 32-bit floating point values
* `complex_float32x8_t` -- 32-bit complex number values

#### 4 64-bit lanes
* `bool64x4_t` -- 64-bit booleans
* `int64x4_t` -- signed 64-bit integers
* `uint64x4_t` -- unsigned 64-bit integers
* `float64x4_t` -- 64-bit floating point values
* `complex_float64x4_t` -- 64-bit complex number values

#### 16 32-bit lanes
* `bool32x16_t` -- 32-bit booleans
* `int32x16_t` -- signed 32-bit integers
* `uint32x16_t` -- unsigned 32-bit integers
* `float32x16_t` -- 32-bit floating point values
* `complex_float32x16_t` -- 32-bit complex number values

#### 8 64-bit lanes
* `bool64x8_t` -- 64-bit booleans
* `int64x8_t` -- signed 64-bit integers
* `uint64x8_t` -- unsigned 64-bit integers
* `float64x8_t` -- 64-bit floating point values
* `complex_float64x8_t` -- 64-bit complex number values

## <a name="neon"></a>namespace `simd::neon`

#### 8 8-bit lanes

* `bool8x8_t` -- 8-bit booleans
* `int8x8_t` -- signed 8-bit integers
* `uint8x8_t` -- unsigned 8-bit integers

#### 4 16-bit lanes
* `bool16x4_t` -- 16-bit booleans
* `int16x4_t` -- signed 16-bit integers
* `uint16x4_t` -- unsigned 16-bit integers

#### 2 32-bit lanes
* `bool32x2_t` -- 32-bit booleans
* `int32x2_t` -- signed 32-bit integers
* `uint32x2_t` -- unsigned 32-bit integers

#### 1 64-bit lane
* `bool64x1_t` -- 64-bit booleans
* `int64x1_t` -- signed 64-bit integers
* `uint64x1_t` -- unsigned 64-bit integers

#### 16 8-bit lanes
* `bool8x16_t` -- 8-bit booleans
* `int8x16_t` -- signed 8-bit integers
* `uint8x16_t` -- unsigned 8-bit integers

#### 8 16-bit lanes
* `bool16x8_t` -- 16-bit booleans
* `int16x8_t` -- signed 16-bit integers
* `uint16x8_t` -- unsigned 16-bit integers

#### 4 32-bit lanes
* `bool32x4_t` -- 32-bit booleans
* `int32x4_t` -- signed 32-bit integers
* `uint32x4_t` -- unsigned 32-bit integers
* `float32x4_t` -- 32-bit floating point values
* `complex_float32x4_t` -- 32-bit complex number values

#### 2 64-bit lanes
* `bool64x2_t` -- 64-bit booleans
* `int64x2_t` -- signed 64-bit integers
* `uint64x2_t` -- unsigned 64-bit integers
