//
// verifies alignment requirements for simd types
//

#include <cassert>   // assert
#include <cstddef>   // std::size_t
#include <cstdint>   // std::uintptr_t
#include <iostream>  // std::cout
#include <new>       // new, delete
#include <vector>    // std::vector

#include <simd>


template <typename T>
bool alignment_test (T const *, std::size_t) noexcept;

void verify_statically_allocated_vars (void);

template <std::size_t>
void verify_statically_allocated_array (void);

void verify_automatic_storage_vars (void);

template <std::size_t>
void verify_automatic_storage_array (void);

void verify_dynamically_allocated_vars (void);

template <std::size_t>
void verify_dynamically_allocated_array (void);

template <std::size_t>
void verify_vector_allocated_vars (void);


template <typename T>
struct simd_allocator
{
    using value_type = T;

    T * allocate (std::size_t n)
    {
        return new T [n];
    }

    void deallocate (T * ptr, std::size_t n)
    {
        (void) n;
        delete [] ptr;
    }
};

template <typename T>
bool alignment_test (T const * ptr, std::size_t expected_alignment)
    noexcept
{
    return reinterpret_cast <std::uintptr_t> (ptr) % expected_alignment == 0;
}

void verify_statically_allocated_vars (void)
{
    using namespace simd::common;

    /* 8 x 8 */
    {
        static bool8x8_t bool8x8;
        static int8x8_t int8x8;
        static uint8x8_t uint8x8;

        assert (alignment_test (&bool8x8, bool8x8_t::alignment));
        assert (alignment_test (&int8x8, int8x8_t::alignment));
        assert (alignment_test (&uint8x8, uint8x8_t::alignment));
    }

    /* 8 x 16 */
    {
        static bool8x16_t bool8x16;
        static int8x16_t int8x16;
        static uint8x16_t uint8x16;

        assert (alignment_test (&bool8x16, bool8x16_t::alignment));
        assert (alignment_test (&int8x16, int8x16_t::alignment));
        assert (alignment_test (&uint8x16, uint8x16_t::alignment));
    }

    /* 8 x 32 */
    {
        static bool8x32_t bool8x32;
        static int8x32_t int8x32;
        static uint8x32_t uint8x32;

        assert (alignment_test (&bool8x32, bool8x32_t::alignment));
        assert (alignment_test (&int8x32, int8x32_t::alignment));
        assert (alignment_test (&uint8x32, uint8x32_t::alignment));
    }

    /* 8 x 64 */
    {
        static bool8x64_t bool8x64;
        static int8x64_t int8x64;
        static uint8x64_t uint8x64;

        assert (alignment_test (&bool8x64, bool8x64_t::alignment));
        assert (alignment_test (&int8x64, int8x64_t::alignment));
        assert (alignment_test (&uint8x64, uint8x64_t::alignment));
    }

    /* 16 x 4 */
    {
        static bool16x4_t bool16x4;
        static int16x4_t int16x4;
        static uint16x4_t uint16x4;

        assert (alignment_test (&bool16x4, bool16x4_t::alignment));
        assert (alignment_test (&int16x4, int16x4_t::alignment));
        assert (alignment_test (&uint16x4, uint16x4_t::alignment));
    }

    /* 16 x 8 */
    {
        static bool16x8_t bool16x8;
        static int16x8_t int16x8;
        static uint16x8_t uint16x8;

        assert (alignment_test (&bool16x8, bool16x8_t::alignment));
        assert (alignment_test (&int16x8, int16x8_t::alignment));
        assert (alignment_test (&uint16x8, uint16x8_t::alignment));
    }

    /* 16 x 16 */
    {
        static bool16x16_t bool16x16;
        static int16x16_t int16x16;
        static uint16x16_t uint16x16;

        assert (alignment_test (&bool16x16, bool16x16_t::alignment));
        assert (alignment_test (&int16x16, int16x16_t::alignment));
        assert (alignment_test (&uint16x16, uint16x16_t::alignment));
    }

    /* 16 x 32 */
    {
        static bool16x32_t bool16x32;
        static int16x32_t int16x32;
        static uint16x32_t uint16x32;

        assert (alignment_test (&bool16x32, bool16x32_t::alignment));
        assert (alignment_test (&int16x32, int16x32_t::alignment));
        assert (alignment_test (&uint16x32, uint16x32_t::alignment));
    }

    /* 32 x 2 */
    {
        static bool32x2_t bool32x2;
        static int32x2_t int32x2;
        static uint32x2_t uint32x2;
        static float32x2_t float32x2;
        static complex_float32x2_t complex_float32x2;

        assert (alignment_test (&bool32x2, bool32x2_t::alignment));
        assert (alignment_test (&int32x2, int32x2_t::alignment));
        assert (alignment_test (&uint32x2, uint32x2_t::alignment));
        assert (alignment_test (&float32x2, float32x2_t::alignment));
        assert (
            alignment_test (&complex_float32x2, complex_float32x2_t::alignment)
        );
    }

    /* 32 x 4 */
    {
        static bool32x4_t bool32x4;
        static int32x4_t int32x4;
        static uint32x4_t uint32x4;
        static float32x4_t float32x4;
        static complex_float32x4_t complex_float32x4;

        assert (alignment_test (&bool32x4, bool32x4_t::alignment));
        assert (alignment_test (&int32x4, int32x4_t::alignment));
        assert (alignment_test (&uint32x4, uint32x4_t::alignment));
        assert (alignment_test (&float32x4, float32x4_t::alignment));
        assert (
            alignment_test (&complex_float32x4, complex_float32x4_t::alignment)
        );
    }

    /* 32 x 8 */
    {
        static bool32x8_t bool32x8;
        static int32x8_t int32x8;
        static uint32x8_t uint32x8;
        static float32x8_t float32x8;
        static complex_float32x8_t complex_float32x8;

        assert (alignment_test (&bool32x8, bool32x8_t::alignment));
        assert (alignment_test (&int32x8, int32x8_t::alignment));
        assert (alignment_test (&uint32x8, uint32x8_t::alignment));
        assert (alignment_test (&float32x8, float32x8_t::alignment));
        assert (
            alignment_test (&complex_float32x8, complex_float32x8_t::alignment)
        );
    }

    /* 32 x 16 */
    {
        static bool32x16_t bool32x16;
        static int32x16_t int32x16;
        static uint32x16_t uint32x16;
        static float32x16_t float32x16;
        static complex_float32x16_t complex_float32x16;

        assert (alignment_test (&bool32x16, bool32x16_t::alignment));
        assert (alignment_test (&int32x16, int32x16_t::alignment));
        assert (alignment_test (&uint32x16, uint32x16_t::alignment));
        assert (alignment_test (&float32x16, float32x16_t::alignment));
        assert (
            alignment_test (
                &complex_float32x16, complex_float32x16_t::alignment
            )
        );
    }

    /* 64 x 1 */
    {
        static bool64x1_t bool64x1;
        static int64x1_t int64x1;
        static uint64x1_t uint64x1;
        static float64x1_t float64x1;
        static complex_float64x1_t complex_float64x1;

        assert (alignment_test (&bool64x1, bool64x1_t::alignment));
        assert (alignment_test (&int64x1, int64x1_t::alignment));
        assert (alignment_test (&uint64x1, uint64x1_t::alignment));
        assert (alignment_test (&float64x1, float64x1_t::alignment));
        assert (
            alignment_test (&complex_float64x1, complex_float64x1_t::alignment)
        );
    }

    /* 64 x 2 */
    {
        static bool64x2_t bool64x2;
        static int64x2_t int64x2;
        static uint64x2_t uint64x2;
        static float64x2_t float64x2;
        static complex_float64x2_t complex_float64x2;

        assert (alignment_test (&bool64x2, bool64x2_t::alignment));
        assert (alignment_test (&int64x2, int64x2_t::alignment));
        assert (alignment_test (&uint64x2, uint64x2_t::alignment));
        assert (alignment_test (&float64x2, float64x2_t::alignment));
        assert (
            alignment_test (&complex_float64x2, complex_float64x2_t::alignment)
        );
    }

    /* 64 x 4 */
    {
        static bool64x4_t bool64x4;
        static int64x4_t int64x4;
        static uint64x4_t uint64x4;
        static float64x4_t float64x4;
        static complex_float64x4_t complex_float64x4;

        assert (alignment_test (&bool64x4, bool64x4_t::alignment));
        assert (alignment_test (&int64x4, int64x4_t::alignment));
        assert (alignment_test (&uint64x4, uint64x4_t::alignment));
        assert (alignment_test (&float64x4, float64x4_t::alignment));
        assert (
            alignment_test (&complex_float64x4, complex_float64x4_t::alignment)
        );
    }

    /* 64 x 8 */
    {
        static bool64x8_t bool64x8;
        static int64x8_t int64x8;
        static uint64x8_t uint64x8;
        static float64x8_t float64x8;
        static complex_float64x8_t complex_float64x8;

        assert (alignment_test (&bool64x8, bool64x8_t::alignment));
        assert (alignment_test (&int64x8, int64x8_t::alignment));
        assert (alignment_test (&uint64x8, uint64x8_t::alignment));
        assert (alignment_test (&float64x8, float64x8_t::alignment));
        assert (
            alignment_test (&complex_float64x8, complex_float64x8_t::alignment)
        );
    }

    /* long double x 2 */
    {
        static long_doublex2_t long_doublex2;
        static complex_long_doublex2_t complex_long_doublex2;

        assert (alignment_test (&long_doublex2, long_doublex2_t::alignment));
        assert (
            alignment_test (
                &complex_long_doublex2, complex_long_doublex2_t::alignment
            )
        );
    }

    /* long double x 4 */
    {
        static long_doublex4_t long_doublex4;
        static complex_long_doublex4_t complex_long_doublex4;

        assert (alignment_test (&long_doublex4, long_doublex4_t::alignment));
        assert (
            alignment_test (
                &complex_long_doublex4, complex_long_doublex4_t::alignment
            )
        );
    }

    /* 128 x 1 */
    {
        static bool128x1_t bool128x1;
        static int128x1_t int128x1;
        static uint128x1_t uint128x1;

        assert (alignment_test (&bool128x1, bool128x1_t::alignment));
        assert (alignment_test (&int128x1, int128x1_t::alignment));
        assert (alignment_test (&uint128x1, uint128x1_t::alignment));
    }

    /* 128 x 2 */
    {
        static bool128x2_t bool128x2;
        static int128x2_t int128x2;
        static uint128x2_t uint128x2;

        assert (alignment_test (&bool128x2, bool128x2_t::alignment));
        assert (alignment_test (&int128x2, int128x2_t::alignment));
        assert (alignment_test (&uint128x2, uint128x2_t::alignment));
    }

    /* 128 x 4 */
    {
        static bool128x4_t bool128x4;
        static int128x4_t int128x4;
        static uint128x4_t uint128x4;

        assert (alignment_test (&bool128x4, bool128x4_t::alignment));
        assert (alignment_test (&int128x4, int128x4_t::alignment));
        assert (alignment_test (&uint128x4, uint128x4_t::alignment));
    }
}

template <std::size_t array_size>
void verify_statically_allocated_array (void)
{
    using namespace simd::common;

    /* 8 x 8 */
    {
        static bool8x8_t bool8x8 [array_size];
        static int8x8_t int8x8 [array_size];
        static uint8x8_t uint8x8 [array_size];

        assert (alignment_test (&bool8x8 [0], bool8x8_t::alignment));
        assert (alignment_test (&int8x8 [0], int8x8_t::alignment));
        assert (alignment_test (&uint8x8 [0], uint8x8_t::alignment));
    }

    /* 8 x 16 */
    {
        static bool8x16_t bool8x16 [array_size];
        static int8x16_t int8x16 [array_size];
        static uint8x16_t uint8x16 [array_size];

        assert (alignment_test (&bool8x16 [0], bool8x16_t::alignment));
        assert (alignment_test (&int8x16 [0], int8x16_t::alignment));
        assert (alignment_test (&uint8x16 [0], uint8x16_t::alignment));
    }

    /* 8 x 32 */
    {
        static bool8x32_t bool8x32 [array_size];
        static int8x32_t int8x32 [array_size];
        static uint8x32_t uint8x32 [array_size];

        assert (alignment_test (&bool8x32 [0], bool8x32_t::alignment));
        assert (alignment_test (&int8x32 [0], int8x32_t::alignment));
        assert (alignment_test (&uint8x32 [0], uint8x32_t::alignment));
    }

    /* 8 x 64 */
    {
        static bool8x64_t bool8x64 [array_size];
        static int8x64_t int8x64 [array_size];
        static uint8x64_t uint8x64 [array_size];

        assert (alignment_test (&bool8x64 [0], bool8x64_t::alignment));
        assert (alignment_test (&int8x64 [0], int8x64_t::alignment));
        assert (alignment_test (&uint8x64 [0], uint8x64_t::alignment));
    }

    /* 16 x 4 */
    {
        static bool16x4_t bool16x4 [array_size];
        static int16x4_t int16x4 [array_size];
        static uint16x4_t uint16x4 [array_size];

        assert (alignment_test (&bool16x4 [0], bool16x4_t::alignment));
        assert (alignment_test (&int16x4 [0], int16x4_t::alignment));
        assert (alignment_test (&uint16x4 [0], uint16x4_t::alignment));
    }

    /* 16 x 8 */
    {
        static bool16x8_t bool16x8 [array_size];
        static int16x8_t int16x8 [array_size];
        static uint16x8_t uint16x8 [array_size];

        assert (alignment_test (&bool16x8 [0], bool16x8_t::alignment));
        assert (alignment_test (&int16x8 [0], int16x8_t::alignment));
        assert (alignment_test (&uint16x8 [0], uint16x8_t::alignment));
    }

    /* 16 x 16 */
    {
        static bool16x16_t bool16x16 [array_size];
        static int16x16_t int16x16 [array_size];
        static uint16x16_t uint16x16 [array_size];

        assert (alignment_test (&bool16x16 [0], bool16x16_t::alignment));
        assert (alignment_test (&int16x16 [0], int16x16_t::alignment));
        assert (alignment_test (&uint16x16 [0], uint16x16_t::alignment));
    }

    /* 16 x 32 */
    {
        static bool16x32_t bool16x32 [array_size];
        static int16x32_t int16x32 [array_size];
        static uint16x32_t uint16x32 [array_size];

        assert (alignment_test (&bool16x32 [0], bool16x32_t::alignment));
        assert (alignment_test (&int16x32 [0], int16x32_t::alignment));
        assert (alignment_test (&uint16x32 [0], uint16x32_t::alignment));
    }

    /* 32 x 2 */
    {
        static bool32x2_t bool32x2 [array_size];
        static int32x2_t int32x2 [array_size];
        static uint32x2_t uint32x2 [array_size];
        static float32x2_t float32x2 [array_size];
        static complex_float32x2_t complex_float32x2 [array_size];

        assert (alignment_test (&bool32x2 [0], bool32x2_t::alignment));
        assert (alignment_test (&int32x2 [0], int32x2_t::alignment));
        assert (alignment_test (&uint32x2 [0], uint32x2_t::alignment));
        assert (alignment_test (&float32x2 [0], float32x2_t::alignment));
        assert (
            alignment_test (
                &complex_float32x2 [0], complex_float32x2_t::alignment
            )
        );
    }

    /* 32 x 4 */
    {
        static bool32x4_t bool32x4 [array_size];
        static int32x4_t int32x4 [array_size];
        static uint32x4_t uint32x4 [array_size];
        static float32x4_t float32x4 [array_size];
        static complex_float32x4_t complex_float32x4 [array_size];

        assert (alignment_test (&bool32x4 [0], bool32x4_t::alignment));
        assert (alignment_test (&int32x4 [0], int32x4_t::alignment));
        assert (alignment_test (&uint32x4 [0], uint32x4_t::alignment));
        assert (alignment_test (&float32x4 [0], float32x4_t::alignment));
        assert (
            alignment_test (
                &complex_float32x4 [0], complex_float32x4_t::alignment
            )
        );
    }

    /* 32 x 8 */
    {
        static bool32x8_t bool32x8 [array_size];
        static int32x8_t int32x8 [array_size];
        static uint32x8_t uint32x8 [array_size];
        static float32x8_t float32x8 [array_size];
        static complex_float32x8_t complex_float32x8 [array_size];

        assert (alignment_test (&bool32x8 [0], bool32x8_t::alignment));
        assert (alignment_test (&int32x8 [0], int32x8_t::alignment));
        assert (alignment_test (&uint32x8 [0], uint32x8_t::alignment));
        assert (alignment_test (&float32x8 [0], float32x8_t::alignment));
        assert (
            alignment_test (
                &complex_float32x8 [0], complex_float32x8_t::alignment
            )
        );
    }

    /* 32 x 16 */
    {
        static bool32x16_t bool32x16 [array_size];
        static int32x16_t int32x16 [array_size];
        static uint32x16_t uint32x16 [array_size];
        static float32x16_t float32x16 [array_size];
        static complex_float32x16_t complex_float32x16 [array_size];

        assert (alignment_test (&bool32x16 [0], bool32x16_t::alignment));
        assert (alignment_test (&int32x16 [0], int32x16_t::alignment));
        assert (alignment_test (&uint32x16 [0], uint32x16_t::alignment));
        assert (alignment_test (&float32x16 [0], float32x16_t::alignment));
        assert (
            alignment_test (
                &complex_float32x16 [0], complex_float32x16_t::alignment
            )
        );
    }

    /* 64 x 1 */
    {
        static bool64x1_t bool64x1 [array_size];
        static int64x1_t int64x1 [array_size];
        static uint64x1_t uint64x1 [array_size];
        static float64x1_t float64x1 [array_size];
        static complex_float64x1_t complex_float64x1 [array_size];

        assert (alignment_test (&bool64x1 [0], bool64x1_t::alignment));
        assert (alignment_test (&int64x1 [0], int64x1_t::alignment));
        assert (alignment_test (&uint64x1 [0], uint64x1_t::alignment));
        assert (alignment_test (&float64x1 [0], float64x1_t::alignment));
        assert (
            alignment_test (
                &complex_float64x1 [0], complex_float64x1_t::alignment
            )
        );
    }

    /* 64 x 2 */
    {
        static bool64x2_t bool64x2 [array_size];
        static int64x2_t int64x2 [array_size];
        static uint64x2_t uint64x2 [array_size];
        static float64x2_t float64x2 [array_size];
        static complex_float64x2_t complex_float64x2 [array_size];

        assert (alignment_test (&bool64x2 [0], bool64x2_t::alignment));
        assert (alignment_test (&int64x2 [0], int64x2_t::alignment));
        assert (alignment_test (&uint64x2 [0], uint64x2_t::alignment));
        assert (alignment_test (&float64x2 [0], float64x2_t::alignment));
        assert (
            alignment_test (
                &complex_float64x2 [0], complex_float64x2_t::alignment
            )
        );
    }

    /* 64 x 4 */
    {
        static bool64x4_t bool64x4 [array_size];
        static int64x4_t int64x4 [array_size];
        static uint64x4_t uint64x4 [array_size];
        static float64x4_t float64x4 [array_size];
        static complex_float64x4_t complex_float64x4 [array_size];

        assert (alignment_test (&bool64x4 [0], bool64x4_t::alignment));
        assert (alignment_test (&int64x4 [0], int64x4_t::alignment));
        assert (alignment_test (&uint64x4 [0], uint64x4_t::alignment));
        assert (alignment_test (&float64x4 [0], float64x4_t::alignment));
        assert (
            alignment_test (
                &complex_float64x4 [0], complex_float64x4_t::alignment
            )
        );
    }

    /* 64 x 8 */
    {
        static bool64x8_t bool64x8 [array_size];
        static int64x8_t int64x8 [array_size];
        static uint64x8_t uint64x8 [array_size];
        static float64x8_t float64x8 [array_size];
        static complex_float64x8_t complex_float64x8 [array_size];

        assert (alignment_test (&bool64x8 [0], bool64x8_t::alignment));
        assert (alignment_test (&int64x8 [0], int64x8_t::alignment));
        assert (alignment_test (&uint64x8 [0], uint64x8_t::alignment));
        assert (alignment_test (&float64x8 [0], float64x8_t::alignment));
        assert (
            alignment_test (
                &complex_float64x8 [0], complex_float64x8_t::alignment
            )
        );
    }

    /* long double x 2 */
    {
        static long_doublex2_t long_doublex2 [array_size];
        static complex_long_doublex2_t complex_long_doublex2 [array_size];

        assert (
            alignment_test (&long_doublex2 [0], long_doublex2_t::alignment)
        );
        assert (
            alignment_test (
                &complex_long_doublex2 [0], complex_long_doublex2_t::alignment
            )
        );
    }

    /* long double x 4 */
    {
        static long_doublex4_t long_doublex4 [array_size];
        static complex_long_doublex4_t complex_long_doublex4 [array_size];

        assert (
            alignment_test (&long_doublex4 [0], long_doublex4_t::alignment)
        );
        assert (
            alignment_test (
                &complex_long_doublex4 [0], complex_long_doublex4_t::alignment
            )
        );
    }

    /* 128 x 1 */
    {
        static bool128x1_t bool128x1 [array_size];
        static int128x1_t int128x1 [array_size];
        static uint128x1_t uint128x1 [array_size];

        assert (alignment_test (&bool128x1 [0], bool128x1_t::alignment));
        assert (alignment_test (&int128x1 [0], int128x1_t::alignment));
        assert (alignment_test (&uint128x1 [0], uint128x1_t::alignment));
    }

    /* 128 x 2 */
    {
        static bool128x2_t bool128x2 [array_size];
        static int128x2_t int128x2 [array_size];
        static uint128x2_t uint128x2 [array_size];

        assert (alignment_test (&bool128x2 [0], bool128x2_t::alignment));
        assert (alignment_test (&int128x2 [0], int128x2_t::alignment));
        assert (alignment_test (&uint128x2 [0], uint128x2_t::alignment));
    }

    /* 128 x 4 */
    {
        static bool128x4_t bool128x4 [array_size];
        static int128x4_t int128x4 [array_size];
        static uint128x4_t uint128x4 [array_size];

        assert (alignment_test (&bool128x4 [0], bool128x4_t::alignment));
        assert (alignment_test (&int128x4 [0], int128x4_t::alignment));
        assert (alignment_test (&uint128x4 [0], uint128x4_t::alignment));
    }
}

void verify_automatic_storage_vars (void)
{
    using namespace simd::common;

    /* 8 x 8 */
    {
        bool8x8_t bool8x8;
        int8x8_t int8x8;
        uint8x8_t uint8x8;

        assert (alignment_test (&bool8x8, bool8x8_t::alignment));
        assert (alignment_test (&int8x8, int8x8_t::alignment));
        assert (alignment_test (&uint8x8, uint8x8_t::alignment));
    }

    /* 8 x 16 */
    {
        bool8x16_t bool8x16;
        int8x16_t int8x16;
        uint8x16_t uint8x16;

        assert (alignment_test (&bool8x16, bool8x16_t::alignment));
        assert (alignment_test (&int8x16, int8x16_t::alignment));
        assert (alignment_test (&uint8x16, uint8x16_t::alignment));
    }

    /* 8 x 32 */
    {
        bool8x32_t bool8x32;
        int8x32_t int8x32;
        uint8x32_t uint8x32;

        assert (alignment_test (&bool8x32, bool8x32_t::alignment));
        assert (alignment_test (&int8x32, int8x32_t::alignment));
        assert (alignment_test (&uint8x32, uint8x32_t::alignment));
    }

    /* 8 x 64 */
    {
        bool8x64_t bool8x64;
        int8x64_t int8x64;
        uint8x64_t uint8x64;

        assert (alignment_test (&bool8x64, bool8x64_t::alignment));
        assert (alignment_test (&int8x64, int8x64_t::alignment));
        assert (alignment_test (&uint8x64, uint8x64_t::alignment));
    }

    /* 16 x 4 */
    {
        bool16x4_t bool16x4;
        int16x4_t int16x4;
        uint16x4_t uint16x4;

        assert (alignment_test (&bool16x4, bool16x4_t::alignment));
        assert (alignment_test (&int16x4, int16x4_t::alignment));
        assert (alignment_test (&uint16x4, uint16x4_t::alignment));
    }

    /* 16 x 8 */
    {
        bool16x8_t bool16x8;
        int16x8_t int16x8;
        uint16x8_t uint16x8;

        assert (alignment_test (&bool16x8, bool16x8_t::alignment));
        assert (alignment_test (&int16x8, int16x8_t::alignment));
        assert (alignment_test (&uint16x8, uint16x8_t::alignment));
    }

    /* 16 x 16 */
    {
        bool16x16_t bool16x16;
        int16x16_t int16x16;
        uint16x16_t uint16x16;

        assert (alignment_test (&bool16x16, bool16x16_t::alignment));
        assert (alignment_test (&int16x16, int16x16_t::alignment));
        assert (alignment_test (&uint16x16, uint16x16_t::alignment));
    }

    /* 16 x 32 */
    {
        bool16x32_t bool16x32;
        int16x32_t int16x32;
        uint16x32_t uint16x32;

        assert (alignment_test (&bool16x32, bool16x32_t::alignment));
        assert (alignment_test (&int16x32, int16x32_t::alignment));
        assert (alignment_test (&uint16x32, uint16x32_t::alignment));
    }

    /* 32 x 2 */
    {
        bool32x2_t bool32x2;
        int32x2_t int32x2;
        uint32x2_t uint32x2;
        float32x2_t float32x2;
        complex_float32x2_t complex_float32x2;

        assert (alignment_test (&bool32x2, bool32x2_t::alignment));
        assert (alignment_test (&int32x2, int32x2_t::alignment));
        assert (alignment_test (&uint32x2, uint32x2_t::alignment));
        assert (alignment_test (&float32x2, float32x2_t::alignment));
        assert (
            alignment_test (&complex_float32x2, complex_float32x2_t::alignment)
        );
    }

    /* 32 x 4 */
    {
        bool32x4_t bool32x4;
        int32x4_t int32x4;
        uint32x4_t uint32x4;
        float32x4_t float32x4;
        complex_float32x4_t complex_float32x4;

        assert (alignment_test (&bool32x4, bool32x4_t::alignment));
        assert (alignment_test (&int32x4, int32x4_t::alignment));
        assert (alignment_test (&uint32x4, uint32x4_t::alignment));
        assert (alignment_test (&float32x4, float32x4_t::alignment));
        assert (
            alignment_test (&complex_float32x4, complex_float32x4_t::alignment)
        );
    }

    /* 32 x 8 */
    {
        bool32x8_t bool32x8;
        int32x8_t int32x8; uint32x8_t uint32x8;
        float32x8_t float32x8;
        complex_float32x8_t complex_float32x8;

        assert (alignment_test (&bool32x8, bool32x8_t::alignment));
        assert (alignment_test (&int32x8, int32x8_t::alignment));
        assert (alignment_test (&uint32x8, uint32x8_t::alignment));
        assert (alignment_test (&float32x8, float32x8_t::alignment));
        assert (
            alignment_test (&complex_float32x8, complex_float32x8_t::alignment)
        );
    }

    /* 32 x 16 */
    {
        bool32x16_t bool32x16;
        int32x16_t int32x16;
        uint32x16_t uint32x16;
        float32x16_t float32x16;
        complex_float32x16_t complex_float32x16;

        assert (alignment_test (&bool32x16, bool32x16_t::alignment));
        assert (alignment_test (&int32x16, int32x16_t::alignment));
        assert (alignment_test (&uint32x16, uint32x16_t::alignment));
        assert (alignment_test (&float32x16, float32x16_t::alignment));
        assert (
            alignment_test (
                &complex_float32x16, complex_float32x16_t::alignment
            )
        );
    }

    /* 64 x 1 */
    {
        bool64x1_t bool64x1;
        int64x1_t int64x1;
        uint64x1_t uint64x1;
        float64x1_t float64x1;
        complex_float64x1_t complex_float64x1;

        assert (alignment_test (&bool64x1, bool64x1_t::alignment));
        assert (alignment_test (&int64x1, int64x1_t::alignment));
        assert (alignment_test (&uint64x1, uint64x1_t::alignment));
        assert (alignment_test (&float64x1, float64x1_t::alignment));
        assert (
            alignment_test (&complex_float64x1, complex_float64x1_t::alignment)
        );
    }

    /* 64 x 2 */
    {
        bool64x2_t bool64x2;
        int64x2_t int64x2;
        uint64x2_t uint64x2;
        float64x2_t float64x2;
        complex_float64x2_t complex_float64x2;

        assert (alignment_test (&bool64x2, bool64x2_t::alignment));
        assert (alignment_test (&int64x2, int64x2_t::alignment));
        assert (alignment_test (&uint64x2, uint64x2_t::alignment));
        assert (alignment_test (&float64x2, float64x2_t::alignment));
        assert (
            alignment_test (&complex_float64x2, complex_float64x2_t::alignment)
        );
    }

    /* 64 x 4 */
    {
        bool64x4_t bool64x4;
        int64x4_t int64x4;
        uint64x4_t uint64x4;
        float64x4_t float64x4;
        complex_float64x4_t complex_float64x4;

        assert (alignment_test (&bool64x4, bool64x4_t::alignment));
        assert (alignment_test (&int64x4, int64x4_t::alignment));
        assert (alignment_test (&uint64x4, uint64x4_t::alignment));
        assert (alignment_test (&float64x4, float64x4_t::alignment));
        assert (
            alignment_test (&complex_float64x4, complex_float64x4_t::alignment)
        );
    }

    /* 64 x 8 */
    {
        bool64x8_t bool64x8;
        int64x8_t int64x8;
        uint64x8_t uint64x8;
        float64x8_t float64x8;
        complex_float64x8_t complex_float64x8;

        assert (alignment_test (&bool64x8, bool64x8_t::alignment));
        assert (alignment_test (&int64x8, int64x8_t::alignment));
        assert (alignment_test (&uint64x8, uint64x8_t::alignment));
        assert (alignment_test (&float64x8, float64x8_t::alignment));
        assert (
            alignment_test (&complex_float64x8, complex_float64x8_t::alignment)
        );
    }

    /* long double x 2 */
    {
        long_doublex2_t long_doublex2;
        complex_long_doublex2_t complex_long_doublex2;

        assert (alignment_test (&long_doublex2, long_doublex2_t::alignment));
        assert (
            alignment_test (
                &complex_long_doublex2, complex_long_doublex2_t::alignment
            )
        );
    }

    /* long double x 4 */
    {
        long_doublex4_t long_doublex4;
        complex_long_doublex4_t complex_long_doublex4;

        assert (alignment_test (&long_doublex4, long_doublex4_t::alignment));
        assert (
            alignment_test (
                &complex_long_doublex4, complex_long_doublex4_t::alignment
            )
        );
    }

    /* 128 x 1 */
    {
        bool128x1_t bool128x1;
        int128x1_t int128x1;
        uint128x1_t uint128x1;

        assert (alignment_test (&bool128x1, bool128x1_t::alignment));
        assert (alignment_test (&int128x1, int128x1_t::alignment));
        assert (alignment_test (&uint128x1, uint128x1_t::alignment));
    }

    /* 128 x 2 */
    {
        bool128x2_t bool128x2;
        int128x2_t int128x2;
        uint128x2_t uint128x2;

        assert (alignment_test (&bool128x2, bool128x2_t::alignment));
        assert (alignment_test (&int128x2, int128x2_t::alignment));
        assert (alignment_test (&uint128x2, uint128x2_t::alignment));
    }

    /* 128 x 4 */
    {
        bool128x4_t bool128x4;
        int128x4_t int128x4;
        uint128x4_t uint128x4;

        assert (alignment_test (&bool128x4, bool128x4_t::alignment));
        assert (alignment_test (&int128x4, int128x4_t::alignment));
        assert (alignment_test (&uint128x4, uint128x4_t::alignment));
    }
}

template <std::size_t array_size>
void verify_automatic_storage_array (void)
{
    using namespace simd::common;

    /* 8 x 8 */
    {
        bool8x8_t bool8x8 [array_size];
        int8x8_t int8x8 [array_size];
        uint8x8_t uint8x8 [array_size];

        assert (alignment_test (&bool8x8 [0], bool8x8_t::alignment));
        assert (alignment_test (&int8x8 [0], int8x8_t::alignment));
        assert (alignment_test (&uint8x8 [0], uint8x8_t::alignment));
    }

    /* 8 x 16 */
    {
        bool8x16_t bool8x16 [array_size];
        int8x16_t int8x16 [array_size];
        uint8x16_t uint8x16 [array_size];

        assert (alignment_test (&bool8x16 [0], bool8x16_t::alignment));
        assert (alignment_test (&int8x16 [0], int8x16_t::alignment));
        assert (alignment_test (&uint8x16 [0], uint8x16_t::alignment));
    }

    /* 8 x 32 */
    {
        bool8x32_t bool8x32 [array_size];
        int8x32_t int8x32 [array_size];
        uint8x32_t uint8x32 [array_size];

        assert (alignment_test (&bool8x32 [0], bool8x32_t::alignment));
        assert (alignment_test (&int8x32 [0], int8x32_t::alignment));
        assert (alignment_test (&uint8x32 [0], uint8x32_t::alignment));
    }

    /* 8 x 64 */
    {
        bool8x64_t bool8x64 [array_size];
        int8x64_t int8x64 [array_size];
        uint8x64_t uint8x64 [array_size];

        assert (alignment_test (&bool8x64 [0], bool8x64_t::alignment));
        assert (alignment_test (&int8x64 [0], int8x64_t::alignment));
        assert (alignment_test (&uint8x64 [0], uint8x64_t::alignment));
    }

    /* 16 x 4 */
    {
        bool16x4_t bool16x4 [array_size];
        int16x4_t int16x4 [array_size];
        uint16x4_t uint16x4 [array_size];

        assert (alignment_test (&bool16x4 [0], bool16x4_t::alignment));
        assert (alignment_test (&int16x4 [0], int16x4_t::alignment));
        assert (alignment_test (&uint16x4 [0], uint16x4_t::alignment));
    }

    /* 16 x 8 */
    {
        bool16x8_t bool16x8 [array_size];
        int16x8_t int16x8 [array_size];
        uint16x8_t uint16x8 [array_size];

        assert (alignment_test (&bool16x8 [0], bool16x8_t::alignment));
        assert (alignment_test (&int16x8 [0], int16x8_t::alignment));
        assert (alignment_test (&uint16x8 [0], uint16x8_t::alignment));
    }

    /* 16 x 16 */
    {
        bool16x16_t bool16x16 [array_size];
        int16x16_t int16x16 [array_size];
        uint16x16_t uint16x16 [array_size];

        assert (alignment_test (&bool16x16 [0], bool16x16_t::alignment));
        assert (alignment_test (&int16x16 [0], int16x16_t::alignment));
        assert (alignment_test (&uint16x16 [0], uint16x16_t::alignment));
    }

    /* 16 x 32 */
    {
        bool16x32_t bool16x32 [array_size];
        int16x32_t int16x32 [array_size];
        uint16x32_t uint16x32 [array_size];

        assert (alignment_test (&bool16x32 [0], bool16x32_t::alignment));
        assert (alignment_test (&int16x32 [0], int16x32_t::alignment));
        assert (alignment_test (&uint16x32 [0], uint16x32_t::alignment));
    }

    /* 32 x 2 */
    {
        bool32x2_t bool32x2 [array_size];
        int32x2_t int32x2 [array_size];
        uint32x2_t uint32x2 [array_size];
        float32x2_t float32x2 [array_size];
        complex_float32x2_t complex_float32x2 [array_size];

        assert (alignment_test (&bool32x2 [0], bool32x2_t::alignment));
        assert (alignment_test (&int32x2 [0], int32x2_t::alignment));
        assert (alignment_test (&uint32x2 [0], uint32x2_t::alignment));
        assert (alignment_test (&float32x2 [0], float32x2_t::alignment));
        assert (
            alignment_test (
                &complex_float32x2 [0], complex_float32x2_t::alignment
            )
        );
    }

    /* 32 x 4 */
    {
        bool32x4_t bool32x4 [array_size];
        int32x4_t int32x4 [array_size];
        uint32x4_t uint32x4 [array_size];
        float32x4_t float32x4 [array_size];
        complex_float32x4_t complex_float32x4 [array_size];

        assert (alignment_test (&bool32x4 [0], bool32x4_t::alignment));
        assert (alignment_test (&int32x4 [0], int32x4_t::alignment));
        assert (alignment_test (&uint32x4 [0], uint32x4_t::alignment));
        assert (alignment_test (&float32x4 [0], float32x4_t::alignment));
        assert (
            alignment_test (
                &complex_float32x4 [0], complex_float32x4_t::alignment
            )
        );
    }

    /* 32 x 8 */
    {
        bool32x8_t bool32x8 [array_size];
        int32x8_t int32x8 [array_size];
        uint32x8_t uint32x8 [array_size];
        float32x8_t float32x8 [array_size];
        complex_float32x8_t complex_float32x8 [array_size];

        assert (alignment_test (&bool32x8 [0], bool32x8_t::alignment));
        assert (alignment_test (&int32x8 [0], int32x8_t::alignment));
        assert (alignment_test (&uint32x8 [0], uint32x8_t::alignment));
        assert (alignment_test (&float32x8 [0], float32x8_t::alignment));
        assert (
            alignment_test (
                &complex_float32x8 [0], complex_float32x8_t::alignment
            )
        );
    }

    /* 32 x 16 */
    {
        bool32x16_t bool32x16 [array_size];
        int32x16_t int32x16 [array_size];
        uint32x16_t uint32x16 [array_size];
        float32x16_t float32x16 [array_size];
        complex_float32x16_t complex_float32x16 [array_size];

        assert (alignment_test (&bool32x16 [0], bool32x16_t::alignment));
        assert (alignment_test (&int32x16 [0], int32x16_t::alignment));
        assert (alignment_test (&uint32x16 [0], uint32x16_t::alignment));
        assert (alignment_test (&float32x16 [0], float32x16_t::alignment));
        assert (
            alignment_test (
                &complex_float32x16 [0], complex_float32x16_t::alignment
            )
        );
    }

    /* 64 x 1 */
    {
        bool64x1_t bool64x1 [array_size];
        int64x1_t int64x1 [array_size];
        uint64x1_t uint64x1 [array_size];
        float64x1_t float64x1 [array_size];
        complex_float64x1_t complex_float64x1 [array_size];

        assert (alignment_test (&bool64x1 [0], bool64x1_t::alignment));
        assert (alignment_test (&int64x1 [0], int64x1_t::alignment));
        assert (alignment_test (&uint64x1 [0], uint64x1_t::alignment));
        assert (alignment_test (&float64x1 [0], float64x1_t::alignment));
        assert (
            alignment_test (
                &complex_float64x1 [0], complex_float64x1_t::alignment
            )
        );
    }

    /* 64 x 2 */
    {
        bool64x2_t bool64x2 [array_size];
        int64x2_t int64x2 [array_size];
        uint64x2_t uint64x2 [array_size];
        float64x2_t float64x2 [array_size];
        complex_float64x2_t complex_float64x2 [array_size];

        assert (alignment_test (&bool64x2 [0], bool64x2_t::alignment));
        assert (alignment_test (&int64x2 [0], int64x2_t::alignment));
        assert (alignment_test (&uint64x2 [0], uint64x2_t::alignment));
        assert (alignment_test (&float64x2 [0], float64x2_t::alignment));
        assert (
            alignment_test (
                &complex_float64x2 [0], complex_float64x2_t::alignment
            )
        );
    }

    /* 64 x 4 */
    {
        bool64x4_t bool64x4 [array_size];
        int64x4_t int64x4 [array_size];
        uint64x4_t uint64x4 [array_size];
        float64x4_t float64x4 [array_size];
        complex_float64x4_t complex_float64x4 [array_size];

        assert (alignment_test (&bool64x4 [0], bool64x4_t::alignment));
        assert (alignment_test (&int64x4 [0], int64x4_t::alignment));
        assert (alignment_test (&uint64x4 [0], uint64x4_t::alignment));
        assert (alignment_test (&float64x4 [0], float64x4_t::alignment));
        assert (
            alignment_test (
                &complex_float64x4 [0], complex_float64x4_t::alignment
            )
        );
    }

    /* 64 x 8 */
    {
        bool64x8_t bool64x8 [array_size];
        int64x8_t int64x8 [array_size];
        uint64x8_t uint64x8 [array_size];
        float64x8_t float64x8 [array_size];
        complex_float64x8_t complex_float64x8 [array_size];

        assert (alignment_test (&bool64x8 [0], bool64x8_t::alignment));
        assert (alignment_test (&int64x8 [0], int64x8_t::alignment));
        assert (alignment_test (&uint64x8 [0], uint64x8_t::alignment));
        assert (alignment_test (&float64x8 [0], float64x8_t::alignment));
        assert (
            alignment_test (
                &complex_float64x8 [0], complex_float64x8_t::alignment
            )
        );
    }

    /* long double x 2 */
    {
        long_doublex2_t long_doublex2 [array_size];
        complex_long_doublex2_t complex_long_doublex2 [array_size];

        assert (
            alignment_test (&long_doublex2 [0], long_doublex2_t::alignment)
        );
        assert (
            alignment_test (
                &complex_long_doublex2 [0], complex_long_doublex2_t::alignment
            )
        );
    }

    /* long double x 4 */
    {
        long_doublex4_t long_doublex4 [array_size];
        complex_long_doublex4_t complex_long_doublex4 [array_size];

        assert (
            alignment_test (&long_doublex4 [0], long_doublex4_t::alignment)
        );
        assert (
            alignment_test (
                &complex_long_doublex4 [0], complex_long_doublex4_t::alignment
            )
        );
    }

    /* 128 x 1 */
    {
        bool128x1_t bool128x1 [array_size];
        int128x1_t int128x1 [array_size];
        uint128x1_t uint128x1 [array_size];

        assert (alignment_test (&bool128x1 [0], bool128x1_t::alignment));
        assert (alignment_test (&int128x1 [0], int128x1_t::alignment));
        assert (alignment_test (&uint128x1 [0], uint128x1_t::alignment));
    }

    /* 128 x 2 */
    {
        bool128x2_t bool128x2 [array_size];
        int128x2_t int128x2 [array_size];
        uint128x2_t uint128x2 [array_size];

        assert (alignment_test (&bool128x2 [0], bool128x2_t::alignment));
        assert (alignment_test (&int128x2 [0], int128x2_t::alignment));
        assert (alignment_test (&uint128x2 [0], uint128x2_t::alignment));
    }

    /* 128 x 4 */
    {
        bool128x4_t bool128x4 [array_size];
        int128x4_t int128x4 [array_size];
        uint128x4_t uint128x4 [array_size];

        assert (alignment_test (&bool128x4 [0], bool128x4_t::alignment));
        assert (alignment_test (&int128x4 [0], int128x4_t::alignment));
        assert (alignment_test (&uint128x4 [0], uint128x4_t::alignment));
    }
}

void verify_dynamically_allocated_vars (void)
{
    using namespace simd::common;

    /* 8 x 8 */
    {
        auto bool8x8 = new bool8x8_t ();
        auto int8x8 = new int8x8_t ();
        auto uint8x8 = new uint8x8_t ();

        assert (alignment_test (bool8x8, bool8x8_t::alignment));
        assert (alignment_test (int8x8, int8x8_t::alignment));
        assert (alignment_test (uint8x8, uint8x8_t::alignment));

        delete bool8x8;
        delete int8x8;
        delete uint8x8;
    }

    /* 8 x 16 */
    {
        auto bool8x16 = new bool8x16_t ();
        auto int8x16 = new int8x16_t ();
        auto uint8x16 = new uint8x16_t ();

        assert (alignment_test (bool8x16, bool8x16_t::alignment));
        assert (alignment_test (int8x16, int8x16_t::alignment));
        assert (alignment_test (uint8x16, uint8x16_t::alignment));

        delete bool8x16;
        delete int8x16;
        delete uint8x16;
    }

    /* 8 x 32 */
    {
        auto bool8x32 = new bool8x32_t ();
        auto int8x32 = new int8x32_t ();
        auto uint8x32 = new uint8x32_t ();

        assert (alignment_test (bool8x32, bool8x32_t::alignment));
        assert (alignment_test (int8x32, int8x32_t::alignment));
        assert (alignment_test (uint8x32, uint8x32_t::alignment));

        delete bool8x32;
        delete int8x32;
        delete uint8x32;
    }

    /* 8 x 64 */
    {
        auto bool8x64 = new bool8x64_t ();
        auto int8x64 = new int8x64_t ();
        auto uint8x64 = new uint8x64_t ();

        assert (alignment_test (bool8x64, bool8x64_t::alignment));
        assert (alignment_test (int8x64, int8x64_t::alignment));
        assert (alignment_test (uint8x64, uint8x64_t::alignment));

        delete bool8x64;
        delete int8x64;
        delete uint8x64;
    }

    /* 16 x 4 */
    {
        auto bool16x4 = new bool16x4_t ();
        auto int16x4 = new int16x4_t ();
        auto uint16x4 = new uint16x4_t ();

        assert (alignment_test (bool16x4, bool16x4_t::alignment));
        assert (alignment_test (int16x4, int16x4_t::alignment));
        assert (alignment_test (uint16x4, uint16x4_t::alignment));

        delete bool16x4;
        delete int16x4;
        delete uint16x4;
    }

    /* 16 x 8 */
    {
        auto bool16x8 = new bool16x8_t ();
        auto int16x8 = new int16x8_t ();
        auto uint16x8 = new uint16x8_t ();

        assert (alignment_test (bool16x8, bool16x8_t::alignment));
        assert (alignment_test (int16x8, int16x8_t::alignment));
        assert (alignment_test (uint16x8, uint16x8_t::alignment));

        delete bool16x8;
        delete int16x8;
        delete uint16x8;
    }

    /* 16 x 16 */
    {
        auto bool16x16 = new bool16x16_t ();
        auto int16x16 = new int16x16_t ();
        auto uint16x16 = new uint16x16_t ();

        assert (alignment_test (bool16x16, bool16x16_t::alignment));
        assert (alignment_test (int16x16, int16x16_t::alignment));
        assert (alignment_test (uint16x16, uint16x16_t::alignment));

        delete bool16x16;
        delete int16x16;
        delete uint16x16;
    }

    /* 16 x 32 */
    {
        auto bool16x32 = new bool16x32_t ();
        auto int16x32 = new int16x32_t ();
        auto uint16x32 = new uint16x32_t ();

        assert (alignment_test (bool16x32, bool16x32_t::alignment));
        assert (alignment_test (int16x32, int16x32_t::alignment));
        assert (alignment_test (uint16x32, uint16x32_t::alignment));

        delete bool16x32;
        delete int16x32;
        delete uint16x32;
    }

    /* 32 x 2 */
    {
        auto bool32x2 = new bool32x2_t ();
        auto int32x2 = new int32x2_t ();
        auto uint32x2 = new uint32x2_t ();
        auto float32x2 = new float32x2_t ();
        auto complex_float32x2 = new complex_float32x2_t ();

        assert (alignment_test (bool32x2, bool32x2_t::alignment));
        assert (alignment_test (int32x2, int32x2_t::alignment));
        assert (alignment_test (uint32x2, uint32x2_t::alignment));
        assert (alignment_test (float32x2, float32x2_t::alignment));
        assert (
            alignment_test (complex_float32x2, complex_float32x2_t::alignment)
        );

        delete bool32x2;
        delete int32x2;
        delete uint32x2;
        delete float32x2;
        delete complex_float32x2;
    }

    /* 32 x 4 */
    {
        auto bool32x4 = new bool32x4_t ();
        auto int32x4 = new int32x4_t ();
        auto uint32x4 = new uint32x4_t ();
        auto float32x4 = new float32x4_t ();
        auto complex_float32x4 = new complex_float32x4_t ();

        assert (alignment_test (bool32x4, bool32x4_t::alignment));
        assert (alignment_test (int32x4, int32x4_t::alignment));
        assert (alignment_test (uint32x4, uint32x4_t::alignment));
        assert (alignment_test (float32x4, float32x4_t::alignment));
        assert (
            alignment_test (complex_float32x4, complex_float32x4_t::alignment)
        );

        delete bool32x4;
        delete int32x4;
        delete uint32x4;
        delete float32x4;
        delete complex_float32x4;
    }

    /* 32 x 8 */
    {
        auto bool32x8 = new bool32x8_t ();
        auto int32x8 = new int32x8_t ();
        auto uint32x8 = new uint32x8_t ();
        auto float32x8 = new float32x8_t ();
        auto complex_float32x8 = new complex_float32x8_t ();

        assert (alignment_test (bool32x8, bool32x8_t::alignment));
        assert (alignment_test (int32x8, int32x8_t::alignment));
        assert (alignment_test (uint32x8, uint32x8_t::alignment));
        assert (alignment_test (float32x8, float32x8_t::alignment));
        assert (
            alignment_test (complex_float32x8, complex_float32x8_t::alignment)
        );

        delete bool32x8;
        delete int32x8;
        delete uint32x8;
        delete float32x8;
        delete complex_float32x8;
    }

    /* 32 x 16 */
    {
        auto bool32x16 = new bool32x16_t ();
        auto int32x16 = new int32x16_t ();
        auto uint32x16 = new uint32x16_t ();
        auto float32x16 = new float32x16_t ();
        auto complex_float32x16 = new complex_float32x16_t ();

        assert (alignment_test (bool32x16, bool32x16_t::alignment));
        assert (alignment_test (int32x16, int32x16_t::alignment));
        assert (alignment_test (uint32x16, uint32x16_t::alignment));
        assert (alignment_test (float32x16, float32x16_t::alignment));
        assert (
            alignment_test (
                complex_float32x16, complex_float32x16_t::alignment
            )
        );

        delete bool32x16;
        delete int32x16;
        delete uint32x16;
        delete float32x16;
        delete complex_float32x16;
    }

    /* 64 x 1 */
    {
        auto bool64x1 = new bool64x1_t ();
        auto int64x1 = new int64x1_t ();
        auto uint64x1 = new uint64x1_t ();
        auto float64x1 = new float64x1_t ();
        auto complex_float64x1 = new complex_float64x1_t ();

        assert (alignment_test (bool64x1, bool64x1_t::alignment));
        assert (alignment_test (int64x1, int64x1_t::alignment));
        assert (alignment_test (uint64x1, uint64x1_t::alignment));
        assert (alignment_test (float64x1, float64x1_t::alignment));
        assert (
            alignment_test (complex_float64x1, complex_float64x1_t::alignment)
        );

        delete bool64x1;
        delete int64x1;
        delete uint64x1;
        delete float64x1;
        delete complex_float64x1;
    }

    /* 64 x 2 */
    {
        auto bool64x2 = new bool64x2_t ();
        auto int64x2 = new int64x2_t ();
        auto uint64x2 = new uint64x2_t ();
        auto float64x2 = new float64x2_t ();
        auto complex_float64x2 = new complex_float64x2_t ();

        assert (alignment_test (bool64x2, bool64x2_t::alignment));
        assert (alignment_test (int64x2, int64x2_t::alignment));
        assert (alignment_test (uint64x2, uint64x2_t::alignment));
        assert (alignment_test (float64x2, float64x2_t::alignment));
        assert (
            alignment_test (complex_float64x2, complex_float64x2_t::alignment)
        );

        delete bool64x2;
        delete int64x2;
        delete uint64x2;
        delete float64x2;
        delete complex_float64x2;
    }

    /* 64 x 4 */
    {
        auto bool64x4 = new bool64x4_t ();
        auto int64x4 = new int64x4_t ();
        auto uint64x4 = new uint64x4_t ();
        auto float64x4 = new float64x4_t ();
        auto complex_float64x4 = new complex_float64x4_t ();

        assert (alignment_test (bool64x4, bool64x4_t::alignment));
        assert (alignment_test (int64x4, int64x4_t::alignment));
        assert (alignment_test (uint64x4, uint64x4_t::alignment));
        assert (alignment_test (float64x4, float64x4_t::alignment));
        assert (
            alignment_test (complex_float64x4, complex_float64x4_t::alignment)
        );

        delete bool64x4;
        delete int64x4;
        delete uint64x4;
        delete float64x4;
        delete complex_float64x4;
    }

    /* 64 x 8 */
    {
        auto bool64x8 = new bool64x8_t ();
        auto int64x8 = new int64x8_t ();
        auto uint64x8 = new uint64x8_t ();
        auto float64x8 = new float64x8_t ();
        auto complex_float64x8 = new complex_float64x8_t ();

        assert (alignment_test (bool64x8, bool64x8_t::alignment));
        assert (alignment_test (int64x8, int64x8_t::alignment));
        assert (alignment_test (uint64x8, uint64x8_t::alignment));
        assert (alignment_test (float64x8, float64x8_t::alignment));
        assert (
            alignment_test (complex_float64x8, complex_float64x8_t::alignment)
        );

        delete bool64x8;
        delete int64x8;
        delete uint64x8;
        delete float64x8;
        delete complex_float64x8;
    }

    /* long double x 2 */
    {
        auto long_doublex2 = new long_doublex2_t ();
        auto complex_long_doublex2 = new complex_long_doublex2_t ();

        assert (alignment_test (long_doublex2, long_doublex2_t::alignment));
        assert (
            alignment_test (
                complex_long_doublex2, complex_long_doublex2_t::alignment
            )
        );

        delete long_doublex2;
        delete complex_long_doublex2;
    }

    /* long double x 4 */
    {
        auto long_doublex4 = new long_doublex4_t ();
        auto complex_long_doublex4 = new complex_long_doublex4_t ();

        assert (alignment_test (long_doublex4, long_doublex4_t::alignment));
        assert (
            alignment_test (
                complex_long_doublex4, complex_long_doublex4_t::alignment
            )
        );

        delete long_doublex4;
        delete complex_long_doublex4;
    }

    /* 128 x 1 */
    {
        auto bool128x1 = new bool128x1_t ();
        auto int128x1 = new int128x1_t ();
        auto uint128x1 = new uint128x1_t ();

        assert (alignment_test (bool128x1, bool128x1_t::alignment));
        assert (alignment_test (int128x1, int128x1_t::alignment));
        assert (alignment_test (uint128x1, uint128x1_t::alignment));

        delete bool128x1;
        delete int128x1;
        delete uint128x1;
    }

    /* 128 x 2 */
    {
        auto bool128x2 = new bool128x2_t ();
        auto int128x2 = new int128x2_t ();
        auto uint128x2 = new uint128x2_t ();

        assert (alignment_test (bool128x2, bool128x2_t::alignment));
        assert (alignment_test (int128x2, int128x2_t::alignment));
        assert (alignment_test (uint128x2, uint128x2_t::alignment));

        delete bool128x2;
        delete int128x2;
        delete uint128x2;
    }

    /* 128 x 4 */
    {
        auto bool128x4 = new bool128x4_t ();
        auto int128x4 = new int128x4_t ();
        auto uint128x4 = new uint128x4_t ();

        assert (alignment_test (bool128x4, bool128x4_t::alignment));
        assert (alignment_test (int128x4, int128x4_t::alignment));
        assert (alignment_test (uint128x4, uint128x4_t::alignment));

        delete bool128x4;
        delete int128x4;
        delete uint128x4;
    }
}

template <std::size_t array_size>
void verify_dynamically_allocated_array (void)
{
    using namespace simd::common;

    /* 8 x 8 */
    {
        auto bool8x8 = new bool8x8_t [array_size];
        auto int8x8 = new int8x8_t [array_size];
        auto uint8x8 = new uint8x8_t [array_size];

        assert (alignment_test (bool8x8, bool8x8_t::alignment));
        assert (alignment_test (int8x8, int8x8_t::alignment));
        assert (alignment_test (uint8x8, uint8x8_t::alignment));

        delete [] bool8x8;
        delete [] int8x8;
        delete [] uint8x8;
    }

    /* 8 x 16 */
    {
        auto bool8x16 = new bool8x16_t [array_size];
        auto int8x16 = new int8x16_t [array_size];
        auto uint8x16 = new uint8x16_t [array_size];

        assert (alignment_test (bool8x16, bool8x16_t::alignment));
        assert (alignment_test (int8x16, int8x16_t::alignment));
        assert (alignment_test (uint8x16, uint8x16_t::alignment));

        delete [] bool8x16;
        delete [] int8x16;
        delete [] uint8x16;
    }

    /* 8 x 32 */
    {
        auto bool8x32 = new bool8x32_t [array_size];
        auto int8x32 = new int8x32_t [array_size];
        auto uint8x32 = new uint8x32_t [array_size];

        assert (alignment_test (bool8x32, bool8x32_t::alignment));
        assert (alignment_test (int8x32, int8x32_t::alignment));
        assert (alignment_test (uint8x32, uint8x32_t::alignment));

        delete [] bool8x32;
        delete [] int8x32;
        delete [] uint8x32;
    }

    /* 8 x 64 */
    {
        auto bool8x64 = new bool8x64_t [array_size];
        auto int8x64 = new int8x64_t [array_size];
        auto uint8x64 = new uint8x64_t [array_size];

        assert (alignment_test (bool8x64, bool8x64_t::alignment));
        assert (alignment_test (int8x64, int8x64_t::alignment));
        assert (alignment_test (uint8x64, uint8x64_t::alignment));

        delete [] bool8x64;
        delete [] int8x64;
        delete [] uint8x64;
    }

    /* 16 x 4 */
    {
        auto bool16x4 = new bool16x4_t [array_size];
        auto int16x4 = new int16x4_t [array_size];
        auto uint16x4 = new uint16x4_t [array_size];

        assert (alignment_test (bool16x4, bool16x4_t::alignment));
        assert (alignment_test (int16x4, int16x4_t::alignment));
        assert (alignment_test (uint16x4, uint16x4_t::alignment));

        delete [] bool16x4;
        delete [] int16x4;
        delete [] uint16x4;
    }

    /* 16 x 8 */
    {
        auto bool16x8 = new bool16x8_t [array_size];
        auto int16x8 = new int16x8_t [array_size];
        auto uint16x8 = new uint16x8_t [array_size];

        assert (alignment_test (bool16x8, bool16x8_t::alignment));
        assert (alignment_test (int16x8, int16x8_t::alignment));
        assert (alignment_test (uint16x8, uint16x8_t::alignment));

        delete [] bool16x8;
        delete [] int16x8;
        delete [] uint16x8;
    }

    /* 16 x 16 */
    {
        auto bool16x16 = new bool16x16_t [array_size];
        auto int16x16 = new int16x16_t [array_size];
        auto uint16x16 = new uint16x16_t [array_size];

        assert (alignment_test (bool16x16, bool16x16_t::alignment));
        assert (alignment_test (int16x16, int16x16_t::alignment));
        assert (alignment_test (uint16x16, uint16x16_t::alignment));

        delete [] bool16x16;
        delete [] int16x16;
        delete [] uint16x16;
    }

    /* 16 x 32 */
    {
        auto bool16x32 = new bool16x32_t [array_size];
        auto int16x32 = new int16x32_t [array_size];
        auto uint16x32 = new uint16x32_t [array_size];

        assert (alignment_test (bool16x32, bool16x32_t::alignment));
        assert (alignment_test (int16x32, int16x32_t::alignment));
        assert (alignment_test (uint16x32, uint16x32_t::alignment));

        delete [] bool16x32;
        delete [] int16x32;
        delete [] uint16x32;
    }

    /* 32 x 2 */
    {
        auto bool32x2 = new bool32x2_t [array_size];
        auto int32x2 = new int32x2_t [array_size];
        auto uint32x2 = new uint32x2_t [array_size];
        auto float32x2 = new float32x2_t [array_size];
        auto complex_float32x2 = new complex_float32x2_t [array_size];

        assert (alignment_test (bool32x2, bool32x2_t::alignment));
        assert (alignment_test (int32x2, int32x2_t::alignment));
        assert (alignment_test (uint32x2, uint32x2_t::alignment));
        assert (alignment_test (float32x2, float32x2_t::alignment));
        assert (
            alignment_test (complex_float32x2, complex_float32x2_t::alignment)
        );

        delete [] bool32x2;
        delete [] int32x2;
        delete [] uint32x2;
        delete [] float32x2;
        delete [] complex_float32x2;
    }

    /* 32 x 4 */
    {
        auto bool32x4 = new bool32x4_t [array_size];
        auto int32x4 = new int32x4_t [array_size];
        auto uint32x4 = new uint32x4_t [array_size];
        auto float32x4 = new float32x4_t [array_size];
        auto complex_float32x4 = new complex_float32x4_t [array_size];

        assert (alignment_test (bool32x4, bool32x4_t::alignment));
        assert (alignment_test (int32x4, int32x4_t::alignment));
        assert (alignment_test (uint32x4, uint32x4_t::alignment));
        assert (alignment_test (float32x4, float32x4_t::alignment));
        assert (
            alignment_test (complex_float32x4, complex_float32x4_t::alignment)
        );

        delete [] bool32x4;
        delete [] int32x4;
        delete [] uint32x4;
        delete [] float32x4;
        delete [] complex_float32x4;
    }

    /* 32 x 8 */
    {
        auto bool32x8 = new bool32x8_t [array_size];
        auto int32x8 = new int32x8_t [array_size];
        auto uint32x8 = new uint32x8_t [array_size];
        auto float32x8 = new float32x8_t [array_size];
        auto complex_float32x8 = new complex_float32x8_t [array_size];

        assert (alignment_test (bool32x8, bool32x8_t::alignment));
        assert (alignment_test (int32x8, int32x8_t::alignment));
        assert (alignment_test (uint32x8, uint32x8_t::alignment));
        assert (alignment_test (float32x8, float32x8_t::alignment));
        assert (
            alignment_test (complex_float32x8, complex_float32x8_t::alignment)
        );

        delete [] bool32x8;
        delete [] int32x8;
        delete [] uint32x8;
        delete [] float32x8;
        delete [] complex_float32x8;
    }

    /* 32 x 16 */
    {
        auto bool32x16 = new bool32x16_t [array_size];
        auto int32x16 = new int32x16_t [array_size];
        auto uint32x16 = new uint32x16_t [array_size];
        auto float32x16 = new float32x16_t [array_size];
        auto complex_float32x16 = new complex_float32x16_t [array_size];

        assert (alignment_test (bool32x16, bool32x16_t::alignment));
        assert (alignment_test (int32x16, int32x16_t::alignment));
        assert (alignment_test (uint32x16, uint32x16_t::alignment));
        assert (alignment_test (float32x16, float32x16_t::alignment));
        assert (
            alignment_test (
                complex_float32x16, complex_float32x16_t::alignment
            )
        );

        delete [] bool32x16;
        delete [] int32x16;
        delete [] uint32x16;
        delete [] float32x16;
        delete [] complex_float32x16;
    }

    /* 64 x 1 */
    {
        auto bool64x1 = new bool64x1_t [array_size];
        auto int64x1 = new int64x1_t [array_size];
        auto uint64x1 = new uint64x1_t [array_size];
        auto float64x1 = new float64x1_t [array_size];
        auto complex_float64x1 = new complex_float64x1_t [array_size];

        assert (alignment_test (bool64x1, bool64x1_t::alignment));
        assert (alignment_test (int64x1, int64x1_t::alignment));
        assert (alignment_test (uint64x1, uint64x1_t::alignment));
        assert (alignment_test (float64x1, float64x1_t::alignment));
        assert (
            alignment_test (complex_float64x1, complex_float64x1_t::alignment)
        );

        delete [] bool64x1;
        delete [] int64x1;
        delete [] uint64x1;
        delete [] float64x1;
        delete [] complex_float64x1;
    }

    /* 64 x 2 */
    {
        auto bool64x2 = new bool64x2_t [array_size];
        auto int64x2 = new int64x2_t [array_size];
        auto uint64x2 = new uint64x2_t [array_size];
        auto float64x2 = new float64x2_t [array_size];
        auto complex_float64x2 = new complex_float64x2_t [array_size];

        assert (alignment_test (bool64x2, bool64x2_t::alignment));
        assert (alignment_test (int64x2, int64x2_t::alignment));
        assert (alignment_test (uint64x2, uint64x2_t::alignment));
        assert (alignment_test (float64x2, float64x2_t::alignment));
        assert (
            alignment_test (complex_float64x2, complex_float64x2_t::alignment)
        );

        delete [] bool64x2;
        delete [] int64x2;
        delete [] uint64x2;
        delete [] float64x2;
        delete [] complex_float64x2;
    }

    /* 64 x 4 */
    {
        auto bool64x4 = new bool64x4_t [array_size];
        auto int64x4 = new int64x4_t [array_size];
        auto uint64x4 = new uint64x4_t [array_size];
        auto float64x4 = new float64x4_t [array_size];
        auto complex_float64x4 = new complex_float64x4_t [array_size];

        assert (alignment_test (bool64x4, bool64x4_t::alignment));
        assert (alignment_test (int64x4, int64x4_t::alignment));
        assert (alignment_test (uint64x4, uint64x4_t::alignment));
        assert (alignment_test (float64x4, float64x4_t::alignment));
        assert (
            alignment_test (complex_float64x4, complex_float64x4_t::alignment)
        );

        delete [] bool64x4;
        delete [] int64x4;
        delete [] uint64x4;
        delete [] float64x4;
        delete [] complex_float64x4;
    }

    /* 64 x 8 */
    {
        auto bool64x8 = new bool64x8_t [array_size];
        auto int64x8 = new int64x8_t [array_size];
        auto uint64x8 = new uint64x8_t [array_size];
        auto float64x8 = new float64x8_t [array_size];
        auto complex_float64x8 = new complex_float64x8_t [array_size];

        assert (alignment_test (bool64x8, bool64x8_t::alignment));
        assert (alignment_test (int64x8, int64x8_t::alignment));
        assert (alignment_test (uint64x8, uint64x8_t::alignment));
        assert (alignment_test (float64x8, float64x8_t::alignment));
        assert (
            alignment_test (complex_float64x8, complex_float64x8_t::alignment)
        );

        delete [] bool64x8;
        delete [] int64x8;
        delete [] uint64x8;
        delete [] float64x8;
        delete [] complex_float64x8;
    }

    /* long double x 2 */
    {
        auto long_doublex2 = new long_doublex2_t [array_size];
        auto complex_long_doublex2 = new complex_long_doublex2_t [array_size];

        assert (alignment_test (long_doublex2, long_doublex2_t::alignment));
        assert (
            alignment_test (
                complex_long_doublex2, complex_long_doublex2_t::alignment
            )
        );

        delete [] long_doublex2;
        delete [] complex_long_doublex2;
    }

    /* long double x 4 */
    {
        auto long_doublex4 = new long_doublex4_t [array_size];
        auto complex_long_doublex4 = new complex_long_doublex4_t [array_size];

        assert (alignment_test (long_doublex4, long_doublex4_t::alignment));
        assert (
            alignment_test (
                complex_long_doublex4, complex_long_doublex4_t::alignment
            )
        );

        delete [] long_doublex4;
        delete [] complex_long_doublex4;
    }

    /* 128 x 1 */
    {
        auto bool128x1 = new bool128x1_t [array_size];
        auto int128x1 = new int128x1_t [array_size];
        auto uint128x1 = new uint128x1_t [array_size];

        assert (alignment_test (bool128x1, bool128x1_t::alignment));
        assert (alignment_test (int128x1, int128x1_t::alignment));
        assert (alignment_test (uint128x1, uint128x1_t::alignment));

        delete [] bool128x1;
        delete [] int128x1;
        delete [] uint128x1;
    }

    /* 128 x 2 */
    {
        auto bool128x2 = new bool128x2_t [array_size];
        auto int128x2 = new int128x2_t [array_size];
        auto uint128x2 = new uint128x2_t [array_size];

        assert (alignment_test (bool128x2, bool128x2_t::alignment));
        assert (alignment_test (int128x2, int128x2_t::alignment));
        assert (alignment_test (uint128x2, uint128x2_t::alignment));

        delete [] bool128x2;
        delete [] int128x2;
        delete [] uint128x2;
    }

    /* 128 x 4 */
    {
        auto bool128x4 = new bool128x4_t [array_size];
        auto int128x4 = new int128x4_t [array_size];
        auto uint128x4 = new uint128x4_t [array_size];

        assert (alignment_test (bool128x4, bool128x4_t::alignment));
        assert (alignment_test (int128x4, int128x4_t::alignment));
        assert (alignment_test (uint128x4, uint128x4_t::alignment));

        delete [] bool128x4;
        delete [] int128x4;
        delete [] uint128x4;
    }
}

template <std::size_t array_size>
void verify_vector_allocated_vars (void)
{
    using namespace simd::common;

    /* 8 x 8 */
    {
        auto bool8x8 = std::vector <bool8x8_t, simd_allocator <bool8x8_t>> (array_size);
        auto int8x8 = std::vector <int8x8_t, simd_allocator <int8x8_t>> (array_size);
        auto uint8x8 = std::vector <uint8x8_t, simd_allocator <uint8x8_t>> (array_size);

        assert (alignment_test (bool8x8.data (), bool8x8_t::alignment));
        assert (alignment_test (int8x8.data (), int8x8_t::alignment));
        assert (alignment_test (uint8x8.data (), uint8x8_t::alignment));
    }

    /* 8 x 16 */
    {
        auto bool8x16 = std::vector <bool8x16_t, simd_allocator <bool8x16_t>> (array_size);
        auto int8x16 = std::vector <int8x16_t, simd_allocator <int8x16_t>> (array_size);
        auto uint8x16 = std::vector <uint8x16_t, simd_allocator <uint8x16_t>> (array_size);

        assert (alignment_test (bool8x16.data (), bool8x16_t::alignment));
        assert (alignment_test (int8x16.data (), int8x16_t::alignment));
        assert (alignment_test (uint8x16.data (), uint8x16_t::alignment));
    }

    /* 8 x 32 */
    {
        auto bool8x32 = std::vector <bool8x32_t, simd_allocator <bool8x32_t>> (array_size);
        auto int8x32 = std::vector <int8x32_t, simd_allocator <int8x32_t>> (array_size);
        auto uint8x32 = std::vector <uint8x32_t, simd_allocator <uint8x32_t>> (array_size);

        assert (alignment_test (bool8x32.data (), bool8x32_t::alignment));
        assert (alignment_test (int8x32.data (), int8x32_t::alignment));
        assert (alignment_test (uint8x32.data (), uint8x32_t::alignment));
    }

    /* 8 x 64 */
    {
        auto bool8x64 = std::vector <bool8x64_t, simd_allocator <bool8x64_t>> (array_size);
        auto int8x64 = std::vector <int8x64_t, simd_allocator <int8x64_t>> (array_size);
        auto uint8x64 = std::vector <uint8x64_t, simd_allocator <uint8x64_t>> (array_size);

        assert (alignment_test (bool8x64.data (), bool8x64_t::alignment));
        assert (alignment_test (int8x64.data (), int8x64_t::alignment));
        assert (alignment_test (uint8x64.data (), uint8x64_t::alignment));
    }

    /* 16 x 4 */
    {
        auto bool16x4 = std::vector <bool16x4_t, simd_allocator <bool16x4_t>> (array_size);
        auto int16x4 = std::vector <int16x4_t, simd_allocator <int16x4_t>> (array_size);
        auto uint16x4 = std::vector <uint16x4_t, simd_allocator <uint16x4_t>> (array_size);

        assert (alignment_test (bool16x4.data (), bool16x4_t::alignment));
        assert (alignment_test (int16x4.data (), int16x4_t::alignment));
        assert (alignment_test (uint16x4.data (), uint16x4_t::alignment));
    }

    /* 16 x 8 */
    {
        auto bool16x8 = std::vector <bool16x8_t, simd_allocator <bool16x8_t>> (array_size);
        auto int16x8 = std::vector <int16x8_t, simd_allocator <int16x8_t>> (array_size);
        auto uint16x8 = std::vector <uint16x8_t, simd_allocator <uint16x8_t>> (array_size);

        assert (alignment_test (bool16x8.data (), bool16x8_t::alignment));
        assert (alignment_test (int16x8.data (), int16x8_t::alignment));
        assert (alignment_test (uint16x8.data (), uint16x8_t::alignment));
    }

    /* 16 x 16 */
    {
        auto bool16x16 = std::vector <bool16x16_t, simd_allocator <bool16x16_t>> (array_size);
        auto int16x16 = std::vector <int16x16_t, simd_allocator <int16x16_t>> (array_size);
        auto uint16x16 = std::vector <uint16x16_t, simd_allocator <uint16x16_t>> (array_size);

        assert (alignment_test (bool16x16.data (), bool16x16_t::alignment));
        assert (alignment_test (int16x16.data (), int16x16_t::alignment));
        assert (alignment_test (uint16x16.data (), uint16x16_t::alignment));
    }

    /* 16 x 32 */
    {
        auto bool16x32 = std::vector <bool16x32_t, simd_allocator <bool16x32_t>> (array_size);
        auto int16x32 = std::vector <int16x32_t, simd_allocator <int16x32_t>> (array_size);
        auto uint16x32 = std::vector <uint16x32_t, simd_allocator <uint16x32_t>> (array_size);

        assert (alignment_test (bool16x32.data (), bool16x32_t::alignment));
        assert (alignment_test (int16x32.data (), int16x32_t::alignment));
        assert (alignment_test (uint16x32.data (), uint16x32_t::alignment));
    }

    /* 32 x 2 */
    {
        auto bool32x2 = std::vector <bool32x2_t, simd_allocator <bool32x2_t>> (array_size);
        auto int32x2 = std::vector <int32x2_t, simd_allocator <int32x2_t>> (array_size);
        auto uint32x2 = std::vector <uint32x2_t, simd_allocator <uint32x2_t>> (array_size);
        auto float32x2 = std::vector <float32x2_t, simd_allocator <float32x2_t>> (array_size);
        auto complex_float32x2 = std::vector <complex_float32x2_t, simd_allocator <complex_float32x2_t>> (array_size);

        assert (alignment_test (bool32x2.data (), bool32x2_t::alignment));
        assert (alignment_test (int32x2.data (), int32x2_t::alignment));
        assert (alignment_test (uint32x2.data (), uint32x2_t::alignment));
        assert (alignment_test (float32x2.data (), float32x2_t::alignment));
        assert (
            alignment_test (complex_float32x2.data (), complex_float32x2_t::alignment)
        );
    }

    /* 32 x 4 */
    {
        auto bool32x4 = std::vector <bool32x4_t, simd_allocator <bool32x4_t>> (array_size);
        auto int32x4 = std::vector <int32x4_t, simd_allocator <int32x4_t>> (array_size);
        auto uint32x4 = std::vector <uint32x4_t, simd_allocator <uint32x4_t>> (array_size);
        auto float32x4 = std::vector <float32x4_t, simd_allocator <float32x4_t>> (array_size);
        auto complex_float32x4 = std::vector <complex_float32x4_t, simd_allocator <complex_float32x4_t>> (array_size);

        assert (alignment_test (bool32x4.data (), bool32x4_t::alignment));
        assert (alignment_test (int32x4.data (), int32x4_t::alignment));
        assert (alignment_test (uint32x4.data (), uint32x4_t::alignment));
        assert (alignment_test (float32x4.data (), float32x4_t::alignment));
        assert (
            alignment_test (complex_float32x4.data (), complex_float32x4_t::alignment)
        );
    }

    /* 32 x 8 */
    {
        auto bool32x8 = std::vector <bool32x8_t, simd_allocator <bool32x8_t>> (array_size);
        auto int32x8 = std::vector <int32x8_t, simd_allocator <int32x8_t>> (array_size);
        auto uint32x8 = std::vector <uint32x8_t, simd_allocator <uint32x8_t>> (array_size);
        auto float32x8 = std::vector <float32x8_t, simd_allocator <float32x8_t>> (array_size);
        auto complex_float32x8 = std::vector <complex_float32x8_t, simd_allocator <complex_float32x8_t>> (array_size);

        assert (alignment_test (bool32x8.data (), bool32x8_t::alignment));
        assert (alignment_test (int32x8.data (), int32x8_t::alignment));
        assert (alignment_test (uint32x8.data (), uint32x8_t::alignment));
        assert (alignment_test (float32x8.data (), float32x8_t::alignment));
        assert (
            alignment_test (complex_float32x8.data (), complex_float32x8_t::alignment)
        );
    }

    /* 32 x 16 */
    {
        auto bool32x16 = std::vector <bool32x16_t, simd_allocator <bool32x16_t>> (array_size);
        auto int32x16 = std::vector <int32x16_t, simd_allocator <int32x16_t>> (array_size);
        auto uint32x16 = std::vector <uint32x16_t, simd_allocator <uint32x16_t>> (array_size);
        auto float32x16 = std::vector <float32x16_t, simd_allocator <float32x16_t>> (array_size);
        auto complex_float32x16 = std::vector <complex_float32x16_t, simd_allocator <complex_float32x16_t>> (array_size);

        assert (alignment_test (bool32x16.data (), bool32x16_t::alignment));
        assert (alignment_test (int32x16.data (), int32x16_t::alignment));
        assert (alignment_test (uint32x16.data (), uint32x16_t::alignment));
        assert (alignment_test (float32x16.data (), float32x16_t::alignment));
        assert (
            alignment_test (
                complex_float32x16.data (), complex_float32x16_t::alignment
            )
        );
    }

    /* 64 x 1 */
    {
        auto bool64x1 = std::vector <bool64x1_t, simd_allocator <bool64x1_t>> (array_size);
        auto int64x1 = std::vector <int64x1_t, simd_allocator <int64x1_t>> (array_size);
        auto uint64x1 = std::vector <uint64x1_t, simd_allocator <uint64x1_t>> (array_size);
        auto float64x1 = std::vector <float64x1_t, simd_allocator <float64x1_t>> (array_size);
        auto complex_float64x1 = std::vector <complex_float64x1_t, simd_allocator <complex_float64x1_t>> (array_size);

        assert (alignment_test (bool64x1.data (), bool64x1_t::alignment));
        assert (alignment_test (int64x1.data (), int64x1_t::alignment));
        assert (alignment_test (uint64x1.data (), uint64x1_t::alignment));
        assert (alignment_test (float64x1.data (), float64x1_t::alignment));
        assert (
            alignment_test (complex_float64x1.data (), complex_float64x1_t::alignment)
        );
    }

    /* 64 x 2 */
    {
        auto bool64x2 = std::vector <bool64x2_t, simd_allocator <bool64x2_t>> (array_size);
        auto int64x2 = std::vector <int64x2_t, simd_allocator <int64x2_t>> (array_size);
        auto uint64x2 = std::vector <uint64x2_t, simd_allocator <uint64x2_t>> (array_size);
        auto float64x2 = std::vector <float64x2_t, simd_allocator <float64x2_t>> (array_size);
        auto complex_float64x2 = std::vector <complex_float64x2_t, simd_allocator <complex_float64x2_t>> (array_size);

        assert (alignment_test (bool64x2.data (), bool64x2_t::alignment));
        assert (alignment_test (int64x2.data (), int64x2_t::alignment));
        assert (alignment_test (uint64x2.data (), uint64x2_t::alignment));
        assert (alignment_test (float64x2.data (), float64x2_t::alignment));
        assert (
            alignment_test (complex_float64x2.data (), complex_float64x2_t::alignment)
        );
    }

    /* 64 x 4 */
    {
        auto bool64x4 = std::vector <bool64x4_t, simd_allocator <bool64x4_t>> (array_size);
        auto int64x4 = std::vector <int64x4_t, simd_allocator <int64x4_t>> (array_size);
        auto uint64x4 = std::vector <uint64x4_t, simd_allocator <uint64x4_t>> (array_size);
        auto float64x4 = std::vector <float64x4_t, simd_allocator <float64x4_t>> (array_size);
        auto complex_float64x4 = std::vector <complex_float64x4_t, simd_allocator <complex_float64x4_t>> (array_size);

        assert (alignment_test (bool64x4.data (), bool64x4_t::alignment));
        assert (alignment_test (int64x4.data (), int64x4_t::alignment));
        assert (alignment_test (uint64x4.data (), uint64x4_t::alignment));
        assert (alignment_test (float64x4.data (), float64x4_t::alignment));
        assert (
            alignment_test (complex_float64x4.data (), complex_float64x4_t::alignment)
        );
    }

    /* 64 x 8 */
    {
        auto bool64x8 = std::vector <bool64x8_t, simd_allocator <bool64x8_t>> (array_size);
        auto int64x8 = std::vector <int64x8_t, simd_allocator <int64x8_t>> (array_size);
        auto uint64x8 = std::vector <uint64x8_t, simd_allocator <uint64x8_t>> (array_size);
        auto float64x8 = std::vector <float64x8_t, simd_allocator <float64x8_t>> (array_size);
        auto complex_float64x8 = std::vector <complex_float64x8_t, simd_allocator <complex_float64x8_t>> (array_size);

        assert (alignment_test (bool64x8.data (), bool64x8_t::alignment));
        assert (alignment_test (int64x8.data (), int64x8_t::alignment));
        assert (alignment_test (uint64x8.data (), uint64x8_t::alignment));
        assert (alignment_test (float64x8.data (), float64x8_t::alignment));
        assert (
            alignment_test (complex_float64x8.data (), complex_float64x8_t::alignment)
        );
    }

    /* long double x 2 */
    {
        auto long_doublex2 = std::vector <long_doublex2_t, simd_allocator <long_doublex2_t>> (array_size);
        auto complex_long_doublex2 = std::vector <complex_long_doublex2_t, simd_allocator <complex_long_doublex2_t>> (array_size);

        assert (alignment_test (long_doublex2.data (), long_doublex2_t::alignment));
        assert (
            alignment_test (
                complex_long_doublex2.data (), complex_long_doublex2_t::alignment
            )
        );
    }

    /* long double x 4 */
    {
        auto long_doublex4 = std::vector <long_doublex4_t, simd_allocator <long_doublex4_t>> (array_size);
        auto complex_long_doublex4 = std::vector <complex_long_doublex4_t, simd_allocator <complex_long_doublex4_t>> (array_size);

        assert (alignment_test (long_doublex4.data (), long_doublex4_t::alignment));
        assert (
            alignment_test (
                complex_long_doublex4.data (), complex_long_doublex4_t::alignment
            )
        );
    }

    /* 128 x 1 */
    {
        auto bool128x1 = std::vector <bool128x1_t, simd_allocator <bool128x1_t>> (array_size);
        auto int128x1 = std::vector <int128x1_t, simd_allocator <int128x1_t>> (array_size);
        auto uint128x1 = std::vector <uint128x1_t, simd_allocator <uint128x1_t>> (array_size);

        assert (alignment_test (bool128x1.data (), bool128x1_t::alignment));
        assert (alignment_test (int128x1.data (), int128x1_t::alignment));
        assert (alignment_test (uint128x1.data (), uint128x1_t::alignment));
    }

    /* 128 x 2 */
    {
        auto bool128x2 = std::vector <bool128x2_t, simd_allocator <bool128x2_t>> (array_size);
        auto int128x2 = std::vector <int128x2_t, simd_allocator <int128x2_t>> (array_size);
        auto uint128x2 = std::vector <uint128x2_t, simd_allocator <uint128x2_t>> (array_size);

        assert (alignment_test (bool128x2.data (), bool128x2_t::alignment));
        assert (alignment_test (int128x2.data (), int128x2_t::alignment));
        assert (alignment_test (uint128x2.data (), uint128x2_t::alignment));
    }

    /* 128 x 4 */
    {
        auto bool128x4 = std::vector <bool128x4_t, simd_allocator <bool128x4_t>> (array_size);
        auto int128x4 = std::vector <int128x4_t, simd_allocator <int128x4_t>> (array_size);
        auto uint128x4 = std::vector <uint128x4_t, simd_allocator <uint128x4_t>> (array_size);

        assert (alignment_test (bool128x4.data (), bool128x4_t::alignment));
        assert (alignment_test (int128x4.data (), int128x4_t::alignment));
        assert (alignment_test (uint128x4.data (), uint128x4_t::alignment));
    }
}

int main (void)
{
    verify_statically_allocated_vars ();
    verify_statically_allocated_array <1> ();
    verify_statically_allocated_array <2> ();
    verify_statically_allocated_array <4> ();
    verify_statically_allocated_array <8> ();
    verify_statically_allocated_array <10> ();
    verify_statically_allocated_array <16> ();
    verify_statically_allocated_array <32> ();
    verify_statically_allocated_array <64> ();
    verify_statically_allocated_array <100> ();

    verify_automatic_storage_vars ();
    verify_automatic_storage_array <1> ();
    verify_automatic_storage_array <2> ();
    verify_automatic_storage_array <4> ();
    verify_automatic_storage_array <8> ();
    verify_automatic_storage_array <10> ();
    verify_automatic_storage_array <16> ();
    verify_automatic_storage_array <32> ();
    verify_automatic_storage_array <64> ();
    verify_automatic_storage_array <100> ();

    verify_dynamically_allocated_vars ();
    verify_dynamically_allocated_array <0> ();
    verify_dynamically_allocated_array <1> ();
    verify_dynamically_allocated_array <2> ();
    verify_dynamically_allocated_array <4> ();
    verify_dynamically_allocated_array <8> ();
    verify_dynamically_allocated_array <10> ();
    verify_dynamically_allocated_array <16> ();
    verify_dynamically_allocated_array <32> ();
    verify_dynamically_allocated_array <64> ();
    verify_dynamically_allocated_array <100> ();

    verify_vector_allocated_vars <0> ();
    verify_vector_allocated_vars <1> ();
    verify_vector_allocated_vars <2> ();
    verify_vector_allocated_vars <4> ();
    verify_vector_allocated_vars <8> ();
    verify_vector_allocated_vars <10> ();
    verify_vector_allocated_vars <16> ();
    verify_vector_allocated_vars <32> ();
    verify_vector_allocated_vars <64> ();
    verify_vector_allocated_vars <100> ();

    return 0;
}
