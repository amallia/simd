//
// simd; an implementation of SIMD vectors for C++ using GCC vector instrinsics
//
// author: Dalton Woodard
// contact: daltonmwoodard@gmail.com
// official repository: https://github.com/daltonwoodard/simd.git
// license:
//
// Copyright (c) 2016 DaltonWoodard. See the COPYRIGHT.md file at the top-level
// directory or at the listed source repository for details.
//
//      Licensed under the Apache License. Version 2.0:
//          https://www.apache.org/licenses/LICENSE-2.0
//      or the MIT License:
//          https://opensource.org/licenses/MIT
//      at the licensee's option. This file may not be copied, modified, or
//      distributed except according to those terms.
//

#ifndef SIMD_IMPLEMENTATION_HEADER
#define SIMD_IMPLEMENTATION_HEADER

#include <array>                // std::array
#include <cassert>              // std::assert
#include <cfloat>               // FLT_RADIX
#include <complex>              // std::complex
#include <cstddef>              // std::size_t
#include <cstdint>              // std::{u}int{8,16,32,64}_t
#include <cstring>              // std::memcpy
#include <functional>           // std::hash
#include <iterator>             // std::iterator, std::reverse_iterator
#include <memory>               // std::align, std::addressof
#include <mutex>                // std::lock_guard, std::mutex
#include <new>                  // std::{get,set}_new_handler
#include <numeric>              // std::accumulate
#include <stdexcept>            // std::bad_alloc
#include <type_traits>          // std::conditional, std::is_arithmetic
#include <utility>              // util::index_sequence

#if !defined (__clang__) && !defined (__GNUG__)
    #error "simd implemention requires clang or gcc vector extensions"
#endif

#if __cplusplus < 201103L
    #error "simd implementation requires C++11 support"
#endif

#if __cplusplus >= 201402L
    #define advanced_constexpr constexpr
#else
    #define advanced_constexpr
#endif

#if defined (__arm__)
    #if !(defined (__ARM_NEON__) || defined (__ARM_NEON))
        #error "simd implementation requires ARM NEON extension support"
    #else
        #include <arm_neon.h>
        #define simd_arm
        #define simd_neon
    #endif
#endif

#if defined (__x86__)\
 || defined (__x86_64__)\
 || defined (__x86_64)\
 || defined (__amd64__)\
 || defined (__amd64)
    #if !(defined (__MMX__)\
       || defined (__SSE__)\
       || defined (__SSE2__)\
       || defined (__SSE3__)\
       || defined (__SSSE3__)\
       || defined (__SSE4_1__)\
       || defined (__SSE4_2__)\
       || defined (__AVX__)\
       || defined (__AVX2__)\
       || defined (__AVX512F__))
        #error "simd implementation requires x86 SIMD extension support"
    #else
        #include <x86intrin.h>
        #define simd_x86
        #if defined (__MMX__)
            #define simd_mmx
        #endif
        #if defined (__SSE__)
            #define simd_sse
        #endif
        #if defined (__SSE2__)
            #define simd_sse2
        #endif
        #if defined (__SSE3__)
            #define simd_sse3
        #endif
        #if defined (__SSSE3__)
            #define simd_ssse3
        #endif
        #if defined (__SSE4_1__)
            #define simd_see4_1
        #endif
        #if defined (__SSE4_2__)
            #define simd_see4_2
        #endif
        #if defined (__AVX__)
            #define simd_avx
        #endif
        #if defined (__AVX2__)
            #define simd_avx2
        #endif
        #if defined (__AVX512F__)
            #define simd_avx512
        #endif
    #endif
#endif


namespace simd
{
namespace detail
{
namespace util
{
#if __cplusplus >= 201402L
    template <std::size_t ... v>
    using index_sequence = std::index_sequence <v...>;

    template <std::size_t N>
    using make_index_sequence = std::make_index_sequence <N>;
#else
    template <std::size_t ... v>
    struct index_sequence
    {
        using type = index_sequence;
        using value_type = std::size_t;

        static constexpr std::size_t size (void) noexcept
        {
            return sizeof... (v);
        }
    };

    template <typename, typename>
    struct merge;

    template <std::size_t ... v1, std::size_t ... v2>
    struct merge <index_sequence <v1...>, index_sequence <v2...>>
        : index_sequence <v1..., (sizeof... (v1) + v2)...>
    {};

    template <std::size_t N>
    struct seq_gen : merge <
        typename seq_gen <N/2>::type,
        typename seq_gen <N - N/2>::type
    >
    {};

    template <>
    struct seq_gen <0> : index_sequence <> {};

    template <>
    struct seq_gen <1> : index_sequence <0> {};

    template <std::size_t N>
    using make_index_sequence = typename seq_gen <N>::type;
#endif

    /*
     * Implemented for use in custom new implementation;
     * this method is threasafe, and consequently calls to new
     * on SIMD vector types are threadsafe as well (this becomes
     * a concern only in failing cases for memory allocation, which
     * typically will not occur on modern OSs that have overcommit
     * semantics).
     */
    void attempt_global_new_handler_call (void);
    void attempt_global_new_handler_call (void)
    {
        static std::mutex m;
        std::lock_guard <std::mutex> lock {m};

        auto global_new_handler = std::set_new_handler (nullptr);
        std::set_new_handler (global_new_handler);

        if (global_new_handler) {
            global_new_handler ();
        } else {
            throw std::bad_alloc {};
        }
    }

    /*
     * Allocates a block of memory of size bytes with alignment align.
     */
    void * aligned_allocate (std::size_t, std::size_t);
    void * aligned_allocate (std::size_t size, std::size_t alignment)
    {
#if __cplusplus > 201402L
        /* do we have C++17 support? */
        /* then use operator new with alignment spec */
        while (true) {
            auto const alloc_mem = ::operator new (size, alignment);
            if (alloc_mem) {
                return alloc_mem;
            } else {
                util::attempt_global_new_handler_call ();
            }
        }
#else
        /* then use our own implementation */
        while (true) {
            auto const alloc_sz  = size + alignment + sizeof (void *);
            auto const alloc_mem = ::operator new (alloc_sz);
            if (alloc_mem) {
                auto parg = static_cast <void *> (
                    reinterpret_cast <void **> (alloc_mem) + 1
                );
                auto sarg = size + alignment;
                auto const ptr = std::align (alignment, size, parg, sarg);
                reinterpret_cast <void **> (ptr) [-1] = alloc_mem;
                return ptr;
            } else {
                util::attempt_global_new_handler_call ();
            }
        }
#endif
    }

    /*
     * Deallocates a block of memory of size bytes with alignment align.
     */
    void aligned_deallocate (void *, std::size_t, std::size_t) noexcept;
    void aligned_deallocate (void * p, std::size_t size, std::size_t alignment)
        noexcept
    {
        if (!p) {
            return;
        }

#if __cplusplus > 201402L
        ::operator delete (p, size, alignment);
#else
        auto const alloc_size = size + alignment + sizeof (void *);
    #if __cplusplus >= 201402L
        ::operator delete (reinterpret_cast <void **> (p) [-1], alloc_size);
    #else
        (void) alloc_size;
        ::operator delete (reinterpret_cast <void **> (p) [-1]);
    #endif
#endif
    }

    /*
     * Hash combine for specialization of std::hash for SIMD vector types.
     */
    template <typename T>
    void hash_combine (std::size_t & seed, T const & t) noexcept
    {
        std::hash <T> hfn {};
        seed ^= hfn (t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    /*
     * Non-modifying hash combine for specialization of std::hash for SIMD
     * vector types.
     */
    template <typename T>
    std::size_t hash_combine (std::size_t const & seed, T const & t)
        noexcept
    {
        std::hash <T> hfn {};
        auto sd = seed;
        sd ^= hfn (t) + 0x9e3779b9 + (sd << 6) + (sd >> 2);
        return sd;
    }

    /*
     * Hash combine for specialization of std::hash for SIMD vector types.
     */
    template <>
#if defined (__clang__)
    void hash_combine (std::size_t & seed, __int128_t const & t) noexcept
#elif defined (__GNUG__)
    void hash_combine (std::size_t & seed, __int128 const & t) noexcept
#endif
    {
        struct alias {
            std::int64_t a;
            std::int64_t b;
        };

        auto const & a = reinterpret_cast <alias const &> (t);
        std::hash <std::int64_t> hfn {};

        seed ^= hfn (a.a) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hfn (a.b) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    /*
     * Non-modifying hash combine for specialization of std::hash for SIMD
     * vector types.
     */
    template <>
#if defined (__clang__)
    std::size_t hash_combine (std::size_t const & seed, __int128_t const & t)
        noexcept
#elif defined (__GNUG__)
    std::size_t hash_combine (std::size_t const & seed, __int128 const & t)
        noexcept
#endif
    {
        struct alias {
            std::int64_t a;
            std::int64_t b;
        };

        auto const & a = reinterpret_cast <alias const &> (t);
        std::hash <std::int64_t> hfn {};

        auto sd = seed;
        sd ^= hfn (a.a) + 0x9e3779b9 + (sd << 6) + (sd >> 2);
        sd ^= hfn (a.b) + 0x9e3779b9 + (sd << 6) + (sd >> 2);
        return sd;
    }

    /*
     * Hash combine for specialization of std::hash for SIMD vector types.
     */
    template <>
#if defined (__clang__)
    void hash_combine (std::size_t & seed, __uint128_t const & t) noexcept
#elif defined (__GNUG__)
    void hash_combine (std::size_t & seed, unsigned __int128 const & t) noexcept
#endif
    {
        struct alias {
            std::uint64_t a;
            std::uint64_t b;
        };

        auto const & a = reinterpret_cast <alias const &> (t);
        std::hash <std::uint64_t> hfn {};

        seed ^= hfn (a.a) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hfn (a.b) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    /*
     * Non-modifying hash combine for specialization of std::hash for SIMD
     * vector types.
     */
    template <>
#if defined (__clang__)
    std::size_t hash_combine (std::size_t const & seed, __uint128_t const & t)
        noexcept
#elif defined (__GNUG__)
    std::size_t hash_combine (std::size_t const & seed,
                              unsigned __int128 const & t) noexcept
#endif
    {
        struct alias {
            std::uint64_t a;
            std::uint64_t b;
        };

        auto const & a = reinterpret_cast <alias const &> (t);
        std::hash <std::uint64_t> hfn {};

        auto sd = seed;
        sd ^= hfn (a.a) + 0x9e3779b9 + (sd << 6) + (sd >> 2);
        sd ^= hfn (a.b) + 0x9e3779b9 + (sd << 6) + (sd >> 2);
        return sd;
    }
}   // namespace util

    /*
     * Due to implementation details in the clang C++ compiler, it is impossible
     * to declare templated typedefs of vector extension types. Moreover, the
     * declaration requires an integer literal for the vector_size attribute.
     * Therefore we must create a list of all the possible specializations for
     * the underlying vector types.
     *
     * It should also be noted that we specialize for vector types which are
     * technically smaller and larger than "true" SIMD vector types for any
     * particular architecture. For each possible base type we provide vector
     * types with lane counts: 1, 2, 4, 8, 16, 32, 64. This is to allow general
     * mappings over SIMD vector types without fear that the resulting SIMD
     * vector type does not have a defined vector type specialization.
     *
     * This is okay since both Clang and GCC will synthesize instructions that
     * are not present on the target architecure.
     *
     * It should also be noted that we provide alignment values equal to the
     * size of each vector type. This is required to prevent penalties or
     * exceptions for unaligned memory accesses on architectures supporting only
     * aligned accesses for SIMD vector types. Often these alignment values will
     * be larger than the value alignof (std::max_align_t), and so with GCC in
     * particular warnings of attribute alignment greater than
     * alignof (std::max_align_t) will be emitted. I have taken the liberty to
     * insert #pragma GCC diagnostic push/pop/ignored "-Wattributes" blocks
     * around declarations using alignas to supporess these warnings (these are
     * the only place #pragma blocks are used). They are not necessary and can
     * be searched for and removed if desired.
     *
     * Each SIMD vector type (depending on underlying type and lane count),
     * requires backing either by a particular SIMD technology or synthesized
     * instructions when no appropriate SIMD technology is available on the
     * target architecure. In the case of the latter GCC may emit warnings
     * about vector return types (in the -Wpsabi catgeory). It may also be
     * useful to the user of this library to explicitly enable the target SIMD
     * technology they wish to use, this may be one of, but is not limited to:
     * -mmmx, -msse, -msse2, -msse3, -mssse3, -msse4, -msse4.{1,2}, -mavx,
     * -mavx2, -mavx512{f,bw,cd,dq,er,ifma,pf,vbmi,vl}, -mneon.
     */
namespace vext
{
    template <typename, std::size_t, typename enable = void>
    struct vector_type_specialization;

template <std::size_t lanes>
struct vector_type_specialization <signed char, lanes>
    : public vector_type_specialization <char, lanes>
{};

#define char_spec(lanes) template <>\
struct vector_type_specialization <char, lanes>\
{\
    typedef char type\
        __attribute__ ((vector_size (lanes * sizeof (char))));\
    static constexpr std::size_t alignment = lanes * alignof (char);\
    static constexpr std::size_t size = lanes * sizeof (char);\
};

    char_spec(1);
    char_spec(2);
    char_spec(4);
    char_spec(8);
    char_spec(16);
    char_spec(32);
    char_spec(64);

#undef char_spec

#define unsigned_char_spec(lanes) template <>\
struct vector_type_specialization <unsigned char, lanes>\
{\
    typedef unsigned char type\
        __attribute__ ((vector_size (lanes * sizeof (unsigned char))));\
    static constexpr std::size_t alignment = lanes * alignof (unsigned char);\
    static constexpr std::size_t size = lanes * sizeof (unsigned char);\
};

    unsigned_char_spec(1);
    unsigned_char_spec(2);
    unsigned_char_spec(4);
    unsigned_char_spec(8);
    unsigned_char_spec(16);
    unsigned_char_spec(32);
    unsigned_char_spec(64);

#undef unsigned_char_spec

#define short_spec(lanes) template <>\
struct vector_type_specialization <short, lanes>\
{\
    typedef short type\
        __attribute__ ((vector_size (lanes * sizeof (short))));\
    static constexpr std::size_t alignment = lanes * alignof (short);\
    static constexpr std::size_t size = lanes * sizeof (short);\
};

    short_spec(1);
    short_spec(2);
    short_spec(4);
    short_spec(8);
    short_spec(16);
    short_spec(32);
    short_spec(64);

#undef short_spec

#define unsigned_short_spec(lanes) template <>\
struct vector_type_specialization <unsigned short, lanes>\
{\
    typedef unsigned short type\
        __attribute__ ((vector_size (lanes * sizeof (unsigned short))));\
    static constexpr std::size_t alignment = lanes * alignof (unsigned short);\
    static constexpr std::size_t size = lanes * sizeof (unsigned short);\
};

    unsigned_short_spec(1);
    unsigned_short_spec(2);
    unsigned_short_spec(4);
    unsigned_short_spec(8);
    unsigned_short_spec(16);
    unsigned_short_spec(32);
    unsigned_short_spec(64);

#undef unsigned_short_spec

#define int_spec(lanes) template <>\
struct vector_type_specialization <int, lanes>\
{\
    typedef int type\
        __attribute__ ((vector_size (lanes * sizeof (int))));\
    static constexpr std::size_t alignment = lanes * alignof (int);\
    static constexpr std::size_t size = lanes * sizeof (int);\
};

    int_spec(1);
    int_spec(2);
    int_spec(4);
    int_spec(8);
    int_spec(16);
    int_spec(32);
    int_spec(64);

#undef int_spec

#define unsigned_int_spec(lanes) template <>\
struct vector_type_specialization <unsigned int, lanes>\
{\
    typedef unsigned int type\
        __attribute__ ((vector_size (lanes * sizeof (unsigned int))));\
    static constexpr std::size_t alignment = lanes * alignof (unsigned int);\
    static constexpr std::size_t size = lanes * sizeof (unsigned int);\
};

    unsigned_int_spec(1);
    unsigned_int_spec(2);
    unsigned_int_spec(4);
    unsigned_int_spec(8);
    unsigned_int_spec(16);
    unsigned_int_spec(32);
    unsigned_int_spec(64);

#undef unsigned_int_spec

#define long_spec(lanes) template <>\
struct vector_type_specialization <long, lanes>\
{\
    typedef long type\
        __attribute__ ((vector_size (lanes * sizeof (long))));\
    static constexpr std::size_t alignment = lanes * alignof (long);\
    static constexpr std::size_t size = lanes * sizeof (long);\
};

    long_spec(1);
    long_spec(2);
    long_spec(4);
    long_spec(8);
    long_spec(16);
    long_spec(32);
    long_spec(64);

#undef long_spec

#define unsigned_long_spec(lanes) template <>\
struct vector_type_specialization <unsigned long, lanes>\
{\
    typedef unsigned long type\
        __attribute__ ((vector_size (lanes * sizeof (unsigned long))));\
    static constexpr std::size_t alignment = lanes * alignof (unsigned long);\
    static constexpr std::size_t size = lanes * sizeof (unsigned long);\
};

    unsigned_long_spec(1);
    unsigned_long_spec(2);
    unsigned_long_spec(4);
    unsigned_long_spec(8);
    unsigned_long_spec(16);
    unsigned_long_spec(32);
    unsigned_long_spec(64);

#undef unsigned_long_spec

#define long_long_spec(lanes) template <>\
struct vector_type_specialization <long long, lanes>\
{\
    typedef long long type\
        __attribute__ ((vector_size (lanes * sizeof (long long))));\
    static constexpr std::size_t alignment = lanes * alignof (long long);\
    static constexpr std::size_t size = lanes * sizeof (long long);\
};

    long_long_spec(1);
    long_long_spec(2);
    long_long_spec(4);
    long_long_spec(8);
    long_long_spec(16);
    long_long_spec(32);
    long_long_spec(64);

#undef long_long_spec

#define unsigned_long_long_spec(lanes) template <>\
struct vector_type_specialization <unsigned long long, lanes>\
{\
    typedef unsigned long long type\
        __attribute__ ((vector_size (lanes * sizeof (unsigned long long))));\
    static constexpr std::size_t alignment =\
        lanes * alignof (unsigned long long);\
    static constexpr std::size_t size = lanes * sizeof (unsigned long long);\
};

    unsigned_long_long_spec(1);
    unsigned_long_long_spec(2);
    unsigned_long_long_spec(4);
    unsigned_long_long_spec(8);
    unsigned_long_long_spec(16);
    unsigned_long_long_spec(32);
    unsigned_long_long_spec(64);

#undef unsigned_long_long_spec

#define float_spec(lanes) template <>\
struct vector_type_specialization <float, lanes>\
{\
    typedef float type\
        __attribute__ ((vector_size (lanes * sizeof (float))));\
    static constexpr std::size_t alignment = lanes * alignof (float);\
    static constexpr std::size_t size = lanes * sizeof (float);\
};

    float_spec(1);
    float_spec(2);
    float_spec(4);
    float_spec(8);
    float_spec(16);
    float_spec(32);
    float_spec(64);

#undef float_spec

#define double_spec(lanes) template <>\
struct vector_type_specialization <double, lanes>\
{\
    typedef double type\
        __attribute__ ((vector_size (lanes * sizeof (double))));\
    static constexpr std::size_t alignment = lanes * alignof (double);\
    static constexpr std::size_t size = lanes * sizeof (double);\
};

    double_spec(1);
    double_spec(2);
    double_spec(4);
    double_spec(8);
    double_spec(16);
    double_spec(32);
    double_spec(64);

#undef double_spec

    template <std::size_t lanes>
    struct long_double_specialization;
    
#define long_double_spec(lanes) template <>\
struct vector_type_specialization <long double, lanes>\
{\
    typedef long double type\
        __attribute__ ((vector_size (lanes * sizeof (long double))));\
    static constexpr std::size_t alignment = lanes * alignof (long double);\
    static constexpr std::size_t size = lanes * sizeof (long double);\
};

    long_double_spec(1);
    long_double_spec(2);
    long_double_spec(4);
    long_double_spec(8);
    long_double_spec(16);
    long_double_spec(32);
    long_double_spec(64);

#undef long_double_spec

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __int128_t,
#elif defined (__GNUG__)
        __int128,
#endif
        1
    >
    {
#if defined (__clang__)
        typedef __int128_t type __attribute__ ((vector_size (16)));
#elif defined (__GNUG__)
        typedef __int128 type __attribute__ ((vector_size (16)));
#endif
        static constexpr std::size_t alignment = 16;
        static constexpr std::size_t size = 16;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __int128_t,
#elif defined (__GNUG__)
        __int128,
#endif
        2
    >
    {
#if defined (__clang__)
        typedef __int128_t type __attribute__ ((vector_size (32)));
#elif defined (__GNUG__)
        typedef __int128 type __attribute__ ((vector_size (32)));
#endif
        static constexpr std::size_t alignment = 32;
        static constexpr std::size_t size = 32;
    };

template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __int128_t,
#elif defined (__GNUG__)
        __int128,
#endif
        4
    >
    {
#if defined (__clang__)
        typedef __int128_t type __attribute__ ((vector_size (64)));
#elif defined (__GNUG__)
        typedef __int128 type __attribute__ ((vector_size (64)));
#endif
        static constexpr std::size_t alignment = 64;
        static constexpr std::size_t size = 64;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __int128_t,
#elif defined (__GNUG__)
        __int128,
#endif
        8
    >
    {
#if defined (__clang__)
        typedef __int128_t type __attribute__ ((vector_size (128)));
#elif defined (__GNUG__)
        typedef __int128 type __attribute__ ((vector_size (128)));
#endif
        static constexpr std::size_t alignment = 128;
        static constexpr std::size_t size = 128;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __int128_t,
#elif defined (__GNUG__)
        __int128,
#endif
        16
    >
    {
#if defined (__clang__)
        typedef __int128_t type __attribute__ ((vector_size (256)));
#elif defined (__GNUG__)
        typedef __int128 type __attribute__ ((vector_size (256)));
#endif
        static constexpr std::size_t alignment = 256;
        static constexpr std::size_t size = 256;
    };

template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __int128_t,
#elif defined (__GNUG__)
        __int128,
#endif
        32
    >
    {
#if defined (__clang__)
        typedef __int128_t type __attribute__ ((vector_size (512)));
#elif defined (__GNUG__)
        typedef __int128 type __attribute__ ((vector_size (512)));
#endif
        static constexpr std::size_t alignment = 512;
        static constexpr std::size_t size = 512;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __int128_t,
#elif defined (__GNUG__)
        __int128,
#endif
        64
    >
    {
#if defined (__clang__)
        typedef __int128_t type __attribute__ ((vector_size (1024)));
#elif defined (__GNUG__)
        typedef __int128 type __attribute__ ((vector_size (1024)));
#endif
        static constexpr std::size_t alignment = 1024;
        static constexpr std::size_t size = 1024;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __uint128_t,
#elif defined (__GNUG__)
        unsigned __int128,
#endif
        1
    >
    {
#if defined (__clang__)
        typedef __uint128_t type __attribute__ ((vector_size (16)));
#elif defined (__GNUG__)
        typedef unsigned __int128 type __attribute__ ((vector_size (16)));
#endif
        static constexpr std::size_t alignment = 16;
        static constexpr std::size_t size = 16;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __uint128_t,
#elif defined (__GNUG__)
        unsigned __int128,
#endif
        2
    >
    {
#if defined (__clang__)
        typedef __uint128_t type __attribute__ ((vector_size (32)));
#elif defined (__GNUG__)
        typedef unsigned __int128 type __attribute__ ((vector_size (32)));
#endif
        static constexpr std::size_t alignment = 32;
        static constexpr std::size_t size = 32;
    };

template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __uint128_t,
#elif defined (__GNUG__)
        unsigned __int128,
#endif
        4
    >
    {
#if defined (__clang__)
        typedef __uint128_t type __attribute__ ((vector_size (64)));
#elif defined (__GNUG__)
        typedef unsigned __int128 type __attribute__ ((vector_size (64)));
#endif
        static constexpr std::size_t alignment = 64;
        static constexpr std::size_t size = 64;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __uint128_t,
#elif defined (__GNUG__)
        unsigned __int128,
#endif
        8
    >
    {
#if defined (__clang__)
        typedef __uint128_t type __attribute__ ((vector_size (128)));
#elif defined (__GNUG__)
        typedef unsigned __int128 type __attribute__ ((vector_size (128)));
#endif
        static constexpr std::size_t alignment = 128;
        static constexpr std::size_t size = 128;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __uint128_t,
#elif defined (__GNUG__)
        unsigned __int128,
#endif
        16
    >
    {
#if defined (__clang__)
        typedef __uint128_t type __attribute__ ((vector_size (256)));
#elif defined (__GNUG__)
        typedef unsigned __int128 type __attribute__ ((vector_size (256)));
#endif
        static constexpr std::size_t alignment = 256;
        static constexpr std::size_t size = 256;
    };

template <>
    struct vector_type_specialization <
#if defined (__clang__)
        __uint128_t,
#elif defined (__GNUG__)
        unsigned __int128,
#endif
        32
    >
    {
#if defined (__clang__)
        typedef __uint128_t type __attribute__ ((vector_size (512)));
#elif defined (__GNUG__)
        typedef unsigned __int128 type __attribute__ ((vector_size (512)));
#endif
        static constexpr std::size_t alignment = 512;
        static constexpr std::size_t size = 512;
    };

    template <>
    struct vector_type_specialization <
#if defined (__clang__)
         __uint128_t,
#elif defined (__GNUG__)
        unsigned __int128,
#endif
        64
    >
    {
#if defined (__clang__)
        typedef __uint128_t type __attribute__ ((vector_size (1024)));
#elif defined (__GNUG__)
        typedef unsigned __int128 type __attribute__ ((vector_size (1024)));
#endif
        static constexpr std::size_t alignment = 1024;
        static constexpr std::size_t size = 1024;
    };

    template <typename T, std::size_t lanes>
    using vector = vector_type_specialization <T, lanes>;
}   // namespace vext

    template <typename T, std::size_t lanes>
    class simd_type_base;

    template <typename T, std::size_t lanes>
    class simd_type_base <std::complex <T>, lanes>
        : public simd_type_base <T, lanes>
    {};

    template <typename T>
    using integral_type_switch = typename std::conditional <
        sizeof (T) == 1,
        std::int8_t,
        typename std::conditional <
            sizeof (T) == 2,
            std::int16_t,
            typename std::conditional <
                sizeof (T) == 4,
                std::int32_t,
                typename std::conditional <
                    sizeof (T) == 8,
                    std::int64_t,
                    typename std::conditional <
                        sizeof (T) == 16 || sizeof (T) == 12 || sizeof (T) == 10,
#if defined (__clang__)
                        __int128_t,
#elif defined (__GNUG__)
                        __int128,
#endif
                        void
                    >::type
                >::type
            >::type
        >::type
    >::type;

    template <typename T>
    using unsigned_integral_type_switch = typename std::conditional <
        sizeof (T) == 1,
        std::uint8_t,
        typename std::conditional <
            sizeof (T) == 2,
            std::uint16_t,
            typename std::conditional <
                sizeof (T) == 4,
                std::uint32_t,
                typename std::conditional <
                    sizeof (T) == 8,
                    std::uint64_t,
                    typename std::conditional <
                        sizeof (T) == 16 || sizeof (T) == 12 || sizeof (T) == 10,
#if defined (__clang__)
                        __uint128_t,
#elif defined (__GNUG__)
                        unsigned __int128,
#endif
                        void
                    >::type
                >::type
            >::type
        >::type
    >::type;

    template <typename T, std::size_t lanes>
    class simd_type_base
    {
#if defined (__clang__)
        static_assert (
            std::is_arithmetic <T>::value ||
                std::is_same <T, __int128_t>::value ||
                std::is_same <T, __uint128_t>::value,
            "template parameter typename T must be an arithmetic type"
        );
#elif defined (__GNUG__)
        static_assert (
            std::is_arithmetic <T>::value ||
                std::is_same <T, __int128>::value ||
                std::is_same <T, unsigned __int128>::value,
            "template parameter typename T must be an arithmetic type"
        );
#endif

        static_assert (
            lanes > 0,
            "template parameter value lanes must be nonzero"
        );

        using base_value_type = T;

    public:
        using integral_type          = integral_type_switch <T>;
        using unsigned_integral_type = unsigned_integral_type_switch <T>;
        using signed_integral_type   = integral_type_switch <T>;

        static_assert (
            !std::is_same <integral_type, void>::value &&
                !std::is_same <unsigned_integral_type, void>::value,
            "template parameter typename T is too large"
        );

        static_assert (
            sizeof (integral_type) == sizeof (unsigned_integral_type) &&
            sizeof (integral_type) == sizeof (signed_integral_type),
            "error in selecting integral types: sizes do not compare equal"
        );

        using vector_type_impl = typename vext::vector <T, lanes>::type;

        static constexpr std::size_t alignment =
            vext::vector <T, lanes>::alignment;
        static constexpr std::size_t size = vext::vector <T, lanes>::size;

        simd_type_base (void) noexcept = default;
        simd_type_base (simd_type_base const &) noexcept = default;

        static void * operator new (std::size_t sz)
        {
            if (sz != size) {
                /* let standard new handle an incorrect size request */
                return ::operator new (sz);
            } else {
                return util::aligned_allocate (sz, alignment);
            }
        }

        static void * operator new [] (std::size_t sz)
        {
            return util::aligned_allocate (sz ? sz : size, alignment);
        }

/* overloads for C++17 operator new with alignement spec */
#if __cplusplus > 201402L
        static void * operator new (std::size_t sz, std::align_val_t al)
        {
            return ::operator new (sz, al)};
        }

        static void * operator new [] (std::size_t sz, std::align_val_t al)
        {
            return ::operator new [] (sz, al);
        }
#endif

#if __cplusplus >= 201402L
        static void operator delete (void * ptr, std::size_t sz) noexcept
#else
        static void operator delete (void * ptr) noexcept
#endif
        {
#if __cplusplus < 201402L
            std::size_t sz {size};
#endif
            if (ptr == nullptr) {
                return;
            } else if (sz != size) {
                /* let ::delete handle incorrectly sized delete requests */
#if __cplusplus >= 201402L
                ::operator delete (ptr, sz);
#else
                ::operator delete (ptr);
#endif
            } else {
                util::aligned_deallocate (ptr, sz, alignment);
            }
        }

#if __cplusplus >= 201402L
        static void operator delete [] (void * ptr, std::size_t sz) noexcept
#else
        static void operator delete [] (void * ptr) noexcept
#endif
        {
#if __cplusplus < 201402L
            std::size_t sz {size};
#endif
            if (ptr == nullptr) {
                return;
            } else if (!sz) {
                /* let ::delete handle incorrectly sized delete requests */
#if __cplusplus >= 201402L
                ::operator delete (ptr, sz);
#else
                ::operator delete (ptr);
#endif
            } else {
                util::aligned_deallocate (ptr, sz, alignment);
            }
        }

/* overloads for C++17 operator delete with alignment spec */
#if __cplusplus > 201402L
        static void
            operator delete (void * ptr, std::size_t sz, std::align_val_t al)
            noexcept
        {
            ::operator delete (ptr, sz, al);
        }

        static void
            operator delete [] (void * ptr, std::size_t sz, std::align_val_t al)
            noexcept
        {
            ::operator delete [] (ptr, sz, al);
        }
#endif
    private:
        template <std::size_t ... L>
        static constexpr
        vector_type_impl unpack_impl (base_value_type const (&arr) [lanes],
                                      util::index_sequence <L...>) noexcept
        {
            return vector_type_impl {arr [L]...};
        }

        template <std::size_t ... L>
        static constexpr
        vector_type_impl
            unpack_impl (std::array <base_value_type, lanes> const & arr,
                         util::index_sequence <L...>) noexcept
        {
            return vector_type_impl {arr [L]...};
        }

    protected:
        static constexpr
        vector_type_impl unpack (base_value_type const (&arr) [lanes]) noexcept
        {
            return simd_type_base::unpack_impl (
                arr, util::make_index_sequence <lanes> {}
            );
        }

        static constexpr vector_type_impl
            unpack (std::array <base_value_type, lanes> const & arr) noexcept
        {
            return simd_type_base::unpack_impl (
                arr, util::make_index_sequence <lanes> {}
            );
        }

        template <typename ... Ts>
        static constexpr vector_type_impl extend (Ts const & ... ts) noexcept
        {
            return vector_type_impl {static_cast <base_value_type> (ts)...};
        }

    public:
        /*
         * This is a pointer proxy object to avoid undefined behavior and
         * type-punning in derived SIMD type classes. It is the returned
         * type from methds such as {c}{r}begin, and {c}{r}end.
         */
        template <typename U>
        class pointer_proxy
        {
        private:
            using element_type = U;
            using void_pointer = typename std::conditional <
                std::is_const <U>::value,
                void const *,
                void *
            >::type;
            using pointer   = typename std::add_pointer <U>::type;
            using reference = typename std::add_lvalue_reference <U>::type;

            pointer _pointer;

        public:
            using iterator_category = std::random_access_iterator_tag;

            constexpr pointer_proxy (void) noexcept
                : _pointer {nullptr}
            {}

            ~pointer_proxy (void) noexcept = default;

            constexpr pointer_proxy (pointer p) noexcept
                : _pointer {p}
            {}

            constexpr pointer_proxy (element_type & e) noexcept
                : _pointer {
                    static_cast <pointer> (static_cast <void_pointer> (&e))
                }
            {}

            constexpr pointer_proxy (pointer_proxy const &) noexcept = default;

            advanced_constexpr pointer_proxy & operator= (pointer_proxy p)
                noexcept
            {
                this->_pointer = p._pointer;
                return *this;
            }

            advanced_constexpr pointer_proxy & operator= (pointer p) noexcept
            {
                this->_pointer = p;
                return *this;
            }

            operator pointer (void) noexcept
            {
                return this->_pointer;
            }

            operator bool (void) noexcept
            {
                return static_cast <bool> (this->_pointer);
            }

            reference operator* (void) const noexcept
            {
                return *this->_pointer;
            }

            pointer operator-> (void) const noexcept
            {
                return this->_pointer;
            }

            reference operator[] (std::ptrdiff_t n) const noexcept
            {
                return this->_pointer [n];
            }

            pointer_proxy & operator++ (void) noexcept
            {
                this->_pointer += 1;
                return *this;
            }

            pointer_proxy & operator-- (void) noexcept
            {
                this->_pointer -= 1;
                return *this;
            }

            pointer_proxy operator++ (int) noexcept
            {
                auto const tmp = *this;
                this->_pointer += 1;
                return tmp;
            }

            pointer_proxy operator-- (int) noexcept
            {
                auto const tmp = *this;
                this->_pointer -= 1;
                return tmp;
            }

            pointer_proxy & operator+= (std::ptrdiff_t n) noexcept
            {
                this->_poiner += n;
                return *this;
            }

            pointer_proxy & operator-= (std::ptrdiff_t n) noexcept
            {
                this->_poiner -= n;
                return *this;
            }

            pointer_proxy operator+ (std::ptrdiff_t n) const noexcept
            {
                auto tmp = *this;
                return tmp += n;
            }

            pointer_proxy operator- (std::ptrdiff_t n) const noexcept
            {
                auto tmp = *this;
                return tmp -= n;
            }

            std::ptrdiff_t operator- (pointer_proxy p) const noexcept
            {
                return this->_pointer - p._pointer;
            }

            std::ptrdiff_t operator- (pointer p) const noexcept
            {
                return this->_pointer - p;
            }

            bool operator== (pointer_proxy p) const noexcept
            {
                return this->_pointer == p._pointer;
            }

            bool operator== (pointer p) const noexcept
            {
                return this->_pointer == p;
            }

            bool operator!= (pointer_proxy p) const noexcept
            {
                return this->_pointer != p._pointer;
            }

            bool operator!= (pointer p) const noexcept
            {
                return this->_pointer != p;
            }

            bool operator< (pointer_proxy p) const noexcept
            {
                return this->_pointer < p._pointer;
            }

            bool operator< (pointer p) const noexcept
            {
                return this->_pointer < p;
            }

            bool operator> (pointer_proxy p) const noexcept
            {
                return this->_pointer > p._pointer;
            }

            bool operator> (pointer p) const noexcept
            {
                return this->_pointer > p;
            }

            bool operator<= (pointer_proxy p) const noexcept
            {
                return this->_pointer <= p._pointer;
            }

            bool operator<= (pointer p) const noexcept
            {
                return this->_pointer <= p;
            }

            bool operator>= (pointer_proxy p) const noexcept
            {
                return this->_pointer >= p._pointer;
            }

            bool operator>= (pointer p) const noexcept
            {
                return this->_pointer >= p;
            }
        };
    };

    template <typename T, std::size_t lanes>
    class integral_simd_type;

    template <typename T, std::size_t lanes>
    class fp_simd_type;

    template <typename T, std::size_t lanes>
    class complex_simd_type;

    template <typename T, std::size_t lanes>
    class boolean_simd_type;

    struct arithmetic_tag;
    struct complex_tag;
    struct boolean_tag;

    template <typename T, std::size_t lanes, typename tag = arithmetic_tag>
    using simd_type = typename std::conditional <
        (std::is_integral <T>::value &&
            std::is_same <tag, arithmetic_tag>::value) ||
#if defined (__clang__)
        (std::is_same <T, __int128_t>::value &&
            std::is_same <tag, arithmetic_tag>::value)||
        (std::is_same <T, __uint128_t>::value &&
            std::is_same <tag, arithmetic_tag>::value),
#elif defined (__GNUG__)
        (std::is_same <T, __int128>::value &&
            std::is_same <tag, arithmetic_tag>::value)||
        (std::is_same <T, unsigned __int128>::value &&
            std::is_same <tag, arithmetic_tag>::value),
#endif
        integral_simd_type <T, lanes>,
        typename std::conditional <
            (std::is_integral <T>::value &&
                std::is_same <tag, boolean_tag>::value) ||
#if defined (__clang__)
            (std::is_same <T, __int128_t>::value &&
                std::is_same <tag, boolean_tag>::value) ||
            (std::is_same <T, __uint128_t>::value &&
                std::is_same <tag, boolean_tag>::value),
#elif defined (__GNUG__)
            (std::is_same <T, __int128>::value &&
                std::is_same <tag, boolean_tag>::value) ||
            (std::is_same <T, unsigned __int128>::value &&
                std::is_same <tag, boolean_tag>::value),
#endif
            boolean_simd_type <T, lanes>,
            typename std::conditional <
                std::is_floating_point <T>::value &&
                    std::is_same <tag, arithmetic_tag>::value,
                fp_simd_type <T, lanes>,
                typename std::conditional <
                    std::is_floating_point <T>::value &&
                        std::is_same <tag, complex_tag>::value,
                    complex_simd_type <T, lanes>,
                    void
                >::type
            >::type
        >::type
    >::type;

    template <typename>
    struct is_simd_type : std::false_type {};

    template <typename T, std::size_t lanes>
    struct is_simd_type <integral_simd_type <T, lanes>>
        : public std::true_type {};

    template <typename T, std::size_t lanes>
    struct is_simd_type <fp_simd_type <T, lanes>>
        : public std::true_type {};

    template <typename T, std::size_t lanes>
    struct is_simd_type <complex_simd_type <T, lanes>>
        : public std::true_type {};

    template <typename T, std::size_t lanes>
    struct is_simd_type <boolean_simd_type <T, lanes>>
        : public std::true_type {};

    template <typename T, std::size_t l, typename tag>
    struct simd_traits_base
    {
        using base                   = simd_type_base <T, l>;
        using vector_type            = typename base::vector_type_impl;
        using value_type             = T;
        using integral_type          = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type   = typename base::signed_integral_type;
        using pointer =
            typename base::template pointer_proxy <value_type>;
        using const_pointer =
            typename base::template pointer_proxy <value_type const>;
        using reference              = value_type &;
        using const_reference        = value_type const &;
        using iterator               = pointer;
        using const_iterator         = const_pointer;
        using reverse_iterator       = std::reverse_iterator <pointer>;
        using const_reverse_iterator = std::reverse_iterator <const_pointer>;
        using category_tag           = tag;

        static constexpr std::size_t alignment = base::alignment;
        static constexpr std::size_t size = base::size;
        static constexpr std::size_t lanes = l;
    };

    template <typename>
    struct simd_traits;

    template <typename T, std::size_t lanes>
    struct simd_traits <integral_simd_type <T, lanes>>
        : public simd_traits_base <T, lanes, arithmetic_tag>
    {};

    template <typename T, std::size_t lanes>
    struct simd_traits <fp_simd_type <T, lanes>>
        : public simd_traits_base <T, lanes, arithmetic_tag>
    {};

    template <typename T, std::size_t lanes>
    struct simd_traits <boolean_simd_type <T, lanes>>
        : public simd_traits_base <T, lanes, boolean_tag>
    {};

    template <typename T, std::size_t l>
    struct simd_traits <complex_simd_type <T, l>>
    {
        using base                   = simd_type_base <T, l>;
        using vector_type            = typename base::vector_type_impl;
        using value_type             = std::complex <T>;
        using integral_type          = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type   = typename base::signed_integral_type;
        using pointer =
            typename base::template pointer_proxy <value_type>;
        using const_pointer =
            typename base::template pointer_proxy <value_type const>;
        using reference              = value_type &;
        using const_reference        = value_type const &;
        using iterator =
            typename complex_simd_type <T, l>::iterator;
        using const_iterator =
            typename complex_simd_type <T, l>::const_iterator;
        using reverse_iterator =
            typename complex_simd_type <T, l>::reverse_iterator;
        using const_reverse_iterator =
            typename complex_simd_type <T, l>::const_reverse_iterator;
        using category_tag    = complex_tag;

        static constexpr std::size_t alignment = base::alignment;
        static constexpr std::size_t size = base::size;
        static constexpr std::size_t lanes = l;
    };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
    template <typename T, std::size_t l>
    class alignas (simd_type_base <T, l>::alignment)
        integral_simd_type : public simd_type_base <T, l>
    {
    private:
        using base = simd_type_base <T, l>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
        union alignas (base::alignment)
        {
            typename base::vector_type_impl _vec;
            T _arr [l];
        };
#pragma GCC diagnostic pop

    public:
        static_assert (
            std::is_integral <T>::value ||
#if defined (__clang__)
            std::is_same <T, __int128_t>::value ||
            std::is_same <T, __uint128_t>::value,
#elif defined (__GNUG__)
            std::is_same <T, __int128>::value ||
            std::is_same <T, unsigned __int128>::value,
#endif
            "template parameter typename T must be an integral type"
        );

        using vector_type            = typename base::vector_type_impl;
        using value_type             = T;
        using integral_type          = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type   = typename base::signed_integral_type;
        using pointer                =
            typename base::template pointer_proxy <value_type>;
        using const_pointer          =
            typename base::template pointer_proxy <value_type const>;
        using reference              = value_type &;
        using const_reference        = value_type const &;
        using iterator               = pointer;
        using const_iterator         = const_pointer;
        using reverse_iterator       = std::reverse_iterator <pointer>;
        using const_reverse_iterator = std::reverse_iterator <const_pointer>;
        using category_tag           = arithmetic_tag;
        static constexpr std::size_t lanes = l;

        static_assert (
            sizeof (value_type) == sizeof (integral_type),
            "error in selecting integral type: size of value type does not"
            " compare equal"
        );

        template <typename U, std::size_t L, typename tag>
        using rebind = simd_type <U, L, tag>;

    private:
        template <std::size_t ... L>
        static constexpr integral_simd_type
            increment_vector_impl (integral_type from,
                                   util::index_sequence <L...>) noexcept
        {
            return integral_simd_type {
                (static_cast <integral_type> (L) + from)...
            };
        }

    public:
        static constexpr
        integral_simd_type increment_vector (value_type from = 0) noexcept
        {
            return increment_vector_impl (
                from, util::make_index_sequence <lanes> {}
            );
        }

        static integral_simd_type load (value_type const * addr) noexcept
        {
            integral_simd_type result {};
            std::memcpy (&result.data (), addr, lanes * sizeof (value_type));
            return result;
        }

        constexpr integral_simd_type (void) noexcept
            : _vec {base::extend (value_type {})}
        {}

        explicit constexpr integral_simd_type (vector_type const & vec) noexcept
            : _vec {vec}
        {}

        template <typename ... value_types>
        explicit constexpr integral_simd_type (value_types const & ... vals) noexcept
            : _vec {base::extend (vals...)}
        {
            static_assert (
                sizeof... (value_types) == 1 ||
                sizeof... (value_types) == lanes,
                "vector constructor must be provided a number of values equal"
                " to one or equal to the number of vector lanes"
            );
        }

        constexpr
        integral_simd_type (integral_simd_type const & sv) noexcept
            : base {}
            , _vec {sv._vec}
        {}

        explicit constexpr
        integral_simd_type (value_type const (&arr) [lanes]) noexcept
            : _vec {base::unpack (arr)}
        {}

        explicit constexpr
        integral_simd_type (std::array <value_type, lanes> const & arr) noexcept
            : _vec {base::unpack (arr)}
        {}

        advanced_constexpr
        integral_simd_type & operator= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec = sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec = base::extend (val);
            return *this;
        }

    private:
        template <std::size_t ... L>
        advanced_constexpr void
            fill_array (std::array <value_type, lanes> & arr,
                        util::index_sequence <L...>) const noexcept
        {
            value_type _unused [] = {
                (std::get <L> (arr) = this->template get <L> ())...
            };
            (void) _unused;
        }

    public:
        explicit advanced_constexpr
            operator std::array <value_type, lanes> (void) const noexcept
        {
            std::array <value_type, lanes> result {};
            this->fill_array (result, util::make_index_sequence <lanes> {});
            return result;
        }

        template <typename SimdT>
        constexpr SimdT as (void) const noexcept
        {
            static_assert (
                is_simd_type <SimdT>::value,
                "cannot perform cast to non-simd type"
            );

            using traits = simd_traits <SimdT>;
            using rebind_type = rebind <
                typename traits::value_type,
                traits::lanes,
                typename traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;

            static_assert (
                sizeof (vector_type) == sizeof (rebind_vector_type) ||
                lanes == traits::lanes,
                "cannot perform up-sizing or down-sizing vector cast unless"
                " the result type and source type have an equal number of lanes"
            );

            return rebind_type {
                static_cast <rebind_vector_type> (this->_vec)
            };
        }

        advanced_constexpr void fill (value_type const & val) noexcept
        {
            this->_vec = base::extend (val);
        }

        advanced_constexpr void swap (integral_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        advanced_constexpr vector_type & data (void) & noexcept
        {
            return this->_vec;
        }

        constexpr vector_type const & data (void) const & noexcept
        {
            return this->_vec;
        }

        template <std::size_t n>
        advanced_constexpr reference get (void) & noexcept
        {
            static_assert (
                n < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_arr [n];
        }

        template <std::size_t n>
        constexpr const_reference get (void) const & noexcept
        {
            static_assert (
                n < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_arr [n];
        }

        advanced_constexpr reference operator[] (std::size_t n) & noexcept
        {
            return this->_arr [n];
        }

        constexpr const_reference operator[] (std::size_t n) const & noexcept
        {
            return this->_arr [n];
        }

        advanced_constexpr reference at (std::size_t n) &
        {
            return n < lanes ?
                this->_arr [n] :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        constexpr const_reference at (std::size_t n) const &
        {
            return n < lanes ?
                this->_arr [n] :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        advanced_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->_arr [0]};
        }

        advanced_constexpr iterator end (void) & noexcept
        {
            return iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->_arr [0]};
        }

        constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->_arr [0]};
        }

        constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {*(&this->_arr [0] + lanes)};
        }

        advanced_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        advanced_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->_arr [0]};
        }

        constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->_arr [0]};
        }

        constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->_arr [0]};
        }

        constexpr integral_simd_type operator+ (void) const noexcept
        {
            return integral_simd_type {+this->_vec};
        }

        constexpr integral_simd_type operator- (void) const noexcept
        {
            return integral_simd_type {-this->_vec};
        }

        advanced_constexpr integral_simd_type & operator++ (void) noexcept
        {
            this->operator+ (1);
            return *this;
        }

        advanced_constexpr integral_simd_type & operator-- (void) noexcept
        {
            this->operator- (1);
            return *this;
        }

        advanced_constexpr integral_simd_type operator++ (int) noexcept
        {
            auto const tmp = *this;
            this->operator+ (1);
            return tmp;
        }

        advanced_constexpr integral_simd_type operator-- (int) noexcept
        {
            auto const tmp = *this;
            this->operator- (1);
            return tmp;
        }

        constexpr
        integral_simd_type operator+ (integral_simd_type const & sv) const
            noexcept
        {
            return integral_simd_type {this->_vec + sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator+ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec + base::extend (val)};
        }

        constexpr integral_simd_type operator- (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec - sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator- (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec - base::extend (val)};
        }

        constexpr integral_simd_type operator* (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec * sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator* (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec * base::extend (val)};
        }

        constexpr integral_simd_type operator/ (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec / sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator/ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec / base::extend (val)};
        }

        constexpr integral_simd_type operator% (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec % sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator% (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec % base::extend (val)};
        }

        constexpr integral_simd_type operator~ (void) const noexcept
        {
            return integral_simd_type {~this->_vec};
        }

        constexpr integral_simd_type operator& (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec & sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator& (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec & base::extend (val)};
        }

        constexpr integral_simd_type operator| (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec | sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator| (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec | base::extend (val)};
        }

        constexpr integral_simd_type operator^ (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec ^ sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator^ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec ^ base::extend (val)};
        }

        constexpr integral_simd_type operator! (void) const noexcept
        {
            return integral_simd_type {!this->_vec};
        }

        constexpr integral_simd_type operator&& (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec && sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator&& (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec && base::extend (val)};
        }

        constexpr integral_simd_type operator|| (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec || sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator|| (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return integral_simd_type {this->_vec || base::extend (val)};
        }

        constexpr integral_simd_type operator<< (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec << sv._vec};
        }

        constexpr integral_simd_type operator<< (value_type shl_val)
            const noexcept
        {
            return *this << integral_simd_type {shl_val};
        }

        constexpr integral_simd_type operator>> (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec >> sv._vec};
        }

        constexpr integral_simd_type operator>> (value_type shl_val)
            const noexcept
        {
            return *this >> integral_simd_type {shl_val};
        }

        advanced_constexpr
        integral_simd_type & operator+= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec += sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator+= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec += base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator-= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec -= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator-= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec -= base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator*= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec *= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator*= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec *= base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator/= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec /= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator/= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec /= base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator%= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec %= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator%= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec %= base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator&= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec &= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator&= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec &= base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator|= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec |= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator|= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec |= base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator^= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec ^= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator^= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec ^= base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator<<= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec <<= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator<<= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec <<= base::extend (val);
            return *this;
        }

        advanced_constexpr
        integral_simd_type & operator>>= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec >>= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr integral_simd_type & operator>>= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec >>= base::extend (val);
            return *this;
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator== (integral_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec == sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes> operator== (U val)
            const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (integral_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec != sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this != integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator> (integral_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec > sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator> (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this > integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator< (integral_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec < sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator< (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this < integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (integral_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec >= sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this >= integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (integral_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec <= sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this <= integral_simd_type {val};
        }
    };
#pragma GCC diagnostic pop

    template <typename T, std::size_t lanes>
    class fp_simd_type;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
    template <typename T, std::size_t l>
    class alignas (simd_type_base <T, l>::alignment)
        fp_simd_type : public simd_type_base <T, l>
    {
    private:
        using base = simd_type_base <T, l>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
        union alignas (base::alignment)
        {
            typename base::vector_type_impl _vec;
            T _arr [l];
        };
#pragma GCC diagnostic pop

    public:
        static_assert (
            std::is_floating_point <T>::value,
            "template parameter typename T must be a floating point type"
        );

        using vector_type     = typename base::vector_type_impl;
        using value_type      = T;
        using integral_type   = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type = typename base::signed_integral_type;
        using pointer         = typename base::template pointer_proxy <value_type>;
        using const_pointer   = typename base::template pointer_proxy <value_type const>;
        using reference       = value_type &;
        using const_reference = value_type const &;
        using iterator        = pointer;
        using const_iterator  = const_pointer;
        using reverse_iterator       = std::reverse_iterator <pointer>;
        using const_reverse_iterator = std::reverse_iterator <const_pointer>;
        using category_tag    = arithmetic_tag;
        static constexpr std::size_t lanes = l;

        template <typename U, std::size_t L, typename tag>
        using rebind = simd_type <U, L, tag>;

        static fp_simd_type load (value_type const * addr) noexcept
        {
            fp_simd_type result {};
            std::memcpy (&result.data (), addr, lanes * sizeof (value_type));
            return result;
        }

        constexpr fp_simd_type (void) noexcept
            : _vec {base::extend (value_type {})}
        {}

        explicit constexpr fp_simd_type (vector_type const & vec) noexcept
            : _vec {vec}
        {}

        template <typename ... value_types>
        explicit constexpr fp_simd_type (value_types const & ... vals) noexcept
            : _vec {base::extend (vals...)}
        {
            static_assert (
                sizeof... (value_types) == 1 || sizeof... (value_types) == lanes,
                "vector constructor must be provided a number of values equal"
                " to one or equal to the number of vector lanes"
            );
        }

        constexpr fp_simd_type (fp_simd_type const & sv) noexcept
            : base {}
            , _vec {sv._vec}
        {}

        explicit constexpr fp_simd_type (value_type const (&arr) [lanes])
            noexcept
            : _vec {base::unpack (arr)}
        {}

        explicit constexpr
        fp_simd_type (std::array <value_type, lanes> const & arr) noexcept
            : _vec {base::unpack (arr)}
        {}

        advanced_constexpr fp_simd_type & operator= (fp_simd_type const & sv) & noexcept
        {
            this->_vec = sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec = base::extend (val);
            return *this;
        }

    private:
        template <std::size_t ... L>
        advanced_constexpr void fill_array (std::array <value_type, lanes> & arr,
                                   util::index_sequence <L...>) const noexcept
        {
            value_type _unused [] = {
                (std::get <L> (arr) = this->template get <L> ())...
            };
            (void) _unused;
        }

    public:
        explicit advanced_constexpr operator std::array <value_type, lanes> (void) const
            noexcept
        {
            std::array <value_type, lanes> result {};
            this->fill_array (result, util::make_index_sequence <lanes> {});
            return result;
        }

        template <typename SimdT>
        constexpr SimdT as (void) const noexcept
        {
            static_assert (
                is_simd_type <SimdT>::value,
                "cannot perform cast to non-simd type"
            );

            using traits = simd_traits <SimdT>;
            using rebind_type = rebind <
                typename traits::value_type,
                traits::lanes,
                typename traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;

            static_assert (
                sizeof (vector_type) == sizeof (rebind_vector_type) ||
                lanes == traits::lanes,
                "cannot perform up-sizing or down-sizing vector cast unless"
                " the result type and source type have an equal number of lanes"
            );

            return rebind_type {
                static_cast <rebind_vector_type> (this->_vec)
            };
        }

        advanced_constexpr void fill (value_type const & val) noexcept
        {
            this->_vec = base::extend (val);
        }

        advanced_constexpr void swap (fp_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        advanced_constexpr vector_type & data (void) & noexcept
        {
            return this->_vec;
        }

        constexpr vector_type const & data (void) const & noexcept
        {
            return this->_vec;
        }

        template <std::size_t n>
        advanced_constexpr reference get (void) & noexcept
        {
            static_assert (
                n < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_arr [n];
        }

        template <std::size_t n>
        constexpr const_reference get (void) const & noexcept
        {
            static_assert (
                n < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_arr [n];
        }

        advanced_constexpr reference operator[] (std::size_t n) & noexcept
        {
            return this->_arr [n];
        }

        constexpr const_reference operator[] (std::size_t n) const & noexcept
        {
            return this->_arr [n];
        }

        advanced_constexpr reference at (std::size_t n) &
        {
            return n < lanes ?
                this->_arr [n] :
                throw std::out_of_range {"access attempt to out-of-bounds vector lane"};
        }

        constexpr const_reference at (std::size_t n) const &
        {
            return n < lanes ?
                this->_arr [n] :
                throw std::out_of_range {"access attempt to out-of-bounds vector lane"};
        }

        advanced_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->_arr [0]};
        }

        advanced_constexpr iterator end (void) & noexcept
        {
            return iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->_arr [0]};
        }

        constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->_arr [0]};
        }

        constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {*(&this->_arr [0] + lanes)};
        }

        advanced_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        advanced_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->_arr [0]};
        }

        constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->_arr [0]};
        }

        constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->_arr [0]};
        }

        constexpr fp_simd_type operator+ (void) const noexcept
        {
            return fp_simd_type {+this->_vec};
        }

        constexpr fp_simd_type operator- (void) const noexcept
        {
            return fp_simd_type {-this->_vec};
        }

        constexpr fp_simd_type operator+ (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec + sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator+ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec + base::extend (val)
            };
        }

        constexpr fp_simd_type operator- (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec - sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator- (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec - base::extend (val)
            };
        }

        constexpr fp_simd_type operator* (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec * sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator* (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec * base::extend (val)
            };
        }

        constexpr fp_simd_type operator/ (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec / sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator/ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec / base::extend (val)
            };
        }

        constexpr fp_simd_type operator! (void) const noexcept
        {
            return fp_simd_type {!this->_vec};
        }

        constexpr fp_simd_type operator&& (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec && sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator&& (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec && base::extend (val)
            };
        }

        constexpr fp_simd_type operator|| (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec || sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator|| (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec || base::extend (val)
            };
        }

        advanced_constexpr fp_simd_type & operator+= (fp_simd_type const & sv) & noexcept
        {
            this->_vec += sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator+= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec += base::extend (val);
            return *this;
        }

        advanced_constexpr fp_simd_type & operator-= (fp_simd_type const & sv) & noexcept
        {
            this->_vec -= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator-= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec -= base::extend (val);
            return *this;
        }

        advanced_constexpr fp_simd_type & operator*= (fp_simd_type const & sv) & noexcept
        {
            this->_vec *= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator*= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec *= base::extend (val);
            return *this;
        }

        advanced_constexpr fp_simd_type & operator/= (fp_simd_type const & sv) &
            noexcept
        {
            this->_vec /= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator/= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec /= base::extend (val);
            return *this;
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator== (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec == sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes> operator== (U val)
            const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec != sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this != fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator> (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec > sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator> (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this > fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator< (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec < sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator< (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this < fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec >= sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this >= fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec <= sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this <= fp_simd_type {val};
        }
    };
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
    template <std::size_t l>
    class alignas (simd_type_base <long double, l>::alignment)
        fp_simd_type <long double, l> : public simd_type_base <long double, l>
    {
    private:
        using T = long double;
        using base = simd_type_base <T, l>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
        union alignas (base::alignment)
        {
            typename base::vector_type_impl _vec;
            T _arr [l];
        };
#pragma GCC diagnostic pop

    public:
        using vector_type     = typename base::vector_type_impl;
        using value_type      = T;
        using integral_type   = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type = typename base::signed_integral_type;
        using pointer         = typename base::template pointer_proxy <value_type>;
        using const_pointer   = typename base::template pointer_proxy <value_type const>;
        using reference       = value_type &;
        using const_reference = value_type const &;
        using iterator        = pointer;
        using const_iterator  = const_pointer;
        using reverse_iterator       = std::reverse_iterator <pointer>;
        using const_reverse_iterator = std::reverse_iterator <const_pointer>;
        using category_tag    = arithmetic_tag;
        static constexpr std::size_t lanes = l;

        template <typename U, std::size_t L, typename tag>
        using rebind = simd_type <U, L, tag>;

        static fp_simd_type load (value_type const * addr) noexcept
        {
            fp_simd_type result {};
            std::memcpy (&result.data (), addr, lanes * sizeof (value_type));
            return result;
        }

        constexpr fp_simd_type (void) noexcept
            : _vec {base::extend (value_type {})}
        {}

        explicit constexpr fp_simd_type (vector_type const & vec) noexcept
            : _vec {vec}
        {}

        template <typename ... value_types>
        explicit constexpr fp_simd_type (value_types const & ... vals) noexcept
            : _vec {base::extend (vals...)}
        {
            static_assert (
                sizeof... (value_types) == 1 || sizeof... (value_types) == lanes,
                "vector constructor must be provided a number of values equal"
                " to one or equal to the number of vector lanes"
            );
        }

        constexpr fp_simd_type (fp_simd_type const & sv) noexcept
            : base {}
            , _vec {sv._vec}
        {}

        explicit constexpr
        fp_simd_type (value_type const (&arr) [lanes]) noexcept
            : _vec {base::unpack (arr)}
        {}

        explicit constexpr
        fp_simd_type (std::array <value_type, lanes> const & arr) noexcept
            : _vec {base::unpack (arr)}
        {}

        advanced_constexpr fp_simd_type & operator= (fp_simd_type const & sv) & noexcept
        {
            this->_vec = sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec = base::extend (val);
            return *this;
        }

    private:
        template <std::size_t ... L>
        advanced_constexpr void fill_array (std::array <value_type, lanes> & arr,
                                   util::index_sequence <L...>) const noexcept
        {
            value_type _unused [] = {
                (std::get <L> (arr) = this->template get <L> ())...
            };
            (void) _unused;
        }

    public:
        explicit advanced_constexpr operator std::array <value_type, lanes> (void) const
            noexcept
        {
            std::array <value_type, lanes> result {};
            this->fill_array (result, util::make_index_sequence <lanes> {});
            return result;
        }

        template <typename SimdT>
        constexpr SimdT as (void) const noexcept
        {
            static_assert (
                is_simd_type <SimdT>::value,
                "cannot perform cast to non-simd type"
            );

            using traits = simd_traits <SimdT>;
            using rebind_type = rebind <
                typename traits::value_type,
                traits::lanes,
                typename traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;

            static_assert (
                sizeof (vector_type) == sizeof (rebind_vector_type) ||
                lanes == traits::lanes,
                "cannot perform up-sizing or down-sizing vector cast unless"
                " the result type and source type have an equal number of lanes"
            );

            return rebind_type {
                static_cast <rebind_vector_type> (this->_vec)
            };
        }

        advanced_constexpr void fill (value_type const & val) noexcept
        {
            this->_vec = base::extend (val);
        }

        advanced_constexpr void swap (fp_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        advanced_constexpr vector_type & data (void) & noexcept
        {
            return this->_vec;
        }

        constexpr vector_type const & data (void) const & noexcept
        {
            return this->_vec;
        }

        template <std::size_t n>
        advanced_constexpr reference get (void) & noexcept
        {
            static_assert (
                n < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_arr [n];
        }

        template <std::size_t n>
        constexpr const_reference get (void) const & noexcept
        {
            static_assert (
                n < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_arr [n];
        }

        advanced_constexpr reference operator[] (std::size_t n) & noexcept
        {
            return this->_arr [n];
        }

        constexpr const_reference operator[] (std::size_t n) const & noexcept
        {
            return this->_arr [n];
        }

        advanced_constexpr reference at (std::size_t n) &
        {
            return n < lanes ?
                this->_arr [n] :
                throw std::out_of_range {"access attempt to out-of-bounds vector lane"};
        }

        constexpr const_reference at (std::size_t n) const &
        {
            return n < lanes ?
                this->_arr [n] :
                throw std::out_of_range {"access attempt to out-of-bounds vector lane"};
        }

        advanced_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->_arr [0]};
        }

        advanced_constexpr iterator end (void) & noexcept
        {
            return iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->_arr [0]};
        }

        constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->_arr [0]};
        }

        constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {*(&this->_arr [0] + lanes)};
        }

        advanced_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        advanced_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->_arr [0]};
        }

        constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->_arr [0]};
        }

        constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->_arr [0]};
        }

        constexpr fp_simd_type operator+ (void) const noexcept
        {
            return fp_simd_type {+this->_vec};
        }

        constexpr fp_simd_type operator- (void) const noexcept
        {
            return fp_simd_type {-this->_vec};
        }

        constexpr fp_simd_type operator+ (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec + sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator+ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec + base::extend (val)
            };
        }

        constexpr fp_simd_type operator- (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec - sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator- (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec - base::extend (val)
            };
        }

        constexpr fp_simd_type operator* (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec * sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator* (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec * base::extend (val)
            };
        }

        constexpr fp_simd_type operator/ (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec / sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator/ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec / base::extend (val)
            };
        }

        constexpr fp_simd_type operator! (void) const noexcept
        {
            return fp_simd_type {!this->_vec};
        }

        constexpr fp_simd_type operator&& (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec && sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator&& (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec && base::extend (val)
            };
        }

        constexpr fp_simd_type operator|| (fp_simd_type const & sv) const
            noexcept
        {
            return fp_simd_type {this->_vec || sv._vec};
        }

        template <typename U>
        constexpr fp_simd_type operator|| (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return fp_simd_type {
                this->_vec || base::extend (val)
            };
        }

        advanced_constexpr fp_simd_type & operator+= (fp_simd_type const & sv) & noexcept
        {
            this->_vec += sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator+= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec += base::extend (val);
            return *this;
        }

        advanced_constexpr fp_simd_type & operator-= (fp_simd_type const & sv) & noexcept
        {
            this->_vec -= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator-= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec -= base::extend (val);
            return *this;
        }

        advanced_constexpr fp_simd_type & operator*= (fp_simd_type const & sv) & noexcept
        {
            this->_vec *= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator*= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec *= base::extend (val);
            return *this;
        }

        advanced_constexpr fp_simd_type & operator/= (fp_simd_type const & sv) &
            noexcept
        {
            this->_vec /= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr fp_simd_type & operator/= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec /= base::extend (val);
            return *this;
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator== (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec == sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes> operator== (U val)
            const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec != sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this != fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator> (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec > sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator> (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this > fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator< (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec < sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator< (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this < fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec >= sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this >= fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (fp_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (this->_vec <= sv._vec)
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this <= fp_simd_type {val};
        }
    };
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
    template <typename T, std::size_t l>
    class alignas (simd_type_base <std::complex <T>, l>::alignment)
        complex_simd_type : public simd_type_base <std::complex <T>, l>
    {
    private:
        using base = simd_type_base <std::complex <T>, l>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
        union alignas (base::alignment)
        {
            typename base::vector_type_impl _realvec;
            T _realarr [l];
        };
#pragma GCC diagnostic pop

        union alignas (base::alignment)
        {
            typename base::vector_type_impl _imagvec;
            T _imagarr [l];
        };

        template <typename U>
        class pointer_proxy
        {
        public:
            using element_type = U;
            using void_pointer = typename std::conditional <
                std::is_const <U>::value,
                void const *,
                void *
            >::type;
            using pointer   = typename std::add_pointer <U>::type;
            using reference = typename std::add_lvalue_reference <U>::type;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::random_access_iterator_tag;

        private:
            pointer _realpointer;
            pointer _imagpointer;

        public:
            pointer_proxy (void) noexcept
                : _realpointer {nullptr}
                , _imagpointer {nullptr}
            {}

            ~pointer_proxy (void) noexcept = default;

            pointer_proxy (pointer real, pointer imag) noexcept
                : _realpointer {real}
                , _imagpointer {imag}
            {}

            pointer_proxy (element_type & real, element_type & imag) noexcept
                : _realpointer {
                    static_cast <pointer> (static_cast <void_pointer> (&real))
                }
                , _imagpointer {
                    static_cast <pointer> (static_cast <void_pointer> (&imag))
                }
            {}

            pointer_proxy (pointer_proxy const &) noexcept = default;
            pointer_proxy & operator= (pointer_proxy const &) noexcept
                = default;

            pointer_proxy & operator++ (void) noexcept
            {
                this->_realpointer += 1;
                this->_imagpointer += 1;
                return *this;
            }

            pointer_proxy & operator-- (void) noexcept
            {
                this->_realpointer -= 1;
                this->_imagpointer -= 1;
                return *this;
            }

            pointer_proxy & operator++ (int) noexcept
            {
                auto const tmp = *this;
                this->_realpointer += 1;
                this->_imagpointer += 1;
                return tmp;
            }

            pointer_proxy & operator-- (int) noexcept
            {
                auto const tmp = *this;
                this->_realpointer -= 1;
                this->_imagpointer -= 1;
                return tmp;
            }

            pointer_proxy & operator+= (difference_type n) noexcept
            {
                this->_realpointer += n;
                this->_imagpointer += n;
                return *this;
            }

            pointer_proxy & operator-= (difference_type n) noexcept
            {
                this->_realpointer -= n;
                this->_imagpointer -= n;
                return *this;
            }

            pointer_proxy operator+ (difference_type n) const noexcept
            {
                auto tmp = *this;
                return tmp += n;
            }

            pointer_proxy operator- (difference_type n) const noexcept
            {
                auto tmp = *this;
                return tmp -= n;
            }

            bool operator== (pointer_proxy const & other) const noexcept
            {
                return (this->_realpointer == other._realpointer) &&
                       (this->_imagpointer == other._imagpointer);
            }

            bool operator< (pointer_proxy const & other) const noexcept
            {
                return (this->_realpointer < other._realpointer) &&
                       (this->_imagpointer < other._imagpointer);
            }

            bool operator> (pointer_proxy const & other) const noexcept
            {
                return (this->_realpointer > other._realpointer) &&
                       (this->_imagpointer > other._imagpointer);
            }

            bool operator<= (pointer_proxy const & other) const noexcept
            {
                return (this->_realpointer <= other._realpointer) &&
                       (this->_imagpointer <= other._imagpointer);
            }

            bool operator>= (pointer_proxy const & other) const noexcept
            {
                return (this->_realpointer >= other._realpointer) &&
                       (this->_imagpointer >= other._imagpointer);
            }

            U operator* (void) const noexcept
            {
                return U {*this->_realpointer, *this->_imagpointer};
            }

            U operator[] (difference_type n) const noexcept
            {
                return *(*this + n);
            }
        };

    public:
        static_assert (
            std::is_floating_point <T>::value,
            "template parameter typename T must be a floating point type"
        );

        using vector_type     = typename base::vector_type_impl;
        using value_type      = std::complex <T>;
        using integral_type   = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type = typename base::signed_integral_type;
        using pointer         = pointer_proxy <value_type>;
        using const_pointer   = pointer_proxy <value_type const>;
        using iterator        = pointer;
        using const_iterator  = const_pointer;
        using reverse_iterator       = std::reverse_iterator <pointer>;
        using const_reverse_iterator = std::reverse_iterator <const_pointer>;
        using category_tag           = complex_tag;
        static constexpr std::size_t lanes = l;
        static constexpr std::size_t alignment = base::alignment;
        static constexpr std::size_t size = 2 * sizeof (vector_type);

        template <typename U, std::size_t L, typename tag>
        using rebind = simd_type <U, L, tag>;

        static void * operator new (std::size_t sz)
        {
            if (sz != size) {
                /* let standard new handle an incorrect size request */
                return ::operator new (sz);
            } else {
                return util::aligned_allocate (sz, alignment);
            }
        }

        static void * operator new [] (std::size_t sz)
        {
            return util::aligned_allocate (sz ? sz : size, alignment);
        }

/* overloads for C++17 operator new with alignement spec */
#if __cplusplus > 201402L
        static void * operator new (std::size_t sz, std::align_val_t al)
        {
            return ::operator new (sz, al)};
        }

        static void * operator new [] (std::size_t sz, std::align_val_t al)
        {
            return ::operator new [] (sz, al);
        }
#endif

#if __cplusplus >= 201402L
        static void operator delete (void * ptr, std::size_t sz) noexcept
#else
        static void operator delete (void * ptr) noexcept
#endif
        {
#if __cplusplus < 201402L
            std::size_t sz {size};
#endif
            if (ptr == nullptr) {
                return;
            } else if (sz != size) {
                /* let ::delete handle incorrectly sized delete requests */
#if __cplusplus >= 201402L
                ::operator delete (ptr, sz);
#else
                ::operator delete (ptr);
#endif
            } else {
                util::aligned_deallocate (ptr, sz, alignment);
            }
        }

#if __cplusplus >= 201402L
        static void operator delete [] (void * ptr, std::size_t sz) noexcept
#else
        static void operator delete [] (void * ptr) noexcept
#endif
        {
#if __cplusplus < 201402L
            std::size_t sz {size};
#endif
            if (ptr == nullptr) {
                return;
            } else if (!sz) {
                /* let ::delete handle incorrectly sized delete requests */
#if __cplusplus >= 201402L
                ::operator delete (ptr, sz);
#else
                ::operator delete (ptr);
#endif
            } else {
                util::aligned_deallocate (ptr, sz, alignment);
            }
        }

/* overloads for C++17 operator delete with alignment spec */
#if __cplusplus > 201402L
        static void
            operator delete (void * ptr, std::size_t sz, std::align_val_t al)
            noexcept
        {
            ::operator delete (ptr, sz, al);
        }

        static void
            operator delete [] (void * ptr, std::size_t sz, std::align_val_t al)
            noexcept
        {
            ::operator delete [] (ptr, sz, al);
        }
#endif

    private:
        template <std::size_t ... L>
        static constexpr vector_type
            unpack_real_impl (value_type const (& arr) [lanes], util::index_sequence <L...>) noexcept
        {
            return vector_type {arr [L].real ()...};
        }

        template <std::size_t ... L>
        static constexpr vector_type
            unpack_real_impl (std::array <value_type, lanes> const & arr, util::index_sequence <L...>) noexcept
        {
            return vector_type {std::get <L> (arr).real ()...};
        }

        template <std::size_t ... L>
        static constexpr vector_type
            unpack_imag_impl (value_type const (& arr) [lanes], util::index_sequence <L...>) noexcept
        {
            return vector_type {arr [L].imag ()...};
        }

        template <std::size_t ... L>
        static constexpr vector_type
            unpack_imag_impl (std::array <value_type, lanes> const & arr, util::index_sequence <L...>) noexcept
        {
            return vector_type {std::get <L> (arr).imag ()...};
        }

        static constexpr vector_type
            unpack_real (value_type const (& arr) [lanes]) noexcept
        {
            return unpack_real_impl (arr, util::make_index_sequence <lanes> {});
        }

        static constexpr vector_type
            unpack_imag (value_type const (& arr) [lanes]) noexcept
        {
            return unpack_imag_impl (arr, util::make_index_sequence <lanes> {});
        }

        static constexpr vector_type
            unpack_real (std::array <value_type, lanes> const & arr) noexcept
        {
            return unpack_real_impl (arr, util::make_index_sequence <lanes> {});
        }

        static constexpr vector_type
            unpack_imag (std::array <value_type, lanes> const & arr) noexcept
        {
            return unpack_imag_impl (arr, util::make_index_sequence <lanes> {});
        }

        template <typename ... Ts>
        static constexpr vector_type extend_real (Ts const & ... ts)
            noexcept
        {
            return vector_type {static_cast <value_type> (ts).real ()...};
        }

        template <typename ... Ts>
        static constexpr vector_type extend_imag (Ts const & ... ts)
            noexcept
        {
            return vector_type {static_cast <value_type> (ts).imag ()...};
        }

    public:
        static complex_simd_type load (value_type const * addr) noexcept
        {
            complex_simd_type result {};
            std::memcpy (&result.data (), addr, lanes * sizeof (value_type));
            return result;
        }

        static complex_simd_type load (complex_simd_type const * addr) noexcept
        {
            return *addr;
        }

        constexpr complex_simd_type (void) noexcept
            : _realvec {extend_real (value_type {})}
            , _imagvec {extend_imag (value_type {})}
        {}

        explicit constexpr
            complex_simd_type (vector_type const & realvec,
                               vector_type const & imagvec) noexcept
            : _realvec {realvec}
            , _imagvec {imagvec}
        {}

        template <typename ... value_types>
        explicit constexpr complex_simd_type (value_types const & ... vals) noexcept
            : _realvec {extend_real (vals...)}
            , _imagvec {extend_imag (vals...)}
        {
            static_assert (
                sizeof... (value_types) == 1 ||
                sizeof... (value_types) == lanes,
                "vector constructor must be provided a number of values equal"
                " to one or equal to the number of vector lanes"
            );
        }

        constexpr
        complex_simd_type (complex_simd_type const & sv)
            noexcept
            : base {}
            , _realvec {sv._realvec}
            , _imagvec {sv._imagvec}
        {}

        explicit constexpr
        complex_simd_type (value_type const (& arr) [lanes]) noexcept
            : _realvec {unpack_real (arr)}
            , _imagvec {unpack_imag (arr)}
        {}

        explicit constexpr
        complex_simd_type (std::array <value_type, lanes> const & arr) noexcept
            : _realvec {unpack_real (arr)}
            , _imagvec {unpack_imag (arr)}
        {}

        advanced_constexpr complex_simd_type &
            operator= (complex_simd_type const & sv) & noexcept
        {
            this->_realvec = sv._realvec;
            this->_imagvec = sv._imagvec;
            return *this;
        }

        template <typename U>
        advanced_constexpr complex_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_realvec = extend_real (static_cast <value_type> (val));
            this->_imagvec = extend_imag (static_cast <value_type> (val));
            return *this;
        }

    private:
        template <std::size_t ... L>
        advanced_constexpr void fill_array (std::array <value_type, lanes> & arr,
                                   util::index_sequence <L...>) const noexcept
        {
            value_type _unused [] = {
                (std::get <L> (arr) = this->template get <L> ())...
            };
            (void) _unused;
        }

    public:
        explicit advanced_constexpr operator std::array <value_type, lanes> (void) const
            noexcept
        {
            std::array <value_type, lanes> result {};
            this->fill_array (result, util::make_index_sequence <lanes> {});
            return result;
        }

        template <typename SimdT>
        constexpr SimdT as (void) const noexcept
        {
            static_assert (
                is_simd_type <SimdT>::value,
                "cannot perform cast to non-simd type"
            );

            using traits = simd_traits <SimdT>;
            using rebind_type = rebind <
                typename traits::value_type,
                traits::lanes,
                typename traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;

            static_assert (
                2 * sizeof (vector_type) == sizeof (rebind_vector_type) ||
                lanes == traits::lanes,
                "cannot perform up-sizing or down-sizing vector cast unless"
                " the result type and source type have an equal number of lanes"
            );
            static_assert (
                sizeof (value_type) ==
                    sizeof (typename rebind_type::value_type),
                "cannot perform cast from complex simd type to simd type with "
                "differently sized value type"
            );

            std::array <typename traits::value_type, traits::lanes> result {};
            for (std::size_t i = 0; i < lanes; ++i) {
                result [2*i] = static_cast <typename traits::value_type> (
                    this->_realvec [i]
                );
                result [2*i + 1] = static_cast <typename traits::value_type> (
                    this->_imagvec [i]
                );
            }
            return rebind_type {result};
        }

        advanced_constexpr void fill (value_type const & val) noexcept
        {
            this->_realvec = extend_real (val);
            this->_realvec = extend_imag (val);
        }

        advanced_constexpr void swap (complex_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        constexpr std::array <value_type, lanes> data (void) const & noexcept
        {
            return static_cast <std::array <value_type, lanes>> (*this);
        }

        template <std::size_t lane>
        advanced_constexpr std::pair <T &, T &> get (void) & noexcept
        {
            static_assert (
                lane < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return std::make_pair (this->_realarr [lane], this->_imagarr [lane]);
        }

        template <std::size_t lane>
        constexpr std::pair <T const &, T const &> get (void) const & noexcept
        {
            static_assert (
                lane < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return std::make_pair (this->_realarr [lane], this->_imagarr [lane]);
        }

        template <std::size_t lane>
        advanced_constexpr T & get_real (void) & noexcept
        {
            static_assert (
                lane < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_realarr [lane];
        }

        template <std::size_t lane>
        constexpr T const & get_real (void) const & noexcept
        {
            static_assert (
                lane < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_realarr [lane];
        }

        template <std::size_t lane>
        advanced_constexpr T & get_imag (void) & noexcept
        {
            static_assert (
                lane < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_imagarr [lane];
        }

        template <std::size_t lane>
        constexpr T const & get_imag (void) const & noexcept
        {
            static_assert (
                lane < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_imagarr [lane];
        }

        advanced_constexpr std::pair <T &, T &> operator[] (std::size_t n) & noexcept
        {
            return std::make_pair (this->_realarr [n], this->_imagarr [n]);
        }

        constexpr std::pair <T const &, T const &> operator[] (std::size_t n) const & noexcept
        {
            return std::make_pair (this->_realarr [n], this->_imagarr [n]);
        }

        advanced_constexpr std::pair <T &, T &> at (std::size_t n) &
        {
            return n < lanes ?
                std::make_pair (this->_realarr [n], this->_imagarr [n]) :
                throw std::out_of_range {"access attempt to out-of-bounds vector lane"};
        }

        constexpr std::pair <T const &, T const &> at (std::size_t n) const &
        {
            return n < lanes ?
                std::make_pair (this->_realarr [n], this->_imagarr [n]) :
                throw std::out_of_range {"access attempt to out-of-bounds vector lane"};
        }

        advanced_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->_realarr [0], this->_imagarr [0]};
        }

        advanced_constexpr iterator end (void) & noexcept
        {
            return iterator {
                *(&this->_realarr [0] + lanes),
                *(&this->_imagarr [0] + lanes)
            };
        }

        advanced_constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->_realarr [0], this->_imagarr [0]};
        }

        advanced_constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {
                *(&this->_realarr [0] + lanes),
                *(&this->_imagarr [0] + lanes)
            };
        }

        advanced_constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->_realarr [0], this->_imagarr [0]};
        }

        advanced_constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {
                *(&this->_realarr [0] + lanes),
                *(&this->_imagarr [0] + lanes)
            };
        }

        advanced_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {
                *(&this->_realarr [0] + lanes),
                *(&this->_imagarr [0] + lanes)
            };
        }

        advanced_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->_realarr [0], this->_imagarr [0]};
        }

        advanced_constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                *(&this->_realarr [0] + lanes),
                *(&this->_imagarr [0] + lanes)
            };
        }

        advanced_constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->_realarr [0], this->_imagarr [0]};
        }

        advanced_constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                *(&this->_realarr [0] + lanes),
                *(&this->_imagarr [0] + lanes)
            };
        }

        advanced_constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->_realarr [0], this->_imagarr [0]};
        }

        constexpr complex_simd_type operator+ (void) const noexcept
        {
            return complex_simd_type {+this->_realvec, +this->_imagvec};
        }

        constexpr complex_simd_type operator- (void) const noexcept
        {
            return complex_simd_type {-this->_realvec, -this->_imagvec};
        }

        constexpr
        complex_simd_type operator+ (complex_simd_type const & sv) const
            noexcept
        {
            return complex_simd_type {
                this->_realvec + sv._realvec, this->_imagvec + sv._imagvec
            };
        }

        template <typename U>
        constexpr complex_simd_type operator+ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this + complex_simd_type {val};
        }

        constexpr complex_simd_type operator- (complex_simd_type const & sv)
            const noexcept
        {
            return complex_simd_type {
                this->_realvec - sv._realvec, this->_imagvec - sv._imagvec
            };
        }

        template <typename U>
        constexpr complex_simd_type operator- (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this - complex_simd_type {val};
        }

        advanced_constexpr complex_simd_type operator* (complex_simd_type const & sv)
            const noexcept
        {
            auto const realmul = this->_realvec * sv._realvec;
            auto const imagmul = this->_imagvec * sv._imagvec;
            auto const real_part = realmul - imagmul;

            auto const crossmul_1 = this->_realvec * sv._imagvec;
            auto const crossmul_2 = this->_imagvec * sv._realvec;
            auto const imag_part = crossmul_1 + crossmul_2;

            return complex_simd_type {real_part, imag_part};
        }

        template <typename U>
        constexpr complex_simd_type operator* (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this * complex_simd_type {val};
        }

        advanced_constexpr complex_simd_type operator/ (complex_simd_type const & sv)
            const noexcept
        {
            auto const divisor = sv._realvec * sv._realvec + sv._imagvec * sv._imagvec;

            auto const realmul = this->_realvec * sv._realvec;
            auto const imagmul = this->_imagvec * sv._imagvec;
            auto const real_part = (realmul + imagmul) / divisor;

            auto const crossmul_1 = this->_imagvec * sv._realvec;
            auto const crossmul_2 = this->_realvec * sv._imagvec;
            auto const imag_part = (crossmul_1 - crossmul_2) / divisor;

            return complex_simd_type {real_part, imag_part};
        }

        template <typename U>
        constexpr complex_simd_type operator/ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this / complex_simd_type {val};
        }

        advanced_constexpr
        complex_simd_type & operator+= (complex_simd_type const & sv) &
            noexcept
        {
            this->_realvec += sv._realvec;
            this->_imagvec += sv._imagvec;
            return *this;
        }

        template <typename U>
        advanced_constexpr complex_simd_type & operator+= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            complex_simd_type const v {val};
            return *this += v;
        }

        advanced_constexpr
        complex_simd_type & operator-= (complex_simd_type const & sv) &
            noexcept
        {
            this->_realvec -= sv._realvec;
            this->_imagvec -= sv._imagvec;
            return *this;
        }

        template <typename U>
        advanced_constexpr complex_simd_type & operator-= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            complex_simd_type const v {val};
            return *this -= v;
        }

        advanced_constexpr
        complex_simd_type & operator*= (complex_simd_type const & sv) &
            noexcept
        {
            auto const result = *this * sv;
            *this = result;
            return *this;
        }

        template <typename U>
        advanced_constexpr complex_simd_type & operator*= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            complex_simd_type const v {val};
            auto const result = *this * v;
            *this = result;
            return *this;
        }

        advanced_constexpr
        complex_simd_type & operator/= (complex_simd_type const & sv) &
            noexcept
        {
            auto const result = *this / sv;
            *this = result;
            return *this;
        }

        template <typename U>
        advanced_constexpr complex_simd_type & operator/= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            complex_simd_type const v {val};
            auto const result = *this / v;
            *this = result;
            return *this;
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator== (complex_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (
                    this->_realvec == sv._realvec &&
                    this->_imagvec == sv._imagvec
                )
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes> operator== (U val)
            const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == complex_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (complex_simd_type const & sv) const noexcept
        {
            using boolean_vector_type =
                typename boolean_simd_type <integral_type, lanes>::vector_type;

            return boolean_simd_type <integral_type, lanes> {
                static_cast <boolean_vector_type> (
                    this->_realvec != sv._realvec ||
                    this->_imagvec != sv._imagvec
                )
            };
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this != complex_simd_type {val};
        }
    };
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
    template <typename T, std::size_t l>
    class alignas (simd_type_base <T, l>::alignment)
        boolean_simd_type : public simd_type_base <T, l>
    {
    private:
        using base = simd_type_base <T, l>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
        union alignas (base::alignment)
        {
            typename base::vector_type_impl _vec;
            T _arr [l];
        };
#pragma GCC diagnostic pop

    public:
        using vector_type     = typename base::vector_type_impl;
        using value_type      = bool;
        using integral_type   = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type = typename base::signed_integral_type;
        using pointer         = typename base::template pointer_proxy <T>;
        using const_pointer   = typename base::template pointer_proxy <T const>;
        using reference       = T &;
        using const_reference = T const &;
        using iterator        = pointer;
        using const_iterator  = const_pointer;
        using reverse_iterator       = std::reverse_iterator <pointer>;
        using const_reverse_iterator = std::reverse_iterator <const_pointer>;
        static constexpr std::size_t lanes = l;

        template <typename U, std::size_t L, typename tag>
        using rebind = simd_type <U, L, tag>;

        static boolean_simd_type load (value_type const * addr) noexcept
        {
            boolean_simd_type result {};
            std::memcpy (&result.data (), addr, lanes * sizeof (value_type));
            return result;
        }

        constexpr boolean_simd_type (void) noexcept
            : _vec {base::extend (value_type {})}
        {}

        explicit constexpr boolean_simd_type (vector_type const & vec) noexcept
            : _vec {vec}
        {}

        template <typename ... value_types>
        explicit constexpr boolean_simd_type (value_types const & ... vals) noexcept
            : _vec {base::extend (vals...)}
        {
            static_assert (
                sizeof... (value_types) == 1 || sizeof... (value_types) == lanes,
                "vector constructor must be provided a number of values equal"
                " to one or equal to the number of vector lanes"
            );
        }

        constexpr boolean_simd_type (boolean_simd_type const & sv) noexcept
            : base {}
            , _vec {sv._vec}
        {}

        explicit constexpr boolean_simd_type (value_type const (&arr) [lanes])
            noexcept
            : _vec {base::unpack (arr)}
        {}

        explicit constexpr
        boolean_simd_type (std::array <value_type, lanes> const & arr) noexcept
            : _vec {base::unpack (arr)}
        {}

        advanced_constexpr boolean_simd_type & operator= (boolean_simd_type const & sv) &
            noexcept
        {
            this->_vec = sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr boolean_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec = base::extend (val);
            return *this;
        }

    private:
        template <std::size_t ... L>
        advanced_constexpr void fill_array (std::array <value_type, lanes> & arr,
                                            util::index_sequence <L...>) const noexcept
        {
            value_type _unused [] = {
                (std::get <L> (arr) = this->template get <L> ())...
            };
            (void) _unused;
        }

    public:
        explicit advanced_constexpr operator std::array <value_type, lanes> (void) const
            noexcept
        {
            std::array <value_type, lanes> result {};
            this->fill_array (result, util::make_index_sequence <lanes> {});
            return result;
        }

        template <typename SimdT>
        constexpr SimdT as (void) const noexcept
        {
            static_assert (
                is_simd_type <SimdT>::value,
                "cannot perform cast to non-simd type"
            );

            using traits = simd_traits <SimdT>;
            using rebind_type = rebind <
                typename traits::value_type,
                traits::lanes,
                typename traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;

            static_assert (
                sizeof (vector_type) == sizeof (rebind_vector_type) ||
                lanes == traits::lanes,
                "cannot perform up-sizing or down-sizing vector cast unless"
                " the result type and source type have an equal number of lanes"
            );

            return rebind_type {
                static_cast <rebind_vector_type> (this->_vec)
            };
        }

        advanced_constexpr void fill (value_type const & val) noexcept
        {
            this->_vec = base::extend (val);
        }

        advanced_constexpr void swap (boolean_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        advanced_constexpr vector_type & data (void) & noexcept
        {
            return this->_vec;
        }

        constexpr vector_type const & data (void) const & noexcept
        {
            return this->_vec;
        }

        template <std::size_t n>
        advanced_constexpr reference get (void) & noexcept
        {
            static_assert (
                n < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_arr [n];
        }

        template <std::size_t n>
        constexpr const_reference get (void) const & noexcept
        {
            static_assert (
                n < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return this->_arr [n];
        }

        advanced_constexpr reference operator[] (std::size_t n) & noexcept
        {
            return this->_arr [n];
        }

        constexpr const_reference operator[] (std::size_t n) const & noexcept
        {
            return this->_arr [n];
        }

        advanced_constexpr reference at (std::size_t n) &
        {
            return n < lanes ?
                this->_arr [n] :
                throw std::out_of_range {"access attempt to out-of-bounds vector lane"};
        }

        constexpr const_reference at (std::size_t n) const &
        {
            return n < lanes ?
                this->_arr [n] :
                throw std::out_of_range {"access attempt to out-of-bounds vector lane"};
        }

        advanced_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->_arr [0]};
        }

        advanced_constexpr iterator end (void) & noexcept
        {
            return iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->_arr [0]};
        }

        constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->_arr [0]};
        }

        constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {*(&this->_arr [0] + lanes)};
        }

        advanced_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        advanced_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->_arr [0]};
        }

        constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->_arr [0]};
        }

        constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {*(&this->_arr [0] + lanes)};
        }

        constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->_arr [0]};
        }

    private:
        advanced_constexpr bool any_of_impl (bool (&&array) [lanes]) const noexcept
        {
            for (auto b : array) {
                if (b) {
                    return true;
                }
            }

            return false;
        }

        advanced_constexpr bool all_of_impl (bool (&&array) [lanes]) const noexcept
        {
            for (auto b : array) {
                if (!b) {
                    return false;
                }
            }

            return true;
        }

        advanced_constexpr bool none_of_impl (bool (&&array) [lanes]) const noexcept
        {
            for (auto b : array) {
                if (b) {
                    return false;
                }
            }

            return true;
        }

        template <std::size_t ... L>
        advanced_constexpr bool any_of_impl (util::index_sequence <L...>) const noexcept
        {
            return any_of_impl (
                {static_cast <bool> (this->template get <L> ())...}
            );
        }

        template <std::size_t ... L>
        advanced_constexpr bool all_of_impl (util::index_sequence <L...>) const noexcept
        {
            return all_of_impl (
                {static_cast <bool> (this->template get <L> ())...}
            );
        }

        template <std::size_t ... L>
        advanced_constexpr bool none_of_impl (util::index_sequence <L...>) const noexcept
        {
            return none_of_impl (
                {static_cast <bool> (this->template get <L> ())...}
            );
        }

    public:
        advanced_constexpr bool any_of (void) const noexcept
        {
            return this->any_of_impl (util::make_index_sequence <lanes> {});
        }

        advanced_constexpr bool all_of (void) const noexcept
        {
            return this->all_of_impl (util::make_index_sequence <lanes> {});
        }

        advanced_constexpr bool none_of (void) const noexcept
        {
            return this->none_of_impl (util::make_index_sequence <lanes> {});
        }

    private:
        template <std::size_t ... L>
        constexpr boolean_simd_type normalize_impl (util::index_sequence <L...>)
            const noexcept
        {
            return boolean_simd_type {
                (this->_arr [L] ? integral_type {1} : integral_type {0})...
            };
        }

    public:
        constexpr boolean_simd_type normalize (void) const noexcept
        {
            return this->normalize_impl (util::make_index_sequence <lanes> {});
        }

        constexpr integral_simd_type <integral_type, lanes> to_integral (void)
            const noexcept
        {
            using cast_vector_type =
                typename integral_simd_type <integral_type, lanes>::vector_type;

            return integral_simd_type <integral_type, lanes> {
                static_cast <cast_vector_type> (this->_vec)
            };
        }

        constexpr boolean_simd_type operator~ (void) const noexcept
        {
            return boolean_simd_type {~this->_vec};
        }

        constexpr boolean_simd_type operator& (boolean_simd_type const & sv)
            const noexcept
        {
            return boolean_simd_type {this->_vec & sv._vec};
        }

        template <typename U>
        constexpr boolean_simd_type operator& (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return boolean_simd_type {
                this->_vec & base::extend (val)
            };
        }

        constexpr boolean_simd_type operator| (boolean_simd_type const & sv)
            const noexcept
        {
            return boolean_simd_type {this->_vec | sv._vec};
        }

        template <typename U>
        constexpr boolean_simd_type operator| (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return boolean_simd_type {
                this->_vec | base::extend (val)
            };
        }

        constexpr boolean_simd_type operator^ (boolean_simd_type const & sv)
            const noexcept
        {
            return boolean_simd_type {this->_vec ^ sv._vec};
        }

        template <typename U>
        constexpr boolean_simd_type operator^ (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return boolean_simd_type {
                this->_vec ^ base::extend (val)
            };
        }

        constexpr boolean_simd_type operator! (void) const noexcept
        {
            return boolean_simd_type {!this->_vec};
        }

        constexpr boolean_simd_type operator&& (boolean_simd_type const & sv)
            const noexcept
        {
            return boolean_simd_type {this->_vec && sv._vec};
        }

        template <typename U>
        constexpr boolean_simd_type operator&& (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return boolean_simd_type {
                this->_vec && base::extend (val)
            };
        }

        constexpr boolean_simd_type operator|| (boolean_simd_type const & sv)
            const noexcept
        {
            return boolean_simd_type {this->_vec || sv._vec};
        }

        template <typename U>
        constexpr boolean_simd_type operator|| (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return boolean_simd_type {
                this->_vec || base::extend (val)
            };
        }

        advanced_constexpr boolean_simd_type &
            operator&= (boolean_simd_type const & sv) & noexcept
        {
            this->_vec &= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr boolean_simd_type & operator&= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec &= base::extend (val);
            return *this;
        }

        advanced_constexpr boolean_simd_type &
            operator|= (boolean_simd_type const & sv) & noexcept
        {
            this->_vec |= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr boolean_simd_type & operator|= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec |= base::extend (val);
            return *this;
        }

        advanced_constexpr boolean_simd_type &
            operator^= (boolean_simd_type const & sv) & noexcept
        {
            this->_vec ^= sv._vec;
            return *this;
        }

        template <typename U>
        advanced_constexpr boolean_simd_type & operator^= (U val) & noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec ^= base::extend (val);
            return *this;
        }

        constexpr boolean_simd_type operator== (boolean_simd_type const & sv)
            const noexcept
        {
            return boolean_simd_type {this->_vec == sv._vec};
        }

        template <typename U>
        constexpr boolean_simd_type operator== (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == boolean_simd_type {val};
        }

        constexpr boolean_simd_type operator!= (boolean_simd_type const & sv)
            const noexcept
        {
            return boolean_simd_type {this->_vec != sv._vec};
        }

        template <typename U>
        constexpr boolean_simd_type operator!= (U val) const noexcept
        {
            static_assert (
                std::is_convertible <U, value_type>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this != boolean_simd_type {val};
        }
    };
#pragma GCC diagnostic pop
}   // namespace detail

    template <typename SimdT>
    struct simd_traits : public detail::simd_traits <SimdT> {};

    template <typename SimdT>
    struct simd_traits <SimdT const> : public simd_traits <SimdT>
    {};

    template <typename SimdT>
    struct simd_traits <SimdT &> : public simd_traits <SimdT>
    {};

    template <typename SimdT>
    struct simd_traits <SimdT &&> : public simd_traits <SimdT>
    {};

    template <typename SimdT>
    struct simd_traits <SimdT const &> : public simd_traits <SimdT>
    {};

    using arithmetic_tag = detail::arithmetic_tag;
    using complex_tag = detail::complex_tag;
    using boolean_tag = detail::boolean_tag;

    template <typename T, std::size_t lanes, typename tag = arithmetic_tag>
    using simd_type = detail::simd_type <T, lanes, tag>;

    template <typename SimdT>
    SimdT load (typename simd_traits <SimdT>::value_type const * addr) noexcept
    {
        return SimdT::load (addr);
    }

    template <std::size_t lane, typename T, std::size_t lanes, typename tag>
    constexpr typename simd_type <T, lanes, tag>::value_type
        get (simd_type <T, lanes, tag> const & sv) noexcept
    {
        static_assert (
            lane < lanes,
            "cannot access out-of-bounds vector lane"
        );

        return sv.template get <lane> ();
    }

    template <std::size_t lane, typename T, std::size_t lanes>
    constexpr typename simd_type <T, lanes, complex_tag>::value_type
        get_real (simd_type <T, lanes, complex_tag> const & sv) noexcept
    {
        static_assert (
            lane < lanes,
            "cannot access out-of-bounds vector lane"
        );

        return sv.template get_real <lane> ();
    }

    template <std::size_t lane, typename T, std::size_t lanes>
    constexpr typename simd_type <T, lanes, complex_tag>::value_type
        get_imag (simd_type <T, lanes, complex_tag> const & sv) noexcept
    {
        static_assert (
            lane < lanes,
            "cannot access out-of-bounds vector lane"
        );

        return sv.template get_imag <lane> ();
    }

    template <typename T, std::size_t lanes, typename tag>
    typename simd_type <T, lanes, tag>::iterator
        begin (simd_type <T, lanes, tag> & sv) noexcept
    {
        return sv.begin ();
    }

    template <typename T, std::size_t lanes, typename tag>
    typename simd_type <T, lanes, tag>::iterator
        end (simd_type <T, lanes, tag> & sv) noexcept
    {
        return sv.end ();
    }

    template <typename T, std::size_t lanes, typename tag>
    typename simd_type <T, lanes, tag>::const_iterator
        begin (simd_type <T, lanes, tag> const & sv) noexcept
    {
        return sv.begin ();
    }

    template <typename T, std::size_t lanes, typename tag>
    typename simd_type <T, lanes, tag>::const_iterator
        end (simd_type <T, lanes, tag> const & sv) noexcept
    {
        return sv.end ();
    }

    template <typename T, std::size_t lanes, typename tag>
    typename simd_type <T, lanes, tag>::const_iterator
        cbegin (simd_type <T, lanes, tag> const & sv) noexcept
    {
        return sv.cbegin ();
    }

    template <typename T, std::size_t lanes, typename tag>
    typename simd_type <T, lanes, tag>::const_iterator
        cend (simd_type <T, lanes, tag> const & sv) noexcept
    {
        return sv.cend ();
    }

    template <typename U, typename T, std::size_t lanes, typename tag>
    constexpr simd_type <T, lanes, tag>
        operator+ (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv + val;
    }

    template <typename U, typename T, std::size_t lanes, typename tag>
    constexpr simd_type <T, lanes, tag>
        operator- (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv - val;
    }

    template <typename U, typename T, std::size_t lanes, typename tag>
    constexpr simd_type <T, lanes, tag>
        operator* (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv * val;
    }

    template <typename U, typename T, std::size_t lanes, typename tag>
    constexpr simd_type <T, lanes, tag>
        operator/ (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv / val;
    }

    template <
        typename U, typename T, std::size_t lanes, typename tag,
        typename = typename std::enable_if <std::is_integral <T>::value>::type
    >
    constexpr simd_type <T, lanes, tag>
        operator& (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv & val;
    }

    template <
        typename U, typename T, std::size_t lanes, typename tag,
        typename = typename std::enable_if <std::is_integral <T>::value>::type
    >
    constexpr simd_type <T, lanes, tag>
        operator| (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv | val;
    }

    template <
        typename U, typename T, std::size_t lanes, typename tag,
        typename = typename std::enable_if <std::is_integral <T>::value>::type
    >
    constexpr simd_type <T, lanes, tag>
        operator^ (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv ^ val;
    }

    template <typename U, typename T, std::size_t lanes, typename tag>
    constexpr simd_type <T, lanes, tag>
        operator&& (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv && val;
    }

    template <typename U, typename T, std::size_t lanes, typename tag>
    constexpr simd_type <T, lanes, tag>
        operator|| (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv || val;
    }

    template <typename U, typename T, std::size_t lanes, typename tag>
    constexpr simd_type <T, lanes, tag>
        operator== (U val, simd_type <T, lanes, tag> const & sv)
    noexcept
    {
        static_assert (
            std::is_convertible <U, T>::value,
            "cannot perform operation between vector type and scalar type"
            " without conversion"
        );

        return sv == val;
    }

    template <typename T, typename U, std::size_t lanes, typename tag>
    simd_type <U, lanes, tag>
        shuffle (simd_type <U, lanes, tag> const & sv,
                 simd_type <T, lanes, tag> const & mask) noexcept
    {
        static_assert (
            std::is_integral <T>::value,
            "template parameter T of mask simd type must be an integral type"
        );

#if defined (__clang__)
    /*
     * clang's __builtin_shufflevector requires constant integer indices,
     * and hence we must implement the function by hand for the general
     * case. This can be overcome by use of the .data () method, which
     * provides access to the underlying SIMD vector type.
     */
        simd_type <U, lanes, tag> shuffle_result {};

        for (std::size_t i = 0; i < lanes; ++i) {
            shuffle_result [i] = sv [mask [i]];
        }

        return shuffle_result;
#elif defined (__GNUG__)
        return simd_type <U, lanes, tag> {
            __builtin_shuffle (sv.data (), mask.data ())
        };
#endif
    }

    template <typename T, typename U, std::size_t lanes, typename tag>
    simd_type <U, lanes, tag>
        shuffle (simd_type <U, lanes, tag> const & sv1,
                 simd_type <U, lanes, tag> const & sv2,
                 simd_type <T, lanes, tag> const & mask) noexcept
    {
        static_assert (
            std::is_integral <T>::value,
            "template parameter T of mask simd type must be an integral type"
        );

#if defined (__clang__)
    /*
     * clang's __builtin_shufflevector requires constant integer indices,
     * and hence we must implement the function by hand for the general
     * case. For the user of this library this limitation can be overcome by
     * using the .data () method, which provides access to the underlying SIMD
     * vector type.
     */
        simd_type <U, lanes, tag> shuffle_result {};

        for (std::size_t i = 0; i < lanes; ++i) {
            if (static_cast <std::size_t> (mask [i]) < lanes) {
                shuffle_result [i] = sv1 [mask [i]];
            } else {
                shuffle_result [i] = sv2 [lanes - mask [i]];
            }
        }

        return shuffle_result;
#elif defined (__GNUG__)
        return simd_type <U, lanes, tag> {
            __builtin_shuffle (sv1.data (), sv2.data (), mask.data ())
        };
#endif
    }

    template <typename T, std::size_t lanes>
    constexpr bool any_of (simd_type <T, lanes, boolean_tag> const & boolvec)
        noexcept
    {
        return boolvec.any_of ();
    }

    template <typename T, std::size_t lanes>
    constexpr bool all_of (simd_type <T, lanes, boolean_tag> const & boolvec)
        noexcept
    {
        return boolvec.all_of ();
    }

    template <typename T, std::size_t lanes>
    constexpr bool none_of (simd_type <T, lanes, boolean_tag> const & boolvec)
        noexcept
    {
        return boolvec.none_of ();
    }

    template <typename T, typename U, std::size_t lanes, typename tag>
    simd_type <U, lanes, tag>
        select (simd_type <T, lanes, arithmetic_tag> const & selector,
                simd_type <U, lanes, tag> const & then_vec,
                simd_type <U, lanes, tag> const & else_vec) noexcept
    {
        using select_type = simd_type <T, lanes, boolean_tag>;
        using integral_simd_type = typename select_type::template rebind <
            typename select_type::integral_type, lanes, arithmetic_tag
        >;

        integral_simd_type mask {
            integral_simd_type::increment_vector (lanes) -
                (selector.to_integral () * integral_simd_type {lanes})
        };

        return shuffle (then_vec, else_vec, mask);
    }

namespace detail
{
    template <typename F, typename SimdT>
    using transform_result = simd_type <
        typename std::result_of <
            F (typename simd_traits <SimdT>::value_type)
        >::type,
        simd_traits <SimdT>::lanes,
        typename simd_traits <SimdT>::category_tag
    >;
}   // namespace detail

    /*
     * Compute a new SIMD vector containing the function results of each lane of
     * the original SIMD vector.
     */
    template <typename F, typename SimdT>
    advanced_constexpr detail::transform_result <F, SimdT>
        transform (F && f, SimdT const & v)
        noexcept (noexcept (
            std::forward <F> (f) (
                std::declval <typename simd_traits <SimdT>::value_type> ()
            )
        ))
    {
        constexpr auto lanes = simd_traits <SimdT>::lanes;

        detail::transform_result <F, SimdT> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::forward <F> (f) (v [i]);
        }
        return result;
    }

    /*
     * Compute a new SIMD vector containing the hash values of each lane of the
     * original SIMD vector.
     */
    template <typename SimdT>
    auto hash (SimdT const & v)
        noexcept (noexcept (
            transform (
                std::hash <typename simd_traits <SimdT>::value_type> {}, v
            )
        ))
        -> decltype (
            transform (
                std::hash <typename simd_traits <SimdT>::value_type> {}, v
            )
        )
    {
        return transform (
            std::hash <typename simd_traits <SimdT>::value_type> {}, v
        );
    }

    /*
     * Compute a new SIMD vector containing the hash values of each lane of the
     * original SIMD vector using the provided hash function.
     */
    template <typename HashFn, typename SimdT>
    auto hash (HashFn && hfn, SimdT const & v)
        noexcept (noexcept (transform (std::forward <HashFn> (hfn), v)))
        -> decltype (
            transform (std::forward <HashFn> (hfn), v)
        )
    {
        return transform (std::forward <HashFn> (hfn), v);
    }

    /*
     * Computes the sum across the SIMD vector by the given binary operation.
     */
    template <typename SimdT, typename U, typename BinaryOp>
    U accumulate (SimdT const & v, U init, BinaryOp op)
    {
        return std::accumulate (v.begin (), v.end (), init, op);
    }

    /*
     * Computes the inner product of two arithmetic (non-boolean) SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    T inner_product (simd_type <T, lanes, arithmetic_tag> const & v,
                     simd_type <T, lanes, arithmetic_tag> const & u)
        noexcept
    {
        return simd::accumulate (v * u, T {0}, std::plus <T> {});
    }

    /*
     * Returns a SIMD vector of the real components of a complex SIMD vector.
     */
    template <typename T, std::size_t lanes>
    constexpr simd_type <T, lanes, arithmetic_tag>
        real (simd_type <T, lanes, complex_tag> const & v) noexcept
    {
        return simd_type <T, lanes, arithmetic_tag> {v.real ()};
    }

    /*
     * Returns a SIMD vector of the imaginary components of a complex SIMD
     * vector.
     */
    template <typename T, std::size_t lanes>
    constexpr simd_type <T, lanes, arithmetic_tag>
        imag (simd_type <T, lanes, complex_tag> const & v) noexcept
    {
        return simd_type <T, lanes, arithmetic_tag> {v.imag ()};
    }

    /*
     * Computes two SIMD vectors respectively containing the pairwise
     * quotient and remainder of integral division.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr std::pair <simd_type <T, lanes, arithmetic_tag>,
                         simd_type <T, lanes, arithmetic_tag>>
        div (simd_type <T, lanes, arithmetic_tag> const & v, T const & a) noexcept
    {
        using result_type = decltype (
            std::div (std::declval <T> (), std::declval <T> ())
        );

        std::array <result_type, lanes> results;
        for (std::size_t i = 0; i < lanes; ++i) {
            results [i] = std::div (v [i], a);
        }

        std::pair <simd_type <T, lanes, arithmetic_tag>,
                   simd_type <T, lanes, arithmetic_tag>>
            qr;

        for (std::size_t i = 0; i < lanes; ++i) {
            qr.first [i] = results [i].quot;
        }

        for (std::size_t i = 0; i < lanes; ++i) {
            qr.second [i] = results [i].rem;
        }

        return qr;
    }

    /*
     * Computes two SIMD vectors respectively containing the pairwise
     * quotient and remainder of integral division.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr std::pair <simd_type <T, lanes, arithmetic_tag>,
                         simd_type <T, lanes, arithmetic_tag>>
        div (simd_type <T, lanes, arithmetic_tag> const & u,
             simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        using result_type = decltype (
            std::div (std::declval <T> (), std::declval <T> ())
        );

        std::array <result_type, lanes> results;
        for (std::size_t i = 0; i < lanes; ++i) {
            results [i] = std::div (u [i], v [i]);
        }

        std::pair <simd_type <T, lanes, arithmetic_tag>,
                   simd_type <T, lanes, arithmetic_tag>>
            qr;

        for (std::size_t i = 0; i < lanes; ++i) {
            qr.first [i] = results [i].quot;
        }

        for (std::size_t i = 0; i < lanes; ++i) {
            qr.second [i] = results [i].rem;
        }

        return qr;
    }

    /*
     * Computes the absolute value for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        abs (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::abs <T>, v);
    }

    /*
     * Computes the absolute value for each lane of a complex SIMD vector
     * without undue underflow or overflow by calling std::hypot.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        abs (simd_type <T, lanes, complex_tag> const & v) noexcept
    {
        return transform (
            [] (T const & a, T const & b) { return std::hypot (a, b); }, v
        );
    }

    /*
     * Computes the phase angle for each lane of a complex SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        arg (simd_type <T, lanes, complex_tag> const & v) noexcept
    {
        return transform (std::arg <T>, v);
    }

    /*
     * Computes the norm for each lane of a complex SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        norm (simd_type <T, lanes, complex_tag> const & v) noexcept
    {
        auto const & data = v.data ();
        auto const & reals = std::get <0> (data);
        auto const & imags = std::get <1> (data);

        return reals * reals + imags * imags;
    }

    /*
     * Computes the complex conjugate for each lane of a complex SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, complex_tag>
        conj (simd_type <T, lanes, complex_tag> const & v) noexcept
    {
        auto const & data = v.data ();
        auto const & reals = std::get <0> (data);
        auto const & imags = std::get <1> (data);

        return simd_type <T, lanes, complex_tag> (reals, -imags);
    }

    /*
     * Computes the projection onto the Riemann Sphere for each lane of a
     * complex SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, complex_tag>
        proj (simd_type <T, lanes, complex_tag> const & v) noexcept
    {
        return transform (std::proj <T>, v);
    }

    /*
     * Computes the exponential for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        exp (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::exp <T>, v);
    }

    /*
     * Computes the exponent base 2 for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        exp2 (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::exp2 <T>, v);
    }

    /*
     * Computes the exponential minus 1 for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        expm1 (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::expm1 <T>, v);
    }

    /*
     * Computes the natural logarithm for each lane of a SIMD vector.
     * For complex types branch cuts occur along the negative real axis.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        log (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::log <T>, v);
    }

    /*
     * Computes the logarithm base 10 for each lane of a SIMD vector.
     * For complex types branch cuts occur along the negative real axis.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        log10 (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::log10 <T>, v);
    }

    /*
     * Computes the logarithm base 2 for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        log2 (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::log2 <T>, v);
    }

    /*
     * Computes the natural logarithm for each lane of a SIMD vector
     * plus one.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        log1p (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::log1p <T>, v);
    }

    /*
     * Computes the square root for each lane of a SIMD vector.
     * For complex types the result lies in the right half-plane.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        sqrt (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::sqrt <T>, v);
    }

    /*
     * Computes the cube root for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        cbrt (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::cbrt <T>, v);
    }

    /*
     * Computes the hypotenuse (sqrt (x^2 + y^2)) for each pairwise lane of
     * SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        hypot (simd_type <T, lanes, arithmetic_tag> const & u,
               simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        using result_type = decltype (
            std::hypot (std::declval <T> (), std::declval <T> ())
        );

        simd_type <result_type, lanes, arithmetic_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::hypot (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the power x^y for each lane, pairwise of two SIMD vectors.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        pow (simd_type <T, lanes, tag> const & u,
             simd_type <T, lanes, tag> const & v) noexcept
    {
        simd_type <T, lanes, tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::pow (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the sine for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        sin (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::sin <T>, v);
    }

    /*
     * Computes the arcsine for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        asin (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::asin <T>, v);
    }

    /*
     * Computes the cosine for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        cos (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::cos <T>, v);
    }

    /*
     * Computes the arcosine for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        acos (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::acos <T>, v);
    }

    /*
     * Computes the tangent for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        tan (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::tan <T>, v);
    }

    /*
     * Computes the arctangent for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        atan (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::atan <T>, v);
    }

    /*
     * Computes the arctangent considering signs for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        atan2 (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::atan2 <T>, v);
    }

    /*
     * Computes the hyperbolic sine for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        sinh (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::sinh <T>, v);
    }

    /*
     * Computes the area hyperbolic sine for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        asinh (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::asinh <T>, v);
    }

    /*
     * Computes the hyperbolic cosine for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        cosh (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::cosh <T>, v);
    }

    /*
     * Computes the area hyperbolic cosine for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        acosh (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::acosh <T>, v);
    }

    /*
     * Computes the hyperbolic tangent for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        tanh (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::tanh <T>, v);
    }

    /*
     * Computes the area hyperbolic tangent for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        atanh (simd_type <T, lanes, tag> const & v) noexcept
    {
        return transform (std::atanh <T>, v);
    }

    /*
     * Computes the error function for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        decltype (std::erf (std::declval <T> ())), lanes, arithmetic_tag
    >
        erf (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::erf, v);
    }

    /*
     * Computes the complementary error function for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        decltype (std::erfc (std::declval <T> ())), lanes, arithmetic_tag
    >
        erfc (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::erfc, v);
    }

    /*
     * Computes the gamma function for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        decltype (std::tgamma (std::declval <T> ())), lanes, arithmetic_tag
    >
        tgamma (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::tgamma, v);
    }

    /*
     * Computes the natural logarithm of the gramma function for each lane of a
     * SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        decltype (std::lgamma (std::declval <T> ())), lanes, arithmetic_tag
    >
        lgamma (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::lgamma, v);
    }

    /*
     * Computes the pairwise maximum of two SIMD vectors.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        max (simd_type <T, lanes, tag> const & u,
             simd_type <T, lanes, tag> const & v) noexcept
    {
        simd_type <T, lanes, tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::max (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the pairwise minimum of two SIMD vectors.
     */
    template <typename T, std::size_t lanes, typename tag>
    advanced_constexpr simd_type <T, lanes, tag>
        min (simd_type <T, lanes, tag> const & u,
             simd_type <T, lanes, tag> const & v) noexcept
    {
        simd_type <T, lanes, tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::min (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the ceil for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        ceil (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform ([] (T const & a) { return std::ceil (a); }, v);
    }

    /*
     * Computes the floor for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        floor (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform ([] (T const & a) { return std::floor (a); }, v);
    }

    /*
     * Computes the truncation value for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        trunc (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform ([] (T const & a) { return std::trunc (a); }, v);
    }

    /*
     * Computes the nearest integer value for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        round (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform ([] (T const & a) { return std::round (a); }, v);
    }

    /*
     * Computes the nearest integer value for each lane of a SIMD vector using the
     * current rounding mode.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        nearbyint (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform ([] (T const & a) { return std::nearbyint (a); }, v);
    }

    /*
     * Computes the decomposition of a number into significand and a power of 2,
     * returning a pair of SIMD vectors with the above values, respectively.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr std::pair <
        simd_type <decltype (
                    std::frexp (std::declval <T> (), std::declval <int *> ())
                   ), lanes, arithmetic_tag>,
        simd_type <int, lanes, arithmetic_tag>
    >
        frexp (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        using result_type = decltype (
            std::frexp (std::declval <T> (), std::declval <int *> ())
        );

        std::pair <simd_type <result_type, lanes, arithmetic_tag>,
                   simd_type <int, lanes, arithmetic_tag>>
            result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result.first [i] = std::frexp (v [i], &result.second [i]);
        }
        return result;
    }

    /*
     * Computes a value times the number 2 raised to the exp power for each
     * lane of a SIMD vector. This overload uses the same exp for each
     * computation.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        typename std::common_type <
            T,
            decltype (std::ldexp (std::declval <T> (), std::declval <int> ()))
        >::type,
        lanes,
        arithmetic_tag
    >
        ldexp (simd_type <T, lanes, arithmetic_tag> const & v, int exp) noexcept
    {
        using result_type = decltype (
            std::ldexp (std::declval <T> (), std::declval <int> ())
        );
        using common_type = typename std::common_type <T, result_type>::type;
        using common_simd_type = simd_type <common_type, lanes, arithmetic_tag>;

        common_simd_type const exp2 {std::exp2 (exp)};
        return static_cast <common_simd_type> (v) * exp2;
    }

    /*
     * Computes a value times the number 2 raised to the exp power for each
     * lane of a SIMD vector. This overload uses a SIMD vector of
     * (potentially different) exponents for the computation.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        typename std::common_type <
            T,
            decltype (std::ldexp (std::declval <T> (), std::declval <int> ()))
        >::type,
        lanes,
        arithmetic_tag
    >
        ldexp (simd_type <T, lanes, arithmetic_tag> const & v,
               simd_type <int, lanes, arithmetic_tag> const & exp) noexcept
    {
        using common_type = simd_type <
            typename std::common_type <
                T,
                decltype (
                    std::ldexp (std::declval <T> (), std::declval <int> ())
                )
            >::type, lanes, arithmetic_tag
        >;

        auto const exp2 = transform (std::exp2, exp);
        return static_cast <common_type> (v) * exp2;
    }

    /*
     * Computes the decomposition of floating point values into integral and
     * fractional parts for each lane of a SIMD vector. Returns a pair
     * of SIMD vectors containing the integral and fractional parts,
     * respectively.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr std::pair <
        simd_type <T, lanes, arithmetic_tag>, simd_type <T, lanes, arithmetic_tag>
    >
        modf (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        std::pair <
            simd_type <T, lanes, arithmetic_tag>, simd_type <T, lanes, arithmetic_tag>
        > result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result.first [i] = std::modf (v [i], &result.second [i]);
        }
        return result;
    }

    /*
     * Computes a value times the number FLT_RADIX raised to the exp power for
     * each lane of a SIMD vector. This overload uses the same exp for each
     * computation.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        typename std::common_type <
            T,
            decltype (std::scalbn (std::declval <T> (), std::declval <int> ()))
        >::type,
        lanes,
        arithmetic_tag
    >
        scalbn (simd_type <T, lanes, arithmetic_tag> const & v, int exp) noexcept
    {
        using result_type = decltype (
            std::scalbn (std::declval <T> (), std::declval <int> ())
        );
        using common_type = typename std::common_type <T, result_type>::type;
        using common_simd_type = simd_type <common_type, lanes, arithmetic_tag>;

        common_simd_type const exp_flt_radix {std::pow (FLT_RADIX, exp)};
        return static_cast <common_simd_type> (v) * exp_flt_radix;
    }

    /*
     * Computes a value times the number FLT_RADIX raised to the exp power for
     * each lane of a SIMD vector. This overload uses a SIMD vector of
     * (potentially different) exponents for the computation.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        typename std::common_type <
            T,
            decltype (std::scalbn (std::declval <T> (), std::declval <int> ()))
        >::type,
        lanes,
        arithmetic_tag
    >
        scalbn (simd_type <T, lanes, arithmetic_tag> const & v,
                simd_type <int, lanes, arithmetic_tag> const & exp) noexcept
    {
        using common_type = simd_type <
            typename std::common_type <
                T,
                decltype (
                    std::scalbn (std::declval <T> (), std::declval <int> ())
                )
            >::type, lanes, arithmetic_tag
        >;

        auto const exp_flt_radix = transform (
            [] (int e) { return std::pow (FLT_RADIX, e); }, exp
        );
        return static_cast <common_type> (v) * exp_flt_radix;
    }

    /*
     * Computes a value times the number FLT_RADIX raised to the long exp power
     * for each lane of a SIMD vector. This overload uses the same exp for
     * each computation.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        typename std::common_type <
            T,
            decltype (
                std::scalbln (std::declval <T> (), std::declval <long> ())
            )
        >::type,
        lanes,
        arithmetic_tag
    >
        scalbln (simd_type <T, lanes, arithmetic_tag> const & v, long exp) noexcept
    {
        using result_type = decltype (
            std::scalbln (std::declval <T> (), std::declval <long> ())
        );
        using common_type = typename std::common_type <T, result_type>::type;
        using common_simd_type = simd_type <common_type, lanes, arithmetic_tag>;

        common_simd_type const exp_flt_radix {std::pow (FLT_RADIX, exp)};
        return static_cast <common_simd_type> (v) * exp_flt_radix;
    }

    /*
     * Computes a value times the number FLT_RADIX raised to the exp power for
     * each lane of a SIMD vector. This overload uses a SIMD vector of
     * (potentially different) exponents for the computation.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        typename std::common_type <
            T,
            decltype (
                std::scalbln (std::declval <T> (), std::declval <long> ())
            )
        >::type,
        lanes,
        arithmetic_tag
    >
        scalbln (simd_type <T, lanes, arithmetic_tag> const & v,
                 simd_type <long, lanes, arithmetic_tag> const & exp) noexcept
    {
        using common_type = simd_type <
            typename std::common_type <
                T,
                decltype (
                    std::scalbln (std::declval <T> (), std::declval <long> ())
                )
            >::type, lanes, arithmetic_tag
        >;

        auto const exp_flt_radix = transform (
            [] (long e) { return std::pow (FLT_RADIX, e); }, exp
        );
        return static_cast <common_type> (v) * exp_flt_radix;
    }

    /*
     * Extracts the integral exponent of a floating point value for each lane
     * of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <int, lanes, arithmetic_tag>
        ilogb (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::ilogb, v);
    }

    /*
     * Extracts the floating point radix independent exponent of a floating
     * point value, as a floating point result for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        decltype (std::logb (std::declval <T> ())), lanes, arithmetic_tag
    >
        logb (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::ilogb, v);
    }

    /*
     * Computes the next representable value from the floating point value from
     * to the floating point value to for each lane of SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        decltype (std::nextafter (std::declval <T> (), std::declval <T> ())),
        lanes, arithmetic_tag
    >
        nextafter (simd_type <T, lanes, arithmetic_tag> const & from,
                   simd_type <T, lanes, arithmetic_tag> const & to) noexcept
    {
        simd_type <
            decltype (
                std::nextafter (std::declval <T> (), std::declval <T> ())
            ),
            lanes, arithmetic_tag
        > result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::nextafter (from [i], to [i]);
        }
        return result;
    }

    /*
     * Computes the next representable value from the floating point value from
     * to the floating point value to for each lane of SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        decltype (std::nextafter (std::declval <T> (), std::declval <T> ())),
        lanes, arithmetic_tag
    >
        nexttoward (simd_type <T, lanes, arithmetic_tag> const & from,
                    simd_type <T, lanes, arithmetic_tag> const & to) noexcept
    {
        simd_type <
            decltype (
                std::nextafter (std::declval <T> (), std::declval <T> ())
            ),
            lanes, arithmetic_tag
        > result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::nexttoward (from [i], to [i]);
        }
        return result;
    }

    /*
     * Computes a floating point value with the magnitude of the first floating
     * point value and the sign of the second floating point value for each lane
     * of SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <
        decltype (std::nextafter (std::declval <T> (), std::declval <T> ())),
        lanes, arithmetic_tag
    >
        copysign (simd_type <T, lanes, arithmetic_tag> const & mag,
                  simd_type <T, lanes, arithmetic_tag> const & sgn) noexcept
    {
        simd_type <
            decltype (
                std::nextafter (std::declval <T> (), std::declval <T> ())
            ),
            lanes, arithmetic_tag
        > result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::copysign (mag [i], sgn [i]);
        }
        return result;
    }

    /*
     * Classifies the floating point value into one of: zero, subnormal, normal,
     * infinite, NaN, or an implementation defined category for each lane of a
     * SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <int, lanes, arithmetic_tag>
        fpclassify (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::fpclassify, v);
    }

    /*
     * Determines if a floating point value is finite for each lane of a SIMD
     * vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isfinite (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        constexpr simd_type <T, lanes, arithmetic_tag> pinf {+INFINITY};
        constexpr simd_type <T, lanes, arithmetic_tag> ninf {-INFINITY};

        return (v == v) && (v != pinf) && (v != ninf);
    }

    /*
     * Determines if a floating point value is infinite for each lane of a SIMD
     * vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isinf (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        constexpr simd_type <T, lanes, arithmetic_tag> pinf {+INFINITY};
        constexpr simd_type <T, lanes, arithmetic_tag> ninf {-INFINITY};

        return (v == pinf) || (v == ninf);
    }

    /*
     * Determines if a floating point value is not-a-number for each lane of a
     * SIMD vector.
     */
    template <typename T, std::size_t lanes>
    constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isnan (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return v != v;
    }

    /*
     * Determines if a floating point value is normal (neither zero, subnormal,
     * infinite, nor NaN) for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isnormal (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::isnormal, v);
    }

    /*
     * Determines if a floating point value is negative for each lane of a SIMD
     * vector.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        signbit (simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        return transform (std::signbit, v);
    }

    /*
     * Determines the result of whether a floating point value is greater than
     * another floating point value for each lane of a SIMD vector.  This
     * function does not set floating point exceptions. This overload uses the
     * same value for comparison across all lanes of the first argument.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isgreater (simd_type <T, lanes, arithmetic_tag> const & v, T const & cmp)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::isgreater (v [i], cmp);
        }
        return result;
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * greater than another floating point value for each lane of SIMD vectors.
     * This function does not set floating point exceptions.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isgreater (simd_type <T, lanes, arithmetic_tag> const & u,
                   simd_type <T, lanes, arithmetic_tag> const & v)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::isgreater (u [i], v [i]);
        }
        return result;
    }

    /*
     * Determines the result of whether a floating point value is greater than
     * or equal to another floating point value for each lane of a SIMD vector.
     * This function does not set floating point exceptions. This overload uses
     * the same value for comparison across all lanes of the first argument.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isgreaterequal (simd_type <T, lanes, arithmetic_tag> const & v, T const & cmp)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::isgreaterequal (v [i], cmp);
        }
        return result;
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * greater than or equal to another floating point value for each lane of
     * SIMD vectors. This function does not set floating point exceptions.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isgreaterequal (simd_type <T, lanes, arithmetic_tag> const & u,
                        simd_type <T, lanes, arithmetic_tag> const & v)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::isgreaterequal (u [i], v [i]);
        }
        return result;
    }

    /*
     * Determines the result of whether a floating point value is less than
     * another floating point value for each lane of a SIMD vector.  This
     * function does not set floating point exceptions. This overload uses the
     * same value for comparison across all lanes of the first argument.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isless (simd_type <T, lanes, arithmetic_tag> const & v, T const & cmp)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::isless (v [i], cmp);
        }
        return result;
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * less than another floating point value for each lane of SIMD vectors.
     * This function does not set floating point exceptions.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isless (simd_type <T, lanes, arithmetic_tag> const & u,
                simd_type <T, lanes, arithmetic_tag> const & v)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::isless (u [i], v [i]);
        }
        return result;
    }

    /*
     * Determines the result of whether a floating point value is less than
     * or equal to another floating point value for each lane of a SIMD vector.
     * This function does not set floating point exceptions. This overload uses
     * the same value for comparison across all lanes of the first argument.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        islessequal (simd_type <T, lanes, arithmetic_tag> const & v, T const & cmp)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::islessequal (v [i], cmp);
        }
        return result;
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * less than or equal to another floating point value for each lane of
     * SIMD vectors. This function does not set floating point exceptions.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        islessequal (simd_type <T, lanes, arithmetic_tag> const & u,
                     simd_type <T, lanes, arithmetic_tag> const & v)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::islessequal (u [i], v [i]);
        }
        return result;
    }

    /*
     * Determines the result of whether a floating point value is less than
     * or greater than another floating point value for each lane of a SIMD
     * vector. This function does not set floating point exceptions. This
     * overload uses the same value for comparison across all lanes of the first
     * argument.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        islessgreater (simd_type <T, lanes, arithmetic_tag> const & v, T const & cmp)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::islessgreater (v [i], cmp);
        }
        return result;
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * less than or greater than another floating point value for each lane of
     * SIMD vectors. This function does not set floating point exceptions.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        islessgreater (simd_type <T, lanes, arithmetic_tag> const & u,
                       simd_type <T, lanes, arithmetic_tag> const & v)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::islessgreater (u [i], v [i]);
        }
        return result;
    }

    /*
     * Determines the result of whether a floating point value is unordered
     * with another floating point value for each lane of a SIMD  vector. This
     * function does not set floating point exceptions. This overload uses the
     * same value for comparison across all lanes of the first argument.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isunordered (simd_type <T, lanes, arithmetic_tag> const & v, T const & cmp)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::isunordered (v [i], cmp);
        }
        return result;
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * unordered with another floating point value for each lane of SIMD
     * vectors. This function does not set floating point exceptions.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <detail::integral_type_switch <T>, lanes, boolean_tag>
        isunordered (simd_type <T, lanes, arithmetic_tag> const & u,
                       simd_type <T, lanes, arithmetic_tag> const & v)
        noexcept
    {
        simd_type <detail::integral_type_switch <T>, lanes, boolean_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::isunordered (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the pairwise fmod of two SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        fmod (simd_type <T, lanes, arithmetic_tag> const & u,
              simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        using result_type = decltype (
            std::fmod (std::declval <T> (), std::declval <T> ())
        );

        simd_type <result_type, lanes, arithmetic_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::fmod (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the pairwise remainder of two SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        remainder (simd_type <T, lanes, arithmetic_tag> const & u,
                   simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        using result_type = decltype (
            std::remainder (std::declval <T> (), std::declval <T> ())
        );

        simd_type <result_type, lanes, arithmetic_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::remainder (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the fused multiply and add operation of three SIMD vectors,
     * in the form (u * v) + w.
     */
    template <typename T, std::size_t lanes>
    advanced_constexpr simd_type <T, lanes, arithmetic_tag>
        fma (simd_type <T, lanes, arithmetic_tag> const & u,
             simd_type <T, lanes, arithmetic_tag> const & v,
             simd_type <T, lanes, arithmetic_tag> const & w) noexcept
    {
        using result_type = decltype (
            std::fma (
                std::declval <T> (), std::declval <T> (), std::declval <T> ()
            )
        );

        simd_type <result_type, lanes, arithmetic_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::fma (u [i], v [i], w [i]);
        }
        return result;
    }

#if __cplusplus > 201402L
    /*
     * Computes the pairwise gcd of two SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    constexpr simd_type <T, lanes, arithmetic_tag>
        gcd (simd_type <T, lanes, arithmetic_tag> const & u,
             simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        static_assert (
            std::is_integral <T>::value,
            "template parameter type T must be an integral type"
        );

        simd_type <T, lanes, arithmetic_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::gcd (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the pairwise lcm of two SIMD vectors.
     */
    template <typename T, std::size_t lanes>
    constexpr simd_type <T, lanes, arithmetic_tag>
        lcm (simd_type <T, lanes, arithmetic_tag> const & u,
             simd_type <T, lanes, arithmetic_tag> const & v) noexcept
    {
        static_assert (
            std::is_integral <T>::value,
            "template parameter type T must be an integral type"
        );

        simd_type <T, lanes, arithmetic_tag> result {};
        for (std::size_t i = 0; i < lanes; ++i) {
            result [i] = std::lcm (u [i], v [i]);
        }
        return result;
    }

    /*
     * Computes the clamped value for each lane of a SIMD vector.
     */
    template <typename T, std::size_t lanes, typename tag>
    constexpr simd_type <T, lanes, tag>
        clamp (simd_type <T, lanes, tag> const & v, T const & lo, T const & hi)
        noexcept
    {
        return transform (
            [&lo, &hi] (T const & a) { return std::clamp (a, lo, hi); }, v
        );
    }
#endif

    /*
     * These are the default provided typedefs for SIMD vector types, they cover
     * the normal range of vector types available on targets with 64 bit,
     * 128 bit, 256 bit, and/or 512 bit support. Clang and GCC will synthesize
     * instructions when target hardware support is not available, so we
     * provide all common SIMD vector types. Target specific typedefs are
     * provided below.
     */
inline namespace common
{
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 32 8-bit lanes */
    using bool8x32_t = simd_type <std::int8_t, 32, simd::boolean_tag>;
    using int8x32_t  = simd_type <std::int8_t, 32>;
    using uint8x32_t = simd_type <std::uint8_t, 32>;

    /* 64 8-bit lanes */
    using bool8x64_t = simd_type <std::int8_t, 64, simd::boolean_tag>;
    using int8x64_t  = simd_type <std::int8_t, 64>;
    using uint8x64_t = simd_type <std::uint8_t, 64>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 16 16-bit lanes */
    using bool16x16_t = simd_type <std::int16_t, 16, simd::boolean_tag>;
    using int16x16_t  = simd_type <std::int16_t, 16>;
    using uint16x16_t = simd_type <std::uint16_t, 16>;

    /* 32 16-bit lanes */
    using bool16x32_t = simd_type <std::int16_t, 32, simd::boolean_tag>;
    using int16x32_t  = simd_type <std::int16_t, 32>;
    using uint16x32_t = simd_type <std::uint16_t, 32>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;
    using float32x2_t = simd_type <float, 2>;
    using complex_float32x2_t = simd_type <float, 2, simd::complex_tag>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 8 32-bit lanes */
    using bool32x8_t  = simd_type <std::int32_t, 8, simd::boolean_tag>;
    using int32x8_t   = simd_type <std::int32_t, 8>;
    using uint32x8_t  = simd_type <std::uint32_t, 8>;
    using float32x8_t = simd_type <float, 8>;
    using complex_float32x8_t = simd_type <float, 8, simd::complex_tag>;

    /* 16 32-bit lanes */
    using bool32x16_t  = simd_type <std::int32_t, 16, simd::boolean_tag>;
    using int32x16_t   = simd_type <std::int32_t, 16>;
    using uint32x16_t  = simd_type <std::uint32_t, 16>;
    using float32x16_t = simd_type <float, 16>;
    using complex_float32x16_t = simd_type <float, 16, simd::complex_tag>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;
    using float64x1_t = simd_type <double, 1>;
    using complex_float64x1_t = simd_type <double, 1, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 4 64-bit lanes */
    using bool64x4_t  = simd_type <std::int64_t, 4, simd::boolean_tag>;
    using int64x4_t   = simd_type <std::int64_t, 4>;
    using uint64x4_t  = simd_type <std::uint64_t, 4>;
    using float64x4_t = simd_type <double, 4>;
    using complex_float64x4_t = simd_type <double, 4, simd::complex_tag>;

    /* 8 64-bit lanes */
    using bool64x8_t  = simd_type <std::int64_t, 8, simd::boolean_tag>;
    using int64x8_t   = simd_type <std::int64_t, 8>;
    using uint64x8_t  = simd_type <std::uint64_t, 8>;
    using float64x8_t = simd_type <double, 8>;
    using complex_float64x8_t = simd_type <float, 8, simd::complex_tag>;

    /*
     * long double specializations; may be 80-bit (x87), 128-bit,
     * or even a synonym for double floating point types depending
     * on the implementation.
     */

    /* long double x 2 */
    using long_doublex2_t = simd_type <long double, 2>;
    using complex_long_doublex2_t =
        simd_type <long double, 2, simd::complex_tag>;

    /* long double x 4 */
    using long_doublex4_t = simd_type <long double, 4>;
    using complex_long_doublex4_t =
        simd_type <long double, 4, simd::complex_tag>;

    /* Guaranteed 128-bit integer SIMD vectors */
    /* 1 128-bit lane */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif

    /* 2 128-bit lanes */
#if defined (__clang__)
    using bool128x2_t = simd_type <__int128_t, 2, simd::boolean_tag>;
    using int128x2_t  = simd_type <__int128_t, 2>;
    using uint128x2_t = simd_type <__uint128_t, 2>;
#elif defined (__GNUG__)
    using bool128x2_t = simd_type <__int128, 2, simd::boolean_tag>;
    using int128x2_t  = simd_type <__int128, 2>;
    using uint128x2_t = simd_type <unsigned __int128, 2>;
#endif

    /* 4 128-bit lanes */
#if defined (__clang__)
    using bool128x4_t = simd_type <__int128_t, 4, simd::boolean_tag>;
    using int128x4_t  = simd_type <__int128_t, 4>;
    using uint128x4_t = simd_type <__uint128_t, 4>;
#elif defined (__GNUG__)
    using bool128x4_t = simd_type <__int128, 4, simd::boolean_tag>;
    using int128x4_t  = simd_type <__int128, 4>;
    using uint128x4_t = simd_type <unsigned __int128, 4>;
#endif
}   // inline namespace common

    /*
     * These are target technology specific typedefs for SIMD vector types, for
     * when it is undesirable to allow Clang or GCC to synthesize instructions
     * for the target archetecture.
     *
     * Please note that for any given target technology Clang and GCC
     * may still have to synthesize instructions if operations that are
     * not backed by hardware support are used. It is up to the end-user
     * to ensure that the used operations have hardware support.
     */
namespace mmx
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;
}   // namespace mmx

namespace sse
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 4 32 bit lanes */
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;
}   // namespace sse

namespace sse2
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif
}   // namespace sse2

/*
 * Available types in SSE3/4.1/4.2 and SSE4a (AMD) are the same as those
 * availalbe in SSE2 since no new registers beyond the MMX and XMM registers
 * were introduced (256 bit registers were not available until the AVX
 * extensions).
 */
namespace sse3
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif
}   // namespace sse3

namespace ssse3
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif
}   // namespace ssse3

namespace sse4
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif
}   // namespace sse4

namespace sse4_1 = sse4;
namespace sse4_2 = sse4;

namespace sse4a
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif
}   // namespace sse4a

namespace avx
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif

    /* ymm registers (256-bit) */
    /* 8 32-bit lanes */
    using float32x8_t = simd_type <float, 8>;
    using complex_float32x8_t = simd_type <float, 8, simd::complex_tag>;

    /* 4 64-bit lanes */
    using float64x4_t = simd_type <double, 4>;
    using complex_float64x4_t = simd_type <double, 4, simd::complex_tag>;
}   // namespace avx

namespace avx2
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif

    /* ymm registers (256-bit) */
    /* 8 32-bit lanes */
    using bool32x8_t  = simd_type <std::int32_t, 8, simd::boolean_tag>;
    using int32x8_t   = simd_type <std::int32_t, 8>;
    using uint32x8_t  = simd_type <std::uint32_t, 8>;
    using float32x8_t = simd_type <float, 8>;
    using complex_float32x8_t = simd_type <float, 8, simd::complex_tag>;

    /* 4 64-bit lanes */
    using bool64x4_t  = simd_type <std::int64_t, 4, simd::boolean_tag>;
    using int64x4_t   = simd_type <std::int64_t, 4>;
    using uint64x4_t  = simd_type <std::uint64_t, 4>;
    using float64x4_t = simd_type <double, 4>;
    using complex_float64x4_t = simd_type <double, 4, simd::complex_tag>;
}   // namespace avx2

namespace avx512
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
    using float64x2_t = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, simd::complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if defined (__clang__)
    using bool128x1_t = simd_type <__int128_t, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif defined (__GNUG__)
    using bool128x1_t = simd_type <__int128, 1, simd::boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif

    /* ymm registers (256-bit) */
    /* 8 32-bit lanes */
    using bool32x8_t  = simd_type <std::int32_t, 8, simd::boolean_tag>;
    using int32x8_t   = simd_type <std::int32_t, 8>;
    using uint32x8_t  = simd_type <std::uint32_t, 8>;
    using float32x8_t = simd_type <float, 8>;
    using complex_float32x8_t = simd_type <float, 8, simd::complex_tag>;

    /* 4 64-bit lanes */
    using bool64x4_t  = simd_type <std::int64_t, 4, simd::boolean_tag>;
    using int64x4_t   = simd_type <std::int64_t, 4>;
    using uint64x4_t  = simd_type <std::uint64_t, 4>;
    using float64x4_t = simd_type <double, 4>;
    using complex_float64x4_t = simd_type <double, 4, simd::complex_tag>;

    /* zmm registers (512-bit) */
    /* 16 32-bit lanes */
    using bool32x16_t  = simd_type <std::int32_t, 16, simd::boolean_tag>;
    using int32x16_t   = simd_type <std::int32_t, 16>;
    using uint32x16_t  = simd_type <std::uint32_t, 16>;
    using float32x16_t = simd_type <float, 16>;
    using complex_float32x16_t = simd_type <float, 16, simd::complex_tag>;

    /* 8 64-bit lanes */
    using bool64x8_t  = simd_type <std::int64_t, 8, simd::boolean_tag>;
    using int64x8_t   = simd_type <std::int64_t, 8>;
    using uint64x8_t  = simd_type <std::uint64_t, 8>;
    using float64x8_t = simd_type <double, 8>;
    using complex_float64x8_t = simd_type <double, 8, simd::complex_tag>;
}   // namespace avx512

namespace neon
{
    /* 64-bit registers (ARM doubleword registers -- D0, D1, ...) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, simd::boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, simd::boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, simd::boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;
    using float32x2_t = simd_type <float, 2>;

    /* 1 64-bit lane */
    using bool64x1_t  = simd_type <std::int64_t, 1, simd::boolean_tag>;
    using int64x1_t   = simd_type <std::int64_t, 1>;
    using uint64x1_t  = simd_type <std::uint64_t, 1>;

    /* 128-bit registers (ARM quadword registers -- Q0, Q1, ...) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, simd::boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, simd::boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t  = simd_type <std::int32_t, 4, simd::boolean_tag>;
    using int32x4_t   = simd_type <std::int32_t, 4>;
    using uint32x4_t  = simd_type <std::uint32_t, 4>;
    using float32x4_t = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, simd::complex_tag>;

    /* 2 64-bit lane */
    using bool64x2_t  = simd_type <std::int64_t, 2, simd::boolean_tag>;
    using int64x2_t   = simd_type <std::int64_t, 2>;
    using uint64x2_t  = simd_type <std::uint64_t, 2>;
}   // namespace neon
}   // namespace simd

#undef advanced_constexpr
#undef simd_arm
#undef simd_neon
#undef simd_x86
#undef simd_mmx
#undef simd_sse
#undef simd_sse2
#undef simd_sse3
#undef simd_ssse3
#undef simd_see4_1
#undef simd_see4_2
#undef simd_avx
#undef simd_avx2
#undef simd_avx512

#include <cctype>    // std::is[x]digit
#include <cwctype>   // std::isw[x]digit
#include <iostream>  // std::(w){i,o}stream

/*
 * The following provide overloads for std namespace functions, including:
 *      - operator<< (narrow and wide character streams)
 *      - operator>> (narrow and wide character streams)
 *      - std::hash
 */

template <
    typename SimdT,
    typename = typename std::enable_if <simd::detail::is_simd_type <SimdT>::value>::type
>
std::ostream & operator<< (std::ostream & os, SimdT const & v)
{
    static constexpr std::size_t lanes = simd::simd_traits <SimdT>::lanes;

    os << '(';
    for (std::size_t i = 0; i < lanes - 1; ++i) {
        os << v [i] << "; ";
    }
    os << v [lanes - 1] << ')';

    return os;
}

template <
    typename SimdT,
    typename = typename std::enable_if <simd::detail::is_simd_type <SimdT>::value>::type
>
std::wostream & operator<< (std::wostream & os, SimdT const & v)
{
    static constexpr std::size_t lanes = simd::simd_traits <SimdT>::lanes;

    os << L'(';
    for (std::size_t i = 0; i < lanes - 1; ++i) {
        os << v [i] << L';' << L' ';
    }
    os << v [lanes - 1] << L')';

    return os;
}

template <
    typename SimdT,
    typename = typename std::enable_if <simd::detail::is_simd_type <SimdT>::value>::type
>
std::istream & operator>> (std::istream & is, SimdT & v)
{
    static constexpr std::size_t lanes = simd::simd_traits <SimdT>::lanes;

    auto nonnum = [](std::istream & _is) -> std::istream &
    {
        while (!_is.eof () && !_is.bad ()) {
            auto const flags {_is.flags ()};
            auto const peek {_is.peek ()};
            if (flags & std::ios_base::dec) {
                if (!std::isdigit (peek)) {
                    _is.ignore ();
                    continue;
                } else {
                    break;
                }
            } else if (flags & std::ios_base::oct) {
                if (!std::isdigit (peek) || peek == '8' || peek == '9') {
                    _is.ignore ();
                    continue;
                } else {
                    break;
                }
            } else if (flags & std::ios_base::hex) {
                if (!std::isxdigit (peek)) {
                    _is.ignore ();
                    continue;
                } else {
                    break;
                }
            }
        }

        return _is;
    };

    {
        std::size_t i = 0;
        while (i < lanes && !is.eof () && !is.bad ()) {
            is >> v [i];
            if (is.fail ()) {
                is.clear ();
                is >> nonnum;
            } else {
                i += 1;
            }
        }
    }

    return is;
}

template <
    typename SimdT,
    typename = typename std::enable_if <simd::detail::is_simd_type <SimdT>::value>::type
>
std::wistream & operator>> (std::wistream & wis, SimdT & v)
{
    static constexpr std::size_t lanes = simd::simd_traits <SimdT>::lanes;

    auto nonnum = [](std::wistream & _wis) -> std::wistream &
    {
        while (!_wis.eof () && !_wis.bad ()) {
            auto const flags {_wis.flags ()};
            auto const peek {_wis.peek ()};
            if (flags & std::ios_base::dec) {
                if (!std::iswdigit (peek)) {
                    _wis.ignore ();
                    continue;
                } else {
                    break;
                }
            } else if (flags & std::ios_base::oct) {
                if (!std::iswdigit (peek) || peek == L'8' || peek == L'9') {
                    _wis.ignore ();
                    continue;
                } else {
                    break;
                }
            } else if (flags & std::ios_base::hex) {
                if (!std::iswxdigit (peek)) {
                    _wis.ignore ();
                    continue;
                } else {
                    break;
                }
            }
        }

        return _wis;
    };

    {
        std::size_t i = 0;
        while (i < lanes && !wis.eof () && !wis.bad ()) {
            wis >> v [i];
            if (wis.fail ()) {
                wis.clear ();
                wis >> nonnum;
            } else {
                i += 1;
            }
        }
    }

    return wis;
}

namespace std
{
    /*
     * Computes a single hash value for an object of a SIMD vector type.
     */
#define std_hash_impl(ty, lanes, tag) template <>\
    struct hash <simd::simd_type <ty, lanes, tag>>\
    {\
        typedef simd::simd_type <ty, lanes, tag> argument_type;\
        typedef std::size_t result_type;\
\
        result_type operator() (argument_type const & s) const noexcept\
        {\
            using value_type = typename simd::simd_traits <\
                argument_type\
            >::value_type;\
\
            return simd::accumulate (\
                simd::hash <argument_type> (s), std::size_t {0},\
                [] (std::size_t const & seed, value_type const & t) {\
                    return simd::detail::util::hash_combine <value_type> (\
                            seed, t\
                    );\
                }\
            );\
        }\
    };

#define std_hash_impl_lanes(ty)\
    std_hash_impl(ty, 1, simd::boolean_tag)\
    std_hash_impl(ty, 2, simd::boolean_tag)\
    std_hash_impl(ty, 4, simd::boolean_tag)\
    std_hash_impl(ty, 8, simd::boolean_tag)\
    std_hash_impl(ty, 16, simd::boolean_tag)\
    std_hash_impl(ty, 32, simd::boolean_tag)\
    std_hash_impl(ty, 64, simd::boolean_tag)\
    std_hash_impl(ty, 1, simd::arithmetic_tag)\
    std_hash_impl(ty, 2, simd::arithmetic_tag)\
    std_hash_impl(ty, 4, simd::arithmetic_tag)\
    std_hash_impl(ty, 8, simd::arithmetic_tag)\
    std_hash_impl(ty, 16, simd::arithmetic_tag)\
    std_hash_impl(ty, 32, simd::arithmetic_tag)\
    std_hash_impl(ty, 64, simd::arithmetic_tag)\
    std_hash_impl(unsigned ty, 1, simd::arithmetic_tag)\
    std_hash_impl(unsigned ty, 2, simd::arithmetic_tag)\
    std_hash_impl(unsigned ty, 4, simd::arithmetic_tag)\
    std_hash_impl(unsigned ty, 8, simd::arithmetic_tag)\
    std_hash_impl(unsigned ty, 16, simd::arithmetic_tag)\
    std_hash_impl(unsigned ty, 32, simd::arithmetic_tag)\
    std_hash_impl(unsigned ty, 64, simd::arithmetic_tag)

    std_hash_impl_lanes(char)
    std_hash_impl_lanes(short)
    std_hash_impl_lanes(int)
    std_hash_impl_lanes(long)
    std_hash_impl_lanes(long long)

#undef std_hash_impl_lanes

#define std_hash_impl_float_lanes(ty)\
    std_hash_impl(ty, 1, simd::arithmetic_tag)\
    std_hash_impl(ty, 2, simd::arithmetic_tag)\
    std_hash_impl(ty, 4, simd::arithmetic_tag)\
    std_hash_impl(ty, 8, simd::arithmetic_tag)\
    std_hash_impl(ty, 16, simd::arithmetic_tag)\
    std_hash_impl(ty, 32, simd::arithmetic_tag)\
    std_hash_impl(ty, 64, simd::arithmetic_tag)

    std_hash_impl_float_lanes(float)
    std_hash_impl_float_lanes(double)
    std_hash_impl_float_lanes(long double)

#undef std_hash_impl_float_lanes
#undef std_hash_impl

#define std_hash_impl_int128(ty, lanes, tag) template <>\
    struct hash <simd::simd_type <ty, lanes, tag>>\
    {\
        typedef simd::simd_type <ty, lanes, tag> argument_type;\
        typedef std::size_t result_type;\
\
        result_type operator() (argument_type const & s) const noexcept\
        {\
            struct alias {\
                simd::simd_type <std::uint64_t, lanes, simd::arithmetic_tag>\
                    v1;\
                simd::simd_type <std::uint64_t, lanes, simd::arithmetic_tag>\
                    v2;\
            };\
\
            auto const & a = reinterpret_cast <alias const &> (s);\
            auto h1 = simd::hash <decltype (a.v1)> (a.v1);\
            auto h2 = simd::hash <decltype (a.v2)> (a.v2);\
\
            using hash_type = decltype (h1);\
            using hash_value_type =\
                typename simd::simd_traits <hash_type>::value_type;\
            return simd::accumulate (\
                h1 ^ ((h2  + hash_type {hash_value_type {0x9e3779b9}}) +\
                      (h1 << hash_type {hash_value_type {6}}) +\
                      (h1 >> hash_type {hash_value_type {2}})),\
                std::size_t {0},\
                [] (std::size_t const & seed, std::uint64_t const & t) {\
                    return simd::detail::util::hash_combine <std::uint64_t> (\
                            seed, t\
                    );\
                }\
            );\
        }\
    };

#if defined (__clang__)
    std_hash_impl_int128(__int128_t, 1, simd::boolean_tag)
    std_hash_impl_int128(__int128_t, 2, simd::boolean_tag)
    std_hash_impl_int128(__int128_t, 4, simd::boolean_tag)
    std_hash_impl_int128(__int128_t, 8, simd::boolean_tag)
    std_hash_impl_int128(__int128_t, 16, simd::boolean_tag)
    std_hash_impl_int128(__int128_t, 32, simd::boolean_tag)
    std_hash_impl_int128(__int128_t, 64, simd::boolean_tag)
    std_hash_impl_int128(__int128_t, 1, simd::arithmetic_tag)
    std_hash_impl_int128(__int128_t, 2, simd::arithmetic_tag)
    std_hash_impl_int128(__int128_t, 4, simd::arithmetic_tag)
    std_hash_impl_int128(__int128_t, 8, simd::arithmetic_tag)
    std_hash_impl_int128(__int128_t, 16, simd::arithmetic_tag)
    std_hash_impl_int128(__int128_t, 32, simd::arithmetic_tag)
    std_hash_impl_int128(__int128_t, 64, simd::arithmetic_tag)
    std_hash_impl_int128(__uint128_t, 1, simd::arithmetic_tag)
    std_hash_impl_int128(__uint128_t, 2, simd::arithmetic_tag)
    std_hash_impl_int128(__uint128_t, 4, simd::arithmetic_tag)
    std_hash_impl_int128(__uint128_t, 8, simd::arithmetic_tag)
    std_hash_impl_int128(__uint128_t, 16, simd::arithmetic_tag)
    std_hash_impl_int128(__uint128_t, 32, simd::arithmetic_tag)
    std_hash_impl_int128(__uint128_t, 64, simd::arithmetic_tag)
#elif defined (__GNUG__)
    std_hash_impl_int128(__int128, 1, simd::boolean_tag)
    std_hash_impl_int128(__int128, 2, simd::boolean_tag)
    std_hash_impl_int128(__int128, 4, simd::boolean_tag)
    std_hash_impl_int128(__int128, 8, simd::boolean_tag)
    std_hash_impl_int128(__int128, 16, simd::boolean_tag)
    std_hash_impl_int128(__int128, 32, simd::boolean_tag)
    std_hash_impl_int128(__int128, 64, simd::boolean_tag)
    std_hash_impl_int128(__int128, 1, simd::arithmetic_tag)
    std_hash_impl_int128(__int128, 2, simd::arithmetic_tag)
    std_hash_impl_int128(__int128, 4, simd::arithmetic_tag)
    std_hash_impl_int128(__int128, 8, simd::arithmetic_tag)
    std_hash_impl_int128(__int128, 16, simd::arithmetic_tag)
    std_hash_impl_int128(__int128, 32, simd::arithmetic_tag)
    std_hash_impl_int128(__int128, 64, simd::arithmetic_tag)
    std_hash_impl_int128(unsigned __int128, 1, simd::arithmetic_tag)
    std_hash_impl_int128(unsigned __int128, 2, simd::arithmetic_tag)
    std_hash_impl_int128(unsigned __int128, 4, simd::arithmetic_tag)
    std_hash_impl_int128(unsigned __int128, 8, simd::arithmetic_tag)
    std_hash_impl_int128(unsigned __int128, 16, simd::arithmetic_tag)
    std_hash_impl_int128(unsigned __int128, 32, simd::arithmetic_tag)
    std_hash_impl_int128(unsigned __int128, 64, simd::arithmetic_tag)
#endif

#undef std_hash_impl_int128
}   // namespace std

#endif  // #ifndef SIMD_IMPLEMENTATION_HEADER
