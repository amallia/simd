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

#include <algorithm>            // std::clamp
#include <array>                // std::array
#include <cfloat>               // FLT_RADIX
#include <cmath>                // **all math functions from namespace std::**
#include <complex>              // std::complex
#include <cstddef>              // std::size_t
#include <cstdint>              // std::{u}int{8,16,32,64}_t
#include <cstring>              // std::memcpy
#include <functional>           // std::hash
#include <initializer_list>     // std::initializer_list
#include <iterator>             // std::iterator, std::reverse_iterator
#include <memory>               // std::align, std::addressof
#include <mutex>                // std::lock_guard, std::mutex
#include <new>                  // std::{get,set}_new_handler
#include <numeric>              // std::accumulate, std::gcd, std::lcm
#include <stdexcept>            // std::bad_alloc
#include <type_traits>          // std::conditional, std::is_arithmetic
#include <utility>              // std::forward, std::index_sequence

#if !defined (__clang__) && !defined (__GNUG__)
    #error "simd implemention requires clang or gcc vector extensions"
#endif

#if defined (__clang__)
    #define SIMD_HEADER_CLANG true
#else
    #define SIMD_HEADER_CLANG false
#endif

#if defined (__GNUG__) && !defined (__clang__)
    #define SIMD_HEADER_GNUG true
#else
    #define SIMD_HEADER_GNUG false
#endif

#if __cplusplus < 201103L
    #error "simd implementation requires C++11 support"
#endif

#if __cplusplus >= 201402L
    #define cpp14_constexpr constexpr
#else
    #define cpp14_constexpr
#endif

#if __cplusplus > 201402L
    #define cpp17_constexpr constexpr
#else
    #define cpp17_constexpr
#endif

/* -- Implementation Notes --
 *  Vector type specializations:
 *
 *  Due to implementation details in the clang C++ compiler, it is impossible to
 *  declare templated typedefs of vector extension types. Moreover, the
 *  declaration requires an integer literal for the `vector_size` attribute.
 *  Therefore we must create a list of all the possible specializations for the
 *  underlying vector types.
 *
 *  It should also be noted that we specialize for vector types which are
 *  technically smaller and larger than "true" SIMD vector types for any
 *  particular architecture. In particular, for each possible base type we
 *  provide vector types with lane counts: 1, 2, 4, 8, 16, 32, 64. This is to
 *  allow general mappings over SIMD vector types without fear that the
 *  resulting SIMD vector type does not have a defined vector type
 *  specialization.
 *
 *  This is okay since both Clang and GCC will synthesize instructions that are
 *  not present on the target architecture.
 *
 *  Alignment concerns:
 *
 *  It should also be noted that we provide alignment values equal to the size
 *  of each vector type. This is required to prevent penalties or exceptions for
 *  unaligned memory accesses on architectures supporting only aligned accesses
 *  for SIMD vector types. Often these alignment values will be larger than the
 *  value `alignof (std::max_align_t)`, and so with GCC in particular warnings
 *  of attribute alignment greater than `alignof (std::max_align_t)` will be
 *  emitted. I have taken the liberty to insert `#pragma GCC diagnostic
 *  push/pop/ignored "-Wattributes"` blocks around declarations using `alignas`
 *  to suppress these warnings (these are the only place `#pragma` blocks are
 *  used). They are not necessary and can be searched for and removed if
 *  desired.
 *
 *  ABI concerns:
 *
 *  Each SIMD vector type (depending on underlying type and lane count),
 *  requires backing either by a particular SIMD technology or synthesized
 *  instructions when no appropriate SIMD technology is available on the target
 *  architecture. In the case of the latter GCC may emit warnings about vector
 *  return types (in the `-Wpsabi` category). It may also be useful to the user
 *  of this library to explicitly enable the target SIMD technology they wish to
 *  use, this may be one of, but is not limited to: `-mmmx`, `-msse`, `-msse2`,
 *  `-msse3`, `-mssse3`, `-msse4`, `-msse4.{1,2}`, `-mavx`, `-mavx2`,
 *  `-mavx512{f,bw,cd,dq,er,fma,pf,vbmi,vl}`, `-mneon`.
 *
 *  Vector comparisons generated by GCC vs. those generated by Clang:
 *
 *  GCC vector comparison operations return vector types with lane type equal to
 *  the signed integer with size equal to the size of the original vector lane
 *  type. However, the result type of vector comparisons generated by Clang are
 *  `ext_vector_types` (OpenCL vector types). For that reason we must perform
 *  manual conversion of comparisons when Clang is being used.
 *
 *  Additionally, GCC vector comparisons return lane values equal to `0` or `-1`
 *  (strictly speaking a value of the appropriate lane type where all bits are
 *  set; i.e. `2's` complement `-1`), representing false and true, respectively,
 *  while Clang vector comparisons return lane values equal to `0` or `1`,
 *  representing false and true, respectively. For reasons of consistency I have
 *  decided to normalize the values to `0` and `1`, which means that the code
 *  compiled with GCC involves a greedily evaluated integer mask operation for
 *  each construction of a boolean SIMD type by an underlying SIMD vector (which
 *  are precisely those constructions produced by comparison methods). For this
 *  reason values of `0` and `1` represent false and true throughout this code,
 *  and should be used in client code when explicitly constructing boolean SIMD
 *  types instead of `0` and `-1`, regardless of compiler.
 *
 *  If it is necessary that `-1` be used as the truth value for compatibility
 *  reasons, there is a class static function for boolean SIMD types named
 *  `make_gcc_compatible` which saturates all bits in vector lanes with logical
 *  true values while leaving all bits unset in vector lanes with logical false
 *  values.  Note, when compiling with Clang this method assumes the lane values
 *  are either `0` or `1`. Incorrect behavior will occur if this precondition is
 *  broken.
 *
 *  General discussion of comparison methods:
 *
 *  For all SIMD types we have overloads of `operator==`, and `operator!=`. For
 *  SIMD types other than boolean types we also have overloads of `operator<`,
 *  `operator>`, `operator<=`, and `operator>=`.
 *
 *  The same applies, of course, to floating point SIMD types. The overloads are
 *  implemented in the naive way and, therefore, should probably not be used in
 *  production code except when strict comparisons are actually desired. More
 *  general and robust floating point comparisons can be easily implemented on
 *  top of the primitive operations provided therein.
 *
 *  `namespace simd` implementations of `namespace std` mathematical functions:
 *  We provide lane-by-lane overloads for the mathematical methods in the
 *  standard headers `<cstdlib>` and `<cmath>`. Planned extensions include
 *  lane-by-lane overloads in the `namespace simd` for those methods specified
 *  in the Special Math TR, merged into the C++ standard as of C++17.
 *
 *  Use of simd::shuffle ():
 *  Clang's `__builtin_shufflevector` requires constant integer indices, and
 *  hence we must implement the function by hand for the general case when this
 *  header is compiled with Clang. For the user of this library this limitation
 *  can be overcome by using the .data () method, which provides access to the
 *  underlying SIMD vector type.
 *
 *  We provide the following overloads for standard library methods:
 *  - `operator<<` (narrow and wide character streams)
 *  - `operator>>` (narrow and wide character streams)
 *  - std::hash
 *
 *  It should be noted that std::hash computes a single value of type
 *  `std::size_t`, while `simd::hash` computes hash values lane-by-lane.
 */

namespace simd
{
namespace detail
{
namespace util
{
#if __cplusplus >= 201402L
    template <std::size_t ... I>
    using index_sequence = std::index_sequence <I...>;

    template <std::size_t N>
    using make_index_sequence = std::make_index_sequence <N>;
#else
    template <std::size_t ... I>
    struct index_sequence
    {
        using type = index_sequence;
        using value_type = std::size_t;

        static constexpr std::size_t size (void) noexcept
        {
            return sizeof... (I);
        }
    };

    template <typename, typename>
    struct merge;

    template <std::size_t ... I1, std::size_t ... I2>
    struct merge <index_sequence <I1...>, index_sequence <I2...>>
        : index_sequence <I1..., (sizeof... (I1) + I2)...>
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

    template <std::size_t L>
    struct lane_tag {};

    /*
     * Implemented for use in custom new implementation;
     * this method is threasafe, and consequently calls to new
     * on SIMD vector types are threadsafe as well (this becomes
     * a concern only in failing cases for memory allocation, which
     * typically will not occur on modern OSs that have overcommit
     * semantics).
     */
    inline void attempt_global_new_handler_call (void)
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
    inline void * aligned_allocate (std::size_t size, std::size_t alignment)
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
    inline void aligned_deallocate (void *p , std::size_t size, std::size_t alignment)
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
    inline void hash_combine (std::size_t & seed, T const & t) noexcept
    {
        std::hash <T> hfn {};
        seed ^= hfn (t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    /*
     * Non-modifying hash combine for specialization of std::hash for SIMD
     * vector types.
     */
    template <typename T>
    inline std::size_t hash_combine (std::size_t const & seed, T const & t)
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
#if SIMD_HEADER_CLANG
    inline void hash_combine (std::size_t & seed, __int128_t const & t) noexcept
#elif SIMD_HEADER_GNUG
    inline void hash_combine (std::size_t & seed, __int128 const & t) noexcept
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
#if SIMD_HEADER_CLANG
    inline std::size_t hash_combine (std::size_t const & seed, __int128_t const & t)
        noexcept
#elif SIMD_HEADER_GNUG
    inline std::size_t hash_combine (std::size_t const & seed, __int128 const & t)
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
#if SIMD_HEADER_CLANG
    inline void hash_combine (std::size_t & seed, __uint128_t const & t) noexcept
#elif SIMD_HEADER_GNUG
    inline void hash_combine (std::size_t & seed, unsigned __int128 const & t) noexcept
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
#if SIMD_HEADER_CLANG
    inline std::size_t hash_combine (std::size_t const & seed, __uint128_t const & t)
        noexcept
#elif SIMD_HEADER_GNUG
    inline std::size_t hash_combine (std::size_t const & seed,
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

namespace vext
{
    template <typename, std::size_t, typename enable = void>
    struct vector_type_specialization;

template <std::size_t lanes>
struct vector_type_specialization <signed char, lanes>
    : public vector_type_specialization <char, lanes>
{};

#define vsize(lanes, size) vector_size ((lanes) * (size))

#define specialize(ty, lanes) template <>\
struct vector_type_specialization <ty, lanes>\
{\
    typedef ty type __attribute__ ((vsize(lanes, sizeof (ty))));\
    static constexpr std::size_t alignment = lanes * alignof (ty);\
    static constexpr std::size_t size = lanes * sizeof (ty);\
};

    specialize(char, 1)
    specialize(char, 2)
    specialize(char, 4)
    specialize(char, 8)
    specialize(char, 16)
    specialize(char, 32)
    specialize(char, 64)

    specialize(unsigned char, 1)
    specialize(unsigned char, 2)
    specialize(unsigned char, 4)
    specialize(unsigned char, 8)
    specialize(unsigned char, 16)
    specialize(unsigned char, 32)
    specialize(unsigned char, 64)

    specialize(short, 1)
    specialize(short, 2)
    specialize(short, 4)
    specialize(short, 8)
    specialize(short, 16)
    specialize(short, 32)
    specialize(short, 64)

    specialize(unsigned short, 1)
    specialize(unsigned short, 2)
    specialize(unsigned short, 4)
    specialize(unsigned short, 8)
    specialize(unsigned short, 16)
    specialize(unsigned short, 32)
    specialize(unsigned short, 64)

    specialize(int, 1)
    specialize(int, 2)
    specialize(int, 4)
    specialize(int, 8)
    specialize(int, 16)
    specialize(int, 32)
    specialize(int, 64)

    specialize(unsigned int, 1)
    specialize(unsigned int, 2)
    specialize(unsigned int, 4)
    specialize(unsigned int, 8)
    specialize(unsigned int, 16)
    specialize(unsigned int, 32)
    specialize(unsigned int, 64)

    specialize(long, 1)
    specialize(long, 2)
    specialize(long, 4)
    specialize(long, 8)
    specialize(long, 16)
    specialize(long, 32)
    specialize(long, 64)

    specialize(unsigned long, 1)
    specialize(unsigned long, 2)
    specialize(unsigned long, 4)
    specialize(unsigned long, 8)
    specialize(unsigned long, 16)
    specialize(unsigned long, 32)
    specialize(unsigned long, 64)

    specialize(long long, 1)
    specialize(long long, 2)
    specialize(long long, 4)
    specialize(long long, 8)
    specialize(long long, 16)
    specialize(long long, 32)
    specialize(long long, 64)

    specialize(unsigned long long, 1)
    specialize(unsigned long long, 2)
    specialize(unsigned long long, 4)
    specialize(unsigned long long, 8)
    specialize(unsigned long long, 16)
    specialize(unsigned long long, 32)
    specialize(unsigned long long, 64)

    specialize(float, 1)
    specialize(float, 2)
    specialize(float, 4)
    specialize(float, 8)
    specialize(float, 16)
    specialize(float, 32)
    specialize(float, 64)

    specialize(double, 1)
    specialize(double, 2)
    specialize(double, 4)
    specialize(double, 8)
    specialize(double, 16)
    specialize(double, 32)
    specialize(double, 64)

    specialize(long double, 1)
    specialize(long double, 2)
    specialize(long double, 4)
    specialize(long double, 8)
    specialize(long double, 16)
    specialize(long double, 32)
    specialize(long double, 64)

#undef specialize

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __int128_t,
#elif SIMD_HEADER_GNUG
        __int128,
#endif
        1
    >
    {
#if SIMD_HEADER_CLANG
        typedef __int128_t type
            __attribute__ ((vsize (1, sizeof (__int128_t))));
        static constexpr std::size_t alignment = 1 * alignof (__int128_t);
        static constexpr std::size_t size      = 1 * sizeof (__int128_t);
#elif SIMD_HEADER_GNUG
        typedef __int128 type __attribute__ ((vsize (1, sizeof (__int128))));
        static constexpr std::size_t alignment = 1 * alignof (__int128);
        static constexpr std::size_t size      = 1 * sizeof (__int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __int128_t,
#elif SIMD_HEADER_GNUG
        __int128,
#endif
        2
    >
    {
#if SIMD_HEADER_CLANG
        typedef __int128_t type
            __attribute__ ((vsize (2, sizeof (__int128_t))));
        static constexpr std::size_t alignment = 2 * alignof (__int128_t);
        static constexpr std::size_t size      = 2 * sizeof (__int128_t);
#elif SIMD_HEADER_GNUG
        typedef __int128 type __attribute__ ((vsize (2, sizeof (__int128))));
        static constexpr std::size_t alignment = 2 * alignof (__int128);
        static constexpr std::size_t size      = 2 * sizeof (__int128);
#endif
    };

template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __int128_t,
#elif SIMD_HEADER_GNUG
        __int128,
#endif
        4
    >
    {
#if SIMD_HEADER_CLANG
        typedef __int128_t type
            __attribute__ ((vsize (4, sizeof (__int128_t))));
        static constexpr std::size_t alignment = 4 * alignof (__int128_t);
        static constexpr std::size_t size      = 4 * sizeof (__int128_t);
#elif SIMD_HEADER_GNUG
        typedef __int128 type __attribute__ ((vsize (4, sizeof (__int128))));
        static constexpr std::size_t alignment = 4 * alignof (__int128);
        static constexpr std::size_t size      = 4 * sizeof (__int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __int128_t,
#elif SIMD_HEADER_GNUG
        __int128,
#endif
        8
    >
    {
#if SIMD_HEADER_CLANG
        typedef __int128_t type
            __attribute__ ((vsize (8, sizeof (__int128_t))));
        static constexpr std::size_t alignment = 8 * alignof (__int128_t);
        static constexpr std::size_t size      = 8 * sizeof (__int128_t);
#elif SIMD_HEADER_GNUG
        typedef __int128 type __attribute__ ((vsize (8, sizeof (__int128))));
        static constexpr std::size_t alignment = 8 * alignof (__int128);
        static constexpr std::size_t size      = 8 * sizeof (__int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __int128_t,
#elif SIMD_HEADER_GNUG
        __int128,
#endif
        16
    >
    {
#if SIMD_HEADER_CLANG
        typedef __int128_t type
            __attribute__ ((vsize (16, sizeof (__int128_t))));
        static constexpr std::size_t alignment = 16 * alignof (__int128_t);
        static constexpr std::size_t size      = 16 * sizeof (__int128_t);
#elif SIMD_HEADER_GNUG
        typedef __int128 type __attribute__ ((vsize (16, sizeof (__int128))));
        static constexpr std::size_t alignment = 16 * alignof (__int128);
        static constexpr std::size_t size      = 16 * sizeof (__int128);
#endif
    };

template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __int128_t,
#elif SIMD_HEADER_GNUG
        __int128,
#endif
        32
    >
    {
#if SIMD_HEADER_CLANG
        typedef __int128_t type
            __attribute__ ((vsize (32, sizeof (__int128_t))));
        static constexpr std::size_t alignment = 32 * alignof (__int128_t);
        static constexpr std::size_t size      = 32 * sizeof (__int128_t);
#elif SIMD_HEADER_GNUG
        typedef __int128 type __attribute__ ((vsize (32, sizeof (__int128))));
        static constexpr std::size_t alignment = 32 * alignof (__int128);
        static constexpr std::size_t size      = 32 * sizeof (__int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __int128_t,
#elif SIMD_HEADER_GNUG
        __int128,
#endif
        64
    >
    {
#if SIMD_HEADER_CLANG
        typedef __int128_t type
            __attribute__ ((vsize (64, sizeof (__int128_t))));
        static constexpr std::size_t alignment = 64 * alignof (__int128_t);
        static constexpr std::size_t size      = 64 * sizeof (__int128_t);
#elif SIMD_HEADER_GNUG
        typedef __int128 type __attribute__ ((vsize (64, sizeof (__int128))));
        static constexpr std::size_t alignment = 64 * alignof (__int128);
        static constexpr std::size_t size      = 64 * sizeof (__int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __uint128_t,
#elif SIMD_HEADER_GNUG
        unsigned __int128,
#endif
        1
    >
    {
#if SIMD_HEADER_CLANG
        typedef __uint128_t type
            __attribute__ ((vsize (1, sizeof (__uint128_t))));
        static constexpr std::size_t alignment = 1 * alignof (__uint128_t);
        static constexpr std::size_t size      = 1 * sizeof (__uint128_t);
#elif SIMD_HEADER_GNUG
        typedef unsigned __int128 type
            __attribute__ ((vsize (1, sizeof (unsigned __int128))));
        static constexpr std::size_t alignment
            = 1 * alignof (unsigned __int128);
        static constexpr std::size_t size = 1 * sizeof (unsigned __int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __uint128_t,
#elif SIMD_HEADER_GNUG
        unsigned __int128,
#endif
        2
    >
    {
#if SIMD_HEADER_CLANG
        typedef __uint128_t type
            __attribute__ ((vsize (2, sizeof (__uint128_t))));
        static constexpr std::size_t alignment = 2 * alignof (__uint128_t);
        static constexpr std::size_t size      = 2 * sizeof (__uint128_t);
#elif SIMD_HEADER_GNUG
        typedef unsigned __int128 type
            __attribute__ ((vsize (2, sizeof (unsigned __int128))));
        static constexpr std::size_t alignment
            = 2 * alignof (unsigned __int128);
        static constexpr std::size_t size = 2 * sizeof (unsigned __int128);
#endif
    };

template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __uint128_t,
#elif SIMD_HEADER_GNUG
        unsigned __int128,
#endif
        4
    >
    {
#if SIMD_HEADER_CLANG
        typedef __uint128_t type
            __attribute__ ((vsize (4, sizeof (__uint128_t))));
        static constexpr std::size_t alignment = 4 * alignof (__uint128_t);
        static constexpr std::size_t size      = 4 * sizeof (__uint128_t);
#elif SIMD_HEADER_GNUG
        typedef unsigned __int128 type
            __attribute__ ((vsize (4, sizeof (unsigned __int128))));
        static constexpr std::size_t alignment
            = 4 * alignof (unsigned __int128);
        static constexpr std::size_t size = 4 * sizeof (unsigned __int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __uint128_t,
#elif SIMD_HEADER_GNUG
        unsigned __int128,
#endif
        8
    >
    {
#if SIMD_HEADER_CLANG
        typedef __uint128_t type
            __attribute__ ((vsize (8, sizeof (__uint128_t))));
        static constexpr std::size_t alignment = 8 * alignof (__uint128_t);
        static constexpr std::size_t size      = 8 * sizeof (__uint128_t);
#elif SIMD_HEADER_GNUG
        typedef unsigned __int128 type
            __attribute__ ((vsize (8, sizeof (unsigned __int128))));
        static constexpr std::size_t alignment
            = 8 * alignof (unsigned __int128);
        static constexpr std::size_t size = 8 * sizeof (unsigned __int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __uint128_t,
#elif SIMD_HEADER_GNUG
        unsigned __int128,
#endif
        16
    >
    {
#if SIMD_HEADER_CLANG
        typedef __uint128_t type
            __attribute__ ((vsize (16, sizeof (__uint128_t))));
        static constexpr std::size_t alignment = 16 * alignof (__uint128_t);
        static constexpr std::size_t size      = 16 * sizeof (__uint128_t);
#elif SIMD_HEADER_GNUG
        typedef unsigned __int128 type
            __attribute__ ((vsize (16, sizeof (unsigned __int128))));
        static constexpr std::size_t alignment
            = 16 * alignof (unsigned __int128);
        static constexpr std::size_t size = 16 * sizeof (unsigned __int128);
#endif
    };

template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
        __uint128_t,
#elif SIMD_HEADER_GNUG
        unsigned __int128,
#endif
        32
    >
    {
#if SIMD_HEADER_CLANG
        typedef __uint128_t type
            __attribute__ ((vsize (32, sizeof (__uint128_t))));
        static constexpr std::size_t alignment
            = 32 * alignof (__uint128_t);
        static constexpr std::size_t size = 32 * sizeof (__uint128_t);
#elif SIMD_HEADER_GNUG
        typedef unsigned __int128 type
            __attribute__ ((vsize (32, sizeof (unsigned __int128))));
        static constexpr std::size_t alignment
            = 32 * alignof (unsigned __int128);
        static constexpr std::size_t size = 32 * sizeof (unsigned __int128);
#endif
    };

    template <>
    struct vector_type_specialization <
#if SIMD_HEADER_CLANG
         __uint128_t,
#elif SIMD_HEADER_GNUG
        unsigned __int128,
#endif
        64
    >
    {
#if SIMD_HEADER_CLANG
        typedef __uint128_t type
            __attribute__ ((vsize (64, sizeof (__uint128_t))));
        static constexpr std::size_t alignment = 64 * alignof (__uint128_t);
        static constexpr std::size_t size      = 64 * sizeof (__uint128_t);
#elif SIMD_HEADER_GNUG
        typedef unsigned __int128 type
            __attribute__ ((vsize (64, sizeof (unsigned __int128))));
        static constexpr std::size_t alignment
            = 64 * alignof (unsigned __int128);
        static constexpr std::size_t size = 64 * sizeof (unsigned __int128);
#endif
    };

#undef vsize

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
                        sizeof (T) == 16 ||
                        sizeof (T) == 12 ||
                        sizeof (T) == 10,
#if SIMD_HEADER_CLANG
                        __int128_t,
#elif SIMD_HEADER_GNUG
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
                        sizeof (T) == 16 ||
                        sizeof (T) == 12 ||
                        sizeof (T) == 10,
#if SIMD_HEADER_CLANG
                        __uint128_t,
#elif SIMD_HEADER_GNUG
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
#if SIMD_HEADER_CLANG
        static_assert (
            std::is_arithmetic <T>::value ||
                std::is_same <T, __int128_t>::value ||
                std::is_same <T, __uint128_t>::value,
            "template parameter typename T must be an arithmetic type"
        );
#elif SIMD_HEADER_GNUG
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

        template <typename U, std::size_t ... L>
        static constexpr vector_type_impl
            extend_impl (U const & u, util::index_sequence <L...>) noexcept
        {
            return vector_type_impl {((void) L, u)...};
        }

    protected:
        static constexpr vector_type_impl
            unpack (base_value_type const (&arr) [lanes]) noexcept
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

        template <typename U>
        static constexpr vector_type_impl extend (U const & u) noexcept
        {
            return extend_impl (u, util::make_index_sequence <lanes> {});
        }

        template <
            typename ... Ts,
            typename = typename std::enable_if <
                sizeof... (Ts) == lanes && lanes != 1
            >::type
        >
        static constexpr vector_type_impl extend (Ts const & ... ts) noexcept
        {
            return vector_type_impl {static_cast <base_value_type> (ts)...};
        }

    private:
        template <typename ResultVector, typename Op, std::size_t ... L>
        static constexpr ResultVector apply_impl (vector_type_impl const & v,
                                                  Op op,
                                                  util::index_sequence <L...>)
            noexcept
        {
            return ResultVector {op (v [L])...};
        }

        template <typename ResultVector, typename Op, std::size_t ... L>
        static constexpr ResultVector apply_impl (vector_type_impl const & u,
                                                  vector_type_impl const & v,
                                                  Op op,
                                                  util::index_sequence <L...>)
            noexcept
        {
            return ResultVector {op (u [L], v [L])...};
        }

    protected:
        template <typename ResultVector, typename Op>
        static constexpr ResultVector
            apply_op (vector_type_impl const & v, Op op) noexcept
        {
            return apply_impl <ResultVector> (
                v, op, util::make_index_sequence <lanes> {}
            );
        }

        template <typename ResultVector, typename Op>
        static constexpr ResultVector apply_op (vector_type_impl const & u,
                                                vector_type_impl const & v,
                                                Op op) noexcept
        {
            return apply_impl <ResultVector> (
                u, v, op, util::make_index_sequence <lanes> {}
            );
        }

        template <typename vec_to, typename valtype, typename vec_from>
        static cpp14_constexpr vec_to vector_convert (vec_from const & v)
            noexcept
        {
            using from_valtype = typename std::remove_reference <
                decltype (v [0])
            >::type;

            return apply_op <vec_to> (
                v,
                [] (from_valtype const & val) cpp17_constexpr noexcept
                    -> valtype
                {
                    return static_cast <valtype> (val);
                }
            );
        }

        /*
         * This is a proxy reference object to avoid undefined behavior and
         * type-punning in derived SIMD type classes. It is the returned
         * type from methds such as operator[] and at().
         */
        template <typename VecType, typename ValType>
        class reference_proxy;

    public:
        using reference = reference_proxy <
            vector_type_impl, base_value_type
        >;
        using const_reference = reference_proxy <
            vector_type_impl const, base_value_type const
        >;

    protected:
        /*
         * This is a proxy pointer object to avoid undefined behavior and
         * type-punning in derived SIMD type classes. It is the returned
         * type from methds such as {c}{r}begin and {c}{r}end.
         */
        template <typename VecType, typename ValType>
        class pointer_proxy;

    public:
        using pointer = pointer_proxy <
            vector_type_impl, base_value_type
        >;
        using const_pointer = pointer_proxy <
            vector_type_impl const, base_value_type const
        >;

    protected:
        template <typename VecType, typename ValType>
        class reference_proxy
        {
        private:
            using vector_type = VecType;
            using value_type  = ValType;
            using pointer     = pointer_proxy <vector_type, value_type>;
            using vecpointer  = typename std::add_pointer <vector_type>::type;

            vecpointer _ref;
            std::ptrdiff_t _index;

        public:
            reference_proxy (void) = delete;
            ~reference_proxy (void) noexcept = default;

            constexpr reference_proxy (vecpointer p, std::ptrdiff_t index)
                noexcept
                : _ref   {p}
                , _index {index}
            {}

            constexpr
                reference_proxy (vector_type & v, std::ptrdiff_t index)
                noexcept
                : _ref   {&v}
                , _index {index}
            {}

            constexpr reference_proxy (reference_proxy const &) noexcept
                = default;

            template <typename U>
            reference_proxy & operator= (U && u) noexcept
            {
                static_assert (
                    std::is_convertible <U, value_type>::value,
                    "cannot assign to vector lane from non-convertible type"
                );

                (*this->_ref) [this->_index] = static_cast <value_type> (
                    std::forward <U> (u)
                );
                return *this;
            }

            reference_proxy & operator= (reference_proxy const & r) noexcept
            {
                this->_ref = r._ref;
                this->_index = r._index;
                return *this;
            }

            void swap (reference_proxy & r) noexcept
            {
                std::swap (this->_ref, r._ref);
                std::swap (this->_index, r._index);
            }

            template <typename U>
            constexpr operator U (void) const noexcept
            {
                static_assert (
                    std::is_convertible <value_type, U>::value,
                    "cannot perform cast"
                );

                return static_cast <U> ((*this->_ref) [this->_index]);
            }

            value_type data (void) const noexcept
            {
                return (*this->_ref) [this->_index];
            }

            pointer operator& (void) const noexcept
            {
                return pointer {this->_ref, this->_index};
            }

            constexpr bool operator== (reference_proxy const & r) const noexcept
            {
                return (*this->_ref) [this->_index] == (*r._ref) [r._index];
            }

            constexpr bool operator!= (reference_proxy const & r) const noexcept
            {
                return (*this->_ref) [this->_index] != (*r._ref) [r._index];
            }

            constexpr bool operator> (reference_proxy const & r) const noexcept
            {
                return (*this->_ref) [this->_index] > (*r._ref) [r._index];
            }

            constexpr bool operator< (reference_proxy const & r) const noexcept
            {
                return (*this->_ref) [this->_index] < (*r._ref) [r._index];
            }

            constexpr bool operator>= (reference_proxy const & r) const noexcept
            {
                return (*this->_ref) [this->_index] >= (*r._ref) [r._index];
            }

            constexpr bool operator<= (reference_proxy const & r) const noexcept
            {
                return (*this->_ref) [this->_index] <= (*r._ref) [r._index];
            }
        };

        template <typename VecType, typename ValType>
        class pointer_proxy
        {
        private:
            using vector_type = VecType;
            using value_type  = ValType;
            using vecpointer  = typename std::add_pointer <vector_type>::type;
            using reference   = reference_proxy <vector_type, value_type>;

            vecpointer _pointer;
            std::ptrdiff_t _index;

        public:
            using iterator_category = std::random_access_iterator_tag;

            constexpr pointer_proxy (void) noexcept
                : _pointer {nullptr}
                , _index   {0}
            {}

            ~pointer_proxy (void) noexcept = default;

            constexpr pointer_proxy (vecpointer p, std::ptrdiff_t index)
                noexcept
                : _pointer {p}
                , _index   {index}
            {}

            constexpr pointer_proxy (vector_type & v, std::ptrdiff_t index)
                noexcept
                : _pointer {&v}
                , _index   {index}
            {}

            constexpr pointer_proxy (pointer_proxy const &) noexcept = default;

            cpp14_constexpr pointer_proxy & operator= (pointer_proxy p)
                noexcept
            {
                this->_pointer = p._pointer;
                this->_index = p._index;
                return *this;
            }

            operator vecpointer (void) noexcept
            {
                return this->_pointer;
            }

            operator bool (void) noexcept
            {
                return static_cast <bool> (this->_pointer);
            }

            reference operator* (void) const noexcept
            {
                return reference {this->_pointer, this->_index};
            }

            reference operator-> (void) const noexcept
            {
                return reference {this->_pointer, this->_index};
            }

            reference operator[] (std::ptrdiff_t n) const noexcept
            {
                return reference {this->_pointer, this->_index + n};
            }

            pointer_proxy & operator++ (void) noexcept
            {
                this->_index += 1;
                return *this;
            }

            pointer_proxy & operator-- (void) noexcept
            {
                this->_index -= 1;
                return *this;
            }

            pointer_proxy operator++ (int) noexcept
            {
                auto const tmp = *this;
                this->_index += 1;
                return tmp;
            }

            pointer_proxy operator-- (int) noexcept
            {
                auto const tmp = *this;
                this->_index -= 1;
                return tmp;
            }

            pointer_proxy & operator+= (std::ptrdiff_t n) noexcept
            {
                this->_index += n;
                return *this;
            }

            pointer_proxy & operator-= (std::ptrdiff_t n) noexcept
            {
                this->_index -= n;
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
                return this->_index - p._index;
            }

            bool operator== (pointer_proxy p) const noexcept
            {
                return this->_pointer == p._pointer &&
                       this->_index == p._index;
            }

            bool operator!= (pointer_proxy p) const noexcept
            {
                return this->_pointer != p._pointer ||
                       this->_index != p._index;
            }

            bool operator< (pointer_proxy p) const noexcept
            {
                return this->_pointer < p._pointer ||
                       (this->_pointer == p._pointer &&
                        this->_index < p._index);
            }

            bool operator> (pointer_proxy p) const noexcept
            {
                return this->_pointer > p._pointer ||
                       (this->_pointer == p._pointer &&
                        this->_index > p._index);
            }

            bool operator<= (pointer_proxy p) const noexcept
            {
                return this->_pointer < p._pointer ||
                       (this->_pointer == p._pointer &&
                        this->_index <= p._index);
            }

            bool operator>= (pointer_proxy p) const noexcept
            {
                return this->_pointer > p._pointer ||
                       (this->_pointer == p._pointer &&
                        this->_index >= p._index);
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

    template <typename T>
    struct is_complex_scalar : std::false_type {};

    template <typename T>
    struct is_complex_scalar <std::complex <T>> : std::true_type {};

    template <typename T>
    struct is_complex_scalar <T const> : is_complex_scalar <T> {};

    template <typename T>
    struct is_complex_scalar <T &> : is_complex_scalar <T> {};

    template <typename T>
    struct is_complex_scalar <T const &> : is_complex_scalar <T> {};

    template <typename T>
    struct extract_template_type
    {
        using type = T;
    };

    template <template <typename> class T, typename U>
    struct extract_template_type <T <U>>
    {
        using type = U;
    };

    template <typename T>
    struct extract_template_type <T const> : extract_template_type <T> {};

    template <typename T>
    struct extract_template_type <T &> : extract_template_type <T> {};

    template <typename T>
    struct extract_template_type <T const &> : extract_template_type <T> {};

    template <typename T>
    struct is_boolean_scalar : std::false_type {};

    template <>
    struct is_boolean_scalar <bool> : std::true_type {};

    template <typename T>
    struct is_boolean_scalar <T const> : is_boolean_scalar <T> {};

    template <typename T>
    struct is_boolean_scalar <T &> : is_boolean_scalar <T> {};

    template <typename T>
    struct is_boolean_scalar <T const &> : is_boolean_scalar <T> {};

    template <typename T, std::size_t lanes, typename tag = arithmetic_tag>
    using simd_type = typename std::conditional <
        (std::is_integral <T>::value &&
            std::is_same <tag, arithmetic_tag>::value) ||
#if SIMD_HEADER_CLANG
        (std::is_same <T, __int128_t>::value &&
            std::is_same <tag, arithmetic_tag>::value)||
        (std::is_same <T, __uint128_t>::value &&
            std::is_same <tag, arithmetic_tag>::value),
#elif SIMD_HEADER_GNUG
        (std::is_same <T, __int128>::value &&
            std::is_same <tag, arithmetic_tag>::value)||
        (std::is_same <T, unsigned __int128>::value &&
            std::is_same <tag, arithmetic_tag>::value),
#endif
        integral_simd_type <T, lanes>,
        typename std::conditional <
            (std::is_integral <T>::value &&
                std::is_same <tag, boolean_tag>::value) ||
#if SIMD_HEADER_CLANG
            (std::is_same <T, __int128_t>::value &&
                std::is_same <tag, boolean_tag>::value) ||
            (std::is_same <T, __uint128_t>::value &&
                std::is_same <tag, boolean_tag>::value),
#elif SIMD_HEADER_GNUG
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
                    typename std::conditional <
                        is_complex_scalar <T>::value &&
                            (std::is_same <tag, complex_tag>::value ||
                             std::is_same <tag, arithmetic_tag>::value),
                        complex_simd_type <
                            typename extract_template_type <T>::type, lanes
                        >,
                        void
                    >::type
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

    template <typename SIMDType>
    struct is_simd_type <SIMDType const> : is_simd_type <SIMDType> {};

    template <typename SIMDType>
    struct is_simd_type <SIMDType &> : is_simd_type <SIMDType> {};

    template <typename SIMDType>
    struct is_simd_type <SIMDType const &> : is_simd_type <SIMDType> {};

    template <typename SIMDType>
    struct is_simd_type <SIMDType &&> : is_simd_type <SIMDType> {};

    template <typename SIMDType>
    struct is_simd_type <SIMDType const &&> : is_simd_type <SIMDType> {};

    template <typename T, std::size_t l, typename tag>
    struct simd_traits_base
    {
        using base                   = simd_type_base <T, l>;
        using vector_type            = typename base::vector_type_impl;
        using value_type             = T;
        using integral_type          = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type   = typename base::signed_integral_type;
        using reference              = typename base::reference;
        using const_reference        = typename base::const_reference;
        using iterator               = typename base::pointer;
        using const_iterator         = typename base::const_pointer;
        using reverse_iterator       = std::reverse_iterator <iterator>;
        using const_reverse_iterator = std::reverse_iterator <const_iterator>;
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
        using lane_type              = T;
        using real_simd_type         = fp_simd_type <T, l>;
        using imag_simd_type         = fp_simd_type <T, l>;
        using integral_type          = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type   = typename base::signed_integral_type;
        using reference =
            typename complex_simd_type <T, l>::reference;
        using const_reference =
            typename complex_simd_type <T, l>::const_reference;
        using iterator =
            typename complex_simd_type <T, l>::iterator;
        using const_iterator =
            typename complex_simd_type <T, l>::const_iterator;
        using reverse_iterator =
            typename complex_simd_type <T, l>::reverse_iterator;
        using const_reverse_iterator =
            typename complex_simd_type <T, l>::const_reverse_iterator;
        using category_tag = complex_tag;

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
        using base      = simd_type_base <T, l>;
        using this_type = integral_simd_type <T, l>; 

        typename base::vector_type_impl _vec;

    public:
        static_assert (
            std::is_integral <T>::value ||
#if SIMD_HEADER_CLANG
            std::is_same <T, __int128_t>::value ||
            std::is_same <T, __uint128_t>::value,
#elif SIMD_HEADER_GNUG
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
        using reference              = typename base::reference;
        using const_reference        = typename base::const_reference;
        using iterator               = typename base::pointer;
        using const_iterator         = typename base::const_pointer;
        using reverse_iterator       = std::reverse_iterator <iterator>;
        using const_reverse_iterator = std::reverse_iterator <const_iterator>;
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
            integral_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *addr++;
            }
            return result;
        }

        static integral_simd_type load (value_type const * addr,
                                        std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;

            integral_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *use_addr++;
            }
            return result;
        }

        static integral_simd_type load (vector_type const * addr) noexcept
        {
            return integral_simd_type {*addr};
        }

        static integral_simd_type load (vector_type const * addr,
                                        std::ptrdiff_t off) noexcept
        {
            return integral_simd_type {*(addr + off)};
        }

        static integral_simd_type load_aligned (value_type const * addr)
            noexcept
        {
            auto aligned_ptr = static_cast <value_type const *> (
                __builtin_assume_aligned (addr, base::alignment)
            );

            integral_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *aligned_ptr++;
            }
            return result;
        }

        static integral_simd_type load_aligned (value_type const * addr,
                                                std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;
            auto aligned_ptr = static_cast <value_type const *> (
                __builtin_assume_aligned (use_addr, base::alignment)
            );

            integral_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *aligned_ptr++;
            }
            return result;
        }

        static integral_simd_type load_aligned (vector_type const * addr)
            noexcept
        {
            auto aligned_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (addr, base::alignment)
            );

            return integral_simd_type {*aligned_ptr};
        }

        static integral_simd_type load_aligned (vector_type const * addr,
                                                std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;
            auto aligned_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (use_addr, base::alignment)
            );

            return integral_simd_type {*aligned_ptr};
        }

        constexpr integral_simd_type (void) noexcept
            : _vec {base::extend (value_type {})}
        {}

        constexpr integral_simd_type (vector_type const & vec) noexcept
            : _vec {vec}
        {}

        explicit constexpr integral_simd_type (value_type const & val) noexcept
            : _vec {base::extend (val)}
        {}

        template <
            typename ... value_types,
            typename = typename std::enable_if <
                sizeof... (value_types) == lanes && lanes != 1
            >::type
        >
        explicit constexpr integral_simd_type (value_types && ... vals) noexcept
            : _vec {
                static_cast <value_type> (std::forward <value_types> (vals))...
            }
        {}

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

        cpp14_constexpr
        integral_simd_type & operator= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec = sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            this->_vec = base::extend (val);
            return *this;
        }

    private:
        template <std::size_t ... L>
        constexpr std::array <value_type, lanes>
            to_array (util::index_sequence <L...>) const noexcept
        {
            return std::array <value_type, lanes> {{this->_vec [L]...}};
        }

    public:
        explicit constexpr operator std::array <value_type, lanes> (void) const
            noexcept
        {
            return this->to_array (util::make_index_sequence <lanes> {});
        }

    private:
        template <typename vec_to, typename valtype, typename vec_from>
        static cpp14_constexpr vec_to vector_convert (vec_from const & v)
            noexcept
        {
            return base::template vector_convert <vec_to, valtype> (v);
        }

    public:
        template <typename SIMDType>
        cpp14_constexpr SIMDType to (void) const noexcept
        {
            static_assert (
                is_simd_type <SIMDType>::value,
                "cannot perform cast to non-simd type"
            );

            using cast_traits = simd_traits <SIMDType>;
            using rebind_type = rebind <
                typename cast_traits::value_type,
                cast_traits::lanes,
                typename cast_traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;
            using rebind_value_type = typename rebind_type::value_type;

            static_assert (
                lanes == cast_traits::lanes,
                "cannot perform conversion of vector type to vector type with a"
                " different number of lanes"
            );

            return rebind_type {
                vector_convert <
                    rebind_vector_type, rebind_value_type, vector_type
                > (this->_vec)
            };
        }

        template <typename SIMDType>
        explicit constexpr operator SIMDType (void) const noexcept
        {
            return this->template to <SIMDType> ();
        }

        template <typename SIMDType>
        SIMDType as (void) const noexcept
        {
            static_assert (
                is_simd_type <SIMDType>::value,
                "cannot perform cast to non-simd type"
            );

            using cast_traits = simd_traits <SIMDType>;
            using rebind_type = rebind <
                typename cast_traits::value_type,
                cast_traits::lanes,
                typename cast_traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;

            static_assert (
                sizeof (vector_type) == sizeof (rebind_vector_type),
                "cannot reinterpret vector to differently sized vector type"
            );

            return rebind_type {
                reinterpret_cast <rebind_vector_type> (this->_vec)
            };
        }

        cpp14_constexpr void swap (integral_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        template <std::size_t N>
        cpp14_constexpr integral_simd_type & set (value_type const & val) &
            noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            this->_vec [N] = val;
            return *this;
        }

        cpp14_constexpr integral_simd_type &
            set (std::size_t n, value_type const & val) & noexcept
        {
            this->_vec [n] = val;
            return *this;
        }

        cpp14_constexpr integral_simd_type &
            set (std::initializer_list <value_type> vlist) & noexcept
        {
            auto lindex = vlist.begin ();
            for (std::size_t i = 0; i < std::min (lanes, vlist.size ()); ++i) {
                this->_vec [i] = *lindex++;
            }
        }

        cpp14_constexpr void fill (value_type const & val) & noexcept
        {
            this->_vec = base::extend (val);
        }

        cpp14_constexpr vector_type & data (void) & noexcept
        {
            return this->_vec;
        }

        constexpr vector_type const & data (void) const & noexcept
        {
            return this->_vec;
        }

        template <std::size_t N>
        constexpr const_reference get (void) const & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return const_reference {
                &this->_vec, static_cast <std::ptrdiff_t> (N)
            };
        }

        template <std::size_t N>
        cpp14_constexpr reference get (void) & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return reference {&this->_vec, static_cast <std::ptrdiff_t> (N)};
        }

        template <std::size_t N>
        constexpr value_type value (void) const noexcept
        {
            return this->_vec [N];
        }

        constexpr value_type value (std::size_t n) const noexcept
        {
            return this->_vec [n];
        }

        constexpr const_reference operator[] (std::size_t n) const & noexcept
        {
            return const_reference {
                &this->_vec, static_cast <std::ptrdiff_t> (n)
            };
        }

        cpp14_constexpr reference operator[] (std::size_t n) & noexcept
        {
            return reference {
                &this->_vec, static_cast <std::ptrdiff_t> (n)
            };
        }

        constexpr const_reference at (std::size_t n) const &
        {
            return n < lanes ?
                const_reference {this->_vec, n} :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        cpp14_constexpr reference at (std::size_t n) &
        {
            return n < lanes ?
                reference {this->_vec, n} :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        cpp14_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->data (), 0};
        }

        cpp14_constexpr iterator end (void) & noexcept
        {
            return iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->data (), 0};
        }

        constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->data (), 0};
        }

        constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        cpp14_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        cpp14_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->data (), 0};
        }

        constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->data (), 0};
        }

        constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->data (), 0};
        }

    private:
        template <typename VectorType, typename LaneType, std::size_t ... L>
        constexpr VectorType convert_vector (util::index_sequence <L...>) const
            noexcept
        {
            return VectorType {static_cast <LaneType> (this->_vec [L])...};
        }

    public:
        constexpr auto operator+ (void) const noexcept
            -> integral_simd_type <
                typename std::conditional <
                    std::is_same <value_type, char>::value ||
                    std::is_same <value_type, signed char>::value ||
                    std::is_same <value_type, unsigned char>::value,
                    int,
                    value_type
                >::type,
                lanes
            >
        {
            using lane_type = typename std::conditional <
                std::is_same <value_type, char>::value ||
                std::is_same <value_type, signed char>::value ||
                std::is_same <value_type, unsigned char>::value,
                int,
                value_type
            >::type;
            using result_type = integral_simd_type <lane_type, lanes>;
            using result_vector_type = typename result_type::vector_type;

            return result_type {
                this->template convert_vector <result_vector_type, lane_type> (
                    util::make_index_sequence <lanes> {}
                )
            };
        }

        constexpr integral_simd_type operator- (void) const noexcept
        {
            return integral_simd_type {-this->_vec};
        }

        cpp14_constexpr integral_simd_type & operator++ (void) noexcept
        {
            this->operator+ (1);
            return *this;
        }

        cpp14_constexpr integral_simd_type & operator-- (void) noexcept
        {
            this->operator- (1);
            return *this;
        }

        cpp14_constexpr integral_simd_type operator++ (int) noexcept
        {
            auto const tmp = *this;
            this->operator+ (1);
            return tmp;
        }

        cpp14_constexpr integral_simd_type operator-- (int) noexcept
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this + integral_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this - integral_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this * integral_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this / integral_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this % integral_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this & integral_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this | integral_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this ^ integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator! (void) const noexcept
        {
#if SIMD_HEADER_CLANG
            return boolean_simd_type <integral_type, lanes> {
                base::template apply_op <vector_type> (
                    this->_vec,
                    [] (value_type const & v) cpp17_constexpr -> value_type
                    {
                        return static_cast <value_type> (!v);
                    }
                )
            };
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {!this->_vec};
#endif
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator&& (integral_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return boolean_simd_type <integral_type, lanes> {
                base::template apply_op <vector_type> (
                    this->_vec,
                    sv._vec,
                    [] (value_type const & u, value_type const & v)
                        cpp17_constexpr -> value_type
                    {
                        return static_cast <value_type> (u && v);
                    }
                )
            };
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec && sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator&& (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this && integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator|| (integral_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return boolean_simd_type <integral_type, lanes> {
                base::template apply_op <vector_type> (
                    this->_vec,
                    sv._vec,
                    [] (value_type const & u, value_type const & v)
                        cpp17_constexpr -> value_type
                    {
                        return static_cast <value_type> (u || v);
                    }
                )
            };
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec || sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator|| (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this || integral_simd_type {val};
        }

        constexpr integral_simd_type operator<< (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec << sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator<< (U val)
            const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this << integral_simd_type {val};
        }

        constexpr integral_simd_type operator>> (integral_simd_type const & sv)
            const noexcept
        {
            return integral_simd_type {this->_vec >> sv._vec};
        }

        template <typename U>
        constexpr integral_simd_type operator>> (U val)
            const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this >> integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator+= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec += sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator+= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this += integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator-= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec -= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator-= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this -= integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator*= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec *= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator*= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this *= integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator/= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec /= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator/= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this /= integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator%= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec %= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator%= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this %= integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator&= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec &= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator&= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this &= integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator|= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec |= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator|= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this |= integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator^= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec ^= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator^= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this ^= integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator<<= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec <<= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator<<= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this <<= integral_simd_type {val};
        }

        cpp14_constexpr
        integral_simd_type & operator>>= (integral_simd_type const & sv) &
            noexcept
        {
            this->_vec >>= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr integral_simd_type & operator>>= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this >>= integral_simd_type {val};
        }

#if SIMD_HEADER_CLANG
    private:
        template <typename Comparison, std::size_t ... L>
        static constexpr boolean_simd_type <integral_type, lanes>
            unpack_comparison (Comparison && c, util::index_sequence <L...>)
            noexcept
        {
            return boolean_simd_type <integral_type, lanes> {
                std::forward <Comparison> (c) [L]...
            };
        }

    public:
#endif
        constexpr boolean_simd_type <integral_type, lanes>
            operator== (integral_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec == sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec == sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes> operator== (U val)
            const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (integral_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec != sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec != sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this != integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator> (integral_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec > sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec > sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator> (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this > integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator< (integral_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec < sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec < sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator< (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this < integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (integral_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec >= sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec >= sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this >= integral_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (integral_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec <= sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec <= sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
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
        using base      = simd_type_base <T, l>;
        using this_type = fp_simd_type <T, l>; 

        typename base::vector_type_impl _vec;

    public:
        static_assert (
            std::is_floating_point <T>::value,
            "template parameter typename T must be a floating point type"
        );

        using vector_type            = typename base::vector_type_impl;
        using value_type             = T;
        using integral_type          = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type   = typename base::signed_integral_type;
        using reference              = typename base::reference;
        using const_reference        = typename base::const_reference;
        using iterator               = typename base::pointer;
        using const_iterator         = typename base::const_pointer;
        using reverse_iterator       = std::reverse_iterator <iterator>;
        using const_reverse_iterator = std::reverse_iterator <const_iterator>;
        using category_tag           = arithmetic_tag;
        static constexpr std::size_t lanes = l;

        template <typename U, std::size_t L, typename tag>
        using rebind = simd_type <U, L, tag>;

        static fp_simd_type load (value_type const * addr) noexcept
        {
            fp_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *addr++;
            }
            return result;
        }

        static fp_simd_type load (value_type const * addr,
                                  std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;

            fp_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *use_addr++;
            }
            return result;
        }

        static fp_simd_type load (vector_type const * addr) noexcept
        {
            return fp_simd_type {*addr};
        }

        static fp_simd_type load (vector_type const * addr,
                                  std::ptrdiff_t off) noexcept
        {
            return fp_simd_type {*(addr + off)};
        }

        static fp_simd_type load_aligned (value_type const * addr)
            noexcept
        {
            auto aligned_ptr = static_cast <value_type const *> (
                __builtin_assume_aligned (addr, base::alignment)
            );

            fp_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *aligned_ptr++;
            }
            return result;
        }

        static fp_simd_type load_aligned (value_type const * addr,
                                          std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;
            auto aligned_ptr = static_cast <value_type const *> (
                __builtin_assume_aligned (use_addr, base::alignment)
            );

            fp_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *aligned_ptr++;
            }
            return result;
        }

        static fp_simd_type load_aligned (vector_type const * addr)
            noexcept
        {
            auto aligned_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (addr, base::alignment)
            );

            return fp_simd_type {*aligned_ptr};
        }

        static fp_simd_type load_aligned (vector_type const * addr,
                                          std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;
            auto aligned_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (use_addr, base::alignment)
            );

            return fp_simd_type {*aligned_ptr};
        }

        constexpr fp_simd_type (void) noexcept
            : _vec {base::extend (value_type {})}
        {}

        constexpr fp_simd_type (vector_type const & vec) noexcept
            : _vec {vec}
        {}

        explicit constexpr fp_simd_type (value_type const & val) noexcept
            : _vec {base::extend (val)}
        {}

        template <
            typename ... value_types,
            typename = typename std::enable_if <
                sizeof... (value_types) == lanes && lanes != 1
            >::type
        >
        explicit constexpr fp_simd_type (value_types && ... vals) noexcept
            : _vec {
                static_cast <value_type> (std::forward <value_types> (vals))...
            }
        {}

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

        cpp14_constexpr fp_simd_type &
            operator= (fp_simd_type const & sv) & noexcept
        {
            this->_vec = sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr fp_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this = fp_simd_type {val};
        }

    private:
        template <std::size_t ... L>
        constexpr std::array <value_type, lanes>
            to_array (util::index_sequence <L...>) const noexcept
        {
            return std::array <value_type, lanes> {{this->_vec [L]...}};
        }

    public:
        explicit constexpr operator std::array <value_type, lanes> (void) const
            noexcept
        {
            return this->to_array (util::make_index_sequence <lanes> {});
        }

    private:
        template <typename vec_to, typename valtype, typename vec_from>
        static cpp14_constexpr vec_to vector_convert (vec_from const & v)
            noexcept
        {
            return base::template vector_convert <vec_to, valtype> (v);
        }

    public:
        template <typename SIMDType>
        cpp14_constexpr SIMDType to (void) const noexcept
        {
            static_assert (
                is_simd_type <SIMDType>::value,
                "cannot perform cast to non-simd type"
            );

            using cast_traits = simd_traits <SIMDType>;
            using rebind_type = rebind <
                typename cast_traits::value_type,
                cast_traits::lanes,
                typename cast_traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;
            using rebind_value_type = typename rebind_type::value_type;

            static_assert (
                lanes == cast_traits::lanes,
                "cannot perform conversion of vector type to vector type with a"
                " different number of lanes"
            );

            return rebind_type {
                vector_convert <
                    rebind_vector_type, rebind_value_type, vector_type
                > (this->_vec)
            };
        }

        template <typename SIMDType>
        explicit constexpr operator SIMDType (void) const noexcept
        {
            return this->template to <SIMDType> ();
        }

        template <typename SIMDType>
        SIMDType as (void) const noexcept
        {
            static_assert (
                is_simd_type <SIMDType>::value,
                "cannot perform cast to non-simd type"
            );

            using cast_traits = simd_traits <SIMDType>;
            using rebind_type = rebind <
                typename cast_traits::value_type,
                cast_traits::lanes,
                typename cast_traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;

            static_assert (
                sizeof (vector_type) == sizeof (rebind_vector_type),
                "cannot reinterpret vector to differently sized vector type"
            );

            return rebind_type {
                reinterpret_cast <rebind_vector_type> (this->_vec)
            };
        }

        cpp14_constexpr void swap (fp_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        template <std::size_t N>
        cpp14_constexpr fp_simd_type & set (value_type const & val) & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            this->_vec [N] = val;
            return *this;
        }

        cpp14_constexpr fp_simd_type &
            set (std::size_t n, value_type const & val) & noexcept
        {
            this->_vec [n] = val;
            return *this;
        }

        cpp14_constexpr fp_simd_type &
            set (std::initializer_list <value_type> vlist) & noexcept
        {
            auto lindex = vlist.begin ();
            for (std::size_t i = 0; i < std::min (lanes, vlist.size ()); ++i) {
                this->_vec [i] = *lindex++;
            }
        }

        cpp14_constexpr void fill (value_type const & val) & noexcept
        {
            this->_vec = base::extend (val);
        }

        cpp14_constexpr vector_type & data (void) & noexcept
        {
            return this->_vec;
        }

        constexpr vector_type const & data (void) const & noexcept
        {
            return this->_vec;
        }

        template <std::size_t N>
        constexpr const_reference get (void) const & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return const_reference {
                &this->_vec, static_cast <std::ptrdiff_t> (N)
            };
        }

        template <std::size_t N>
        cpp14_constexpr reference get (void) & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return reference {&this->_vec, static_cast <std::ptrdiff_t> (N)};
        }

        template <std::size_t N>
        constexpr value_type value (void) const noexcept
        {
            return this->_vec [N];
        }

        constexpr value_type value (std::size_t n) const noexcept
        {
            return this->_vec [n];
        }

        constexpr const_reference operator[] (std::size_t n) const & noexcept
        {
            return const_reference {
                &this->_vec, static_cast <std::ptrdiff_t> (n)
            };
        }

        cpp14_constexpr reference operator[] (std::size_t n) & noexcept
        {
            return reference {
                &this->_vec, static_cast <std::ptrdiff_t> (n)
            };
        }

        constexpr const_reference at (std::size_t n) const &
        {
            return n < lanes ?
                const_reference {this->_vec, n} :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        cpp14_constexpr reference at (std::size_t n) &
        {
            return n < lanes ?
                reference {this->_vec, n} :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        cpp14_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->data (), 0};
        }

        cpp14_constexpr iterator end (void) & noexcept
        {
            return iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->data (), 0};
        }

        constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->data (), 0};
        }

        constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        cpp14_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        cpp14_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->data (), 0};
        }

        constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->data (), 0};
        }

        constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->data (), 0};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this + fp_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this - fp_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this * fp_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this / fp_simd_type {val};
        }

        constexpr boolean_simd_type <value_type, lanes> operator! (void) const
            noexcept
        {
            return boolean_simd_type <value_type, lanes> {!this->_vec};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this && fp_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this || fp_simd_type {val};
        }

        cpp14_constexpr fp_simd_type & operator+= (fp_simd_type const & sv) &
            noexcept
        {
            this->_vec += sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr fp_simd_type & operator+= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this += fp_simd_type {val};
        }

        cpp14_constexpr fp_simd_type & operator-= (fp_simd_type const & sv) &
            noexcept
        {
            this->_vec -= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr fp_simd_type & operator-= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this -= fp_simd_type {val};
        }

        cpp14_constexpr fp_simd_type & operator*= (fp_simd_type const & sv) &
            noexcept
        {
            this->_vec *= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr fp_simd_type & operator*= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this *= fp_simd_type {val};
        }

        cpp14_constexpr fp_simd_type & operator/= (fp_simd_type const & sv) &
            noexcept
        {
            this->_vec /= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr fp_simd_type & operator/= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this /= fp_simd_type {val};
        }

#if SIMD_HEADER_CLANG
    private:
        template <typename Comparison, std::size_t ... L>
        static constexpr boolean_simd_type <integral_type, lanes>
            unpack_comparison (Comparison && c, util::index_sequence <L...>)
            noexcept
        {
            return boolean_simd_type <integral_type, lanes> {
                std::forward <Comparison> (c) [L]...
            };
        }

    public:
#endif
        constexpr boolean_simd_type <integral_type, lanes>
            operator== (fp_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec == sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec == sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes> operator== (U val)
            const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (fp_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec != sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec != sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this != fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator> (fp_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec > sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec > sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator> (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this > fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator< (fp_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec < sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec < sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator< (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this < fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (fp_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec >= sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec >= sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator>= (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this >= fp_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (fp_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec <= sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_vec <= sv._vec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator<= (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
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
        using base      = simd_type_base <std::complex <T>, l>;
        using this_type = complex_simd_type <T, l>;

        typename base::vector_type_impl _realvec;
        typename base::vector_type_impl _imagvec;

        /*
         * This is a proxy reference object to avoid undefined behavior and
         * type-punning in derived SIMD type classes. It is the returned
         * type from methds such as operator[] and at().
         */
        template <typename VecType, typename ValType>
        class reference_proxy;

        /*
         * This is a proxy pointer object to avoid undefined behavior and
         * type-punning in derived SIMD type classes. It is the returned
         * type from methds such as {c}{r}begin and {c}{r}end.
         */
        template <typename VecType, typename ValType>
        class pointer_proxy;

        template <typename VecType, typename ValType>
        class reference_proxy
        {
        private:
            using vector_type = VecType;
            using value_type  = ValType;
            using pointer     = pointer_proxy <vector_type, value_type>;
            using vecpointer  = typename std::add_pointer <vector_type>::type;

            vecpointer _realref;
            vecpointer _imagref;
            std::ptrdiff_t _index;

        public:
            reference_proxy (void) = delete;
            ~reference_proxy (void) noexcept = default;

            constexpr reference_proxy (vecpointer real,
                                       vecpointer imag,
                                       std::ptrdiff_t index = 0)
                noexcept
                : _realref {real}
                , _imagref {imag}
                , _index   {index}
            {}

            constexpr
                reference_proxy (vector_type & real,
                                 vector_type & imag,
                                 std::ptrdiff_t index = 0)
                noexcept
                : _realref {&real}
                , _imagref {&imag}
                , _index   {index}
            {}

            constexpr reference_proxy (reference_proxy const &) noexcept
                = default;

            template <typename U>
            reference_proxy & operator= (U && u) noexcept
            {
                static_assert (
                    std::is_constructible <this_type, U>::value,
                    "cannot assign to vector lane from non-convertible type"
                );

                (*this->_realref) [this->_index] = static_cast <value_type> (
                    std::forward <U> (u)
                ).real ();
                (*this->_imagref) [this->_index] = static_cast <value_type> (
                    std::forward <U> (u)
                ).imag ();
                return *this;
            }

            reference_proxy & operator= (reference_proxy const & r) noexcept
            {
                this->_realref = r._realref;
                this->_imagref = r._imagref;
                this->_index = r._index;
                return *this;
            }

            void swap (reference_proxy & r) noexcept
            {
                std::swap (this->_realref, r._realref);
                std::swap (this->_imagref, r._imagref);
                std::swap (this->_index, r._index);
            }

            template <typename U>
            constexpr operator U (void) const noexcept
            {
                static_assert (
                    std::is_convertible <value_type, U>::value,
                    "cannot perform cast"
                );

                return static_cast <U> (
                    value_type {
                        (*this->_realref) [this->_index],
                        (*this->_imagref) [this->_index]
                    }
                );
            }

            pointer operator& (void) const & noexcept
            {
                return pointer {this->_realref, this->_imagref, this->_index};
            }

            constexpr bool operator== (reference_proxy const & r) const noexcept
            {
                return (*this->_realref) [this->_index] == (*r._realref) [r._index] &&
                       (*this->_imagref) [this->_index] == (*r._imagref) [r._index];
            }

            constexpr bool operator!= (reference_proxy const & r) const noexcept
            {
                return (*this->_realref) [this->_index] != (*r._realref) [r._index] ||
                       (*this->_imagref) [this->_index] != (*r._imagref) [r._index];
            }
        };

        template <typename VecType, typename ValType>
        class pointer_proxy
        {
        private:
            using vector_type = VecType;
            using value_type  = ValType;
            using vecpointer  = typename std::add_pointer <vector_type>::type;
            using reference   = reference_proxy <vector_type, value_type>;

            vecpointer _realpointer;
            vecpointer _imagpointer;
            std::ptrdiff_t _index;

        public:
            using iterator_category = std::random_access_iterator_tag;

            constexpr pointer_proxy (void) noexcept
                : _realpointer {nullptr}
                , _imagpointer {nullptr}
                , _index       {0}
            {}

            ~pointer_proxy (void) noexcept = default;

            constexpr pointer_proxy (vecpointer real,
                                     vecpointer imag,
                                     std::ptrdiff_t index = 0)
                noexcept
                : _realpointer {real}
                , _imagpointer {imag}
                , _index       {index}
            {}

            constexpr pointer_proxy (vector_type & real,
                                     vector_type & imag,
                                     std::ptrdiff_t index = 0)
                noexcept
                : _realpointer {&real}
                , _imagpointer {&imag}
                , _index       {index}
            {}

            constexpr pointer_proxy (pointer_proxy const &) noexcept = default;

            cpp14_constexpr pointer_proxy & operator= (pointer_proxy p)
                noexcept
            {
                this->_realpointer = p._realpointer;
                this->_imagpointer = p._imagpointer;
                this->_index = p._index;
                return *this;
            }

            operator bool (void) noexcept
            {
                return static_cast <bool> (this->_realpointer) &&
                       static_cast <bool> (this->_imagpointer);
            }

            reference operator* (void) const noexcept
            {
                return reference {
                    this->_realpointer, this->_imagpointer, this->_index
                };
            }

            reference operator[] (std::ptrdiff_t n) const noexcept
            {
                return reference {
                    this->_realpointer, this->_imagpointer, this->_index + n
                };
            }

            pointer_proxy & operator++ (void) noexcept
            {
                this->_index += 1;
                return *this;
            }

            pointer_proxy & operator-- (void) noexcept
            {
                this->_index -= 1;
                return *this;
            }

            pointer_proxy operator++ (int) noexcept
            {
                auto const tmp = *this;
                this->_index += 1;
                return tmp;
            }

            pointer_proxy operator-- (int) noexcept
            {
                auto const tmp = *this;
                this->_index -= 1;
                return tmp;
            }

            pointer_proxy & operator+= (std::ptrdiff_t n) noexcept
            {
                this->_index += n;
                return *this;
            }

            pointer_proxy & operator-= (std::ptrdiff_t n) noexcept
            {
                this->_index -= n;
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
                return this->_index - p._index;
            }

            bool operator== (pointer_proxy p) const noexcept
            {
                return this->_realpointer == p._realpointer &&
                       this->_imagpointer == p._imagpointer &&
                       this->_index == p._index;
            }

            bool operator!= (pointer_proxy p) const noexcept
            {
                return this->_realpointer != p._realpointer ||
                       this->_imagpointer != p._imagpointer ||
                       this->_index != p._index;
            }

            bool operator< (pointer_proxy p) const noexcept
            {
                return (this->_realpointer < p._realpointer &&
                        this->_imagpointer < p._imagpointer)||
                       (this->_realpointer == p._realpointer &&
                        this->_imagpointer == p._imagpointer &&
                        this->_index < p._index);
            }

            bool operator> (pointer_proxy p) const noexcept
            {
                return (this->_realpointer > p._realpointer &&
                        this->_imagpointer > p._imagpointer)||
                       (this->_realpointer == p._realpointer &&
                        this->_imagpointer == p._imagpointer &&
                        this->_index > p._index);
            }

            bool operator<= (pointer_proxy p) const noexcept
            {
                return *this == p || *this < p;
            }

            bool operator>= (pointer_proxy p) const noexcept
            {
                return *this == p || *this > p;
            }
        };

    public:
        static_assert (
            std::is_floating_point <T>::value,
            "template parameter typename T must be a floating point type"
        );

        using vector_type    = typename base::vector_type_impl;
        using value_type     = std::complex <T>;
        using lane_type      = T;
        using real_simd_type = fp_simd_type <T, l>;
        using imag_simd_type = fp_simd_type <T, l>;
        using integral_type = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type   = typename base::signed_integral_type;
        using reference       = reference_proxy <vector_type, value_type>;
        using const_reference =
            reference_proxy <vector_type const, value_type const>;
        using iterator       = pointer_proxy <vector_type, value_type>;
        using const_iterator =
            pointer_proxy <vector_type const, value_type const>;
        using reverse_iterator       = std::reverse_iterator <iterator>;
        using const_reverse_iterator = std::reverse_iterator <const_iterator>;
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
            unpack_real_impl (value_type const (& arr) [lanes],
                              util::index_sequence <L...>) noexcept
        {
            return vector_type {arr [L].real ()...};
        }

        template <std::size_t ... L>
        static constexpr vector_type
            unpack_real_impl (std::array <value_type, lanes> const & arr,
                              util::index_sequence <L...>) noexcept
        {
            return vector_type {std::get <L> (arr).real ()...};
        }

        template <std::size_t ... L>
        static constexpr vector_type
            unpack_imag_impl (value_type const (& arr) [lanes],
                              util::index_sequence <L...>) noexcept
        {
            return vector_type {arr [L].imag ()...};
        }

        template <std::size_t ... L>
        static constexpr vector_type
            unpack_imag_impl (std::array <value_type, lanes> const & arr,
                              util::index_sequence <L...>) noexcept
        {
            return vector_type {std::get <L> (arr).imag ()...};
        }

        template <std::size_t ... L>
        static constexpr vector_type
            extend_impl (T const & t, util::index_sequence <L...>) noexcept
        {
            return vector_type {((void) L, t)...};
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

        static constexpr vector_type extend (T const & t) noexcept
        {
            return extend_impl (t, util::make_index_sequence <lanes> {});
        }

        template <
            typename ... Ts,
            typename = typename std::enable_if <
                sizeof... (Ts) == lanes && lanes != 1
            >::type
        >
        static constexpr vector_type extend (Ts const & ... ts)
            noexcept
        {
            return vector_type {ts...};
        }

    public:
        static complex_simd_type load (value_type const * addr) noexcept
        {
            complex_simd_type result {};
            for (std::size_t i = 0; i < lanes; ++i) {
                result._realvec [i] = *addr++;
                result._imagvec [i] = *addr++;
            }
            return result;
        }

        static complex_simd_type load (value_type const * addr,
                                       std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;

            complex_simd_type result {};
            for (std::size_t i = 0; i < lanes; ++i) {
                result._realvec [i] = *use_addr++;
                result._imagvec [i] = *use_addr++;
            }
            return result;
        }

        static complex_simd_type load (vector_type const * real_addr,
                                       vector_type const * imag_addr) noexcept
        {
            return complex_simd_type {*real_addr, *imag_addr};
        }

        static complex_simd_type load (vector_type const * real_addr,
                                       vector_type const * imag_addr,
                                       std::ptrdiff_t real_off,
                                       std::ptrdiff_t imag_off)
            noexcept
        {
            return complex_simd_type {
                *(real_addr + real_off), *(imag_addr + imag_off)
            };
        }

        static complex_simd_type load_aligned (value_type const * addr) noexcept
        {
            auto aligned_ptr = static_cast <value_type const *> (
                __builtin_assume_aligned (addr, base::alignment)
            );

            complex_simd_type result {};
            for (std::size_t i = 0; i < lanes; ++i) {
                result._realvec [i] = *aligned_ptr++;
                result._imagvec [i] = *aligned_ptr++;
            }
            return result;
        }

        static complex_simd_type load_aligned (value_type const * addr,
                                               std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;
            auto aligned_ptr = static_cast <value_type const *> (
                __builtin_assume_aligned (use_addr, base::alignment)
            );

            complex_simd_type result {};
            for (std::size_t i = 0; i < lanes; ++i) {
                result._realvec [i] = *aligned_ptr++;
                result._imagvec [i] = *aligned_ptr++;
            }
            return result;
        }

        static complex_simd_type load_aligned (vector_type const * real_addr,
                                               vector_type const * imag_addr)
            noexcept
        {
            auto aligned_real_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (real_addr, base::alignment)
            );
            auto aligned_imag_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (imag_addr, base::alignment)
            );

            return complex_simd_type {*aligned_real_ptr, *aligned_imag_ptr};
        }

        static complex_simd_type load_aligned (vector_type const * real_addr,
                                               vector_type const * imag_addr,
                                               std::ptrdiff_t real_off,
                                               std::ptrdiff_t imag_off)
            noexcept
        {
            auto real_use_addr = real_addr + real_off;
            auto imag_use_addr = imag_addr + imag_off;

            auto aligned_real_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (real_use_addr, base::alignment)
            );
            auto aligned_imag_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (imag_use_addr, base::alignment)
            );

            return complex_simd_type {*aligned_real_ptr, *aligned_imag_ptr};
        }

        constexpr complex_simd_type (void) noexcept
            : _realvec {extend (value_type {}.real ())}
            , _imagvec {extend (value_type {}.imag ())}
        {}

        constexpr
            complex_simd_type (vector_type const & realvec,
                               vector_type const & imagvec) noexcept
            : _realvec {realvec}
            , _imagvec {imagvec}
        {}

        explicit constexpr complex_simd_type (value_type const & val) noexcept
            : _realvec {extend (val.real ())}
            , _imagvec {extend (val.imag ())}
        {}

        template <
            typename ... value_types,
            typename = typename std::enable_if <
                sizeof... (value_types) == lanes && lanes != 1
            >::type
        >
        explicit constexpr complex_simd_type (value_types && ... vals) noexcept
            : _realvec {
                extend (
                    static_cast <value_type> (
                        std::forward <value_types> (vals)
                    ).real ()...
                )
            }
            , _imagvec {
                extend (
                    static_cast <value_type> (
                        std::forward <value_types> (vals)
                    ).imag ()...
                )
            }
        {}

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

        cpp14_constexpr complex_simd_type &
            operator= (complex_simd_type const & sv) & noexcept
        {
            this->_realvec = sv._realvec;
            this->_imagvec = sv._imagvec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr complex_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this = complex_simd_type {val};
        }

    private:
        template <std::size_t ... L>
        constexpr std::array <value_type, lanes>
            to_array (util::index_sequence <L...>) const noexcept
        {
            return std::array <value_type, lanes> {{
                value_type {this->_realvec [L], this->_imagvec [L]}...
            }};
        }

    public:
        explicit constexpr operator std::array <value_type, lanes> (void) const
            noexcept
        {
            return this->to_array (util::make_index_sequence <lanes> {});
        }

        template <typename SIMDType>
        cpp14_constexpr SIMDType to (void) const noexcept
        {
            static_assert (
                is_simd_type <SIMDType>::value,
                "cannot perform cast to non-simd type"
            );

            using cast_traits = simd_traits <SIMDType>;
            using rebind_type = rebind <
                typename cast_traits::value_type,
                cast_traits::lanes,
                typename cast_traits::category_tag
            >;
            using rebind_value_type = typename rebind_type::value_type;

            static_assert (
                2 * lanes == cast_traits::lanes,
                "cannot perform conversion of vector type to vector type with a"
                " different number of lanes"
            );

            std::array <rebind_value_type, cast_traits::lanes> result {};
            for (std::size_t i = 0; i < lanes; ++i) {
                result [2*i] = static_cast <rebind_value_type> (
                    this->_realvec [i]
                );
                result [2*i + 1] = static_cast <rebind_value_type> (
                    this->_imagvec [i]
                );
            }
            return rebind_type {result};
        }

        template <typename SIMDType>
        explicit cpp14_constexpr operator SIMDType (void) const noexcept
        {
            return this->template to <SIMDType> ();
        }

        cpp14_constexpr void swap (complex_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        template <std::size_t N>
        cpp14_constexpr complex_simd_type & set (value_type const & val) &
            noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            this->_realvec [N] = val.real ();
            this->_imagvec [N] = val.imag ();
            return *this;
        }

        cpp14_constexpr complex_simd_type &
            set (std::size_t n, value_type const & val) & noexcept
        {
            this->_realvec [n] = val.real ();
            this->_imagvec [n] = val.imag ();
            return *this;
        }

        cpp14_constexpr complex_simd_type &
            set (std::initializer_list <value_type> vlist) & noexcept
        {
            auto lindex = vlist.begin ();
            for (std::size_t i = 0; i < std::min (lanes, vlist.size ()); ++i) {
                this->_realvec [i] = lindex->real ();
                this->_imagvec [i] = lindex->imag ();
                lindex += 1;
            }

            return *this;
        }

        cpp14_constexpr complex_simd_type & fill (value_type const & val) &
            noexcept
        {
            this->_realvec = extend (val.real ());
            this->_realvec = extend (val.imag ());
            return *this;
        }

        constexpr std::pair <vector_type const &, vector_type const &>
            data (void) const & noexcept
        {
            return std::pair <vector_type const &, vector_type const &> (
                this->_realvec, this->_imagvec
            );
        }

        cpp14_constexpr std::pair <vector_type &, vector_type &>
            data (void) & noexcept
        {
            return std::pair <vector_type &, vector_type &> (
                this->_realvec, this->_imagvec
            );
        }

        template <std::size_t N>
        constexpr const_reference get (void) const & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return const_reference {
                this->_realvec, this->_imagvec, static_cast <std::ptrdiff_t> (N)
            };
        }

        template <std::size_t N>
        cpp14_constexpr reference get (void) & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return reference {
                this->_realvec, this->_imagvec, static_cast <std::ptrdiff_t> (N)
            };
        }

        template <std::size_t N>
        constexpr value_type value (void) const noexcept
        {
            return value_type {this->_realvec [N], this->_imagvec [N]};
        }

        constexpr value_type value (std::size_t n) const noexcept
        {
            return value_type {this->_realvec [n], this->_imagvec [n]};
        }

        constexpr real_simd_type real (void) const noexcept
        {
            return real_simd_type {this->_realvec};
        }

        constexpr imag_simd_type imag (void) const noexcept
        {
            return imag_simd_type {this->_imagvec};
        }

        constexpr const_reference
            operator[] (std::size_t n) const & noexcept
        {
            return const_reference {
                this->_realvec, this->_imagvec, static_cast <std::ptrdiff_t> (n)
            };
        }

        cpp14_constexpr reference operator[] (std::size_t n) & noexcept
        {
            return reference {
                this->_realvec, this->_imagvec, static_cast <std::ptrdiff_t> (n)
            };
        }

        constexpr const_reference at (std::size_t n) const &
        {
            return n < lanes ?
                const_reference {
                    this->_realvec, this->_imagvec,
                    static_cast <std::ptrdiff_t> (n)
                } :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        cpp14_constexpr reference at (std::size_t n) &
        {
            return n < lanes ?
                reference {
                    this->_realvec, this->_imagvec,
                    static_cast <std::ptrdiff_t> (n)
                } :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        cpp14_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->_realvec, this->_imagvec, 0};
        }

        cpp14_constexpr iterator end (void) & noexcept
        {
            return iterator {
                this->_realvec, this->_imagvec,
                static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->_realvec, this->_imagvec, 0};
        }

        constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {
                this->_realvec, this->_imagvec,
                static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->_realvec, this->_imagvec, 0};
        }

        constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {
                this->_realvec, this->_imagvec,
                static_cast <std::ptrdiff_t> (lanes)
            };
        }

        cpp14_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {
                this->_realvec, this->_imagvec,
                static_cast <std::ptrdiff_t> (lanes)
            };
        }

        cpp14_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->_realvec, this->_imagvec, 0};
        }

        constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                this->_realvec, this->_imagvec,
                static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->_realvec, this->_imagvec, 0};
        }

        constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                this->_realvec, this->_imagvec,
                static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->_realvec, this->_imagvec, 0};
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
                std::is_constructible <this_type, U>::value,
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this - complex_simd_type {val};
        }

        cpp14_constexpr complex_simd_type operator* (complex_simd_type const & sv)
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this * complex_simd_type {val};
        }

        cpp14_constexpr complex_simd_type operator/ (complex_simd_type const & sv)
            const noexcept
        {
            auto const divisor = sv._realvec * sv._realvec
                               + sv._imagvec * sv._imagvec;

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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this / complex_simd_type {val};
        }

        cpp14_constexpr
        complex_simd_type & operator+= (complex_simd_type const & sv) &
            noexcept
        {
            this->_realvec += sv._realvec;
            this->_imagvec += sv._imagvec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr complex_simd_type & operator+= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this += complex_simd_type {val};
        }

        cpp14_constexpr
        complex_simd_type & operator-= (complex_simd_type const & sv) &
            noexcept
        {
            this->_realvec -= sv._realvec;
            this->_imagvec -= sv._imagvec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr complex_simd_type & operator-= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this -= complex_simd_type {val};
        }

        cpp14_constexpr
        complex_simd_type & operator*= (complex_simd_type const & sv) &
            noexcept
        {
            auto const result = *this * sv;
            *this = result;
            return *this;
        }

        template <typename U>
        cpp14_constexpr complex_simd_type & operator*= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this *= complex_simd_type {val};
        }

        cpp14_constexpr
        complex_simd_type & operator/= (complex_simd_type const & sv) &
            noexcept
        {
            auto const result = *this / sv;
            *this = result;
            return *this;
        }

        template <typename U>
        cpp14_constexpr complex_simd_type & operator/= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this /= complex_simd_type {val};
        }

#if SIMD_HEADER_CLANG
    private:
        template <typename Comparison, std::size_t ... L>
        static constexpr boolean_simd_type <integral_type, lanes>
            unpack_comparison (Comparison && c, util::index_sequence <L...>)
            noexcept
        {
            return boolean_simd_type <integral_type, lanes> {
                std::forward <Comparison> (c) [L]...
            };
        }

    public:
#endif
        constexpr boolean_simd_type <integral_type, lanes>
            operator== (complex_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_realvec == sv._realvec &&
                this->_imagvec == sv._imagvec,
                util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_realvec == sv._realvec &&
                this->_imagvec == sv._imagvec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes> operator== (U val)
            const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == complex_simd_type {val};
        }

        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (complex_simd_type const & sv) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_realvec != sv._realvec ||
                this->_imagvec != sv._imagvec,
                util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type <integral_type, lanes> {
                this->_realvec != sv._realvec ||
                this->_imagvec != sv._imagvec
            };
#endif
        }

        template <typename U>
        constexpr boolean_simd_type <integral_type, lanes>
            operator!= (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
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
        using base      = simd_type_base <T, l>;
        using this_type = boolean_simd_type <T, l>;

        typename base::vector_type_impl _vec;

    public:
        using vector_type            = typename base::vector_type_impl;
        using value_type             = bool;
        using integral_type          = typename base::integral_type;
        using unsigned_integral_type = typename base::unsigned_integral_type;
        using signed_integral_type   = typename base::signed_integral_type;
        using reference              = typename base::reference;
        using const_reference        = typename base::const_reference;
        using iterator               = typename base::pointer;
        using const_iterator         = typename base::const_pointer;
        using reverse_iterator       = std::reverse_iterator <iterator>;
        using const_reverse_iterator = std::reverse_iterator <const_iterator>;
        static constexpr std::size_t lanes = l;

        template <typename U, std::size_t L, typename tag>
        using rebind = simd_type <U, L, tag>;

        static boolean_simd_type load (value_type const * addr) noexcept
        {
            boolean_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *addr++;
            }
            return result;
        }

        static boolean_simd_type load (value_type const * addr,
                                       std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;

            boolean_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *use_addr++;
            }
            return result;
        }

        static boolean_simd_type load (vector_type const * addr) noexcept
        {
            return boolean_simd_type {*addr};
        }

        static boolean_simd_type load (vector_type const * addr,
                                       std::ptrdiff_t off) noexcept
        {
            return boolean_simd_type {*(addr + off)};
        }

        static boolean_simd_type load_aligned (value_type const * addr)
            noexcept
        {
            auto aligned_ptr = static_cast <value_type const *> (
                __builtin_assume_aligned (addr, base::alignment)
            );

            boolean_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *aligned_ptr++;
            }
            return result;
        }

        static boolean_simd_type load_aligned (value_type const * addr,
                                               std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;
            auto aligned_ptr = static_cast <value_type const *> (
                __builtin_assume_aligned (use_addr, base::alignment)
            );

            boolean_simd_type result;
            for (std::size_t i = 0; i < lanes; ++i) {
                result._vec [i] = *aligned_ptr++;
            }
            return result;
        }

        static boolean_simd_type load_aligned (vector_type const * addr)
            noexcept
        {
            auto aligned_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (addr, base::alignment)
            );

            return boolean_simd_type {*aligned_ptr};
        }

        static boolean_simd_type load_aligned (vector_type const * addr,
                                               std::ptrdiff_t off) noexcept
        {
            auto use_addr = addr + off;
            auto aligned_ptr = static_cast <vector_type const *> (
                __builtin_assume_aligned (use_addr, base::alignment)
            );

            return boolean_simd_type {*aligned_ptr};
        }

        static constexpr vector_type make_gcc_compatible (vector_type const & v)
            noexcept
        {
#if SIMD_HEADER_CLANG
            using uvector_type = vext::vector <unsigned_integral_type, lanes>;

            return __builtin_convertvector (
                __builtin_convertvector (v * -1, uvector_type), vector_type
            );
#elif SIMD_HEADER_GNUG
            return v ? ~integral_type {0} : integral_type {0};
#endif
        }

        constexpr boolean_simd_type (void) noexcept
            : _vec {base::extend (value_type {0})}
        {}

        constexpr boolean_simd_type (vector_type const & vec) noexcept
#if SIMD_HEADER_CLANG
            : _vec {vec}
#elif SIMD_HEADER_GNUG
            : _vec {vec & integral_type {1}}
#endif
        {}

        explicit constexpr boolean_simd_type (value_type const & val) noexcept
            : _vec {base::extend (val)}
        {}

        template <
            typename ... value_types,
            typename = typename std::enable_if <
                sizeof... (value_types) == lanes && lanes != 1
            >::type
        >
        explicit constexpr boolean_simd_type (value_types && ... vals) noexcept
            : _vec {
                static_cast <value_type> (std::forward <value_types> (vals))...
            }
        {}

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

        cpp14_constexpr boolean_simd_type &
            operator= (boolean_simd_type const & sv) & noexcept
        {
            this->_vec = sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr boolean_simd_type & operator= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this = boolean_simd_type {val};
        }

    private:
        template <std::size_t ... L>
        constexpr std::array <value_type, lanes>
            to_array (util::index_sequence <L...>) const noexcept
        {
            return std::array <value_type, lanes> {{this->_vec [L]...}};
        }

    public:
        explicit constexpr operator std::array <value_type, lanes> (void) const
            noexcept
        {
            return this->to_array (util::make_index_sequence <lanes> {});
        }

    private:
        template <typename vec_to, typename valtype, typename vec_from>
        static cpp14_constexpr vec_to vector_convert (vec_from const & v)
            noexcept
        {
            return base::template vector_convert <vec_to, valtype> (v);
        }

    public:
        template <typename SIMDType>
        cpp14_constexpr SIMDType to (void) const noexcept
        {
            static_assert (
                is_simd_type <SIMDType>::value,
                "cannot perform cast to non-simd type"
            );

            using cast_traits = simd_traits <SIMDType>;
            using rebind_type = rebind <
                typename cast_traits::value_type,
                cast_traits::lanes,
                typename cast_traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;
            using rebind_value_type = typename rebind_type::value_type;

            static_assert (
                lanes == cast_traits::lanes,
                "cannot perform conversion of vector type to vector type with a"
                " different number of lanes"
            );

            return rebind_type {
                vector_convert <
                    rebind_vector_type, rebind_value_type, vector_type
                > (this->_vec)
            };
        }

        template <typename SIMDType>
        explicit constexpr operator SIMDType (void) const noexcept
        {
            return this->template to <SIMDType> ();
        }

        template <typename SIMDType>
        SIMDType as (void) const noexcept
        {
            static_assert (
                is_simd_type <SIMDType>::value,
                "cannot perform cast to non-simd type"
            );

            using cast_traits = simd_traits <SIMDType>;
            using rebind_type = rebind <
                typename cast_traits::value_type,
                cast_traits::lanes,
                typename cast_traits::category_tag
            >;
            using rebind_vector_type = typename rebind_type::vector_type;

            static_assert (
                sizeof (vector_type) == sizeof (rebind_vector_type),
                "cannot reinterpret vector to differently sized vector type"
            );

            return rebind_type {
                reinterpret_cast <rebind_vector_type> (this->_vec)
            };
        }

        cpp14_constexpr void swap (boolean_simd_type & other) noexcept
        {
            auto tmp = *this;
            *this = other;
            other = tmp;
        }

        template <std::size_t N>
        cpp14_constexpr boolean_simd_type & set (value_type const & val) &
            noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            this->_vec [N] = val;
            return *this;
        }

        cpp14_constexpr boolean_simd_type &
            set (std::size_t n, value_type const & val) & noexcept
        {
            this->_vec [n] = val;
            return *this;
        }

        cpp14_constexpr boolean_simd_type &
            set (std::initializer_list <value_type> vlist) & noexcept
        {
            auto lindex = vlist.begin ();
            for (std::size_t i = 0; i < std::min (lanes, vlist.size ()); ++i) {
                this->_vec [i] = *lindex++;
            }
        }

        cpp14_constexpr void fill (value_type const & val) & noexcept
        {
            this->_vec = base::extend (val);
        }

        cpp14_constexpr vector_type & data (void) & noexcept
        {
            return this->_vec;
        }

        constexpr vector_type const & data (void) const & noexcept
        {
            return this->_vec;
        }

        template <std::size_t N>
        constexpr const_reference get (void) const & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return const_reference {
                &this->_vec, static_cast <std::ptrdiff_t> (N)
            };
        }

        template <std::size_t N>
        cpp14_constexpr reference get (void) & noexcept
        {
            static_assert (
                N < lanes,
                "cannot access out-of-bounds vector lane"
            );

            return reference {&this->_vec, static_cast <std::ptrdiff_t> (N)};
        }

        template <std::size_t N>
        constexpr value_type value (void) const noexcept
        {
            return this->_vec [N];
        }

        constexpr value_type value (std::size_t n) const noexcept
        {
            return this->_vec [n];
        }

        constexpr const_reference operator[] (std::size_t n) const & noexcept
        {
            return const_reference {
                &this->_vec, static_cast <std::ptrdiff_t> (n)
            };
        }

        cpp14_constexpr reference operator[] (std::size_t n) & noexcept
        {
            return reference {
                &this->_vec, static_cast <std::ptrdiff_t> (n)
            };
        }

        constexpr const_reference at (std::size_t n) const &
        {
            return n < lanes ?
                const_reference {this->_vec, n} :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        cpp14_constexpr reference at (std::size_t n) &
        {
            return n < lanes ?
                reference {this->_vec, n} :
                throw std::out_of_range {
                    "access attempt to out-of-bounds vector lane"
                };
        }

        cpp14_constexpr iterator begin (void) & noexcept
        {
            return iterator {this->data (), 0};
        }

        cpp14_constexpr iterator end (void) & noexcept
        {
            return iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_iterator begin (void) const & noexcept
        {
            return const_iterator {this->data (), 0};
        }

        constexpr const_iterator end (void) const & noexcept
        {
            return const_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_iterator cbegin (void) const & noexcept
        {
            return const_iterator {this->data (), 0};
        }

        constexpr const_iterator cend (void) const & noexcept
        {
            return const_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        cpp14_constexpr reverse_iterator rbegin (void) & noexcept
        {
            return reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        cpp14_constexpr reverse_iterator rend (void) & noexcept
        {
            return reverse_iterator {this->data (), 0};
        }

        constexpr const_reverse_iterator rbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_reverse_iterator rend (void) const & noexcept
        {
            return const_reverse_iterator {this->data (), 0};
        }

        constexpr const_reverse_iterator crbegin (void) const & noexcept
        {
            return const_reverse_iterator {
                this->data (), static_cast <std::ptrdiff_t> (lanes)
            };
        }

        constexpr const_reverse_iterator crend (void) const & noexcept
        {
            return const_reverse_iterator {this->data (), 0};
        }

    private:
        constexpr bool any_of_impl (util::lane_tag <1>) const noexcept
        {
            return static_cast <bool> (this->_vec [0]);
        }

        constexpr bool any_of_impl (util::lane_tag <2>) const noexcept
        {
            return this->_vec [0] || this->_vec [1];
        }

        constexpr bool any_of_impl (util::lane_tag <4>) const noexcept
        {
            return
                this->_vec [0] || this->_vec [1] ||
                this->_vec [2] || this->_vec [3];
        }

        constexpr bool any_of_impl (util::lane_tag <8>) const noexcept
        {
            return
                this->_vec [0] || this->_vec [1] ||
                this->_vec [2] || this->_vec [3] ||
                this->_vec [4] || this->_vec [5] ||
                this->_vec [6] || this->_vec [7];
        }

        constexpr bool any_of_impl (util::lane_tag <16>) const noexcept
        {
            return
                this->_vec [0]  || this->_vec [1]  ||
                this->_vec [2]  || this->_vec [3]  ||
                this->_vec [4]  || this->_vec [5]  ||
                this->_vec [6]  || this->_vec [7]  ||
                this->_vec [8]  || this->_vec [9]  ||
                this->_vec [10] || this->_vec [11] ||
                this->_vec [12] || this->_vec [13] ||
                this->_vec [14] || this->_vec [15];
        }

        constexpr bool any_of_impl (util::lane_tag <32>) const noexcept
        {
            return
                this->_vec [0]  || this->_vec [1]  ||
                this->_vec [2]  || this->_vec [3]  ||
                this->_vec [4]  || this->_vec [5]  ||
                this->_vec [6]  || this->_vec [7]  ||
                this->_vec [8]  || this->_vec [9]  ||
                this->_vec [10] || this->_vec [11] ||
                this->_vec [12] || this->_vec [13] ||
                this->_vec [14] || this->_vec [15] ||
                this->_vec [16] || this->_vec [17] ||
                this->_vec [18] || this->_vec [19] ||
                this->_vec [20] || this->_vec [21] ||
                this->_vec [22] || this->_vec [23] ||
                this->_vec [24] || this->_vec [25] ||
                this->_vec [26] || this->_vec [27] ||
                this->_vec [28] || this->_vec [29] ||
                this->_vec [30] || this->_vec [31];
        }

        constexpr bool any_of_impl (util::lane_tag <64>) const noexcept
        {
            return
                this->_vec [0]  || this->_vec [1]  ||
                this->_vec [2]  || this->_vec [3]  ||
                this->_vec [4]  || this->_vec [5]  ||
                this->_vec [6]  || this->_vec [7]  ||
                this->_vec [8]  || this->_vec [9]  ||
                this->_vec [10] || this->_vec [11] ||
                this->_vec [12] || this->_vec [13] ||
                this->_vec [14] || this->_vec [15] ||
                this->_vec [16] || this->_vec [17] ||
                this->_vec [18] || this->_vec [19] ||
                this->_vec [20] || this->_vec [21] ||
                this->_vec [22] || this->_vec [23] ||
                this->_vec [24] || this->_vec [25] ||
                this->_vec [26] || this->_vec [27] ||
                this->_vec [28] || this->_vec [29] ||
                this->_vec [30] || this->_vec [31] ||
                this->_vec [32] || this->_vec [33] ||
                this->_vec [34] || this->_vec [35] ||
                this->_vec [36] || this->_vec [37] ||
                this->_vec [38] || this->_vec [39] ||
                this->_vec [40] || this->_vec [41] ||
                this->_vec [42] || this->_vec [43] ||
                this->_vec [44] || this->_vec [45] ||
                this->_vec [46] || this->_vec [47] ||
                this->_vec [48] || this->_vec [49] ||
                this->_vec [50] || this->_vec [51] ||
                this->_vec [52] || this->_vec [53] ||
                this->_vec [54] || this->_vec [55] ||
                this->_vec [56] || this->_vec [57] ||
                this->_vec [58] || this->_vec [59] ||
                this->_vec [60] || this->_vec [61] ||
                this->_vec [62] || this->_vec [63];
        }

        constexpr bool all_of_impl (util::lane_tag <1>) const noexcept
        {
            return static_cast <bool> (this->_vec [0]);
        }

        constexpr bool all_of_impl (util::lane_tag <2>) const noexcept
        {
            return this->_vec [0] && this->_vec [1];
        }

        constexpr bool all_of_impl (util::lane_tag <4>) const noexcept
        {
            return
                this->_vec [0] && this->_vec [1] &&
                this->_vec [2] && this->_vec [3];
        }

        constexpr bool all_of_impl (util::lane_tag <8>) const noexcept
        {
            return
                this->_vec [0] && this->_vec [1] &&
                this->_vec [2] && this->_vec [3] &&
                this->_vec [4] && this->_vec [5] &&
                this->_vec [6] && this->_vec [7];
        }

        constexpr bool all_of_impl (util::lane_tag <16>) const noexcept
        {
            return
                this->_vec [0]  && this->_vec [1]  &&
                this->_vec [2]  && this->_vec [3]  &&
                this->_vec [4]  && this->_vec [5]  &&
                this->_vec [6]  && this->_vec [7]  &&
                this->_vec [8]  && this->_vec [9]  &&
                this->_vec [10] && this->_vec [11] &&
                this->_vec [12] && this->_vec [13] &&
                this->_vec [14] && this->_vec [15];
        }

        constexpr bool all_of_impl (util::lane_tag <32>) const noexcept
        {
            return
                this->_vec [0]  && this->_vec [1]  &&
                this->_vec [2]  && this->_vec [3]  &&
                this->_vec [4]  && this->_vec [5]  &&
                this->_vec [6]  && this->_vec [7]  &&
                this->_vec [8]  && this->_vec [9]  &&
                this->_vec [10] && this->_vec [11] &&
                this->_vec [12] && this->_vec [13] &&
                this->_vec [14] && this->_vec [15] &&
                this->_vec [16] && this->_vec [17] &&
                this->_vec [18] && this->_vec [19] &&
                this->_vec [20] && this->_vec [21] &&
                this->_vec [22] && this->_vec [23] &&
                this->_vec [24] && this->_vec [25] &&
                this->_vec [26] && this->_vec [27] &&
                this->_vec [28] && this->_vec [29] &&
                this->_vec [30] && this->_vec [31];
        }

        constexpr bool all_of_impl (util::lane_tag <64>) const noexcept
        {
            return
                this->_vec [0]  && this->_vec [1]  &&
                this->_vec [2]  && this->_vec [3]  &&
                this->_vec [4]  && this->_vec [5]  &&
                this->_vec [6]  && this->_vec [7]  &&
                this->_vec [8]  && this->_vec [9]  &&
                this->_vec [10] && this->_vec [11] &&
                this->_vec [12] && this->_vec [13] &&
                this->_vec [14] && this->_vec [15] &&
                this->_vec [16] && this->_vec [17] &&
                this->_vec [18] && this->_vec [19] &&
                this->_vec [20] && this->_vec [21] &&
                this->_vec [22] && this->_vec [23] &&
                this->_vec [24] && this->_vec [25] &&
                this->_vec [26] && this->_vec [27] &&
                this->_vec [28] && this->_vec [29] &&
                this->_vec [30] && this->_vec [31] &&
                this->_vec [32] && this->_vec [33] &&
                this->_vec [34] && this->_vec [35] &&
                this->_vec [36] && this->_vec [37] &&
                this->_vec [38] && this->_vec [39] &&
                this->_vec [40] && this->_vec [41] &&
                this->_vec [42] && this->_vec [43] &&
                this->_vec [44] && this->_vec [45] &&
                this->_vec [46] && this->_vec [47] &&
                this->_vec [48] && this->_vec [49] &&
                this->_vec [50] && this->_vec [51] &&
                this->_vec [52] && this->_vec [53] &&
                this->_vec [54] && this->_vec [55] &&
                this->_vec [56] && this->_vec [57] &&
                this->_vec [58] && this->_vec [59] &&
                this->_vec [60] && this->_vec [61] &&
                this->_vec [62] && this->_vec [63];
        }

        constexpr bool none_of_impl (util::lane_tag <1>) const noexcept
        {
            return !this->any_of_impl (util::lane_tag <1> {});
        }

        constexpr bool none_of_impl (util::lane_tag <2>) const noexcept
        {
            return !this->any_of_impl (util::lane_tag <2> {});
        }

        constexpr bool none_of_impl (util::lane_tag <4>) const noexcept
        {
            return !this->any_of_impl (util::lane_tag <4> {});
        }

        constexpr bool none_of_impl (util::lane_tag <8>) const noexcept
        {
            return !this->any_of_impl (util::lane_tag <8> {});
        }

        constexpr bool none_of_impl (util::lane_tag <16>) const noexcept
        {
            return !this->any_of_impl (util::lane_tag <16> {});
        }

        constexpr bool none_of_impl (util::lane_tag <32>) const noexcept
        {
            return !this->any_of_impl (util::lane_tag <32> {});
        }

        constexpr bool none_of_impl (util::lane_tag <64>) const noexcept
        {
            return !this->any_of_impl (util::lane_tag <64> {});
        }

    public:
        constexpr bool any_of (void) const noexcept
        {
            return this->any_of_impl (util::lane_tag <lanes> {});
        }

        cpp14_constexpr bool all_of (void) const noexcept
        {
            return this->all_of_impl (util::lane_tag <lanes> {});
        }

        cpp14_constexpr bool none_of (void) const noexcept
        {
            return this->none_of_impl (util::lane_tag <lanes> {});
        }

    private:
        template <std::size_t ... L>
        constexpr boolean_simd_type normalize_impl (util::index_sequence <L...>)
            const noexcept
        {
            return boolean_simd_type {
                (this->_vec [L] ? integral_type {1} : integral_type {0})...
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this & boolean_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this | boolean_simd_type {val};
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
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this ^ boolean_simd_type {val};
        }

#if SIMD_HEADER_CLANG
    private:
        template <typename Comparison, std::size_t ... L>
        static constexpr boolean_simd_type
            unpack_comparison (Comparison && c, util::index_sequence <L...>)
            noexcept
        {
            return boolean_simd_type <integral_type, lanes> {
                std::forward <Comparison> (c) [L]...
            };
        }

    public:
#endif
        constexpr boolean_simd_type operator! (void) const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                !this->_vec, util::make_index_sequence <lanes> {}
            );
#else
            return boolean_simd_type {!this->_vec};
#endif
        }

        constexpr boolean_simd_type operator&& (boolean_simd_type const & sv)
            const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec && sv._vec, util::make_index_sequence <lanes> {}
            );
#else
            return boolean_simd_type {this->_vec && sv._vec};
#endif
        }

        template <typename U>
        constexpr boolean_simd_type operator&& (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this && boolean_simd_type {val};
        }

        constexpr boolean_simd_type operator|| (boolean_simd_type const & sv)
            const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec || sv._vec, util::make_index_sequence <lanes> {}
            );
#else
            return boolean_simd_type {this->_vec || sv._vec};
#endif
        }

        template <typename U>
        constexpr boolean_simd_type operator|| (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this || boolean_simd_type {val};
        }

        cpp14_constexpr boolean_simd_type &
            operator&= (boolean_simd_type const & sv) & noexcept
        {
            this->_vec &= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr boolean_simd_type & operator&= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this &= boolean_simd_type {val};
        }

        cpp14_constexpr boolean_simd_type &
            operator|= (boolean_simd_type const & sv) & noexcept
        {
            this->_vec |= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr boolean_simd_type & operator|= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this |= boolean_simd_type {val};
        }

        cpp14_constexpr boolean_simd_type &
            operator^= (boolean_simd_type const & sv) & noexcept
        {
            this->_vec ^= sv._vec;
            return *this;
        }

        template <typename U>
        cpp14_constexpr boolean_simd_type & operator^= (U val) & noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this ^= boolean_simd_type {val};
        }

        constexpr boolean_simd_type operator== (boolean_simd_type const & sv)
            const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec == sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type {this->_vec == sv._vec};
#endif
        }

        template <typename U>
        constexpr boolean_simd_type operator== (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this == boolean_simd_type {val};
        }

        constexpr boolean_simd_type operator!= (boolean_simd_type const & sv)
            const noexcept
        {
#if SIMD_HEADER_CLANG
            return unpack_comparison (
                this->_vec != sv._vec, util::make_index_sequence <lanes> {}
            );
#elif SIMD_HEADER_GNUG
            return boolean_simd_type {this->_vec != sv._vec};
#endif
        }

        template <typename U>
        constexpr boolean_simd_type
            operator!= (U val) const noexcept
        {
            static_assert (
                std::is_constructible <this_type, U>::value,
                "cannot perform operation between vector type and scalar type"
                " without conversion"
            );

            return *this != boolean_simd_type {val};
        }
    };
#pragma GCC diagnostic pop
}   // namespace detail

    template <typename SIMDType>
    struct simd_traits : public detail::simd_traits <SIMDType> {};

    template <typename SIMDType>
    struct simd_traits <SIMDType const> : public simd_traits <SIMDType>
    {};

    template <typename SIMDType>
    struct simd_traits <SIMDType &> : public simd_traits <SIMDType>
    {};

    template <typename SIMDType>
    struct simd_traits <SIMDType &&> : public simd_traits <SIMDType>
    {};

    template <typename SIMDType>
    struct simd_traits <SIMDType const &> : public simd_traits <SIMDType>
    {};

    using arithmetic_tag = detail::arithmetic_tag;
    using complex_tag = detail::complex_tag;
    using boolean_tag = detail::boolean_tag;

    template <typename T, std::size_t lanes, typename tag = arithmetic_tag>
    using simd_type = detail::simd_type <T, lanes, tag>;

    template <typename>
    struct is_boolean : std::false_type {};

    template <typename T, std::size_t lanes>
    struct is_boolean <detail::boolean_simd_type <T, lanes>>
        : std::true_type {};

    template <typename>
    struct is_arithmetic : std::false_type {};

    template <typename T, std::size_t lanes>
    struct is_arithmetic <detail::integral_simd_type <T, lanes>>
        : std::true_type {};

    template <typename T, std::size_t lanes>
    struct is_arithmetic <detail::fp_simd_type <T, lanes>>
        : std::true_type {};

    template <typename T, std::size_t lanes>
    struct is_arithmetic <detail::complex_simd_type <T, lanes>>
        : std::true_type {};

    template <typename>
    struct is_complex : std::false_type {};

    template <typename T, std::size_t lanes>
    struct is_complex <detail::complex_simd_type <T, lanes>>
        : std::true_type {};

    template <typename>
    struct is_arithmetic_integral : std::false_type {};

    template <typename T, std::size_t lanes>
    struct is_arithmetic_integral <detail::integral_simd_type <T, lanes>>
        : std::true_type
    {};

    template <typename>
    struct is_arithmetic_signed_integral : std::false_type {};

    template <typename T, std::size_t lanes>
    struct is_arithmetic_signed_integral <detail::integral_simd_type <T, lanes>>
        : std::conditional <
            std::is_signed <T>::value,
            std::true_type,
            std::false_type
        >::type
    {};

    template <typename>
    struct is_arithmetic_unsigned_integral : std::false_type {};

    template <typename T, std::size_t lanes>
    struct is_arithmetic_unsigned_integral <
        detail::integral_simd_type <T, lanes>
    >
        : std::conditional <
            std::is_unsigned <T>::value,
            std::true_type,
            std::false_type
        >::type
    {};

    template <typename>
    struct is_arithmetic_floating_point : std::false_type {};

    template <typename T, std::size_t lanes>
    struct is_arithmetic_floating_point <detail::fp_simd_type <T, lanes>>
        : std::true_type {};

    template <typename SIMDType>
    SIMDType load (typename simd_traits <SIMDType>::value_type const * addr)
        noexcept
    {
        return typename std::decay <SIMDType>::type::load (addr);
    }

    template <typename SIMDType>
    SIMDType load (typename simd_traits <SIMDType>::value_type const * addr,
                   std::ptrdiff_t off) noexcept
    {
        return typename std::decay <SIMDType>::type::load (addr, off);
    }

    template <typename SIMDType>
    SIMDType load (typename simd_traits <SIMDType>::vector_type const * addr)
        noexcept
    {
        return typename std::decay <SIMDType>::type::load (addr);
    }

    template <typename SIMDType>
    SIMDType load (typename simd_traits <SIMDType>::vector_type const * addr,
                   std::ptrdiff_t off) noexcept
    {
        return typename std::decay <SIMDType>::type::load (addr, off);
    }

    template <typename SIMDType>
    SIMDType
        load_aligned (typename simd_traits <SIMDType>::value_type const * addr)
        noexcept
    {
        return typename std::decay <SIMDType>::type::load_aligned (addr);
    }

    template <typename SIMDType>
    SIMDType
        load_aligned (typename simd_traits <SIMDType>::value_type const * addr,
                      std::ptrdiff_t off) noexcept
    {
        return typename std::decay <SIMDType>::type::load_aligned (addr, off);
    }

    template <typename SIMDType>
    SIMDType
        load_aligned (typename simd_traits <SIMDType>::vector_type const * addr)
        noexcept
    {
        return typename std::decay <SIMDType>::type::load_aligned (addr);
    }

    template <typename SIMDType>
    SIMDType
        load_aligned (typename simd_traits <SIMDType>::vector_type const * addr,
                      std::ptrdiff_t off)
        noexcept
    {
        return typename std::decay <SIMDType>::type::load_aligned (addr, off);
    }

    template <std::size_t N, typename SIMDType>
    constexpr typename simd_traits <SIMDType>::const_reference
        get (SIMDType const & sv) noexcept
    {
        static_assert (
            N < simd_traits <SIMDType>::lanes,
            "cannot access out-of-bounds vector lane"
        );

        return sv.template get <N> ();
    }

    template <std::size_t N, typename SIMDType>
    constexpr typename simd_traits <SIMDType>::reference
        get (SIMDType & sv) noexcept
    {
        static_assert (
            N < simd_traits <SIMDType>::lanes,
            "cannot access out-of-bounds vector lane"
        );

        return sv.template get <N> ();
    }

    template <typename SIMDType>
    constexpr typename simd_traits <SIMDType>::const_reference
        get (std::size_t n, SIMDType const & sv) noexcept
    {
        return sv.get (n);
    }

    template <typename SIMDType>
    constexpr typename simd_traits <SIMDType>::reference
        get (std::size_t n, SIMDType & sv) noexcept
    {
        return sv.get (n);
    }

    template <std::size_t N, typename SIMDType>
    constexpr typename simd_traits <SIMDType>::value_type
        value (SIMDType const & sv) noexcept
    {
        static_assert (
            N < simd_traits <SIMDType>::lanes,
            "cannot access out-of-bounds vector lane"
        );

        return sv.template value <N> ();
    }

    template <typename SIMDType>
    constexpr typename simd_traits <SIMDType>::value_type
        value (std::size_t n, SIMDType const & sv) noexcept
    {
        return sv.value (n);
    }

    template <std::size_t N, typename SIMDType>
    cpp14_constexpr void
    set (SIMDType & sv, typename simd_traits <SIMDType>::value_type const & v)
        noexcept
    {
        static_assert (
            N < simd_traits <SIMDType>::lanes,
            "cannot access out-of-bounds vector lane"
        );

        sv.template set <N> (v);
    }

    template <typename SIMDType>
    cpp14_constexpr void
        set (std::size_t n, SIMDType & sv,
             typename simd_traits <SIMDType>::value_type const & v) noexcept
    {
        sv.set (n, v);
    }

    template <typename SIMDTypeTo, typename SIMDTypeFrom>
    constexpr SIMDTypeTo static_convert (SIMDTypeFrom const & sv) noexcept
    {
        return sv.template to <SIMDTypeTo> ();
    }

    template <typename SIMDTypeAs, typename SIMDTypeFrom>
    SIMDTypeAs reinterpret_convert (SIMDTypeFrom const & sv) noexcept
    {
        return sv.template as <SIMDTypeAs> ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::iterator begin (SIMDType & sv) noexcept
    {
        return sv.begin ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::iterator end (SIMDType & sv) noexcept
    {
        return sv.end ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::const_iterator begin (SIMDType const & sv)
        noexcept
    {
        return sv.begin ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::const_iterator end (SIMDType const & sv)
        noexcept
    {
        return sv.end ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::const_iterator cbegin (SIMDType const & sv)
        noexcept
    {
        return sv.cbegin ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::const_iterator cend (SIMDType const & sv)
        noexcept
    {
        return sv.cend ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::reverse_iterator rbegin (SIMDType & sv)
        noexcept
    {
        return sv.rbegin ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::iterator rend (SIMDType & sv) noexcept
    {
        return sv.rend ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::const_reverse_iterator
        rbegin (SIMDType const & sv) noexcept
    {
        return sv.rbegin ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::const_reverse_iterator
        rend (SIMDType const & sv) noexcept
    {
        return sv.rend ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::const_reverse_iterator
        crbegin (SIMDType const & sv) noexcept
    {
        return sv.crbegin ();
    }

    template <typename SIMDType>
    typename simd_traits <SIMDType>::const_reverse_iterator
        crend (SIMDType const & sv) noexcept
    {
        return sv.crend ();
    }

namespace detail
{
    template <typename T>
    struct is_scalar_or_complex : std::conditional <
        std::is_scalar <T>::value || is_complex_scalar <T>::value,
        std::true_type,
        std::false_type
    >::type {};

    template <typename T>
    struct is_scalar_or_complex <T const> : is_scalar_or_complex <T> {};

    template <typename T>
    struct is_scalar_or_complex <T &> : is_scalar_or_complex <T> {};

    template <typename T>
    struct is_scalar_or_complex <T const &> : is_scalar_or_complex <T> {};
}   // namespace detail

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator+ (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv + static_cast <value_type> (val);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator- (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return -sv + static_cast <value_type> (val);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator* (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv * static_cast <value_type> (val);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator/ (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return SIMDType {static_cast <value_type> (val)} / sv;
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator% (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return SIMDType {static_cast <value_type> (val)} % sv;
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator<< (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return SIMDType {static_cast <value_type> (val)} << sv;
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator>> (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return SIMDType {static_cast <value_type> (val)} >> sv;
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator& (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv & static_cast <value_type> (val);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator| (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv | static_cast <value_type> (val);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr SIMDType operator^ (T const & val, SIMDType const & sv) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv ^ static_cast <value_type> (val);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr auto operator&& (T const & t, SIMDType const & sv) noexcept
        -> decltype (
            sv && static_cast <typename simd_traits <SIMDType>::value_type> (t)
        )
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv && static_cast <value_type> (t);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr auto operator|| (T const & t, SIMDType const & sv) noexcept
        -> decltype (
            sv || static_cast <typename simd_traits <SIMDType>::value_type> (t)
        )
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv || static_cast <value_type> (t);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr auto operator== (T const & t, SIMDType const & sv) noexcept
        -> decltype (
            sv == static_cast <typename simd_traits <SIMDType>::value_type> (t)
        )
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv == static_cast <value_type> (t);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr auto operator!= (T const & t, SIMDType const & sv) noexcept
        -> decltype (
            sv != static_cast <typename simd_traits <SIMDType>::value_type> (t)
        )
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv != static_cast <value_type> (t);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr auto operator> (T const & t, SIMDType const & sv) noexcept
        -> decltype (
            sv > static_cast <typename simd_traits <SIMDType>::value_type> (t)
        )
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv > static_cast <value_type> (t);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr auto operator< (T const & t, SIMDType const & sv) noexcept
        -> decltype (
            sv < static_cast <typename simd_traits <SIMDType>::value_type> (t)
        )
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv < static_cast <value_type> (t);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr auto operator>= (T const & t, SIMDType const & sv) noexcept
        -> decltype (
            sv >= static_cast <typename simd_traits <SIMDType>::value_type> (t)
        )
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv >= static_cast <value_type> (t);
    }

    template <
        typename T, typename SIMDType,
        typename = typename std::enable_if <
            detail::is_scalar_or_complex <T>::value
        >::type
    >
    constexpr auto operator<= (T const & t, SIMDType const & sv) noexcept
        -> decltype (
            sv <= static_cast <typename simd_traits <SIMDType>::value_type> (t)
        )
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        static_assert (
            std::is_convertible <T, value_type>::value,
            "cannot perform binary operation between different types without"
            " conversion"
        );

        return sv <= static_cast <value_type> (t);
    }

    template <typename SIMDType>
    struct plus
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize plus with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u + v)
        {
            return u + v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u + val)
        {
            return u + val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val + u)
        {
            return val + u;
        }
    };

    template <typename SIMDType>
    struct minus
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize minus with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u - v)
        {
            return u - v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u - val)
        {
            return u - val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val - u)
        {
            return val - u;
        }
    };

    template <typename SIMDType>
    struct multiplies
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize multiplies with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u * v)
        {
            return u * v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u * val)
        {
            return u * val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val * u)
        {
            return val * u;
        }
    };

    template <typename SIMDType>
    struct divides
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize divides with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u / v)
        {
            return u / v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u / val)
        {
            return u / val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val / u)
        {
            return val / u;
        }
    };

    template <typename SIMDType>
    struct modulus
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize modulus with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u % v)
        {
            return u % v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u % val)
        {
            return u % val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val % u)
        {
            return val % u;
        }
    };

    template <typename SIMDType>
    struct negate
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize negate with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u) const
            noexcept
            -> decltype (-u)
        {
            return -u;
        }
    };

    template <typename SIMDType>
    struct shift_left
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize shift_left with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u << v)
        {
            return u << v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u << val)
        {
            return u << val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val << u)
        {
            return val << u;
        }
    };

    template <typename SIMDType>
    struct shift_right
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize shift_right with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u >> v)
        {
            return u >> v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u >> val)
        {
            return u >> val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val >> u)
        {
            return val >> u;
        }
    };

    template <typename SIMDType>
    struct equal_to
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize equal_to with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u == v)
        {
            return u == v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u == val)
        {
            return u == val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val == u)
        {
            return val == u;
        }
    };

    template <typename SIMDType>
    struct not_equal_to
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize not_equal_to with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u != v)
        {
            return u != v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u != val)
        {
            return u != val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val != u)
        {
            return val != u;
        }
    };

    template <typename SIMDType>
    struct greater
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize greater with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u > v)
        {
            return u > v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u > val)
        {
            return u > val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val > u)
        {
            return val > u;
        }
    };

    template <typename SIMDType>
    struct less
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize less with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u < v)
        {
            return u < v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u < val)
        {
            return u < val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val < u)
        {
            return val < u;
        }
    };

    template <typename SIMDType>
    struct greater_equal
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize greater_equal with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u >= v)
        {
            return u >= v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u >= val)
        {
            return u >= val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val >= u)
        {
            return val >= u;
        }
    };

    template <typename SIMDType>
    struct less_equal
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize less_equal with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u <= v)
        {
            return u <= v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u <= val)
        {
            return u <= val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val <= u)
        {
            return val <= u;
        }
    };

    template <typename SIMDType>
    struct logical_and
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize logical_and with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u && v)
        {
            return u && v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u && val)
        {
            return u && val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val && u)
        {
            return val && u;
        }
    };

    template <typename SIMDType>
    struct logical_or
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize logical_or with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u || v)
        {
            return u || v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u || val)
        {
            return u || val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val || u)
        {
            return val || u;
        }
    };

    template <typename SIMDType>
    struct logical_not
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize logical_not with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u) const
            noexcept
            -> decltype (!u)
        {
            return !u;
        }
    };

    template <typename SIMDType>
    struct bit_and
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize bit_and with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u & v)
        {
            return u & v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u & val)
        {
            return u & val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val & u)
        {
            return val & u;
        }
    };

    template <typename SIMDType>
    struct bit_or
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize bit_or with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u | v)
        {
            return u | v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u | val)
        {
            return u | val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val | u)
        {
            return val | u;
        }
    };

    template <typename SIMDType>
    struct bit_xor
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize bit_xor with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u, SIMDType const & v) const
            noexcept
            -> decltype (u ^ v)
        {
            return u ^ v;
        }

        template <typename T>
        constexpr auto operator() (SIMDType const & u, T const & val) const
            noexcept
            -> decltype (u ^ val)
        {
            return u ^ val;
        }

        template <typename T>
        constexpr auto operator() (T const & val, SIMDType const & u) const
            noexcept
            -> decltype (val ^ u)
        {
            return val ^ u;
        }
    };

    template <typename SIMDType>
    struct bit_not
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "cannot specialize bit_not with non-SIMD type"
        );

        constexpr auto operator() (SIMDType const & u) const
            noexcept
            -> decltype (~u)
        {
            return ~u;
        }
    };

#if SIMD_HEADER_CLANG
namespace detail
{
    template <typename SIMDType, typename IntegralSIMDType, std::size_t ... L>
    SIMDType shuffle_impl (SIMDType const & sv1,
                           SIMDType const & sv2,
                           IntegralSIMDType const & mask,
                           util::index_sequence <L...>) noexcept
    {
        return SIMDType {
            __builtin_shufflevector (
                sv1.data (), sv2.data (), simd::value <L> (mask)...
            )
        };
    }
}   // namespace detail

    template <typename SIMDType, typename IntegralSIMDType>
    SIMDType shuffle (SIMDType const & sv, IntegralSIMDType const & mask)
        noexcept
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );

        static_assert (
            detail::is_simd_type <IntegralSIMDType>::value,
            "template parameter IntegralSIMDType must be a simd type"
        );

        using mask_traits_type = simd_traits <IntegralSIMDType>;
        using mask_value_type  = typename mask_traits_type::value_type;
        static_assert (
            std::is_integral <mask_value_type>::value,
            "template parameter T of mask simd type must be an integral type"
        );

        return detail::shuffle_impl (
            sv, sv, mask,
            detail::util::make_index_sequence <mask_traits_type::lanes> {}
        );
    }

    template <typename SIMDType, typename IntegralSIMDType>
    SIMDType shuffle (SIMDType const & sv1,
                      SIMDType const & sv2,
                      IntegralSIMDType const & mask) noexcept
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );

        static_assert (
            detail::is_simd_type <IntegralSIMDType>::value,
            "template parameter IntegralSIMDType must be a simd type"
        );

        using mask_traits_type = simd_traits <IntegralSIMDType>;
        using mask_value_type  = typename mask_traits_type::value_type;
        static_assert (
            std::is_integral <mask_value_type>::value,
            "template parameter T of mask simd type must be an integral type"
        );

        return detail::shuffle_impl (
            sv1, sv2, mask,
            detail::util::make_index_sequence <mask_traits_type::lanes> {}
        );
    }

    template <std::size_t ... Mask, typename SIMDType>
    SIMDType shuffle (SIMDType const & sv) noexcept
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );
        static_assert (
            simd_traits <SIMDType>::lanes == sizeof... (Mask),
            "shuffle with explicit parameters requires the same number as "
            "simd type lanes"
        );

        return SIMDType {
            __builtin_shufflevector (sv.data (), sv.data (), Mask...)
        };
    }

    template <std::size_t ... Mask, typename SIMDType>
    SIMDType shuffle (SIMDType const & sv1, SIMDType const & sv2) noexcept
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );
        static_assert (
            simd_traits <SIMDType>::lanes == sizeof... (Mask),
            "shuffle with explicit parameters requires the same number as "
            "simd type lanes"
        );

        return SIMDType {
            __builtin_shufflevector (sv1.data (), sv2.data (), Mask...)
        };
    }
#elif SIMD_HEADER_GNUG
    template <typename SIMDType, typename IntegralSIMDType>
    SIMDType shuffle (SIMDType const & sv, IntegralSIMDType const & mask)
        noexcept
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );

        static_assert (
            detail::is_simd_type <IntegralSIMDType>::value,
            "template parameter IntegralSIMDType must be a simd type"
        );

        using mask_traits_type = simd_traits <IntegralSIMDType>;
        using mask_value_type  = typename mask_traits_type::value_type;
        static_assert (
            std::is_integral <mask_value_type>::value,
            "template parameter T of mask simd type must be an integral type"
        );

        return SIMDType {__builtin_shuffle (sv.data (), mask.data ())};
    }

    template <typename SIMDType, typename IntegralSIMDType>
    SIMDType shuffle (SIMDType const & sv1,
                      SIMDType const & sv2,
                      IntegralSIMDType const & mask) noexcept
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );

        static_assert (
            detail::is_simd_type <IntegralSIMDType>::value,
            "template parameter IntegralSIMDType must be a simd type"
        );

        using mask_traits_type = simd_traits <IntegralSIMDType>;
        using mask_value_type  = typename mask_traits_type::value_type;
        static_assert (
            std::is_integral <mask_value_type>::value,
            "template parameter T of mask simd type must be an integral type"
        );

        return SIMDType {
            __builtin_shuffle (sv1.data (), sv2.data (), mask.data ())
        };
    }

    template <std::size_t ... Mask, typename SIMDType>
    SIMDType shuffle (SIMDType const & sv) noexcept
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );
        static_assert (
            simd_traits <SIMDType>::lanes == sizeof... (Mask),
            "shuffle with explicit parameters requires the same number as "
            "simd type lanes"
        );

        using mask_type = detail::integral_simd_type <
            std::size_t, sizeof... (Mask)
        >;

        return SIMDType {
            __builtin_shuffle (sv.data (), mask_type {Mask...}.data ())
        };
    }

    template <std::size_t ... Mask, typename SIMDType>
    SIMDType shuffle (SIMDType const & sv1, SIMDType const & sv2) noexcept
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );
        static_assert (
            simd_traits <SIMDType>::lanes == sizeof... (Mask),
            "shuffle with explicit parameters requires the same number as "
            "simd type lanes"
        );

        using mask_type = detail::integral_simd_type <
            std::size_t, sizeof... (Mask)
        >;

        return SIMDType {
            __builtin_shuffle (
                sv1.data (), sv2.data (), mask_type {Mask...}.data ()
            )
        };
    }
#endif

    template <typename BooleanSIMDType>
    constexpr bool any_of (BooleanSIMDType const & sv) noexcept
    {
        return sv.any_of ();
    }

    template <typename BooleanSIMDType>
    constexpr bool all_of (BooleanSIMDType const & sv) noexcept
    {
        return sv.all_of ();
    }

    template <typename BooleanSIMDType>
    constexpr bool none_of (BooleanSIMDType const & sv) noexcept
    {
        return sv.none_of ();
    }

    /*
     * General iterator for SIMD vector types constructed either from a pointer
     * to a contiguous array of scalars in memory or a pointer to a contiguous
     * array of SIMD vector types in memory. This iterator does not assume the
     * contents in memory are aligned to the requirements of the SIMD type.
     *
     * This iterator conforms to the ContiguousIterator concept and, if the
     * provided template parameter is not const-qualified, OutputIterator
     * concepts; it is therefore a contiguous mutable iterator in this case,
     * and a contiguous iterator otherwise.
     */
    template <typename SIMDType>
    class iterator
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a SIMD vector type"
        );

    private:
        using simd_type        = typename std::remove_cv <SIMDType>::type;
        using traits_type      = simd_traits <simd_type>;
        using simd_value_type  = typename traits_type::value_type;
        using simd_vector_type = typename traits_type::vector_type;

        static constexpr bool is_const = std::is_const <SIMDType>::value;
        using non_const_simd_pointer = simd_type *;
        using const_simd_pointer     = simd_type const *;
        using pointer_type           = typename std::conditional <
            is_const, const_simd_pointer, non_const_simd_pointer
        >;

        using non_const_simd_reference = simd_type &;
        using const_simd_reference     = simd_type const &;
        using reference_type           = typename std::conditional <
            is_const, const_simd_reference, non_const_simd_reference
        >;

        pointer_type _iter;

    public:
        using difference_type   = std::ptrdiff_t;
        using value_type        = simd_type;
        using pointer           = pointer_type;
        using reference         = reference_type;
        using iterator_category = std::random_access_iterator_tag;

        iterator (void) noexcept
            : _iter {nullptr}
        {}

        iterator (simd_value_type * p) noexcept
            : _iter {reinterpret_cast <pointer_type> (p)}
        {}

        iterator (simd_value_type const * p,
                  typename std::enable_if <is_const>::type * = nullptr) noexcept
            : _iter {reinterpret_cast <pointer_type> (p)}
        {}

        iterator (simd_type * p) noexcept
            : _iter {p}
        {}

        iterator (simd_type const * p,
                  typename std::enable_if <is_const>::type * = nullptr) noexcept
            : _iter {p}
        {}

        iterator (iterator const &) noexcept             = default;
        iterator & operator= (iterator const &) noexcept = default;

        ~iterator (void) noexcept = default;

        reference operator* (void) const noexcept
        {
            return *this->_iter;
        }

        reference operator[] (difference_type n) const noexcept
        {
            return *(this->_iter + n);
        }

        pointer operator-> (void) const noexcept
        {
            return this->_iter;
        }

        iterator & operator++ (void) noexcept
        {
            this->_iter += 1;
            return *this;
        }

        iterator & operator-- (void) noexcept
        {
            this->_iter -= 1;
            return *this;
        }

        iterator operator++ (int) noexcept
        {
            auto tmp = *this;
            this->_iter += 1;
            return tmp;
        }

        iterator operator-- (int) noexcept
        {
            auto tmp = *this;
            this->_iter -= 1;
            return tmp;
        }

        iterator & operator+= (difference_type n) noexcept
        {
            this->_iter += n;
            return *this;
        }

        iterator & operator-= (difference_type n) noexcept
        {
            this->_iter -= n;
            return *this;
        }

        iterator operator+ (difference_type n) const noexcept
        {
            auto tmp = *this;
            return tmp += n;
        }

        friend iterator operator+ (difference_type n, iterator const & it)
            noexcept
        {
            return it + n;
        }

        iterator operator- (difference_type n) const noexcept
        {
            auto tmp = *this;
            return tmp -= n;
        }

        difference_type operator- (iterator const & other) const noexcept
        {
            return this->_iter - other._iter;
        }

        bool operator== (iterator const & other) const noexcept
        {
            return this->_iter == other._iter;
        }

        bool operator!= (iterator const & other) const noexcept
        {
            return this->_iter != other._iter;
        }

        bool operator> (iterator const & other) const noexcept
        {
            return this->_iter > other._iter;
        }

        bool operator< (iterator const & other) const noexcept
        {
            return this->_iter < other._iter;
        }

        bool operator>= (iterator const & other) const noexcept
        {
            return this->_iter >= other._iter;
        }

        bool operator<= (iterator const & other) const noexcept
        {
            return this->_iter <= other._iter;
        }
    };

    /*
     * General allocator for SIMD vector types; this should be used with
     * containers to ensure correct data alignment.
     */
    template <typename SIMDType>
    class allocator
    {
    public:
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template typename parameter SIMDType must be a SIMD vector type"
        );

        using value_type      = typename std::decay <SIMDType>::type;
        using is_always_equal = std::true_type;
        using propogate_on_container_move_assignment = std::true_type;

#if __cplusplus > 201402L
        [[deprecated("member type pointer is deprecated in C++17")]]
#endif
        using pointer = value_type *;

#if __cplusplus > 201402L
        [[deprecated("member type const_pointer is deprecated in C++17")]]
#endif
        using const_pointer = value_type const *;

#if __cplusplus > 201402L
        [[deprecated("member type reference is deprecated in C++17")]]
#endif
        using reference = value_type &;

#if __cplusplus > 201402L
        [[deprecated("member type const_reference is deprecated in C++17")]]
#endif
        using const_reference = value_type const &;

#if __cplusplus > 201402L
        [[deprecated("member type size_type is deprecated in C++17")]]
#endif
        using size_type = std::size_t;

#if __cplusplus > 201402L
        [[deprecated("member type difference_type is deprecated in C++17")]]
#endif
        using difference_type = std::ptrdiff_t;

        template <class U>
#if __cplusplus > 201402L
        [[deprecated("member template type rebind is deprecated in C++17")]]
#endif
        struct rebind
        {
            using other = allocator <U>;
        };

        allocator (void) noexcept = default;
        ~allocator (void) noexcept = default;

#if __cplusplus > 201402L
        [[deprecated(
            "member function allocate (with hint) is deprecated in C++17"
        )]]
#endif
        pointer allocate (size_type n, void const *)
        {
            return new value_type [n];
        }

        value_type * allocate (size_type n) const
        {
            return new value_type [n];
        }

        void deallocate (value_type * p, std::size_t) const noexcept
        {
            value_type::operator delete [] (p);
        }

#if __cplusplus > 201402L
        [[deprecated("member function address is deprecated in C++17")]]
#endif
        pointer address (reference r) const noexcept
        {
            return &r;
        }

#if __cplusplus > 201402L
        [[deprecated("member function address is deprecated in C++17")]]
#endif
        const_pointer address (const_reference r) const noexcept
        {
            return &r;
        }

#if __cplusplus > 201402L
        [[deprecated("member function max_size is deprecated in C++17")]]
#endif
        size_type max_size (void) const noexcept
        {
            return std::numeric_limits <size_type>::max () /
                sizeof (value_type);
        }

        template <class U, class ... Args>
#if __cplusplus > 201402L
        [[deprecated("member function construct is deprecated in C++17")]]
#endif
        void construct (U * p, Args && ... args) const
        {
            ::new (static_cast <void *> (p)) U (std::forward <Args> (args)...);
        }

        template <class U>
#if __cplusplus > 201402L
        [[deprecated("member function construct is deprecated in C++17")]]
#endif
        void destroy (U * p) const noexcept (noexcept (p->~U ()))
        {
            p->~U ();
        }

        template <class U>
        bool operator== (allocator <U> const &) const noexcept
        {
            return std::is_same <
                typename allocator <U>::value_type, value_type
            >::value;
        }

        template <class U>
        bool operator!= (allocator <U> const &) const noexcept
        {
            return !std::is_same <
                typename allocator <U>::value_type, value_type
            >::value;
        }
    };

namespace detail
{
    template <typename SIMDType, typename ... SIMDTypes>
    struct common_lane_count;

    template <typename SIMDType>
    struct common_lane_count <SIMDType>
    {
        using type = std::integral_constant <
            std::size_t, simd::simd_traits <SIMDType>::lanes
        >;
    };

    template <typename SIMDType1, typename SIMDType2, typename ... SIMDTypes>
    struct common_lane_count <SIMDType1, SIMDType2, SIMDTypes...>
    {
        using type = typename std::conditional <
            simd_traits <SIMDType1>::lanes == simd_traits <SIMDType2>::lanes,
            typename common_lane_count <SIMDType2, SIMDTypes...>::type,
            void
        >::type;

        static_assert (
            !std::is_same <type, void>::value,
            "no common lane count for SIMD vector types"
        );
    };

    template <typename ... SIMDTypes>
    struct common_integral_type
    {
        using type = typename std::common_type <
            typename simd_traits <SIMDTypes>::integral_type...
        >::type;
    };

    template <typename F, typename ... SIMDTypes>
    using function_result = typename std::result_of <
        F (typename simd::simd_traits <SIMDTypes>::value_type const &...)
    >::type;

    template <typename F, typename ... SIMDTypes>
    using transform_result = simd_type <
        typename std::conditional <
            is_boolean_scalar <function_result <F, SIMDTypes...>>::value,
            typename common_integral_type <SIMDTypes...>::type,
            function_result <F, SIMDTypes...>
        >::type,
        common_lane_count <SIMDTypes...>::type::value,
        typename std::conditional <
            std::is_integral <function_result <F, SIMDTypes...>>::value ||
            std::is_floating_point <function_result <F, SIMDTypes...>>::value,
            arithmetic_tag,
            typename std::conditional <
                is_complex_scalar <function_result <F, SIMDTypes...>>::value,
                complex_tag,
                typename std::conditional <
                    is_boolean_scalar <function_result <F, SIMDTypes...>>::value,
                    boolean_tag,
                    arithmetic_tag
                >::type
            >::type
        >::type
    >;

    template <
        std::size_t ... L, typename F, typename SIMDType
    >
    constexpr transform_result <F, SIMDType>
        transform_impl (util::index_sequence <L...>,
                        F && f, SIMDType const & sv)
        noexcept (noexcept (
            std::forward <F> (f) (
                std::declval <typename simd::simd_traits <SIMDType>::value_type>
                    ()
            )
        ))
    {
        static_assert (
            is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );

        return transform_result <F, SIMDType> {
            std::forward <F> (f) (sv.template value <L> ())...
        };
    }

    template <
        std::size_t ... L, typename F, typename SIMDType1, typename SIMDType2
    >
    constexpr transform_result <F, SIMDType1, SIMDType2>
        transform_impl (util::index_sequence <L...>, F && f,
                        SIMDType1 const & sv1, SIMDType2 const & sv2)
        noexcept (noexcept (
            std::forward <F> (f) (
                std::declval <
                    typename simd::simd_traits <SIMDType1>::value_type
                > (),
                std::declval <
                    typename simd::simd_traits <SIMDType2>::value_type
                > ()
            )
        ))
    {
        static_assert (
            is_simd_type <SIMDType1>::value,
            "template parameter SIMDType1 must be a simd type"
        );
        static_assert (
            is_simd_type <SIMDType2>::value,
            "template parameter SIMDType2 must be a simd type"
        );

        return transform_result <F, SIMDType1, SIMDType2> {
            std::forward <F> (f) (
                sv1.template value <L> (), sv2.template value <L> ()
            )...
        };
    }

    template <
        std::size_t ... L, typename F,
        typename SIMDType1, typename SIMDType2, typename SIMDType3
    >
    constexpr transform_result <F, SIMDType1, SIMDType2, SIMDType3>
        transform_impl (util::index_sequence <L...>, F && f,
                        SIMDType1 const & sv1,
                        SIMDType2 const & sv2,
                        SIMDType3 const & sv3)
        noexcept (noexcept (
            std::forward <F> (f) (
                std::declval <
                    typename simd::simd_traits <SIMDType1>::value_type
                > (),
                std::declval <
                    typename simd::simd_traits <SIMDType2>::value_type
                > (),
                std::declval <
                    typename simd::simd_traits <SIMDType3>::value_type
                > ()
            )
        ))
    {
        static_assert (
            is_simd_type <SIMDType1>::value,
            "template parameter SIMDType1 must be a simd type"
        );
        static_assert (
            is_simd_type <SIMDType2>::value,
            "template parameter SIMDType2 must be a simd type"
        );
        static_assert (
            is_simd_type <SIMDType3>::value,
            "template parameter SIMDType3 must be a simd type"
        );

        return transform_result <F, SIMDType1, SIMDType2, SIMDType3> {
            std::forward <F> (f) (
                sv1.template value <L> (),
                sv2.template value <L> (),
                sv3.template value <L> ()
            )...
        };
    }

    template <
        std::size_t ... L, typename F,
        typename SIMDType1, typename SIMDType2,
        typename SIMDType3, typename SIMDType4
    >
    constexpr transform_result <F, SIMDType1, SIMDType2, SIMDType3, SIMDType4>
        transform_impl (util::index_sequence <L...>, F && f,
                        SIMDType1 const & sv1,
                        SIMDType2 const & sv2,
                        SIMDType3 const & sv3,
                        SIMDType4 const & sv4)
        noexcept (noexcept (
            std::forward <F> (f) (
                std::declval <
                    typename simd::simd_traits <SIMDType1>::value_type
                > (),
                std::declval <
                    typename simd::simd_traits <SIMDType2>::value_type
                > (),
                std::declval <
                    typename simd::simd_traits <SIMDType3>::value_type
                > (),
                std::declval <
                    typename simd::simd_traits <SIMDType4>::value_type
                > ()
            )
        ))
    {
        static_assert (
            is_simd_type <SIMDType1>::value,
            "template parameter SIMDType1 must be a simd type"
        );
        static_assert (
            is_simd_type <SIMDType2>::value,
            "template parameter SIMDType2 must be a simd type"
        );
        static_assert (
            is_simd_type <SIMDType3>::value,
            "template parameter SIMDType3 must be a simd type"
        );
        static_assert (
            is_simd_type <SIMDType4>::value,
            "template parameter SIMDType4 must be a simd type"
        );

        return transform_result <F, SIMDType1, SIMDType2, SIMDType3, SIMDType4>
        {
            std::forward <F> (f) (
                sv1.template value <L> (),
                sv2.template value <L> (),
                sv3.template value <L> (),
                sv4.template value <L> ()
            )...
        };
    }
}   // namespace detail

    /*
     * Compute a new SIMD vector containing the function results of each
     * collection of lane values of the original SIMD vectors.
     */
    template <typename F, typename ... SIMDTypes>
    constexpr detail::transform_result <F, SIMDTypes...>
        transform (F && f, SIMDTypes const &... sv)
        noexcept (noexcept (
            std::forward <F> (f) (
                std::declval <
                    typename simd::simd_traits <SIMDTypes>::value_type
                > () ...
            )
        ))
    {
        using common_lanes =
            typename detail::common_lane_count <SIMDTypes...>::type;

        return detail::transform_impl (
            detail::util::make_index_sequence <common_lanes::value> {},
            std::forward <F> (f), sv...
        );
    }

    /*
     * Compute a new SIMD vector containing the hash values of each lane of the
     * original SIMD vector, optionally with a provided hash function.
     */
    template <typename SIMDType>
    struct hash
    {
        static_assert (
            detail::is_simd_type <SIMDType>::value,
            "template parameter SIMDType must be a simd type"
        );

    private:
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

    public:
        using argument_type = SIMDType;
        using result_type   = simd::simd_type <std::size_t, traits_type::lanes>;

        constexpr result_type operator() (argument_type const & sv) const
            noexcept (noexcept (
                std::declval <std::hash <value_type>> () (
                    std::declval <value_type> ()
                )
            ))
        {
            return transform (std::hash <value_type> {}, sv);
        }

        template <typename HashFn>
        constexpr result_type operator() (HashFn && h, argument_type const & sv)
            const noexcept (noexcept (
                std::forward <HashFn> (h) (std::declval <value_type> ())
            ))
        {
            return transform (std::forward <HashFn> (h), sv);
        }
    };

namespace math
{
    /*
     * Computes the sum across the SIMD vector by the given binary operation.
     */
    template <typename SIMDType, typename U, typename BinaryOp>
    U accumulate (SIMDType const & v, U init, BinaryOp op)
    {
        return std::accumulate (v.begin (), v.end (), init, op);
    }

    /*
     * Computes the inner product of two arithmetic (non-boolean) SIMD vectors.
     */
    template <typename SIMDType, std::size_t lanes>
    auto inner_product (SIMDType const & sv1, SIMDType const & sv2) noexcept
        -> typename simd_traits <SIMDType>::value_type
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return simd::math::accumulate (
            sv1 * sv2, value_type {0}, std::plus <value_type> {}
        );
    }

    /*
     * Returns a SIMD vector of the real components of a complex SIMD vector.
     */
    template <typename ComplexSIMDType>
    constexpr auto real (ComplexSIMDType const & sv) noexcept
        -> decltype (sv.real ())
    {
        return sv.real ();
    }

    /*
     * Returns a SIMD vector of the real components of a complex SIMD vector.
     */
    template <typename ComplexSIMDType>
    constexpr auto imag (ComplexSIMDType const & sv) noexcept
        -> decltype (sv.imag ())
    {
        return sv.imag ();
    }

    /*
     * Computes two SIMD vectors respectively containing the pairwise
     * quotient and remainder of integral division.
     */
    template <typename SIMDType>
    std::pair <SIMDType, SIMDType>
        div (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;

        std::pair <SIMDType, SIMDType> qr;

        for (std::size_t i = 0; i < traits_type::lanes; ++i) {
            auto const result = std::div (u.value (i), v.value (i));
            qr.first.set  (i, result.quot);
            qr.second.set (i, result.rem);
        }

        return qr;
    }

    /*
     * Computes the absolute value for each lane of a SIMD vector using
     * std::abs.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::abs (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > abs (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::abs (a); }, v
        );
    }

    /*
     * Computes the absolute value for each lane of a SIMD vector using
     * std::fabs.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::fabs (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > fabs (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::fabs (a); }, v
        );
    }

    /*
     * Computes the phase angle for each lane of a complex SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::arg (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > arg (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::arg (a); }, v
        );
    }

    /*
     * Computes the hypotenuse (sqrt (x^2 + y^2)) for each pairwise lane of
     * SIMD vectors without undue underflow or overflow in intermediate steps.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::hypot (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > hypot (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept {
                return std::hypot (a, b);
            },
            u, v
        );
    }

#if __cplusplus > 201402L
    /*
     * Computes the hypotenuse (sqrt (x^2 + y^2 + z^2)) for each tripple-wise
     * lane of SIMD vectors without undue underflow or overflow in intermediate
     * steps.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::hypot (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > hypot (SIMDType const & u, SIMDType const & v, SIMDType const & w)
        noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a,
                value_type const & b,
                value_type const & c) noexcept
            {
                return std::hypot (a, b, c);
            },
            u, v, w
        );
    }
#endif

    /*
     * Computes the euclidean norm for each lane of a complex SIMD vector.
     */
    template <typename ComplexSIMDType>
    simd_type <
        typename simd_traits <ComplexSIMDType>::lane_type,
        simd_traits <ComplexSIMDType>::lanes
    > norm (ComplexSIMDType const & v) noexcept
    {
        return simd::math::hypot (v.real (), v.imag ());
    }

    /*
     * Computes the complex conjugate for each lane of a complex SIMD vector.
     */
    template <typename ComplexSIMDType>
    constexpr ComplexSIMDType conj (ComplexSIMDType const & v) noexcept
    {
        return ComplexSIMDType {v.real (), -v.imag ()};
    }

    /*
     * Computes the projection onto the Riemann Sphere for each lane of a
     * complex SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::proj (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > proj (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::proj (a); }, v
        );
    }

    /*
     * Computes the exponential for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::exp (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > exp (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::exp (a); }, v
        );
    }

    /*
     * Computes the exponent base 2 for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::exp2 (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > exp2 (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::exp2 (a); }, v
        );
    }

    /*
     * Computes the exponential minus 1 for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::expm1 (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > expm1 (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::expm1 (a); }, v
        );
    }

    /*
     * Computes the natural logarithm for each lane of a SIMD vector.
     * For complex types branch cuts occur along the negative real axis.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::log (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > log (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::log (a); }, v
        );
    }

    /*
     * Computes the logarithm base 10 for each lane of a SIMD vector.
     * For complex types branch cuts occur along the negative real axis.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::log10 (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > log10 (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::log10 (a); }, v
        );
    }

    /*
     * Computes the logarithm base 2 for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::log2 (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > log2 (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::log2 (a); }, v
        );
    }

    /*
     * Computes the natural logarithm for each lane of a SIMD vector
     * plus one.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::log1p (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > log1p (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::log1p (a); }, v
        );
    }

    /*
     * Computes the square root for each lane of a SIMD vector.
     * For complex types the result lies in the right half-plane.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::sqrt (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > sqrt (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::sqrt (a); }, v
        );
    }

    /*
     * Computes the cube root for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::cbrt (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > cbrt (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::cbrt (a); }, v
        );
    }

    /*
     * Computes the power x^y for each lane, pairwise of two SIMD vectors.
     */
    template <typename SIMDTypeBase, typename SIMDTypeExp>
    simd_type <
        decltype (std::pow (
            std::declval <typename simd_traits <SIMDTypeBase>::value_type> (),
            std::declval <typename simd_traits <SIMDTypeExp>::value_type> ()
        )),
        simd_traits <SIMDTypeBase>::lanes
    > pow (SIMDTypeBase const & base, SIMDTypeExp const & exp) noexcept
    {
        using base_traits_type = simd_traits <SIMDTypeBase>;
        using base_value_type  = typename base_traits_type::value_type;

        using exp_traits_type = simd_traits <SIMDTypeExp>;
        using exp_value_type  = typename exp_traits_type::value_type;

        static_assert (
            base_traits_type::lanes == exp_traits_type::lanes,
            "cannot apply pow function to SIMD vectors of different lenghts"
        );

        return transform (
            [] (base_value_type const & b, exp_value_type const & e) noexcept
            {
                return std::pow (b, e);
            },
            base, exp
        );
    }

    /*
     * Computes the sine for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::sin (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > sin (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::sin (a); }, v
        );
    }

    /*
     * Computes the arcsine for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::asin (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > asin (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::asin (a); }, v
        );
    }

    /*
     * Computes the cosine for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::cos (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > cos (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::cos (a); }, v
        );
    }

    /*
     * Computes the arcosine for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::acos (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > acos (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::acos (a); }, v
        );
    }

    /*
     * Computes the tangent for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::tan (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > tan (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::tan (a); }, v
        );
    }

    /*
     * Computes the arctangent for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::atan (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > atan (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::atan (a); }, v
        );
    }

    /*
     * Computes the arctangent considering signs for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::atan2 (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > atan2 (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::atan2 (a); }, v
        );
    }

    /*
     * Computes the hyperbolic sine for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::sinh (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > sinh (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::sinh (a); }, v
        );
    }

    /*
     * Computes the area hyperbolic sine for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::asinh (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > asinh (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::asinh (a); }, v
        );
    }

    /*
     * Computes the hyperbolic cosine for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::cosh (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > cosh (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::cosh (a); }, v
        );
    }

    /*
     * Computes the area hyperbolic cosine for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::acosh (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > acosh (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::acosh (a); }, v
        );
    }

    /*
     * Computes the hyperbolic tangent for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::tanh (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > tanh (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::tanh (a); }, v
        );
    }

    /*
     * Computes the area hyperbolic tangent for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::atanh (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > atanh (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::atanh (a); }, v
        );
    }

    /*
     * Computes the error function for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::erf (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > erf (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::erf (a); }, v
        );
    }

    /*
     * Computes the complementary error function for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::erfc (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > erfc (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::erfc (a); }, v
        );
    }

    /*
     * Computes the gamma function for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::tgamma (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > tgamma (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::tgamma (a); }, v
        );
    }

    /*
     * Computes the natural logarithm of the gramma function for each lane of a
     * SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::lgamma (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > lgamma (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::lgamma (a); }, v
        );
    }

    /*
     * Computes the pairwise maximum of two SIMD vectors.
     */
    template <typename SIMDType>
    cpp14_constexpr SIMDType max (SIMDType const & u, SIMDType const & v)
        noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a, value_type const & b) cpp17_constexpr
            {
                return std::max (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the pairwise minimum of two SIMD vectors.
     */
    template <typename SIMDType>
    cpp14_constexpr SIMDType min (SIMDType const & u, SIMDType const & v)
        noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a, value_type const & b) cpp17_constexpr
            {
                return std::min (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the floating point pairwise maximum of two SIMD vectors.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::fmax (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > fmax (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a, value_type const & b) {
                return std::fmax (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the floating point pairwise minimum of two SIMD vectors.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::fmin (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > fmin (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a, value_type const & b) {
                return std::fmin (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the positive floating point pairwise difference of two SIMD
     * vectors.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::fdim (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > fdim (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a, value_type const & b) {
                return std::fdim (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the ceil for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::ceil (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > ceil (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::ceil (a); }, v
        );
    }

    /*
     * Computes the floor for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::floor (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > floor (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::floor (a); }, v
        );
    }

    /*
     * Computes the truncation value for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::trunc (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > trunc (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::trunc (a); }, v
        );
    }

    /*
     * Computes the nearest integer value for each lane of a SIMD vector,
     * rounding away from zero in half-way cases. Returns the result as a SIMD
     * vector of either the original floating point lane type or, if the lane
     * type is a non-floating point arithmetic type, the promoted type (double).
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::round (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > round (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::round (a); }, v
        );
    }

    /*
     * Computes the nearest integer value for each lane of a SIMD vector,
     * rounding away from zero in half-way cases. Returns the result as a SIMD
     * vector of long values.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::lround (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > lround (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::lround (a); }, v
        );
    }

    /*
     * Computes the nearest integer value for each lane of a SIMD vector,
     * rounding away from zero in half-way cases. Returns the result as a SIMD
     * vector of long long values.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::llround (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > llround (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::llround (a); }, v
        );
    }

    /*
     * Computes the nearest integer value for each lane of a SIMD vector using
     * the current rounding mode.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::nearbyint (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > nearbyint (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::nearbyint (a); }, v
        );
    }

    /*
     * Computes the nearest integer using current rounding mode with F.P.
     * exception if the result differs. Returns the result as a SIMD vector of
     * either the original floating point lane type or, if the lane type is a
     * non-floating point arithmetic type, the promoted type (double).
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::rint (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > rint (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::rint (a); }, v
        );
    }

    /*
     * Computes the nearest integer using current rounding mode with F.P.
     * exception if the result differs. Returns the result as a SIMD vector of
     * long values.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::lrint (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > lrint (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::lrint (a); }, v
        );
    }

    /*
     * Computes the nearest integer using current rounding mode with F.P.
     * exception if the result differs. Returns the result as a SIMD vector of
     * long long values.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::llrint (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > llrint (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::llrint (a); }, v
        );
    }

    /*
     * Computes the decomposition of a number into significand and a power of 2,
     * returning a pair of SIMD vectors with the above values, respectively.
     */
    template <typename SIMDType>
    std::pair <
        simd_type <
            decltype (std::frexp (
                std::declval <typename simd_traits <SIMDType>::value_type> (),
                std::declval <int *> ()
            )),
            simd_traits <SIMDType>::lanes
        >,
        simd_type <int, simd_traits <SIMDType>::lanes>
    > frexp (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = std::pair <
            simd_type <
                decltype (std::frexp (
                    std::declval <value_type> (), std::declval <int *> ()
                )),
                traits_type::lanes
            >,
            simd_type <int, traits_type::lanes>
        >;

        result_type result {};
        for (std::size_t i = 0; i < traits_type::lanes; ++i) {
            int exp;
            result.first.set (i, std::frexp (v.value (i), &exp));
            result.second.set (i, exp);
        }
        return result;
    }

    /*
     * Computes a value times the number 2 raised to the exp power for each
     * pairwise lanes of SIMD vectors. The value type of the second SIMD vector
     * must be implicitly convertible to int.
     */
    template <typename SIMDTypeMultiplicand, typename SIMDTypeExp>
    simd_type <
        decltype (std::ldexp (
            std::declval <
                typename simd_traits <SIMDTypeMultiplicand>::value_type
            > (),
            std::declval <typename simd_traits <SIMDTypeExp>::value_type> ()
        )),
        simd_traits <SIMDTypeMultiplicand>::lanes
    > ldexp (SIMDTypeMultiplicand const & x, SIMDTypeExp const & exp) noexcept
    {
        using m_traits_type = simd_traits <SIMDTypeMultiplicand>;
        using m_value_type  = typename m_traits_type::value_type;

        using exp_traits_type = simd_traits <SIMDTypeExp>;
        using exp_value_type  = typename exp_traits_type::value_type;

        static_assert (
            m_traits_type::lanes == exp_traits_type::lanes,
            "cannot apply ldexp function to SIMD vectors of different lenghts"
        );

        return transform (
            [] (m_value_type const & m, exp_value_type const & e) noexcept
            {
                return std::ldexp (m, e);
            },
            x, exp
        );
    }

    /*
     * Computes the decomposition of floating point values into integral and
     * fractional parts for each lane of a SIMD vector. Returns a pair
     * of SIMD vectors containing the integral and fractional parts,
     * respectively.
     */
    template <typename SIMDType>
    std::pair <
        simd_type <
            decltype (std::modf (
                std::declval <typename simd_traits <SIMDType>::value_type> (),
                std::declval <typename simd_traits <SIMDType>::value_type *> ()
            )),
            simd_traits <SIMDType>::lanes
        >,
        simd_type <
            typename simd_traits <SIMDType>::value_type,
            simd_traits <SIMDType>::lanes
        >
    > modf (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = std::pair <
            simd_type <
                decltype (std::modf (
                    std::declval <value_type> (), std::declval <value_type *> ()
                )),
                traits_type::lanes
            >,
            simd_type <value_type, traits_type::lanes>
        >;

        result_type result {};
        for (std::size_t i = 0; i < traits_type::lanes; ++i) {
            value_type integral_val;
            result.first.set (i, std::modf (v.value (i), &integral_val));
            result.second.set (i, integral_val);
        }
        return result;
    }

    /*
     * Computes a value times the number FLT_RADIX raised to the exp power for
     * each pairwise lanes of SIMD vectors. The value type of the second SIMD
     * vector must be convertible to int.
     */
    template <typename SIMDTypeMultiplicand, typename SIMDTypeExp>
    simd_type <
        decltype (std::scalbn (
            std::declval <
                typename simd_traits <SIMDTypeMultiplicand>::value_type
            > (),
            std::declval <typename simd_traits <SIMDTypeExp>::value_type> ()
        )),
        simd_traits <SIMDTypeMultiplicand>::lanes
    > scalbn (SIMDTypeMultiplicand const & x, SIMDTypeExp const & exp) noexcept
    {
        using m_traits_type = simd_traits <SIMDTypeMultiplicand>;
        using m_value_type  = typename m_traits_type::value_type;

        using exp_traits_type = simd_traits <SIMDTypeExp>;
        using exp_value_type  = typename exp_traits_type::value_type;

        static_assert (
            m_traits_type::lanes == exp_traits_type::lanes,
            "cannot apply scalbn function to SIMD vectors of different lenghts"
        );

        return transform (
            [] (m_value_type const & m, exp_value_type const & e) noexcept
            {
                return std::scalbn (m, e);
            },
            x, exp
        );
    }

    /*
     * Computes a value times the number FLT_RADIX raised to the exp power for
     * each pairwise lanes of SIMD vectors. The value type of the second SIMD
     * vector must be convertible to long.
     */
    template <typename SIMDTypeMultiplicand, typename SIMDTypeExp>
    simd_type <
        decltype (std::scalbln (
            std::declval <
                typename simd_traits <SIMDTypeMultiplicand>::value_type
            > (),
            std::declval <typename simd_traits <SIMDTypeExp>::value_type> ()
        )),
        simd_traits <SIMDTypeMultiplicand>::lanes
    > scalbln (SIMDTypeMultiplicand const & x, SIMDTypeExp const & exp) noexcept
    {
        using m_traits_type = simd_traits <SIMDTypeMultiplicand>;
        using m_value_type  = typename m_traits_type::value_type;

        using exp_traits_type = simd_traits <SIMDTypeExp>;
        using exp_value_type  = typename exp_traits_type::value_type;

        static_assert (
            m_traits_type::lanes == exp_traits_type::lanes,
            "cannot apply scalbln function to SIMD vectors of different lenghts"
        );

        return transform (
            [] (m_value_type const & m, exp_value_type const & e) noexcept
            {
                return std::scalbln (m, e);
            },
            x, exp
        );
    }

    /*
     * Extracts the integral exponent of a floating point value for each lane
     * of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::ilogb (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > ilogb (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::ilogb (a); }, v
        );
    }

    /*
     * Extracts the floating point radix independent exponent of a floating
     * point value, as a floating point result for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::logb (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > logb (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::logb (a); }, v
        );
    }

    /*
     * Computes the next representable value from the floating point value from
     * to the floating point value to for each lane of SIMD vectors.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::nextafter (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > nextafter (SIMDType const & from, SIMDType const & to) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & f, value_type const & t) noexcept
            {
                return std::nextafter (f, t);
            },
            from, to
        );
    }

    /*
     * Computes the next representable value from the floating point value from
     * to the floating point value to for each lane of SIMD vectors.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::nexttoward (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > nexttoward (SIMDType const & from, SIMDType const & to) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & f, value_type const & t) noexcept
            {
                return std::nexttoward (f, t);
            },
            from, to
        );
    }

    /*
     * Computes a floating point value with the magnitude of the first floating
     * point value and the sign of the second floating point value for each lane
     * of SIMD vectors.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::copysign (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > copysign (SIMDType const & mag, SIMDType const & sgn) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & m, value_type const & s) noexcept
            {
                return std::copysign (m, s);
            },
            mag, sgn
        );
    }

    /*
     * Classifies the floating point value into one of: zero, subnormal, normal,
     * infinite, NaN, or an implementation defined category for each lane of a
     * SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::fpclassify (
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > fpclassify (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a) noexcept { return std::fpclassify (a); },
            v
        );
    }

    /*
     * Determines if a floating point value is finite for each lane of a SIMD
     * vector.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > isfinite (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a) noexcept { return std::isfinite (a); }, v
        ).template to <result_type> ();
    }

    /*
     * Determines if a floating point value is infinite for each lane of a SIMD
     * vector.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > isinf (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a) noexcept { return std::isinf (a); }, v
        ).template to <result_type> ();
    }

    /*
     * Determines if a floating point value is not-a-number for each lane of a
     * SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > isnan (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a) noexcept { return std::isnan (a); }, v
        ).template to <result_type> ();
    }

    /*
     * Determines if a floating point value is normal (neither zero, subnormal,
     * infinite, nor NaN) for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > isnormal (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a) noexcept { return std::isnormal (a); }, v
        ).template to <result_type> ();
    }

    /*
     * Determines if a floating point value is negative for each lane of a SIMD
     * vector.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > signbit (SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a) noexcept { return std::signbit (a); }, v
        ).template to <result_type> ();
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * greater than another floating point value for each lane of SIMD vectors.
     * This function does not set floating point exceptions.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > isgreater (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept
            {
                return std::isgreater (a, b);
            },
            u, v
        ).template to <result_type> ();
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * greater than or equal to another floating point value for each lane of
     * SIMD vectors. This function does not set floating point exceptions.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > isgreaterequal (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept
            {
                return std::isgreaterequal (a, b);
            },
            u, v
        ).template to <result_type> ();
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * less than another floating point value for each lane of SIMD vectors.
     * This function does not set floating point exceptions.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > isless (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept
            {
                return std::isless (a, b);
            },
            u, v
        ).template to <result_type> ();
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * less than or equal to another floating point value for each lane of
     * SIMD vectors. This function does not set floating point exceptions.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > islessequal (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept
            {
                return std::islessequal (a, b);
            },
            u, v
        ).template to <result_type> ();
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * less than or greater than another floating point value for each lane of
     * SIMD vectors. This function does not set floating point exceptions.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > islessgreater (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept
            {
                return std::islessgreater (a, b);
            },
            u, v
        ).template to <result_type> ();
    }

    /*
     * Determines the pairwise result of whether a floating point value is
     * unordered with another floating point value for each lane of SIMD
     * vectors. This function does not set floating point exceptions.
     */
    template <typename SIMDType>
    simd_type <
        typename simd_traits <SIMDType>::integral_type,
        simd_traits <SIMDType>::lanes,
        boolean_tag
    > isunordered (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = simd_type <
            typename simd_traits <SIMDType>::integral_type,
            simd_traits <SIMDType>::lanes,
            boolean_tag
        >;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept
            {
                return std::isunordered (a, b);
            },
            u, v
        ).template to <result_type> ();
    }

    /*
     * Computes the pairwise fmod of two SIMD vectors.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::fmod (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > fmod (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept
            {
                return std::fmod (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the pairwise remainder of two SIMD vectors.
     */
    template <typename SIMDType>
    simd_type <
        decltype (std::remainder (
            std::declval <typename simd_traits <SIMDType>::value_type> (),
            std::declval <typename simd_traits <SIMDType>::value_type> ()
        )),
        simd_traits <SIMDType>::lanes
    > remainder (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & a, value_type const & b) noexcept
            {
                return std::remainder (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the pairwise remainder of two SIMD vectors as well as the sign
     * and at least the three last bits of the division, which is stored in the
     * second SIMD vector of the result pair.
     */
    template <typename SIMDType>
    std::pair <
        simd_type <
            decltype (std::remquo (
                std::declval <typename simd_traits <SIMDType>::value_type> (),
                std::declval <typename simd_traits <SIMDType>::value_type> ()
            )),
            simd_traits <SIMDType>::lanes
        >,
        simd_type <int, simd_traits <SIMDType>::lanes>
    > remquo (SIMDType const & u, SIMDType const & v) noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;
        using result_type = std::pair <
            simd_type <
                decltype (std::remquo (
                    std::declval <value_type> (), std::declval <value_type> ()
                )),
                traits_type::lanes
            >,
            simd_type <int, traits_type::lanes>
        >;

        result_type result {};
        for (std::size_t i = 0; i < traits_type::lanes; ++i) {
            int quo;
            result.first.set (
                i, std::remquo (u.value (i), v.value (i), &quo)
            );
            result.second.set (i, quo);
        }
        return result;
    }

    /*
     * Computes the fused multiply and add operation of three SIMD vectors,
     * in the form (u * v) + w.
     */
    template <typename SIMDType1, typename SIMDType2, typename SIMDType3>
    simd_type <
        decltype (std::fma (
            std::declval <typename simd_traits <SIMDType1>::value_type> (),
            std::declval <typename simd_traits <SIMDType2>::value_type> (),
            std::declval <typename simd_traits <SIMDType3>::value_type> ()
        )),
        simd_traits <SIMDType1>::lanes
    > fma (SIMDType1 const & u, SIMDType2 const & v, SIMDType3 const & w)
        noexcept
    {
        using traits_type1 = simd_traits <SIMDType1>;
        using value_type1  = typename traits_type1::value_type;

        using traits_type2 = simd_traits <SIMDType2>;
        using value_type2  = typename traits_type2::value_type;

        using traits_type3 = simd_traits <SIMDType3>;
        using value_type3  = typename traits_type3::value_type;

        static_assert (
            traits_type1::lanes == traits_type2::lanes &&
            traits_type1::lanes == traits_type3::lanes,
            "cannot compute fma across SIMD vectors of different lengths"
        );

        return transform (
            [] (value_type1 const & a,
                value_type2 const & b,
                value_type3 const & c) noexcept
            {
                return std::fma (a, b, c);
            },
            u, v, w
        );
    }

#if __cplusplus > 201402L
    /*
     * Computes the pairwise gcd of two SIMD vectors.
     */
    template <typename SIMDType1, typename SIMDType2>
    constexpr simd_type <
        decltype (std::gcd (
            std::declval <typename simd_traits <SIMDType1>::value_type> (),
            std::declval <typename simd_traits <SIMDType2>::value_type> ()
        )),
        simd_traits <SIMDType1>::lanes
    > gcd (SIMDType1 const & u, SIMDType2 const & v) noexcept
    {
        using traits_type1 = simd_traits <SIMDType1>;
        using value_type1  = typename traits_type1::value_type;

        using traits_type2 = simd_traits <SIMDType2>;
        using value_type2  = typename traits_type2::value_type;

        static_assert (
            traits_type1::lanes == traits_type2::lanes,
            "cannot compute gcd across SIMD vectors of different length"
        );

        return transform (
            [] (value_type1 const & a, value_type2 const & b) constexpr noexcept
            {
                return std::gcd (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the pairwise lcm of two SIMD vectors.
     */
    template <typename SIMDType1, typename SIMDType2>
    constexpr simd_type <
        decltype (std::lcm (
            std::declval <typename simd_traits <SIMDType1>::value_type> (),
            std::declval <typename simd_traits <SIMDType2>::value_type> ()
        )),
        simd_traits <SIMDType1>::lanes
    > lcm (SIMDType1 const & u, SIMDType2 const & v) noexcept
    {
        using traits_type1 = simd_traits <SIMDType1>;
        using value_type1  = typename traits_type1::value_type;

        using traits_type2 = simd_traits <SIMDType2>;
        using value_type2  = typename traits_type2::value_type;

        static_assert (
            traits_type1::lanes == traits_type2::lanes,
            "cannot compute lcm across SIMD vectors of different length"
        );

        return transform (
            [] (value_type1 const & a, value_type2 const & b) constexpr noexcept
            {
                return std::lcm (a, b);
            },
            u, v
        );
    }

    /*
     * Computes the clamped value for each lane of a SIMD vector.
     */
    template <typename SIMDType>
    constexpr SIMDType
        clamp (SIMDType const & u, SIMDType const & lo, SIMDType const & hi)
        noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        uisng value_type  = typename traits_type::value_type;

        return transform (
            [] (value_type const & val,
                value_type const & l,
                value_type const & h) constexpr noexcept
            {
                return std::clamp (val, l, h);
            },
            u, lo, hi
        );
    }

    /*
     * Computes the clamped value for each lane of a SIMD vector using the
     * provided comparison function.
     */
    template <typename SIMDType, typename Compare>
    constexpr SIMDType clamp (SIMDType const & u,
                              SIMDType const & lo,
                              SIMDType const & hi,
                              Compare && comp)
        noexcept
    {
        using traits_type = simd_traits <SIMDType>;
        uisng value_type  = typename traits_type::value_type;

        return transform (
            [c = std::forward <Compare> (comp)] (value_type const & val,
                                                 value_type const & l,
                                                 value_type const & h)
                constexpr noexcept
            {
                return std::clamp (val, l, h, c);
            },
            u, lo, hi
        );
    }
#endif
}   // namespace math

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
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 32 8-bit lanes */
    using bool8x32_t = simd_type <std::int8_t, 32, boolean_tag>;
    using int8x32_t  = simd_type <std::int8_t, 32>;
    using uint8x32_t = simd_type <std::uint8_t, 32>;

    /* 64 8-bit lanes */
    using bool8x64_t = simd_type <std::int8_t, 64, boolean_tag>;
    using int8x64_t  = simd_type <std::int8_t, 64>;
    using uint8x64_t = simd_type <std::uint8_t, 64>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 16 16-bit lanes */
    using bool16x16_t = simd_type <std::int16_t, 16, boolean_tag>;
    using int16x16_t  = simd_type <std::int16_t, 16>;
    using uint16x16_t = simd_type <std::uint16_t, 16>;

    /* 32 16-bit lanes */
    using bool16x32_t = simd_type <std::int16_t, 32, boolean_tag>;
    using int16x32_t  = simd_type <std::int16_t, 32>;
    using uint16x32_t = simd_type <std::uint16_t, 32>;

    /* 2 32-bit lanes */
    using bool32x2_t          = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t           = simd_type <std::int32_t, 2>;
    using uint32x2_t          = simd_type <std::uint32_t, 2>;
    using float32x2_t         = simd_type <float, 2>;
    using complex_float32x2_t = simd_type <float, 2, complex_tag>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 8 32-bit lanes */
    using bool32x8_t          = simd_type <std::int32_t, 8, boolean_tag>;
    using int32x8_t           = simd_type <std::int32_t, 8>;
    using uint32x8_t          = simd_type <std::uint32_t, 8>;
    using float32x8_t         = simd_type <float, 8>;
    using complex_float32x8_t = simd_type <float, 8, complex_tag>;

    /* 16 32-bit lanes */
    using bool32x16_t          = simd_type <std::int32_t, 16, boolean_tag>;
    using int32x16_t           = simd_type <std::int32_t, 16>;
    using uint32x16_t          = simd_type <std::uint32_t, 16>;
    using float32x16_t         = simd_type <float, 16>;
    using complex_float32x16_t = simd_type <float, 16, complex_tag>;

    /* 1 64-bit lane */
    using bool64x1_t          = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t           = simd_type <std::int64_t, 1>;
    using uint64x1_t          = simd_type <std::uint64_t, 1>;
    using float64x1_t         = simd_type <double, 1>;
    using complex_float64x1_t = simd_type <double, 1, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 4 64-bit lanes */
    using bool64x4_t          = simd_type <std::int64_t, 4, boolean_tag>;
    using int64x4_t           = simd_type <std::int64_t, 4>;
    using uint64x4_t          = simd_type <std::uint64_t, 4>;
    using float64x4_t         = simd_type <double, 4>;
    using complex_float64x4_t = simd_type <double, 4, complex_tag>;

    /* 8 64-bit lanes */
    using bool64x8_t          = simd_type <std::int64_t, 8, boolean_tag>;
    using int64x8_t           = simd_type <std::int64_t, 8>;
    using uint64x8_t          = simd_type <std::uint64_t, 8>;
    using float64x8_t         = simd_type <double, 8>;
    using complex_float64x8_t = simd_type <float, 8, complex_tag>;

    /*
     * long double specializations; may be 80-bit (x87), 128-bit,
     * or even a synonym for double floating point types depending
     * on the implementation.
     */

    /* long double x 2 */
    using long_doublex2_t         = simd_type <long double, 2>;
    using complex_long_doublex2_t = simd_type <long double, 2, complex_tag>;

    /* long double x 4 */
    using long_doublex4_t         = simd_type <long double, 4>;
    using complex_long_doublex4_t = simd_type <long double, 4, complex_tag>;

    /* Guaranteed 128-bit integer SIMD vectors */
    /* 1 128-bit lane */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif

    /* 2 128-bit lanes */
#if SIMD_HEADER_CLANG
    using bool128x2_t = simd_type <__int128_t, 2, boolean_tag>;
    using int128x2_t  = simd_type <__int128_t, 2>;
    using uint128x2_t = simd_type <__uint128_t, 2>;
#elif SIMD_HEADER_GNUG
    using bool128x2_t = simd_type <__int128, 2, boolean_tag>;
    using int128x2_t  = simd_type <__int128, 2>;
    using uint128x2_t = simd_type <unsigned __int128, 2>;
#endif

    /* 4 128-bit lanes */
#if SIMD_HEADER_CLANG
    using bool128x4_t = simd_type <__int128_t, 4, boolean_tag>;
    using int128x4_t  = simd_type <__int128_t, 4>;
    using uint128x4_t = simd_type <__uint128_t, 4>;
#elif SIMD_HEADER_GNUG
    using bool128x4_t = simd_type <__int128, 4, boolean_tag>;
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
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;
}   // namespace mmx

namespace sse
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 4 32 bit lanes */
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;
}   // namespace sse

namespace sse2
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
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
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif
}   // namespace sse3

namespace ssse3
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif
}   // namespace ssse3

namespace sse4
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
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
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif
}   // namespace sse4a

namespace avx
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif

    /* ymm registers (256-bit) */
    /* 8 32-bit lanes */
    using float32x8_t         = simd_type <float, 8>;
    using complex_float32x8_t = simd_type <float, 8, complex_tag>;

    /* 4 64-bit lanes */
    using float64x4_t         = simd_type <double, 4>;
    using complex_float64x4_t = simd_type <double, 4, complex_tag>;
}   // namespace avx

namespace avx2
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif

    /* ymm registers (256-bit) */
    /* 8 32-bit lanes */
    using bool32x8_t          = simd_type <std::int32_t, 8, boolean_tag>;
    using int32x8_t           = simd_type <std::int32_t, 8>;
    using uint32x8_t          = simd_type <std::uint32_t, 8>;
    using float32x8_t         = simd_type <float, 8>;
    using complex_float32x8_t = simd_type <float, 8, complex_tag>;

    /* 4 64-bit lanes */
    using bool64x4_t          = simd_type <std::int64_t, 4, boolean_tag>;
    using int64x4_t           = simd_type <std::int64_t, 4>;
    using uint64x4_t          = simd_type <std::uint64_t, 4>;
    using float64x4_t         = simd_type <double, 4>;
    using complex_float64x4_t = simd_type <double, 4, complex_tag>;
}   // namespace avx2

namespace avx512
{
    /* mmx registers (64-bit) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t  = simd_type <std::int32_t, 2>;
    using uint32x2_t = simd_type <std::uint32_t, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* xmm registers (128-bit) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lanes */
    using bool64x2_t          = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t           = simd_type <std::int64_t, 2>;
    using uint64x2_t          = simd_type <std::uint64_t, 2>;
    using float64x2_t         = simd_type <double, 2>;
    using complex_float64x2_t = simd_type <double, 2, complex_tag>;

    /* 1 128-bit lane (x86 doublequadword) */
#if SIMD_HEADER_CLANG
    using bool128x1_t = simd_type <__int128_t, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128_t, 1>;
    using uint128x1_t = simd_type <__uint128_t, 1>;
#elif SIMD_HEADER_GNUG
    using bool128x1_t = simd_type <__int128, 1, boolean_tag>;
    using int128x1_t  = simd_type <__int128, 1>;
    using uint128x1_t = simd_type <unsigned __int128, 1>;
#endif

    /* ymm registers (256-bit) */
    /* 8 32-bit lanes */
    using bool32x8_t          = simd_type <std::int32_t, 8, boolean_tag>;
    using int32x8_t           = simd_type <std::int32_t, 8>;
    using uint32x8_t          = simd_type <std::uint32_t, 8>;
    using float32x8_t         = simd_type <float, 8>;
    using complex_float32x8_t = simd_type <float, 8, complex_tag>;

    /* 4 64-bit lanes */
    using bool64x4_t          = simd_type <std::int64_t, 4, boolean_tag>;
    using int64x4_t           = simd_type <std::int64_t, 4>;
    using uint64x4_t          = simd_type <std::uint64_t, 4>;
    using float64x4_t         = simd_type <double, 4>;
    using complex_float64x4_t = simd_type <double, 4, complex_tag>;

    /* zmm registers (512-bit) */
    /* 16 32-bit lanes */
    using bool32x16_t          = simd_type <std::int32_t, 16, boolean_tag>;
    using int32x16_t           = simd_type <std::int32_t, 16>;
    using uint32x16_t          = simd_type <std::uint32_t, 16>;
    using float32x16_t         = simd_type <float, 16>;
    using complex_float32x16_t = simd_type <float, 16, complex_tag>;

    /* 8 64-bit lanes */
    using bool64x8_t          = simd_type <std::int64_t, 8, boolean_tag>;
    using int64x8_t           = simd_type <std::int64_t, 8>;
    using uint64x8_t          = simd_type <std::uint64_t, 8>;
    using float64x8_t         = simd_type <double, 8>;
    using complex_float64x8_t = simd_type <double, 8, complex_tag>;
}   // namespace avx512

namespace neon
{
    /* 64-bit registers (ARM doubleword registers -- D0, D1, ...) */
    /* 8 8-bit lanes */
    using bool8x8_t = simd_type <std::int8_t, 8, boolean_tag>;
    using int8x8_t  = simd_type <std::int8_t, 8>;
    using uint8x8_t = simd_type <std::uint8_t, 8>;

    /* 4 16-bit lanes */
    using bool16x4_t = simd_type <std::int16_t, 4, boolean_tag>;
    using int16x4_t  = simd_type <std::int16_t, 4>;
    using uint16x4_t = simd_type <std::uint16_t, 4>;

    /* 2 32-bit lanes */
    using bool32x2_t  = simd_type <std::int32_t, 2, boolean_tag>;
    using int32x2_t   = simd_type <std::int32_t, 2>;
    using uint32x2_t  = simd_type <std::uint32_t, 2>;
    using float32x2_t = simd_type <float, 2>;

    /* 1 64-bit lane */
    using bool64x1_t = simd_type <std::int64_t, 1, boolean_tag>;
    using int64x1_t  = simd_type <std::int64_t, 1>;
    using uint64x1_t = simd_type <std::uint64_t, 1>;

    /* 128-bit registers (ARM quadword registers -- Q0, Q1, ...) */
    /* 16 8-bit lanes */
    using bool8x16_t = simd_type <std::int8_t, 16, boolean_tag>;
    using int8x16_t  = simd_type <std::int8_t, 16>;
    using uint8x16_t = simd_type <std::uint8_t, 16>;

    /* 8 16-bit lanes */
    using bool16x8_t = simd_type <std::int16_t, 8, boolean_tag>;
    using int16x8_t  = simd_type <std::int16_t, 8>;
    using uint16x8_t = simd_type <std::uint16_t, 8>;

    /* 4 32-bit lanes */
    using bool32x4_t          = simd_type <std::int32_t, 4, boolean_tag>;
    using int32x4_t           = simd_type <std::int32_t, 4>;
    using uint32x4_t          = simd_type <std::uint32_t, 4>;
    using float32x4_t         = simd_type <float, 4>;
    using complex_float32x4_t = simd_type <float, 4, complex_tag>;

    /* 2 64-bit lane */
    using bool64x2_t = simd_type <std::int64_t, 2, boolean_tag>;
    using int64x2_t  = simd_type <std::int64_t, 2>;
    using uint64x2_t = simd_type <std::uint64_t, 2>;
}   // namespace neon
}   // namespace simd

#undef cpp14_constexpr

#include <cctype>    // std::ctype_base
#include <istream>   // std::basic_istream, sentry
#include <locale>    // std::num_get, std::use_facet
#include <ostream>   // std::basic_ostream, sentry
#include <sstream>   // std::baisc_ostringstream

/*
 * The following provide overloads for std namespace functions, including:
 *      - operator<< (arbitrary character type streams)
 *      - operator>> (arbitrary character type streams)
 *      - std::hash
 *
 * I/O of 128 bit integers is not supported, and so to use operator<< and
 * operator>> on vector types with 128 bit lane values requires a user
 * defined implementation available for lookup in the calling scope.
 *
 * I/O operations are always of the format:
 *
 * 1. ostream output:
 *
 *      ostream_object << v;
 *
 * is semantically equivalent to:
 *
 *      ostream_object << '(' << v[0] << ';' << ... << v[n] << ')';
 *
 * where n is one less than the number of vector lanes and all flags, locality,
 * and precision of the output stream is preserved over the operation. That is,
 * vector lane values are output in a semi-colon separated tuple format with the
 * correct precision and locality constraints.
 *
 * 2. istream input:
 *
 *      istream_object >> v;
 *
 * has only one expectation: the istream_object must supply at least as many
 * values as there are vector lanes. If this is not the case then
 * std::ios_base::failbit will be set. Besides this requirement the input format
 * is flexible and all non-numeric characters (depending on whether std::dec,
 * std::oct, or std::hex is set) will be consumed in the process of reading
 * the required number of lane values.
 */

namespace simd
{
    template <
        typename SIMDType, typename CharT, typename CharTraits,
        typename = typename std::enable_if <
            simd::detail::is_simd_type <SIMDType>::value
        >::type
    >
    std::basic_ostream <CharT, CharTraits> &
        operator<< (std::basic_ostream <CharT, CharTraits> & os,
                    SIMDType const & v)
    {
        static constexpr std::size_t lanes = simd_traits <SIMDType>::lanes;

        typename std::basic_ostream <CharT, CharTraits>::sentry sentry {os};
        if (sentry) {
            std::basic_ostringstream <CharT, CharTraits> ss;
            ss.flags (os.flags ());
            ss.imbue (os.getloc ());
            ss.precision (os.precision ());

            ss << '(';
            for (std::size_t i = 0; i < lanes - 1; ++i) {
                ss << +v.value (i) << ';';
            }
            ss << +v.value (lanes - 1) << ')';
            os << ss.str ();
        }

        return os;
    }

    template <
        typename SIMDType, typename CharT, typename CharTraits,
        typename = typename std::enable_if <
            simd::detail::is_simd_type <SIMDType>::value
        >::type
    >
    std::basic_istream <CharT, CharTraits> &
        operator>> (std::basic_istream <CharT, CharTraits> & is, SIMDType & v)
    {
        using traits_type = simd_traits <SIMDType>;
        using value_type  = typename traits_type::value_type;

        /* select type to read into from std::num_get::get */
        using in_type = typename std::conditional <
            /* boolean type */
            std::is_same <value_type, bool>::value,
            value_type,
            /* non-boolean types */
            typename std::conditional <
                std::is_integral <value_type>::value,
                /* integral values */
                typename std::conditional <
                    std::is_signed <value_type>::value,
                    /* signed types */
                    typename std::conditional <
                        sizeof (value_type) < sizeof (long),
                        long,
                        value_type
                    >::type,
                    /* unsigned types */
                    typename std::conditional <
                        sizeof (value_type) == 1,
                        unsigned short,
                        value_type
                    >::type
                >::type,
                /* floating point values */
                value_type
            >::type
        >::type;

        using stream_type  = std::basic_istream <CharT, CharTraits>;
        using isb_iterator = std::istreambuf_iterator <CharT, CharTraits>;

        using char_traits = CharTraits;
        using char_type   = typename CharTraits::char_type;

        auto const flags    = is.flags ();
        auto const & locale = is.getloc ();
        auto const & ctype  = std::use_facet <std::ctype <char_type>> (locale);
        auto const & num_get =
            std::use_facet <std::num_get <char_type>> (locale);

        auto discard_non_numeric =
        [&flags, &locale, &ctype] (stream_type & _is) -> stream_type &
        {
            if (flags & std::ios_base::dec) {
                while (!_is.eof () && !_is.bad ()) {
                    if (!ctype.is (std::ctype_base::digit,
                                   static_cast <char_type> (_is.peek ())))
                    {
                        _is.ignore ();
                        continue;
                    } else {
                        break;
                    }
                }
            } else if (flags & std::ios_base::oct) {
                auto const eight = ctype.widen ('8');
                auto const nine  = ctype.widen ('9');

                while (!_is.eof () && !_is.bad ()) {
                    auto const peek = _is.peek ();
                    if (!ctype.is (std::ctype_base::digit,
                                   static_cast <char_type> (peek)) ||
                        char_traits::eq (
                            static_cast <char_type> (peek), eight) ||
                        char_traits::eq (
                            static_cast <char_type> (peek), nine))
                    {
                        _is.ignore ();
                        continue;
                    } else {
                        break;
                    }
                }
            } else if (flags & std::ios_base::hex) {
                while (!_is.eof () && !_is.bad ()) {
                    if (!ctype.is (std::ctype_base::xdigit,
                                   static_cast <char_type> (_is.peek ())))
                    {
                        _is.ignore ();
                        continue;
                    } else {
                        break;
                    }
                }
            } else {
                /* assume decimal if no flags are set */
                while (!_is.eof () && !_is.bad ()) {
                    if (!ctype.is (std::ctype_base::digit,
                                   static_cast <char_type> (_is.peek ())))
                    {
                        _is.ignore ();
                        continue;
                    } else {
                        break;
                    }
                }
            }

            return _is;
        };

        typename std::ios_base::iostate err_state = std::ios_base::goodbit;

        try {
            typename stream_type::sentry sentry {is};

            if (sentry) {
                isb_iterator end;
                in_type in_val;
                std::size_t count = 0;

                do {
                    discard_non_numeric (is);
                    num_get.get (is, end, is, err_state, in_val);

                    if (std::ios_base::failbit & err_state) {
                        is.setstate (std::ios_base::failbit);
                        return is;
                    } else {
                        v.set (count, static_cast <value_type> (in_val));
                        count += 1;
                    }
                } while (count < traits_type::lanes);
            }
        } catch (std::ios_base::failure &) {
            is.setstate (std::ios_base::failbit);
        }

        return is;
    }
}   // namespace simd

namespace std
{
    /*
     * Computes a single hash value for an object of a SIMD vector type.
     */
#define std_hash_impl(ty, lanes, tag) template <>\
    struct hash <simd::simd_type <ty, lanes, tag>>\
    {\
        using argument_type = simd::simd_type <ty, lanes, tag>;\
        using result_type   = std::size_t;\
\
        result_type operator() (argument_type const & s) const noexcept\
        {\
            using value_type = typename simd::simd_traits <\
                argument_type\
            >::value_type;\
\
            simd::hash <argument_type> hasher {};\
            return simd::math::accumulate (\
                hasher (s), std::size_t {0},\
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
        using argument_type = simd::simd_type <ty, lanes, tag>;\
        using result_type   = std::size_t;\
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
            simd::hash <decltype (a.v1)> hasher {};\
            auto const h1 = hasher (a.v1);\
            auto const h2 = hasher (a.v2);\
\
            using hash_type = decltype (h1);\
            using hash_value_type =\
                typename simd::simd_traits <hash_type>::value_type;\
            return simd::math::accumulate (\
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

#if SIMD_HEADER_CLANG
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
#elif SIMD_HEADER_GNUG
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

#undef SIMD_HEADER_CLANG
#undef SIMD_HEADER_GNUG

#endif  // #ifndef SIMD_IMPLEMENTATION_HEADER
