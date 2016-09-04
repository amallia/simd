//
// verifies that arithmetic works for simd types
//

#if defined (NDEBUG)
    #undef NDEBUG
#endif

#include <algorithm>    // std::generate
#include <array>        // std::array
#include <cassert>      // assert
#include <climits>      // CHAR_BIT
#include <cstdint>      // std::[u]int*_t
#include <functional>   // std::{plus, minus, multiplies, divides,
                        // modulus, bit_and, bit_or, bit_xor}
#include <iomanip>      // std::dec
#include <iostream>     // std::cerr
#include <random>       // std::random_device, std::mt19937,
                        // std::uniform_{int,real}_distribution
#include <string>       // std::string
#include <sstream>      // std::stringstream
#include <type_traits>  // std::is_same
#include <vector>       // std::vector

#include "simd.hpp"

template <typename T>
struct shiftl
{
    T operator() (T const & a, T const & b) noexcept
    {
        return a << b;
    }
};

template <typename T>
struct shiftr
{
    T operator() (T const & a, T const & b) noexcept
    {
        return a >> b;
    }
};

std::random_device & random_device (void) noexcept
{
    static std::random_device rd;
    return rd;
}

enum status : bool
{
    pass = true,
    fail = false
};

template <typename>
struct is_shift : std::false_type {};

template <typename T>
struct is_shift <shiftl <T>> : std::true_type {};

template <typename T>
struct is_shift <shiftr <T>> : std::true_type {};

template <typename>
struct is_divides_or_modulus : std::false_type {};

template <typename T>
struct is_divides_or_modulus <std::divides <T>> : std::true_type {};

template <typename T>
struct is_divides_or_modulus <std::modulus <T>> : std::true_type {};

template <typename U>
struct lower_bound
{
    using value_type = U;

    static value_type value (void) noexcept
    {
        if (std::is_integral <value_type>::value) {
            return value_type {1};
        } else {
            return std::nextafter (
                value_type {1},
                std::numeric_limits <value_type>::max ()
            );                    
        }
    }
};

template <>
#if defined (__clang__)
struct lower_bound <__int128_t>
#elif defined (__GNUG__)
struct lower_bound <__int128>
#endif
{
#if defined (__clang__)
    using value_type = __int128_t;
#elif defined (__GNUG__)
    using value_type = __int128;
#endif
    static value_type value (void) noexcept
    {
        return value_type {1};
    }
};

template <>
#if defined (__clang__)
struct lower_bound <__uint128_t>
#elif defined (__GNUG__)
struct lower_bound <unsigned __int128>
#endif
{
#if defined (__clang__)
    using value_type = __uint128_t;
#elif defined (__GNUG__)
    using value_type = unsigned __int128;
#endif
    static value_type value (void) noexcept
    {
        return value_type {1};
    }
};

#if defined (__clang__)
std::ostream & operator<< (std::ostream & os, __int128_t val)
#elif defined (__GNUG__)
std::ostream & operator<< (std::ostream & os, __int128 val)
#endif
{
#if defined (__clang__)
    using type = __int128_t;
#elif defined (__GNUG__)
    using type = __int128;
#endif
    static char digit_switch [10] = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    };

    std::ostream::sentry s {os};
    if (s) {
        type tmp = val < type {0} ? -val : val;
        char buffer [128];
        auto d = std::end (buffer);

        do {
            d -= 1;
            *d = digit_switch [tmp % 10];
            tmp /= type {10};
        } while (tmp != type {0});

        if (val < 0) {
            d -= 1;
            *d = '-';
        }

        auto len = std::end (buffer) - d;
        if (os.rdbuf ()->sputn (d, len) != len) {
            os.setstate (std::ios_base::badbit);
        }
    }

    return os;
}

#if defined (__clang__)
std::ostream & operator<< (std::ostream & os, __uint128_t val)
#elif defined (__GNUG__)
std::ostream & operator<< (std::ostream & os, unsigned __int128 val)
#endif
{
#if defined (__clang__)
    using type = __uint128_t;
#elif defined (__GNUG__)
    using type = unsigned __int128;
#endif
    static char digit_switch [10] = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    };

    std::ostream::sentry s {os};
    if (s) {
        type tmp = val;
        char buffer [128];
        auto d = std::end (buffer);

        do {
            d -= 1;
            *d = digit_switch [tmp % 10];
            tmp /= type {10};
        } while (tmp != type {0});

        auto len = std::end (buffer) - d;
        if (os.rdbuf ()->sputn (d, len) != len) {
            os.setstate (std::ios_base::badbit);
        }
    }

    return os;
}

template <typename Op, typename T, std::size_t N>
#if defined (__GNUG__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
std::array <T, N> map (Op op, std::array <T, N> const & lhs, std::array <T, N> const & rhs)
{
    std::array <T, N> result;
#if defined (__clang__)
#pragma clang loop vectorize(disable)
#endif
    for (std::size_t i = 0; i < N; ++i) {
        result [i] = op (lhs [i], rhs [i]);
    }

    return result;
}

template <typename ScalarOp, typename SimdOp, typename SimdT>
enum status compute_and_verify (SimdT const & lhs,
                                SimdT const & rhs,
                                std::vector <std::string> & errors)
{
    using traits     = simd::simd_traits <SimdT>;
    using value_type = typename traits::value_type;
    static constexpr auto lanes = traits::lanes;

    static ScalarOp scalar_op {};
    static SimdOp vector_op {};

    auto const result = vector_op (lhs, rhs);
    auto const lhs_arr = static_cast <std::array <value_type, lanes>> (lhs);
    auto const rhs_arr = static_cast <std::array <value_type, lanes>> (rhs);

    auto const expected_result = map (scalar_op, lhs_arr, rhs_arr);
    SimdT const expected_vector {expected_result};

    if ((result != expected_vector).any_of ()) {
        std::ostringstream err;
        err << "incorrect value obtained for:\n";
        for (std::size_t i = 0; i < lanes; ++i) {
            if (result [i] != expected_vector [i]) {
                using cast_type = typename std::conditional <
                    std::is_integral <value_type>::value &&
                        sizeof (value_type) == 1,
                    typename std::conditional <
                        std::is_unsigned <value_type>::value,
                        unsigned int,
                        int
                    >::type,
                    value_type
                >::type;

                err << "\t[expected: "
                    << static_cast <cast_type> (expected_vector [i])
                    << "] [obtained: "
                    << static_cast <cast_type> (result [i])
                    << "] [arguments: "
                    << static_cast <cast_type> (lhs_arr [i]) << "; "
                    << static_cast <cast_type> (rhs_arr [i]) << "]"
                    << std::endl;
            }
        }
        errors.push_back (err.str ());
        return status::fail;
    } else {
        return status::pass;
    }
}

template <typename ScalarOp, typename SimdOp, typename SimdT>
std::uint64_t generate_and_test_cases (std::size_t len,
                                       std::ostream & logos,
                                       std::vector <std::string> & errors)
{
    using operand_type = SimdT;
    using traits_type  = simd::simd_traits <operand_type>;
    using value_type   = typename traits_type::value_type;
#if defined (__clang__)
    using gen_type = typename std::conditional <
        std::is_same <value_type, __int128_t>::value ||
        std::is_same <value_type, __uint128_t>::value,
        typename std::conditional <
            std::is_same <value_type, __int128_t>::value,
            std::int64_t,
            std::uint64_t
        >::type,
        value_type
    >::type;
#elif defined (__GNUG__)
    using gen_type = typename std::conditional <
        std::is_same <value_type, __int128>::value ||
        std::is_same <value_type, unsigned __int128>::value,
        typename std::conditional <
            std::is_same <value_type, __int128>::value,
            std::int64_t,
            std::uint64_t
        >::type,
        value_type
    >::type;
#endif

    static constexpr auto lanes = traits_type::lanes;

    static auto gen = [] (void) -> operand_type
    {
        using distribution = typename std::conditional <
            std::is_integral <gen_type>::value,
            std::uniform_int_distribution <gen_type>,
            std::uniform_real_distribution <gen_type>
        >::type;

        auto & rd = random_device ();
        auto g = std::bind (distribution {}, std::mt19937 {rd ()});

        std::array <value_type, lanes> values;
        std::generate_n (values.begin (), lanes, g);
        return operand_type {values};
    };

    static auto gen_nonzero = [] (void) -> operand_type
    {
        using distribution = typename std::conditional <
            std::is_integral <gen_type>::value,
            std::uniform_int_distribution <gen_type>,
            std::uniform_real_distribution <gen_type>
        >::type;

        static auto lower = lower_bound <gen_type>::value ();
        static auto upper = std::numeric_limits <gen_type>::max ();
        static distribution dist {lower, upper};

        auto & rd = random_device ();
        auto g = std::bind (dist, std::mt19937 {rd ()});

        std::array <value_type, lanes> values;
        std::generate_n (values.begin (), lanes, g);
        return operand_type {values};
    };

    static auto gen_bounded = [] (void) -> operand_type
    {
        static constexpr auto bits = CHAR_BIT * sizeof (value_type);
        using distribution = typename std::conditional <
            std::is_integral <gen_type>::value,
            std::uniform_int_distribution <gen_type>,
            std::uniform_real_distribution <gen_type>
        >::type;

        auto & rd = random_device ();
        auto g = std::bind (
            distribution {
                typename distribution::result_type {0},
                static_cast <typename distribution::result_type> (bits) - 1
            },
            std::mt19937 {rd ()}
        );

        std::array <value_type, lanes> values;
        std::generate_n (values.begin (), lanes, g);
        return operand_type {values};
    };

    std::vector <operand_type> lhs;
    lhs.resize (len);
    std::vector <operand_type> rhs;
    rhs.resize (len);

    std::generate (lhs.begin (), lhs.end (), gen);

    if (is_divides_or_modulus <ScalarOp>::value) {
        std::generate (rhs.begin (), rhs.end (), gen_nonzero);
    } else if (is_shift <ScalarOp>::value) {
        std::generate (rhs.begin (), rhs.end (), gen_bounded);
    } else {
        std::generate (rhs.begin (), rhs.end (), gen);
    }

    std::uint64_t fail_count = 0;
    for (std::size_t i = 0; i < len; ++i) {
        switch (compute_and_verify <ScalarOp, SimdOp> (lhs [i], rhs [i], errors)) {
            case status::fail:
                fail_count += 1;
                break;
            case status::pass:
                break;
            default:
                break;
        }

        logos << "\r\t" << "[" << i + 1 << "/" << len << "]" << std::flush;
    }

    return fail_count;
}

template <typename ScalarType, typename SimdType>
std::uint64_t run_integral_tests (std::string name, std::size_t test_length)
{
    std::vector <std::string> errors;
    std::uint64_t test_fail_count = 0;

    {
        std::cerr << name << " (+)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::plus <ScalarType>, std::plus <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (-)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::minus <ScalarType>, std::minus <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (*)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::multiplies <ScalarType>, std::multiplies <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (/)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::divides <ScalarType>, std::divides <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (%)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::modulus <ScalarType>, std::modulus <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (<<)" << std::endl;
        auto fail_count = generate_and_test_cases <
            shiftl <ScalarType>, shiftl <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (>>)" << std::endl;
        auto fail_count = generate_and_test_cases <
            shiftr <ScalarType>, shiftr <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (&)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::bit_and <ScalarType>, std::bit_and <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (|)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::bit_or <ScalarType>, std::bit_or <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (^)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::bit_xor <ScalarType>, std::bit_xor <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    return test_fail_count;
}

template <typename ScalarType, typename SimdType>
std::uint64_t run_float_tests (std::string name, std::size_t test_length)
{
    std::vector <std::string> errors;
    std::uint64_t test_fail_count = 0;

    {
        std::cerr << name << " (+)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::plus <ScalarType>, std::plus <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (-)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::minus <ScalarType>, std::minus <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (*)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::multiplies <ScalarType>, std::multiplies <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    {
        std::cerr << name << " (/)" << std::endl;
        auto fail_count = generate_and_test_cases <
            std::divides <ScalarType>, std::divides <SimdType>, SimdType
        > (test_length, std::cerr, errors);

        if (fail_count != 0) {
            std::cerr << "\t... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }

    return test_fail_count;
}

int main (void)
{
    constexpr std::size_t test_length = 5000;
    std::uint64_t failures = 0;

    // 8-bit integer 
    {
        failures += run_integral_tests <std::int8_t, simd::int8x8_t> (
			"simd::int8x8_t", test_length
        );
        failures += run_integral_tests <std::int8_t, simd::int8x16_t> (
			"simd::int8x16_t", test_length
        );
        failures += run_integral_tests <std::int8_t, simd::int8x32_t> (
			"simd::int8x32_t", test_length
        );
        failures += run_integral_tests <std::int8_t, simd::int8x64_t> (
			"simd::int8x64_t", test_length
        );
    }

    // 8-bit unsigned integer 
    {
        failures += run_integral_tests <std::uint8_t, simd::uint8x8_t> (
			"simd::uint8x8_t", test_length
        );
        failures += run_integral_tests <std::uint8_t, simd::uint8x16_t> (
			"simd::uint8x16_t", test_length
        );
        failures += run_integral_tests <std::uint8_t, simd::uint8x32_t> (
			"simd::uint8x32_t", test_length
        );
        failures += run_integral_tests <std::uint8_t, simd::uint8x64_t> (
			"simd::uint8x64_t", test_length
        );
    }

    // 16-bit integer 
    {
        failures += run_integral_tests <std::int16_t, simd::int16x8_t> (
			"simd::int16x8_t", test_length
        );
        failures += run_integral_tests <std::int16_t, simd::int16x16_t> (
			"simd::int16x16_t", test_length
        );
        failures += run_integral_tests <std::int16_t, simd::int16x16_t> (
			"simd::int16x16_t", test_length
        );
        failures += run_integral_tests <std::int16_t, simd::int16x32_t> (
			"simd::int16x32_t", test_length
        );
    }

    // 16-bit unsigned integer 
    {
        failures += run_integral_tests <std::uint16_t, simd::uint16x8_t> (
			"simd::uint16x8_t", test_length
        );
        failures += run_integral_tests <std::uint16_t, simd::uint16x16_t> (
			"simd::uint16x16_t", test_length
        );
        failures += run_integral_tests <std::uint16_t, simd::uint16x16_t> (
			"simd::uint16x16_t", test_length
        );
        failures += run_integral_tests <std::uint16_t, simd::uint16x32_t> (
			"simd::uint16x32_t", test_length
        );
    }

    // 32-bit integer 
    {
        failures += run_integral_tests <std::int32_t, simd::int32x2_t> (
			"simd::int32x2_t", test_length
        );
        failures += run_integral_tests <std::int32_t, simd::int32x4_t> (
			"simd::int32x4_t", test_length
        );
        failures += run_integral_tests <std::int32_t, simd::int32x8_t> (
			"simd::int32x8_t", test_length
        );
        failures += run_integral_tests <std::int32_t, simd::int32x16_t> (
			"simd::int32x16_t", test_length
        );
    }

    // 32-bit unsigned integer 
    {
        failures += run_integral_tests <std::uint32_t, simd::uint32x2_t> (
			"simd::uint32x2_t", test_length
        );
        failures += run_integral_tests <std::uint32_t, simd::uint32x4_t> (
			"simd::uint32x4_t", test_length
        );
        failures += run_integral_tests <std::uint32_t, simd::uint32x8_t> (
			"simd::uint32x8_t", test_length
        );
        failures += run_integral_tests <std::uint32_t, simd::uint32x16_t> (
			"simd::uint32x16_t", test_length
        );
    }

    // 64-bit integer 
    {
        failures += run_integral_tests <std::int64_t, simd::int64x2_t> (
			"simd::int64x2_t", test_length
        );
        failures += run_integral_tests <std::int64_t, simd::int64x4_t> (
			"simd::int64x4_t", test_length
        );
        failures += run_integral_tests <std::int64_t, simd::int64x8_t> (
			"simd::int64x8_t", test_length
        );
    }

    // 64-bit unsigned integer 
    {
        failures += run_integral_tests <std::uint64_t, simd::uint64x2_t> (
			"simd::uint64x2_t", test_length
        );
        failures += run_integral_tests <std::uint64_t, simd::uint64x4_t> (
			"simd::uint64x4_t", test_length
        );
        failures += run_integral_tests <std::uint64_t, simd::uint64x8_t> (
			"simd::uint64x8_t", test_length
        );
    }

    // 128-bit signed integer
    {
#if defined (__clang__)
        failures += run_integral_tests <__int128_t, simd::int128x1_t> (
            "simd::int128x1_t", test_length
        );
        failures += run_integral_tests <__int128_t, simd::int128x2_t> (
            "simd::int128x2_t", test_length
        );
        failures += run_integral_tests <__int128_t, simd::int128x4_t> (
            "simd::int128x4_t", test_length
        );
#elif defined (__GNUG__)
        failures += run_integral_tests <__int128, simd::int128x1_t> (
            "simd::int128x1_t", test_length
        );
        failures += run_integral_tests <__int128, simd::int128x2_t> (
            "simd::int128x2_t", test_length
        );
        failures += run_integral_tests <__int128, simd::int128x4_t> (
            "simd::int128x4_t", test_length
        );
#endif
    }

    // 128-bit unsigned integer
    {
#if defined (__clang__)
        failures += run_integral_tests <__uint128_t, simd::uint128x1_t> (
            "simd::uint128x1_t", test_length
        );
        failures += run_integral_tests <__uint128_t, simd::uint128x2_t> (
            "simd::uint128x2_t", test_length
        );
        failures += run_integral_tests <__uint128_t, simd::uint128x4_t> (
            "simd::uint128x4_t", test_length
        );
#elif defined (__GNUG__)
        failures +=
            run_integral_tests <unsigned __int128, simd::uint128x1_t> (
                "simd::uint128x1_t", test_length
            );
        failures +=
            run_integral_tests <unsigned __int128, simd::uint128x2_t> (
                "simd::uint128x2_t", test_length
            );
        failures +=
            run_integral_tests <unsigned __int128, simd::uint128x4_t> (
                "simd::uint128x4_t", test_length
            );
#endif
    }

    // 32-bit float 
    {
        failures += run_float_tests <float, simd::float32x4_t> (
            "simd::float32x4_t", test_length
        );
        failures += run_float_tests <float, simd::float32x8_t> (
            "simd::float32x8_t", test_length
        );
        failures += run_float_tests <float, simd::float32x16_t> (
            "simd::float32x16_t", test_length
        );
    }

    // 64-bit float 
    {
        failures += run_float_tests <double, simd::float64x2_t> (
            "simd::float64x2_t", test_length
        );
        failures += run_float_tests <double, simd::float64x4_t> (
            "simd::float64x4_t", test_length
        );
        failures += run_float_tests <double, simd::float64x8_t> (
            "simd::float64x8_t", test_length
        );
    }

    // long double 
    {
        failures += run_float_tests <long double, simd::long_doublex2_t> (
            "simd::long_doublex2_t", test_length
        );
        failures += run_float_tests <long double, simd::long_doublex4_t> (
            "simd::long_doublex4_t", test_length
        );
    }

    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
