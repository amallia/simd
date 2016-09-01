//
// verifies that arithmetic works for simd types
//

#include <algorithm>    // std::generate
#include <array>        // std::array
#include <cassert>      // assert
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

#include <simd>


template <typename>
struct is_divides_or_modulus : std::false_type {};

template <typename T>
struct is_divides_or_modulus <std::divides <T>> : std::true_type {};

template <typename T>
struct is_divides_or_modulus <std::modulus <T>> : std::true_type {};

template <typename T>
struct is_divides_or_modulus <T const> : is_divides_or_modulus <T> {};

template <typename T>
struct is_divides_or_modulus <T &> : is_divides_or_modulus <T> {};

template <typename T>
struct is_divides_or_modulus <T const &> : is_divides_or_modulus <T> {};

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

enum status : bool
{
    pass = true,
    fail = false
};

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
enum status verify (SimdT const & lhs,
                    SimdT const & rhs,
                    std::vector <std::string> & errors)
{
    using traits = simd::simd_traits <SimdT>;
    using value_type = typename traits::value_type;
    static constexpr auto lanes = traits::lanes;

    static ScalarOp scalar_op {};
    static SimdOp vector_op {};

    auto const lhs_arr = static_cast <std::array <value_type, lanes>> (lhs);
    auto const rhs_arr = static_cast <std::array <value_type, lanes>> (rhs);

    auto const expected_result = map (scalar_op, lhs_arr, rhs_arr);
    auto const expected_vector = SimdT {expected_result};
    auto const result = vector_op (lhs, rhs);

    if ((result != expected_vector).any_of ()) {
        std::ostringstream err;
        err << "incorrect value obtained for:\n";
        for (std::size_t i = 0; i < lanes; ++i) {
            if (result [i] != expected_vector [i]) {
                err << "\t[expected: "
                    << std::dec << expected_vector [i]
                    << "] [obtained: "
                    << std::dec << result [i]
                    << "] [arguments: "
                    << std::dec << lhs_arr [i] << "; "
                    << std::dec << rhs_arr [i] << "]"
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
    static constexpr auto lanes = traits_type::lanes;

    static auto gen = [] (void) -> operand_type
    {
        using distribution = typename std::conditional <
            std::is_integral <value_type>::value,
            std::uniform_int_distribution <value_type>,
            std::uniform_real_distribution <value_type>
        >::type;

        static std::random_device rd;
        auto g = std::bind (distribution {}, std::mt19937 {rd ()});

        std::array <value_type, lanes> values;
        std::generate_n (values.begin (), lanes, g);
        return operand_type {values};
    };

    static auto gen_nonzero = [] (void) -> operand_type
    {
        using distribution = typename std::conditional <
            std::is_integral <value_type>::value,
            std::uniform_int_distribution <value_type>,
            std::uniform_real_distribution <value_type>
        >::type;

        static std::random_device rd;
        auto g = std::bind (
            distribution {value_type {1}}, std::mt19937 {rd ()}
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
    } else {
        std::generate (rhs.begin (), rhs.end (), gen);
    }

    std::uint64_t fail_count = 0;
    for (std::size_t i = 0; i < len; ++i) {
        switch (verify <ScalarOp, SimdOp> (lhs [i], rhs [i], errors)) {
            case status::fail:
                fail_count += 1;
                break;
            case status::pass:
                break;
        }

        logos << "\r\t" << "[" << i + 1 << "/" << len << "]" << std::flush;
    }

    return fail_count;
}

template <typename ScalarType, typename SimdType>
void run_integral_tests (std::string name, std::size_t test_length)
{
    std::vector <std::string> errors;

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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }
}

template <typename ScalarType, typename SimdType>
void run_float_tests (std::string name, std::size_t test_length)
{
    std::vector <std::string> errors;

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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
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
            std::cerr << "... failed: " << errors.size () << " ..." << std::endl;

            if (errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i) {
                std::cerr << errors [i];
            }

            errors.clear ();
        } else {
            std::cerr << "\t... ok ..." << std::endl;
        }
    }
}

int main (void)
{
    constexpr std::size_t test_length = 50000;

    /* 8-bit integer */
    {
        run_integral_tests <std::int8_t, simd::int8x8_t> ("simd::int8x8_t", test_length);
        run_integral_tests <std::int8_t, simd::int8x16_t> ("simd::int8x16_t", test_length);
        run_integral_tests <std::int8_t, simd::int8x32_t> ("simd::int8x32_t", test_length);
        run_integral_tests <std::int8_t, simd::int8x64_t> ("simd::int8x64_t", test_length);
    }

    /* 8-bit unsigned integer */
    {
        run_integral_tests <std::uint8_t, simd::uint8x8_t> ("simd::uint8x8_t", test_length);
        run_integral_tests <std::uint8_t, simd::uint8x16_t> ("simd::uint8x16_t", test_length);
        run_integral_tests <std::uint8_t, simd::uint8x32_t> ("simd::uint8x32_t", test_length);
        run_integral_tests <std::uint8_t, simd::uint8x64_t> ("simd::uint8x64_t", test_length);
    }

    /* 16-bit integer */
    {
        run_integral_tests <std::int16_t, simd::int16x8_t> ("simd::int16x8_t", test_length);
        run_integral_tests <std::int16_t, simd::int16x16_t> ("simd::int16x16_t", test_length);
        run_integral_tests <std::int16_t, simd::int16x16_t> ("simd::int16x16_t", test_length);
        run_integral_tests <std::int16_t, simd::int16x32_t> ("simd::int16x32_t", test_length);
    }

    /* 16-bit unsigned integer */
    {
        run_integral_tests <std::uint16_t, simd::uint16x8_t> ("simd::uint16x8_t", test_length);
        run_integral_tests <std::uint16_t, simd::uint16x16_t> ("simd::uint16x16_t", test_length);
        run_integral_tests <std::uint16_t, simd::uint16x16_t> ("simd::uint16x16_t", test_length);
        run_integral_tests <std::uint16_t, simd::uint16x32_t> ("simd::uint16x32_t", test_length);
    }

    /* 32-bit integer */
    {
        run_integral_tests <std::int32_t, simd::int32x2_t> ("simd::int32x2_t", test_length);
        run_integral_tests <std::int32_t, simd::int32x4_t> ("simd::int32x4_t", test_length);
        run_integral_tests <std::int32_t, simd::int32x8_t> ("simd::int32x8_t", test_length);
        run_integral_tests <std::int32_t, simd::int32x16_t> ("simd::int32x16_t", test_length);
    }

    /* 32-bit unsigned integer */
    {
        run_integral_tests <std::uint32_t, simd::uint32x2_t> ("simd::uint32x2_t", test_length);
        run_integral_tests <std::uint32_t, simd::uint32x4_t> ("simd::uint32x4_t", test_length);
        run_integral_tests <std::uint32_t, simd::uint32x8_t> ("simd::uint32x8_t", test_length);
        run_integral_tests <std::uint32_t, simd::uint32x16_t> ("simd::uint32x16_t", test_length);
    }

    /* 64-bit integer */
    {
        run_integral_tests <std::int64_t, simd::int64x2_t> ("simd::int64x2_t", test_length);
        run_integral_tests <std::int64_t, simd::int64x4_t> ("simd::int64x4_t", test_length);
        run_integral_tests <std::int64_t, simd::int64x8_t> ("simd::int64x8_t", test_length);
    }

    /* 64-bit unsigned integer */
    {
        run_integral_tests <std::uint64_t, simd::uint64x2_t> ("simd::uint64x2_t", test_length);
        run_integral_tests <std::uint64_t, simd::uint64x4_t> ("simd::uint64x4_t", test_length);
        run_integral_tests <std::uint64_t, simd::uint64x8_t> ("simd::uint64x8_t", test_length);
    }

    /* 32-bit float */
    {
        run_float_tests <float, simd::float32x4_t> ("simd::float32x4_t", test_length);
        run_float_tests <float, simd::float32x8_t> ("simd::float32x8_t", test_length);
        run_float_tests <float, simd::float32x16_t> ("simd::float32x16_t", test_length);
    }

    /* 64-bit float */
    {
        run_float_tests <double, simd::float64x2_t> ("simd::float64x2_t", test_length);
        run_float_tests <double, simd::float64x4_t> ("simd::float64x4_t", test_length);
        run_float_tests <double, simd::float64x8_t> ("simd::float64x8_t", test_length);
    }

    return 0;
}
