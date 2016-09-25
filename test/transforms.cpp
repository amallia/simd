//
// verifies correctness of functions in the simd::math namespace
//

#if defined (NDEBUG)
    #undef NDEBUG
#endif
#include <algorithm>    // std::find_if, std::generate
#include <array>        // std::array
#include <cassert>      // assert
#include <climits>      // ULONG_MAX
#include <cmath>
#include <cstdint>      // std::[u]int*_t
#include <cstdlib>      // std::strtoul
#include <cstring>      // std::strcmp
#include <iostream>     // std::cerr
#include <random>       // std::random_device, std::mt19937,
                        // std::uniform_{int,real}_distribution
#include <string>       // std::string
#include <sstream>      // std::stringstream
#include <type_traits>  // std::is_same
#include <vector>       // std::vector

#include "simd.hpp"

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

template <typename Op, typename T, std::size_t N>
#if defined (__GNUG__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
auto map (Op op, std::array <T, N> const & args)
    -> std::array <
        typename std::decay <
            decltype (op (std::declval <T const &> ()))
        >::type,
        N
    >
{
    std::array <
        typename std::remove_cv <
            typename std::remove_reference <
                decltype (op (std::declval <T const &> ()))
            >::type
        >::type,
        N
    > result;
#if defined (__clang__)
#pragma clang loop vectorize(disable)
#endif
    for (std::size_t i = 0; i < N; ++i) {
        result [i] = op (args [i]);
    }

    return result;
}

template <typename Op, typename T, std::size_t N>
#if defined (__GNUG__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
auto map (Op op, std::array <T, N> const & lhs, std::array <T, N> const & rhs)
    -> std::array <
        typename std::decay <
            decltype (op (
                std::declval <T const &> (), std::declval <T const &> ()
            ))
        >::type,
        N
    >
{
    std::array <
        typename std::remove_cv <
            typename std::remove_reference <
                decltype (op (
                    std::declval <T const &> (), std::declval <T const &> ()
                ))
            >::type
        >::type,
        N
    > result;
#if defined (__clang__)
#pragma clang loop vectorize(disable)
#endif
    for (std::size_t i = 0; i < N; ++i) {
        result [i] = op (lhs [i], rhs [i]);
    }

    return result;
}

template <typename ScalarOp, typename SimdOp, typename SimdT>
enum status
compute_and_verify (SimdT const & arg, std::vector <std::string> & errors)
{
    using traits     = simd::simd_traits <SimdT>;
    using value_type = typename traits::value_type;
    static constexpr auto lanes = traits::lanes;

    static ScalarOp scalar_op {};
    static SimdOp vector_op {};

    auto const result = vector_op (arg);
    auto const arg_arr = static_cast <std::array <value_type, lanes>> (arg);

    auto const expected_result = map (scalar_op, arg_arr);
    typename std::remove_cv <decltype (result)>::type expected_vector;
    for (std::size_t i = 0; i < lanes; ++i) {
        expected_vector.assign (i, expected_result [i]);
    }

    if (simd::any_of (result != expected_vector)) {
        /* check for nan's */
        if (simd::any_of (
            result != result || expected_vector != expected_vector
        ))
        {
            return status::pass;
        }

        std::ostringstream err;
        err << "incorrect value obtained for:\n";
        for (std::size_t i = 0; i < lanes; ++i) {
            if (result.value (i) != expected_vector.value (i)) {
                err << "\t[expected: "
                    << +expected_vector.value (i)
                    << "] [obtained: "
                    << +result.value (i)
                    << "] [argument: "
                    << +arg_arr [i] << "]"
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
    typename std::remove_cv <decltype (result)>::type expected_vector;
    for (std::size_t i = 0; i < lanes; ++i) {
        expected_vector.assign (i, expected_result [i]);
    }

    if (simd::any_of (result != expected_vector)) {
        /* check for nan's */
        if (simd::any_of (
            result != result || expected_vector != expected_vector
        ))
        {
            return status::pass;
        }

        std::ostringstream err;
        err << "incorrect value obtained for:\n";
        for (std::size_t i = 0; i < lanes; ++i) {
            if (result.value (i) != expected_vector.value (i)) {
                err << "\t[expected: "
                    << +expected_vector.value (i)
                    << "] [obtained: "
                    << +result.value (i)
                    << "] [arguments: "
                    << +lhs_arr [i] << "; "
                    << +rhs_arr [i] << "]"
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
std::uint64_t
generate_and_test_unary_cases (std::size_t len,
                               std::ostream & logos,
                               std::vector <std::string> & errors,
                               bool verbose_output)
{
    using operand_type = SimdT;
    using traits_type  = simd::simd_traits <operand_type>;
    using value_type   = typename traits_type::value_type;
    using gen_type     = value_type; 

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

    std::vector <operand_type> args;
    args.resize (len);
    std::generate (args.begin (), args.end (), gen);

    std::uint64_t fail_count = 0;
    for (std::size_t i = 0; i < len; ++i) {
        switch (compute_and_verify <ScalarOp, SimdOp> (args [i], errors))
        {
            case status::fail:
                fail_count += 1;
                break;
            case status::pass:
                break;
            default:
                break;
        }

        if (verbose_output) {
            logos << "\r\t" << "[" << i + 1 << "/" << len << "]" << std::flush;
        }
    }

    return fail_count;
}

template <typename ScalarOp, typename SimdOp, typename SimdT>
std::uint64_t
generate_and_test_binary_cases (std::size_t len,
                                std::ostream & logos,
                                std::vector <std::string> & errors,
                                bool verbose_output)
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

    std::vector <operand_type> lhs;
    lhs.resize (len);
    std::vector <operand_type> rhs;
    rhs.resize (len);

    std::generate (lhs.begin (), lhs.end (), gen);
    std::generate (rhs.begin (), rhs.end (), gen);

    std::uint64_t fail_count = 0;
    for (std::size_t i = 0; i < len; ++i) {
        switch (
            compute_and_verify <ScalarOp, SimdOp> (lhs [i], rhs [i], errors)
        )
        {
            case status::fail:
                fail_count += 1;
                break;
            case status::pass:
                break;
            default:
                break;
        }

        if (verbose_output) {
            logos << "\r\t" << "[" << i + 1 << "/" << len << "]" << std::flush;
        }
    }

    return fail_count;
}

template <typename ScalarType, typename SIMDType>
std::uint64_t run_integral_tests (std::string name,
                               std::size_t test_length,
                               bool verbose_output)
{
    std::vector <std::string> errors;
    std::uint64_t test_fail_count = 0;

    auto process_fail_count = [&] (std::size_t fail_count)
    {
        if (fail_count != 0) {
            if (verbose_output) {
                std::cerr << "\t... failed: " << errors.size () << " ..."
                          << std::endl;
            }

            if (!verbose_output && errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            if (verbose_output) {
                for (auto const & e : errors) {
                    std::cerr << e;
                }
            } else {
                for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i)
                {
                    std::cerr << errors [i];
                }
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            if (verbose_output) {
                std::cerr << "\t... ok ..." << std::endl;
            }
        }
    };

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::fabs (a))
            {
                return std::fabs (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::fabs (a))
            {
                return simd::math::fabs (a);
            }
        };

        std::cout << name << " (fabs)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::arg (a))
            {
                return std::arg (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::arg (a))
            {
                return simd::math::arg (a);
            }
        };

        std::cout << name << " (arg)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::proj (a))
            {
                return std::proj (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::proj (a))
            {
                return simd::math::proj (a);
            }
        };

        std::cout << name << " (proj)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::exp (a))
            {
                return std::exp (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::exp (a))
            {
                return simd::math::exp (a);
            }
        };

        std::cout << name << " (exp)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::exp2 (a))
            {
                return std::exp2 (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::exp2 (a))
            {
                return simd::math::exp2 (a);
            }
        };

        std::cout << name << " (exp2)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::expm1 (a))
            {
                return std::expm1 (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::expm1 (a))
            {
                return simd::math::expm1 (a);
            }
        };

        std::cout << name << " (expm1)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::log (a))
            {
                return std::log (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::log (a))
            {
                return simd::math::log (a);
            }
        };

        std::cout << name << " (log)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::log10 (a))
            {
                return std::log10 (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::log10 (a))
            {
                return simd::math::log10 (a);
            }
        };

        std::cout << name << " (log10)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::log2 (a))
            {
                return std::log2 (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::log2 (a))
            {
                return simd::math::log2 (a);
            }
        };

        std::cout << name << " (log2)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::log1p (a))
            {
                return std::log1p (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::log1p (a))
            {
                return simd::math::log1p (a);
            }
        };

        std::cout << name << " (log1p)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::sqrt (a))
            {
                return std::sqrt (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::sqrt (a))
            {
                return simd::math::sqrt (a);
            }
        };

        std::cout << name << " (sqrt)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::cbrt (a))
            {
                return std::cbrt (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::cbrt (a))
            {
                return simd::math::cbrt (a);
            }
        };

        std::cout << name << " (cbrt)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::sin (a))
            {
                return std::sin (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::sin (a))
            {
                return simd::math::sin (a);
            }
        };

        std::cout << name << " (sin)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::asin (a))
            {
                return std::asin (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::asin (a))
            {
                return simd::math::asin (a);
            }
        };

        std::cout << name << " (asin)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::cos (a))
            {
                return std::cos (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::cos (a))
            {
                return simd::math::cos (a);
            }
        };

        std::cout << name << " (cos)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::acos (a))
            {
                return std::acos (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::acos (a))
            {
                return simd::math::acos (a);
            }
        };

        std::cout << name << " (acos)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::tan (a))
            {
                return std::tan (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::tan (a))
            {
                return simd::math::tan (a);
            }
        };

        std::cout << name << " (tan)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::atan (a))
            {
                return std::atan (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::atan (a))
            {
                return simd::math::atan (a);
            }
        };

        std::cout << name << " (atan)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::sinh (a))
            {
                return std::sinh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::sinh (a))
            {
                return simd::math::sinh (a);
            }
        };

        std::cout << name << " (sinh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::asinh (a))
            {
                return std::asinh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::asinh (a))
            {
                return simd::math::asinh (a);
            }
        };

        std::cout << name << " (asinh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::cosh (a))
            {
                return std::cosh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::cosh (a))
            {
                return simd::math::cosh (a);
            }
        };

        std::cout << name << " (cosh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::acosh (a))
            {
                return std::acosh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::acosh (a))
            {
                return simd::math::acosh (a);
            }
        };

        std::cout << name << " (acosh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::tanh (a))
            {
                return std::tanh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::tanh (a))
            {
                return simd::math::tanh (a);
            }
        };

        std::cout << name << " (tanh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::atanh (a))
            {
                return std::atanh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::atanh (a))
            {
                return simd::math::atanh (a);
            }
        };

        std::cout << name << " (atanh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::erf (a))
            {
                return std::erf (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::erf (a))
            {
                return simd::math::erf (a);
            }
        };

        std::cout << name << " (erf)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::erfc (a))
            {
                return std::erfc (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::erfc (a))
            {
                return simd::math::erfc (a);
            }
        };

        std::cout << name << " (erfc)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::tgamma (a))
            {
                return std::tgamma (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::tgamma (a))
            {
                return simd::math::tgamma (a);
            }
        };

        std::cout << name << " (tgamma)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::lgamma (a))
            {
                return std::lgamma (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::lgamma (a))
            {
                return simd::math::lgamma (a);
            }
        };

        std::cout << name << " (lgamma)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::ceil (a))
            {
                return std::ceil (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::ceil (a))
            {
                return simd::math::ceil (a);
            }
        };

        std::cout << name << " (ceil)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::floor (a))
            {
                return std::floor (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::floor (a))
            {
                return simd::math::floor (a);
            }
        };

        std::cout << name << " (floor)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::trunc (a))
            {
                return std::trunc (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::trunc (a))
            {
                return simd::math::trunc (a);
            }
        };

        std::cout << name << " (trunc)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::round (a))
            {
                return std::round (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::round (a))
            {
                return simd::math::round (a);
            }
        };

        std::cout << name << " (round)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::lround (a))
            {
                return std::lround (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::lround (a))
            {
                return simd::math::lround (a);
            }
        };

        std::cout << name << " (lround)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::llround (a))
            {
                return std::llround (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::llround (a))
            {
                return simd::math::llround (a);
            }
        };

        std::cout << name << " (llround)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::nearbyint (a))
            {
                return std::nearbyint (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::nearbyint (a))
            {
                return simd::math::nearbyint (a);
            }
        };

        std::cout << name << " (nearbyint)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::rint (a))
            {
                return std::rint (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::rint (a))
            {
                return simd::math::rint (a);
            }
        };

        std::cout << name << " (rint)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::lrint (a))
            {
                return std::lrint (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::lrint (a))
            {
                return simd::math::lrint (a);
            }
        };

        std::cout << name << " (lrint)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::llrint (a))
            {
                return std::llrint (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::llrint (a))
            {
                return simd::math::llrint (a);
            }
        };

        std::cout << name << " (llrint)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::ilogb (a))
            {
                return std::ilogb (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::ilogb (a))
            {
                return simd::math::ilogb (a);
            }
        };

        std::cout << name << " (ilogb)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::logb (a))
            {
                return std::logb (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::logb (a))
            {
                return simd::math::logb (a);
            }
        };

        std::cout << name << " (logb)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::fpclassify (a))
            {
                return std::fpclassify (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::fpclassify (a))
            {
                return simd::math::fpclassify (a);
            }
        };

        std::cout << name << " (fpclassify)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::isfinite (a))
            {
                return std::isfinite (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::isfinite (a))
            {
                return simd::math::isfinite (a);
            }
        };

        std::cout << name << " (isfinite)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::isinf (a))
            {
                return std::isinf (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::isinf (a))
            {
                return simd::math::isinf (a);
            }
        };

        std::cout << name << " (isinf)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::isnan (a))
            {
                return std::isnan (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::isnan (a))
            {
                return simd::math::isnan (a);
            }
        };

        std::cout << name << " (isnan)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::isnormal (a))
            {
                return std::isnormal (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::isnormal (a))
            {
                return simd::math::isnormal (a);
            }
        };

        std::cout << name << " (isnormal)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::signbit (a))
            {
                return std::signbit (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::signbit (a))
            {
                return simd::math::signbit (a);
            }
        };

        std::cout << name << " (signbit)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::pow (a, b))
            {
                return std::pow (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::pow (a, b))
            {
                return simd::math::pow (a, b);
            }
        };

        std::cout << name << " (pow)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::hypot (a, b))
            {
                return std::hypot (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::hypot (a, b))
            {
                return simd::math::hypot (a, b);
            }
        };

        std::cout << name << " (hypot)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::max (a, b))
            {
                return std::max (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::max (a, b))
            {
                return simd::math::max (a, b);
            }
        };

        std::cout << name << " (max)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::min (a, b))
            {
                return std::min (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::min (a, b))
            {
                return simd::math::min (a, b);
            }
        };

        std::cout << name << " (min)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::fmax (a, b))
            {
                return std::fmax (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::fmax (a, b))
            {
                return simd::math::fmax (a, b);
            }
        };

        std::cout << name << " (fmax)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::fmin (a, b))
            {
                return std::fmin (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::fmin (a, b))
            {
                return simd::math::fmin (a, b);
            }
        };

        std::cout << name << " (fmin)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::fdim (a, b))
            {
                return std::fdim (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::fdim (a, b))
            {
                return simd::math::fdim (a, b);
            }
        };

        std::cout << name << " (fdim)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::nextafter (a, b))
            {
                return std::nextafter (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::nextafter (a, b))
            {
                return simd::math::nextafter (a, b);
            }
        };

        std::cout << name << " (nextafter)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::nexttoward (a, b))
            {
                return std::nexttoward (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::nexttoward (a, b))
            {
                return simd::math::nexttoward (a, b);
            }
        };

        std::cout << name << " (nexttoward)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::copysign (a, b))
            {
                return std::copysign (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::copysign (a, b))
            {
                return simd::math::copysign (a, b);
            }
        };

        std::cout << name << " (copysign)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::isgreater (a, b))
            {
                return std::isgreater (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::isgreater (a, b))
            {
                return simd::math::isgreater (a, b);
            }
        };

        std::cout << name << " (isgreater)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::isgreaterequal (a, b))
            {
                return std::isgreaterequal (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::isgreaterequal (a, b))
            {
                return simd::math::isgreaterequal (a, b);
            }
        };

        std::cout << name << " (isgreaterequal)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::isless (a, b))
            {
                return std::isless (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::isless (a, b))
            {
                return simd::math::isless (a, b);
            }
        };

        std::cout << name << " (isless)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::islessequal (a, b))
            {
                return std::islessequal (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::islessequal (a, b))
            {
                return simd::math::islessequal (a, b);
            }
        };

        std::cout << name << " (islessequal)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::islessgreater (a, b))
            {
                return std::islessgreater (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::islessgreater (a, b))
            {
                return simd::math::islessgreater (a, b);
            }
        };

        std::cout << name << " (islessgreater)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::isunordered (a, b))
            {
                return std::isunordered (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::isunordered (a, b))
            {
                return simd::math::isunordered (a, b);
            }
        };

        std::cout << name << " (isunordered)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::fmod (a, b))
            {
                return std::fmod (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::fmod (a, b))
            {
                return simd::math::fmod (a, b);
            }
        };

        std::cout << name << " (fmod)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::remainder (a, b))
            {
                return std::remainder (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::remainder (a, b))
            {
                return simd::math::remainder (a, b);
            }
        };

        std::cout << name << " (remainder)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    return test_fail_count;
}

template <typename ScalarType, typename SIMDType>
std::uint64_t run_float_tests (std::string name,
                               std::size_t test_length,
                               bool verbose_output)
{
    std::vector <std::string> errors;
    std::uint64_t test_fail_count = 0;

    auto process_fail_count = [&] (std::size_t fail_count)
    {
        if (fail_count != 0) {
            if (verbose_output) {
                std::cerr << "\t... failed: " << errors.size () << " ..."
                          << std::endl;
            }

            if (!verbose_output && errors.size () > 5) {
                std::cerr << "truncating output to 5 error logs...\n";
            }

            if (verbose_output) {
                for (auto const & e : errors) {
                    std::cerr << e;
                }
            } else {
                for (std::size_t i = 0; i < std::min (5ul, errors.size ()); ++i)
                {
                    std::cerr << errors [i];
                }
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            if (verbose_output) {
                std::cerr << "\t... ok ..." << std::endl;
            }
        }
    };

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::fabs (a))
            {
                return std::fabs (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::fabs (a))
            {
                return simd::math::fabs (a);
            }
        };

        std::cout << name << " (fabs)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::arg (a))
            {
                return std::arg (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::arg (a))
            {
                return simd::math::arg (a);
            }
        };

        std::cout << name << " (arg)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::proj (a))
            {
                return std::proj (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::proj (a))
            {
                return simd::math::proj (a);
            }
        };

        std::cout << name << " (proj)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::exp (a))
            {
                return std::exp (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::exp (a))
            {
                return simd::math::exp (a);
            }
        };

        std::cout << name << " (exp)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::exp2 (a))
            {
                return std::exp2 (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::exp2 (a))
            {
                return simd::math::exp2 (a);
            }
        };

        std::cout << name << " (exp2)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::expm1 (a))
            {
                return std::expm1 (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::expm1 (a))
            {
                return simd::math::expm1 (a);
            }
        };

        std::cout << name << " (expm1)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::log (a))
            {
                return std::log (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::log (a))
            {
                return simd::math::log (a);
            }
        };

        std::cout << name << " (log)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::log10 (a))
            {
                return std::log10 (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::log10 (a))
            {
                return simd::math::log10 (a);
            }
        };

        std::cout << name << " (log10)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::log2 (a))
            {
                return std::log2 (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::log2 (a))
            {
                return simd::math::log2 (a);
            }
        };

        std::cout << name << " (log2)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::log1p (a))
            {
                return std::log1p (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::log1p (a))
            {
                return simd::math::log1p (a);
            }
        };

        std::cout << name << " (log1p)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::sqrt (a))
            {
                return std::sqrt (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::sqrt (a))
            {
                return simd::math::sqrt (a);
            }
        };

        std::cout << name << " (sqrt)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::cbrt (a))
            {
                return std::cbrt (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::cbrt (a))
            {
                return simd::math::cbrt (a);
            }
        };

        std::cout << name << " (cbrt)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::sin (a))
            {
                return std::sin (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::sin (a))
            {
                return simd::math::sin (a);
            }
        };

        std::cout << name << " (sin)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::asin (a))
            {
                return std::asin (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::asin (a))
            {
                return simd::math::asin (a);
            }
        };

        std::cout << name << " (asin)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::cos (a))
            {
                return std::cos (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::cos (a))
            {
                return simd::math::cos (a);
            }
        };

        std::cout << name << " (cos)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::acos (a))
            {
                return std::acos (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::acos (a))
            {
                return simd::math::acos (a);
            }
        };

        std::cout << name << " (acos)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::tan (a))
            {
                return std::tan (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::tan (a))
            {
                return simd::math::tan (a);
            }
        };

        std::cout << name << " (tan)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::atan (a))
            {
                return std::atan (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::atan (a))
            {
                return simd::math::atan (a);
            }
        };

        std::cout << name << " (atan)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::sinh (a))
            {
                return std::sinh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::sinh (a))
            {
                return simd::math::sinh (a);
            }
        };

        std::cout << name << " (sinh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::asinh (a))
            {
                return std::asinh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::asinh (a))
            {
                return simd::math::asinh (a);
            }
        };

        std::cout << name << " (asinh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::cosh (a))
            {
                return std::cosh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::cosh (a))
            {
                return simd::math::cosh (a);
            }
        };

        std::cout << name << " (cosh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::acosh (a))
            {
                return std::acosh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::acosh (a))
            {
                return simd::math::acosh (a);
            }
        };

        std::cout << name << " (acosh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::tanh (a))
            {
                return std::tanh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::tanh (a))
            {
                return simd::math::tanh (a);
            }
        };

        std::cout << name << " (tanh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::atanh (a))
            {
                return std::atanh (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::atanh (a))
            {
                return simd::math::atanh (a);
            }
        };

        std::cout << name << " (atanh)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::erf (a))
            {
                return std::erf (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::erf (a))
            {
                return simd::math::erf (a);
            }
        };

        std::cout << name << " (erf)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::erfc (a))
            {
                return std::erfc (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::erfc (a))
            {
                return simd::math::erfc (a);
            }
        };

        std::cout << name << " (erfc)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::tgamma (a))
            {
                return std::tgamma (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::tgamma (a))
            {
                return simd::math::tgamma (a);
            }
        };

        std::cout << name << " (tgamma)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::lgamma (a))
            {
                return std::lgamma (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::lgamma (a))
            {
                return simd::math::lgamma (a);
            }
        };

        std::cout << name << " (lgamma)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::ceil (a))
            {
                return std::ceil (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::ceil (a))
            {
                return simd::math::ceil (a);
            }
        };

        std::cout << name << " (ceil)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::floor (a))
            {
                return std::floor (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::floor (a))
            {
                return simd::math::floor (a);
            }
        };

        std::cout << name << " (floor)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::trunc (a))
            {
                return std::trunc (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::trunc (a))
            {
                return simd::math::trunc (a);
            }
        };

        std::cout << name << " (trunc)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::round (a))
            {
                return std::round (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::round (a))
            {
                return simd::math::round (a);
            }
        };

        std::cout << name << " (round)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::lround (a))
            {
                return std::lround (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::lround (a))
            {
                return simd::math::lround (a);
            }
        };

        std::cout << name << " (lround)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::llround (a))
            {
                return std::llround (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::llround (a))
            {
                return simd::math::llround (a);
            }
        };

        std::cout << name << " (llround)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::nearbyint (a))
            {
                return std::nearbyint (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::nearbyint (a))
            {
                return simd::math::nearbyint (a);
            }
        };

        std::cout << name << " (nearbyint)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::rint (a))
            {
                return std::rint (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::rint (a))
            {
                return simd::math::rint (a);
            }
        };

        std::cout << name << " (rint)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::lrint (a))
            {
                return std::lrint (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::lrint (a))
            {
                return simd::math::lrint (a);
            }
        };

        std::cout << name << " (lrint)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::llrint (a))
            {
                return std::llrint (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::llrint (a))
            {
                return simd::math::llrint (a);
            }
        };

        std::cout << name << " (llrint)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::ilogb (a))
            {
                return std::ilogb (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::ilogb (a))
            {
                return simd::math::ilogb (a);
            }
        };

        std::cout << name << " (ilogb)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::logb (a))
            {
                return std::logb (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::logb (a))
            {
                return simd::math::logb (a);
            }
        };

        std::cout << name << " (logb)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::fpclassify (a))
            {
                return std::fpclassify (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::fpclassify (a))
            {
                return simd::math::fpclassify (a);
            }
        };

        std::cout << name << " (fpclassify)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::isfinite (a))
            {
                return std::isfinite (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::isfinite (a))
            {
                return simd::math::isfinite (a);
            }
        };

        std::cout << name << " (isfinite)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::isinf (a))
            {
                return std::isinf (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::isinf (a))
            {
                return simd::math::isinf (a);
            }
        };

        std::cout << name << " (isinf)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::isnan (a))
            {
                return std::isnan (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::isnan (a))
            {
                return simd::math::isnan (a);
            }
        };

        std::cout << name << " (isnan)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::isnormal (a))
            {
                return std::isnormal (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::isnormal (a))
            {
                return simd::math::isnormal (a);
            }
        };

        std::cout << name << " (isnormal)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a) const noexcept
                -> decltype (std::signbit (a))
            {
                return std::signbit (a);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a) const noexcept
                -> decltype (simd::math::signbit (a))
            {
                return simd::math::signbit (a);
            }
        };

        std::cout << name << " (signbit)" << std::endl;
        process_fail_count (
            generate_and_test_unary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::pow (a, b))
            {
                return std::pow (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::pow (a, b))
            {
                return simd::math::pow (a, b);
            }
        };

        std::cout << name << " (pow)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::hypot (a, b))
            {
                return std::hypot (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::hypot (a, b))
            {
                return simd::math::hypot (a, b);
            }
        };

        std::cout << name << " (hypot)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::max (a, b))
            {
                return std::max (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::max (a, b))
            {
                return simd::math::max (a, b);
            }
        };

        std::cout << name << " (max)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::min (a, b))
            {
                return std::min (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::min (a, b))
            {
                return simd::math::min (a, b);
            }
        };

        std::cout << name << " (min)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::fmax (a, b))
            {
                return std::fmax (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::fmax (a, b))
            {
                return simd::math::fmax (a, b);
            }
        };

        std::cout << name << " (fmax)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::fmin (a, b))
            {
                return std::fmin (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::fmin (a, b))
            {
                return simd::math::fmin (a, b);
            }
        };

        std::cout << name << " (fmin)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::fdim (a, b))
            {
                return std::fdim (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::fdim (a, b))
            {
                return simd::math::fdim (a, b);
            }
        };

        std::cout << name << " (fdim)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::nextafter (a, b))
            {
                return std::nextafter (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::nextafter (a, b))
            {
                return simd::math::nextafter (a, b);
            }
        };

        std::cout << name << " (nextafter)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::nexttoward (a, b))
            {
                return std::nexttoward (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::nexttoward (a, b))
            {
                return simd::math::nexttoward (a, b);
            }
        };

        std::cout << name << " (nexttoward)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::copysign (a, b))
            {
                return std::copysign (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::copysign (a, b))
            {
                return simd::math::copysign (a, b);
            }
        };

        std::cout << name << " (copysign)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::isgreater (a, b))
            {
                return std::isgreater (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::isgreater (a, b))
            {
                return simd::math::isgreater (a, b);
            }
        };

        std::cout << name << " (isgreater)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::isgreaterequal (a, b))
            {
                return std::isgreaterequal (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::isgreaterequal (a, b))
            {
                return simd::math::isgreaterequal (a, b);
            }
        };

        std::cout << name << " (isgreaterequal)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::isless (a, b))
            {
                return std::isless (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::isless (a, b))
            {
                return simd::math::isless (a, b);
            }
        };

        std::cout << name << " (isless)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::islessequal (a, b))
            {
                return std::islessequal (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::islessequal (a, b))
            {
                return simd::math::islessequal (a, b);
            }
        };

        std::cout << name << " (islessequal)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::islessgreater (a, b))
            {
                return std::islessgreater (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::islessgreater (a, b))
            {
                return simd::math::islessgreater (a, b);
            }
        };

        std::cout << name << " (islessgreater)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::isunordered (a, b))
            {
                return std::isunordered (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::isunordered (a, b))
            {
                return simd::math::isunordered (a, b);
            }
        };

        std::cout << name << " (isunordered)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::fmod (a, b))
            {
                return std::fmod (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::fmod (a, b))
            {
                return simd::math::fmod (a, b);
            }
        };

        std::cout << name << " (fmod)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    {
        struct std_test_function
        {
            auto operator() (ScalarType const & a, ScalarType const & b) const
                noexcept
                -> decltype (std::remainder (a, b))
            {
                return std::remainder (a, b);
            }
        };

        struct simd_test_function
        {
            auto operator() (SIMDType const & a, SIMDType const & b) const
                noexcept
                -> decltype (simd::math::remainder (a, b))
            {
                return simd::math::remainder (a, b);
            }
        };

        std::cout << name << " (remainder)" << std::endl;
        process_fail_count (
            generate_and_test_binary_cases <
                std_test_function, simd_test_function, SIMDType
            > (test_length, std::cout, errors, verbose_output)
        );
    }

    return test_fail_count;
}

int main (int argc, char ** argv)
{
    auto const test_length =
        [argc, argv] (void) -> std::size_t
        {
            constexpr std::size_t default_test_length = 500;
            auto const pos = std::find_if (
                argv + 1, argv + argc,
                [] (char const * s) {
                    return std::strcmp (s, "--test-length") == 0 ||
                           std::strcmp (s, "-l") == 0;
                }
            );

            if (pos == argv + argc || pos == argv + argc - 1) {
                return default_test_length;
            } else {
                auto const conv = std::strtoul (*(pos + 1), nullptr, 10);
                if (conv == 0 || conv == ULONG_MAX) {
                    return default_test_length;
                } else {
                    return conv;
                }
            }
        }();

    auto const verbose_output =
        [argc, argv] (void) -> bool
        {
            auto const vpos = std::find_if (
                argv + 1, argv + argc,
                [] (char const * s) {
                    return std::strcmp (s, "--verbose") == 0;
                }
            );

            return vpos != argv + argc;
        }();

    std::uint64_t failures = 0;

    // 8-bit integer 
    {
        failures += run_integral_tests <std::int8_t, simd::int8x8_t> (
			"simd::int8x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int8_t, simd::int8x16_t> (
			"simd::int8x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int8_t, simd::int8x32_t> (
			"simd::int8x32_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int8_t, simd::int8x64_t> (
			"simd::int8x64_t", test_length, verbose_output
        );
    }

    // 8-bit unsigned integer 
    {
        failures += run_integral_tests <std::uint8_t, simd::uint8x8_t> (
			"simd::uint8x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint8_t, simd::uint8x16_t> (
			"simd::uint8x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint8_t, simd::uint8x32_t> (
			"simd::uint8x32_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint8_t, simd::uint8x64_t> (
			"simd::uint8x64_t", test_length, verbose_output
        );
    }

    // 16-bit integer 
    {
        failures += run_integral_tests <std::int16_t, simd::int16x8_t> (
			"simd::int16x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int16_t, simd::int16x16_t> (
			"simd::int16x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int16_t, simd::int16x16_t> (
			"simd::int16x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int16_t, simd::int16x32_t> (
			"simd::int16x32_t", test_length, verbose_output
        );
    }

    // 16-bit unsigned integer 
    {
        failures += run_integral_tests <std::uint16_t, simd::uint16x8_t> (
			"simd::uint16x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint16_t, simd::uint16x16_t> (
			"simd::uint16x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint16_t, simd::uint16x16_t> (
			"simd::uint16x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint16_t, simd::uint16x32_t> (
			"simd::uint16x32_t", test_length, verbose_output
        );
    }

    // 32-bit integer 
    {
        failures += run_integral_tests <std::int32_t, simd::int32x2_t> (
			"simd::int32x2_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int32_t, simd::int32x4_t> (
			"simd::int32x4_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int32_t, simd::int32x8_t> (
			"simd::int32x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int32_t, simd::int32x16_t> (
			"simd::int32x16_t", test_length, verbose_output
        );
    }

    // 32-bit unsigned integer 
    {
        failures += run_integral_tests <std::uint32_t, simd::uint32x2_t> (
			"simd::uint32x2_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint32_t, simd::uint32x4_t> (
			"simd::uint32x4_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint32_t, simd::uint32x8_t> (
			"simd::uint32x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint32_t, simd::uint32x16_t> (
			"simd::uint32x16_t", test_length, verbose_output
        );
    }

    // 64-bit integer 
    {
        failures += run_integral_tests <std::int64_t, simd::int64x2_t> (
			"simd::int64x2_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int64_t, simd::int64x4_t> (
			"simd::int64x4_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::int64_t, simd::int64x8_t> (
			"simd::int64x8_t", test_length, verbose_output
        );
    }

    // 64-bit unsigned integer 
    {
        failures += run_integral_tests <std::uint64_t, simd::uint64x2_t> (
			"simd::uint64x2_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint64_t, simd::uint64x4_t> (
			"simd::uint64x4_t", test_length, verbose_output
        );
        failures += run_integral_tests <std::uint64_t, simd::uint64x8_t> (
			"simd::uint64x8_t", test_length, verbose_output
        );
    }

    // 32-bit float 
    {
        failures += run_float_tests <float, simd::float32x4_t> (
            "simd::float32x4_t", test_length, verbose_output
        );
        failures += run_float_tests <float, simd::float32x8_t> (
            "simd::float32x8_t", test_length, verbose_output
        );
        failures += run_float_tests <float, simd::float32x16_t> (
            "simd::float32x16_t", test_length, verbose_output
        );
    }

    // 64-bit float 
    {
        failures += run_float_tests <double, simd::float64x2_t> (
            "simd::float64x2_t", test_length, verbose_output
        );
        failures += run_float_tests <double, simd::float64x4_t> (
            "simd::float64x4_t", test_length, verbose_output
        );
        failures += run_float_tests <double, simd::float64x8_t> (
            "simd::float64x8_t", test_length, verbose_output
        );
    }

    // long double (usually either 80-bit x87 float or 128-bit float)
    {
        failures += run_float_tests <long double, simd::long_doublex2_t> (
            "simd::long_doublex2_t", test_length, verbose_output
        );
        failures += run_float_tests <long double, simd::long_doublex4_t> (
            "simd::long_doublex4_t", test_length, verbose_output
        );
    }

    if (failures != 0) {
        std::cerr << "failed: " << failures << " cases" << std::endl;
        return EXIT_FAILURE;
    } else {
        return EXIT_SUCCESS;
    }
}
