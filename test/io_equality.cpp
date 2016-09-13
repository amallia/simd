//
// verifies that input and output works for simd types by
// performing input -> output and comparing strings and
// performing output -> input and comparing values.
//

#include <algorithm>    // std::generate, std::min
#include <climits>      // ULONG_MAX
#include <iomanip>      // std::precision, std::dec, std::hex, std::oct
#include <iostream>     // std::cout, std::cerr
#include <random>       // std::random_device, std::mt19937
                        // std::uniform_{int,real}_distribution
#include <string>       // std::string
#include <sstream>      // std::stringstream
#include <type_traits>  // std::is_same
#include <vector>       // std::vector

#include "simd.hpp"


struct input_tag {};
struct output_tag {};

enum status : bool
{
    pass = true,
    fail = false
};

std::random_device & random_device (void) noexcept
{
    static std::random_device rd;
    return rd;
}

/*
 * This method tests that deserializing a vector type (in various formats)
 * produces the correct vector value.
 */
template <typename SIMDType, typename T, std::size_t N>
enum status compute_and_verify (std::array <T, N> const & arg,
                                std::vector <std::string> & errors,
                                input_tag)
{
    using traits     = simd::simd_traits <SIMDType>;
    using value_type = typename traits::value_type;
    static constexpr auto lanes = traits::lanes;

    static_assert (
        std::is_same <T, value_type>::value && N == lanes,
        "argument mismatch"
    );

    SIMDType const expected_vector {arg};

    bool okay = true;

    // whitespace separated (if thousands sep is not whitespace)
    {
        std::stringstream str_form;
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << " ";
        }
        str_form << +arg [lanes - 1];

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[ws sep]] incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // bracketed, separated by whitespace (if thousands sep is not space)
    {
        std::stringstream str_form;
        str_form << "[";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << " ";
        }
        str_form << +arg [lanes - 1] << "]";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[brackets w/ ws sep]] incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // bracketed, separated by commas
    {
        std::stringstream str_form;
        str_form << "[";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << ",";
        }
        str_form << +arg [lanes - 1] << "]";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[brackets w/ comma sep]] incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // bracketed, separated by commas && trailing space
    {
        std::stringstream str_form;
        str_form << "[";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << ", ";
        }
        str_form << +arg [lanes - 1] << "]";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[brackets w/ comma sep & trailing ws]]"
                   " incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // bracketed, separated by semicolons
    {
        std::stringstream str_form;
        str_form << "[";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << ";";
        }
        str_form << +arg [lanes - 1] << "]";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[brackets w/ semicolon sep]] incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // bracketed, separated by semicolons && trailing space
    {
        std::stringstream str_form;
        str_form << "[";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << "; ";
        }
        str_form << +arg [lanes - 1] << "]";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[brackets w/ semicolon sep & trailing ws]]"
                   " incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // parentheses, separated by whitespace (if thousands sep is not space)
    {
        std::stringstream str_form;
        str_form << "(";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << " ";
        }
        str_form << +arg [lanes - 1] << ")";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[parens w/ ws sep]] incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // parentheses, separated by commas
    {
        std::stringstream str_form;
        str_form << "(";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << ",";
        }
        str_form << +arg [lanes - 1] << ")";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[parens w/ comma sep]] incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // parentheses, separated by commas && trailing space
    {
        std::stringstream str_form;
        str_form << "(";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << ", ";
        }
        str_form << +arg [lanes - 1] << ")";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[parens w/ comma sep & trailing ws]]"
                   " incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // parentheses, separated by semicolons
    {
        std::stringstream str_form;
        str_form << "(";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << ";";
        }
        str_form << +arg [lanes - 1] << ")";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[parens w/ semicolon sep]] incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // parentheses, separated by semicolons && trailing space
    {
        std::stringstream str_form;
        str_form << "(";
        for (std::size_t i = 0; i < lanes - 1; ++i) {
            str_form << +arg [i] << "; ";
        }
        str_form << +arg [lanes - 1] << ")";

        SIMDType result;
        {
            using namespace simd;
            str_form >> result;
        }

        if ((result != expected_vector).any_of ()) {
            std::ostringstream err;
            err << "[[parens w/ semicolon sep & trailing ws]]"
                   " incorrect values obtained:\n";
            for (std::size_t i = 0; i < lanes; ++i) {
                err << "\t[" << i << "]\t"
                    << +expected_vector.value (i) << ", "
                    << +result.value (i)
                    << "\n";
            }
            errors.push_back (err.str ());
            okay = false;
        }
    }

    return okay ? status::pass : status::fail;
}

/*
 * This method tests that serializing a vector type produces the correct output.
 */
template <typename SIMDType, typename T, std::size_t N>
enum status compute_and_verify (std::array <T, N> const & arg,
                                std::vector <std::string> & errors,
                                output_tag)
{
    using traits     = simd::simd_traits <SIMDType>;
    using value_type = typename traits::value_type;
    static constexpr auto lanes = traits::lanes;

    static_assert (
        std::is_same <T, value_type>::value && N == lanes,
        "argument mismatch"
    );

    SIMDType const test_vector {arg};

    bool okay = true;

    // decimal format
    {
        std::stringstream expected_output;
        {
            expected_output << "(";
            for (std::size_t i = 0; i < lanes - 1; ++i) {
                expected_output << std::dec << +arg [i] << ";";
            }
            expected_output << std::dec << +arg [lanes - 1] << ")";
        }

        std::stringstream result_output;
        {
            using namespace simd;
            result_output << std::dec << test_vector; 
        }

        if (expected_output.str () != result_output.str ()) {
            std::ostringstream err;
            err << "[[decimal]] incorrect output obtained:\n";
            err << "\texpected: " << expected_output.str () << "\n";
            err << "\tobtained: " << result_output.str () << "\n";
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // octal format
    {
        std::stringstream expected_output;
        {
            expected_output << "(";
            for (std::size_t i = 0; i < lanes - 1; ++i) {
                expected_output << std::oct << +arg [i] << ";";
            }
            expected_output << std::oct << +arg [lanes - 1] << ")";
        }

        std::stringstream result_output;
        {
            using namespace simd;
            result_output << std::oct << test_vector; 
        }

        if (expected_output.str () != result_output.str ()) {
            std::ostringstream err;
            err << "[[octal]] incorrect output obtained:\n";
            err << "\texpected: " << expected_output.str () << "\n";
            err << "\tobtained: " << result_output.str () << "\n";
            errors.push_back (err.str ());
            okay = false;
        }
    }

    // hexadecimal format
    {
        std::stringstream expected_output;
        {
            expected_output << "(";
            for (std::size_t i = 0; i < lanes - 1; ++i) {
                expected_output << std::hex << +arg [i] << ";";
            }
            expected_output << std::hex << +arg [lanes - 1] << ")";
        }

        std::stringstream result_output;
        {
            using namespace simd;
            result_output << std::hex << test_vector; 
        }

        if (expected_output.str () != result_output.str ()) {
            std::ostringstream err;
            err << "[[hexadecimal]] incorrect output obtained:\n";
            err << "\texpected: " << expected_output.str () << "\n";
            err << "\tobtained: " << result_output.str () << "\n";
            errors.push_back (err.str ());
            okay = false;
        }
    }

    return okay ? status::pass : status::fail;
}

template <typename SIMDType, typename IOTag>
std::uint64_t generate_and_test_cases (std::size_t len,
                                       std::ostream & logos,
                                       std::vector <std::string> & errors,
                                       bool verbose_output,
                                       IOTag)
{
    using traits_type  = simd::simd_traits <SIMDType>;
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

    static auto gen = [] (void) -> std::array <value_type, lanes>
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
        return values;
    };

    std::vector <std::array <value_type, lanes>> args;
    {
        args.resize (len);
        std::generate (args.begin (), args.end (), gen);
    }

    std::uint64_t fail_count = 0;
    for (std::size_t i = 0; i < len; ++i) {
        switch (compute_and_verify <SIMDType> (args [i], errors, IOTag {})) {
            case status::fail:
                fail_count += 1;
                break;
            case status::pass:
                break;
            default:
                break;
        }

        if (verbose_output) {
            logos << "\r\t" << "[" << i + 1 << "/" << len << "]"
                  << std::flush;
        }
    }

    return fail_count;
}

template <typename SIMDType>
std::uint64_t run_integral_tests (std::string name,
                                  std::size_t test_length,
                                  bool verbose_output)
{
    std::vector <std::string> errors;
    std::uint64_t test_fail_count = 0;

    {
        std::cout << name << " (>>)" << std::endl;
        auto fail_count = generate_and_test_cases <SIMDType> (
            test_length, std::cout, errors, verbose_output, input_tag {}
        );

        if (fail_count != 0) {
            if (verbose_output) {
                std::cout << "\t... failed: " << fail_count << " ..."
                          << std::endl;
            }

            if (!verbose_output && fail_count > 5) {
                std::cout << "truncating output to 5 error logs...\n";
            }

            if (verbose_output) {
                for (auto const & e : errors) {
                    std::cerr << e;
                }
            } else {
                using fctype = decltype (fail_count);
                for (std::size_t i = 0;
                     i < std::min (fctype {5ull}, fail_count);
                     ++i)
                {
                    std::cerr << errors [i];
                }
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            if (verbose_output) {
                std::cout << "\t... ok ..." << std::endl;
            }
        }
    }

    {
        std::cout << name << " (<<)" << std::endl;
        auto fail_count = generate_and_test_cases <SIMDType> (
            test_length, std::cout, errors, verbose_output, output_tag {}
        );

        if (fail_count != 0) {
            if (verbose_output) {
                std::cout << "\t... failed: " << fail_count << " ..."
                          << std::endl;
            }

            if (!verbose_output && fail_count > 5) {
                std::cout << "truncating output to 5 error logs...\n";
            }

            if (verbose_output) {
                for (auto const & e : errors) {
                    std::cerr << e;
                }
            } else {
                using fctype = decltype (fail_count);
                for (std::size_t i = 0;
                     i < std::min (fctype {5ull}, fail_count);
                     ++i)
                {
                    std::cerr << errors [i];
                }
            }

            errors.clear ();
            test_fail_count += fail_count;
        } else {
            if (verbose_output) {
                std::cout << "\t... ok ..." << std::endl;
            }
        }
    }

    return test_fail_count;
}
/*
template <typename SIMDType>
std::uint64_t run_float_tests (std::string name,
                               std::size_t test_length,
                               bool verbose_output)
{
    std::vector <std::string> errors;
    std::uint64_t test_fail_count = 0;

    return test_fail_count;
}
*/
int main (int argc, char ** argv)
{
    auto const test_length =
        [argc, argv] (void) -> std::size_t
        {
            constexpr std::size_t default_test_length = 10000;
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
        failures += run_integral_tests <simd::int8x8_t> (
			"simd::int8x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int8x16_t> (
			"simd::int8x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int8x32_t> (
			"simd::int8x32_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int8x64_t> (
			"simd::int8x64_t", test_length, verbose_output
        );
    }

    // 8-bit unsigned integer 
    {
        failures += run_integral_tests <simd::uint8x8_t> (
			"simd::uint8x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint8x16_t> (
			"simd::uint8x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint8x32_t> (
			"simd::uint8x32_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint8x64_t> (
			"simd::uint8x64_t", test_length, verbose_output
        );
    }

    // 16-bit integer 
    {
        failures += run_integral_tests <simd::int16x8_t> (
			"simd::int16x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int16x16_t> (
			"simd::int16x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int16x16_t> (
			"simd::int16x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int16x32_t> (
			"simd::int16x32_t", test_length, verbose_output
        );
    }

    // 16-bit unsigned integer 
    {
        failures += run_integral_tests <simd::uint16x8_t> (
			"simd::uint16x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint16x16_t> (
			"simd::uint16x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint16x16_t> (
			"simd::uint16x16_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint16x32_t> (
			"simd::uint16x32_t", test_length, verbose_output
        );
    }

    // 32-bit integer 
    {
        failures += run_integral_tests <simd::int32x2_t> (
			"simd::int32x2_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int32x4_t> (
			"simd::int32x4_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int32x8_t> (
			"simd::int32x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int32x16_t> (
			"simd::int32x16_t", test_length, verbose_output
        );
    }

    // 32-bit unsigned integer 
    {
        failures += run_integral_tests <simd::uint32x2_t> (
			"simd::uint32x2_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint32x4_t> (
			"simd::uint32x4_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint32x8_t> (
			"simd::uint32x8_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint32x16_t> (
			"simd::uint32x16_t", test_length, verbose_output
        );
    }

    // 64-bit integer 
    {
        failures += run_integral_tests <simd::int64x2_t> (
			"simd::int64x2_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int64x4_t> (
			"simd::int64x4_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::int64x8_t> (
			"simd::int64x8_t", test_length, verbose_output
        );
    }

    // 64-bit unsigned integer 
    {
        failures += run_integral_tests <simd::uint64x2_t> (
			"simd::uint64x2_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint64x4_t> (
			"simd::uint64x4_t", test_length, verbose_output
        );
        failures += run_integral_tests <simd::uint64x8_t> (
			"simd::uint64x8_t", test_length, verbose_output
        );
    }
/*
    // 32-bit float 
    {
        failures += run_float_tests <simd::float32x4_t> (
            "simd::float32x4_t", test_length, verbose_output
        );
        failures += run_float_tests <simd::float32x8_t> (
            "simd::float32x8_t", test_length, verbose_output
        );
        failures += run_float_tests <simd::float32x16_t> (
            "simd::float32x16_t", test_length, verbose_output
        );
    }

    // 64-bit float 
    {
        failures += run_float_tests <simd::float64x2_t> (
            "simd::float64x2_t", test_length, verbose_output
        );
        failures += run_float_tests <simd::float64x4_t> (
            "simd::float64x4_t", test_length, verbose_output
        );
        failures += run_float_tests <simd::float64x8_t> (
            "simd::float64x8_t", test_length, verbose_output
        );
    }

    // long double 
    {
        failures += run_float_tests <simd::long_doublex2_t> (
            "simd::long_doublex2_t", test_length, verbose_output
        );
        failures += run_float_tests <simd::long_doublex4_t> (
            "simd::long_doublex4_t", test_length, verbose_output
        );
    }
*/
    if (failures != 0) {
        std::cerr << "failed: " << failures << " cases" << std::endl;
        return EXIT_FAILURE;
    } else {
        return EXIT_SUCCESS;
    }
}
