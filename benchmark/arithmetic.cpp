//
// example useage of simd types
//

#include <algorithm>    // std::generate, std::transform, std::minmax_element,
                        // std::sort
#include <array>        // std::array
#include <cassert>      // assert
#include <chrono>       // std::chrono::*
#include <cstddef>      // std::size_t
#include <cstdint>      // std::uint64_t
#include <cstdlib>      // std::atol
#include <fstream>      // std::ofstream
#include <functional>   // std::{plus,minus,multiplies,divides,negate}
                        // std::bind
#include <future>       // std::async, std::launch::*
#include <iostream>     // std::cout
#include <iterator>     // std::begin, std::end, std::iterator_traits
#include <limits>       // std::numeric_limits
#include <numeric>      // std::accumulate
#include <random>       // std::random_device, std::mt19937,
                        // std::uniform_{int,real}_distribution
#include <string>       // std::string
#include <tuple>        // std::tuple
#include <typeinfo>     // typeid
#include <type_traits>  // std::result_of, std::is_{integral,floating_point},
                        // std::conditional
#include <utility>      // std::move
#include <vector>       // std::vector

#include "chrono_io.hpp"
#include "docopt/docopt.h"
#include "simd.hpp"

using clock_type = std::chrono::high_resolution_clock;
using duration = typename clock_type::duration;
using fpduration = std::chrono::duration <double, typename duration::period>;

static std::mutex write_lock;

static std::vector <std::size_t> block_sizes {{
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576
}};

struct benchmark_statistics
{
    // time statistics
    duration time_average;
    duration time_corrected_average;
    duration time_minimum;
    duration time_quartile_1;
    duration time_quartile_2;
    duration time_quartile_3;
    duration time_maximum;

    // throughput statistics in bytes_per_second
    std::uint64_t bytes_per_second_average;
    std::uint64_t bytes_per_second_corrected_average;
    std::uint64_t bytes_per_second_minimum;
    std::uint64_t bytes_per_second_quartile_1;
    std::uint64_t bytes_per_second_quartile_2;
    std::uint64_t bytes_per_second_quartile_3;
    std::uint64_t bytes_per_second_maximum;
};

benchmark_statistics prepare_statistics (std::vector <duration> const & data,
                                         std::size_t block_size)
{
    using time_value_type = typename fpduration::rep;
    using time_period = typename fpduration::period;

    std::vector <time_value_type> numerical_data;
    numerical_data.reserve (data.size ());

    std::transform (
        data.begin (), data.end (), std::back_inserter (numerical_data),
        [] (duration const & d) -> time_value_type {
            return std::chrono::duration_cast <fpduration> (d).count ();
        }
    );
    std::sort (numerical_data.begin (), numerical_data.end ());

    auto const len = numerical_data.size ();
    assert (len >= 5 && "results data length must be at least 5!");

    auto const sum = std::accumulate (
        numerical_data.begin (), numerical_data.end (), time_value_type {0}
    );
    assert (sum > 0.0 && "non-positive sum");

    auto const avg = sum / len;
    assert (avg > 0.0 && "non-positive avg");

    auto const min = numerical_data.front ();
    assert (min > 0.0 && "non-positive min");

    auto const q1  = numerical_data [len / 4];
    assert (q1 > 0.0 && "non-positive q1");

    auto const q2  = numerical_data [len / 2];
    assert (q2 > 0.0 && "non-positive q2");

    auto const q3  = numerical_data [(len * 3) / 4];
    assert (q3 > 0.0 && "non-positive q3");

    auto const max = numerical_data.back ();
    assert (max > 0.0 && "non-positive max");

    {
        auto const iqr = q3 - q1;
        auto const lower = std::max (0.0, q1 - 1.5 * iqr);
        auto const upper = q3 + 1.5 * iqr;
        std::remove_if (
            numerical_data.begin (), numerical_data.end (),
            [lower, upper] (time_value_type v) -> bool
            {
                return v < lower || v > upper;
            }
        );
    }

    auto const corrected_len = numerical_data.size ();
    auto const corrected_sum = std::accumulate (
        numerical_data.begin (), numerical_data.end (), time_value_type {0}
    );
    auto const corrected_avg = corrected_sum / corrected_len;

    return benchmark_statistics {
        std::chrono::duration_cast <duration> (fpduration {avg}),
        std::chrono::duration_cast <duration> (fpduration {corrected_avg}),
        std::chrono::duration_cast <duration> (fpduration {min}),
        std::chrono::duration_cast <duration> (fpduration {q1}),
        std::chrono::duration_cast <duration> (fpduration {q2}),
        std::chrono::duration_cast <duration> (fpduration {q3}),
        std::chrono::duration_cast <duration> (fpduration {max}),
        static_cast <std::uint64_t> (
            block_size / (avg * time_period::num / time_period::den)
        ),
        static_cast <std::uint64_t> (
            block_size / (corrected_avg * time_period::num / time_period::den)
        ),
        static_cast <std::uint64_t> (
            block_size / (min * time_period::num / time_period::den)
        ),
        static_cast <std::uint64_t> (
            block_size / (q1  * time_period::num / time_period::den)
        ),
        static_cast <std::uint64_t> (
            block_size / (q2  * time_period::num / time_period::den)
        ),
        static_cast <std::uint64_t> (
            block_size / (q3  * time_period::num / time_period::den)
        ),
        static_cast <std::uint64_t> (
            block_size / (max * time_period::num / time_period::den)
        )
    };
}

template <typename T, typename BinaryOp>
#if defined (__GNUG__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
duration run_benchmark_non_vectorized (std::vector <T> const & lhs,
                                       std::vector <T> const & rhs,
                                       BinaryOp op)
{
    assert (lhs.size () == rhs.size ());

    std::vector <T> result;
    result.resize (lhs.capacity ());

    auto const start = clock_type::now ();
    {
#if defined(__clang__)
    #pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (std::size_t i = 0; i < lhs.size (); ++i)
            result [i] = op (lhs [i], rhs [i]);
    }
    return clock_type::now () - start;
}

template <typename T, typename Alloc, typename BinaryOp>
duration run_benchmark_vectorized (std::vector <T, Alloc> const & lhs,
                                   std::vector <T, Alloc> const & rhs,
                                   BinaryOp op)
{
    assert (lhs.size () == rhs.size ());

    std::vector <T> result;
    result.resize (lhs.capacity ());

    auto const start = clock_type::now ();
    {
        for (std::size_t i = 0; i < lhs.size (); ++i)
            result [i] = op (lhs [i], rhs [i]);
    }
    return clock_type::now () - start;
}

template <typename Op, typename Operand>
std::vector <duration> bench_non_vectorized (std::size_t rep, std::size_t len)
{
    using operand_type = Operand;
    using distribution = typename std::conditional <
        std::is_integral <operand_type>::value,
        std::uniform_int_distribution <operand_type>,
        std::uniform_real_distribution <operand_type>
    >::type;

    std::random_device rd;
    auto const gen {
        std::bind (distribution {}, std::mt19937 {rd ()})
    };

    std::vector <duration> runtimes;
    runtimes.reserve (rep);

    for (std::size_t i = 0; i < rep; ++i) {
        std::vector <operand_type> lhs (len);
        std::vector <operand_type> rhs (len);

        std::generate (lhs.begin (), lhs.end (), gen);
        std::generate (rhs.begin (), rhs.end (), gen);

        if (std::is_same <Op, std::divides <Operand>>::value)
        {
            std::replace (rhs.begin (), rhs.end (), Operand {0}, Operand {1});
        }

        {
            static Op op {};
            runtimes.emplace_back (run_benchmark_non_vectorized (lhs, rhs, op));
        }
    }

    return runtimes;
}

template <typename Op, typename Operand>
std::vector <duration> bench_vectorized (std::size_t rep, std::size_t len)
{
    using operand_type = Operand;
    using traits_type = simd::simd_traits <operand_type>;
    using value_type = typename traits_type::value_type;
    static constexpr auto lanes = traits_type::lanes;

    struct simd_allocator
    {
        using value_type = Operand;

        simd_allocator (void) noexcept = default;
        ~simd_allocator (void) noexcept = default;
        simd_allocator (simd_allocator const &) noexcept = default;
        simd_allocator & operator= (simd_allocator const &) noexcept = default;

        value_type * allocate (std::size_t n)
        {
            return new value_type [n];
        }

        void deallocate (value_type * ptr, std::size_t n) noexcept
        {
            (void) n;
            delete [] ptr;
        }

        bool operator== (simd_allocator const &) const noexcept
        {
            return true;
        }

        bool operator!= (simd_allocator const &) const noexcept
        {
            return false;
        }
    };

    static auto gen = [] (void) -> operand_type
    {
        using distribution = typename std::conditional <
            std::is_integral <value_type>::value,
            std::uniform_int_distribution <value_type>,
            std::uniform_real_distribution <value_type>
        >::type;

        static std::random_device rd;
        static auto g = std::bind (distribution {}, std::mt19937 {rd ()});

        std::array <value_type, lanes> values;
        std::generate_n (values.begin (), lanes, g);
        return operand_type {values};
    };

    std::vector <duration> runtimes;
    runtimes.reserve (rep);

    assert (len % lanes == 0 && "cannot evenly distribute operands across SIMD vector");
    auto const use_length = len / lanes;

    for (std::size_t i = 0; i < rep; ++i) {
        std::vector <operand_type, simd_allocator> lhs (use_length);
        std::vector <operand_type, simd_allocator> rhs (use_length);

        std::generate (lhs.begin (), lhs.end (), gen);
        std::generate (rhs.begin (), rhs.end (), gen);

        if (std::is_same <Op, std::divides <Operand>>::value)
        {
            static operand_type const one_vec (value_type {1});
            static auto any_zero = [](operand_type const & o) -> bool
            {
                static operand_type const zero_vec (value_type {0});
                return (o == zero_vec).any_of ();
            };

            std::replace_if (rhs.begin (), rhs.end (), any_zero, one_vec);
        }

        {
            static Op op {};
            runtimes.emplace_back (run_benchmark_vectorized (lhs, rhs, op));
        }
    }

    return runtimes;
}

template <
    typename ScalarT, typename Vec128, typename Vec256, typename Vec512
>
void benchmark (std::string const & name,
                std::size_t reps,
                std::vector <std::size_t> lengths,
                std::ostream & status_log,
                std::ostream & result_log)
{
    status_log << "running benchmarks for: " << name << std::endl;
    result_log << "[[type:" << name << "]]" << std::endl;
    auto add_results = std::async (
        std::launch::async|std::launch::deferred,
        [reps, lengths](void) {
            auto results = std::make_tuple (
                std::vector <std::vector <duration>> {},
                std::vector <std::vector <duration>> {},
                std::vector <std::vector <duration>> {},
                std::vector <std::vector <duration>> {}
            );

            std::get <0> (results).reserve (lengths.size ());
            std::get <1> (results).reserve (lengths.size ());
            std::get <2> (results).reserve (lengths.size ());
            std::get <3> (results).reserve (lengths.size ());

            auto const r = reps;
            for (auto const l : lengths) {
                std::get <0> (results).emplace_back (
                    bench_non_vectorized <std::plus <ScalarT>, ScalarT> (r, l)
                );
                std::get <1> (results).emplace_back (
                    bench_vectorized <std::plus <Vec128>, Vec128> (r, l)
                );
                std::get <2> (results).emplace_back (
                    bench_vectorized <std::plus <Vec256>, Vec256> (r, l)
                );
                std::get <3> (results).emplace_back (
                    bench_vectorized <std::plus <Vec512>, Vec512> (r, l)
                );
            }

            return results;
        }
    );

    auto sub_results = std::async (
        std::launch::async|std::launch::deferred,
        [reps, lengths](void) {
            auto results {
                std::make_tuple (
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {}
                )
            };

            std::get <0> (results).reserve (lengths.size ());
            std::get <1> (results).reserve (lengths.size ());
            std::get <2> (results).reserve (lengths.size ());
            std::get <3> (results).reserve (lengths.size ());

            auto const r {reps};
            for (auto l : lengths) {
                std::get <0> (results).emplace_back (
                    bench_non_vectorized <std::minus <ScalarT>, ScalarT> (r, l)
                );
                std::get <1> (results).emplace_back (
                    bench_vectorized <std::minus <Vec128>, Vec128> (r, l)
                );
                std::get <2> (results).emplace_back (
                    bench_vectorized <std::minus <Vec256>, Vec256> (r, l)
                );
                std::get <3> (results).emplace_back (
                    bench_vectorized <std::minus <Vec512>, Vec512> (r, l)
                );
            }

            return results;
        }
    );

    auto mul_results = std::async (
        std::launch::async|std::launch::deferred,
        [reps, lengths](void) {
            auto results {
                std::make_tuple (
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {}
                )
            };

            std::get <0> (results).reserve (lengths.size ());
            std::get <1> (results).reserve (lengths.size ());
            std::get <2> (results).reserve (lengths.size ());
            std::get <3> (results).reserve (lengths.size ());

            auto const r {reps};
            for (auto l : lengths) {
                std::get <0> (results).emplace_back (
                    bench_non_vectorized <std::multiplies <ScalarT>, ScalarT> (
                        r, l
                    )
                );
                std::get <1> (results).emplace_back (
                    bench_vectorized <std::multiplies <Vec128>, Vec128> (r, l)
                );
                std::get <2> (results).emplace_back (
                    bench_vectorized <std::multiplies <Vec256>, Vec256> (r, l)
                );
                std::get <3> (results).emplace_back (
                    bench_vectorized <std::multiplies <Vec512>, Vec512> (r, l)
                );
            }

            return results;
        }
    );

    auto div_results = std::async (
        std::launch::async|std::launch::deferred,
        [reps, lengths](void) {
            auto results {
                std::make_tuple (
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {},
                    std::vector <std::vector <duration>> {}
                )
            };

            std::get <0> (results).reserve (lengths.size ());
            std::get <1> (results).reserve (lengths.size ());
            std::get <2> (results).reserve (lengths.size ());
            std::get <3> (results).reserve (lengths.size ());

            auto const r {reps};
            for (auto l : lengths) {
                std::get <0> (results).emplace_back (
                    bench_non_vectorized <std::divides <ScalarT>, ScalarT> (
                        r, l
                    )
                );
                std::get <1> (results).emplace_back (
                    bench_vectorized <std::divides <Vec128>, Vec128> (r, l)
                );
                std::get <2> (results).emplace_back (
                    bench_vectorized <std::divides <Vec256>, Vec256> (r, l)
                );
                std::get <3> (results).emplace_back (
                    bench_vectorized <std::divides <Vec512>, Vec512> (r, l)
                );
            }

            return results;
        }
    );

    auto print_results = [&result_log, &lengths] (auto const & results)
    {
        auto print_stats = [&result_log] (auto const & result, auto block_size)
        {
            auto const stats = prepare_statistics (result, block_size);

            /* result logging */
            {
                using namespace date;

                using rep = typename fpduration::rep;
                static constexpr auto prec {
                    std::numeric_limits <rep>::digits10 + 1
                };

                result_log << "[[section:timing]]\n";
                result_log << "avg=" << stats.time_average << '\n';
                result_log << "cavg=" << stats.time_corrected_average << '\n';
                result_log << "min=" << stats.time_minimum << '\n';
                result_log << "q1=" << stats.time_quartile_1 << '\n';
                result_log << "q2=" << stats.time_quartile_2 << '\n';
                result_log << "q3=" << stats.time_quartile_3 << '\n';
                result_log << "max=" << stats.time_maximum << std::endl;

                result_log << "[[section:throughput]]\n";
                result_log << "avg=" << stats.bytes_per_second_average << '\n';
                result_log << "cavg=" << stats.bytes_per_second_corrected_average << '\n';
                result_log << "min=" << stats.bytes_per_second_minimum << '\n';
                result_log << "q1=" << stats.bytes_per_second_quartile_1 << '\n';
                result_log << "q2=" << stats.bytes_per_second_quartile_2 << '\n';
                result_log << "q3=" << stats.bytes_per_second_quartile_3 << '\n';
                result_log << "max=" << stats.bytes_per_second_maximum << std::endl;
            }
        };

        for (std::size_t i = 0; i < lengths.size (); ++i) {
            auto const len {lengths [i]};
            result_log << "[[len:" << len << "]]\n";

            /* vec types: */
            auto const block_size {len * sizeof (ScalarT)};
            /* non-vectorized code */
            result_log << "[[vec-type:non-vec]]\n";
            print_stats (std::get <0> (results) [i], block_size);

            /* 128-bit vectorized */
            result_log << "[[vec-type:128bit-vec]]\n";
            print_stats (std::get <1> (results) [i], block_size);

            /* 256-bit vectorized */
            result_log << "[[vec-type:256bit-vec]]\n";
            print_stats (std::get <2> (results) [i], block_size);

            /* 512-bit vectorized */
            result_log << "[[vec-type:512bit-vec]]\n";
            print_stats (std::get <3> (results) [i], block_size);
        }
    };

    status_log << "waiting on results..." << std::flush;

    add_results.wait ();
    sub_results.wait ();
    mul_results.wait ();
    div_results.wait ();

    status_log << " done" << std::endl;

    result_log << "[[op:+]]\n";
    print_results (add_results.get ());

    result_log << "[[op:-]]\n";
    print_results (sub_results.get ());

    result_log << "[[op:*]]\n";
    print_results (mul_results.get ());

    result_log << "[[op:/]]\n";
    print_results (div_results.get ());
}

static const char * command_line_usage =
R"(
usage:
    benchmark [--reps=<reps>] [--status_log=<slog>] [--results_log=<rlog>]
    benchmark (-h | --help)

options:
    -h --help             display this information
    --reps=<reps>         the number of repetitions for each benchmark length, minimum is 5 [default: 25]
    --status_log=<slog>   output location for benchmark status updates [default: stderr]
    --results_log=<rlog>  output location for benchmark results [default: stdout]
)";


int main (int argc, char ** argv)
{
    /*
     * Computes benchmark time statistics for regular and vectorized code over:
     *
     * non-vectorized types and,
     *
     * vectorized types (some or all may be emulated based on cpu arch):
     *      128 bit, 256 bit, 512 bit
     *
     * over data types:
     *      signed integers:   8 bit, 16 bit, 32 bit, 64 bit
     *      floating point:    32 bit, 64 bit
     *
     * with operations:
     *      +, -, *, /
     *
     * with # of elements:
     *      512, 1024, 2056, 4092, 8192, 16384, 32786,
     *      65536, 131072, 262144, 524288, 1048576
     */

    auto args = docopt::docopt (
        command_line_usage, {argv + 1, argv + argc}, true
    );

    auto const reps {
        [&args] (void) -> std::size_t
        {
            static auto is_number_string = [](std::string const & s) {
                for (auto const & c : s) {
                    if (!std::isdigit (c)) {
                        return false;
                    }
                }
                return true;
            };

            auto const str {args ["--reps"].asString ()};
            if (!is_number_string (str)) {
                std::cerr << "benchmark: illegal option: --reps="
                          << str
                          << " -- value must be a number.\n";
                std::cerr << command_line_usage << std::endl;
                std::exit (1);
            } else {
                return static_cast <std::size_t> (std::atol (str.c_str ()));
            }
        } ()
    };

    if (reps < 5) {
        std::cerr << "benchmark: illegal option: --reps="
                  << reps
                  << " -- value must be at least 5.\n";
        std::cerr << command_line_usage << std::endl;
        std::exit (1);
    }

    auto const status_log_name  = args ["--status_log"].asString ();
    auto const results_log_name = args ["--results_log"].asString ();

    std::ofstream status_file;
    if (status_log_name != "stderr") {
        status_file.open (status_log_name);
    }

    std::ofstream results_file;
    if (results_log_name != "stdout") {
        results_file.open (results_log_name);
    }

    std::ostream & status  = status_file.is_open () ? status_file : std::cerr;
    std::ostream & results = results_file.is_open () ? results_file : std::cout;

    benchmark <
        std::int8_t, simd::int8x16_t, simd::int8x32_t, simd::int8x64_t
    > ("int8_t", reps, block_sizes, status, results);

    benchmark <
        std::int16_t, simd::int16x8_t, simd::int16x16_t, simd::int16x32_t
    > ("int16_t", reps, block_sizes, status, results);

    benchmark <
        std::int32_t, simd::int32x4_t, simd::int32x8_t, simd::int32x16_t
    > ("int32_t", reps, block_sizes, status, results);

    benchmark <
        std::int64_t, simd::int64x2_t, simd::int64x4_t, simd::int64x8_t
    > ("int64_t", reps, block_sizes, status, results);

    benchmark <
        float, simd::float32x4_t, simd::float32x8_t, simd::float32x16_t
    > ("float32", reps, block_sizes, status, results);

    benchmark <
        double, simd::float64x2_t, simd::float64x4_t, simd::float64x8_t
    > ("float64", reps, block_sizes, status, results);

    return 0;
}
