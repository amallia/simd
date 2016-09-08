//
// mandelbrot image generation example using simd types in the region
// [-2, 1] x [-1, -1] of the complex plane.
//
// Runs both non-vectorized and vectorized code to compare times and assert
// equality of results. Outputs two grey-scale PGM images to compare
// nonvectorized and vectorized results.
//
// Two integral command line arguments define the xdim and ydim of the resulting
// image, respectively, and therefore also define the x-step and y-setp of the
// computation. The provided values are always rounded up to an even multiple of
// eight for convenience in vectorization. Values of 0 are interpreted as
// default.
//
// A third command line argument can also be provided to determine the maximum
// number of iterations in each calculation. A value of 0 is interpreted as
// default.
//

#include <algorithm>    // std::generate, std::transform
#include <cassert>
#include <chrono>       // std::chrono::*
#include <complex>      // std::complex
#include <cstddef>      // std::size_t
#include <cstdint>      // std::uint8_t
#include <fstream>      // std::ofstream
#include <iomanip>      // std::setprecision
#include <climits>      // ULONG_MAX
#include <string>       // std::string, std::to_string
#include <tuple>        // std::tuple
#include <utility>      // std::make_pair, std::pair
#include <vector>       // std::vector

#include "chrono_io.hpp"
#include "simd.hpp"


#if defined (__GNUG__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
std::uint32_t mandelbrot_nonvec (float re, float im, std::uint32_t max_iter)
    noexcept
{
    auto const re_start = re;
    auto const im_start = im;

    std::uint32_t count = 0;
#if defined (__clang__)
    #pragma clang loop vectorize(disable)
#endif
    for (; count < max_iter; ++count) {
        auto const ri  = re * im;
        auto const rr  = re * re;
        auto const ii  = im * im;
        auto const sum = rr + ii;

        if (sum > float (4.0)) {
            break;
        }

        re = rr - ii + re_start;
        im = ri + ri + im_start;
    }

    return count;
}

simd::uint32x4_t mandelbrot_vec128 (simd::float32x4_t re,
                                    simd::float32x4_t im,
                                    std::uint32_t max_iter) noexcept
{
    static constexpr simd::float32x4_t four {4.0};
    auto const re_start = re;
    auto const im_start = im;

    simd::uint32x4_t count {0};

    while (max_iter--) {
        auto const ri  = re * im;
        auto const rr  = re * re;
        auto const ii  = im * im;
        auto const msq = rr + ii;

        auto const compare = msq < four;
        if (!compare.any_of ()) {
            break;
        } else {
            count += compare.as <simd::uint32x4_t> ();
        }

        re = rr - ii + re_start;
        im = ri + ri + im_start;
    }

    return count;
}

simd::uint32x8_t mandelbrot_vec256 (simd::float32x8_t re,
                                    simd::float32x8_t im,
                                    std::uint32_t max_iter) noexcept
{
    static constexpr simd::float32x8_t four {4.0};
    auto const re_start = re;
    auto const im_start = im;

    simd::uint32x8_t count {0};

    while (max_iter--) {
        auto const ri  = re * im;
        auto const rr  = re * re;
        auto const ii  = im * im;
        auto const msq = rr + ii;

        auto const compare = msq < four;
        if (!compare.any_of ()) {
            break;
        } else {
            count += compare.as <simd::uint32x8_t> ();
        }

        re = rr - ii + re_start;
        im = ri + ri + im_start;
    }

    return count;
}

int main (int argc, char ** argv)
{
    using clock = std::chrono::high_resolution_clock;

    auto const dims =
        [argc, argv] (void) -> std::pair <std::size_t, std::size_t>
        {
            static constexpr std::size_t default_xdim = 1200;
            static constexpr std::size_t default_ydim = 800;

            if (argc < 3) {
                return std::make_pair (default_xdim, default_ydim);
            } else {
                auto const xarg = std::strtoul (argv [1], nullptr, 10);
                auto const yarg = std::strtoul (argv [2], nullptr, 10);

                return std::make_pair (
                    xarg == 0 || xarg == ULONG_MAX ? default_xdim
                                                   : xarg + xarg % 8,
                    yarg == 0 || yarg == ULONG_MAX ? default_ydim
                                                   : yarg + yarg % 8
                );
            }
        }();

    auto const max_iter =
        [argc, argv] (void) -> std::uint32_t
        {
            static constexpr std::uint32_t default_max_iter = 100;

            if (argc < 4) {
                return default_max_iter;
            } else {
                auto const iarg = std::strtoul (argv [3], nullptr, 10);
                return iarg == 0 || iarg == ULONG_MAX ? default_max_iter : iarg;
            }
        }();

    auto const re_step = float (3.0) / dims.first;
    auto const im_step = float (2.0) / dims.second;
    auto const data_count = dims.first * dims.second;

    clock::duration nonvec_time;
    clock::duration vec128_time;
    clock::duration vec256_time;

    // non-vectorized
    {
        std::vector <std::uint32_t> step_counts;
        step_counts.reserve (data_count);

        {
            auto const start = clock::now ();
            for (std::size_t y = 0; y < dims.second; ++y) {
                auto const im = float(1.0) - y * im_step;
                for (std::size_t x = 0; x < dims.first; ++x) {
                    auto const re = float(-2.0) + x * re_step;
                    step_counts.emplace_back (
                        mandelbrot_nonvec (re, im, max_iter)
                    );
                }
            }
            nonvec_time = clock::now () - start;

            using namespace date;
            std::cout << "non-vectorized time: " << nonvec_time << std::endl;
        }

        {
            std::string const ofile = std::string ("mandelbrot-")
                                    + std::to_string (dims.first)
                                    + std::string ("x")
                                    + std::to_string (dims.second)
                                    + std::string ("-nonvec.pgm");

            std::ofstream out (ofile, std::ios_base::out|std::ios_base::binary);
            out << "P5\n" << dims.first << " " << dims.second << "\n255\n";

            for (auto const c : step_counts) {
                // interpolate between 0 (black) and 255 (white) from the
                // step count which may range from 0 to max_iter.
                auto const lerp = static_cast <std::uint8_t> (
                    255 - 255 * (static_cast <float> (c) /
                                 static_cast <float> (max_iter))
                );
                out << lerp;
            }
        }
    }

    // 128-bit vectorized
    {
        std::vector <simd::uint32x4_t, simd::allocator <simd::uint32x4_t>>
            step_counts;
        step_counts.reserve (data_count / 4);

        {
            auto const start = clock::now ();
            for (std::size_t y = 0; y < dims.second; ++y) {
                simd::float32x4_t const im {float(1.0) - y * im_step};
                for (std::size_t x = 0; x < dims.first; x += 4) {
                    simd::float32x4_t const re {
                        float(-2.0) + (x + 0) * re_step,
                        float(-2.0) + (x + 1) * re_step,
                        float(-2.0) + (x + 2) * re_step,
                        float(-2.0) + (x + 3) * re_step
                    };
                    step_counts.emplace_back (
                        mandelbrot_vec128 (re, im, max_iter)
                    );
                }
            }
            vec128_time = clock::now () - start;

            using namespace date;
            std::cout << "128-bit vectorized time: "
                      << vec128_time
                      << std::endl;
        }

        {
            std::string const ofile = std::string ("mandelbrot-")
                                    + std::to_string (dims.first)
                                    + std::string ("x")
                                    + std::to_string (dims.second)
                                    + std::string ("-vec128.pgm");

            std::ofstream out (ofile, std::ios_base::out|std::ios_base::binary);
            out << "P5\n" << dims.first << " " << dims.second << "\n255\n";

            for (auto const v : step_counts) {
                // interpolate between 0 (black) and 255 (white) from the
                // step count which may range from 0 to max_iter.
                for (auto const c : v) {
                    auto const lerp = static_cast <std::uint8_t> (
                        255 - 255 * (static_cast <float> (c) /
                                     static_cast <float> (max_iter))
                    );
                    out << lerp;
                }
            }
        }
    }

    // 256-bit vectorized
    {
        std::vector <simd::uint32x8_t, simd::allocator <simd::uint32x8_t>>
            step_counts;
        step_counts.reserve (data_count / 8);

        {
            auto const start = clock::now ();
            for (std::size_t y = 0; y < dims.second; ++y) {
                simd::float32x8_t const im {float(1.0) - y * im_step};
                for (std::size_t x = 0; x < dims.first; x += 8) {
                    simd::float32x8_t const re {
                        float(-2.0) + (x + 0) * re_step,
                        float(-2.0) + (x + 1) * re_step,
                        float(-2.0) + (x + 2) * re_step,
                        float(-2.0) + (x + 3) * re_step,
                        float(-2.0) + (x + 4) * re_step,
                        float(-2.0) + (x + 5) * re_step,
                        float(-2.0) + (x + 6) * re_step,
                        float(-2.0) + (x + 7) * re_step
                    };
                    step_counts.emplace_back (
                        mandelbrot_vec256 (re, im, max_iter)
                    );
                }
            }
            vec256_time = clock::now () - start;

            using namespace date;
            std::cout << "256-bit vectorized time: "
                      << vec256_time
                      << std::endl;
        }

        {
            std::string const ofile = std::string ("mandelbrot-")
                                    + std::to_string (dims.first)
                                    + std::string ("x")
                                    + std::to_string (dims.second)
                                    + std::string ("-vec256.pgm");

            std::ofstream out (ofile, std::ios_base::out|std::ios_base::binary);
            out << "P5\n" << dims.first << " " << dims.second << "\n255\n";

            for (auto const v : step_counts) {
                // interpolate between 0 (black) and 255 (white) from the
                // step count which may range from 0 to max_iter.
                for (auto const c : v) {
                    auto const lerp = static_cast <std::uint8_t> (
                        255 - 255 * (static_cast <float> (c) /
                                     static_cast <float> (max_iter))
                    );
                    out << lerp;
                }
            }
        }
    }

    std::cout << "128-bit speed-up: "
              << std::setprecision (2)
              << static_cast <float> (nonvec_time.count ()) /
                 static_cast <float> (vec128_time.count ())
              << "x"
              << std::endl;
    std::cout << "256-bit speed-up: "
              << std::setprecision (2)
              << static_cast <float> (nonvec_time.count ()) /
                 static_cast <float> (vec256_time.count ())
              << "x"
              << std::endl;
}
