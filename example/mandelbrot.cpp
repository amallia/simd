//
// mandelbrot image generation example using simd types in the region
// [-2, 1] x [-1, -1] of the complex plane.
//
// Runs both non-vectorized and vectorized code to compare times and assert
// equality of results. Outputs a single black and white PPM image result.
//
// Two integral command line arguments define the xdim and ydim of the resulting
// image, respectively, and therefore also define the x-step and y-setp of the
// computation. The provided values are always rounded up to an even multiple of
// four for convenience in vectorization.
//

#include <algorithm>    // std::generate, std::transform
#include <chrono>       // std::chrono::*
#include <complex>      // std::complex
#include <cstddef>      // std::size_t
#include <cstdint>      // std::uint8_t
#include <fstream>      // std::ofstream
#include <limits>       // ULONG_MAX
#include <string>       // std::string, std::to_string
#include <tuple>        // std::tuple
#include <utility>      // std::make_pair, std::pair
#include <vector>       // std::vector

#include "chrono_io.hpp"
#include "simd.hpp"

using rgbcolor = std::tuple <std::uint8_t, std::uint8_t, std::uint8_t>;

static rgbcolor const white {255, 255, 255};
static rgbcolor const black {0, 0, 0};

std::ofstream & operator<< (std::ofstream & of, rgbcolor const & c)
{
    of.put (std::get <0> (c));
    of.put (std::get <1> (c));
    of.put (std::get <2> (c));
    return of;
}

int main (int argc, char ** argv)
{
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

                if (xarg == 0 || xarg == ULONG_MAX ||
                    yarg == 0 || yarg == ULONG_MAX) {
                    return std::make_pair (default_xdim, default_ydim);
                } else {
                    return std::make_pair (xarg + xarg % 4, yarg + yarg % 4);
                }
            }
        }();

    std::string const ofile_name = std::string ("mandelbrot-")
                                 + std::to_string (dims.first)
                                 + std::string ("x")
                                 + std::to_string (dims.second)
                                 + std::string (".ppm");

    std::ofstream out (ofile_name, std::ios_base::out|std::ios_base::binary);
    out << "P6\n" << dims.first << " " << dims.second << "\n255\n";
}
