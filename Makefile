base := .

include_dir   := $(base)/include
benchmark_dir := $(base)/benchmark
test_dir      := $(base)/test
example_dir   := $(base)/example
build_dir     := $(base)/build

benchmark_bin_dir := $(build_dir)/bin/benchmark
test_bin_dir      := $(build_dir)/bin/test
example_bin_dir   := $(build_dir)/bin/example

cxx      := $(CXX)
cxx_std  := c++11

ifeq ($(build), debug)
cxx_gflg := -g
cxx_oflg := -O0
cxx_dflg := -DDEBUG
cxx_vflg := --verbose
else ifeq ($(build), opt1)
cxx_gflg := -g0
cxx_oflg := -O1
cxx_dflg := -DNDEBUG
else ifeq ($(build), opt2)
cxx_gflg := -g0
cxx_oflg := -O2
cxx_dflg := -DNDEBUG
else ifeq ($(build), opt3)
cxx_gflg := -g0
cxx_oflg := -O3
cxx_dflg := -DNDEBUG
else ifeq ($(build), sanitize_debug)
cxx_gflg := -g
cxx_oflg := -O0
cxx_dflg := -DDEBUG
cxx_vflg := --verbose
cxx_fflg := -fsanitize=address -fsanitize=undefined
else ifeq ($(build), sanitize_opt1)
cxx_gflg := -g0
cxx_oflg := -O1
cxx_dflg := -DNDEBUG
cxx_fflg := -fsanitize=address -fsanitize=undefined
else ifeq ($(build), sanitize_opt2)
cxx_gflg := -g0
cxx_oflg := -O2
cxx_dflg := -DNDEBUG
cxx_fflg := -fsanitize=address -fsanitize=undefined
else ifeq ($(build), sanitize_opt3)
cxx_gflg := -g0
cxx_oflg := -O3
cxx_dflg := -DNDEBUG
cxx_fflg := -fsanitize=address -fsanitize=undefined
endif

cxx_iflg := -I$(include_dir)
cxx_lflg :=
cxx_slflg :=
cxx_wflg := -Wall -Wextra -Wmissing-braces -Wmissing-include-dirs \
	-Wsequence-point -Wswitch-default -Wswitch-bool -Wunused-local-typedefs \
	-Wunused-result -Wnarrowing -Wshadow -Wpointer-arith -Wcast-qual \
	-Wcast-align -Wwrite-strings -Wsign-conversion -Wpacked \
	-Wredundant-decls -Winline -Wvla
ifeq ($(findstring clang++, $(cxx)),)
cxx_wflg += -Wsuggest-final-types -Wsuggest-final-methods -Wsuggest-override \
			-Wtrampolines -Wunsafe-loop-optimizations -Wlogical-op \
			-Wzero-as-null-pointer-constant -Wno-psabi -Wuseless-cast
cxx_lflg += -lstdc++
else
cxx_wflg += -Wint-to-void-pointer-cast -Wshift-overflow
cxx_slflg += -stdlib=libc++
endif
cxx_flgs := $(cxx_vflg) -std=$(cxx_std) $(cxx_slflg) $(cxx_gflg) $(cxx_oflg) \
	$(cxx_fflg) $(cxx_dflg) $(cxx_iflg) $(cxx_wflg) $(OPTFLAG)

benchmark_executables := $(benchmark_bin_dir)/binary_operations \
	$(benchmark_bin_dir)/unary_operations
test_executables := $(test_bin_dir)/alignment $(test_bin_dir)/arithmetic \
	$(test_bin_dir)/io_equality $(test_bin_dir)/transforms
example_executables := $(example_bin_dir)/mandelbrot

.PHONY: benchmark test example clean

benchmark: $(benchmark_executables)

test: $(test_executables)

example: $(example_executables)

clean:
	@rm -rf $(build_dir)

$(benchmark_bin_dir)/%: $(benchmark_dir)/%.cpp
	@mkdir -p $(dir $@)
	$(cxx) $(cxx_flgs) $< -o $@ $(cxx_lflg)

$(test_bin_dir)/%: $(test_dir)/%.cpp
	@mkdir -p $(dir $@)
	$(cxx) $(cxx_flgs) $< -o $@ $(cxx_lflg)

$(example_bin_dir)/%: $(example_dir)/%.cpp
	@mkdir -p $(dir $@)
	$(cxx) $(cxx_flgs) $< -o $@ $(cxx_lflg)
