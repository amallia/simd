base := .

include_dir  := $(base)/include
test_dir     := $(base)/test
test_bin_dir := $(test_dir)/bin

cxx      := $(CXX)
cxx_std  := c++11
ifeq ($(build), debug)
cxx_gflg := -g
cxx_oflg := -O0
cxx_dflg := -DDEBUG
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
endif
cxx_iflg := -I$(include_dir)
cxx_fflg :=
cxx_lflg :=
cxx_wflg := -Wall -Wextra -Wcast-qual -Wctor-dtor-privacy -Wold-style-cast \
			-Wdisabled-optimization -Wformat=2 -Winit-self -Wreturn-type \
			-Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual \
			-Wredundant-decls -Wsign-conversion -Wsign-promo -Wsign-compare \
			-Wstrict-overflow=2 -Wswitch-default -Wswitch-enum -Wundef \
			-Wshadow -Wmissing-braces -Wparentheses -Wuninitialized \
			-Wstrict-aliasing
ifeq ($(findstring clang++, $(cxx)),)
cxx_fflg += -fpermissive
cxx_wflg += -Wno-psabi
cxx_lflg += -lstdc++
else
cxx_fflg += -fsanitize=address
cxx_lflg += -lc++
endif
cxx_flgs := -std=$(cxx_std) $(cxx_gflg) $(cxx_oflg) $(cxx_fflg) $(cxx_dflg)\
	$(cxx_iflg) $(cxx_wflg) $(OPTFLAG)

executables := $(test_bin_dir)/alignment $(test_bin_dir)/arithmetic

.PHONY: test clean

test: $(executables)

clean:
	@rm -rf $(test_bin_dir)

$(test_bin_dir)/%: $(test_dir)/%.cpp
	@mkdir -p $(dir $@)
	$(cxx) $(cxx_flgs) $< -o $@ $(cxx_lflg)
