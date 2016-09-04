simd
====

### status
[![Build Status](https://travis-ci.org/daltonwoodard/simd.svg?branch=master)]
(https://travis-ci.org/daltonwoodard/simd)

## description

A single-header C++11 (and onwards) library of SIMD vector types and operations
for MMX, SSE, AVX, AVX2, AVX512, NEON, and other SIMD technologies implemented
using GCC and Clang vector extensions.

## dependencies

Clang or GCC compiler support for C++11 or later. Clang or GCC support for
SIMD vector extensions.

## possible future extensions

* Implement support for Intel C++ Compiler using direct x86 intrinsics.
* Implement support for ARM C++ Compiler using direct NEON intrinsics.

## known successful configurations

### Ubuntu Linux (travis-ci: 12.04 precise)

| -Olvl/Compiler | GCC 5.4 | LLVM Clang 3.6 | LLVM Clang 3.7 | LLVM Clang 3.8 |
|:--------------:|:-------:|:--------------:|:--------------:|:--------------:|
| -O0            | passing | passing        | passing        | passing        |
| -O1            | passing | passing        | passing        | passing        |
| -O2            | passing | passing        | passing        | passing        |
| -O3            | passing | passing        | passing        | passing        |

### macOS

| -Olvl/Compiler | GCC 5.4 | GCC 6.2 | Apple LLVM 7.3 |
|:--------------:|:-------:|:-------:|:--------------:|
| -O0            | passing | passing | passing        |
| -O1            | passing | passing | passing        |
| -O2            | passing | passing | passing        |
| -O3            | passing | passing | passing        |

## known failing configurations

### macOS

Test builds fail on GCC 6.1, LLVM Clang 3.7, and LLVM Clang 3.8 due to internal
compiler errors.

## info

### author

Dalton Woodard

### contact

daltonmwoodard@gmail.com

### official repository

https://github.com/daltonwoodard/simd.git

### License

All of the following information is reproduced in [COPYRIGHT](COPYRIGHT.txt).

The simd header is distributed under a dual MIT License and Apache-2.0 License.
You, the licensee, may choose either at your option. The MIT License is GPL
compatible, while the Apache-2.0 License is not, so please take this into
consideration.

The terms of each can be found in the files [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE-2.0](LICENSE-APACHE-2.0), respectively. The notices of each are
reproduced here for convenience:

---

MIT License (MIT)

Copyright (c) 2016 Dalton M. Woodard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Copyright 2016 Dalton M. Woodard

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
