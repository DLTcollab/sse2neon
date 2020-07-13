# sse2neon
[![Build Status](https://travis-ci.com/DLTcollab/sse2neon.svg?branch=master)](https://travis-ci.com/DLTcollab/sse2neon)

A C/C++ header file that converts Intel SSE intrinsics to Arm/Aarch64 NEON intrinsics.

## Introduction

`sse2neon` is a translator of Intel SSE (Streaming SIMD Extensions) intrinsics
to [Arm NEON](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon),
shortening the time needed to get an Arm working program that then can be used to
extract profiles and to identify hot paths in the code.
The header file `sse2neon.h` contains several of the functions provided by Intel
intrinsic headers such as `<xmmintrin.h>`, only implemented with NEON-based counterparts
to produce the exact semantics of the intrinsics.

## Mapping and Coverage

Header file | Extension |
---|---|
`<mmintrin.h>` | MMX |
`<xmmintrin.h>` | SSE |
`<emmintrin.h>` | SSE2 |
`<pmmintrin.h>` | SSE3 |
`<tmmintrin.h>` | SSSE3 |
`<smmintrin.h>` | SSE4.1 |
`<nmmintrin.h>` | SSE4.2 |
`<wmmintrin.h>` | AES  |

`sse2neon` aims to support SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2 and AES extension.

In order to deliver NEON-equivalent intrinsics for all SSE intrinsics used widely,
please be aware that some SSE intrinsics exist a direct mapping with a concrete
NEON-equivalent intrinsic. However, others lack of 1-to-1 mapping, that means the
equivalents are implemented using several NEON intrinsics.

For example, SSE intrinsic `_mm_loadu_si128` has a direct NEON mapping (`vld1q_s32`),
but SSE intrinsic `_mm_maddubs_epi16` has to be implemented with 13+ NEON instructions.

## Usage

- Put the file `sse2neon.h` in to your source code directory.

- Locate the following SSE header files included in the code:
```C
#include <xmmintrin.h>
#include <emmintrin.h>
```
  {p,t,s,n,w}mmintrin.h should be replaceable, but the coverage of these extensions might be limited though.

- Replace them with:
```C
#include "sse2neon.h"
```

- Explicitly specify platform-specific options to gcc/clang compilers.
  * On ARMv8-A targets, you should specify the following compiler option: (Remove `crypto` and/or `crc` if your architecture does not support cryptographic and/or CRC32 extensions)
  ```shell
  -march=armv8-a+fp+simd+crypto+crc
  ```
  * On ARMv7-A targets, you need to append the following compiler option:
  ```shell
  -mfpu=neon
  ```

## Run Built-in Test Suite

`sse2neon` provides a unified interface for developing test cases. These test
cases are located in `tests` directory, and the input data is specified at
runtime. Use the following commands to perform test cases:
```shell
$ make check
```

You can specify GNU toolchain for cross compilation as well.
[QEMU](https://www.qemu.org/) should be installed in advance.
```shell
$ make CROSS_COMPILE=aarch64-linux-gnu- check # ARMv8-A
```
or
```shell
$ make CROSS_COMPILE=arm-linux-gnueabihf- check # ARMv7-A
```

:warning: **Warning: The test suite is based on the little-endian architecture.**

### Add More Test Items
Once the conversion is implemented, the test can be added with the following steps:

* File `tests/impl.h`

  Add the intrinsic in `enum InstructionTest`. The naming convention should be `IT_MM_XXX`.

* File `tests/impl.cpp`

  * For the test name generation:

    Add the corresponding switch-case in `getInstructionTestString()`.
    ```c
    case IT_MM_XX:
        ret = "MM_XXX";
        break;
    ```

  * For running the test:

    Add the corresponding switch-case in `runSingleTest()`.
    ```c
    case IT_MM_XXX:
        ret = test_mm_xxx();
        break;
    ```

  * The test implementation:

    ```c
    bool test_mm_xxx()
    {
        // The C implementation
        ...

        // The Neon implementation
        ret = _mm_xxx();

        // Compare the result of two implementations and return it
        ...
    }
    ```

## Coding Convention
Use the command `$ make indent` to follow the coding convention.

## Adoptions
Here is a partial list of open source projects that have adopted `sse2neon` for Arm/Aarch64 support.
* [Apache Kudu](https://kudu.apache.org/) completes Hadoop's storage layer to enable fast analytics on fast data.
* [FoundationDB](https://www.foundationdb.org) is a distributed database designed to handle large volumes of structured data across clusters of commodity servers.
* [parallel-n64](https://github.com/libretro/parallel-n64) is an optimized/rewritten Nintendo 64 emulator made specifically for [Libretro](https://www.libretro.com/).
* [MMseqs2](https://github.com/soedinglab/MMseqs2) (Many-against-Many sequence searching) is a software suite to search and cluster huge protein and nucleotide sequence sets.
* [OpenXRay](https://github.com/OpenXRay/xray-16) is an improved version of the X-Ray engine, used in world famous S.T.A.L.K.E.R. game series by GSC Game World.
* [PAQ8PX](https://github.com/hxim/paq8px) is one of the most powerful lossless compression software that is actively under development.
* [Pygame](https://www.pygame.org) is cross-platform and designed to make it easy to write multimedia software, such as games, in Python.
* [srsLTE](https://github.com/srsLTE/srsLTE) is an open source SDR LTE software suite.
* [Surge](https://github.com/surge-synthesizer/surge) is an open source digital synthesizer.

## Related Projects
* [SIMDe](https://github.com/nemequ/simde): fast and portable implementations of SIMD
  intrinsics on hardware which doesn't natively support them, such as calling SSE functions on ARM.
* [SSE2NEON.h : A porting guide and header file to convert SSE intrinsics to their ARM NEON equivalent](https://codesuppository.blogspot.com/2015/02/sse2neonh-porting-guide-and-header-file.html)
* [CatBoost's sse2neon](https://github.com/catboost/catboost/blob/master/library/cpp/sse/sse2neon.h)
* [ARM\_NEON\_2\_x86\_SSE](https://github.com/intel/ARM_NEON_2_x86_SSE)
* [SSE2NEON - High Performance MPC on ARM](https://github.com/rons1404/biu-cybercenter-proj-sse2neon)
* [AvxToNeon](https://github.com/kunpengcompute/AvxToNeon)

## Licensing

`sse2neon` is freely redistributable under the MIT License.
