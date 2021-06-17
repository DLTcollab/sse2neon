# sse2neon
![Github Actions](https://github.com/DLTcollab/sse2neon/workflows/Github%20Actions/badge.svg?branch=master)

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

## Compile-time Configurations

Considering the balance between correctness and performance, `sse2neon` recognizes the following compile-time configurations:
* `SSE2NEON_PRECISE_MINMAX`: Enable precise implementation of `_mm_min_ps` and `_mm_max_ps`. If you need consistent results such as NaN special cases, enable it.
* `SSE2NEON_PRECISE_DIV`: Enable precise implementation of `_mm_rcp_ps` and `_mm_div_ps` by additional Netwon-Raphson iteration for accuracy.
* `SSE2NEON_PRECISE_SQRT`: Enable precise implementation of `_mm_sqrt_ps` and `_mm_rsqrt_ps` by additional Netwon-Raphson iteration for accuracy.

The above are turned off by default, and you should define the corresponding macro(s) as `1` before including `sse2neon.h` if you need the precise implementations.

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

Check the details via [Test Suite for SSE2NEON](tests/README.md).

## Adoptions
Here is a partial list of open source projects that have adopted `sse2neon` for Arm/Aarch64 support.
* [aether-game-utils](https://github.com/johnhues/aether-game-utils) is a collection of cross platform utilities for quickly creating small game prototypes in C++.
* [Apache Impala](https://impala.apache.org/) is a lightning-fast, distributed SQL queries for petabytes of data stored in Apache Hadoop clusters.
* [Apache Kudu](https://kudu.apache.org/) completes Hadoop's storage layer to enable fast analytics on fast data.
* [ART](https://github.com/dinosaure/art) is an implementation in OCaml of [Adaptive Radix Tree](https://db.in.tum.de/~leis/papers/ART.pdf) (ART).
* [Async](https://github.com/romange/async) is a set of c++ primitives that allows efficient and rapid development in C++17 on GNU/Linux systems.
* [Blender](https://www.blender.org/) is the free and open source 3D creation suite, supporting the entirety of the 3D pipeline.
* [Boo](https://github.com/AxioDL/boo) is a cross-platform windowing and event manager similar to SDL or SFML, with additional 3D rendering functionality.
* [CARTA](https://github.com/CARTAvis/carta-backend) is a new visualization tool designed for viewing radio astronomy images in CASA, FITS, MIRIAD, and HDF5 formats (using the IDIA custom schema for HDF5).
* [Catcoon](https://github.com/i-evi/catcoon) is a [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) implementation in C.
* [dab-cmdline](https://github.com/JvanKatwijk/dab-cmdline) provides entries for the functionality to handle Digital audio broadcasting (DAB)/DAB+ through some simple calls.
* [EDGE](https://github.com/3dfxdev/EDGE) is an advanced OpenGL source port spawned from the DOOM engine, with focus on easy development and expansion for modders and end-users.
* [Embree](https://github.com/embree/embree) a collection of high-performance ray tracing kernels. Its target users are graphics application engineers who want to improve the performance of their photo-realistic rendering application by leveraging Embree's performance-optimized ray tracing kernels.
* [emp-tool](https://github.com/emp-toolkit/emp-tool) aims to provide a benchmark for secure computation and allowing other researchers to experiment and extend.
* [FoundationDB](https://www.foundationdb.org) is a distributed database designed to handle large volumes of structured data across clusters of commodity servers.
* [iqtree_arm_neon](https://github.com/joshlvmh/iqtree_arm_neon) is the Arm NEON port of [IQ-TREE](http://www.iqtree.org/), fast and effective stochastic algorithm to infer phylogenetic trees by maximum likelihood.
* [kram](https://github.com/alecazam/kram) is a wrapper to several popular encoders to and from PNG/[KTX](https://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/) files with [LDR/HDR and BC/ASTC/ETC2](https://developer.arm.com/solutions/graphics-and-gaming/developer-guides/learn-the-basics/adaptive-scalable-texture-compression/single-page).
* [libscapi](https://github.com/cryptobiu/libscapi) stands for the "Secure Computation API", providing  reliable, efficient, and highly flexible cryptographic infrastructure.
* [libmatoya](https://github.com/matoya/libmatoya) is a cross-platform application development library, providing various features such as common cryptography tasks.
* [Madronalib](https://github.com/madronalabs/madronalib) enables efficient audio DSP on SIMD processors with readable and brief C++ code.
* [minimap2](https://github.com/lh3/minimap2) is a versatile sequence alignment program that aligns DNA or mRNA sequences against a large reference database.
* [MMseqs2](https://github.com/soedinglab/MMseqs2) (Many-against-Many sequence searching) is a software suite to search and cluster huge protein and nucleotide sequence sets.
* [MRIcroGL](https://github.com/rordenlab/MRIcroGL) is a cross-platform tool for viewing NIfTI, DICOM, MGH, MHD, NRRD, AFNI format medical images.
* [N2](https://github.com/oddconcepts/n2o) is an approximate nearest neighborhoods algorithm library written in C++, providing a much faster search speed than other implementations when modeling large dataset.
* [niimath](https://github.com/rordenlab/niimath) is a general image calculator with superior performance.
* [OBS Studio](https://github.com/obsproject/obs-studio) is software designed for capturing, compositing, encoding, recording, and streaming video content, efficiently.
* [OGRE](https://github.com/OGRECave/ogre) is a scene-oriented, flexible 3D engine written in C++ designed to make it easier and more intuitive for developers to produce games and demos utilising 3D hardware.
* [OpenXRay](https://github.com/OpenXRay/xray-16) is an improved version of the X-Ray engine, used in world famous S.T.A.L.K.E.R. game series by GSC Game World.
* [parallel-n64](https://github.com/libretro/parallel-n64) is an optimized/rewritten Nintendo 64 emulator made specifically for [Libretro](https://www.libretro.com/).
* [PFFFT](https://github.com/marton78/pffft) does 1D Fast Fourier Transforms, of single precision real and complex vectors.
* [PlutoSDR Firmware](https://github.com/seanstone/plutosdr-fw) is the customized firmware for the [PlutoSDR](https://wiki.analog.com/university/tools/pluto) that can be used to introduce fundamentals of Software Defined Radio (SDR) or Radio Frequency (RF) or Communications as advanced topics in electrical engineering in a self or instructor lead setting.
* [Pygame](https://www.pygame.org) is cross-platform and designed to make it easy to write multimedia software, such as games, in Python.
* [simd_utils](https://github.com/JishinMaster/simd_utils) is a header-only library implementing common mathematical functions using SIMD intrinsics.
* [SMhasher](https://github.com/rurban/smhasher) provides comprehensive Hash function quality and speed tests.
* [Spack](https://github.com/spack/spack) is a multi-platform package manager that builds and installs multiple versions and configurations of software.
* [srsLTE](https://github.com/srsLTE/srsLTE) is an open source SDR LTE software suite.
* [Surge](https://github.com/surge-synthesizer/surge) is an open source digital synthesizer.
* [XMRig](https://github.com/xmrig/xmrig) is an open source CPU miner for [Monero](https://web.getmonero.org/) cryptocurrency.

## Related Projects
* [SIMDe](https://github.com/simd-everywhere/simde): fast and portable implementations of SIMD
  intrinsics on hardware which doesn't natively support them, such as calling SSE functions on ARM.
* [CatBoost's sse2neon](https://github.com/catboost/catboost/blob/master/library/cpp/sse/sse2neon.h)
* [ARM\_NEON\_2\_x86\_SSE](https://github.com/intel/ARM_NEON_2_x86_SSE)
* [AvxToNeon](https://github.com/kunpengcompute/AvxToNeon)
* [POWER/PowerPC support for GCC](https://github.com/gcc-mirror/gcc/blob/master/gcc/config/rs6000) contains a series of headers simplifying porting x86_64 code that
   makes explicit use of Intel intrinsics to powerpc64le (pure little-endian mode that has been introduced with the [POWER8](https://en.wikipedia.org/wiki/POWER8)).
    - implementation: [xmmintrin.h](https://github.com/gcc-mirror/gcc/blob/master/gcc/config/rs6000/xmmintrin.h), [emmintrin.h](https://github.com/gcc-mirror/gcc/blob/master/gcc/config/rs6000/emmintrin.h), [pmmintrin.h](https://github.com/gcc-mirror/gcc/blob/master/gcc/config/rs6000/pmmintrin.h), [tmmintrin.h](https://github.com/gcc-mirror/gcc/blob/master/gcc/config/rs6000/tmmintrin.h), [smmintrin.h](https://github.com/gcc-mirror/gcc/blob/master/gcc/config/rs6000/smmintrin.h)

## Reference
* [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
* [Arm Neon Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics)
* [Neon Programmer's Guide for Armv8-A](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/neon-programmers-guide-for-armv8-a)
* [NEON Programmer's Guide](https://static.docs.arm.com/den0018/a/DEN0018A_neon_programmers_guide_en.pdf)
* [qemu/target/i386/ops_sse.h](https://github.com/qemu/qemu/blob/master/target/i386/ops_sse.h): Comprehensive SSE instruction emulation in C. Ideal for semantic checks.

## Licensing

`sse2neon` is freely redistributable under the MIT License.
