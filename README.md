# sse2neon

A C/C++ header file that converts Intel SSE intrinsics to ARN NEON intrinsics.

## Info
The SIMD instruction set of Intel, which is known as SSE is used in many
applications for improved performance.  ARM also have introduced an SIMD
instruction set called Neon to their processors.
Rewriting code written for SSE to work on Neon is very time consuming. and
this is a header file that can automatically convert some of the SSE
instricts into NEON instricts.

## Usage

- Put the file `sse2neon.h` in to your source code directory.

- Locate the following SSE header files included in the code: 
```C
#include <xmmintrin.h>
#include <emmintrin.h>
```

- Replace them with : 
```C
#include "sse2neon.h"
```

- On ARMv7-A targets, you need to append the following compiler option:
```shell
-mfpu=neon
```

## Reference
* [SIMDe](https://github.com/nemequ/simde): fast and portable implementations of SIMD
  intrinsics on hardware which doesn't natively support them, such as calling SSE functions on ARM.
* [SSE2NEON.h : A porting guide and header file to convert SSE intrinsics to their ARM NEON equivalent](https://codesuppository.blogspot.tw/2015/02/sse2neonh-porting-guide-and-header-file.html)
* [ARM_NEON_2_x86_SSE](https://github.com/intel/ARM_NEON_2_x86_SSE)

## Licensing

`sse2neon` is freely redistributable under the MIT License.
