# sse2neon

A C/C++ header file that converts Intel SSE intrinsics to ARN NEON intrinsics.

## Info
The SIMD instruction set of Intel, which is known as SSE is used in many applications for improved performance.  ARM also have introduced an SIMD instruction set called Neon to their processors.
Rewriting code written for SSE to work on Neon is very time consuming.  This is a header file that can automatically convert some of the SSE instricts into NEON instricts.


## Usage

- Put the *SSE2NEON.h* file in to your source code directory.

- Locate the following SSE header files included in the code: 
```    
    #include <xmmintrin.h>
    #include <emmintrin.h>
```

- Replace them with : 
```
#include "SSE2NEON.h"
```

- On Linux compile your code with the following gcc/g++ flag:   
 ```
 -mfpu=neon 
 ```
