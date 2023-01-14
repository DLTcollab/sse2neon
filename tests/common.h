#ifndef SSE2NEONCOMMON_H
#define SSE2NEONCOMMON_H
#include <cstdint>
#if defined(__aarch64__) || defined(__arm__)
#include "sse2neon.h"
#elif defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h>
#include <x86intrin.h>
#include <xmmintrin.h>

// __int64 is defined in the Intrinsics Guide which maps to different datatype
// in different data model
#if !(defined(_WIN32) || defined(_WIN64) || defined(__int64))
#if (defined(__x86_64__) || defined(__i386__))
#define __int64 long long
#else
#define __int64 int64_t
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma push_macro("ALIGN_STRUCT")
#define ALIGN_STRUCT(x) __attribute__((aligned(x)))
#else
#define ALIGN_STRUCT(x) __declspec(align(x))
#endif

typedef union ALIGN_STRUCT(16) SIMDVec {
    float m128_f32[4];     // as floats - DON'T USE. Added for convenience.
    int8_t m128_i8[16];    // as signed 8-bit integers.
    int16_t m128_i16[8];   // as signed 16-bit integers.
    int32_t m128_i32[4];   // as signed 32-bit integers.
    int64_t m128_i64[2];   // as signed 64-bit integers.
    uint8_t m128_u8[16];   // as unsigned 8-bit integers.
    uint16_t m128_u16[8];  // as unsigned 16-bit integers.
    uint32_t m128_u32[4];  // as unsigned 32-bit integers.
    uint64_t m128_u64[2];  // as unsigned 64-bit integers.
} SIMDVec;

#if defined(__GNUC__) || defined(__clang__)
#pragma pop_macro("ALIGN_STRUCT")
#endif

/* Tunable testing configuration for precise testing */
/* _mm_min|max_ps|ss|pd|sd */
#ifndef SSE2NEON_PRECISE_MINMAX
#define SSE2NEON_PRECISE_MINMAX (0)
#endif
#endif

#define ASSERT_RETURN(x) \
    if (!(x))            \
        return TEST_FAIL;

namespace SSE2NEON
{
enum result_t {
    TEST_SUCCESS = 1,
    TEST_FAIL = 0,
    TEST_UNIMPL = -1,
};
extern int32_t NaN;
extern int64_t NaN64;
#define ALL_BIT_1_32 (*(float *) &NaN)
#define ALL_BIT_1_64 (*(double *) &NaN64)

template <typename T>
result_t validate128(T a, T b)
{
    const int32_t *t1 = (const int32_t *) &a;
    const int32_t *t2 = (const int32_t *) &b;

    ASSERT_RETURN(t1[0] == t2[0]);
    ASSERT_RETURN(t1[1] == t2[1]);
    ASSERT_RETURN(t1[2] == t2[2]);
    ASSERT_RETURN(t1[3] == t2[3]);
    return TEST_SUCCESS;
}
result_t validateInt64(__m128i a, int64_t i0, int64_t i1);
result_t validateInt64(__m64 a, int64_t i0);
result_t validateUInt64(__m128i a, uint64_t u0, uint64_t u1);
result_t validateUInt64(__m64 a, uint64_t u0);
result_t validateInt32(__m128i a,
                       int32_t i0,
                       int32_t i1,
                       int32_t i2,
                       int32_t i3);
result_t validateUInt32(__m128i a,
                        uint32_t u0,
                        uint32_t u1,
                        uint32_t u2,
                        uint32_t u3);
result_t validateUInt32(__m64 a, uint32_t u0, uint32_t u1);
result_t validateInt32(__m64 a, int32_t u0, int32_t u1);
result_t validateInt16(__m128i a,
                       int16_t i0,
                       int16_t i1,
                       int16_t i2,
                       int16_t i3,
                       int16_t i4,
                       int16_t i5,
                       int16_t i6,
                       int16_t i7);
result_t validateInt16(__m64 a, int16_t i0, int16_t i1, int16_t i2, int16_t i3);
result_t validateUInt16(__m128i a,
                        uint16_t u0,
                        uint16_t u1,
                        uint16_t u2,
                        uint16_t u3,
                        uint16_t u4,
                        uint16_t u5,
                        uint16_t u6,
                        uint16_t u7);
result_t validateUInt16(__m64 a,
                        uint16_t u0,
                        uint16_t u1,
                        uint16_t u2,
                        uint16_t u3);
result_t validateInt8(__m128i a,
                      int8_t i0,
                      int8_t i1,
                      int8_t i2,
                      int8_t i3,
                      int8_t i4,
                      int8_t i5,
                      int8_t i6,
                      int8_t i7,
                      int8_t i8,
                      int8_t i9,
                      int8_t i10,
                      int8_t i11,
                      int8_t i12,
                      int8_t i13,
                      int8_t i14,
                      int8_t i15);
result_t validateInt8(__m64 a,
                      int8_t i0,
                      int8_t i1,
                      int8_t i2,
                      int8_t i3,
                      int8_t i4,
                      int8_t i5,
                      int8_t i6,
                      int8_t i7);
result_t validateUInt8(__m128i a,
                       uint8_t u0,
                       uint8_t u1,
                       uint8_t u2,
                       uint8_t u3,
                       uint8_t u4,
                       uint8_t u5,
                       uint8_t u6,
                       uint8_t u7,
                       uint8_t u8,
                       uint8_t u9,
                       uint8_t u10,
                       uint8_t u11,
                       uint8_t u12,
                       uint8_t u13,
                       uint8_t u14,
                       uint8_t u15);
result_t validateUInt8(__m64 a,
                       uint8_t u0,
                       uint8_t u1,
                       uint8_t u2,
                       uint8_t u3,
                       uint8_t u4,
                       uint8_t u5,
                       uint8_t u6,
                       uint8_t u7);
result_t validateSingleFloatPair(float a, float b);
result_t validateSingleDoublePair(double a, double b);
result_t validateFloat(__m128 a, float f0, float f1, float f2, float f3);
result_t validateFloatEpsilon(__m128 a,
                              float f0,
                              float f1,
                              float f2,
                              float f3,
                              float epsilon);
result_t validateFloatError(__m128 a,
                            float f0,
                            float f1,
                            float f2,
                            float f3,
                            float err);
result_t validateDouble(__m128d a, double d0, double d1);
result_t validateFloatError(__m128d a, double d0, double d1, double err);

#define VALIDATE_INT8_M128(A, B)                                          \
    validateInt8(A, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], \
                 B[9], B[10], B[11], B[12], B[13], B[14], B[15])
#define VALIDATE_UINT8_M128(A, B)                                          \
    validateUInt8(A, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], \
                  B[9], B[10], B[11], B[12], B[13], B[14], B[15])
#define VALIDATE_INT16_M128(A, B) \
    validateInt16(A, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7])
#define VALIDATE_UINT16_M128(A, B) \
    validateUInt16(A, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7])
#define VALIDATE_INT32_M128(A, B) validateInt32(A, B[0], B[1], B[2], B[3])
#define VALIDATE_UINT32_M128(A, B) validateUInt32(A, B[0], B[1], B[2], B[3])

#define VALIDATE_INT8_M64(A, B) \
    validateInt8(A, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7])
#define VALIDATE_UINT8_M64(A, B) \
    validateUInt8(A, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7])
#define VALIDATE_INT16_M64(A, B) validateInt16(A, B[0], B[1], B[2], B[3])
#define VALIDATE_UINT16_M64(A, B) validateUInt16(A, B[0], B[1], B[2], B[3])
#define VALIDATE_INT32_M64(A, B) validateInt32(A, B[0], B[1])
#define VALIDATE_UINT32_M64(A, B) validateUInt32(A, B[0], B[1])

#define IMM_8_ITER \
    TEST(0)        \
    TEST(1)        \
    TEST(2)        \
    TEST(3)        \
    TEST(4)        \
    TEST(5)        \
    TEST(6)        \
    TEST(7)
}  // namespace SSE2NEON

#endif
