#include "impl.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include "binding.h"

// This program a set of unit tests to ensure that each SSE call provide the
// output we expect.  If this fires an assert, then something didn't match up.

#if defined(__aarch64__) || defined(__arm__)
#include "sse2neon.h"
#elif defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h>
#include <xmmintrin.h>

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

#endif  // __aarch64__ || __arm__ || __x86_64__ || __i386__

namespace SSE2NEON
{
// hex representation of an IEEE NAN
const uint32_t inan = 0xffffffff;

static inline float getNAN(void)
{
    const float *fn = (const float *) &inan;
    return *fn;
}

static inline bool isNAN(float a)
{
    const uint32_t *ia = (const uint32_t *) &a;
    return (*ia) == inan ? true : false;
}

// Do a round operation that produces results the same as SSE instructions
static inline float bankersRounding(float val)
{
    if (val < 0)
        return -bankersRounding(-val);

    float ret;
    int32_t truncateInteger = int32_t(val);
    int32_t roundInteger = int32_t(val + 0.5f);
    float diff1 = val - float(truncateInteger);  // Truncate value
    float diff2 = val - float(roundInteger);     // Round up value
    if (diff2 < 0)
        diff2 *= -1;  // get the positive difference from the round up value
    // If it's closest to the truncate integer; then use it
    if (diff1 < diff2) {
        ret = float(truncateInteger);
    } else if (diff2 <
               diff1) {  // if it's closest to the round-up integer; use it
        ret = float(roundInteger);
    } else {
        // If it's equidistant between rounding up and rounding down, pick the
        // one which is an even number
        if (truncateInteger &
            1) {  // If truncate is odd, then return the rounded integer
            ret = float(roundInteger);
        } else {
            // If the rounded up value is odd, use return the truncated integer
            ret = float(truncateInteger);
        }
    }
    return ret;
}

const char *SSE2NEONTest::getInstructionTestString(InstructionTest test)
{
    const char *ret = "UNKNOWN!";

    switch (test) {
    case IT_MM_SETZERO_SI128:
        ret = "MM_SETZERO_SI128";
        break;
    case IT_MM_SETZERO_PS:
        ret = "MM_SETZERO_PS";
        break;
    case IT_MM_SET1_PS:
        ret = "MM_SET1_PS";
        break;
    case IT_MM_SET_PS1:
        ret = "MM_SET_PS1";
        break;
    case IT_MM_SET_PS:
        ret = "MM_SET_PS";
        break;
    case IT_MM_SET_SS:
        ret = "MM_SET_SS";
        break;
    case IT_MM_SETR_PS:
        ret = "MM_SETR_PS";
        break;
    case IT_MM_SET_EPI8:
        ret = "MM_SET_EPI8";
        break;
    case IT_MM_SET1_EPI32:
        ret = "MM_SET1_EPI32";
        break;
    case IT_MM_SET_EPI32:
        ret = "MM_SET_EPI32";
        break;
    case IT_MM_STORE_PS:
        ret = "MM_STORE_PS";
        break;
    case IT_MM_STOREL_PI:
        ret = "MM_STOREL_PI";
        break;
    case IT_MM_STOREH_PI:
        ret = "MM_STOREH_PI";
        break;
    case IT_MM_STOREU_PS:
        ret = "MM_STOREU_PS";
        break;
    case IT_MM_STORE_SI128:
        ret = "MM_STORE_SI128";
        break;
    case IT_MM_STORE_SS:
        ret = "MM_STORE_SS";
        break;
    case IT_MM_STOREL_EPI64:
        ret = "MM_STOREL_EPI64";
        break;
    case IT_MM_LOAD1_PS:
        ret = "MM_LOAD1_PS";
        break;
    case IT_MM_LOADL_PI:
        ret = "MM_LOADL_PI";
        break;
    case IT_MM_LOADH_PI:
        ret = "MM_LOADH_PI";
        break;
    case IT_MM_LOAD_PS:
        ret = "MM_LOAD_PS";
        break;
    case IT_MM_LOADU_PS:
        ret = "MM_LOADU_PS";
        break;
    case IT_MM_LOAD_SD:
        ret = "MM_LOAD_SD";
        break;
    case IT_MM_LOAD_SS:
        ret = "MM_LOAD_SS";
        break;
    case IT_MM_CMPNEQ_PS:
        ret = "MM_CMPNEQ_PS";
        break;
    case IT_MM_ANDNOT_PS:
        ret = "MM_ANDNOT_PS";
        break;
    case IT_MM_ANDNOT_SI128:
        ret = "MM_ANDNOT_SI128";
        break;
    case IT_MM_AND_SI128:
        ret = "MM_AND_SI128";
        break;
    case IT_MM_AND_PS:
        ret = "MM_AND_PS";
        break;
    case IT_MM_OR_PS:
        ret = "MM_OR_PS";
        break;
    case IT_MM_XOR_PS:
        ret = "MM_XOR_PS";
        break;
    case IT_MM_OR_SI128:
        ret = "MM_OR_SI128";
        break;
    case IT_MM_XOR_SI128:
        ret = "MM_XOR_SI128";
        break;
    case IT_MM_MOVEMASK_PS:
        ret = "MM_MOVEMASK_PS";
        break;
    case IT_MM_SHUFFLE_EPI8:
        ret = "MM_SHUFFLE_EPI8";
        break;
    case IT_MM_SHUFFLE_EPI32_DEFAULT:
        ret = "MM_SHUFFLE_EPI32_DEFAULT";
        break;
    case IT_MM_SHUFFLE_EPI32_FUNCTION:
        ret = "MM_SHUFFLE_EPI32_FUNCTION";
        break;
    case IT_MM_SHUFFLE_EPI32_SPLAT:
        ret = "MM_SHUFFLE_EPI32_SPLAT";
        break;
    case IT_MM_SHUFFLE_EPI32_SINGLE:
        ret = "MM_SHUFFLE_EPI32_SINGLE";
        break;
    case IT_MM_SHUFFLEHI_EPI16_FUNCTION:
        ret = "MM_SHUFFLEHI_EPI16_FUNCTION";
        break;
    case IT_MM_MOVEMASK_EPI8:
        ret = "MM_MOVEMASK_EPI8";
        break;
    case IT_MM_SUB_PS:
        ret = "MM_SUB_PS";
        break;
    case IT_MM_SUB_EPI64:
        ret = "MM_SUB_EPI64";
        break;
    case IT_MM_SUB_EPI32:
        ret = "MM_SUB_EPI32";
        break;
    case IT_MM_ADD_PS:
        ret = "MM_ADD_PS";
        break;
    case IT_MM_ADD_SS:
        ret = "MM_ADD_SS";
        break;
    case IT_MM_ADD_EPI32:
        ret = "MM_ADD_EPI32";
        break;
    case IT_MM_ADD_EPI16:
        ret = "MM_ADD_EPI16";
        break;
    case IT_MM_MADD_EPI16:
        ret = "MM_MADD_EPI16";
        break;
    case IT_MM_MADDUBS_EPI16:
        ret = "MM_MADDUBS_EPI16";
        break;
    case IT_MM_MULLO_EPI16:
        ret = "MM_MULLO_EPI16";
        break;
    case IT_MM_MULLO_EPI32:
        ret = "MM_MULLO_EPI32";
        break;
    case IT_MM_MUL_EPU32:
        ret = "MM_MUL_EPU32";
        break;
    case IT_MM_MUL_PS:
        ret = "MM_MUL_PS";
        break;
    case IT_MM_DIV_PS:
        ret = "MM_DIV_PS";
        break;
    case IT_MM_DIV_SS:
        ret = "MM_DIV_SS";
        break;
    case IT_MM_RCP_PS:
        ret = "MM_RCP_PS";
        break;
    case IT_MM_SQRT_PS:
        ret = "MM_SQRT_PS";
        break;
    case IT_MM_SQRT_SS:
        ret = "MM_SQRT_SS";
        break;
    case IT_MM_RSQRT_PS:
        ret = "MM_RSQRT_PS";
        break;
    case IT_MM_MAX_PS:
        ret = "MM_MAX_PS";
        break;
    case IT_MM_MIN_PS:
        ret = "MM_MIN_PS";
        break;
    case IT_MM_MAX_SS:
        ret = "MM_MAX_SS";
        break;
    case IT_MM_MIN_SS:
        ret = "MM_MIN_SS";
        break;
    case IT_MM_MIN_EPI16:
        ret = "MM_MIN_EPI16";
        break;
    case IT_MM_MAX_EPI32:
        ret = "MM_MAX_EPI32";
        break;
    case IT_MM_MIN_EPI32:
        ret = "MM_MIN_EPI32";
        break;
    case IT_MM_MINPOS_EPU16:
        ret = "MM_MINPOS_EPU16";
        break;
    case IT_MM_MULHI_EPI16:
        ret = "MM_MULHI_EPI16";
        break;
    case IT_MM_HADD_PS:
        ret = "MM_HADD_PS";
        break;
    case IT_MM_HADD_EPI16:
        ret = "MM_HADD_EPI16";
        break;
    case IT_MM_CMPLT_PS:
        ret = "MM_CMPLT_PS";
        break;
    case IT_MM_CMPGT_PS:
        ret = "MM_CMPGT_PS";
        break;
    case IT_MM_CMPGE_PS:
        ret = "MM_CMPGE_PS";
        break;
    case IT_MM_CMPLE_PS:
        ret = "MM_CMPLE_PS";
        break;
    case IT_MM_CMPEQ_PS:
        ret = "MM_CMPEQ_PS";
        break;
    case IT_MM_CMPLT_EPI32:
        ret = "MM_CMPLT_EPI32";
        break;
    case IT_MM_CMPGT_EPI32:
        ret = "MM_CMPGT_EPI32";
        break;
    case IT_MM_CMPORD_PS:
        ret = "MM_CMPORD_PS";
        break;
    case IT_MM_COMILT_SS:
        ret = "MM_COMILT_SS";
        break;
    case IT_MM_COMIGT_SS:
        ret = "MM_COMIGT_SS";
        break;
    case IT_MM_COMILE_SS:
        ret = "MM_COMILE_SS";
        break;
    case IT_MM_COMIGE_SS:
        ret = "MM_COMIGE_SS";
        break;
    case IT_MM_COMIEQ_SS:
        ret = "MM_COMIEQ_SS";
        break;
    case IT_MM_COMINEQ_SS:
        ret = "MM_COMINEQ_SS";
        break;
    case IT_MM_CVTTPS_EPI32:
        ret = "MM_CVTTPS_EPI32";
        break;
    case IT_MM_CVTEPI32_PS:
        ret = "MM_CVTEPI32_PS";
        break;
    case IT_MM_CVTPS_EPI32:
        ret = "MM_CVTPS_EPI32";
        break;
    case IT_MM_CVTSI128_SI32:
        ret = "MM_CVTSI128_SI32";
        break;
    case IT_MM_CVTSI32_SI128:
        ret = "MM_CVTSI32_SI128";
        break;
    case IT_MM_CASTPS_SI128:
        ret = "MM_CASTPS_SI128";
        break;
    case IT_MM_CASTSI128_PS:
        ret = "MM_CASTSI128_PS";
        break;
    case IT_MM_LOAD_SI128:
        ret = "MM_LOAD_SI128";
        break;
    case IT_MM_PACKS_EPI16:
        ret = "MM_PACKS_EPI16";
        break;
    case IT_MM_PACKUS_EPI16:
        ret = "MM_PACKUS_EPI16";
        break;
    case IT_MM_PACKS_EPI32:
        ret = "MM_PACKS_EPI32";
        break;
    case IT_MM_UNPACKLO_EPI8:
        ret = "MM_UNPACKLO_EPI8";
        break;
    case IT_MM_UNPACKLO_EPI16:
        ret = "MM_UNPACKLO_EPI16";
        break;
    case IT_MM_UNPACKLO_EPI32:
        ret = "MM_UNPACKLO_EPI32";
        break;
    case IT_MM_UNPACKLO_PS:
        ret = "MM_UNPACKLO_PS";
        break;
    case IT_MM_UNPACKHI_PS:
        ret = "MM_UNPACKHI_PS";
        break;
    case IT_MM_UNPACKHI_EPI8:
        ret = "MM_UNPACKHI_EPI8";
        break;
    case IT_MM_UNPACKHI_EPI16:
        ret = "MM_UNPACKHI_EPI16";
        break;
    case IT_MM_UNPACKHI_EPI32:
        ret = "MM_UNPACKHI_EPI32";
        break;
    case IT_MM_SFENCE:
        ret = "MM_SFENCE";
        break;
    case IT_MM_STREAM_SI128:
        ret = "MM_STREAM_SI128";
        break;
    case IT_MM_CLFLUSH:
        ret = "MM_CLFLUSH";
        break;
    case IT_MM_SHUFFLE_PS:
        ret = "MM_SHUFFLE_PS";
        break;

    case IT_MM_CVTSS_F32:
        ret = "MM_CVTSS_F32";
        break;

    case IT_MM_SET1_EPI16:
        ret = "MM_SET1_EPI16";
        break;
    case IT_MM_SET_EPI16:
        ret = "MM_SET_EPI16";
        break;
    case IT_MM_SRA_EPI16:
        ret = "MM_SRA_EPI16";
        break;
    case IT_MM_SRA_EPI32:
        ret = "MM_SRA_EPI32";
        break;
    case IT_MM_SLLI_EPI16:
        ret = "MM_SLLI_EPI16";
        break;
    case IT_MM_SLL_EPI16:
        ret = "IT_MM_SLL_EPI16";
        break;
    case IT_MM_SLL_EPI32:
        ret = "IT_MM_SLL_EPI32";
        break;
    case IT_MM_SLL_EPI64:
        ret = "IT_MM_SLL_EPI64";
        break;
    case IT_MM_SRL_EPI16:
        ret = "IT_MM_SRL_EPI16";
        break;
    case IT_MM_SRL_EPI32:
        ret = "IT_MM_SRL_EPI32";
        break;
    case IT_MM_SRL_EPI64:
        ret = "IT_MM_SRL_EPI64";
        break;
    case IT_MM_SRLI_EPI16:
        ret = "MM_SRLI_EPI16";
        break;
    case IT_MM_CMPEQ_EPI16:
        ret = "MM_CMPEQ_EPI16";
        break;
    case IT_MM_CMPEQ_EPI64:
        ret = "MM_CMPEQ_EPI64";
        break;

    case IT_MM_SET1_EPI8:
        ret = "MM_SET1_EPI8";
        break;
    case IT_MM_ADDS_EPU8:
        ret = "MM_ADDS_EPU8";
        break;
    case IT_MM_SUBS_EPU8:
        ret = "MM_SUBS_EPU8";
        break;
    case IT_MM_MAX_EPU8:
        ret = "MM_MAX_EPU8";
        break;
    case IT_MM_CMPEQ_EPI8:
        ret = "MM_CMPEQ_EPI8";
        break;
    case IT_MM_ADDS_EPI16:
        ret = "MM_ADDS_EPI16";
        break;
    case IT_MM_MAX_EPI16:
        ret = "MM_MAX_EPI16";
        break;
    case IT_MM_SUBS_EPU16:
        ret = "MM_SUBS_EPU16";
        break;
    case IT_MM_CMPLT_EPI16:
        ret = "MM_CMPLT_EPI16";
        break;
    case IT_MM_CMPGT_EPI16:
        ret = "MM_CMPGT_EPI16";
        break;
    case IT_MM_LOADU_SI128:
        ret = "MM_LOADU_SI128";
        break;
    case IT_MM_STOREU_SI128:
        ret = "MM_STOREU_SI128";
        break;
    case IT_MM_ADD_EPI8:
        ret = "MM_ADD_EPI8";
        break;
    case IT_MM_CMPGT_EPI8:
        ret = "MM_CMPGT_EPI8";
        break;
    case IT_MM_CMPLT_EPI8:
        ret = "MM_CMPLT_EPI8";
        break;
    case IT_MM_SUB_EPI8:
        ret = "MM_SUB_EPI8";
        break;
    case IT_MM_SETR_EPI32:
        ret = "MM_SETR_EPI32";
        break;
    case IT_MM_MIN_EPU8:
        ret = "MM_MIN_EPU8";
        break;
    case IT_MM_TEST_ALL_ZEROS:
        ret = "MM_TEST_ALL_ZEROS";
        break;
    case IT_MM_AVG_EPU8:
        ret = "MM_AVG_EPU8";
        break;
    case IT_MM_AVG_EPU16:
        ret = "MM_AVG_EPU16";
        break;
    case IT_MM_POPCNT_U32:
        ret = "MM_POPCNT_U32";
        break;
    case IT_MM_POPCNT_U64:
        ret = "MM_POPCNT_U64";
        break;
#if !defined(__arm__) && __ARM_ARCH != 7
    case IT_MM_AESENC_SI128:
        ret = "IT_MM_AESENC_SI128";
        break;
#endif
    case IT_MM_CLMULEPI64_SI128:
        ret = "IT_MM_CLMULEPI64_SI128";
        break;
    case IT_MM_MALLOC:
        ret = "IT_MM_MALLOC";
        break;
    case IT_MM_CRC32_U8:
        ret = "IT_MM_CRC32_U8";
        break;
    case IT_MM_CRC32_U16:
        ret = "IT_MM_CRC32_U16";
        break;
    case IT_MM_CRC32_U32:
        ret = "IT_MM_CRC32_U32";
        break;
    case IT_MM_CRC32_U64:
        ret = "IT_MM_CRC32_U64";
        break;
    case IT_LAST: /* should not happend */
        break;
    }

    return ret;
}

#define ASSERT_RETURN(x) \
    if (!(x))            \
        return false;

static float ranf(void)
{
    uint32_t ir = rand() & 0x7FFF;
    return (float) ir * (1.0f / 32768.0f);
}

static float ranf(float low, float high)
{
    return ranf() * (high - low) + low;
}

bool validate128(__m128i a, __m128i b)
{
    const int32_t *t1 = (const int32_t *) &a;
    const int32_t *t2 = (const int32_t *) &b;

    ASSERT_RETURN(t1[0] == t2[0]);
    ASSERT_RETURN(t1[1] == t2[1]);
    ASSERT_RETURN(t1[2] == t2[2]);
    ASSERT_RETURN(t1[3] == t2[3]);
    return true;
}

bool validateInt64(__m128i a, int64_t i0, int64_t i1)
{
    const int64_t *t = (const int64_t *) &a;
    ASSERT_RETURN(t[0] == i0);
    ASSERT_RETURN(t[1] == i1);
    return true;
}

bool validateUInt64(__m128i a, uint64_t u0, uint64_t u1)
{
    const uint64_t *t = (const uint64_t *) &a;
    ASSERT_RETURN(t[0] == u0);
    ASSERT_RETURN(t[1] == u1);
    return true;
}

bool validateInt32(__m128i a, int32_t i0, int32_t i1, int32_t i2, int32_t i3)
{
    const int32_t *t = (const int32_t *) &a;
    ASSERT_RETURN(t[0] == i0);
    ASSERT_RETURN(t[1] == i1);
    ASSERT_RETURN(t[2] == i2);
    ASSERT_RETURN(t[3] == i3);
    return true;
}

bool validateInt16(__m128i a,
                   int16_t i0,
                   int16_t i1,
                   int16_t i2,
                   int16_t i3,
                   int16_t i4,
                   int16_t i5,
                   int16_t i6,
                   int16_t i7)
{
    const int16_t *t = (const int16_t *) &a;
    ASSERT_RETURN(t[0] == i0);
    ASSERT_RETURN(t[1] == i1);
    ASSERT_RETURN(t[2] == i2);
    ASSERT_RETURN(t[3] == i3);
    ASSERT_RETURN(t[4] == i4);
    ASSERT_RETURN(t[5] == i5);
    ASSERT_RETURN(t[6] == i6);
    ASSERT_RETURN(t[7] == i7);
    return true;
}

bool validateUInt16(__m128i a,
                    uint16_t u0,
                    uint16_t u1,
                    uint16_t u2,
                    uint16_t u3,
                    uint16_t u4,
                    uint16_t u5,
                    uint16_t u6,
                    uint16_t u7)
{
    const uint16_t *t = (const uint16_t *) &a;
    ASSERT_RETURN(t[0] == u0);
    ASSERT_RETURN(t[1] == u1);
    ASSERT_RETURN(t[2] == u2);
    ASSERT_RETURN(t[3] == u3);
    ASSERT_RETURN(t[4] == u4);
    ASSERT_RETURN(t[5] == u5);
    ASSERT_RETURN(t[6] == u6);
    ASSERT_RETURN(t[7] == u7);
    return true;
}

bool validateInt8(__m128i a,
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
                  int8_t i15)
{
    const int8_t *t = (const int8_t *) &a;
    ASSERT_RETURN(t[0] == i0);
    ASSERT_RETURN(t[1] == i1);
    ASSERT_RETURN(t[2] == i2);
    ASSERT_RETURN(t[3] == i3);
    ASSERT_RETURN(t[4] == i4);
    ASSERT_RETURN(t[5] == i5);
    ASSERT_RETURN(t[6] == i6);
    ASSERT_RETURN(t[7] == i7);
    ASSERT_RETURN(t[8] == i8);
    ASSERT_RETURN(t[9] == i9);
    ASSERT_RETURN(t[10] == i10);
    ASSERT_RETURN(t[11] == i11);
    ASSERT_RETURN(t[12] == i12);
    ASSERT_RETURN(t[13] == i13);
    ASSERT_RETURN(t[14] == i14);
    ASSERT_RETURN(t[15] == i15);
    return true;
}

bool validateUInt8(__m128i a,
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
                   uint8_t u15)
{
    const uint8_t *t = (const uint8_t *) &a;
    ASSERT_RETURN(t[0] == u0);
    ASSERT_RETURN(t[1] == u1);
    ASSERT_RETURN(t[2] == u2);
    ASSERT_RETURN(t[3] == u3);
    ASSERT_RETURN(t[4] == u4);
    ASSERT_RETURN(t[5] == u5);
    ASSERT_RETURN(t[6] == u6);
    ASSERT_RETURN(t[7] == u7);
    ASSERT_RETURN(t[8] == u8);
    ASSERT_RETURN(t[9] == u9);
    ASSERT_RETURN(t[10] == u10);
    ASSERT_RETURN(t[11] == u11);
    ASSERT_RETURN(t[12] == u12);
    ASSERT_RETURN(t[13] == u13);
    ASSERT_RETURN(t[14] == u14);
    ASSERT_RETURN(t[15] == u15);
    return true;
}

bool validateSingleFloatPair(float a, float b)
{
    const uint32_t *ua = (const uint32_t *) &a;
    const uint32_t *ub = (const uint32_t *) &b;
    // We do an integer (binary) compare rather than a
    // floating point compare to take nands and infinities
    // into account as well.
    return (*ua) == (*ub);
}

bool validateSingleDoublePair(double a, double b)
{
    const uint64_t *ua = (const uint64_t *) &a;
    const uint64_t *ub = (const uint64_t *) &b;
    // We do an integer (binary) compare rather than a
    // floating point compare to take nands and infinities
    // into account as well.
    return (*ua) == (*ub);
}

bool validateFloat(__m128 a, float f0, float f1, float f2, float f3)
{
    const float *t = (const float *) &a;
    ASSERT_RETURN(validateSingleFloatPair(t[0], f0));
    ASSERT_RETURN(validateSingleFloatPair(t[1], f1));
    ASSERT_RETURN(validateSingleFloatPair(t[2], f2));
    ASSERT_RETURN(validateSingleFloatPair(t[3], f3));
    return true;
}

bool validateFloatEpsilon(__m128 a,
                          float f0,
                          float f1,
                          float f2,
                          float f3,
                          float epsilon)
{
    const float *t = (const float *) &a;
    float df0 = fabsf(t[0] - f0);
    float df1 = fabsf(t[1] - f1);
    float df2 = fabsf(t[2] - f2);
    float df3 = fabsf(t[3] - f3);
    ASSERT_RETURN(df0 < epsilon);
    ASSERT_RETURN(df1 < epsilon);
    ASSERT_RETURN(df2 < epsilon);
    ASSERT_RETURN(df3 < epsilon);
    return true;
}

bool validateDouble(__m128d a, double d0, double d1)
{
    const double *t = (const double *) &a;
    ASSERT_RETURN(validateSingleDoublePair(t[0], d0));
    ASSERT_RETURN(validateSingleDoublePair(t[1], d1));
    return true;
}

bool test_mm_setzero_si128(void)
{
    __m128i a = _mm_setzero_si128();
    return validateInt32(a, 0, 0, 0, 0);
}

bool test_mm_setzero_ps(void)
{
    __m128 a = _mm_setzero_ps();
    return validateFloat(a, 0, 0, 0, 0);
}

bool test_mm_set1_ps(float w)
{
    __m128 a = _mm_set1_ps(w);
    return validateFloat(a, w, w, w, w);
}

bool test_mm_set_ps(float x, float y, float z, float w)
{
    __m128 a = _mm_set_ps(x, y, z, w);
    return validateFloat(a, w, z, y, x);
}

bool test_mm_set_ss(float a)
{
    __m128 c = _mm_set_ss(a);
    return validateFloat(c, a, 0, 0, 0);
}

bool test_mm_set_epi8(const int8_t *_a)
{
    int8_t d0 = _a[0];
    int8_t d1 = _a[1];
    int8_t d2 = _a[2];
    int8_t d3 = _a[3];
    int8_t d4 = _a[4];
    int8_t d5 = _a[5];
    int8_t d6 = _a[6];
    int8_t d7 = _a[7];
    int8_t d8 = _a[8];
    int8_t d9 = _a[9];
    int8_t d10 = _a[10];
    int8_t d11 = _a[11];
    int8_t d12 = _a[12];
    int8_t d13 = _a[13];
    int8_t d14 = _a[14];
    int8_t d15 = _a[15];

    __m128i c = _mm_set_epi8(d15, d14, d13, d12, d11, d10, d9, d8, d7, d6, d5,
                             d4, d3, d2, d1, d0);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_set1_epi32(int32_t i)
{
    __m128i a = _mm_set1_epi32(i);
    return validateInt32(a, i, i, i, i);
}

bool testret_mm_set_epi32(int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    return validateInt32(a, w, z, y, x);
}

__m128i test_mm_set_epi32(int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    validateInt32(a, w, z, y, x);
    return a;
}

bool test_mm_store_ps(float *p, float x, float y, float z, float w)
{
    __m128 a = _mm_set_ps(x, y, z, w);
    _mm_store_ps(p, a);
    ASSERT_RETURN(p[0] == w);
    ASSERT_RETURN(p[1] == z);
    ASSERT_RETURN(p[2] == y);
    ASSERT_RETURN(p[3] == x);
    return true;
}

bool test_mm_store_ps(int32_t *p, int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    _mm_store_ps((float *) p, *(const __m128 *) &a);
    ASSERT_RETURN(p[0] == w);
    ASSERT_RETURN(p[1] == z);
    ASSERT_RETURN(p[2] == y);
    ASSERT_RETURN(p[3] == x);
    return true;
}

bool test_mm_storel_pi(const float *p)
{
    __m128 a = _mm_load_ps(p);

    float d[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    __m64 *b = (__m64 *) d;

    _mm_storel_pi(b, a);

    ASSERT_RETURN(d[0] == p[0]);
    ASSERT_RETURN(d[1] == p[1]);
    ASSERT_RETURN(d[2] == 3.0f);
    ASSERT_RETURN(d[3] == 4.0f);
    return true;
}

bool test_mm_storeh_pi(const float *p)
{
    __m128 a = _mm_load_ps(p);

    float d[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    __m64 *b = (__m64 *) d;

    _mm_storeh_pi(b, a);

    ASSERT_RETURN(d[0] == p[2]);
    ASSERT_RETURN(d[1] == p[3]);
    ASSERT_RETURN(d[2] == 3.0f);
    ASSERT_RETURN(d[3] == 4.0f);
    return true;
}

bool test_mm_load1_ps(const float *p)
{
    __m128 a = _mm_load1_ps(p);
    return validateFloat(a, p[0], p[0], p[0], p[0]);
}

bool test_mm_loadl_pi(const float *p1, const float *p2)
{
    __m128 a = _mm_load_ps(p1);
    const __m64 *b = (const __m64 *) p2;
    __m128 c = _mm_loadl_pi(a, b);

    return validateFloat(c, p2[0], p2[1], p1[2], p1[3]);
}

bool test_mm_loadh_pi(const float *p1, const float *p2)
{
    __m128 a = _mm_load_ps(p1);
    const __m64 *b = (const __m64 *) p2;
    __m128 c = _mm_loadh_pi(a, b);

    return validateFloat(c, p1[0], p1[1], p2[0], p2[1]);
}

__m128 test_mm_load_ps(const float *p)
{
    __m128 a = _mm_load_ps(p);
    validateFloat(a, p[0], p[1], p[2], p[3]);
    return a;
}

__m128i test_mm_load_ps(const int32_t *p)
{
    __m128 a = _mm_load_ps((const float *) p);
    __m128i ia = *(const __m128i *) &a;
    validateInt32(ia, p[0], p[1], p[2], p[3]);
    return ia;
}

bool test_mm_load_sd(const double *p)
{
    __m128d a = _mm_load_sd(p);
    return validateDouble(a, p[0], 0);
}

// r0 := ~a0 & b0
// r1 := ~a1 & b1
// r2 := ~a2 & b2
// r3 := ~a3 & b3
bool test_mm_andnot_ps(const float *_a, const float *_b)
{
    bool r = false;

    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    __m128 c = _mm_andnot_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ~ia[0] & ib[0];
    uint32_t r1 = ~ia[1] & ib[1];
    uint32_t r2 = ~ia[2] & ib[2];
    uint32_t r3 = ~ia[3] & ib[3];
    __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
    r = validateInt32(*(const __m128i *) &c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

bool test_mm_and_ps(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    __m128 c = _mm_and_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ia[0] & ib[0];
    uint32_t r1 = ia[1] & ib[1];
    uint32_t r2 = ia[2] & ib[2];
    uint32_t r3 = ia[3] & ib[3];
    __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
    bool r = validateInt32(*(const __m128i *) &c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

bool test_mm_or_ps(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    __m128 c = _mm_or_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ia[0] | ib[0];
    uint32_t r1 = ia[1] | ib[1];
    uint32_t r2 = ia[2] | ib[2];
    uint32_t r3 = ia[3] | ib[3];
    __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
    bool r = validateInt32(*(const __m128i *) &c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

bool test_mm_andnot_si128(const int32_t *_a, const int32_t *_b)
{
    bool r = true;
    __m128i a = test_mm_load_ps(_a);
    __m128i b = test_mm_load_ps(_b);
    __m128 fc = _mm_andnot_ps(*(const __m128 *) &a, *(const __m128 *) &b);
    __m128i c = *(const __m128i *) &fc;
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ~ia[0] & ib[0];
    uint32_t r1 = ~ia[1] & ib[1];
    uint32_t r2 = ~ia[2] & ib[2];
    uint32_t r3 = ~ia[3] & ib[3];
    __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
    r = validateInt32(c, r0, r1, r2, r3);
    if (r) {
        validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

bool test_mm_and_si128(const int32_t *_a, const int32_t *_b)
{
    __m128i a = test_mm_load_ps(_a);
    __m128i b = test_mm_load_ps(_b);
    __m128 fc = _mm_and_ps(*(const __m128 *) &a, *(const __m128 *) &b);
    __m128i c = *(const __m128i *) &fc;
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ia[0] & ib[0];
    uint32_t r1 = ia[1] & ib[1];
    uint32_t r2 = ia[2] & ib[2];
    uint32_t r3 = ia[3] & ib[3];
    __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
    bool r = validateInt32(c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

bool test_mm_or_si128(const int32_t *_a, const int32_t *_b)
{
    __m128i a = test_mm_load_ps(_a);
    __m128i b = test_mm_load_ps(_b);
    __m128 fc = _mm_or_ps(*(const __m128 *) &a, *(const __m128 *) &b);
    __m128i c = *(const __m128i *) &fc;
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ia[0] | ib[0];
    uint32_t r1 = ia[1] | ib[1];
    uint32_t r2 = ia[2] | ib[2];
    uint32_t r3 = ia[3] | ib[3];
    __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
    bool r = validateInt32(c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

bool test_mm_movemask_ps(const float *p)
{
    int ret = 0;

    const uint32_t *ip = (const uint32_t *) p;
    if (ip[0] & 0x80000000) {
        ret |= 1;
    }
    if (ip[1] & 0x80000000) {
        ret |= 2;
    }
    if (ip[2] & 0x80000000) {
        ret |= 4;
    }
    if (ip[3] & 0x80000000) {
        ret |= 8;
    }
    __m128 a = test_mm_load_ps(p);
    int val = _mm_movemask_ps(a);
    return val == ret ? true : false;
}

// Note, NEON does not have a general purpose shuffled command like SSE.
// When invoking this method, there is special code for a number of the most
// common shuffle permutations
bool test_mm_shuffle_ps(const float *_a, const float *_b)
{
    bool isValid = true;
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    // Test many permutations of the shuffle operation, including all
    // permutations which have an optimized/customized implementation
    __m128 ret;
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 1, 2, 3));
    if (!validateFloat(ret, _a[3], _a[2], _b[1], _b[0])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 1, 0));
    if (!validateFloat(ret, _a[0], _a[1], _b[2], _b[3])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 1, 1));
    if (!validateFloat(ret, _a[1], _a[1], _b[0], _b[0])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 0, 2));
    if (!validateFloat(ret, _a[2], _a[0], _b[1], _b[3])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 0, 3, 2));
    if (!validateFloat(ret, _a[2], _a[3], _b[0], _b[1])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 3, 0, 1));
    if (!validateFloat(ret, _a[1], _a[0], _b[3], _b[2])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 2, 2));
    if (!validateFloat(ret, _a[2], _a[2], _b[0], _b[0])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 2, 0, 0));
    if (!validateFloat(ret, _a[0], _a[0], _b[2], _b[2])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 0, 2));
    if (!validateFloat(ret, _a[2], _a[0], _b[2], _b[3])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 1, 3, 3));
    if (!validateFloat(ret, _a[3], _a[3], _b[1], _b[1])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 1, 0));
    if (!validateFloat(ret, _a[0], _a[1], _b[0], _b[2])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 0, 1));
    if (!validateFloat(ret, _a[1], _a[0], _b[0], _b[2])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 3, 2));
    if (!validateFloat(ret, _a[2], _a[3], _b[0], _b[2])) {
        isValid = false;
    }

    return isValid;
}

bool test_mm_movemask_epi8(const int32_t *_a)
{
    __m128i a = test_mm_load_ps(_a);

    const uint8_t *ip = (const uint8_t *) _a;
    int ret = 0;
    uint32_t mask = 1;
    for (uint32_t i = 0; i < 16; i++) {
        if (ip[i] & 0x80) {
            ret |= mask;
        }
        mask = mask << 1;
    }
    int test = _mm_movemask_epi8(a);
    ASSERT_RETURN(test == ret);
    return true;
}

bool test_mm_sub_ps(const float *_a, const float *_b)
{
    float dx = _a[0] - _b[0];
    float dy = _a[1] - _b[1];
    float dz = _a[2] - _b[2];
    float dw = _a[3] - _b[3];

    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    __m128 c = _mm_sub_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

bool test_mm_sub_epi32(const int32_t *_a, const int32_t *_b)
{
    int32_t dx = _a[0] - _b[0];
    int32_t dy = _a[1] - _b[1];
    int32_t dz = _a[2] - _b[2];
    int32_t dw = _a[3] - _b[3];

    __m128i a = test_mm_load_ps(_a);
    __m128i b = test_mm_load_ps(_b);
    __m128i c = _mm_sub_epi32(a, b);
    return validateInt32(c, dx, dy, dz, dw);
}

bool test_mm_sub_epi64(const int64_t *_a, const int64_t *_b)
{
    int64_t d0 = _a[0] - _b[0];
    int64_t d1 = _a[1] - _b[1];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sub_epi64(a, b);
    return validateInt64(c, d0, d1);
}

bool test_mm_add_ps(const float *_a, const float *_b)
{
    float dx = _a[0] + _b[0];
    float dy = _a[1] + _b[1];
    float dz = _a[2] + _b[2];
    float dw = _a[3] + _b[3];

    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    __m128 c = _mm_add_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

bool test_mm_add_epi32(const int32_t *_a, const int32_t *_b)
{
    int32_t dx = _a[0] + _b[0];
    int32_t dy = _a[1] + _b[1];
    int32_t dz = _a[2] + _b[2];
    int32_t dw = _a[3] + _b[3];

    __m128i a = test_mm_load_ps(_a);
    __m128i b = test_mm_load_ps(_b);
    __m128i c = _mm_add_epi32(a, b);
    return validateInt32(c, dx, dy, dz, dw);
}

bool test_mm_mullo_epi16(const int16_t *_a, const int16_t *_b)
{
    int16_t d0 = _a[0] * _b[0];
    int16_t d1 = _a[1] * _b[1];
    int16_t d2 = _a[2] * _b[2];
    int16_t d3 = _a[3] * _b[3];
    int16_t d4 = _a[4] * _b[4];
    int16_t d5 = _a[5] * _b[5];
    int16_t d6 = _a[6] * _b[6];
    int16_t d7 = _a[7] * _b[7];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_mullo_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_mul_epu32(const uint32_t *_a, const uint32_t *_b)
{
    uint64_t dx = (uint64_t)(_a[0]) * (uint64_t)(_b[0]);
    uint64_t dy = (uint64_t)(_a[2]) * (uint64_t)(_b[2]);

    __m128i a = _mm_loadu_si128((const __m128i *) _a);
    __m128i b = _mm_loadu_si128((const __m128i *) _b);
    __m128i r = _mm_mul_epu32(a, b);
    return validateUInt64(r, dx, dy);
}

bool test_mm_madd_epi16(const int16_t *_a, const int16_t *_b)
{
    int32_t d0 = (int32_t) _a[0] * _b[0];
    int32_t d1 = (int32_t) _a[1] * _b[1];
    int32_t d2 = (int32_t) _a[2] * _b[2];
    int32_t d3 = (int32_t) _a[3] * _b[3];
    int32_t d4 = (int32_t) _a[4] * _b[4];
    int32_t d5 = (int32_t) _a[5] * _b[5];
    int32_t d6 = (int32_t) _a[6] * _b[6];
    int32_t d7 = (int32_t) _a[7] * _b[7];

    int32_t e0 = d0 + d1;
    int32_t e1 = d2 + d3;
    int32_t e2 = d4 + d5;
    int32_t e3 = d6 + d7;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_madd_epi16(a, b);
    return validateInt32(c, e0, e1, e2, e3);
}

static inline int16_t saturate_16(int32_t a)
{
    int32_t max = (1 << 15) - 1;
    int32_t min = -(1 << 15);
    if (a > max)
        return max;
    if (a < min)
        return min;
    return a;
}

bool test_mm_maddubs_epi16(const uint8_t *_a, const int8_t *_b)
{
    int32_t d0 = (int32_t)(_a[0] * _b[0]);
    int32_t d1 = (int32_t)(_a[1] * _b[1]);
    int32_t d2 = (int32_t)(_a[2] * _b[2]);
    int32_t d3 = (int32_t)(_a[3] * _b[3]);
    int32_t d4 = (int32_t)(_a[4] * _b[4]);
    int32_t d5 = (int32_t)(_a[5] * _b[5]);
    int32_t d6 = (int32_t)(_a[6] * _b[6]);
    int32_t d7 = (int32_t)(_a[7] * _b[7]);
    int32_t d8 = (int32_t)(_a[8] * _b[8]);
    int32_t d9 = (int32_t)(_a[9] * _b[9]);
    int32_t d10 = (int32_t)(_a[10] * _b[10]);
    int32_t d11 = (int32_t)(_a[11] * _b[11]);
    int32_t d12 = (int32_t)(_a[12] * _b[12]);
    int32_t d13 = (int32_t)(_a[13] * _b[13]);
    int32_t d14 = (int32_t)(_a[14] * _b[14]);
    int32_t d15 = (int32_t)(_a[15] * _b[15]);

    int16_t e0 = saturate_16(d0 + d1);
    int16_t e1 = saturate_16(d2 + d3);
    int16_t e2 = saturate_16(d4 + d5);
    int16_t e3 = saturate_16(d6 + d7);
    int16_t e4 = saturate_16(d8 + d9);
    int16_t e5 = saturate_16(d10 + d11);
    int16_t e6 = saturate_16(d12 + d13);
    int16_t e7 = saturate_16(d14 + d15);

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_maddubs_epi16(a, b);
    return validateInt16(c, e0, e1, e2, e3, e4, e5, e6, e7);
}

bool test_mm_shuffle_epi8(const int32_t *a, const int32_t *b)
{
    const uint8_t *tbl = (const uint8_t *) a;
    const uint8_t *idx = (const uint8_t *) b;
    int32_t r[4];

    r[0] = ((idx[3] & 0x80) ? 0 : tbl[idx[3] % 16]) << 24;
    r[0] |= ((idx[2] & 0x80) ? 0 : tbl[idx[2] % 16]) << 16;
    r[0] |= ((idx[1] & 0x80) ? 0 : tbl[idx[1] % 16]) << 8;
    r[0] |= ((idx[0] & 0x80) ? 0 : tbl[idx[0] % 16]);

    r[1] = ((idx[7] & 0x80) ? 0 : tbl[idx[7] % 16]) << 24;
    r[1] |= ((idx[6] & 0x80) ? 0 : tbl[idx[6] % 16]) << 16;
    r[1] |= ((idx[5] & 0x80) ? 0 : tbl[idx[5] % 16]) << 8;
    r[1] |= ((idx[4] & 0x80) ? 0 : tbl[idx[4] % 16]);

    r[2] = ((idx[11] & 0x80) ? 0 : tbl[idx[11] % 16]) << 24;
    r[2] |= ((idx[10] & 0x80) ? 0 : tbl[idx[10] % 16]) << 16;
    r[2] |= ((idx[9] & 0x80) ? 0 : tbl[idx[9] % 16]) << 8;
    r[2] |= ((idx[8] & 0x80) ? 0 : tbl[idx[8] % 16]);

    r[3] = ((idx[15] & 0x80) ? 0 : tbl[idx[15] % 16]) << 24;
    r[3] |= ((idx[14] & 0x80) ? 0 : tbl[idx[14] % 16]) << 16;
    r[3] |= ((idx[13] & 0x80) ? 0 : tbl[idx[13] % 16]) << 8;
    r[3] |= ((idx[12] & 0x80) ? 0 : tbl[idx[12] % 16]);

    __m128i ret = _mm_shuffle_epi8(test_mm_load_ps(a), test_mm_load_ps(b));

    return validateInt32(ret, r[0], r[1], r[2], r[3]);
}

bool test_mm_mul_ps(const float *_a, const float *_b)
{
    float dx = _a[0] * _b[0];
    float dy = _a[1] * _b[1];
    float dz = _a[2] * _b[2];
    float dw = _a[3] * _b[3];

    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    __m128 c = _mm_mul_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

bool test_mm_rcp_ps(const float *_a)
{
    float dx = 1.0f / _a[0];
    float dy = 1.0f / _a[1];
    float dz = 1.0f / _a[2];
    float dw = 1.0f / _a[3];

    __m128 a = test_mm_load_ps(_a);
    __m128 c = _mm_rcp_ps(a);
    return validateFloatEpsilon(c, dx, dy, dz, dw, 300.0f);
}

bool test_mm_max_ps(const float *_a, const float *_b)
{
    float c[4];

    c[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    __m128 ret = _mm_max_ps(a, b);
    return validateFloat(ret, c[0], c[1], c[2], c[3]);
}

bool test_mm_min_ps(const float *_a, const float *_b)
{
    float c[4];

    c[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);
    __m128 ret = _mm_min_ps(a, b);
    return validateFloat(ret, c[0], c[1], c[2], c[3]);
}

bool test_mm_min_epi16(const int16_t *_a, const int16_t *_b)
{
    int16_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    int16_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
    int16_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
    int16_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];
    int16_t d4 = _a[4] < _b[4] ? _a[4] : _b[4];
    int16_t d5 = _a[5] < _b[5] ? _a[5] : _b[5];
    int16_t d6 = _a[6] < _b[6] ? _a[6] : _b[6];
    int16_t d7 = _a[7] < _b[7] ? _a[7] : _b[7];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_min_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_mulhi_epi16(const int16_t *_a, const int16_t *_b)
{
    int16_t d[8];
    for (uint32_t i = 0; i < 8; i++) {
        int32_t m = (int32_t) _a[i] * (int32_t) _b[i];
        d[i] = (int16_t)(m >> 16);
    }

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_mulhi_epi16(a, b);
    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

bool test_mm_cmplt_ps(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] < _b[0] ? -1 : 0;
    result[1] = _a[1] < _b[1] ? -1 : 0;
    result[2] = _a[2] < _b[2] ? -1 : 0;
    result[3] = _a[3] < _b[3] ? -1 : 0;

    __m128 ret = _mm_cmplt_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

bool test_mm_cmpgt_ps(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] > _b[0] ? -1 : 0;
    result[1] = _a[1] > _b[1] ? -1 : 0;
    result[2] = _a[2] > _b[2] ? -1 : 0;
    result[3] = _a[3] > _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpgt_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

bool test_mm_cmpge_ps(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] >= _b[0] ? -1 : 0;
    result[1] = _a[1] >= _b[1] ? -1 : 0;
    result[2] = _a[2] >= _b[2] ? -1 : 0;
    result[3] = _a[3] >= _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpge_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

bool test_mm_cmple_ps(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] <= _b[0] ? -1 : 0;
    result[1] = _a[1] <= _b[1] ? -1 : 0;
    result[2] = _a[2] <= _b[2] ? -1 : 0;
    result[3] = _a[3] <= _b[3] ? -1 : 0;

    __m128 ret = _mm_cmple_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

bool test_mm_cmpeq_ps(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] == _b[0] ? -1 : 0;
    result[1] = _a[1] == _b[1] ? -1 : 0;
    result[2] = _a[2] == _b[2] ? -1 : 0;
    result[3] = _a[3] == _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpeq_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

bool test_mm_cmplt_epi32(const int32_t *_a, const int32_t *_b)
{
    __m128i a = test_mm_load_ps(_a);
    __m128i b = test_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] < _b[0] ? -1 : 0;
    result[1] = _a[1] < _b[1] ? -1 : 0;
    result[2] = _a[2] < _b[2] ? -1 : 0;
    result[3] = _a[3] < _b[3] ? -1 : 0;

    __m128i iret = _mm_cmplt_epi32(a, b);
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

bool test_mm_cmpgt_epi32(const int32_t *_a, const int32_t *_b)
{
    __m128i a = test_mm_load_ps(_a);
    __m128i b = test_mm_load_ps(_b);

    int32_t result[4];

    result[0] = _a[0] > _b[0] ? -1 : 0;
    result[1] = _a[1] > _b[1] ? -1 : 0;
    result[2] = _a[2] > _b[2] ? -1 : 0;
    result[3] = _a[3] > _b[3] ? -1 : 0;

    __m128i iret = _mm_cmpgt_epi32(a, b);
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

float compord(float a, float b)
{
    float ret;

    bool isNANA = isNAN(a);
    bool isNANB = isNAN(b);
    if (!isNANA && !isNANB) {
        ret = getNAN();
    } else {
        ret = 0.0f;
    }
    return ret;
}

bool test_mm_cmpord_ps(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    float result[4];

    for (uint32_t i = 0; i < 4; i++) {
        result[i] = compord(_a[i], _b[i]);
    }

    __m128 ret = _mm_cmpord_ps(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

int32_t comilt_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isNAN(a);
    bool isNANB = isNAN(b);
    if (!isNANA && !isNANB) {
        ret = a < b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

bool test_mm_comilt_ss(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    if (isNAN(_a[0]) || isNAN(_b[0]))
        // Test disabled: GCC and Clang on x86_64 return different values.
        return true;

    int32_t result = comilt_ss(_a[0], _b[0]);

    int32_t ret = _mm_comilt_ss(a, b);

    return result == ret ? true : false;
}

int32_t comigt_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isNAN(a);
    bool isNANB = isNAN(b);
    if (!isNANA && !isNANB) {
        ret = a > b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

bool test_mm_comigt_ss(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    int32_t result = comigt_ss(_a[0], _b[0]);
    int32_t ret = _mm_comigt_ss(a, b);

    return result == ret ? true : false;
}

int32_t comile_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isNAN(a);
    bool isNANB = isNAN(b);
    if (!isNANA && !isNANB) {
        ret = a <= b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

bool test_mm_comile_ss(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    if (isNAN(_a[0]) || isNAN(_b[0]))
        // Test disabled: GCC and Clang on x86_64 return different values.
        return true;

    int32_t result = comile_ss(_a[0], _b[0]);
    int32_t ret = _mm_comile_ss(a, b);

    return result == ret ? true : false;
}

int32_t comige_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isNAN(a);
    bool isNANB = isNAN(b);
    if (!isNANA && !isNANB) {
        ret = a >= b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

bool test_mm_comige_ss(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    int32_t result = comige_ss(_a[0], _b[0]);
    int32_t ret = _mm_comige_ss(a, b);

    return result == ret ? true : false;
}

int32_t comieq_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isNAN(a);
    bool isNANB = isNAN(b);
    if (!isNANA && !isNANB) {
        ret = a == b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

bool test_mm_comieq_ss(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    if (isNAN(_a[0]) || isNAN(_b[0]))
        // Test disabled: GCC and Clang on x86_64 return different values.
        return true;

    int32_t result = comieq_ss(_a[0], _b[0]);
    int32_t ret = _mm_comieq_ss(a, b);

    return result == ret ? true : false;
}

int32_t comineq_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isNAN(a);
    bool isNANB = isNAN(b);
    if (!isNANA && !isNANB) {
        ret = a != b ? 1 : 0;
    } else {
        ret = 1;
    }
    return ret;
}

bool test_mm_comineq_ss(const float *_a, const float *_b)
{
    __m128 a = test_mm_load_ps(_a);
    __m128 b = test_mm_load_ps(_b);

    if (isNAN(_a[0]) || isNAN(_b[0]))
        // Test disabled: GCC and Clang on x86_64 return different values.
        return true;

    int32_t result = comineq_ss(_a[0], _b[0]);
    int32_t ret = _mm_comineq_ss(a, b);

    return result == ret ? true : false;
}

bool test_mm_hadd_epi16(const int16_t *_a, const int16_t *_b)
{
    int16_t d0 = _a[0] + _a[1];
    int16_t d1 = _a[2] + _a[3];
    int16_t d2 = _a[4] + _a[5];
    int16_t d3 = _a[6] + _a[7];
    int16_t d4 = _b[0] + _b[1];
    int16_t d5 = _b[2] + _b[3];
    int16_t d6 = _b[4] + _b[5];
    int16_t d7 = _b[6] + _b[7];
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_hadd_epi16(a, b);
    return validateInt16(ret, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_cvttps_epi32(const float *_a)
{
    __m128 a = test_mm_load_ps(_a);
    int32_t trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (int32_t) _a[i];
    }

    __m128i ret = _mm_cvttps_epi32(a);
    return validateInt32(ret, trun[0], trun[1], trun[2], trun[3]);
}

bool test_mm_cvtepi32_ps(const int32_t *_a)
{
    __m128i a = test_mm_load_ps(_a);
    float trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (float) _a[i];
    }

    __m128 ret = _mm_cvtepi32_ps(a);
    return validateFloat(ret, trun[0], trun[1], trun[2], trun[3]);
}

// https://msdn.microsoft.com/en-us/library/xdc42k5e%28v=vs.90%29.aspx?f=255&MSPPError=-2147217396
bool test_mm_cvtps_epi32(const float _a[4])
{
    __m128 a = test_mm_load_ps(_a);
    int32_t trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (int32_t)(bankersRounding(_a[i]));
    }

    __m128i ret = _mm_cvtps_epi32(a);
    return validateInt32(ret, trun[0], trun[1], trun[2], trun[3]);
}

bool test_mm_set1_epi16(const int16_t *_a)
{
    int16_t d0 = _a[0];

    __m128i c = _mm_set1_epi16(d0);
    return validateInt16(c, d0, d0, d0, d0, d0, d0, d0, d0);
}

bool test_mm_set_epi16(const int16_t *_a)
{
    int16_t d0 = _a[0];
    int16_t d1 = _a[1];
    int16_t d2 = _a[2];
    int16_t d3 = _a[3];
    int16_t d4 = _a[4];
    int16_t d5 = _a[5];
    int16_t d6 = _a[6];
    int16_t d7 = _a[7];

    __m128i c = _mm_set_epi16(d7, d6, d5, d4, d3, d2, d1, d0);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_sra_epi16(const int16_t *_a, const int64_t count)
{
    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sra_epi16(a, b);
    if (count > 15) {
        int16_t d0 = _a[0] < 0 ? 0xffff : 0;
        int16_t d1 = _a[1] < 0 ? 0xffff : 0;
        int16_t d2 = _a[2] < 0 ? 0xffff : 0;
        int16_t d3 = _a[3] < 0 ? 0xffff : 0;
        int16_t d4 = _a[4] < 0 ? 0xffff : 0;
        int16_t d5 = _a[5] < 0 ? 0xffff : 0;
        int16_t d6 = _a[6] < 0 ? 0xffff : 0;
        int16_t d7 = _a[7] < 0 ? 0xffff : 0;

        return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
    }

    int16_t d0 = _a[0] >> count;
    int16_t d1 = _a[1] >> count;
    int16_t d2 = _a[2] >> count;
    int16_t d3 = _a[3] >> count;
    int16_t d4 = _a[4] >> count;
    int16_t d5 = _a[5] >> count;
    int16_t d6 = _a[6] >> count;
    int16_t d7 = _a[7] >> count;

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_sra_epi32(const int32_t *_a, const int64_t count)
{
    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sra_epi32(a, b);
    if (count > 31) {
        int32_t d0 = _a[0] < 0 ? 0xffffffff : 0;
        int32_t d1 = _a[1] < 0 ? 0xffffffff : 0;
        int32_t d2 = _a[2] < 0 ? 0xffffffff : 0;
        int32_t d3 = _a[3] < 0 ? 0xffffffff : 0;

        return validateInt32(c, d0, d1, d2, d3);
    }

    int32_t d0 = _a[0] >> count;
    int32_t d1 = _a[1] >> count;
    int32_t d2 = _a[2] >> count;
    int32_t d3 = _a[3] >> count;

    return validateInt32(c, d0, d1, d2, d3);
}

bool test_mm_slli_epi16(const int16_t *_a)
{
    const int count = 3;

    int16_t d0 = _a[0] << count;
    int16_t d1 = _a[1] << count;
    int16_t d2 = _a[2] << count;
    int16_t d3 = _a[3] << count;
    int16_t d4 = _a[4] << count;
    int16_t d5 = _a[5] << count;
    int16_t d6 = _a[6] << count;
    int16_t d7 = _a[7] << count;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_slli_epi16(a, count);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_sll_epi16(const int16_t *_a, const int64_t count)
{
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sll_epi16(a, b);
    if (count < 0 || count > 15)
        return validateInt16(c, 0, 0, 0, 0, 0, 0, 0, 0);

    uint16_t d0 = _a[0] << count;
    uint16_t d1 = _a[1] << count;
    uint16_t d2 = _a[2] << count;
    uint16_t d3 = _a[3] << count;
    uint16_t d4 = _a[4] << count;
    uint16_t d5 = _a[5] << count;
    uint16_t d6 = _a[6] << count;
    uint16_t d7 = _a[7] << count;
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_sll_epi32(const int32_t *_a, const int64_t count)
{
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sll_epi32(a, b);
    if (count < 0 || count > 31)
        return validateInt32(c, 0, 0, 0, 0);

    uint32_t d0 = _a[0] << count;
    uint32_t d1 = _a[1] << count;
    uint32_t d2 = _a[2] << count;
    uint32_t d3 = _a[3] << count;
    return validateInt32(c, d0, d1, d2, d3);
}

bool test_mm_sll_epi64(const int64_t *_a, const int64_t count)
{
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sll_epi64(a, b);
    if (count < 0 || count > 63)
        return validateInt64(c, 0, 0);

    uint64_t d0 = _a[0] << count;
    uint64_t d1 = _a[1] << count;
    return validateInt64(c, d0, d1);
}

bool test_mm_srl_epi16(const int16_t *_a, const int64_t count)
{
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi16(a, b);
    if (count < 0 || count > 15)
        return validateInt16(c, 0, 0, 0, 0, 0, 0, 0, 0);

    uint16_t d0 = (uint16_t)(_a[0]) >> count;
    uint16_t d1 = (uint16_t)(_a[1]) >> count;
    uint16_t d2 = (uint16_t)(_a[2]) >> count;
    uint16_t d3 = (uint16_t)(_a[3]) >> count;
    uint16_t d4 = (uint16_t)(_a[4]) >> count;
    uint16_t d5 = (uint16_t)(_a[5]) >> count;
    uint16_t d6 = (uint16_t)(_a[6]) >> count;
    uint16_t d7 = (uint16_t)(_a[7]) >> count;
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_srl_epi32(const int32_t *_a, const int64_t count)
{
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi32(a, b);
    if (count < 0 || count > 31)
        return validateInt32(c, 0, 0, 0, 0);

    uint32_t d0 = (uint32_t)(_a[0]) >> count;
    uint32_t d1 = (uint32_t)(_a[1]) >> count;
    uint32_t d2 = (uint32_t)(_a[2]) >> count;
    uint32_t d3 = (uint32_t)(_a[3]) >> count;
    return validateInt32(c, d0, d1, d2, d3);
}

bool test_mm_srl_epi64(const int64_t *_a, const int64_t count)
{
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi64(a, b);
    if (count < 0 || count > 63)
        return validateInt64(c, 0, 0);

    uint64_t d0 = (uint64_t)(_a[0]) >> count;
    uint64_t d1 = (uint64_t)(_a[1]) >> count;
    return validateInt64(c, d0, d1);
}

bool test_mm_srli_epi16(const int16_t *_a)
{
    const int count = 3;

    int16_t d0 = (uint16_t)(_a[0]) >> count;
    int16_t d1 = (uint16_t)(_a[1]) >> count;
    int16_t d2 = (uint16_t)(_a[2]) >> count;
    int16_t d3 = (uint16_t)(_a[3]) >> count;
    int16_t d4 = (uint16_t)(_a[4]) >> count;
    int16_t d5 = (uint16_t)(_a[5]) >> count;
    int16_t d6 = (uint16_t)(_a[6]) >> count;
    int16_t d7 = (uint16_t)(_a[7]) >> count;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_srli_epi16(a, count);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_cmpeq_epi16(const int16_t *_a, const int16_t *_b)
{
    int16_t d0 = (_a[0] == _b[0]) ? 0xffff : 0x0;
    int16_t d1 = (_a[1] == _b[1]) ? 0xffff : 0x0;
    int16_t d2 = (_a[2] == _b[2]) ? 0xffff : 0x0;
    int16_t d3 = (_a[3] == _b[3]) ? 0xffff : 0x0;
    ;
    int16_t d4 = (_a[4] == _b[4]) ? 0xffff : 0x0;
    ;
    int16_t d5 = (_a[5] == _b[5]) ? 0xffff : 0x0;
    int16_t d6 = (_a[6] == _b[6]) ? 0xffff : 0x0;
    ;
    int16_t d7 = (_a[7] == _b[7]) ? 0xffff : 0x0;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpeq_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_cmpeq_epi64(const int64_t *_a, const int64_t *_b)
{
    int64_t d0 = (_a[0] == _b[0]) ? 0xffffffffffffffff : 0x0;
    int64_t d1 = (_a[1] == _b[1]) ? 0xffffffffffffffff : 0x0;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpeq_epi64(a, b);
    return validateInt64(c, d0, d1);
}

bool test_mm_set1_epi8(const int8_t *_a)
{
    int8_t d0 = _a[0];
    __m128i c = _mm_set1_epi8(d0);
    return validateInt8(c, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0,
                        d0, d0, d0);
}

bool test_mm_adds_epu8(const int8_t *_a, const int8_t *_b)
{
    uint8_t d0 = (uint8_t) _a[0] + (uint8_t) _b[0];
    if (d0 < (uint8_t) _a[0])
        d0 = 255;
    uint8_t d1 = (uint8_t) _a[1] + (uint8_t) _b[1];
    if (d1 < (uint8_t) _a[1])
        d1 = 255;
    uint8_t d2 = (uint8_t) _a[2] + (uint8_t) _b[2];
    if (d2 < (uint8_t) _a[2])
        d2 = 255;
    uint8_t d3 = (uint8_t) _a[3] + (uint8_t) _b[3];
    if (d3 < (uint8_t) _a[3])
        d3 = 255;
    uint8_t d4 = (uint8_t) _a[4] + (uint8_t) _b[4];
    if (d4 < (uint8_t) _a[4])
        d4 = 255;
    uint8_t d5 = (uint8_t) _a[5] + (uint8_t) _b[5];
    if (d5 < (uint8_t) _a[5])
        d5 = 255;
    uint8_t d6 = (uint8_t) _a[6] + (uint8_t) _b[6];
    if (d6 < (uint8_t) _a[6])
        d6 = 255;
    uint8_t d7 = (uint8_t) _a[7] + (uint8_t) _b[7];
    if (d7 < (uint8_t) _a[7])
        d7 = 255;
    uint8_t d8 = (uint8_t) _a[8] + (uint8_t) _b[8];
    if (d8 < (uint8_t) _a[8])
        d8 = 255;
    uint8_t d9 = (uint8_t) _a[9] + (uint8_t) _b[9];
    if (d9 < (uint8_t) _a[9])
        d9 = 255;
    uint8_t d10 = (uint8_t) _a[10] + (uint8_t) _b[10];
    if (d10 < (uint8_t) _a[10])
        d10 = 255;
    uint8_t d11 = (uint8_t) _a[11] + (uint8_t) _b[11];
    if (d11 < (uint8_t) _a[11])
        d11 = 255;
    uint8_t d12 = (uint8_t) _a[12] + (uint8_t) _b[12];
    if (d12 < (uint8_t) _a[12])
        d12 = 255;
    uint8_t d13 = (uint8_t) _a[13] + (uint8_t) _b[13];
    if (d13 < (uint8_t) _a[13])
        d13 = 255;
    uint8_t d14 = (uint8_t) _a[14] + (uint8_t) _b[14];
    if (d14 < (uint8_t) _a[14])
        d14 = 255;
    uint8_t d15 = (uint8_t) _a[15] + (uint8_t) _b[15];
    if (d15 < (uint8_t) _a[15])
        d15 = 255;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_adds_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_subs_epu8(const int8_t *_a, const int8_t *_b)
{
    uint8_t d0 = (uint8_t) _a[0] - (uint8_t) _b[0];
    if (d0 > (uint8_t) _a[0])
        d0 = 0;
    uint8_t d1 = (uint8_t) _a[1] - (uint8_t) _b[1];
    if (d1 > (uint8_t) _a[1])
        d1 = 0;
    uint8_t d2 = (uint8_t) _a[2] - (uint8_t) _b[2];
    if (d2 > (uint8_t) _a[2])
        d2 = 0;
    uint8_t d3 = (uint8_t) _a[3] - (uint8_t) _b[3];
    if (d3 > (uint8_t) _a[3])
        d3 = 0;
    uint8_t d4 = (uint8_t) _a[4] - (uint8_t) _b[4];
    if (d4 > (uint8_t) _a[4])
        d4 = 0;
    uint8_t d5 = (uint8_t) _a[5] - (uint8_t) _b[5];
    if (d5 > (uint8_t) _a[5])
        d5 = 0;
    uint8_t d6 = (uint8_t) _a[6] - (uint8_t) _b[6];
    if (d6 > (uint8_t) _a[6])
        d6 = 0;
    uint8_t d7 = (uint8_t) _a[7] - (uint8_t) _b[7];
    if (d7 > (uint8_t) _a[7])
        d7 = 0;
    uint8_t d8 = (uint8_t) _a[8] - (uint8_t) _b[8];
    if (d8 > (uint8_t) _a[8])
        d8 = 0;
    uint8_t d9 = (uint8_t) _a[9] - (uint8_t) _b[9];
    if (d9 > (uint8_t) _a[9])
        d9 = 0;
    uint8_t d10 = (uint8_t) _a[10] - (uint8_t) _b[10];
    if (d10 > (uint8_t) _a[10])
        d10 = 0;
    uint8_t d11 = (uint8_t) _a[11] - (uint8_t) _b[11];
    if (d11 > (uint8_t) _a[11])
        d11 = 0;
    uint8_t d12 = (uint8_t) _a[12] - (uint8_t) _b[12];
    if (d12 > (uint8_t) _a[12])
        d12 = 0;
    uint8_t d13 = (uint8_t) _a[13] - (uint8_t) _b[13];
    if (d13 > (uint8_t) _a[13])
        d13 = 0;
    uint8_t d14 = (uint8_t) _a[14] - (uint8_t) _b[14];
    if (d14 > (uint8_t) _a[14])
        d14 = 0;
    uint8_t d15 = (uint8_t) _a[15] - (uint8_t) _b[15];
    if (d15 > (uint8_t) _a[15])
        d15 = 0;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_subs_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_max_epu8(const int8_t *_a, const int8_t *_b)
{
    uint8_t d0 = ((uint8_t) _a[0] > (uint8_t) _b[0]) ? ((uint8_t) _a[0])
                                                     : ((uint8_t) _b[0]);
    uint8_t d1 = ((uint8_t) _a[1] > (uint8_t) _b[1]) ? ((uint8_t) _a[1])
                                                     : ((uint8_t) _b[1]);
    uint8_t d2 = ((uint8_t) _a[2] > (uint8_t) _b[2]) ? ((uint8_t) _a[2])
                                                     : ((uint8_t) _b[2]);
    uint8_t d3 = ((uint8_t) _a[3] > (uint8_t) _b[3]) ? ((uint8_t) _a[3])
                                                     : ((uint8_t) _b[3]);
    uint8_t d4 = ((uint8_t) _a[4] > (uint8_t) _b[4]) ? ((uint8_t) _a[4])
                                                     : ((uint8_t) _b[4]);
    uint8_t d5 = ((uint8_t) _a[5] > (uint8_t) _b[5]) ? ((uint8_t) _a[5])
                                                     : ((uint8_t) _b[5]);
    uint8_t d6 = ((uint8_t) _a[6] > (uint8_t) _b[6]) ? ((uint8_t) _a[6])
                                                     : ((uint8_t) _b[6]);
    uint8_t d7 = ((uint8_t) _a[7] > (uint8_t) _b[7]) ? ((uint8_t) _a[7])
                                                     : ((uint8_t) _b[7]);
    uint8_t d8 = ((uint8_t) _a[8] > (uint8_t) _b[8]) ? ((uint8_t) _a[8])
                                                     : ((uint8_t) _b[8]);
    uint8_t d9 = ((uint8_t) _a[9] > (uint8_t) _b[9]) ? ((uint8_t) _a[9])
                                                     : ((uint8_t) _b[9]);
    uint8_t d10 = ((uint8_t) _a[10] > (uint8_t) _b[10]) ? ((uint8_t) _a[10])
                                                        : ((uint8_t) _b[10]);
    uint8_t d11 = ((uint8_t) _a[11] > (uint8_t) _b[11]) ? ((uint8_t) _a[11])
                                                        : ((uint8_t) _b[11]);
    uint8_t d12 = ((uint8_t) _a[12] > (uint8_t) _b[12]) ? ((uint8_t) _a[12])
                                                        : ((uint8_t) _b[12]);
    uint8_t d13 = ((uint8_t) _a[13] > (uint8_t) _b[13]) ? ((uint8_t) _a[13])
                                                        : ((uint8_t) _b[13]);
    uint8_t d14 = ((uint8_t) _a[14] > (uint8_t) _b[14]) ? ((uint8_t) _a[14])
                                                        : ((uint8_t) _b[14]);
    uint8_t d15 = ((uint8_t) _a[15] > (uint8_t) _b[15]) ? ((uint8_t) _a[15])
                                                        : ((uint8_t) _b[15]);

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_max_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_cmpeq_epi8(const int8_t *_a, const int8_t *_b)
{
    int8_t d0 = (_a[0] == _b[0]) ? 0xff : 0x00;
    int8_t d1 = (_a[1] == _b[1]) ? 0xff : 0x00;
    int8_t d2 = (_a[2] == _b[2]) ? 0xff : 0x00;
    int8_t d3 = (_a[3] == _b[3]) ? 0xff : 0x00;
    int8_t d4 = (_a[4] == _b[4]) ? 0xff : 0x00;
    int8_t d5 = (_a[5] == _b[5]) ? 0xff : 0x00;
    int8_t d6 = (_a[6] == _b[6]) ? 0xff : 0x00;
    int8_t d7 = (_a[7] == _b[7]) ? 0xff : 0x00;
    int8_t d8 = (_a[8] == _b[8]) ? 0xff : 0x00;
    int8_t d9 = (_a[9] == _b[9]) ? 0xff : 0x00;
    int8_t d10 = (_a[10] == _b[10]) ? 0xff : 0x00;
    int8_t d11 = (_a[11] == _b[11]) ? 0xff : 0x00;
    int8_t d12 = (_a[12] == _b[12]) ? 0xff : 0x00;
    int8_t d13 = (_a[13] == _b[13]) ? 0xff : 0x00;
    int8_t d14 = (_a[14] == _b[14]) ? 0xff : 0x00;
    int8_t d15 = (_a[15] == _b[15]) ? 0xff : 0x00;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpeq_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_adds_epi16(const int16_t *_a, const int16_t *_b)
{
    int32_t d0 = (int32_t) _a[0] + (int32_t) _b[0];
    if (d0 > 32767)
        d0 = 32767;
    if (d0 < -32768)
        d0 = -32768;
    int32_t d1 = (int32_t) _a[1] + (int32_t) _b[1];
    if (d1 > 32767)
        d1 = 32767;
    if (d1 < -32768)
        d1 = -32768;
    int32_t d2 = (int32_t) _a[2] + (int32_t) _b[2];
    if (d2 > 32767)
        d2 = 32767;
    if (d2 < -32768)
        d2 = -32768;
    int32_t d3 = (int32_t) _a[3] + (int32_t) _b[3];
    if (d3 > 32767)
        d3 = 32767;
    if (d3 < -32768)
        d3 = -32768;
    int32_t d4 = (int32_t) _a[4] + (int32_t) _b[4];
    if (d4 > 32767)
        d4 = 32767;
    if (d4 < -32768)
        d4 = -32768;
    int32_t d5 = (int32_t) _a[5] + (int32_t) _b[5];
    if (d5 > 32767)
        d5 = 32767;
    if (d5 < -32768)
        d5 = -32768;
    int32_t d6 = (int32_t) _a[6] + (int32_t) _b[6];
    if (d6 > 32767)
        d6 = 32767;
    if (d6 < -32768)
        d6 = -32768;
    int32_t d7 = (int32_t) _a[7] + (int32_t) _b[7];
    if (d7 > 32767)
        d7 = 32767;
    if (d7 < -32768)
        d7 = -32768;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);

    __m128i c = _mm_adds_epi16(a, b);
    return validateInt16(c, (int16_t) d0, (int16_t) d1, (int16_t) d2,
                         (int16_t) d3, (int16_t) d4, (int16_t) d5, (int16_t) d6,
                         (int16_t) d7);
}

bool test_mm_max_epi16(const int16_t *_a, const int16_t *_b)
{
    int16_t d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    int16_t d1 = _a[1] > _b[1] ? _a[1] : _b[1];
    int16_t d2 = _a[2] > _b[2] ? _a[2] : _b[2];
    int16_t d3 = _a[3] > _b[3] ? _a[3] : _b[3];
    int16_t d4 = _a[4] > _b[4] ? _a[4] : _b[4];
    int16_t d5 = _a[5] > _b[5] ? _a[5] : _b[5];
    int16_t d6 = _a[6] > _b[6] ? _a[6] : _b[6];
    int16_t d7 = _a[7] > _b[7] ? _a[7] : _b[7];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);

    __m128i c = _mm_max_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_subs_epu16(const int16_t *_a, const int16_t *_b)
{
    uint16_t d0 = (uint16_t) _a[0] - (uint16_t) _b[0];
    if (d0 > (uint16_t) _a[0])
        d0 = 0;
    uint16_t d1 = (uint16_t) _a[1] - (uint16_t) _b[1];
    if (d1 > (uint16_t) _a[1])
        d1 = 0;
    uint16_t d2 = (uint16_t) _a[2] - (uint16_t) _b[2];
    if (d2 > (uint16_t) _a[2])
        d2 = 0;
    uint16_t d3 = (uint16_t) _a[3] - (uint16_t) _b[3];
    if (d3 > (uint16_t) _a[3])
        d3 = 0;
    uint16_t d4 = (uint16_t) _a[4] - (uint16_t) _b[4];
    if (d4 > (uint16_t) _a[4])
        d4 = 0;
    uint16_t d5 = (uint16_t) _a[5] - (uint16_t) _b[5];
    if (d5 > (uint16_t) _a[5])
        d5 = 0;
    uint16_t d6 = (uint16_t) _a[6] - (uint16_t) _b[6];
    if (d6 > (uint16_t) _a[6])
        d6 = 0;
    uint16_t d7 = (uint16_t) _a[7] - (uint16_t) _b[7];
    if (d7 > (uint16_t) _a[7])
        d7 = 0;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);

    __m128i c = _mm_subs_epu16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_cmplt_epi16(const int16_t *_a, const int16_t *_b)
{
    uint16_t d0 = _a[0] < _b[0] ? 0xffff : 0;
    uint16_t d1 = _a[1] < _b[1] ? 0xffff : 0;
    uint16_t d2 = _a[2] < _b[2] ? 0xffff : 0;
    uint16_t d3 = _a[3] < _b[3] ? 0xffff : 0;
    uint16_t d4 = _a[4] < _b[4] ? 0xffff : 0;
    uint16_t d5 = _a[5] < _b[5] ? 0xffff : 0;
    uint16_t d6 = _a[6] < _b[6] ? 0xffff : 0;
    uint16_t d7 = _a[7] < _b[7] ? 0xffff : 0;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmplt_epi16(a, b);

    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_cmpgt_epi16(const int16_t *_a, const int16_t *_b)
{
    uint16_t d0 = _a[0] > _b[0] ? 0xffff : 0;
    uint16_t d1 = _a[1] > _b[1] ? 0xffff : 0;
    uint16_t d2 = _a[2] > _b[2] ? 0xffff : 0;
    uint16_t d3 = _a[3] > _b[3] ? 0xffff : 0;
    uint16_t d4 = _a[4] > _b[4] ? 0xffff : 0;
    uint16_t d5 = _a[5] > _b[5] ? 0xffff : 0;
    uint16_t d6 = _a[6] > _b[6] ? 0xffff : 0;
    uint16_t d7 = _a[7] > _b[7] ? 0xffff : 0;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpgt_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_loadu_si128(const int32_t *_a)
{
    __m128i c = _mm_loadu_si128((const __m128i *) _a);
    return validateInt32(c, _a[0], _a[1], _a[2], _a[3]);
}

bool test_mm_storeu_si128(const int32_t *_a)
{
    __m128i b;
    __m128i a = _mm_loadu_si128((const __m128i *) _a);
    _mm_storeu_si128(&b, a);
    int32_t *_b = (int32_t *) &b;
    return validateInt32(a, _b[0], _b[1], _b[2], _b[3]);
    return 1;
}

bool test_mm_add_epi8(const int8_t *_a, const int8_t *_b)
{
    int8_t d0 = _a[0] + _b[0];
    int8_t d1 = _a[1] + _b[1];
    int8_t d2 = _a[2] + _b[2];
    int8_t d3 = _a[3] + _b[3];
    int8_t d4 = _a[4] + _b[4];
    int8_t d5 = _a[5] + _b[5];
    int8_t d6 = _a[6] + _b[6];
    int8_t d7 = _a[7] + _b[7];
    int8_t d8 = _a[8] + _b[8];
    int8_t d9 = _a[9] + _b[9];
    int8_t d10 = _a[10] + _b[10];
    int8_t d11 = _a[11] + _b[11];
    int8_t d12 = _a[12] + _b[12];
    int8_t d13 = _a[13] + _b[13];
    int8_t d14 = _a[14] + _b[14];
    int8_t d15 = _a[15] + _b[15];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_add_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_cmpgt_epi8(const int8_t *_a, const int8_t *_b)
{
    int8_t d0 = (_a[0] > _b[0]) ? 0xff : 0x00;
    int8_t d1 = (_a[1] > _b[1]) ? 0xff : 0x00;
    int8_t d2 = (_a[2] > _b[2]) ? 0xff : 0x00;
    int8_t d3 = (_a[3] > _b[3]) ? 0xff : 0x00;
    int8_t d4 = (_a[4] > _b[4]) ? 0xff : 0x00;
    int8_t d5 = (_a[5] > _b[5]) ? 0xff : 0x00;
    int8_t d6 = (_a[6] > _b[6]) ? 0xff : 0x00;
    int8_t d7 = (_a[7] > _b[7]) ? 0xff : 0x00;
    int8_t d8 = (_a[8] > _b[8]) ? 0xff : 0x00;
    int8_t d9 = (_a[9] > _b[9]) ? 0xff : 0x00;
    int8_t d10 = (_a[10] > _b[10]) ? 0xff : 0x00;
    int8_t d11 = (_a[11] > _b[11]) ? 0xff : 0x00;
    int8_t d12 = (_a[12] > _b[12]) ? 0xff : 0x00;
    int8_t d13 = (_a[13] > _b[13]) ? 0xff : 0x00;
    int8_t d14 = (_a[14] > _b[14]) ? 0xff : 0x00;
    int8_t d15 = (_a[15] > _b[15]) ? 0xff : 0x00;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpgt_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_cmplt_epi8(const int8_t *_a, const int8_t *_b)
{
    int8_t d0 = (_a[0] < _b[0]) ? 0xff : 0x00;
    int8_t d1 = (_a[1] < _b[1]) ? 0xff : 0x00;
    int8_t d2 = (_a[2] < _b[2]) ? 0xff : 0x00;
    int8_t d3 = (_a[3] < _b[3]) ? 0xff : 0x00;
    int8_t d4 = (_a[4] < _b[4]) ? 0xff : 0x00;
    int8_t d5 = (_a[5] < _b[5]) ? 0xff : 0x00;
    int8_t d6 = (_a[6] < _b[6]) ? 0xff : 0x00;
    int8_t d7 = (_a[7] < _b[7]) ? 0xff : 0x00;
    int8_t d8 = (_a[8] < _b[8]) ? 0xff : 0x00;
    int8_t d9 = (_a[9] < _b[9]) ? 0xff : 0x00;
    int8_t d10 = (_a[10] < _b[10]) ? 0xff : 0x00;
    int8_t d11 = (_a[11] < _b[11]) ? 0xff : 0x00;
    int8_t d12 = (_a[12] < _b[12]) ? 0xff : 0x00;
    int8_t d13 = (_a[13] < _b[13]) ? 0xff : 0x00;
    int8_t d14 = (_a[14] < _b[14]) ? 0xff : 0x00;
    int8_t d15 = (_a[15] < _b[15]) ? 0xff : 0x00;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmplt_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_sub_epi8(const int8_t *_a, const int8_t *_b)
{
    int8_t d0 = _a[0] - _b[0];
    int8_t d1 = _a[1] - _b[1];
    int8_t d2 = _a[2] - _b[2];
    int8_t d3 = _a[3] - _b[3];
    int8_t d4 = _a[4] - _b[4];
    int8_t d5 = _a[5] - _b[5];
    int8_t d6 = _a[6] - _b[6];
    int8_t d7 = _a[7] - _b[7];
    int8_t d8 = _a[8] - _b[8];
    int8_t d9 = _a[9] - _b[9];
    int8_t d10 = _a[10] - _b[10];
    int8_t d11 = _a[11] - _b[11];
    int8_t d12 = _a[12] - _b[12];
    int8_t d13 = _a[13] - _b[13];
    int8_t d14 = _a[14] - _b[14];
    int8_t d15 = _a[15] - _b[15];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sub_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_setr_epi32(const int32_t *_a)
{
    __m128i c = _mm_setr_epi32(_a[0], _a[1], _a[2], _a[3]);
    return validateInt32(c, _a[0], _a[1], _a[2], _a[3]);
}

bool test_mm_min_epu8(const int8_t *_a, const int8_t *_b)
{
    uint8_t d0 =
        ((uint8_t) _a[0] < (uint8_t) _b[0]) ? (uint8_t) _a[0] : (uint8_t) _b[0];
    uint8_t d1 =
        ((uint8_t) _a[1] < (uint8_t) _b[1]) ? (uint8_t) _a[1] : (uint8_t) _b[1];
    uint8_t d2 =
        ((uint8_t) _a[2] < (uint8_t) _b[2]) ? (uint8_t) _a[2] : (uint8_t) _b[2];
    uint8_t d3 =
        ((uint8_t) _a[3] < (uint8_t) _b[3]) ? (uint8_t) _a[3] : (uint8_t) _b[3];
    uint8_t d4 =
        ((uint8_t) _a[4] < (uint8_t) _b[4]) ? (uint8_t) _a[4] : (uint8_t) _b[4];
    uint8_t d5 =
        ((uint8_t) _a[5] < (uint8_t) _b[5]) ? (uint8_t) _a[5] : (uint8_t) _b[5];
    uint8_t d6 =
        ((uint8_t) _a[6] < (uint8_t) _b[6]) ? (uint8_t) _a[6] : (uint8_t) _b[6];
    uint8_t d7 =
        ((uint8_t) _a[7] < (uint8_t) _b[7]) ? (uint8_t) _a[7] : (uint8_t) _b[7];
    uint8_t d8 =
        ((uint8_t) _a[8] < (uint8_t) _b[8]) ? (uint8_t) _a[8] : (uint8_t) _b[8];
    uint8_t d9 =
        ((uint8_t) _a[9] < (uint8_t) _b[9]) ? (uint8_t) _a[9] : (uint8_t) _b[9];
    uint8_t d10 = ((uint8_t) _a[10] < (uint8_t) _b[10]) ? (uint8_t) _a[10]
                                                        : (uint8_t) _b[10];
    uint8_t d11 = ((uint8_t) _a[11] < (uint8_t) _b[11]) ? (uint8_t) _a[11]
                                                        : (uint8_t) _b[11];
    uint8_t d12 = ((uint8_t) _a[12] < (uint8_t) _b[12]) ? (uint8_t) _a[12]
                                                        : (uint8_t) _b[12];
    uint8_t d13 = ((uint8_t) _a[13] < (uint8_t) _b[13]) ? (uint8_t) _a[13]
                                                        : (uint8_t) _b[13];
    uint8_t d14 = ((uint8_t) _a[14] < (uint8_t) _b[14]) ? (uint8_t) _a[14]
                                                        : (uint8_t) _b[14];
    uint8_t d15 = ((uint8_t) _a[15] < (uint8_t) _b[15]) ? (uint8_t) _a[15]
                                                        : (uint8_t) _b[15];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_min_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

bool test_mm_minpos_epu16(const int16_t *_a)
{
    uint16_t index = 0, min = (uint16_t) _a[0];
    for (int i = 0; i < 8; i++) {
        if ((uint16_t) _a[i] < min) {
            index = (uint16_t) i;
            min = (uint16_t) _a[i];
        }
    }
    uint16_t d0 = min;
    uint16_t d1 = index;
    uint16_t d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0;

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_minpos_epu16(a);
    return validateUInt16(ret, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_test_all_zeros(const int32_t *_a, const int32_t *_mask)
{
    __m128i a = test_mm_load_ps(_a);
    __m128i mask = test_mm_load_ps(_mask);

    int32_t d0 = _a[0] & _mask[0];
    int32_t d1 = _a[1] & _mask[1];
    int32_t d2 = _a[2] & _mask[2];
    int32_t d3 = _a[3] & _mask[3];
    int32_t result = ((d0 | d1 | d2 | d3) == 0) ? 1 : 0;

    int32_t ret = _mm_test_all_zeros(a, mask);

    return result == ret ? true : false;
}

bool test_mm_avg_epu8(const int8_t *_a, const int8_t *_b)
{
    uint8_t d0 = ((uint8_t) _a[0] + (uint8_t) _b[0] + 1) >> 1;
    uint8_t d1 = ((uint8_t) _a[1] + (uint8_t) _b[1] + 1) >> 1;
    uint8_t d2 = ((uint8_t) _a[2] + (uint8_t) _b[2] + 1) >> 1;
    uint8_t d3 = ((uint8_t) _a[3] + (uint8_t) _b[3] + 1) >> 1;
    uint8_t d4 = ((uint8_t) _a[4] + (uint8_t) _b[4] + 1) >> 1;
    uint8_t d5 = ((uint8_t) _a[5] + (uint8_t) _b[5] + 1) >> 1;
    uint8_t d6 = ((uint8_t) _a[6] + (uint8_t) _b[6] + 1) >> 1;
    uint8_t d7 = ((uint8_t) _a[7] + (uint8_t) _b[7] + 1) >> 1;
    uint8_t d8 = ((uint8_t) _a[8] + (uint8_t) _b[8] + 1) >> 1;
    uint8_t d9 = ((uint8_t) _a[9] + (uint8_t) _b[9] + 1) >> 1;
    uint8_t d10 = ((uint8_t) _a[10] + (uint8_t) _b[10] + 1) >> 1;
    uint8_t d11 = ((uint8_t) _a[11] + (uint8_t) _b[11] + 1) >> 1;
    uint8_t d12 = ((uint8_t) _a[12] + (uint8_t) _b[12] + 1) >> 1;
    uint8_t d13 = ((uint8_t) _a[13] + (uint8_t) _b[13] + 1) >> 1;
    uint8_t d14 = ((uint8_t) _a[14] + (uint8_t) _b[14] + 1) >> 1;
    uint8_t d15 = ((uint8_t) _a[15] + (uint8_t) _b[15] + 1) >> 1;
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_avg_epu8(a, b);
    return validateUInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                         d12, d13, d14, d15);
}

bool test_mm_avg_epu16(const int16_t *_a, const int16_t *_b)
{
    uint16_t d0 = ((uint16_t) _a[0] + (uint16_t) _b[0] + 1) >> 1;
    uint16_t d1 = ((uint16_t) _a[1] + (uint16_t) _b[1] + 1) >> 1;
    uint16_t d2 = ((uint16_t) _a[2] + (uint16_t) _b[2] + 1) >> 1;
    uint16_t d3 = ((uint16_t) _a[3] + (uint16_t) _b[3] + 1) >> 1;
    uint16_t d4 = ((uint16_t) _a[4] + (uint16_t) _b[4] + 1) >> 1;
    uint16_t d5 = ((uint16_t) _a[5] + (uint16_t) _b[5] + 1) >> 1;
    uint16_t d6 = ((uint16_t) _a[6] + (uint16_t) _b[6] + 1) >> 1;
    uint16_t d7 = ((uint16_t) _a[7] + (uint16_t) _b[7] + 1) >> 1;
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_avg_epu16(a, b);
    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

bool test_mm_popcnt_u32(const uint32_t *a)
{
    ASSERT_RETURN(__builtin_popcount(a[0]) == _mm_popcnt_u32(a[0]));
    return true;
}

bool test_mm_popcnt_u64(const uint64_t *a)
{
    ASSERT_RETURN(__builtin_popcountll(a[0]) == _mm_popcnt_u64(a[0]));
    return true;
}

static inline uint64_t MUL(uint32_t a, uint32_t b)
{
    return (uint64_t) a * (uint64_t) b;
}

// From BearSSL. Performs a 32-bit->64-bit carryless/polynomial
// long multiply.
//
// This implementation was chosen because it is reasonably fast
// without a lookup table or branching.
//
// This does it by splitting up the bits in a way that they
// would not carry, then combine them together with xor (a
// carryless add).
//
// https://www.bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/hash/ghash_ctmul.c;h=3623202;hb=5f045c7#l164
static uint64_t clmul_32(uint32_t x, uint32_t y)
{
    uint32_t x0, x1, x2, x3;
    uint32_t y0, y1, y2, y3;
    uint64_t z0, z1, z2, z3;

    x0 = x & (uint32_t) 0x11111111;
    x1 = x & (uint32_t) 0x22222222;
    x2 = x & (uint32_t) 0x44444444;
    x3 = x & (uint32_t) 0x88888888;
    y0 = y & (uint32_t) 0x11111111;
    y1 = y & (uint32_t) 0x22222222;
    y2 = y & (uint32_t) 0x44444444;
    y3 = y & (uint32_t) 0x88888888;
    z0 = MUL(x0, y0) ^ MUL(x1, y3) ^ MUL(x2, y2) ^ MUL(x3, y1);
    z1 = MUL(x0, y1) ^ MUL(x1, y0) ^ MUL(x2, y3) ^ MUL(x3, y2);
    z2 = MUL(x0, y2) ^ MUL(x1, y1) ^ MUL(x2, y0) ^ MUL(x3, y3);
    z3 = MUL(x0, y3) ^ MUL(x1, y2) ^ MUL(x2, y1) ^ MUL(x3, y0);
    z0 &= (uint64_t) 0x1111111111111111;
    z1 &= (uint64_t) 0x2222222222222222;
    z2 &= (uint64_t) 0x4444444444444444;
    z3 &= (uint64_t) 0x8888888888888888;
    return z0 | z1 | z2 | z3;
}

// Performs a 64x64->128-bit carryless/polynomial long
// multiply, using the above routine to calculate the
// subproducts needed for the full-size multiply.
//
// This uses the Karatsuba algorithm.
//
// Normally, the Karatsuba algorithm isn't beneficial
// until very large numbers due to carry tracking and
// multiplication being relatively cheap.
//
// However, we have no carries and multiplication is
// definitely not cheap, so the Karatsuba algorithm is
// a low cost and easy optimization.
//
// https://en.m.wikipedia.org/wiki/Karatsuba_algorithm
//
// Note that addition and subtraction are both
// performed with xor, since all operations are
// carryless.
//
// The comments represent the actual mathematical
// operations being performed (instead of the bitwise
// operations) and to reflect the linked Wikipedia article.
static std::pair<uint64_t, uint64_t> clmul_64(uint64_t x, uint64_t y)
{
    // B = 2
    // m = 32
    // x = (x1 * B^m) + x0
    uint32_t x0 = x & 0xffffffff;
    uint32_t x1 = x >> 32;
    // y = (y1 * B^m) + y0
    uint32_t y0 = y & 0xffffffff;
    uint32_t y1 = y >> 32;

    // z0 = x0 * y0
    uint64_t z0 = clmul_32(x0, y0);
    // z2 = x1 * y1
    uint64_t z2 = clmul_32(x1, y1);
    // z1 = (x0 + x1) * (y0 + y1) - z0 - z2
    uint64_t z1 = clmul_32(x0 ^ x1, y0 ^ y1) ^ z0 ^ z2;

    // xy = z0 + (z1 * B^m) + (z2 * B^2m)
    // note: z1 is split between the low and high halves
    uint64_t xy0 = z0 ^ (z1 << 32);
    uint64_t xy1 = z2 ^ (z1 >> 32);

    return std::make_pair(xy0, xy1);
}

bool test_mm_clmulepi64_si128(const uint64_t *_a, const uint64_t *_b)
{
    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    auto result = clmul_64(_a[0], _b[0]);
    if (!validateUInt64(_mm_clmulepi64_si128(a, b, 0x00), result.first,
                        result.second))
        return false;
    result = clmul_64(_a[1], _b[0]);
    if (!validateUInt64(_mm_clmulepi64_si128(a, b, 0x01), result.first,
                        result.second))
        return false;
    result = clmul_64(_a[0], _b[1]);
    if (!validateUInt64(_mm_clmulepi64_si128(a, b, 0x10), result.first,
                        result.second))
        return false;
    result = clmul_64(_a[1], _b[1]);
    if (!validateUInt64(_mm_clmulepi64_si128(a, b, 0x11), result.first,
                        result.second))
        return false;
    return true;
}

#define XT(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))
inline __m128i aesenc_128_reference(__m128i a, __m128i b)
{
    static const uint8_t sbox[256] = {
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b,
        0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
        0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
        0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
        0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
        0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed,
        0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f,
        0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec,
        0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
        0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
        0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
        0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f,
        0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
        0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11,
        0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
        0xb0, 0x54, 0xbb, 0x16};
    uint8_t i, t, u, v[4][4];
    for (i = 0; i < 16; ++i) {
        v[((i / 4) + 4 - (i % 4)) % 4][i % 4] =
            sbox[((SIMDVec *) &a)->m128_u8[i]];
    }
    for (i = 0; i < 4; ++i) {
        t = v[i][0];
        u = v[i][0] ^ v[i][1] ^ v[i][2] ^ v[i][3];
        v[i][0] ^= u ^ XT(v[i][0] ^ v[i][1]);
        v[i][1] ^= u ^ XT(v[i][1] ^ v[i][2]);
        v[i][2] ^= u ^ XT(v[i][2] ^ v[i][3]);
        v[i][3] ^= u ^ XT(v[i][3] ^ t);
    }
    for (i = 0; i < 16; ++i) {
        ((SIMDVec *) &a)->m128_u8[i] =
            v[i / 4][i % 4] ^ ((SIMDVec *) &b)->m128_u8[i];
    }
    return a;
}

#if !defined(__arm__) && __ARM_ARCH != 7
bool test_mm_aesenc_si128(const int32_t *a, const int32_t *b)
{
    __m128i data = _mm_loadu_si128((const __m128i *) a);
    __m128i rk = _mm_loadu_si128((const __m128i *) b);

    __m128i resultReference = aesenc_128_reference(data, rk);
    __m128i resultIntrinsic = _mm_aesenc_si128(data, rk);

    return validate128(resultReference, resultIntrinsic);
}
#endif

bool test_mm_malloc(const size_t *a, const size_t *b)
{
    size_t size = *a % (1024 * 16) + 1;
    size_t align = 2 << (*b % 5);

    void *p = _mm_malloc(size, align);
    if (!p)
        return false;
    bool res = ((uintptr_t) p % align) == 0;
    _mm_free(p);
    return res;
}

uint32_t canonical_crc32_u8(uint32_t crc, uint8_t v)
{
    crc ^= v;
    for (int bit = 0; bit < 8; bit++) {
        if (crc & 1)
            crc = (crc >> 1) ^ uint32_t(0x82f63b78);
        else
            crc = (crc >> 1);
    }
    return crc;
}

uint32_t canonical_crc32_u16(uint32_t crc, uint16_t v)
{
    crc = canonical_crc32_u8(crc, v & 0xff);
    crc = canonical_crc32_u8(crc, (v >> 8) & 0xff);
    return crc;
}

uint32_t canonical_crc32_u32(uint32_t crc, uint32_t v)
{
    crc = canonical_crc32_u16(crc, v & 0xffff);
    crc = canonical_crc32_u16(crc, (v >> 16) & 0xffff);
    return crc;
}

uint64_t canonical_crc32_u64(uint64_t crc, uint64_t v)
{
    crc = canonical_crc32_u32((uint32_t)(crc), v & 0xffffffff);
    crc = canonical_crc32_u32((uint32_t)(crc), (v >> 32) & 0xffffffff);
    return crc;
}

bool test_mm_crc32_u8(uint32_t crc, uint8_t v)
{
    uint32_t result = _mm_crc32_u8(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u8(crc, v));
    return true;
}

bool test_mm_crc32_u16(uint32_t crc, uint16_t v)
{
    uint32_t result = _mm_crc32_u16(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u16(crc, v));
    return true;
}

bool test_mm_crc32_u32(uint32_t crc, uint32_t v)
{
    uint32_t result = _mm_crc32_u32(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u32(crc, v));
    return true;
}

bool test_mm_crc32_u64(uint64_t crc, uint64_t v)
{
    uint64_t result = _mm_crc32_u64(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u64(crc, v));
    return true;
}

// Try 10,000 random floating point values for each test we run
#define MAX_TEST_VALUE 10000

class SSE2NEONTestImpl : public SSE2NEONTest
{
public:
    SSE2NEONTestImpl(void)
    {
        mTestFloatPointer1 = (float *) platformAlignedAlloc(sizeof(__m128));
        mTestFloatPointer2 = (float *) platformAlignedAlloc(sizeof(__m128));
        mTestIntPointer1 = (int32_t *) platformAlignedAlloc(sizeof(__m128i));
        mTestIntPointer2 = (int32_t *) platformAlignedAlloc(sizeof(__m128i));
        srand(0);
        for (uint32_t i = 0; i < MAX_TEST_VALUE; i++) {
            mTestFloats[i] = ranf(-100000, 100000);
            mTestInts[i] = (int32_t) ranf(-100000, 100000);
        }
    }

    virtual ~SSE2NEONTestImpl(void)
    {
        platformAlignedFree(mTestFloatPointer1);
        platformAlignedFree(mTestFloatPointer2);
        platformAlignedFree(mTestIntPointer1);
        platformAlignedFree(mTestIntPointer2);
    }

    bool loadTestFloatPointers(uint32_t i)
    {
        bool ret = test_mm_store_ps(mTestFloatPointer1, mTestFloats[i],
                                    mTestFloats[i + 1], mTestFloats[i + 2],
                                    mTestFloats[i + 3]);
        if (ret) {
            ret = test_mm_store_ps(mTestFloatPointer2, mTestFloats[i + 4],
                                   mTestFloats[i + 5], mTestFloats[i + 6],
                                   mTestFloats[i + 7]);
        }
        return ret;
    }

    bool loadTestIntPointers(uint32_t i)
    {
        bool ret =
            test_mm_store_ps(mTestIntPointer1, mTestInts[i], mTestInts[i + 1],
                             mTestInts[i + 2], mTestInts[i + 3]);
        if (ret) {
            ret = test_mm_store_ps(mTestIntPointer2, mTestInts[i + 4],
                                   mTestInts[i + 5], mTestInts[i + 6],
                                   mTestInts[i + 7]);
        }

        return ret;
    }

    bool runSingleTest(InstructionTest test, uint32_t i)
    {
        bool ret = true;

        switch (test) {
        case IT_MM_SETZERO_SI128:
            ret = test_mm_setzero_si128();
            break;
        case IT_MM_SETZERO_PS:
            ret = test_mm_setzero_ps();
            break;
        case IT_MM_SET1_PS:
            ret = test_mm_set1_ps(mTestFloats[i]);
            break;
        case IT_MM_SET_PS1:
            ret = test_mm_set1_ps(mTestFloats[i]);
            break;
        case IT_MM_SET_PS:
            ret = test_mm_set_ps(mTestFloats[i], mTestFloats[i + 1],
                                 mTestFloats[i + 2], mTestFloats[i + 3]);
            break;
        case IT_MM_SET_SS:
            ret = test_mm_set_ss(mTestFloats[i]);
            break;
        case IT_MM_SET_EPI8:
            ret = test_mm_set_epi8((const int8_t *) mTestIntPointer1);
            break;
        case IT_MM_SET1_EPI32:
            ret = test_mm_set1_epi32(mTestInts[i]);
            break;
        case IT_MM_SET_EPI32:
            ret = testret_mm_set_epi32(mTestInts[i], mTestInts[i + 1],
                                       mTestInts[i + 2], mTestInts[i + 3]);
            break;
        case IT_MM_STORE_PS:
            ret = test_mm_store_ps(mTestIntPointer1, mTestInts[i],
                                   mTestInts[i + 1], mTestInts[i + 2],
                                   mTestInts[i + 3]);
            break;
        case IT_MM_STOREL_PI:
            ret = test_mm_storel_pi(mTestFloatPointer1);
            break;
        case IT_MM_STOREH_PI:
            ret = test_mm_storeh_pi(mTestFloatPointer1);
            break;
        case IT_MM_LOAD1_PS:
            ret = test_mm_load1_ps(mTestFloatPointer1);
            break;
        case IT_MM_LOADL_PI:
            ret = test_mm_loadl_pi(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_LOADH_PI:
            ret = test_mm_loadh_pi(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_ANDNOT_PS:
            ret = test_mm_andnot_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_ANDNOT_SI128:
            ret = test_mm_andnot_si128(mTestIntPointer1, mTestIntPointer2);
            break;
        case IT_MM_AND_SI128:
            ret = test_mm_and_si128(mTestIntPointer1, mTestIntPointer2);
            break;
        case IT_MM_AND_PS:
            ret = test_mm_and_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_OR_PS:
            ret = test_mm_or_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_OR_SI128:
            ret = test_mm_or_si128(mTestIntPointer1, mTestIntPointer2);
            break;
        case IT_MM_MOVEMASK_PS:
            ret = test_mm_movemask_ps(mTestFloatPointer1);
            break;
        case IT_MM_SHUFFLE_PS:
            ret = test_mm_shuffle_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_MOVEMASK_EPI8:
            ret = test_mm_movemask_epi8(mTestIntPointer1);
            break;
        case IT_MM_SUB_PS:
            ret = test_mm_sub_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_SUB_EPI64:
            ret = test_mm_sub_epi64((int64_t *) mTestIntPointer1,
                                    (int64_t *) mTestIntPointer2);
            break;
        case IT_MM_SUB_EPI32:
            ret = test_mm_sub_epi32(mTestIntPointer1, mTestIntPointer2);
            break;
        case IT_MM_ADD_PS:
            ret = test_mm_add_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_ADD_EPI32:
            ret = test_mm_add_epi32(mTestIntPointer1, mTestIntPointer2);
            break;
        case IT_MM_MULLO_EPI16:
            ret = test_mm_mullo_epi16((const int16_t *) mTestIntPointer1,
                                      (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_MUL_PS:
            ret = test_mm_mul_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_RCP_PS:
            ret = test_mm_rcp_ps(mTestFloatPointer1);
            break;
        case IT_MM_MAX_PS:
            ret = test_mm_max_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_MIN_PS:
            ret = test_mm_min_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_MIN_EPI16:
            ret = test_mm_min_epi16((const int16_t *) mTestIntPointer1,
                                    (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_MULHI_EPI16:
            ret = test_mm_mulhi_epi16((const int16_t *) mTestIntPointer1,
                                      (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_CMPLT_PS:
            ret = test_mm_cmplt_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_CMPGT_PS:
            ret = test_mm_cmpgt_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_CMPGE_PS:
            ret = test_mm_cmpge_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_CMPLE_PS:
            ret = test_mm_cmple_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_CMPEQ_PS:
            ret = test_mm_cmpeq_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_CMPLT_EPI32:
            ret = test_mm_cmplt_epi32(mTestIntPointer1, mTestIntPointer2);
            break;
        case IT_MM_CMPGT_EPI32:
            ret = test_mm_cmpgt_epi32(mTestIntPointer1, mTestIntPointer2);
            break;
        case IT_MM_CVTTPS_EPI32:
            ret = test_mm_cvttps_epi32(mTestFloatPointer1);
            break;
        case IT_MM_CVTEPI32_PS:
            ret = test_mm_cvtepi32_ps(mTestIntPointer1);
            break;
        case IT_MM_CVTPS_EPI32:
            ret = test_mm_cvtps_epi32(mTestFloatPointer1);
            break;
        case IT_MM_CMPORD_PS:
            ret = test_mm_cmpord_ps(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_COMILT_SS:
            ret = test_mm_comilt_ss(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_COMIGT_SS:
            ret = test_mm_comigt_ss(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_COMILE_SS:
            ret = test_mm_comile_ss(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_COMIGE_SS:
            ret = test_mm_comige_ss(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_COMIEQ_SS:
            ret = test_mm_comieq_ss(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_COMINEQ_SS:
            ret = test_mm_comineq_ss(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_HADD_PS:
            ret = true;
            break;
        case IT_MM_HADD_EPI16:
            ret = test_mm_hadd_epi16((const int16_t *) mTestIntPointer1,
                                     (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_MAX_EPI32:
            ret = true;
            break;
        case IT_MM_MIN_EPI32:
            ret = true;
            break;
        case IT_MM_MINPOS_EPU16:
            ret = test_mm_minpos_epu16((const int16_t *) mTestIntPointer1);
            break;
        case IT_MM_MAX_SS:
            ret = true;
            break;
        case IT_MM_MIN_SS:
            ret = true;
            break;
        case IT_MM_SQRT_PS:
            ret = true;
            break;
        case IT_MM_SQRT_SS:
            ret = true;
            break;
        case IT_MM_RSQRT_PS:
            ret = true;
            break;
        case IT_MM_DIV_PS:
            ret = true;
            break;
        case IT_MM_DIV_SS:
            ret = true;
            break;
        case IT_MM_MULLO_EPI32:
            ret = true;
            break;
        case IT_MM_MUL_EPU32:
            ret = test_mm_mul_epu32((const uint32_t *) mTestIntPointer1,
                                    (const uint32_t *) mTestIntPointer2);
            break;
        case IT_MM_ADD_EPI16:
            ret = true;
            break;
        case IT_MM_MADD_EPI16:
            ret = test_mm_madd_epi16((const int16_t *) mTestIntPointer1,
                                     (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_MADDUBS_EPI16:
            ret = test_mm_maddubs_epi16((const uint8_t *) mTestIntPointer1,
                                        (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_ADD_SS:
            ret = true;
            break;
        case IT_MM_SHUFFLE_EPI8:
            ret = test_mm_shuffle_epi8(mTestIntPointer1, mTestIntPointer2);
            break;
        case IT_MM_SHUFFLE_EPI32_DEFAULT:
            ret = true;
            break;
        case IT_MM_SHUFFLE_EPI32_FUNCTION:
            ret = true;
            break;
        case IT_MM_SHUFFLE_EPI32_SPLAT:
            ret = true;
            break;
        case IT_MM_SHUFFLE_EPI32_SINGLE:
            ret = true;
            break;
        case IT_MM_SHUFFLEHI_EPI16_FUNCTION:
            ret = true;
            break;
        case IT_MM_XOR_SI128:
            ret = true;
            break;
        case IT_MM_XOR_PS:
            ret = true;
            break;
        case IT_MM_LOAD_PS:
            ret = true;
            break;
        case IT_MM_LOADU_PS:
            ret = true;
            break;
        case IT_MM_LOAD_SD:
            ret = test_mm_load_sd((const double *) mTestFloatPointer1);
            break;
        case IT_MM_LOAD_SS:
            ret = true;
            break;
        case IT_MM_CMPNEQ_PS:
            ret = true;
            break;
        case IT_MM_STOREU_PS:
            ret = true;
            break;
        case IT_MM_STORE_SI128:
            ret = true;
            break;
        case IT_MM_STORE_SS:
            ret = true;
            break;
        case IT_MM_STOREL_EPI64:
            ret = true;
            break;
        case IT_MM_SETR_PS:
            ret = true;
            break;
        case IT_MM_CVTSI128_SI32:
            ret = true;
            break;
        case IT_MM_CVTSI32_SI128:
            ret = true;
            break;
        case IT_MM_CASTPS_SI128:
            ret = true;
            break;
        case IT_MM_CASTSI128_PS:
            ret = true;
            break;
        case IT_MM_LOAD_SI128:
            ret = true;
            break;
        case IT_MM_PACKS_EPI16:
            ret = true;
            break;
        case IT_MM_PACKUS_EPI16:
            ret = true;
            break;
        case IT_MM_PACKS_EPI32:
            ret = true;
            break;
        case IT_MM_UNPACKLO_EPI8:
            ret = true;
            break;
        case IT_MM_UNPACKLO_EPI16:
            ret = true;
            break;
        case IT_MM_UNPACKLO_EPI32:
            ret = true;
            break;
        case IT_MM_UNPACKLO_PS:
            ret = true;
            break;
        case IT_MM_UNPACKHI_PS:
            ret = true;
            break;
        case IT_MM_UNPACKHI_EPI8:
            ret = true;
            break;
        case IT_MM_UNPACKHI_EPI16:
            ret = true;
            break;
        case IT_MM_UNPACKHI_EPI32:
            ret = true;
            break;
        case IT_MM_SFENCE:
            ret = true;
            break;
        case IT_MM_STREAM_SI128:
            ret = true;
            break;
        case IT_MM_CLFLUSH:
            ret = true;
            break;

        case IT_MM_CVTSS_F32:
            ret = true;
            break;

        case IT_MM_SET1_EPI16:
            ret = test_mm_set1_epi16((const int16_t *) mTestIntPointer1);
            break;
        case IT_MM_SET_EPI16:
            ret = test_mm_set_epi16((const int16_t *) mTestIntPointer1);
            break;
        case IT_MM_SRA_EPI16:
            ret = test_mm_sra_epi16((const int16_t *) mTestIntPointer1,
                                    (int64_t) i);
            break;
        case IT_MM_SRA_EPI32:
            ret = test_mm_sra_epi32((const int32_t *) mTestIntPointer1,
                                    (int64_t) i);
            break;
        case IT_MM_SLLI_EPI16:
            ret = test_mm_slli_epi16((const int16_t *) mTestIntPointer1);
            break;
        case IT_MM_SLL_EPI16:
            ret = test_mm_sll_epi16((const int16_t *) mTestIntPointer1,
                                    (int64_t) i);
            break;
        case IT_MM_SLL_EPI32:
            ret = test_mm_sll_epi32((const int32_t *) mTestIntPointer1,
                                    (int64_t) i);
            break;
        case IT_MM_SLL_EPI64:
            ret = test_mm_sll_epi64((const int64_t *) mTestIntPointer1,
                                    (int64_t) i);
            break;
        case IT_MM_SRL_EPI16:
            ret = test_mm_srl_epi16((const int16_t *) mTestIntPointer1,
                                    (int64_t) i);
            break;
        case IT_MM_SRL_EPI32:
            ret = test_mm_srl_epi32((const int32_t *) mTestIntPointer1,
                                    (int64_t) i);
            break;
        case IT_MM_SRL_EPI64:
            ret = test_mm_srl_epi64((const int64_t *) mTestIntPointer1,
                                    (int64_t) i);
            break;
        case IT_MM_SRLI_EPI16:
            ret = test_mm_srli_epi16((const int16_t *) mTestIntPointer1);
            break;
        case IT_MM_CMPEQ_EPI16:
            ret = test_mm_cmpeq_epi16((const int16_t *) mTestIntPointer1,
                                      (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_CMPEQ_EPI64:
            ret = test_mm_cmpeq_epi64((const int64_t *) mTestIntPointer1,
                                      (const int64_t *) mTestIntPointer2);
            break;
        case IT_MM_SET1_EPI8:
            ret = test_mm_set1_epi8((const int8_t *) mTestIntPointer1);
            break;
        case IT_MM_ADDS_EPU8:
            ret = test_mm_adds_epu8((const int8_t *) mTestIntPointer1,
                                    (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_SUBS_EPU8:
            ret = test_mm_subs_epu8((const int8_t *) mTestIntPointer1,
                                    (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_MAX_EPU8:
            ret = test_mm_max_epu8((const int8_t *) mTestIntPointer1,
                                   (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_CMPEQ_EPI8:
            ret = test_mm_cmpeq_epi8((const int8_t *) mTestIntPointer1,
                                     (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_ADDS_EPI16:
            ret = test_mm_adds_epi16((const int16_t *) mTestIntPointer1,
                                     (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_MAX_EPI16:
            ret = test_mm_max_epi16((const int16_t *) mTestIntPointer1,
                                    (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_SUBS_EPU16:
            ret = test_mm_subs_epu16((const int16_t *) mTestIntPointer1,
                                     (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_CMPLT_EPI16:
            ret = test_mm_cmplt_epi16((const int16_t *) mTestIntPointer1,
                                      (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_CMPGT_EPI16:
            ret = test_mm_cmpgt_epi16((const int16_t *) mTestIntPointer1,
                                      (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_LOADU_SI128:
            ret = test_mm_loadu_si128((const int32_t *) mTestIntPointer1);
            break;
        case IT_MM_STOREU_SI128:
            ret = test_mm_storeu_si128((const int32_t *) mTestIntPointer1);
            break;
        case IT_MM_ADD_EPI8:
            ret = test_mm_add_epi8((const int8_t *) mTestIntPointer1,
                                   (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_CMPGT_EPI8:
            ret = test_mm_cmpgt_epi8((const int8_t *) mTestIntPointer1,
                                     (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_CMPLT_EPI8:
            ret = test_mm_cmplt_epi8((const int8_t *) mTestIntPointer1,
                                     (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_SUB_EPI8:
            ret = test_mm_sub_epi8((const int8_t *) mTestIntPointer1,
                                   (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_SETR_EPI32:
            ret = test_mm_setr_epi32((const int32_t *) mTestIntPointer1);
            break;
        case IT_MM_MIN_EPU8:
            ret = test_mm_min_epu8((const int8_t *) mTestIntPointer1,
                                   (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_TEST_ALL_ZEROS:
            ret = test_mm_test_all_zeros((const int32_t *) mTestIntPointer1,
                                         (const int32_t *) mTestIntPointer2);
            break;
        case IT_MM_AVG_EPU8:
            ret = test_mm_avg_epu8((const int8_t *) mTestIntPointer1,
                                   (const int8_t *) mTestIntPointer2);
            break;
        case IT_MM_AVG_EPU16:
            ret = test_mm_avg_epu16((const int16_t *) mTestIntPointer1,
                                    (const int16_t *) mTestIntPointer2);
            break;
        case IT_MM_POPCNT_U32:
            ret = test_mm_popcnt_u32((const uint32_t *) mTestIntPointer1);
            break;
        case IT_MM_POPCNT_U64:
            ret = test_mm_popcnt_u64((const uint64_t *) mTestIntPointer1);
            break;
#if !defined(__arm__) && __ARM_ARCH != 7
        case IT_MM_AESENC_SI128:
            ret = test_mm_aesenc_si128(mTestIntPointer1, mTestIntPointer2);
            break;
#endif
        case IT_MM_CLMULEPI64_SI128:
            ret = test_mm_clmulepi64_si128((const uint64_t *) mTestIntPointer1,
                                           (const uint64_t *) mTestIntPointer2);
            break;
        case IT_MM_MALLOC:
            ret = test_mm_malloc((const size_t *) mTestIntPointer1,
                                 (const size_t *) mTestIntPointer2);
            break;
        case IT_MM_CRC32_U8:
            ret = test_mm_crc32_u8(*(const uint32_t *) mTestIntPointer1, i);
            break;
        case IT_MM_CRC32_U16:
            ret = test_mm_crc32_u16(*(const uint32_t *) mTestIntPointer1, i);
            break;
        case IT_MM_CRC32_U32:
            ret = test_mm_crc32_u32(*(const uint32_t *) mTestIntPointer1,
                                    *(const uint32_t *) mTestIntPointer2);
            break;
        case IT_MM_CRC32_U64:
            ret = test_mm_crc32_u64(*(const uint64_t *) mTestIntPointer1,
                                    *(const uint64_t *) mTestIntPointer2);
            break;
        case IT_LAST: /* should not happend */
            break;
        }

        return ret;
    }

    virtual bool runTest(InstructionTest test)
    {
        bool ret = true;

        // Test a whole bunch of values
        for (uint32_t i = 0; i < (MAX_TEST_VALUE - 8); i++) {
            ret = loadTestFloatPointers(i);  // Load some random float values
            if (!ret)
                break;                     // load test float failed??
            ret = loadTestIntPointers(i);  // load some random int values
            if (!ret)
                break;  // load test float failed??
            // If we are testing the reciprocal, then invert the input data
            // (easier for debugging)
            if (test == IT_MM_RCP_PS) {
                mTestFloatPointer1[0] = 1.0f / mTestFloatPointer1[0];
                mTestFloatPointer1[1] = 1.0f / mTestFloatPointer1[1];
                mTestFloatPointer1[2] = 1.0f / mTestFloatPointer1[2];
                mTestFloatPointer1[3] = 1.0f / mTestFloatPointer1[3];
            }
            if (test == IT_MM_CMPGE_PS || test == IT_MM_CMPLE_PS ||
                test == IT_MM_CMPEQ_PS) {
                // Make sure at least one value is the same.
                mTestFloatPointer1[3] = mTestFloatPointer2[3];
            }

            if (test == IT_MM_CMPORD_PS || test == IT_MM_COMILT_SS ||
                test == IT_MM_COMILE_SS || test == IT_MM_COMIGE_SS ||
                test == IT_MM_COMIEQ_SS || test == IT_MM_COMINEQ_SS ||
                test == IT_MM_COMIGT_SS) {  // if testing for NAN's make sure we
                                            // have some nans
                // One out of four times
                // Make sure a couple of values have NANs for testing purposes
                if ((rand() & 3) == 0) {
                    uint32_t r1 = rand() & 3;
                    uint32_t r2 = rand() & 3;
                    mTestFloatPointer1[r1] = getNAN();
                    mTestFloatPointer2[r2] = getNAN();
                }
            }

            // one out of every random 64 times or so, mix up the test floats to
            // contain some integer values
            if ((rand() & 63) == 0) {
                uint32_t option = rand() & 3;
                switch (option) {
                // All integers..
                case 0:
                    mTestFloatPointer1[0] = float(mTestIntPointer1[0]);
                    mTestFloatPointer1[1] = float(mTestIntPointer1[1]);
                    mTestFloatPointer1[2] = float(mTestIntPointer1[2]);
                    mTestFloatPointer1[3] = float(mTestIntPointer1[3]);

                    mTestFloatPointer2[0] = float(mTestIntPointer2[0]);
                    mTestFloatPointer2[1] = float(mTestIntPointer2[1]);
                    mTestFloatPointer2[2] = float(mTestIntPointer2[2]);
                    mTestFloatPointer2[3] = float(mTestIntPointer2[3]);

                    break;
                case 1: {
                    uint32_t index = rand() & 3;
                    mTestFloatPointer1[index] = float(mTestIntPointer1[index]);
                    index = rand() & 3;
                    mTestFloatPointer2[index] = float(mTestIntPointer2[index]);
                } break;
                case 2: {
                    uint32_t index1 = rand() & 3;
                    uint32_t index2 = rand() & 3;
                    mTestFloatPointer1[index1] =
                        float(mTestIntPointer1[index1]);
                    mTestFloatPointer1[index2] =
                        float(mTestIntPointer1[index2]);
                    index1 = rand() & 3;
                    index2 = rand() & 3;
                    mTestFloatPointer1[index1] =
                        float(mTestIntPointer1[index1]);
                    mTestFloatPointer1[index2] =
                        float(mTestIntPointer1[index2]);
                } break;
                case 3:
                    mTestFloatPointer1[0] = float(mTestIntPointer1[0]);
                    mTestFloatPointer1[1] = float(mTestIntPointer1[1]);
                    mTestFloatPointer1[2] = float(mTestIntPointer1[2]);
                    mTestFloatPointer1[3] = float(mTestIntPointer1[3]);
                    break;
                }
                if ((rand() & 3) == 0) {  // one out of 4 times, make halves
                    for (uint32_t j = 0; j < 4; j++) {
                        mTestFloatPointer1[j] *= 0.5f;
                        mTestFloatPointer2[j] *= 0.5f;
                    }
                }
            }
#if 0
            {
                mTestFloatPointer1[0] = getNAN();
                mTestFloatPointer2[0] = getNAN();
                bool ok = test_mm_comilt_ss(mTestFloatPointer1, mTestFloatPointer1);
                if (!ok) {
                    printf("Debug me");
                }
            }
#endif
            ret = runSingleTest(test, i);
            if (!ret)  // the test failed...
            {
                // Set a breakpoint here if you want to step through the failure
                // case in the debugger
                ret = runSingleTest(test, i);
                break;
            }
        }
        return ret;
    }

    virtual void release(void) { delete this; }

    float *mTestFloatPointer1;
    float *mTestFloatPointer2;
    int32_t *mTestIntPointer1;
    int32_t *mTestIntPointer2;
    float mTestFloats[MAX_TEST_VALUE];
    int32_t mTestInts[MAX_TEST_VALUE];
};

SSE2NEONTest *SSE2NEONTest::create(void)
{
    SSE2NEONTestImpl *st = new SSE2NEONTestImpl;
    return static_cast<SSE2NEONTest *>(st);
}

}  // namespace SSE2NEON
