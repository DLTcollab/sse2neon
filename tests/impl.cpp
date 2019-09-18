#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "binding.h"
#include "impl.h"

// This program a set of unit tests to ensure that each SSE call provide the
// output we expect.  If this fires an assert, then something didn't match up.

#include "sse2neon.h"

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
    case IT_MM_LOAD_PS:
        ret = "MM_LOAD_PS";
        break;
    case IT_MM_LOADU_PS:
        ret = "MM_LOADU_PS";
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
    case IT_MM_MULHI_EPI16:
        ret = "MM_MULHI_EPI16";
        break;
    case IT_MM_HADD_PS:
        ret = "MM_HADD_PS";
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
    case IT_MM_SLLI_EPI16:
        ret = "MM_SLLI_EPI16";
        break;
    case IT_MM_SRLI_EPI16:
        ret = "MM_SRLI_EPI16";
        break;
    case IT_MM_CMPEQ_EPI16:
        ret = "MM_CMPEQ_EPI16";
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
    case IT_MM_AESENC_SI128:
        ret = "IT_MM_AESENC_SI128";
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

    ASSERT_RETURN(t1[3] == t2[3]);
    ASSERT_RETURN(t1[2] == t2[2]);
    ASSERT_RETURN(t1[1] == t2[1]);
    ASSERT_RETURN(t1[0] == t2[0]);
    return true;
}

bool validateUInt64(__m128i a, uint64_t x, uint64_t y)
{
    const uint64_t *t = (const uint64_t *) &a;
    ASSERT_RETURN(t[1] == x);
    ASSERT_RETURN(t[0] == y);
    return true;
}

bool validateInt64(__m128i a, int64_t x, int64_t y)
{
    const int64_t *t = (const int64_t *) &a;
    ASSERT_RETURN(t[1] == x);
    ASSERT_RETURN(t[0] == y);
    return true;
}

bool validateInt(__m128i a, int32_t x, int32_t y, int32_t z, int32_t w)
{
    const int32_t *t = (const int32_t *) &a;
    ASSERT_RETURN(t[3] == x);
    ASSERT_RETURN(t[2] == y);
    ASSERT_RETURN(t[1] == z);
    ASSERT_RETURN(t[0] == w);
    return true;
}

bool validateInt16(__m128i a,
                   int16_t d0,
                   int16_t d1,
                   int16_t d2,
                   int16_t d3,
                   int16_t d4,
                   int16_t d5,
                   int16_t d6,
                   int16_t d7)
{
    const int16_t *t = (const int16_t *) &a;
    ASSERT_RETURN(t[0] == d0);
    ASSERT_RETURN(t[1] == d1);
    ASSERT_RETURN(t[2] == d2);
    ASSERT_RETURN(t[3] == d3);
    ASSERT_RETURN(t[4] == d4);
    ASSERT_RETURN(t[5] == d5);
    ASSERT_RETURN(t[6] == d6);
    ASSERT_RETURN(t[7] == d7);
    return true;
}

bool validateInt8(__m128i a,
                  int8_t d0,
                  int8_t d1,
                  int8_t d2,
                  int8_t d3,
                  int8_t d4,
                  int8_t d5,
                  int8_t d6,
                  int8_t d7,
                  int8_t d8,
                  int8_t d9,
                  int8_t d10,
                  int8_t d11,
                  int8_t d12,
                  int8_t d13,
                  int8_t d14,
                  int8_t d15)
{
    const int8_t *t = (const int8_t *) &a;
    ASSERT_RETURN(t[0] == d0);
    ASSERT_RETURN(t[1] == d1);
    ASSERT_RETURN(t[2] == d2);
    ASSERT_RETURN(t[3] == d3);
    ASSERT_RETURN(t[4] == d4);
    ASSERT_RETURN(t[5] == d5);
    ASSERT_RETURN(t[6] == d6);
    ASSERT_RETURN(t[7] == d7);
    ASSERT_RETURN(t[8] == d8);
    ASSERT_RETURN(t[9] == d9);
    ASSERT_RETURN(t[10] == d10);
    ASSERT_RETURN(t[11] == d11);
    ASSERT_RETURN(t[12] == d12);
    ASSERT_RETURN(t[13] == d13);
    ASSERT_RETURN(t[14] == d14);
    ASSERT_RETURN(t[15] == d15);
    return true;
}

bool validateSingleFloatPair(float a, float b)
{
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    return (*ia) == (*ib)
               ? true
               : false;  // We do an integer (binary) compare rather than a
                         // floating point compare to take nands and infinities
                         // into account as well.
}

bool validateFloat(__m128 a, float x, float y, float z, float w)
{
    const float *t = (const float *) &a;
    ASSERT_RETURN(validateSingleFloatPair(t[3], x));
    ASSERT_RETURN(validateSingleFloatPair(t[2], y));
    ASSERT_RETURN(validateSingleFloatPair(t[1], z));
    ASSERT_RETURN(validateSingleFloatPair(t[0], w));
    return true;
}

bool validateFloatEpsilon(__m128 a,
                          float x,
                          float y,
                          float z,
                          float w,
                          float epsilon)
{
    const float *t = (const float *) &a;
    float dx = fabsf(t[3] - x);
    float dy = fabsf(t[2] - y);
    float dz = fabsf(t[1] - z);
    float dw = fabsf(t[0] - w);
    ASSERT_RETURN(dx < epsilon);
    ASSERT_RETURN(dy < epsilon);
    ASSERT_RETURN(dz < epsilon);
    ASSERT_RETURN(dw < epsilon);
    return true;
}

bool test_mm_setzero_si128(void)
{
    __m128i a = _mm_setzero_si128();
    return validateInt(a, 0, 0, 0, 0);
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
    return validateFloat(a, x, y, z, w);
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
    return validateInt(a, i, i, i, i);
}

bool testret_mm_set_epi32(int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    return validateInt(a, x, y, z, w);
}

__m128i test_mm_set_epi32(int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    validateInt(a, x, y, z, w);
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

bool test_mm_load1_ps(const float *p)
{
    __m128 a = _mm_load1_ps(p);
    return validateFloat(a, p[0], p[0], p[0], p[0]);
}

bool test_mm_loadl_pi(const float *p1, const float *p2)
{
    __m128 a = _mm_load_ps(p1);
    __m64 *b = (__m64 *) p2;
    __m128 c = _mm_loadl_pi(a, b);

    return validateFloat(c, p1[3], p1[2], p2[1], p2[0]);
}

__m128 test_mm_load_ps(const float *p)
{
    __m128 a = _mm_load_ps(p);
    validateFloat(a, p[3], p[2], p[1], p[0]);
    return a;
}

__m128i test_mm_load_ps(const int32_t *p)
{
    __m128 a = _mm_load_ps((const float *) p);
    __m128i ia = *(const __m128i *) &a;
    validateInt(ia, p[3], p[2], p[1], p[0]);
    return ia;
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
    r = validateInt(*(const __m128i *) &c, r3, r2, r1, r0);
    if (r) {
        r = validateInt(ret, r3, r2, r1, r0);
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
    bool r = validateInt(*(const __m128i *) &c, r3, r2, r1, r0);
    if (r) {
        r = validateInt(ret, r3, r2, r1, r0);
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
    bool r = validateInt(*(const __m128i *) &c, r3, r2, r1, r0);
    if (r) {
        r = validateInt(ret, r3, r2, r1, r0);
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
    r = validateInt(c, r3, r2, r1, r0);
    if (r) {
        validateInt(ret, r3, r2, r1, r0);
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
    bool r = validateInt(c, r3, r2, r1, r0);
    if (r) {
        r = validateInt(ret, r3, r2, r1, r0);
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
    bool r = validateInt(c, r3, r2, r1, r0);
    if (r) {
        r = validateInt(ret, r3, r2, r1, r0);
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
    // permutations which have an optmized/custom implementation
    __m128 ret;
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 1, 2, 3));
    if (!validateFloat(ret, _b[0], _b[1], _a[2], _a[3])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 1, 0));
    if (!validateFloat(ret, _b[3], _b[2], _a[1], _a[0])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 1, 1));
    if (!validateFloat(ret, _b[0], _b[0], _a[1], _a[1])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 0, 2));
    if (!validateFloat(ret, _b[3], _b[1], _a[0], _a[2])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 0, 3, 2));
    if (!validateFloat(ret, _b[1], _b[0], _a[3], _a[2])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 3, 0, 1));
    if (!validateFloat(ret, _b[2], _b[3], _a[0], _a[1])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 2, 2));
    if (!validateFloat(ret, _b[0], _b[0], _a[2], _a[2])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 2, 0, 0));
    if (!validateFloat(ret, _b[2], _b[2], _a[0], _a[0])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 0, 2));
    if (!validateFloat(ret, _b[3], _b[2], _a[0], _a[2])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 1, 3, 3));
    if (!validateFloat(ret, _b[1], _b[1], _a[3], _a[3])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 1, 0));
    if (!validateFloat(ret, _b[2], _b[0], _a[1], _a[0])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 0, 1));
    if (!validateFloat(ret, _b[2], _b[0], _a[0], _a[1])) {
        isValid = false;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 3, 2));
    if (!validateFloat(ret, _b[2], _b[0], _a[3], _a[2])) {
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
    return validateFloat(c, dw, dz, dy, dx);
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
    return validateInt(c, dw, dz, dy, dx);
}

bool test_mm_sub_epi64(const int64_t *_a, const int64_t *_b)
{
    int64_t d0 = _a[0] - _b[0];
    int64_t d1 = _a[1] - _b[1];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sub_epi64(a, b);
    return validateInt64(c, d1, d0);
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
    return validateFloat(c, dw, dz, dy, dx);
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
    return validateInt(c, dw, dz, dy, dx);
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
    uint64_t dx = _a[0] * _b[0];
    uint64_t dy = _a[2] * _b[2];

    __m128i a = test_mm_load_ps((const int32_t *) _a);
    __m128i b = test_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_mul_epu32(a, b);
    return validateUInt64(c, dy, dx);
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
    return validateInt(c, e3, e2, e1, e0);
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

    return validateInt(ret, r[3], r[2], r[1], r[0]);
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
    return validateFloat(c, dw, dz, dy, dx);
}

bool test_mm_rcp_ps(const float *_a)
{
    float dx = 1.0f / _a[0];
    float dy = 1.0f / _a[1];
    float dz = 1.0f / _a[2];
    float dw = 1.0f / _a[3];

    __m128 a = test_mm_load_ps(_a);
    __m128 c = _mm_rcp_ps(a);
    return validateFloatEpsilon(c, dw, dz, dy, dx, 300.0f);
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
    return validateFloat(ret, c[3], c[2], c[1], c[0]);
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
    return validateFloat(ret, c[3], c[2], c[1], c[0]);
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
    return validateInt(iret, result[3], result[2], result[1], result[0]);
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
    return validateInt(iret, result[3], result[2], result[1], result[0]);
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
    return validateInt(iret, result[3], result[2], result[1], result[0]);
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
    return validateInt(iret, result[3], result[2], result[1], result[0]);
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
    return validateInt(iret, result[3], result[2], result[1], result[0]);
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
    return validateInt(iret, result[3], result[2], result[1], result[0]);
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
    return validateInt(iret, result[3], result[2], result[1], result[0]);
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

    return validateFloat(ret, result[3], result[2], result[1], result[0]);
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

    int32_t result = comineq_ss(_a[0], _b[0]);
    int32_t ret = _mm_comineq_ss(a, b);

    return result == ret ? true : false;
}

bool test_mm_cvttps_epi32(const float *_a)
{
    __m128 a = test_mm_load_ps(_a);
    int32_t trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (int32_t) _a[i];
    }

    __m128i ret = _mm_cvttps_epi32(a);
    return validateInt(ret, trun[3], trun[2], trun[1], trun[0]);
}

bool test_mm_cvtepi32_ps(const int32_t *_a)
{
    __m128i a = test_mm_load_ps(_a);
    float trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (float) _a[i];
    }

    __m128 ret = _mm_cvtepi32_ps(a);
    return validateFloat(ret, trun[3], trun[2], trun[1], trun[0]);
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
    return validateInt(ret, trun[3], trun[2], trun[1], trun[0]);
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
    return validateInt(c, _a[3], _a[2], _a[1], _a[0]);
}

bool test_mm_storeu_si128(const int32_t *_a)
{
    __m128i b;
    __m128i a = _mm_loadu_si128((const __m128i *) _a);
    _mm_storeu_si128(&b, a);
    int32_t *_b = (int32_t *) &b;
    return validateInt(a, _b[3], _b[2], _b[1], _b[0]);
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
    return validateInt(c, _a[3], _a[2], _a[1], _a[0]);
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

bool test_mm_test_all_zeros(const int32_t *_a, const int32_t *_mask)
{
    __m128i a = test_mm_load_ps(_a);
    __m128i mask = test_mm_load_ps(_mask);

    int32_t d0 = a[0] & mask[0];
    int32_t d1 = a[1] & mask[1];
    int32_t d2 = a[2] & mask[2];
    int32_t d3 = a[3] & mask[3];
    int32_t result = ((d0 | d1 | d2 | d3) == 0) ? 1 : 0;

    int32_t ret = _mm_test_all_zeros(a, mask);

    return result == ret ? true : false;
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

bool test_mm_aesenc_si128(const int32_t *a, const int32_t *b)
{
    __m128i data = _mm_loadu_si128((const __m128i *) a);
    __m128i rk = _mm_loadu_si128((const __m128i *) b);

    __m128i resultReference = aesenc_128_reference(data, rk);
    __m128i resultIntrinsic = _mm_aesenc_si128(data, rk);

    return validate128(resultReference, resultIntrinsic);
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
        case IT_MM_LOAD1_PS:
            ret = test_mm_load1_ps(mTestFloatPointer1);
            break;
        case IT_MM_LOADL_PI:
            ret = test_mm_loadl_pi(mTestFloatPointer1, mTestFloatPointer2);
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
            if (!ret) {
                // FIXME: incomplete
                ret = test_mm_comilt_ss(mTestFloatPointer1, mTestFloatPointer2);
            }
            break;
        case IT_MM_COMIGT_SS:
            ret = test_mm_comigt_ss(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_COMILE_SS:
            ret = test_mm_comile_ss(mTestFloatPointer1, mTestFloatPointer2);
            if (!ret) {
                // FIXME: incomplete
                ret = test_mm_comile_ss(mTestFloatPointer1, mTestFloatPointer2);
            }
            break;
        case IT_MM_COMIGE_SS:
            ret = test_mm_comige_ss(mTestFloatPointer1, mTestFloatPointer2);
            break;
        case IT_MM_COMIEQ_SS:
            ret = test_mm_comieq_ss(mTestFloatPointer1, mTestFloatPointer2);
            if (!ret) {
                // FIXME: incomplete
                ret = test_mm_comieq_ss(mTestFloatPointer1, mTestFloatPointer2);
            }
            break;
        case IT_MM_COMINEQ_SS:
            ret = test_mm_comineq_ss(mTestFloatPointer1, mTestFloatPointer2);
            if (!ret) {
                // FIXME: incomplete
                ret =
                    test_mm_comineq_ss(mTestFloatPointer1, mTestFloatPointer2);
            }
            break;
        case IT_MM_HADD_PS:
            ret = true;
            break;
        case IT_MM_MAX_EPI32:
            ret = true;
            break;
        case IT_MM_MIN_EPI32:
            ret = true;
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
        case IT_MM_SLLI_EPI16:
            ret = test_mm_slli_epi16((const int16_t *) mTestIntPointer1);
            break;
        case IT_MM_SRLI_EPI16:
            ret = test_mm_srli_epi16((const int16_t *) mTestIntPointer1);
            break;
        case IT_MM_CMPEQ_EPI16:
            ret = test_mm_cmpeq_epi16((const int16_t *) mTestIntPointer1,
                                      (const int16_t *) mTestIntPointer2);
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
        case IT_MM_AESENC_SI128:
            ret = test_mm_aesenc_si128(mTestIntPointer1, mTestIntPointer2);
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
