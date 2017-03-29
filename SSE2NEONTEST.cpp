#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <math.h>

#include "SSE2NEONBinding.h"
#include "SSE2NEONTEST.h"

// SSE2NEONTEST performs a set of 'unit tests' making sure that each SSE call
// provides the output we expect.  If this fires an assert, then something didn't match up.


#ifdef WIN32

#pragma warning(disable:4211)

#include <xmmintrin.h>
#include <emmintrin.h>

#else

#include "SSE2NEON.h"

#endif

namespace SSE2NEON
{

// hex representation of an IEEE NAN
const uint32_t inan = 0xffffffff;

static inline float getNAN(void)
{
    const float *fn = (const float *)&inan;
    return *fn;
}

static inline bool isNAN(float a)
{
    const uint32_t *ia = (const uint32_t *)&a;
    return (*ia) == inan ? true : false;
}

// Do a round operation that produces results the same as SSE instructions
static inline float bankersRounding(float val)
{
    if (val < 0)
    {
        return -bankersRounding(-val);
    }
    float ret;
    int32_t truncateInteger = int32_t(val);
    int32_t roundInteger = int32_t(val + 0.5f);
    float diff1 = val - float(truncateInteger); // Truncate value
    float diff2 = val - float(roundInteger);    // Round up value
    if (diff2 < 0) diff2 *= -1; // get the positive difference from the round up value
    // If it's closest to the truncate integer; then use it
    if (diff1 < diff2)
    {
        ret = float(truncateInteger);
    }
    else if (diff2 < diff1) // if it's closest to the round-up integer; use it
    {
        ret = float(roundInteger);
    }
    else
    {
        // If it's equidistant between rounding up and rounding down, pick the one which is an even number
        if (truncateInteger & 1) // If truncate is odd, then return the rounded integer
        {
            ret = float(roundInteger);
        }
        else
        {
            // If the rounded up value is odd, use return the truncated integer
            ret = float(truncateInteger);
        }
    }
    return ret;
}


    const char *SSE2NEONTest::getInstructionTestString(InstructionTest test)
    {
        const char *ret = "UNKNOWN!";

        switch (test)
        {
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
        case IT_MM_SET1_EPI32:
            ret = "MM_SET1_EPI32";
            break;
        case IT_MM_SET_EPI32:
            ret = "MM_SET_EPI32";
            break;
        case IT_MM_STORE_PS:
            ret = "MM_STORE_PS";
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
        case IT_MM_MULLO_EPI16:
            ret = "MM_MULLO_EPI16";
            break;
        case IT_MM_MULLO_EPI32:
            ret = "MM_MULLO_EPI32";
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
        }


        return ret;
    }


#define ASSERT_RETURN(x) if ( !(x) ) return false;

    static float ranf(void)
    {
        uint32_t ir = rand() & 0x7FFF;
        return (float)ir*(1.0f / 32768.0f);
    }

    static float ranf(float low, float high)
    {
        return ranf()*(high - low) + low;
    }

    bool validateInt(__m128i a, int32_t x, int32_t y, int32_t z, int32_t w)
    {
        const int32_t *t = (const int32_t *)&a;
        ASSERT_RETURN(t[3] == x);
        ASSERT_RETURN(t[2] == y);
        ASSERT_RETURN(t[1] == z);
        ASSERT_RETURN(t[0] == w);
        return true;
    }

    bool validateInt16(__m128i a, int16_t d0, int16_t d1, int16_t d2, int16_t d3, int16_t d4, int16_t d5, int16_t d6, int16_t d7)
    {
        const int16_t *t = (const int16_t *)&a;
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

    bool validateSingleFloatPair(float a, float b)
    {
        const uint32_t *ia = (const uint32_t *)&a;
        const uint32_t *ib = (const uint32_t *)&b;
        return (*ia) == (*ib) ? true : false;   // We do an integer (binary) compare rather than a floating point compare to take nands and infinities into account as well.
    }

    bool validateFloat(__m128 a, float x, float y, float z, float w)
    {
        const float *t = (const float *)&a;
        ASSERT_RETURN(validateSingleFloatPair(t[3],x));
        ASSERT_RETURN(validateSingleFloatPair(t[2],y));
        ASSERT_RETURN(validateSingleFloatPair(t[1],z));
        ASSERT_RETURN(validateSingleFloatPair(t[0],w));
        return true;
    }

    bool validateFloatEpsilon(__m128 a, float x, float y, float z, float w, float epsilon)
    {
        const float *t = (const float *)&a;
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
        _mm_store_ps((float *)p, *(const __m128 *)&a);
        ASSERT_RETURN(p[0] == w);
        ASSERT_RETURN(p[1] == z);
        ASSERT_RETURN(p[2] == y);
        ASSERT_RETURN(p[3] == x);
        return true;
    }

    bool test_mm_load1_ps(const float *p)
    {
        __m128 a = _mm_load1_ps(p);
        return validateFloat(a, p[0], p[0], p[0], p[0]);
    }

    __m128 test_mm_load_ps(const float *p)
    {
        __m128 a = _mm_load_ps(p);
        validateFloat(a, p[3], p[2], p[1], p[0]);
        return a;
    }

    __m128i test_mm_load_ps(const int32_t *p)
    {
        __m128 a = _mm_load_ps((const float *)p);
        __m128i ia = *(const __m128i *)&a;
        validateInt(ia, p[3], p[2], p[1], p[0]);
        return ia;
    }


    //r0 := ~a0 & b0
    //r1 := ~a1 & b1
    //r2 := ~a2 & b2
    //r3 := ~a3 & b3
    bool test_mm_andnot_ps(const float *_a, const float *_b)
    {
        bool r = false;

        __m128 a = test_mm_load_ps(_a);
        __m128 b = test_mm_load_ps(_b);
        __m128 c = _mm_andnot_ps(a, b);
        // now for the assertion...
        const uint32_t *ia = (const uint32_t *)&a;
        const uint32_t *ib = (const uint32_t *)&b;
        uint32_t r0 = ~ia[0] & ib[0];
        uint32_t r1 = ~ia[1] & ib[1];
        uint32_t r2 = ~ia[2] & ib[2];
        uint32_t r3 = ~ia[3] & ib[3];
        __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
        r = validateInt(*(const __m128i *)&c, r3, r2, r1, r0);
        if (r)
        {
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
        const uint32_t *ia = (const uint32_t *)&a;
        const uint32_t *ib = (const uint32_t *)&b;
        uint32_t r0 = ia[0] & ib[0];
        uint32_t r1 = ia[1] & ib[1];
        uint32_t r2 = ia[2] & ib[2];
        uint32_t r3 = ia[3] & ib[3];
        __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
        bool r = validateInt(*(const __m128i *)&c, r3, r2, r1, r0);
        if (r)
        {
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
        const uint32_t *ia = (const uint32_t *)&a;
        const uint32_t *ib = (const uint32_t *)&b;
        uint32_t r0 = ia[0] | ib[0];
        uint32_t r1 = ia[1] | ib[1];
        uint32_t r2 = ia[2] | ib[2];
        uint32_t r3 = ia[3] | ib[3];
        __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
        bool r = validateInt(*(const __m128i *)&c, r3, r2, r1, r0);
        if (r)
        {
            r = validateInt(ret, r3, r2, r1, r0);
        }
        return r;
    }


    bool test_mm_andnot_si128(const int32_t *_a, const int32_t *_b)
    {
        bool r = true;
        __m128i a = test_mm_load_ps(_a);
        __m128i b = test_mm_load_ps(_b);
        __m128 fc = _mm_andnot_ps(*(const __m128 *)&a, *(const __m128 *)&b);
        __m128i c = *(const __m128i *)&fc;
        // now for the assertion...
        const uint32_t *ia = (const uint32_t *)&a;
        const uint32_t *ib = (const uint32_t *)&b;
        uint32_t r0 = ~ia[0] & ib[0];
        uint32_t r1 = ~ia[1] & ib[1];
        uint32_t r2 = ~ia[2] & ib[2];
        uint32_t r3 = ~ia[3] & ib[3];
        __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
        r = validateInt(c, r3, r2, r1, r0);
        if (r)
        {
            validateInt(ret, r3, r2, r1, r0);
        }
        return r;
    }

    bool test_mm_and_si128(const int32_t *_a, const int32_t *_b)
    {
        __m128i a = test_mm_load_ps(_a);
        __m128i b = test_mm_load_ps(_b);
        __m128 fc = _mm_and_ps(*(const __m128 *)&a, *(const __m128 *)&b);
        __m128i c = *(const __m128i *)&fc;
        // now for the assertion...
        const uint32_t *ia = (const uint32_t *)&a;
        const uint32_t *ib = (const uint32_t *)&b;
        uint32_t r0 = ia[0] & ib[0];
        uint32_t r1 = ia[1] & ib[1];
        uint32_t r2 = ia[2] & ib[2];
        uint32_t r3 = ia[3] & ib[3];
        __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
        bool r = validateInt(c, r3, r2, r1, r0);
        if (r)
        {
            r = validateInt(ret, r3, r2, r1, r0);
        }
        return r;
    }

    bool test_mm_or_si128(const int32_t *_a, const int32_t *_b)
    {
        __m128i a = test_mm_load_ps(_a);
        __m128i b = test_mm_load_ps(_b);
        __m128 fc = _mm_or_ps(*(const __m128 *)&a, *(const __m128 *)&b);
        __m128i c = *(const __m128i *)&fc;
        // now for the assertion...
        const uint32_t *ia = (const uint32_t *)&a;
        const uint32_t *ib = (const uint32_t *)&b;
        uint32_t r0 = ia[0] | ib[0];
        uint32_t r1 = ia[1] | ib[1];
        uint32_t r2 = ia[2] | ib[2];
        uint32_t r3 = ia[3] | ib[3];
        __m128i ret = test_mm_set_epi32(r3, r2, r1, r0);
        bool r = validateInt(c, r3, r2, r1, r0);
        if (r)
        {
            r = validateInt(ret, r3, r2, r1, r0);
        }
        return r;
    }

    bool test_mm_movemask_ps(const float *p)
    {
        int ret = 0;

        const uint32_t *ip = (const uint32_t *)p;
        if (ip[0] & 0x80000000)
        {
            ret |= 1;
        }
        if (ip[1] & 0x80000000)
        {
            ret |= 2;
        }
        if (ip[2] & 0x80000000)
        {
            ret |= 4;
        }
        if (ip[3] & 0x80000000)
        {
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
        // Test many permutations of the shuffle operation, including all permutations which have an optmized/custom implementation
        __m128 ret;
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 1, 2, 3));
        if (!validateFloat(ret, _b[0], _b[1], _a[2], _a[3]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 1, 0));
        if (!validateFloat(ret, _b[3], _b[2], _a[1], _a[0]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 1, 1));
        if (!validateFloat(ret, _b[0], _b[0], _a[1], _a[1]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 0, 2));
        if (!validateFloat(ret, _b[3], _b[1], _a[0], _a[2]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 0, 3, 2));
        if (!validateFloat(ret, _b[1], _b[0], _a[3], _a[2]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 3, 0, 1));
        if (!validateFloat(ret, _b[2], _b[3], _a[0], _a[1]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 2, 2));
        if (!validateFloat(ret, _b[0], _b[0], _a[2], _a[2]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 2, 0, 0));
        if (!validateFloat(ret, _b[2], _b[2], _a[0], _a[0]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 0, 2));
        if (!validateFloat(ret, _b[3], _b[2], _a[0], _a[2]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 1, 3, 3));
        if (!validateFloat(ret, _b[1], _b[1], _a[3], _a[3]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 1, 0));
        if (!validateFloat(ret, _b[2], _b[0], _a[1], _a[0]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 0, 1));
        if (!validateFloat(ret, _b[2], _b[0], _a[0], _a[1]))
        {
            isValid = false;
        }
        ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 3, 2));
        if (!validateFloat(ret, _b[2], _b[0], _a[3], _a[2]))
        {
            isValid = false;
        }

        return isValid;
    }

    bool test_mm_movemask_epi8(const int32_t *_a)
    {
        __m128i a = test_mm_load_ps(_a);

        const uint8_t *ip = (const uint8_t *)_a;
        int ret = 0;
        uint32_t mask = 1;
        for (uint32_t i = 0; i < 16; i++)
        {
            if (ip[i] & 0x80)
            {
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

        __m128i a = test_mm_load_ps((const int32_t *)_a);
        __m128i b = test_mm_load_ps((const int32_t *)_b);

        __m128i c = _mm_mullo_epi16(a, b);
        return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
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

        __m128i a = test_mm_load_ps((const int32_t *)_a);
        __m128i b = test_mm_load_ps((const int32_t *)_b);

        __m128i c = _mm_min_epi16(a, b);
        return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
    }

    bool test_mm_mulhi_epi16(const int16_t *_a, const int16_t *_b)
    {
        int16_t d[8];
        for (uint32_t i = 0; i < 8; i++)
        {
            int32_t m = (int32_t)_a[i] * (int32_t)_b[i];
            d[i] = (int16_t)(m >> 16);
        }

        __m128i a = test_mm_load_ps((const int32_t *)_a);
        __m128i b = test_mm_load_ps((const int32_t *)_b);

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
        __m128i iret = *(const __m128i *)&ret;
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
        __m128i iret = *(const __m128i *)&ret;
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
        __m128i iret = *(const __m128i *)&ret;
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
        __m128i iret = *(const __m128i *)&ret;
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
        __m128i iret = *(const __m128i *)&ret;
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
        if ( !isNANA &&  !isNANB)
        {
            ret = getNAN();
        }
        else
        {
            ret = 0.0f;
        }
        return ret;
    }

    bool test_mm_cmpord_ps(const float *_a, const float *_b)
    {
        __m128 a = test_mm_load_ps(_a);
        __m128 b = test_mm_load_ps(_b);

        float result[4];

        for (uint32_t i = 0; i < 4; i++)
        {
            result[i] = compord(_a[i], _b[i]);
        }

        __m128 ret = _mm_cmpord_ps(a, b);

        return validateFloat(ret, result[3], result[2], result[1], result[0]);
    }
//********************************************
    int32_t comilt_ss(float a, float b)
    {
        int32_t ret;

        bool isNANA = isNAN(a);
        bool isNANB = isNAN(b);
        if (!isNANA && !isNANB)
        {
            ret = a < b ? 1 : 0;
        }
        else
        {
            ret = 0;        // **NOTE** The documentation on MSDN is in error!  The actual hardware returns a 0, not a 1 if either of the values is a NAN!
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
//********************************************

//********************************************
    int32_t comigt_ss(float a, float b)
    {
        int32_t ret;

        bool isNANA = isNAN(a);
        bool isNANB = isNAN(b);
        if (!isNANA && !isNANB)
        {
            ret = a > b ? 1 : 0;
        }
        else
        {
            ret = 0;        // **NOTE** The documentation on MSDN is in error!  The actual hardware returns a 0, not a 1 if either of the values is a NAN!
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
//********************************************

//********************************************
    int32_t comile_ss(float a, float b)
    {
        int32_t ret;

        bool isNANA = isNAN(a);
        bool isNANB = isNAN(b);
        if (!isNANA && !isNANB)
        {
            ret = a <= b ? 1 : 0;
        }
        else
        {
            ret = 0;        // **NOTE** The documentation on MSDN is in error!  The actual hardware returns a 0, not a 1 if either of the values is a NAN!
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
//********************************************

//********************************************
    int32_t comige_ss(float a, float b)
    {
        int32_t ret;

        bool isNANA = isNAN(a);
        bool isNANB = isNAN(b);
        if (!isNANA && !isNANB)
        {
            ret = a >= b ? 1 : 0;
        }
        else
        {
            ret = 0;        // **NOTE** The documentation on MSDN is in error!  The actual hardware returns a 0, not a 1 if either of the values is a NAN!
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
    //********************************************

    //********************************************
    int32_t comieq_ss(float a, float b)
    {
        int32_t ret;

        bool isNANA = isNAN(a);
        bool isNANB = isNAN(b);
        if (!isNANA && !isNANB)
        {
            ret = a == b ? 1 : 0;
        }
        else
        {
            ret = 0;        // **NOTE** The documentation on MSDN is in error!  The actual hardware returns a 0, not a 1 if either of the values is a NAN!
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
    //********************************************


    //********************************************
    int32_t comineq_ss(float a, float b)
    {
        int32_t ret;

        bool isNANA = isNAN(a);
        bool isNANB = isNAN(b);
        if (!isNANA && !isNANB)
        {
            ret = a != b ? 1 : 0;
        }
        else
        {
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
    //********************************************




    bool test_mm_cvttps_epi32(const float *_a)
    {
        __m128 a = test_mm_load_ps(_a);
        int32_t trun[4];
        for (uint32_t i = 0; i < 4; i++)
        {
            trun[i] = (int32_t)_a[i];
        }

        __m128i ret = _mm_cvttps_epi32(a);
        return validateInt(ret, trun[3], trun[2], trun[1], trun[0]);
    }

    bool test_mm_cvtepi32_ps(const int32_t *_a)
    {
        __m128i a = test_mm_load_ps(_a);
        float trun[4];
        for (uint32_t i = 0; i < 4; i++)
        {
            trun[i] = (float)_a[i];
        }

        __m128 ret = _mm_cvtepi32_ps(a);
        return validateFloat(ret, trun[3], trun[2], trun[1], trun[0]);
    }

    // https://msdn.microsoft.com/en-us/library/xdc42k5e%28v=vs.90%29.aspx?f=255&MSPPError=-2147217396
    bool test_mm_cvtps_epi32(const float _a[4])
    {
        __m128 a = test_mm_load_ps(_a);
        int32_t trun[4];
        for (uint32_t i = 0; i < 4; i++)
        {
            trun[i] = (int32_t)(bankersRounding(_a[i]));
        }

        __m128i ret = _mm_cvtps_epi32(a);
        return validateInt(ret, trun[3], trun[2], trun[1], trun[0]);
    }


// Try 10,000 random floating point values for each test we run
#define MAX_TEST_VALUE 10000


class SSE2NEONTestImpl : public SSE2NEONTest
{
public:
    SSE2NEONTestImpl(void)
    {
        mTestFloatPointer1 = (float *)platformAlignedAlloc(sizeof(__m128));
        mTestFloatPointer2 = (float *)platformAlignedAlloc(sizeof(__m128));
        mTestIntPointer1 = (int32_t *)platformAlignedAlloc(sizeof(__m128i));
        mTestIntPointer2 = (int32_t *)platformAlignedAlloc(sizeof(__m128i));
        srand(0);
        for (uint32_t i = 0; i < MAX_TEST_VALUE; i++)
        {
            mTestFloats[i] = ranf(-100000, 100000);
            mTestInts[i] = (int32_t)ranf(-100000, 100000);
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
        bool ret = test_mm_store_ps(mTestFloatPointer1, mTestFloats[i], mTestFloats[i + 1], mTestFloats[i + 2], mTestFloats[i + 3]);
        if (ret)
        {
            ret = test_mm_store_ps(mTestFloatPointer2, mTestFloats[i + 4], mTestFloats[i + 5], mTestFloats[i + 6], mTestFloats[i + 7]);
        }
        return ret;
    }

    bool loadTestIntPointers(uint32_t i)
    {
        bool ret = test_mm_store_ps(mTestIntPointer1, mTestInts[i], mTestInts[i + 1], mTestInts[i + 2], mTestInts[i + 3]);
        if (ret)
        {
            ret = test_mm_store_ps(mTestIntPointer2, mTestInts[i + 4], mTestInts[i + 5], mTestInts[i + 6], mTestInts[i + 7]);
        }

        return ret;
    }

    bool runSingleTest(InstructionTest test,uint32_t i)
    {
        bool ret = true;

        switch ( test )
        {
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
                ret = test_mm_set_ps(mTestFloats[i], mTestFloats[i + 1], mTestFloats[i + 2], mTestFloats[i + 3]);
                break;
            case IT_MM_SET1_EPI32:
                ret = test_mm_set1_epi32(mTestInts[i]);
                break;
            case IT_MM_SET_EPI32:
                ret = testret_mm_set_epi32(mTestInts[i], mTestInts[i + 1], mTestInts[i + 2], mTestInts[i + 3]);
                break;
            case IT_MM_STORE_PS:
                ret = test_mm_store_ps(mTestIntPointer1, mTestInts[i], mTestInts[i + 1], mTestInts[i + 2], mTestInts[i + 3]);
                break;
            case IT_MM_LOAD1_PS:
                ret = test_mm_load1_ps(mTestFloatPointer1);
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
                ret = test_mm_mullo_epi16((const int16_t *)mTestIntPointer1, (const int16_t *)mTestIntPointer2);
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
                ret = test_mm_min_epi16((const int16_t *)mTestIntPointer1, (const int16_t *)mTestIntPointer2);
                break;
            case IT_MM_MULHI_EPI16:
                ret = test_mm_mulhi_epi16((const int16_t *)mTestIntPointer1, (const int16_t *)mTestIntPointer2);
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
                if (!ret)
                {
                    // Note to Alexander, you need to fix this.
                    ret = test_mm_comilt_ss(mTestFloatPointer1, mTestFloatPointer2);
                }
                break;
            case IT_MM_COMIGT_SS:
                ret = test_mm_comigt_ss(mTestFloatPointer1, mTestFloatPointer2);
                break;
            case IT_MM_COMILE_SS:
                ret = test_mm_comile_ss(mTestFloatPointer1, mTestFloatPointer2);
                if (!ret)
                {
                    // Note to Alexander, you need to fix this.
                    ret = test_mm_comile_ss(mTestFloatPointer1, mTestFloatPointer2);
                }
                break;
            case IT_MM_COMIGE_SS:
                ret = test_mm_comige_ss(mTestFloatPointer1, mTestFloatPointer2);
                break;
            case IT_MM_COMIEQ_SS:
                ret = test_mm_comieq_ss(mTestFloatPointer1, mTestFloatPointer2);
                if (!ret)
                {
                    // Note to Alexander, you need to fix this.
                    ret = test_mm_comieq_ss(mTestFloatPointer1, mTestFloatPointer2);
                }
                break;
            case IT_MM_COMINEQ_SS:
                ret = test_mm_comineq_ss(mTestFloatPointer1, mTestFloatPointer2);
                if (!ret)
                {
                    // Note to Alexander, you need to fix this.
                    ret = test_mm_comineq_ss(mTestFloatPointer1, mTestFloatPointer2);
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
            case IT_MM_ADD_EPI16:
                ret = true;
                break;
            case IT_MM_ADD_SS:
                ret = true;
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
        }


        return ret;
    }


    virtual bool runTest(InstructionTest test)
    {
        bool ret = true;


        // Test a whole bunch of values
        for (uint32_t i = 0; i < (MAX_TEST_VALUE - 8); i++)
        {
            ret = loadTestFloatPointers(i);	// Load some random float values
            if ( !ret ) break; // load test float failed??
            ret = loadTestIntPointers(i);	// load some random int values
            if ( !ret ) break; // load test float failed??
            // If we are testing the reciprocal, then invert the input data (easier for debugging)
            if ( test == IT_MM_RCP_PS )
            {

                mTestFloatPointer1[0] = 1.0f / mTestFloatPointer1[0];
                mTestFloatPointer1[1] = 1.0f / mTestFloatPointer1[1];
                mTestFloatPointer1[2] = 1.0f / mTestFloatPointer1[2];
                mTestFloatPointer1[3] = 1.0f / mTestFloatPointer1[3];
            }
            if ( test == IT_MM_CMPGE_PS || test == IT_MM_CMPLE_PS || test == IT_MM_CMPEQ_PS )
            {
               // Make sure at least one value is the same.
               mTestFloatPointer1[3] = mTestFloatPointer2[3];
            }

            if (test == IT_MM_CMPORD_PS || 
                test == IT_MM_COMILT_SS || 
                test == IT_MM_COMILE_SS ||
                test == IT_MM_COMIGE_SS ||
                test == IT_MM_COMIEQ_SS ||
                test == IT_MM_COMINEQ_SS ||
                test == IT_MM_COMIGT_SS) // if testing for NAN's make sure we have some nans
            {
                // One out of four times
                // Make sure a couple of values have NANs for testing purposes
                if ((rand() & 3) == 0)
                {
                    uint32_t r1 = rand() & 3;
                    uint32_t r2 = rand() & 3;
                    mTestFloatPointer1[r1] = getNAN();
                    mTestFloatPointer2[r2] = getNAN();
                }
            }

            // one out of every random 64 times or so, mix up the test floats to contain some integer values
            if ((rand() & 63) == 0)
            {
                uint32_t option = rand() & 3;
                switch (option)
                {
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
                    case 1:
                        {
                            uint32_t index = rand() & 3;
                            mTestFloatPointer1[index] = float(mTestIntPointer1[index]);
                            index = rand() & 3;
                            mTestFloatPointer2[index] = float(mTestIntPointer2[index]);
                        }
                        break;
                    case 2:
                        {
                            uint32_t index1 = rand() & 3;
                            uint32_t index2 = rand() & 3;
                            mTestFloatPointer1[index1] = float(mTestIntPointer1[index1]);
                            mTestFloatPointer1[index2] = float(mTestIntPointer1[index2]);
                            index1 = rand() & 3;
                            index2 = rand() & 3;
                            mTestFloatPointer1[index1] = float(mTestIntPointer1[index1]);
                            mTestFloatPointer1[index2] = float(mTestIntPointer1[index2]);
                        }
                        break;
                    case 3:
                        mTestFloatPointer1[0] = float(mTestIntPointer1[0]);
                        mTestFloatPointer1[1] = float(mTestIntPointer1[1]);
                        mTestFloatPointer1[2] = float(mTestIntPointer1[2]);
                        mTestFloatPointer1[3] = float(mTestIntPointer1[3]);
                        break;
                }
                if ((rand() & 3) == 0) // one out of 4 times, make halves
                {
                    for (uint32_t j = 0; j < 4; j++)
                    {
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
                if (!ok)
                {
                    printf("Debug me");
                }
            }
#endif
            ret = runSingleTest(test,i);
            if ( !ret ) // the test failed...
            {
                // Set a breakpoint here if you want to step through the failure case in the debugger
                ret = runSingleTest(test,i);
                break;
            }
        }
        return ret;
    }

    virtual void release(void)
    {
        delete this;
    }

    float       *mTestFloatPointer1;
    float       *mTestFloatPointer2;
    int32_t     *mTestIntPointer1;
    int32_t     *mTestIntPointer2;
    float       mTestFloats[MAX_TEST_VALUE];
    int32_t     mTestInts[MAX_TEST_VALUE];
};

SSE2NEONTest *SSE2NEONTest::create(void)
{
    SSE2NEONTestImpl *st = new SSE2NEONTestImpl;
    return static_cast<SSE2NEONTest *>(st);
}

} // end of SSE2NEON namespace
