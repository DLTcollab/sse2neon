#include "impl.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include "binding.h"

#include <inttypes.h>

// Try 10,000 random floating point values for each test we run
#define MAX_TEST_VALUE 10000

/* Pattern Matching for C macros.
 * https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
 */

/* catenate */
#define PRIMITIVE_CAT(a, ...) a##__VA_ARGS__

#define IIF(c) PRIMITIVE_CAT(IIF_, c)
/* run the 2nd parameter */
#define IIF_0(t, ...) __VA_ARGS__
/* run the 1st parameter */
#define IIF_1(t, ...) t

// This program a set of unit tests to ensure that each SSE call provide the
// output we expect.  If this fires an assert, then something didn't match up.
//
// Functions with `test_` prefix will be called in runSingleTest.
namespace SSE2NEON
{
// Forward declaration
class SSE2NEONTestImpl : public SSE2NEONTest
{
public:
    SSE2NEONTestImpl(void);
    result_t loadTestFloatPointers(uint32_t i);
    result_t loadTestIntPointers(uint32_t i);
    result_t runSingleTest(InstructionTest test, uint32_t i);

    float *mTestFloatPointer1;
    float *mTestFloatPointer2;
    int32_t *mTestIntPointer1;
    int32_t *mTestIntPointer2;
    float mTestFloats[MAX_TEST_VALUE];
    int32_t mTestInts[MAX_TEST_VALUE];

    virtual ~SSE2NEONTestImpl(void)
    {
        platformAlignedFree(mTestFloatPointer1);
        platformAlignedFree(mTestFloatPointer2);
        platformAlignedFree(mTestIntPointer1);
        platformAlignedFree(mTestIntPointer2);
    }
    virtual void release(void) { delete this; }
    virtual result_t runTest(InstructionTest test)
    {
        result_t ret = TEST_SUCCESS;

        // Test a whole bunch of values
        for (uint32_t i = 0; i < (MAX_TEST_VALUE - 8); i++) {
            ret = loadTestFloatPointers(i);  // Load some random float values
            if (ret == TEST_FAIL)
                break;                     // load test float failed??
            ret = loadTestIntPointers(i);  // load some random int values
            if (ret == TEST_FAIL)
                break;  // load test float failed??
            // If we are testing the reciprocal, then invert the input data
            // (easier for debugging)
            if (test == it_mm_rcp_ps) {
                mTestFloatPointer1[0] = 1.0f / mTestFloatPointer1[0];
                mTestFloatPointer1[1] = 1.0f / mTestFloatPointer1[1];
                mTestFloatPointer1[2] = 1.0f / mTestFloatPointer1[2];
                mTestFloatPointer1[3] = 1.0f / mTestFloatPointer1[3];
            }
            if (test == it_mm_cmpge_ps || test == it_mm_cmpge_ss ||
                test == it_mm_cmple_ps || test == it_mm_cmple_ss ||
                test == it_mm_cmpeq_ps || test == it_mm_cmpeq_ss) {
                // Make sure at least one value is the same.
                mTestFloatPointer1[3] = mTestFloatPointer2[3];
            }

            if (test == it_mm_cmpord_ps || test == it_mm_cmpord_ss ||
                test == it_mm_cmpunord_ps || test == it_mm_cmpunord_ss ||
                test == it_mm_cmpeq_ps || test == it_mm_cmpeq_ss ||
                test == it_mm_cmpge_ps || test == it_mm_cmpge_ss ||
                test == it_mm_cmpgt_ps || test == it_mm_cmpgt_ss ||
                test == it_mm_cmple_ps || test == it_mm_cmple_ss ||
                test == it_mm_cmplt_ps || test == it_mm_cmplt_ss ||
                test == it_mm_cmpneq_ps || test == it_mm_cmpneq_ss ||
                test == it_mm_cmpnge_ps || test == it_mm_cmpnge_ss ||
                test == it_mm_cmpngt_ps || test == it_mm_cmpngt_ss ||
                test == it_mm_cmpnle_ps || test == it_mm_cmpnle_ss ||
                test == it_mm_cmpnlt_ps || test == it_mm_cmpnlt_ss ||
                test == it_mm_comieq_ss || test == it_mm_ucomieq_ss ||
                test == it_mm_comige_ss || test == it_mm_ucomige_ss ||
                test == it_mm_comigt_ss || test == it_mm_ucomigt_ss ||
                test == it_mm_comile_ss || test == it_mm_ucomile_ss ||
                test == it_mm_comilt_ss || test == it_mm_ucomilt_ss ||
                test == it_mm_comineq_ss || test == it_mm_ucomineq_ss) {
                // Make sure the NaN values are included in the testing
                // one out of four times.
                if ((rand() & 3) == 0) {
                    uint32_t r1 = rand() & 3;
                    uint32_t r2 = rand() & 3;
                    mTestFloatPointer1[r1] = nanf("");
                    mTestFloatPointer2[r2] = nanf("");
                }
            }

            if (test == it_mm_cmpord_pd || test == it_mm_cmpord_sd ||
                test == it_mm_cmpunord_pd || test == it_mm_cmpunord_sd ||
                test == it_mm_cmpeq_pd || test == it_mm_cmpeq_sd ||
                test == it_mm_cmpge_pd || test == it_mm_cmpge_sd ||
                test == it_mm_cmpgt_pd || test == it_mm_cmpgt_sd ||
                test == it_mm_cmple_pd || test == it_mm_cmple_sd ||
                test == it_mm_cmplt_pd || test == it_mm_cmplt_sd ||
                test == it_mm_cmpneq_pd || test == it_mm_cmpneq_sd ||
                test == it_mm_cmpnge_pd || test == it_mm_cmpnge_sd ||
                test == it_mm_cmpngt_pd || test == it_mm_cmpngt_sd ||
                test == it_mm_cmpnle_pd || test == it_mm_cmpnle_sd ||
                test == it_mm_cmpnlt_pd || test == it_mm_cmpnlt_sd ||
                test == it_mm_comieq_sd || test == it_mm_ucomieq_sd ||
                test == it_mm_comige_sd || test == it_mm_ucomige_sd ||
                test == it_mm_comigt_sd || test == it_mm_ucomigt_sd ||
                test == it_mm_comile_sd || test == it_mm_ucomile_sd ||
                test == it_mm_comilt_sd || test == it_mm_ucomilt_sd ||
                test == it_mm_comineq_sd || test == it_mm_ucomineq_sd) {
                // Make sure the NaN values are included in the testing
                // one out of four times.
                if ((rand() & 3) == 0) {
                    // FIXME:
                    // The argument "0xFFFFFFFFFFFF" is a tricky workaround to
                    // set the NaN value for doubles. The code is not intuitive
                    // and should be fixed in the future.
                    uint32_t r1 = ((rand() & 1) << 1) + 1;
                    uint32_t r2 = ((rand() & 1) << 1) + 1;
                    mTestFloatPointer1[r1] = nanf("0xFFFFFFFFFFFF");
                    mTestFloatPointer2[r2] = nanf("0xFFFFFFFFFFFF");
                }
            }

#if SSE2NEON_PRECISE_MINMAX
            if (test == it_mm_max_ps || test == it_mm_max_ss ||
                test == it_mm_min_ps || test == it_mm_min_ss) {
                // Make sure the NaN values are included in the testing
                // one out of four times.
                if ((rand() & 3) == 0) {
                    uint32_t r1 = rand() & 3;
                    uint32_t r2 = rand() & 3;
                    mTestFloatPointer1[r1] = nanf("");
                    mTestFloatPointer2[r2] = nanf("");
                }
            }

            if (test == it_mm_max_pd || test == it_mm_max_sd ||
                test == it_mm_min_pd || test == it_mm_min_sd) {
                // Make sure the NaN values are included in the testing
                // one out of four times.
                if ((rand() & 3) == 0) {
                    // FIXME:
                    // The argument "0xFFFFFFFFFFFF" is a tricky workaround to
                    // set the NaN value for doubles. The code is not intuitive
                    // and should be fixed in the future.
                    uint32_t r1 = ((rand() & 1) << 1) + 1;
                    uint32_t r2 = ((rand() & 1) << 1) + 1;
                    mTestFloatPointer1[r1] = nanf("0xFFFFFFFFFFFF");
                    mTestFloatPointer2[r2] = nanf("0xFFFFFFFFFFFF");
                }
            }
#endif

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

            ret = runSingleTest(test, i);
            if (ret == TEST_FAIL)  // the test failed...
            {
                // Set a breakpoint here if you want to step through the failure
                // case in the debugger
                ret = runSingleTest(test, i);
                break;
            }
        }
        return ret;
    }
};

const char *instructionString[] = {INTRIN_FOREACH(STR)};

// Produce rounding which is the same as SSE instructions with _MM_ROUND_NEAREST
// rounding mode
static inline float bankersRounding(float val)
{
    if (val < 0)
        return -bankersRounding(-val);

    float ret;
    float roundDown = floorf(val);  // Round down value
    float roundUp = ceilf(val);     // Round up value
    float diffDown = val - roundDown;
    float diffUp = roundUp - val;

    if (diffDown < diffUp) {
        /* If it's closer to the round down value, then use it */
        ret = roundDown;
    } else if (diffDown > diffUp) {
        /* If it's closer to the round up value, then use it */
        ret = roundUp;
    } else {
        /* If it's equidistant between round up and round down value, pick the
         * one which is an even number */
        float half = roundDown / 2;
        if (half != floorf(half)) {
            /* If the round down value is odd, return the round up value */
            ret = roundUp;
        } else {
            /* If the round up value is odd, return the round down value */
            ret = roundDown;
        }
    }
    return ret;
}

static inline double bankersRounding(double val)
{
    if (val < 0)
        return -bankersRounding(-val);

    double ret;
    double roundDown = floor(val);  // Round down value
    double roundUp = ceil(val);     // Round up value
    double diffDown = val - roundDown;
    double diffUp = roundUp - val;

    if (diffDown < diffUp) {
        /* If it's closer to the round down value, then use it */
        ret = roundDown;
    } else if (diffDown > diffUp) {
        /* If it's closer to the round up value, then use it */
        ret = roundUp;
    } else {
        /* If it's equidistant between round up and round down value, pick the
         * one which is an even number */
        double half = roundDown / 2;
        if (half != floor(half)) {
            /* If the round down value is odd, return the round up value */
            ret = roundUp;
        } else {
            /* If the round up value is odd, return the round down value */
            ret = roundDown;
        }
    }
    return ret;
}

static float ranf(void)
{
    uint32_t ir = rand() & 0x7FFF;
    return (float) ir * (1.0f / 32768.0f);
}

static float ranf(float low, float high)
{
    return ranf() * (high - low) + low;
}

// Enable the tests which are using the macro of another tests
result_t test_mm_slli_si128(const SSE2NEONTestImpl &impl, uint32_t iter);
result_t test_mm_srli_si128(const SSE2NEONTestImpl &impl, uint32_t iter);
result_t test_mm_shuffle_pi16(const SSE2NEONTestImpl &impl, uint32_t iter);

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_set_epi32`.
__m128i do_mm_set_epi32(int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    validateInt32(a, w, z, y, x);
    return a;
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to load __m64 data.
template <class T>
__m64 load_m64(const T *p)
{
    return *((const __m64 *) p);
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_load_ps`.
template <class T>
__m128 load_m128(const T *p)
{
    return _mm_loadu_ps((const float *) p);
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_load_ps`.
template <class T>
__m128i load_m128i(const T *p)
{
    __m128 a = _mm_loadu_ps((const float *) p);
    __m128i ia = *(const __m128i *) &a;
    return ia;
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_load_pd`.
template <class T>
__m128d load_m128d(const T *p)
{
    return _mm_loadu_pd((const double *) p);
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_store_ps`.
result_t do_mm_store_ps(float *p, float x, float y, float z, float w)
{
    __m128 a = _mm_set_ps(x, y, z, w);
    _mm_store_ps(p, a);
    ASSERT_RETURN(p[0] == w);
    ASSERT_RETURN(p[1] == z);
    ASSERT_RETURN(p[2] == y);
    ASSERT_RETURN(p[3] == x);
    return TEST_SUCCESS;
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_store_ps`.
result_t do_mm_store_ps(int32_t *p, int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    _mm_store_ps((float *) p, *(const __m128 *) &a);
    ASSERT_RETURN(p[0] == w);
    ASSERT_RETURN(p[1] == z);
    ASSERT_RETURN(p[2] == y);
    ASSERT_RETURN(p[3] == x);
    return TEST_SUCCESS;
}

float cmp_noNaN(float a, float b)
{
    return (!isnan(a) && !isnan(b)) ? ALL_BIT_1_32 : 0.0f;
}

double cmp_noNaN(double a, double b)
{
    return (!isnan(a) && !isnan(b)) ? ALL_BIT_1_64 : 0.0f;
}

float cmp_hasNaN(float a, float b)
{
    return (isnan(a) || isnan(b)) ? ALL_BIT_1_32 : 0.0f;
}

double cmp_hasNaN(double a, double b)
{
    return (isnan(a) || isnan(b)) ? ALL_BIT_1_64 : 0.0f;
}

int32_t comilt_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isnan(a);
    bool isNANB = isnan(b);
    if (!isNANA && !isNANB) {
        ret = a < b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

int32_t comigt_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isnan(a);
    bool isNANB = isnan(b);
    if (!isNANA && !isNANB) {
        ret = a > b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

int32_t comile_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isnan(a);
    bool isNANB = isnan(b);
    if (!isNANA && !isNANB) {
        ret = a <= b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

int32_t comige_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isnan(a);
    bool isNANB = isnan(b);
    if (!isNANA && !isNANB) {
        ret = a >= b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

int32_t comieq_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isnan(a);
    bool isNANB = isnan(b);
    if (!isNANA && !isNANB) {
        ret = a == b ? 1 : 0;
    } else {
        ret = 0;  // **NOTE** The documentation on MSDN is in error!  The actual
                  // hardware returns a 0, not a 1 if either of the values is a
                  // NAN!
    }
    return ret;
}

int32_t comineq_ss(float a, float b)
{
    int32_t ret;

    bool isNANA = isnan(a);
    bool isNANB = isnan(b);
    if (!isNANA && !isNANB) {
        ret = a != b ? 1 : 0;
    } else {
        ret = 1;
    }
    return ret;
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
    crc = canonical_crc32_u32((uint32_t) (crc), v & 0xffffffff);
    crc = canonical_crc32_u32((uint32_t) (crc), (v >> 32) & 0xffffffff);
    return crc;
}

static const uint8_t crypto_aes_sbox[256] = {
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
    0xb0, 0x54, 0xbb, 0x16,
};

#define XT(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))
inline __m128i aesenc_128_reference(__m128i a, __m128i b)
{
    uint8_t i, t, u, v[4][4];
    for (i = 0; i < 16; ++i) {
        v[((i / 4) + 4 - (i % 4)) % 4][i % 4] =
            crypto_aes_sbox[((SIMDVec *) &a)->m128_u8[i]];
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

inline __m128i aesenclast_128_reference(__m128i s, __m128i rk)
{
    uint8_t i, v[4][4];
    for (i = 0; i < 16; ++i)
        v[((i / 4) + 4 - (i % 4)) % 4][i % 4] =
            crypto_aes_sbox[((SIMDVec *) &s)->m128_u8[i]];
    for (i = 0; i < 16; ++i)
        ((SIMDVec *) &s)->m128_u8[i] =
            v[i / 4][i % 4] ^ ((SIMDVec *) &rk)->m128_u8[i];
    return s;
}

static inline uint32_t sub_word(uint32_t key)
{
    return (crypto_aes_sbox[key >> 24] << 24) |
           (crypto_aes_sbox[(key >> 16) & 0xff] << 16) |
           (crypto_aes_sbox[(key >> 8) & 0xff] << 8) |
           crypto_aes_sbox[key & 0xff];
}

// Rotates right (circular right shift) value by "amount" positions
static inline uint32_t rotr(uint32_t value, uint32_t amount)
{
    return (value >> amount) | (value << ((32 - amount) & 31));
}

inline __m128i aeskeygenassist_128_reference(__m128i a, const int rcon)
{
    const uint32_t X1 = sub_word(_mm_cvtsi128_si32(_mm_shuffle_epi32(a, 0x55)));
    const uint32_t X3 = sub_word(_mm_cvtsi128_si32(_mm_shuffle_epi32(a, 0xFF)));
    return _mm_set_epi32(rotr(X3, 8) ^ rcon, X3, rotr(X1, 8) ^ rcon, X1);
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

/* MMX */
result_t test_mm_empty(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_SUCCESS;
}

/* SSE */
result_t test_mm_add_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float dx = _a[0] + _b[0];
    float dy = _a[1] + _b[1];
    float dz = _a[2] + _b[2];
    float dw = _a[3] + _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_add_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_add_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f0 = _a[0] + _b[0];
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_add_ss(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_and_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_and_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ia[0] & ib[0];
    uint32_t r1 = ia[1] & ib[1];
    uint32_t r2 = ia[2] & ib[2];
    uint32_t r3 = ia[3] & ib[3];
    __m128i ret = do_mm_set_epi32(r3, r2, r1, r0);
    result_t r = validateInt32(*(const __m128i *) &c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

// r0 := ~a0 & b0
// r1 := ~a1 & b1
// r2 := ~a2 & b2
// r3 := ~a3 & b3
result_t test_mm_andnot_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    result_t r = TEST_FAIL;
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_andnot_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ~ia[0] & ib[0];
    uint32_t r1 = ~ia[1] & ib[1];
    uint32_t r2 = ~ia[2] & ib[2];
    uint32_t r3 = ~ia[3] & ib[3];
    __m128i ret = do_mm_set_epi32(r3, r2, r1, r0);
    r = validateInt32(*(const __m128i *) &c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

result_t test_mm_avg_pu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;
    uint16_t d0 = (_a[0] + _b[0] + 1) >> 1;
    uint16_t d1 = (_a[1] + _b[1] + 1) >> 1;
    uint16_t d2 = (_a[2] + _b[2] + 1) >> 1;
    uint16_t d3 = (_a[3] + _b[3] + 1) >> 1;

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_avg_pu16(a, b);

    return validateUInt16(c, d0, d1, d2, d3);
}

result_t test_mm_avg_pu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    uint8_t d0 = (_a[0] + _b[0] + 1) >> 1;
    uint8_t d1 = (_a[1] + _b[1] + 1) >> 1;
    uint8_t d2 = (_a[2] + _b[2] + 1) >> 1;
    uint8_t d3 = (_a[3] + _b[3] + 1) >> 1;
    uint8_t d4 = (_a[4] + _b[4] + 1) >> 1;
    uint8_t d5 = (_a[5] + _b[5] + 1) >> 1;
    uint8_t d6 = (_a[6] + _b[6] + 1) >> 1;
    uint8_t d7 = (_a[7] + _b[7] + 1) >> 1;

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_avg_pu8(a, b);

    return validateUInt8(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_cmpeq_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result[4];
    result[0] = _a[0] == _b[0] ? -1 : 0;
    result[1] = _a[1] == _b[1] ? -1 : 0;
    result[2] = _a[2] == _b[2] ? -1 : 0;
    result[3] = _a[3] == _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpeq_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpeq_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = _a[0] == _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpeq_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpge_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result[4];
    result[0] = _a[0] >= _b[0] ? -1 : 0;
    result[1] = _a[1] >= _b[1] ? -1 : 0;
    result[2] = _a[2] >= _b[2] ? -1 : 0;
    result[3] = _a[3] >= _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpge_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpge_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = _a[0] >= _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpge_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpgt_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result[4];
    result[0] = _a[0] > _b[0] ? -1 : 0;
    result[1] = _a[1] > _b[1] ? -1 : 0;
    result[2] = _a[2] > _b[2] ? -1 : 0;
    result[3] = _a[3] > _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpgt_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpgt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = _a[0] > _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpgt_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmple_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result[4];
    result[0] = _a[0] <= _b[0] ? -1 : 0;
    result[1] = _a[1] <= _b[1] ? -1 : 0;
    result[2] = _a[2] <= _b[2] ? -1 : 0;
    result[3] = _a[3] <= _b[3] ? -1 : 0;

    __m128 ret = _mm_cmple_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmple_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = _a[0] <= _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmple_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmplt_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result[4];
    result[0] = _a[0] < _b[0] ? -1 : 0;
    result[1] = _a[1] < _b[1] ? -1 : 0;
    result[2] = _a[2] < _b[2] ? -1 : 0;
    result[3] = _a[3] < _b[3] ? -1 : 0;

    __m128 ret = _mm_cmplt_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmplt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = _a[0] < _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmplt_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpneq_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result[4];
    result[0] = _a[0] != _b[0] ? -1 : 0;
    result[1] = _a[1] != _b[1] ? -1 : 0;
    result[2] = _a[2] != _b[2] ? -1 : 0;
    result[3] = _a[3] != _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpneq_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpneq_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = _a[0] != _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpneq_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnge_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = !(_a[0] >= _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = !(_a[1] >= _b[1]) ? ALL_BIT_1_32 : 0;
    result[2] = !(_a[2] >= _b[2]) ? ALL_BIT_1_32 : 0;
    result[3] = !(_a[3] >= _b[3]) ? ALL_BIT_1_32 : 0;

    __m128 ret = _mm_cmpnge_ps(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnge_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = !(_a[0] >= _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpnge_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpngt_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = !(_a[0] > _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = !(_a[1] > _b[1]) ? ALL_BIT_1_32 : 0;
    result[2] = !(_a[2] > _b[2]) ? ALL_BIT_1_32 : 0;
    result[3] = !(_a[3] > _b[3]) ? ALL_BIT_1_32 : 0;

    __m128 ret = _mm_cmpngt_ps(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpngt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = !(_a[0] > _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpngt_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnle_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = !(_a[0] <= _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = !(_a[1] <= _b[1]) ? ALL_BIT_1_32 : 0;
    result[2] = !(_a[2] <= _b[2]) ? ALL_BIT_1_32 : 0;
    result[3] = !(_a[3] <= _b[3]) ? ALL_BIT_1_32 : 0;

    __m128 ret = _mm_cmpnle_ps(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnle_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = !(_a[0] <= _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpnle_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnlt_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = !(_a[0] < _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = !(_a[1] < _b[1]) ? ALL_BIT_1_32 : 0;
    result[2] = !(_a[2] < _b[2]) ? ALL_BIT_1_32 : 0;
    result[3] = !(_a[3] < _b[3]) ? ALL_BIT_1_32 : 0;

    __m128 ret = _mm_cmpnlt_ps(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnlt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = !(_a[0] < _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpnlt_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpord_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];

    for (uint32_t i = 0; i < 4; i++) {
        result[i] = cmp_noNaN(_a[i], _b[i]);
    }

    __m128 ret = _mm_cmpord_ps(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpord_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = cmp_noNaN(_a[0], _b[0]);
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpord_ss(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpunord_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];

    for (uint32_t i = 0; i < 4; i++) {
        result[i] = cmp_hasNaN(_a[i], _b[i]);
    }

    __m128 ret = _mm_cmpunord_ps(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpunord_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = cmp_hasNaN(_a[0], _b[0]);
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpunord_ss(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_comieq_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comieq_ss correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__)
    return TEST_UNIMPL;
#else
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result = comieq_ss(_a[0], _b[0]);
    int32_t ret = _mm_comieq_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
#endif
}

result_t test_mm_comige_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result = comige_ss(_a[0], _b[0]);
    int32_t ret = _mm_comige_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_comigt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result = comigt_ss(_a[0], _b[0]);
    int32_t ret = _mm_comigt_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_comile_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comile_ss correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__)
    return TEST_UNIMPL;
#else
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result = comile_ss(_a[0], _b[0]);
    int32_t ret = _mm_comile_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
#endif
}

result_t test_mm_comilt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comilt_ss correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__)
    return TEST_UNIMPL;
#else
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result = comilt_ss(_a[0], _b[0]);

    int32_t ret = _mm_comilt_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
#endif
}

result_t test_mm_comineq_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comineq_ss correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__)
    return TEST_UNIMPL;
#else
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    int32_t result = comineq_ss(_a[0], _b[0]);
    int32_t ret = _mm_comineq_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
#endif
}

result_t test_mm_cvt_pi2ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t *_b = impl.mTestIntPointer2;

    float dx = (float) _b[0];
    float dy = (float) _b[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = load_m128(_a);
    __m64 b = load_m64(_b);
    __m128 c = _mm_cvt_pi2ps(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvt_ps2pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    for (int idx = 0; idx < 2; idx++) {
        switch (iter & 0x3) {
        case 0:
            _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
            d[idx] = (int32_t) (bankersRounding(_a[idx]));
            break;
        case 1:
            _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
            d[idx] = (int32_t) (floorf(_a[idx]));
            break;
        case 2:
            _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
            d[idx] = (int32_t) (ceilf(_a[idx]));
            break;
        case 3:
            _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
            d[idx] = (int32_t) (_a[idx]);
            break;
        }
    }

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvt_ps2pi(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvt_si2ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t b = *impl.mTestIntPointer2;

    float dx = (float) b;
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_cvt_si2ss(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvt_ss2si(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d0;

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d0 = (int32_t) (bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d0 = (int32_t) (floorf(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d0 = (int32_t) (ceilf(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d0 = (int32_t) (_a[0]);
        break;
    }

    __m128 a = load_m128(_a);
    int32_t ret = _mm_cvt_ss2si(a);
    return ret == d0 ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtpi16_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _a[2];
    float dw = (float) _a[3];

    __m64 a = load_m64(_a);
    __m128 c = _mm_cvtpi16_ps(a);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtpi32_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    float dx = (float) _b[0];
    float dy = (float) _b[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = load_m128(_a);
    __m64 b = load_m64(_b);
    __m128 c = _mm_cvtpi32_ps(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtpi32x2_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _b[0];
    float dw = (float) _b[1];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m128 c = _mm_cvtpi32x2_ps(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtpi8_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _a[2];
    float dw = (float) _a[3];

    __m64 a = load_m64(_a);
    __m128 c = _mm_cvtpi8_ps(a);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtps_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int16_t rnd[4];

    for (int i = 0; i < 4; i++) {
        if ((float) INT16_MAX <= _a[i] && _a[i] <= (float) INT32_MAX) {
            rnd[i] = INT16_MAX;
        } else if (INT16_MIN < _a[i] && _a[i] < INT16_MAX) {
            switch (iter & 0x3) {
            case 0:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
                rnd[i] = (int16_t) bankersRounding(_a[i]);
                break;
            case 1:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
                rnd[i] = (int16_t) floorf(_a[i]);
                break;
            case 2:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
                rnd[i] = (int16_t) ceilf(_a[i]);
                break;
            case 3:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
                rnd[i] = (int16_t) _a[i];
                break;
            }
        } else {
            rnd[i] = INT16_MIN;
        }
    }

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvtps_pi16(a);
    return validateInt16(ret, rnd[0], rnd[1], rnd[2], rnd[3]);
}

result_t test_mm_cvtps_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d[0] = (int32_t) bankersRounding(_a[0]);
        d[1] = (int32_t) bankersRounding(_a[1]);
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d[0] = (int32_t) floorf(_a[0]);
        d[1] = (int32_t) floorf(_a[1]);
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d[0] = (int32_t) ceilf(_a[0]);
        d[1] = (int32_t) ceilf(_a[1]);
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d[0] = (int32_t) _a[0];
        d[1] = (int32_t) _a[1];
        break;
    }

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvtps_pi32(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvtps_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int8_t rnd[4];

    for (int i = 0; i < 4; i++) {
        if ((float) INT8_MAX <= _a[i] && _a[i] <= (float) INT32_MAX) {
            rnd[i] = INT8_MAX;
        } else if (INT8_MIN < _a[i] && _a[i] < INT8_MAX) {
            switch (iter & 0x3) {
            case 0:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
                rnd[i] = (int8_t) bankersRounding(_a[i]);
                break;
            case 1:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
                rnd[i] = (int8_t) floorf(_a[i]);
                break;
            case 2:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
                rnd[i] = (int8_t) ceilf(_a[i]);
                break;
            case 3:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
                rnd[i] = (int8_t) _a[i];
                break;
            }
        } else {
            rnd[i] = INT8_MIN;
        }
    }

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvtps_pi8(a);
    return validateInt8(ret, rnd[0], rnd[1], rnd[2], rnd[3], 0, 0, 0, 0);
}

result_t test_mm_cvtpu16_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _a[2];
    float dw = (float) _a[3];

    __m64 a = load_m64(_a);
    __m128 c = _mm_cvtpu16_ps(a);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtpu8_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _a[2];
    float dw = (float) _a[3];

    __m64 a = load_m64(_a);
    __m128 c = _mm_cvtpu8_ps(a);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtsi32_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t b = *impl.mTestIntPointer2;

    float dx = (float) b;
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_cvtsi32_ss(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtsi64_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int64_t b = *(int64_t *) impl.mTestIntPointer2;

    float dx = (float) b;
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_cvtsi64_ss(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtss_f32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    float f = _a[0];

    __m128 a = load_m128(_a);
    float c = _mm_cvtss_f32(a);

    return f == c ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtss_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    int32_t d0;
    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d0 = (int32_t) (bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d0 = (int32_t) (floorf(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d0 = (int32_t) (ceilf(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d0 = (int32_t) (_a[0]);
        break;
    }

    __m128 a = load_m128(_a);
    int32_t ret = _mm_cvtss_si32(a);

    return ret == d0 ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtss_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    int64_t d0;
    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d0 = (int64_t) (bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d0 = (int64_t) (floorf(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d0 = (int64_t) (ceilf(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d0 = (int64_t) (_a[0]);
        break;
    }

    __m128 a = load_m128(_a);
    int64_t ret = _mm_cvtss_si64(a);

    return ret == d0 ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtt_ps2pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    d[0] = (int32_t) _a[0];
    d[1] = (int32_t) _a[1];

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvtt_ps2pi(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvtt_ss2si(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = load_m128(_a);
    int ret = _mm_cvtt_ss2si(a);

    return ret == (int32_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvttps_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    d[0] = (int32_t) _a[0];
    d[1] = (int32_t) _a[1];

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvttps_pi32(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvttss_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = load_m128(_a);
    int ret = _mm_cvttss_si32(a);

    return ret == (int32_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvttss_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = load_m128(_a);
    int64_t ret = _mm_cvttss_si64(a);

    return ret == (int64_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_div_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float f0 = _a[0] / _b[0];
    float f1 = _a[1] / _b[1];
    float f2 = _a[2] / _b[2];
    float f3 = _a[3] / _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_div_ps(a, b);

#if defined(__arm__) && !defined(__aarch64__)
    // The implementation of "_mm_div_ps()" on ARM 32bit doesn't use "DIV"
    // instruction directly, instead it uses "FRECPE" instruction to approximate
    // it. Therefore, the precision is not as small as other architecture
    return validateFloatError(c, f0, f1, f2, f3, 0.00001f);
#else
    return validateFloat(c, f0, f1, f2, f3);
#endif
}

result_t test_mm_div_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float d0 = _a[0] / _b[0];
    float d1 = _a[1];
    float d2 = _a[2];
    float d3 = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_div_ss(a, b);

#if defined(__arm__) && !defined(__aarch64__)
    // The implementation of "_mm_div_ps()" on ARM 32bit doesn't use "DIV"
    // instruction directly, instead it uses "FRECPE" instruction to approximate
    // it. Therefore, the precision is not as small as other architecture
    return validateFloatError(c, d0, d1, d2, d3, 0.00001f);
#else
    return validateFloat(c, d0, d1, d2, d3);
#endif
}

result_t test_mm_extract_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME GCC has bug on `_mm_extract_pi16` intrinsics. We will enable this
    // test when GCC fix this bug.
    // see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98495 for more
    // information
#if defined(__clang__)
    uint64_t *_a = (uint64_t *) impl.mTestIntPointer1;
    const int idx = iter & 0x3;

    __m64 a = load_m64(_a);
    int c;
    switch (idx) {
    case 0:
        c = _mm_extract_pi16(a, 0);
        break;
    case 1:
        c = _mm_extract_pi16(a, 1);
        break;
    case 2:
        c = _mm_extract_pi16(a, 2);
        break;
    case 3:
        c = _mm_extract_pi16(a, 3);
        break;
    }

    ASSERT_RETURN((uint64_t) c == ((*_a >> (idx * 16)) & 0xFFFF));
    ASSERT_RETURN(0 == ((uint64_t) c & 0xFFFF0000));
    return TEST_SUCCESS;
#else
    return TEST_UNIMPL;
#endif
}

result_t test_mm_malloc(const SSE2NEONTestImpl &impl, uint32_t iter);
result_t test_mm_free(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    /* We verify _mm_malloc first, and there is no need to check _mm_free . */
    return test_mm_malloc(impl, iter);
}

result_t test_mm_get_flush_zero_mode(const SSE2NEONTestImpl &impl,
                                     uint32_t iter)
{
    int res_flush_zero_on, res_flush_zero_off;
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    res_flush_zero_on = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    res_flush_zero_off = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_OFF;

    return (res_flush_zero_on && res_flush_zero_off) ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_get_rounding_mode(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int res_toward_zero, res_to_neg_inf, res_to_pos_inf, res_nearest;
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
    res_toward_zero = _MM_GET_ROUNDING_MODE() == _MM_ROUND_TOWARD_ZERO ? 1 : 0;
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    res_to_neg_inf = _MM_GET_ROUNDING_MODE() == _MM_ROUND_DOWN ? 1 : 0;
    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    res_to_pos_inf = _MM_GET_ROUNDING_MODE() == _MM_ROUND_UP ? 1 : 0;
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    res_nearest = _MM_GET_ROUNDING_MODE() == _MM_ROUND_NEAREST ? 1 : 0;

    if (res_toward_zero && res_to_neg_inf && res_to_pos_inf && res_nearest) {
        return TEST_SUCCESS;
    } else {
        return TEST_FAIL;
    }
}

result_t test_mm_getcsr(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // store original csr value for post test restoring
    unsigned int originalCsr = _mm_getcsr();

    unsigned int roundings[] = {_MM_ROUND_TOWARD_ZERO, _MM_ROUND_DOWN,
                                _MM_ROUND_UP, _MM_ROUND_NEAREST};
    for (size_t i = 0; i < sizeof(roundings) / sizeof(roundings[0]); i++) {
        _mm_setcsr(_mm_getcsr() | roundings[i]);
        if ((_mm_getcsr() & roundings[i]) != roundings[i]) {
            return TEST_FAIL;
        }
    }

    // restore original csr value for remaining tests
    _mm_setcsr(originalCsr);

    return TEST_SUCCESS;
}

result_t test_mm_insert_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t insert = (int16_t) impl.mTestInts[iter];
    const int imm8 = 2;

    int16_t d[4];
    for (int i = 0; i < 4; i++) {
        d[i] = _a[i];
    }
    d[imm8] = insert;

    __m64 a = load_m64(_a);
    __m64 b = _mm_insert_pi16(a, insert, imm8);

    return validateInt16(b, d[0], d[1], d[2], d[3]);
}

result_t test_mm_load_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_load_ps(addr);

    return validateFloat(ret, addr[0], addr[1], addr[2], addr[3]);
}

result_t test_mm_load_ps1(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_load_ps1(addr);

    return validateFloat(ret, addr[0], addr[0], addr[0], addr[0]);
}

result_t test_mm_load_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_load_ss(addr);

    return validateFloat(ret, addr[0], 0, 0, 0);
}

result_t test_mm_load1_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p = impl.mTestFloatPointer1;
    __m128 a = _mm_load1_ps(p);
    return validateFloat(a, p[0], p[0], p[0], p[0]);
}

result_t test_mm_loadh_pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p1 = impl.mTestFloatPointer1;
    const float *p2 = impl.mTestFloatPointer2;
    const __m64 *b = (const __m64 *) p2;
    __m128 a = _mm_load_ps(p1);
    __m128 c = _mm_loadh_pi(a, b);

    return validateFloat(c, p1[0], p1[1], p2[0], p2[1]);
}

result_t test_mm_loadl_pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p1 = impl.mTestFloatPointer1;
    const float *p2 = impl.mTestFloatPointer2;
    __m128 a = _mm_load_ps(p1);
    const __m64 *b = (const __m64 *) p2;
    __m128 c = _mm_loadl_pi(a, b);

    return validateFloat(c, p2[0], p2[1], p1[2], p1[3]);
}

result_t test_mm_loadr_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_loadr_ps(addr);

    return validateFloat(ret, addr[3], addr[2], addr[1], addr[0]);
}

result_t test_mm_loadu_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_loadu_ps(addr);

    return validateFloat(ret, addr[0], addr[1], addr[2], addr[3]);
}

result_t test_mm_loadu_si16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_loadu_si16. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
    return TEST_UNIMPL;
#else
    const int16_t *addr = (const int16_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_loadu_si16((const void *) addr);

    return validateInt16(ret, addr[0], 0, 0, 0, 0, 0, 0, 0);
#endif
}

result_t test_mm_loadu_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // Versions of GCC prior to 9 do not implement intrinsic function
    // _mm_loadu_si64. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=78782
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ < 9)
    return TEST_UNIMPL;
#else
    const int64_t *addr = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_loadu_si64((const void *) addr);

    return validateInt64(ret, addr[0], 0);
#endif
}

result_t test_mm_malloc(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const size_t *a = (const size_t *) impl.mTestIntPointer1;
    const size_t *b = (const size_t *) impl.mTestIntPointer2;
    size_t size = *a % (1024 * 16) + 1;
    size_t align = 2 << (*b % 5);

    void *p = _mm_malloc(size, align);
    if (!p)
        return TEST_FAIL;
    result_t res = (((uintptr_t) p % align) == 0) ? TEST_SUCCESS : TEST_FAIL;
    _mm_free(p);
    return res;
}

result_t test_mm_maskmove_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_mask = (const uint8_t *) impl.mTestIntPointer2;
    char mem_addr[16];

    const __m64 *a = (const __m64 *) _a;
    const __m64 *mask = (const __m64 *) _mask;
    _mm_maskmove_si64(*a, *mask, (char *) mem_addr);

    for (int i = 0; i < 8; i++) {
        if (_mask[i] >> 7) {
            ASSERT_RETURN(_a[i] == (uint8_t) mem_addr[i]);
        }
    }

    return TEST_SUCCESS;
}

result_t test_m_maskmovq(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_maskmove_si64(impl, iter);
}

result_t test_mm_max_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t c[4];

    c[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] > _b[3] ? _a[3] : _b[3];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_max_pi16(a, b);
    return validateInt16(ret, c[0], c[1], c[2], c[3]);
}

result_t test_mm_max_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float c[4];

    c[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 ret = _mm_max_ps(a, b);
    return validateFloat(ret, c[0], c[1], c[2], c[3]);
}

result_t test_mm_max_pu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    uint8_t c[8];

    c[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] > _b[3] ? _a[3] : _b[3];
    c[4] = _a[4] > _b[4] ? _a[4] : _b[4];
    c[5] = _a[5] > _b[5] ? _a[5] : _b[5];
    c[6] = _a[6] > _b[6] ? _a[6] : _b[6];
    c[7] = _a[7] > _b[7] ? _a[7] : _b[7];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_max_pu8(a, b);
    return validateUInt8(ret, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}

result_t test_mm_max_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f0 = _a[0] > _b[0] ? _a[0] : _b[0];
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_max_ss(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_min_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t c[4];

    c[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] < _b[3] ? _a[3] : _b[3];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_min_pi16(a, b);
    return validateInt16(ret, c[0], c[1], c[2], c[3]);
}

result_t test_mm_min_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float c[4];

    c[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 ret = _mm_min_ps(a, b);
    return validateFloat(ret, c[0], c[1], c[2], c[3]);
}

result_t test_mm_min_pu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    uint8_t c[8];

    c[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] < _b[3] ? _a[3] : _b[3];
    c[4] = _a[4] < _b[4] ? _a[4] : _b[4];
    c[5] = _a[5] < _b[5] ? _a[5] : _b[5];
    c[6] = _a[6] < _b[6] ? _a[6] : _b[6];
    c[7] = _a[7] < _b[7] ? _a[7] : _b[7];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_min_pu8(a, b);
    return validateUInt8(ret, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}

result_t test_mm_min_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float c;

    c = _a[0] < _b[0] ? _a[0] : _b[0];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 ret = _mm_min_ss(a, b);

    return validateFloat(ret, c, _a[1], _a[2], _a[3]);
}

result_t test_mm_move_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    float result[4];
    result[0] = b[0];
    result[1] = a[1];
    result[2] = a[2];
    result[3] = a[3];

    __m128 ret = _mm_move_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_movehl_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _b[2];
    float f1 = _b[3];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 ret = _mm_movehl_ps(a, b);

    return validateFloat(ret, f0, f1, f2, f3);
}

result_t test_mm_movelh_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _a[0];
    float f1 = _a[1];
    float f2 = _b[0];
    float f3 = _b[1];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 ret = _mm_movelh_ps(a, b);

    return validateFloat(ret, f0, f1, f2, f3);
}

result_t test_mm_movemask_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    unsigned int _c = 0;
    for (int i = 0; i < 8; i++) {
        if (_a[i] & 0x80) {
            _c |= (1 << i);
        }
    }

    const __m64 *a = (const __m64 *) _a;
    int c = _mm_movemask_pi8(*a);

    ASSERT_RETURN((unsigned int) c == _c);
    return TEST_SUCCESS;
}

result_t test_mm_movemask_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p = impl.mTestFloatPointer1;
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
    __m128 a = load_m128(p);
    int val = _mm_movemask_ps(a);
    return val == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_mul_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float dx = _a[0] * _b[0];
    float dy = _a[1] * _b[1];
    float dz = _a[2] * _b[2];
    float dw = _a[3] * _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_mul_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_mul_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float dx = _a[0] * _b[0];
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_mul_ss(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_mulhi_pu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;
    uint16_t d[4];
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t m = (uint32_t) _a[i] * (uint32_t) _b[i];
        d[i] = (uint16_t) (m >> 16);
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_mulhi_pu16(a, b);
    return validateUInt16(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_or_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_or_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ia[0] | ib[0];
    uint32_t r1 = ia[1] | ib[1];
    uint32_t r2 = ia[2] | ib[2];
    uint32_t r3 = ia[3] | ib[3];
    __m128i ret = do_mm_set_epi32(r3, r2, r1, r0);
    result_t r = validateInt32(*(const __m128i *) &c, r0, r1, r2, r3);
    if (r)
        r = validateInt32(ret, r0, r1, r2, r3);
    return r;
}

result_t test_m_pavgb(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_avg_pu8(impl, iter);
}

result_t test_m_pavgw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_avg_pu16(impl, iter);
}

result_t test_m_pextrw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_extract_pi16(impl, iter);
}

result_t test_m_pinsrw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_insert_pi16(impl, iter);
}

result_t test_m_pmaxsw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_max_pi16(impl, iter);
}

result_t test_m_pmaxub(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_max_pu8(impl, iter);
}

result_t test_m_pminsw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_min_pi16(impl, iter);
}

result_t test_m_pminub(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_min_pu8(impl, iter);
}

result_t test_m_pmovmskb(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_movemask_pi8(impl, iter);
}

result_t test_m_pmulhuw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_mulhi_pu16(impl, iter);
}

result_t test_mm_prefetch(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    typedef struct {
        __m128 a;
        float r[4];
    } prefetch_test_t;
    prefetch_test_t test_vec[8] = {
        {
            _mm_set_ps(-0.1f, 0.2f, 0.3f, 0.4f),
            {0.4f, 0.3f, 0.2f, -0.1f},
        },
        {
            _mm_set_ps(0.5f, 0.6f, -0.7f, -0.8f),
            {-0.8f, -0.7f, 0.6f, 0.5f},
        },
        {
            _mm_set_ps(0.9f, 0.10f, -0.11f, 0.12f),
            {0.12f, -0.11f, 0.10f, 0.9f},
        },
        {
            _mm_set_ps(-1.1f, -2.1f, -3.1f, -4.1f),
            {-4.1f, -3.1f, -2.1f, -1.1f},
        },
        {
            _mm_set_ps(100.0f, -110.0f, 120.0f, -130.0f),
            {-130.0f, 120.0f, -110.0f, 100.0f},
        },
        {
            _mm_set_ps(200.5f, 210.5f, -220.5f, 230.5f),
            {995.74f, -93.04f, 144.03f, 902.50f},
        },
        {
            _mm_set_ps(10.11f, -11.12f, -12.13f, 13.14f),
            {13.14f, -12.13f, -11.12f, 10.11f},
        },
        {
            _mm_set_ps(10.1f, -20.2f, 30.3f, 40.4f),
            {40.4f, 30.3f, -20.2f, 10.1f},
        },
    };

    for (size_t i = 0; i < (sizeof(test_vec) / (sizeof(test_vec[0]))); i++) {
        _mm_prefetch(((const char *) &test_vec[i].a), _MM_HINT_T0);
        _mm_prefetch(((const char *) &test_vec[i].a), _MM_HINT_T1);
        _mm_prefetch(((const char *) &test_vec[i].a), _MM_HINT_T2);
        _mm_prefetch(((const char *) &test_vec[i].a), _MM_HINT_NTA);
    }

    return TEST_SUCCESS;
}

result_t test_m_psadbw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    uint16_t d = 0;
    for (int i = 0; i < 8; i++) {
        d += abs(_a[i] - _b[i]);
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _m_psadbw(a, b);
    return validateUInt16(c, d, 0, 0, 0);
}

result_t test_m_pshufw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_shuffle_pi16(impl, iter);
}

result_t test_mm_rcp_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    float dx = 1.0f / _a[0];
    float dy = 1.0f / _a[1];
    float dz = 1.0f / _a[2];
    float dw = 1.0f / _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_rcp_ps(a);
    return validateFloatEpsilon(c, dx, dy, dz, dw, 300.0f);
}

result_t test_mm_rcp_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    float dx = 1.0f / _a[0];
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_rcp_ss(a);
    return validateFloatEpsilon(c, dx, dy, dz, dw, 300.0f);
}

result_t test_mm_rsqrt_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    float f0 = 1 / sqrt(_a[0]);
    float f1 = 1 / sqrt(_a[1]);
    float f2 = 1 / sqrt(_a[2]);
    float f3 = 1 / sqrt(_a[3]);

    __m128 a = load_m128(_a);
    __m128 c = _mm_rsqrt_ps(a);

    // Here, we ensure `_mm_rsqrt_ps()`'s error is under 1% compares to the C
    // implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.01f);
}

result_t test_mm_rsqrt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    float f0 = 1 / sqrt(_a[0]);
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_rsqrt_ss(a);

    // Here, we ensure `_mm_rsqrt_ps()`'s error is under 1% compares to the C
    // implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.01f);
}

result_t test_mm_sad_pu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    uint16_t d = 0;
    for (int i = 0; i < 8; i++) {
        d += abs(_a[i] - _b[i]);
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_sad_pu8(a, b);
    return validateUInt16(c, d, 0, 0, 0);
}

result_t test_mm_set_flush_zero_mode(const SSE2NEONTestImpl &impl,
                                     uint32_t iter)
{
    // TODO:
    // After the behavior of denormal number and flush zero mode is fully
    // investigated, the testing would be added.
    return TEST_UNIMPL;
}

result_t test_mm_set_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float x = impl.mTestFloats[iter];
    float y = impl.mTestFloats[iter + 1];
    float z = impl.mTestFloats[iter + 2];
    float w = impl.mTestFloats[iter + 3];
    __m128 a = _mm_set_ps(x, y, z, w);
    return validateFloat(a, w, z, y, x);
}

result_t test_mm_set_ps1(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float a = impl.mTestFloats[iter];

    __m128 ret = _mm_set_ps1(a);

    return validateFloat(ret, a, a, a, a);
}

result_t test_mm_set_rounding_mode(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    result_t res_toward_zero, res_to_neg_inf, res_to_pos_inf, res_nearest;

    __m128 a = load_m128(_a);
    __m128 b, c;

    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
    b = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    c = _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    res_toward_zero = validate128(c, b);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    b = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    c = _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    res_to_neg_inf = validate128(c, b);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    b = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    c = _mm_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    res_to_pos_inf = validate128(c, b);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    b = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    c = _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    res_nearest = validate128(c, b);

    if (res_toward_zero == TEST_SUCCESS && res_to_neg_inf == TEST_SUCCESS &&
        res_to_pos_inf == TEST_SUCCESS && res_nearest == TEST_SUCCESS) {
        return TEST_SUCCESS;
    } else {
        return TEST_FAIL;
    }
}

result_t test_mm_set_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float a = impl.mTestFloats[iter];
    __m128 c = _mm_set_ss(a);
    return validateFloat(c, a, 0, 0, 0);
}

result_t test_mm_set1_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float w = impl.mTestFloats[iter];
    __m128 a = _mm_set1_ps(w);
    return validateFloat(a, w, w, w, w);
}

result_t test_mm_setcsr(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_set_rounding_mode(impl, iter);
}

result_t test_mm_setr_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float x = impl.mTestFloats[iter];
    float y = impl.mTestFloats[iter + 1];
    float z = impl.mTestFloats[iter + 2];
    float w = impl.mTestFloats[iter + 3];

    __m128 ret = _mm_setr_ps(w, z, y, x);

    return validateFloat(ret, w, z, y, x);
}

result_t test_mm_setzero_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128 a = _mm_setzero_ps();
    return validateFloat(a, 0, 0, 0, 0);
}

result_t test_mm_sfence(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

result_t test_mm_shuffle_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int32_t imm = 73;

    __m64 a = load_m64(_a);

    int16_t d0 = _a[imm & 0x3];
    int16_t d1 = _a[(imm >> 2) & 0x3];
    int16_t d2 = _a[(imm >> 4) & 0x3];
    int16_t d3 = _a[(imm >> 6) & 0x3];

    __m64 d = _mm_shuffle_pi16(a, imm);

    return validateInt16(d, d0, d1, d2, d3);
}

// Note, NEON does not have a general purpose shuffled command like SSE.
// When invoking this method, there is special code for a number of the most
// common shuffle permutations
result_t test_mm_shuffle_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    result_t isValid = TEST_SUCCESS;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    // Test many permutations of the shuffle operation, including all
    // permutations which have an optimized/customized implementation
    __m128 ret;
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 1, 2, 3));
    if (!validateFloat(ret, _a[3], _a[2], _b[1], _b[0])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 1, 0));
    if (!validateFloat(ret, _a[0], _a[1], _b[2], _b[3])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 1, 1));
    if (!validateFloat(ret, _a[1], _a[1], _b[0], _b[0])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 0, 2));
    if (!validateFloat(ret, _a[2], _a[0], _b[1], _b[3])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 0, 3, 2));
    if (!validateFloat(ret, _a[2], _a[3], _b[0], _b[1])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 3, 0, 1));
    if (!validateFloat(ret, _a[1], _a[0], _b[3], _b[2])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 2, 2));
    if (!validateFloat(ret, _a[2], _a[2], _b[0], _b[0])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 2, 0, 0));
    if (!validateFloat(ret, _a[0], _a[0], _b[2], _b[2])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 0, 2));
    if (!validateFloat(ret, _a[2], _a[0], _b[2], _b[3])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 1, 3, 3));
    if (!validateFloat(ret, _a[3], _a[3], _b[1], _b[1])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 1, 0));
    if (!validateFloat(ret, _a[0], _a[1], _b[0], _b[2])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 0, 1));
    if (!validateFloat(ret, _a[1], _a[0], _b[0], _b[2])) {
        isValid = TEST_FAIL;
    }
    ret = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 3, 2));
    if (!validateFloat(ret, _a[2], _a[3], _b[0], _b[2])) {
        isValid = TEST_FAIL;
    }

    return isValid;
}

result_t test_mm_sqrt_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    float f0 = sqrt(_a[0]);
    float f1 = sqrt(_a[1]);
    float f2 = sqrt(_a[2]);
    float f3 = sqrt(_a[3]);

    __m128 a = load_m128(_a);
    __m128 c = _mm_sqrt_ps(a);

    // Here, we ensure `_mm_sqrt_ps()`'s error is under 1% compares to the C
    // implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.01f);
}

result_t test_mm_sqrt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    float f0 = sqrt(_a[0]);
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_sqrt_ss(a);

    // Here, we ensure `_mm_sqrt_ps()`'s error is under 1% compares to the C
    // implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.01f);
}

result_t test_mm_store_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t *p = impl.mTestIntPointer1;
    int32_t x = impl.mTestInts[iter];
    int32_t y = impl.mTestInts[iter + 1];
    int32_t z = impl.mTestInts[iter + 2];
    int32_t w = impl.mTestInts[iter + 3];
    __m128i a = _mm_set_epi32(x, y, z, w);
    _mm_store_ps((float *) p, *(const __m128 *) &a);
    ASSERT_RETURN(p[0] == w);
    ASSERT_RETURN(p[1] == z);
    ASSERT_RETURN(p[2] == y);
    ASSERT_RETURN(p[3] == x);
    return TEST_SUCCESS;
}

result_t test_mm_store_ps1(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float *p = impl.mTestFloatPointer1;
    float d[4];

    __m128 a = load_m128(p);
    _mm_store_ps1(d, a);

    ASSERT_RETURN(d[0] == *p);
    ASSERT_RETURN(d[1] == *p);
    ASSERT_RETURN(d[2] == *p);
    ASSERT_RETURN(d[3] == *p);
    return TEST_SUCCESS;
}

result_t test_mm_store_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float x = impl.mTestFloats[iter];
    float p[4];

    __m128 a = _mm_set_ss(x);
    _mm_store_ss(p, a);
    ASSERT_RETURN(p[0] == x);
    return TEST_SUCCESS;
}

result_t test_mm_store1_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float *p = impl.mTestFloatPointer1;
    float d[4];

    __m128 a = load_m128(p);
    _mm_store1_ps(d, a);

    ASSERT_RETURN(d[0] == *p);
    ASSERT_RETURN(d[1] == *p);
    ASSERT_RETURN(d[2] == *p);
    ASSERT_RETURN(d[3] == *p);
    return TEST_SUCCESS;
}

result_t test_mm_storeh_pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p = impl.mTestFloatPointer1;
    float d[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    __m128 a = _mm_load_ps(p);
    __m64 *b = (__m64 *) d;

    _mm_storeh_pi(b, a);
    ASSERT_RETURN(d[0] == p[2]);
    ASSERT_RETURN(d[1] == p[3]);
    ASSERT_RETURN(d[2] == 3.0f);
    ASSERT_RETURN(d[3] == 4.0f);
    return TEST_SUCCESS;
}

result_t test_mm_storel_pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p = impl.mTestFloatPointer1;
    float d[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    __m128 a = _mm_load_ps(p);
    __m64 *b = (__m64 *) d;

    _mm_storel_pi(b, a);
    ASSERT_RETURN(d[0] == p[0]);
    ASSERT_RETURN(d[1] == p[1]);
    ASSERT_RETURN(d[2] == 3.0f);
    ASSERT_RETURN(d[3] == 4.0f);
    return TEST_SUCCESS;
}

result_t test_mm_storer_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float *p = impl.mTestFloatPointer1;
    float d[4];

    __m128 a = load_m128(p);
    _mm_storer_ps(d, a);

    ASSERT_RETURN(d[0] == p[3]);
    ASSERT_RETURN(d[1] == p[2]);
    ASSERT_RETURN(d[2] == p[1]);
    ASSERT_RETURN(d[3] == p[0]);
    return TEST_SUCCESS;
}

result_t test_mm_storeu_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float *_a = impl.mTestFloatPointer1;
    float f[4];
    __m128 a = _mm_load_ps(_a);

    _mm_storeu_ps(f, a);
    return validateFloat(a, f[0], f[1], f[2], f[3]);
}

result_t test_mm_storeu_si16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_storeu_si16. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
    return TEST_UNIMPL;
#else
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i b;
    __m128i a = load_m128i(_a);
    _mm_storeu_si16(&b, a);
    int16_t *_b = (int16_t *) &b;
    int16_t *_c = (int16_t *) &a;
    return validateInt16(b, _c[0], _b[1], _b[2], _b[3], _b[4], _b[5], _b[6],
                         _b[7]);
#endif
}

result_t test_mm_storeu_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // Versions of GCC prior to 9 do not implement intrinsic function
    // _mm_storeu_si64. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=87558
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ < 9)
    return TEST_UNIMPL;
#else
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i b;
    __m128i a = load_m128i(_a);
    _mm_storeu_si64(&b, a);
    int64_t *_b = (int64_t *) &b;
    int64_t *_c = (int64_t *) &a;
    return validateInt64(b, _c[0], _b[1]);
#endif
}

result_t test_mm_stream_pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    __m64 a = load_m64(_a);
    __m64 p;

    _mm_stream_pi(&p, a);
    return validateInt64(p, _a[0]);
}

result_t test_mm_stream_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    __m128 a = load_m128(_a);
    alignas(16) float p[4];

    _mm_stream_ps(p, a);
    ASSERT_RETURN(p[0] == _a[0]);
    ASSERT_RETURN(p[1] == _a[1]);
    ASSERT_RETURN(p[2] == _a[2]);
    ASSERT_RETURN(p[3] == _a[3]);
    return TEST_SUCCESS;
}

result_t test_mm_sub_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float dx = _a[0] - _b[0];
    float dy = _a[1] - _b[1];
    float dz = _a[2] - _b[2];
    float dw = _a[3] - _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_sub_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_sub_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float dx = _a[0] - _b[0];
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_sub_ss(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_ucomieq_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_ucomieq_ss is equal to _mm_comieq_ss
    return test_mm_comieq_ss(impl, iter);
}

result_t test_mm_ucomige_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_ucomige_ss is equal to _mm_comige_ss
    return test_mm_comige_ss(impl, iter);
}

result_t test_mm_ucomigt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_ucomigt_ss is equal to _mm_comigt_ss
    return test_mm_comigt_ss(impl, iter);
}

result_t test_mm_ucomile_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_ucomile_ss is equal to _mm_comile_ss
    return test_mm_comile_ss(impl, iter);
}

result_t test_mm_ucomilt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_ucomilt_ss is equal to _mm_comilt_ss
    return test_mm_comilt_ss(impl, iter);
}

result_t test_mm_ucomineq_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_ucomineq_ss is equal to _mm_comineq_ss
    return test_mm_comineq_ss(impl, iter);
}

result_t test_mm_undefined_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128 a = _mm_undefined_ps();
    a = _mm_xor_ps(a, a);
    return validateFloat(a, 0, 0, 0, 0);
}

result_t test_mm_unpackhi_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float *_a = impl.mTestFloatPointer1;
    float *_b = impl.mTestFloatPointer1;

    float f0 = _a[2];
    float f1 = _b[2];
    float f2 = _a[3];
    float f3 = _b[3];

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_unpackhi_ps(a, b);
    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_unpacklo_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float *_a = impl.mTestFloatPointer1;
    float *_b = impl.mTestFloatPointer1;

    float f0 = _a[0];
    float f1 = _b[0];
    float f2 = _a[1];
    float f3 = _b[1];

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_unpacklo_ps(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_xor_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestFloatPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestFloatPointer2;

    int32_t d0 = _a[0] ^ _b[0];
    int32_t d1 = _a[1] ^ _b[1];
    int32_t d2 = _a[2] ^ _b[2];
    int32_t d3 = _a[3] ^ _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_xor_ps(a, b);

    return validateFloat(c, *((float *) &d0), *((float *) &d1),
                         *((float *) &d2), *((float *) &d3));
}

/* SSE2 */
result_t test_mm_add_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    int16_t d0 = _a[0] + _b[0];
    int16_t d1 = _a[1] + _b[1];
    int16_t d2 = _a[2] + _b[2];
    int16_t d3 = _a[3] + _b[3];
    int16_t d4 = _a[4] + _b[4];
    int16_t d5 = _a[5] + _b[5];
    int16_t d6 = _a[6] + _b[6];
    int16_t d7 = _a[7] + _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_add_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_add_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t dx = _a[0] + _b[0];
    int32_t dy = _a[1] + _b[1];
    int32_t dz = _a[2] + _b[2];
    int32_t dw = _a[3] + _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_add_epi32(a, b);
    return validateInt32(c, dx, dy, dz, dw);
}

result_t test_mm_add_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t d0 = _a[0] + _b[0];
    int64_t d1 = _a[1] + _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_add_epi64(a, b);

    return validateInt64(c, d0, d1);
}

result_t test_mm_add_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_add_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_add_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] + _b[0];
    double d1 = _a[1] + _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_add_pd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_add_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] + _b[0];
    double d1 = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_add_sd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_add_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t d0 = _a[0] + _b[0];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_add_si64(a, b);

    return validateInt64(c, d0);
}

result_t test_mm_adds_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_adds_epi16(a, b);
    return validateInt16(c, (int16_t) d0, (int16_t) d1, (int16_t) d2,
                         (int16_t) d3, (int16_t) d4, (int16_t) d5, (int16_t) d6,
                         (int16_t) d7);
}

result_t test_mm_adds_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;

    int16_t d[16];
    for (int i = 0; i < 16; i++) {
        d[i] = (int16_t) _a[i] + (int16_t) _b[i];
        if (d[i] > 127)
            d[i] = 127;
        if (d[i] < -128)
            d[i] = -128;
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_adds_epi8(a, b);

    return validateInt8(
        c, (int8_t) d[0], (int8_t) d[1], (int8_t) d[2], (int8_t) d[3],
        (int8_t) d[4], (int8_t) d[5], (int8_t) d[6], (int8_t) d[7],
        (int8_t) d[8], (int8_t) d[9], (int8_t) d[10], (int8_t) d[11],
        (int8_t) d[12], (int8_t) d[13], (int8_t) d[14], (int8_t) d[15]);
}

result_t test_mm_adds_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint32_t max = 0xFFFF;
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;

    uint16_t d0 =
        (uint32_t) _a[0] + (uint32_t) _b[0] > max ? max : _a[0] + _b[0];
    uint16_t d1 =
        (uint32_t) _a[1] + (uint32_t) _b[1] > max ? max : _a[1] + _b[1];
    uint16_t d2 =
        (uint32_t) _a[2] + (uint32_t) _b[2] > max ? max : _a[2] + _b[2];
    uint16_t d3 =
        (uint32_t) _a[3] + (uint32_t) _b[3] > max ? max : _a[3] + _b[3];
    uint16_t d4 =
        (uint32_t) _a[4] + (uint32_t) _b[4] > max ? max : _a[4] + _b[4];
    uint16_t d5 =
        (uint32_t) _a[5] + (uint32_t) _b[5] > max ? max : _a[5] + _b[5];
    uint16_t d6 =
        (uint32_t) _a[6] + (uint32_t) _b[6] > max ? max : _a[6] + _b[6];
    uint16_t d7 =
        (uint32_t) _a[7] + (uint32_t) _b[7] > max ? max : _a[7] + _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_adds_epu16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_adds_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_adds_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_and_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestFloatPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestFloatPointer2;

    int64_t d0 = _a[0] & _b[0];
    int64_t d1 = _a[1] & _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_and_pd(a, b);

    return validateDouble(c, *((double *) &d0), *((double *) &d1));
}

result_t test_mm_and_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128 fc = _mm_and_ps(*(const __m128 *) &a, *(const __m128 *) &b);
    __m128i c = *(const __m128i *) &fc;
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ia[0] & ib[0];
    uint32_t r1 = ia[1] & ib[1];
    uint32_t r2 = ia[2] & ib[2];
    uint32_t r3 = ia[3] & ib[3];
    __m128i ret = do_mm_set_epi32(r3, r2, r1, r0);
    result_t r = validateInt32(c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

result_t test_mm_andnot_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_andnot_pd(a, b);

    // Take AND operation a complement of 'a' and 'b'. Bitwise operations are
    // not allowed on float/double datatype, so 'a' and 'b' are calculated in
    // uint64_t datatype.
    const uint64_t *ia = (const uint64_t *) &a;
    const uint64_t *ib = (const uint64_t *) &b;
    uint64_t r0 = ~ia[0] & ib[0];
    uint64_t r1 = ~ia[1] & ib[1];
    return validateUInt64(*(const __m128i *) &c, r0, r1);
}

result_t test_mm_andnot_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    result_t r = TEST_SUCCESS;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128 fc = _mm_andnot_ps(*(const __m128 *) &a, *(const __m128 *) &b);
    __m128i c = *(const __m128i *) &fc;
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ~ia[0] & ib[0];
    uint32_t r1 = ~ia[1] & ib[1];
    uint32_t r2 = ~ia[2] & ib[2];
    uint32_t r3 = ~ia[3] & ib[3];
    __m128i ret = do_mm_set_epi32(r3, r2, r1, r0);
    r = validateInt32(c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

result_t test_mm_avg_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    uint16_t d0 = ((uint16_t) _a[0] + (uint16_t) _b[0] + 1) >> 1;
    uint16_t d1 = ((uint16_t) _a[1] + (uint16_t) _b[1] + 1) >> 1;
    uint16_t d2 = ((uint16_t) _a[2] + (uint16_t) _b[2] + 1) >> 1;
    uint16_t d3 = ((uint16_t) _a[3] + (uint16_t) _b[3] + 1) >> 1;
    uint16_t d4 = ((uint16_t) _a[4] + (uint16_t) _b[4] + 1) >> 1;
    uint16_t d5 = ((uint16_t) _a[5] + (uint16_t) _b[5] + 1) >> 1;
    uint16_t d6 = ((uint16_t) _a[6] + (uint16_t) _b[6] + 1) >> 1;
    uint16_t d7 = ((uint16_t) _a[7] + (uint16_t) _b[7] + 1) >> 1;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_avg_epu16(a, b);
    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_avg_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
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
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_avg_epu8(a, b);
    return validateUInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                         d12, d13, d14, d15);
}

result_t test_mm_bslli_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_slli_si128(impl, iter);
}

result_t test_mm_bsrli_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_srli_si128(impl, iter);
}

result_t test_mm_castpd_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const __m128d a = load_m128d(_a);
    const __m128 _c = load_m128(_a);

    __m128 r = _mm_castpd_ps(a);

    return validate128(r, _c);
}

result_t test_mm_castpd_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const __m128d a = load_m128d(_a);
    const __m128i *_c = (const __m128i *) _a;

    __m128i r = _mm_castpd_si128(a);

    return validate128(r, *_c);
}

result_t test_mm_castps_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const __m128 a = load_m128(_a);
    const __m128d *_c = (const __m128d *) _a;

    __m128d r = _mm_castps_pd(a);

    return validate128(r, *_c);
}

result_t test_mm_castps_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    const __m128i *_c = (const __m128i *) _a;

    const __m128 a = load_m128(_a);
    __m128i r = _mm_castps_si128(a);

    return validate128(r, *_c);
}

result_t test_mm_castsi128_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;

    const __m128d *_c = (const __m128d *) _a;

    const __m128i a = load_m128i(_a);
    __m128d r = _mm_castsi128_pd(a);

    return validate128(r, *_c);
}

result_t test_mm_castsi128_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;

    const __m128 *_c = (const __m128 *) _a;

    const __m128i a = load_m128i(_a);
    __m128 r = _mm_castsi128_ps(a);

    return validate128(r, *_c);
}

result_t test_mm_clflush(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpeq_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d0 = (_a[0] == _b[0]) ? ~UINT16_C(0) : 0x0;
    int16_t d1 = (_a[1] == _b[1]) ? ~UINT16_C(0) : 0x0;
    int16_t d2 = (_a[2] == _b[2]) ? ~UINT16_C(0) : 0x0;
    int16_t d3 = (_a[3] == _b[3]) ? ~UINT16_C(0) : 0x0;
    int16_t d4 = (_a[4] == _b[4]) ? ~UINT16_C(0) : 0x0;
    int16_t d5 = (_a[5] == _b[5]) ? ~UINT16_C(0) : 0x0;
    int16_t d6 = (_a[6] == _b[6]) ? ~UINT16_C(0) : 0x0;
    int16_t d7 = (_a[7] == _b[7]) ? ~UINT16_C(0) : 0x0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpeq_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_cmpeq_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;

    int32_t d0 = (_a[0] == _b[0]) ? ~UINT32_C(0) : 0x0;
    int32_t d1 = (_a[1] == _b[1]) ? ~UINT32_C(0) : 0x0;
    int32_t d2 = (_a[2] == _b[2]) ? ~UINT32_C(0) : 0x0;
    int32_t d3 = (_a[3] == _b[3]) ? ~UINT32_C(0) : 0x0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpeq_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_cmpeq_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    int8_t d0 = (_a[0] == _b[0]) ? ~UINT8_C(0) : 0x00;
    int8_t d1 = (_a[1] == _b[1]) ? ~UINT8_C(0) : 0x00;
    int8_t d2 = (_a[2] == _b[2]) ? ~UINT8_C(0) : 0x00;
    int8_t d3 = (_a[3] == _b[3]) ? ~UINT8_C(0) : 0x00;
    int8_t d4 = (_a[4] == _b[4]) ? ~UINT8_C(0) : 0x00;
    int8_t d5 = (_a[5] == _b[5]) ? ~UINT8_C(0) : 0x00;
    int8_t d6 = (_a[6] == _b[6]) ? ~UINT8_C(0) : 0x00;
    int8_t d7 = (_a[7] == _b[7]) ? ~UINT8_C(0) : 0x00;
    int8_t d8 = (_a[8] == _b[8]) ? ~UINT8_C(0) : 0x00;
    int8_t d9 = (_a[9] == _b[9]) ? ~UINT8_C(0) : 0x00;
    int8_t d10 = (_a[10] == _b[10]) ? ~UINT8_C(0) : 0x00;
    int8_t d11 = (_a[11] == _b[11]) ? ~UINT8_C(0) : 0x00;
    int8_t d12 = (_a[12] == _b[12]) ? ~UINT8_C(0) : 0x00;
    int8_t d13 = (_a[13] == _b[13]) ? ~UINT8_C(0) : 0x00;
    int8_t d14 = (_a[14] == _b[14]) ? ~UINT8_C(0) : 0x00;
    int8_t d15 = (_a[15] == _b[15]) ? ~UINT8_C(0) : 0x00;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpeq_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_cmpeq_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] == _b[0]) ? 0xffffffffffffffff : 0;
    uint64_t d1 = (_a[1] == _b[1]) ? 0xffffffffffffffff : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpeq_pd(a, b);
    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpeq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    const uint64_t d0 = (_a[0] == _b[0]) ? ~UINT64_C(0) : 0;
    const uint64_t d1 = ((const uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpeq_sd(a, b);

    return validateDouble(c, *(const double *) &d0, *(const double *) &d1);
}

result_t test_mm_cmpge_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] >= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = (_a[1] >= _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpge_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpge_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] >= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpge_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpgt_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    uint16_t d0 = _a[0] > _b[0] ? ~UINT16_C(0) : 0;
    uint16_t d1 = _a[1] > _b[1] ? ~UINT16_C(0) : 0;
    uint16_t d2 = _a[2] > _b[2] ? ~UINT16_C(0) : 0;
    uint16_t d3 = _a[3] > _b[3] ? ~UINT16_C(0) : 0;
    uint16_t d4 = _a[4] > _b[4] ? ~UINT16_C(0) : 0;
    uint16_t d5 = _a[5] > _b[5] ? ~UINT16_C(0) : 0;
    uint16_t d6 = _a[6] > _b[6] ? ~UINT16_C(0) : 0;
    uint16_t d7 = _a[7] > _b[7] ? ~UINT16_C(0) : 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpgt_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_cmpgt_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    int32_t result[4];

    result[0] = _a[0] > _b[0] ? -1 : 0;
    result[1] = _a[1] > _b[1] ? -1 : 0;
    result[2] = _a[2] > _b[2] ? -1 : 0;
    result[3] = _a[3] > _b[3] ? -1 : 0;

    __m128i iret = _mm_cmpgt_epi32(a, b);
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpgt_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    int8_t d0 = (_a[0] > _b[0]) ? ~UINT8_C(0) : 0x00;
    int8_t d1 = (_a[1] > _b[1]) ? ~UINT8_C(0) : 0x00;
    int8_t d2 = (_a[2] > _b[2]) ? ~UINT8_C(0) : 0x00;
    int8_t d3 = (_a[3] > _b[3]) ? ~UINT8_C(0) : 0x00;
    int8_t d4 = (_a[4] > _b[4]) ? ~UINT8_C(0) : 0x00;
    int8_t d5 = (_a[5] > _b[5]) ? ~UINT8_C(0) : 0x00;
    int8_t d6 = (_a[6] > _b[6]) ? ~UINT8_C(0) : 0x00;
    int8_t d7 = (_a[7] > _b[7]) ? ~UINT8_C(0) : 0x00;
    int8_t d8 = (_a[8] > _b[8]) ? ~UINT8_C(0) : 0x00;
    int8_t d9 = (_a[9] > _b[9]) ? ~UINT8_C(0) : 0x00;
    int8_t d10 = (_a[10] > _b[10]) ? ~UINT8_C(0) : 0x00;
    int8_t d11 = (_a[11] > _b[11]) ? ~UINT8_C(0) : 0x00;
    int8_t d12 = (_a[12] > _b[12]) ? ~UINT8_C(0) : 0x00;
    int8_t d13 = (_a[13] > _b[13]) ? ~UINT8_C(0) : 0x00;
    int8_t d14 = (_a[14] > _b[14]) ? ~UINT8_C(0) : 0x00;
    int8_t d15 = (_a[15] > _b[15]) ? ~UINT8_C(0) : 0x00;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpgt_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_cmpgt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] > _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = (_a[1] > _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpgt_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpgt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] > _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpgt_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmple_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = (_a[1] <= _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmple_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmple_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmple_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmplt_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    uint16_t d0 = _a[0] < _b[0] ? ~UINT16_C(0) : 0;
    uint16_t d1 = _a[1] < _b[1] ? ~UINT16_C(0) : 0;
    uint16_t d2 = _a[2] < _b[2] ? ~UINT16_C(0) : 0;
    uint16_t d3 = _a[3] < _b[3] ? ~UINT16_C(0) : 0;
    uint16_t d4 = _a[4] < _b[4] ? ~UINT16_C(0) : 0;
    uint16_t d5 = _a[5] < _b[5] ? ~UINT16_C(0) : 0;
    uint16_t d6 = _a[6] < _b[6] ? ~UINT16_C(0) : 0;
    uint16_t d7 = _a[7] < _b[7] ? ~UINT16_C(0) : 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmplt_epi16(a, b);

    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_cmplt_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    int32_t result[4];
    result[0] = _a[0] < _b[0] ? -1 : 0;
    result[1] = _a[1] < _b[1] ? -1 : 0;
    result[2] = _a[2] < _b[2] ? -1 : 0;
    result[3] = _a[3] < _b[3] ? -1 : 0;

    __m128i iret = _mm_cmplt_epi32(a, b);
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmplt_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    int8_t d0 = (_a[0] < _b[0]) ? ~UINT8_C(0) : 0x00;
    int8_t d1 = (_a[1] < _b[1]) ? ~UINT8_C(0) : 0x00;
    int8_t d2 = (_a[2] < _b[2]) ? ~UINT8_C(0) : 0x00;
    int8_t d3 = (_a[3] < _b[3]) ? ~UINT8_C(0) : 0x00;
    int8_t d4 = (_a[4] < _b[4]) ? ~UINT8_C(0) : 0x00;
    int8_t d5 = (_a[5] < _b[5]) ? ~UINT8_C(0) : 0x00;
    int8_t d6 = (_a[6] < _b[6]) ? ~UINT8_C(0) : 0x00;
    int8_t d7 = (_a[7] < _b[7]) ? ~UINT8_C(0) : 0x00;
    int8_t d8 = (_a[8] < _b[8]) ? ~UINT8_C(0) : 0x00;
    int8_t d9 = (_a[9] < _b[9]) ? ~UINT8_C(0) : 0x00;
    int8_t d10 = (_a[10] < _b[10]) ? ~UINT8_C(0) : 0x00;
    int8_t d11 = (_a[11] < _b[11]) ? ~UINT8_C(0) : 0x00;
    int8_t d12 = (_a[12] < _b[12]) ? ~UINT8_C(0) : 0x00;
    int8_t d13 = (_a[13] < _b[13]) ? ~UINT8_C(0) : 0x00;
    int8_t d14 = (_a[14] < _b[14]) ? ~UINT8_C(0) : 0x00;
    int8_t d15 = (_a[15] < _b[15]) ? ~UINT8_C(0) : 0x00;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmplt_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_cmplt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    int64_t f0 = (_a[0] < _b[0]) ? ~UINT64_C(0) : UINT64_C(0);
    int64_t f1 = (_a[1] < _b[1]) ? ~UINT64_C(0) : UINT64_C(0);

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmplt_pd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmplt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmplt_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpneq_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    int64_t f0 = (_a[0] != _b[0]) ? ~UINT64_C(0) : UINT64_C(0);
    int64_t f1 = (_a[1] != _b[1]) ? ~UINT64_C(0) : UINT64_C(0);

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpneq_pd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmpneq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;

    int64_t f0 = (_a[0] != _b[0]) ? ~UINT64_C(0) : UINT64_C(0);
    int64_t f1 = ((int64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpneq_sd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmpnge_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = !(_a[0] >= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = !(_a[1] >= _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnge_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpnge_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = !(_a[0] >= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnge_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpngt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = !(_a[0] > _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = !(_a[1] > _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpngt_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpngt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = !(_a[0] > _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpngt_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpnle_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = !(_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = !(_a[1] <= _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnle_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpnle_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = !(_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnle_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpnlt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = !(_a[0] < _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = !(_a[1] < _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnlt_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpnlt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = !(_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnlt_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpord_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);

    double result[2];

    for (uint32_t i = 0; i < 2; i++) {
        result[i] = cmp_noNaN(_a[i], _b[i]);
    }

    __m128d ret = _mm_cmpord_pd(a, b);

    return validateDouble(ret, result[0], result[1]);
}

result_t test_mm_cmpord_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);

    double c0 = cmp_noNaN(_a[0], _b[0]);
    double c1 = _a[1];

    __m128d ret = _mm_cmpord_sd(a, b);
    return validateDouble(ret, c0, c1);
}

result_t test_mm_cmpunord_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);

    double result[2];
    result[0] = cmp_hasNaN(_a[0], _b[0]);
    result[1] = cmp_hasNaN(_a[1], _b[1]);

    __m128d ret = _mm_cmpunord_pd(a, b);
    return validateDouble(ret, result[0], result[1]);
}

result_t test_mm_cmpunord_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);

    double result[2];
    result[0] = cmp_hasNaN(_a[0], _b[0]);
    result[1] = _a[1];

    __m128d ret = _mm_cmpunord_sd(a, b);
    return validateDouble(ret, result[0], result[1]);
}

result_t test_mm_comieq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comieq_sd correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__)
    return TEST_UNIMPL;
#else
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    int32_t _c = (_a[0] == _b[0]) ? 1 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    int32_t c = _mm_comieq_sd(a, b);

    ASSERT_RETURN(c == _c);
    return TEST_SUCCESS;
#endif
}

result_t test_mm_comige_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    int32_t _c = (_a[0] >= _b[0]) ? 1 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    int32_t c = _mm_comige_sd(a, b);

    ASSERT_RETURN(c == _c);
    return TEST_SUCCESS;
}

result_t test_mm_comigt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    int32_t _c = (_a[0] > _b[0]) ? 1 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    int32_t c = _mm_comigt_sd(a, b);

    ASSERT_RETURN(c == _c);
    return TEST_SUCCESS;
}

result_t test_mm_comile_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comile_sd correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__)
    return TEST_UNIMPL;
#else
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    int32_t _c = (_a[0] <= _b[0]) ? 1 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    int32_t c = _mm_comile_sd(a, b);

    ASSERT_RETURN(c == _c);
    return TEST_SUCCESS;
#endif
}

result_t test_mm_comilt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comilt_sd correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__)
    return TEST_UNIMPL;
#else
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    int32_t _c = (_a[0] < _b[0]) ? 1 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    int32_t c = _mm_comilt_sd(a, b);

    ASSERT_RETURN(c == _c);
    return TEST_SUCCESS;
#endif
}

result_t test_mm_comineq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comineq_sd correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__)
    return TEST_UNIMPL;
#else
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    int32_t _c = (_a[0] != _b[0]) ? 1 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    int32_t c = _mm_comineq_sd(a, b);

    ASSERT_RETURN(c == _c);
    return TEST_SUCCESS;
#endif
}

result_t test_mm_cvtepi32_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);
    double trun[2] = {(double) _a[0], (double) _a[1]};

    __m128d ret = _mm_cvtepi32_pd(a);
    return validateDouble(ret, trun[0], trun[1]);
}

result_t test_mm_cvtepi32_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);
    float trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (float) _a[i];
    }

    __m128 ret = _mm_cvtepi32_ps(a);
    return validateFloat(ret, trun[0], trun[1], trun[2], trun[3]);
}

result_t test_mm_cvtpd_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    int32_t d[2];

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d[0] = (int32_t) (bankersRounding(_a[0]));
        d[1] = (int32_t) (bankersRounding(_a[1]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d[0] = (int32_t) (floor(_a[0]));
        d[1] = (int32_t) (floor(_a[1]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d[0] = (int32_t) (ceil(_a[0]));
        d[1] = (int32_t) (ceil(_a[1]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d[0] = (int32_t) (_a[0]);
        d[1] = (int32_t) (_a[1]);
        break;
    }

#if defined(__ARM_FEATURE_FRINT) && !defined(__clang__)
    /* Floats that cannot fit into 32-bits should instead return
     * indefinite integer value (INT32_MIN). This behaviour is
     * currently only emulated when using the round-to-integral
     * instructions. */
    for (int i = 0; i < 2; i++) {
        if (_a[i] > (float) INT32_MAX || _a[i] < (float) INT32_MIN)
            d[i] = INT32_MIN;
    }
#endif

    __m128d a = load_m128d(_a);
    __m128i ret = _mm_cvtpd_epi32(a);

    return validateInt32(ret, d[0], d[1], 0, 0);
}

result_t test_mm_cvtpd_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    int32_t d[2];

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d[0] = (int32_t) (bankersRounding(_a[0]));
        d[1] = (int32_t) (bankersRounding(_a[1]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d[0] = (int32_t) (floor(_a[0]));
        d[1] = (int32_t) (floor(_a[1]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d[0] = (int32_t) (ceil(_a[0]));
        d[1] = (int32_t) (ceil(_a[1]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d[0] = (int32_t) (_a[0]);
        d[1] = (int32_t) (_a[1]);
        break;
    }

    __m128d a = load_m128d(_a);
    __m64 ret = _mm_cvtpd_pi32(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvtpd_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    float f0 = (float) _a[0];
    float f1 = (float) _a[1];
    const __m128d a = load_m128d(_a);

    __m128 r = _mm_cvtpd_ps(a);

    return validateFloat(r, f0, f1, 0, 0);
}

result_t test_mm_cvtpi32_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m64 a = load_m64(_a);

    double trun[2] = {(double) _a[0], (double) _a[1]};

    __m128d ret = _mm_cvtpi32_pd(a);

    return validateDouble(ret, trun[0], trun[1]);
}

result_t test_mm_cvtps_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    __m128 a = load_m128(_a);
    int32_t d[4];
    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        for (uint32_t i = 0; i < 4; i++) {
            d[i] = (int32_t) (bankersRounding(_a[i]));
        }
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        for (uint32_t i = 0; i < 4; i++) {
            d[i] = (int32_t) (floorf(_a[i]));
        }
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        for (uint32_t i = 0; i < 4; i++) {
            d[i] = (int32_t) (ceilf(_a[i]));
        }
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        for (uint32_t i = 0; i < 4; i++) {
            d[i] = (int32_t) (_a[i]);
        }
        break;
    }

    __m128i ret = _mm_cvtps_epi32(a);
    return validateInt32(ret, d[0], d[1], d[2], d[3]);
}

result_t test_mm_cvtps_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    double d0 = (double) _a[0];
    double d1 = (double) _a[1];
    const __m128 a = load_m128(_a);

    __m128d r = _mm_cvtps_pd(a);

    return validateDouble(r, d0, d1);
}

result_t test_mm_cvtsd_f64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    double d = _a[0];

    const __m128d *a = (const __m128d *) _a;
    double r = _mm_cvtsd_f64(*a);

    return r == d ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsd_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    int32_t d;

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d = (int32_t) (bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d = (int32_t) (floor(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d = (int32_t) (ceil(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d = (int32_t) (_a[0]);
        break;
    }

    __m128d a = load_m128d(_a);
    int32_t ret = _mm_cvtsd_si32(a);

    return ret == d ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsd_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    int64_t d;

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d = (int64_t) (bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d = (int64_t) (floor(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d = (int64_t) (ceil(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d = (int64_t) (_a[0]);
        break;
    }

    __m128d a = load_m128d(_a);
    int64_t ret = _mm_cvtsd_si64(a);

    return ret == d ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsd_si64x(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_cvtsd_si64(impl, iter);
}

result_t test_mm_cvtsd_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    float f0 = _b[0];
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = load_m128(_a);
    __m128d b = load_m128d(_b);
    __m128 c = _mm_cvtsd_ss(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_cvtsi128_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;

    int32_t d = _a[0];

    __m128i a = load_m128i(_a);
    int c = _mm_cvtsi128_si32(a);

    return d == c ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsi128_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d = _a[0];

    __m128i a = load_m128i(_a);
    int64_t c = _mm_cvtsi128_si64(a);

    return d == c ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsi128_si64x(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_cvtsi128_si64(impl, iter);
}

result_t test_mm_cvtsi32_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const int32_t b = (const int32_t) impl.mTestInts[iter];

    __m128d a = load_m128d(_a);
    __m128d c = _mm_cvtsi32_sd(a, b);

    return validateDouble(c, b, _a[1]);
}

result_t test_mm_cvtsi32_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;

    int32_t d = _a[0];

    __m128i c = _mm_cvtsi32_si128(*_a);

    return validateInt32(c, d, 0, 0, 0);
}

result_t test_mm_cvtsi64_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const int64_t b = (const int64_t) impl.mTestInts[iter];

    __m128d a = load_m128d(_a);
    __m128d c = _mm_cvtsi64_sd(a, b);

    return validateDouble(c, b, _a[1]);
}

result_t test_mm_cvtsi64_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d = _a[0];

    __m128i c = _mm_cvtsi64_si128(*_a);

    return validateInt64(c, d, 0);
}

result_t test_mm_cvtsi64x_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_cvtsi64_sd(impl, iter);
}

result_t test_mm_cvtsi64x_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_cvtsi64_si128(impl, iter);
}

result_t test_mm_cvtss_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    double d0 = double(_b[0]);
    double d1 = _a[1];

    __m128d a = load_m128d(_a);
    __m128 b = load_m128(_b);
    __m128d c = _mm_cvtss_sd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_cvttpd_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    __m128d a = load_m128d(_a);
    int32_t d0 = (int32_t) (_a[0]);
    int32_t d1 = (int32_t) (_a[1]);

    __m128i ret = _mm_cvttpd_epi32(a);
    return validateInt32(ret, d0, d1, 0, 0);
}

result_t test_mm_cvttpd_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    __m128d a = load_m128d(_a);
    int32_t d0 = (int32_t) (_a[0]);
    int32_t d1 = (int32_t) (_a[1]);

    __m64 ret = _mm_cvttpd_pi32(a);
    return validateInt32(ret, d0, d1);
}

result_t test_mm_cvttps_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    __m128 a = load_m128(_a);
    int32_t trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (int32_t) _a[i];
    }

    __m128i ret = _mm_cvttps_epi32(a);
    return validateInt32(ret, trun[0], trun[1], trun[2], trun[3]);
}

result_t test_mm_cvttsd_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    __m128d a = _mm_load_sd(_a);
    int32_t ret = _mm_cvttsd_si32(a);

    return ret == (int32_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvttsd_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    __m128d a = _mm_load_sd(_a);
    int64_t ret = _mm_cvttsd_si64(a);

    return ret == (int64_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvttsd_si64x(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if defined(__clang__)
    // The intrinsic _mm_cvttsd_si64x() does not exist in Clang
    return TEST_UNIMPL;
#else
    const double *_a = (const double *) impl.mTestFloatPointer1;

    __m128d a = _mm_load_sd(_a);
    int64_t ret = _mm_cvttsd_si64x(a);

    return ret == (int64_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
#endif
}

result_t test_mm_div_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = 0.0, d1 = 0.0;

    if (_b[0] != 0.0)
        d0 = _a[0] / _b[0];
    if (_b[1] != 0.0)
        d1 = _a[1] / _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_div_pd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_div_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double d0 = _a[0] / _b[0];
    double d1 = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);

    __m128d c = _mm_div_sd(a, b);

    return validateDouble(c, d0, d1);
}

result_t test_mm_extract_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint16_t *_a = (uint16_t *) impl.mTestIntPointer1;
    const int idx = iter & 0x7;
    __m128i a = load_m128i(_a);
    int c;
    switch (idx) {
    case 0:
        c = _mm_extract_epi16(a, 0);
        break;
    case 1:
        c = _mm_extract_epi16(a, 1);
        break;
    case 2:
        c = _mm_extract_epi16(a, 2);
        break;
    case 3:
        c = _mm_extract_epi16(a, 3);
        break;
    case 4:
        c = _mm_extract_epi16(a, 4);
        break;
    case 5:
        c = _mm_extract_epi16(a, 5);
        break;
    case 6:
        c = _mm_extract_epi16(a, 6);
        break;
    case 7:
        c = _mm_extract_epi16(a, 7);
        break;
    }

    ASSERT_RETURN(c == *(_a + idx));
    return TEST_SUCCESS;
}

result_t test_mm_insert_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t insert = (int16_t) *impl.mTestIntPointer2;
    const int imm8 = 2;

    int16_t d[8];
    for (int i = 0; i < 8; i++) {
        d[i] = _a[i];
    }
    d[imm8] = insert;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_insert_epi16(a, insert, imm8);
    return validateInt16(b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_lfence(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

result_t test_mm_load_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = _mm_load_pd(p);
    return validateDouble(a, p[0], p[1]);
}

result_t test_mm_load_pd1(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = _mm_load_pd1(p);
    return validateDouble(a, p[0], p[0]);
}

result_t test_mm_load_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = _mm_load_sd(p);
    return validateDouble(a, p[0], 0);
}

result_t test_mm_load_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *addr = impl.mTestIntPointer1;

    __m128i ret = _mm_load_si128((const __m128i *) addr);

    return validateInt32(ret, addr[0], addr[1], addr[2], addr[3]);
}

result_t test_mm_load1_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *addr = (const double *) impl.mTestFloatPointer1;

    __m128d ret = _mm_load1_pd(addr);

    return validateDouble(ret, addr[0], addr[0]);
}

result_t test_mm_loadh_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *addr = (const double *) impl.mTestFloatPointer2;

    __m128d a = load_m128d(_a);
    __m128d ret = _mm_loadh_pd(a, addr);

    return validateDouble(ret, _a[0], addr[0]);
}

result_t test_mm_loadl_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *addr = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_loadl_epi64((const __m128i *) addr);

    return validateInt64(ret, addr[0], 0);
}

result_t test_mm_loadl_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *addr = (const double *) impl.mTestFloatPointer2;

    __m128d a = load_m128d(_a);
    __m128d ret = _mm_loadl_pd(a, addr);

    return validateDouble(ret, addr[0], _a[1]);
}

result_t test_mm_loadr_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *addr = (const double *) impl.mTestFloatPointer1;

    __m128d ret = _mm_loadr_pd(addr);

    return validateDouble(ret, addr[1], addr[0]);
}

result_t test_mm_loadu_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = _mm_loadu_pd(p);
    return validateDouble(a, p[0], p[1]);
}

result_t test_mm_loadu_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i c = _mm_loadu_si128((const __m128i *) _a);
    return validateInt32(c, _a[0], _a[1], _a[2], _a[3]);
}

result_t test_mm_loadu_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_loadu_si32. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
    return TEST_UNIMPL;
#else
    const int32_t *addr = (const int32_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_loadu_si32((const void *) addr);

    return validateInt32(ret, addr[0], 0, 0, 0);
#endif
}

result_t test_mm_madd_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_madd_epi16(a, b);
    return validateInt32(c, e0, e1, e2, e3);
}

result_t test_mm_maskmoveu_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_mask = (const uint8_t *) impl.mTestIntPointer2;
    char mem_addr[16];

    __m128i a = load_m128i(_a);
    __m128i mask = load_m128i(_mask);
    _mm_maskmoveu_si128(a, mask, mem_addr);

    for (int i = 0; i < 16; i++) {
        if (_mask[i] >> 7) {
            ASSERT_RETURN(_a[i] == (uint8_t) mem_addr[i]);
        }
    }

    return TEST_SUCCESS;
}

result_t test_mm_max_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    int16_t d1 = _a[1] > _b[1] ? _a[1] : _b[1];
    int16_t d2 = _a[2] > _b[2] ? _a[2] : _b[2];
    int16_t d3 = _a[3] > _b[3] ? _a[3] : _b[3];
    int16_t d4 = _a[4] > _b[4] ? _a[4] : _b[4];
    int16_t d5 = _a[5] > _b[5] ? _a[5] : _b[5];
    int16_t d6 = _a[6] > _b[6] ? _a[6] : _b[6];
    int16_t d7 = _a[7] > _b[7] ? _a[7] : _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_max_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_max_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_max_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_max_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double f0 = _a[0] > _b[0] ? _a[0] : _b[0];
    double f1 = _a[1] > _b[1] ? _a[1] : _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_max_pd(a, b);

    return validateDouble(c, f0, f1);
}

result_t test_mm_max_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    double d1 = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_max_sd(a, b);

    return validateDouble(c, d0, d1);
}

result_t test_mm_mfence(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

result_t test_mm_min_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    int16_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
    int16_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
    int16_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];
    int16_t d4 = _a[4] < _b[4] ? _a[4] : _b[4];
    int16_t d5 = _a[5] < _b[5] ? _a[5] : _b[5];
    int16_t d6 = _a[6] < _b[6] ? _a[6] : _b[6];
    int16_t d7 = _a[7] < _b[7] ? _a[7] : _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_min_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_min_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double f0 = _a[0] < _b[0] ? _a[0] : _b[0];
    double f1 = _a[1] < _b[1] ? _a[1] : _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);

    __m128d c = _mm_min_pd(a, b);
    return validateDouble(c, f0, f1);
}

result_t test_mm_min_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    double d1 = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_min_sd(a, b);

    return validateDouble(c, d0, d1);
}

result_t test_mm_move_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d0 = _a[0];
    int64_t d1 = 0;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_move_epi64(a);

    return validateInt64(c, d0, d1);
}

result_t test_mm_move_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);

    double result[2];
    result[0] = _b[0];
    result[1] = _a[1];

    __m128d ret = _mm_move_sd(a, b);
    return validateDouble(ret, result[0], result[1]);
}

result_t test_mm_movemask_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);

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
    return TEST_SUCCESS;
}

result_t test_mm_movemask_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    unsigned int _c = 0;
    _c |= ((*(const uint64_t *) _a) >> 63) & 0x1;
    _c |= (((*(const uint64_t *) (_a + 1)) >> 62) & 0x2);

    __m128d a = load_m128d(_a);
    int c = _mm_movemask_pd(a);

    ASSERT_RETURN((unsigned int) c == _c);
    return TEST_SUCCESS;
}

result_t test_mm_movepi64_pi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d0 = _a[0];

    __m128i a = load_m128i(_a);
    __m64 c = _mm_movepi64_pi64(a);

    return validateInt64(c, d0);
}

result_t test_mm_movpi64_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d0 = _a[0];

    __m64 a = load_m64(_a);
    __m128i c = _mm_movpi64_epi64(a);

    return validateInt64(c, d0, 0);
}

result_t test_mm_mul_epu32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;
    const uint32_t *_b = (const uint32_t *) impl.mTestIntPointer2;
    uint64_t dx = (uint64_t) (_a[0]) * (uint64_t) (_b[0]);
    uint64_t dy = (uint64_t) (_a[2]) * (uint64_t) (_b[2]);

    __m128i a = _mm_loadu_si128((const __m128i *) _a);
    __m128i b = _mm_loadu_si128((const __m128i *) _b);
    __m128i r = _mm_mul_epu32(a, b);
    return validateUInt64(r, dx, dy);
}

result_t test_mm_mul_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] * _b[0];
    double d1 = _a[1] * _b[1];

    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);
    __m128d c = _mm_mul_pd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_mul_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double dx = _a[0] * _b[0];
    double dy = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_mul_sd(a, b);
    return validateDouble(c, dx, dy);
}

result_t test_mm_mul_su32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;
    const uint32_t *_b = (const uint32_t *) impl.mTestIntPointer2;

    uint64_t u = (uint64_t) (_a[0]) * (uint64_t) (_b[0]);

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 r = _mm_mul_su32(a, b);

    return validateUInt64(r, u);
}

result_t test_mm_mulhi_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d[8];
    for (uint32_t i = 0; i < 8; i++) {
        int32_t m = (int32_t) _a[i] * (int32_t) _b[i];
        d[i] = (int16_t) (m >> 16);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_mulhi_epi16(a, b);
    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_mulhi_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;
    uint16_t d[8];
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t m = (uint32_t) _a[i] * (uint32_t) _b[i];
        d[i] = (uint16_t) (m >> 16);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_mulhi_epu16(a, b);
    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_mullo_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d0 = _a[0] * _b[0];
    int16_t d1 = _a[1] * _b[1];
    int16_t d2 = _a[2] * _b[2];
    int16_t d3 = _a[3] * _b[3];
    int16_t d4 = _a[4] * _b[4];
    int16_t d5 = _a[5] * _b[5];
    int16_t d6 = _a[6] * _b[6];
    int16_t d7 = _a[7] * _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_mullo_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_or_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestFloatPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestFloatPointer2;

    int64_t d0 = _a[0] | _b[0];
    int64_t d1 = _a[1] | _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_or_pd(a, b);

    return validateDouble(c, *((double *) &d0), *((double *) &d1));
}

result_t test_mm_or_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128 fc = _mm_or_ps(*(const __m128 *) &a, *(const __m128 *) &b);
    __m128i c = *(const __m128i *) &fc;
    // now for the assertion...
    const uint32_t *ia = (const uint32_t *) &a;
    const uint32_t *ib = (const uint32_t *) &b;
    uint32_t r0 = ia[0] | ib[0];
    uint32_t r1 = ia[1] | ib[1];
    uint32_t r2 = ia[2] | ib[2];
    uint32_t r3 = ia[3] | ib[3];
    __m128i ret = do_mm_set_epi32(r3, r2, r1, r0);
    result_t r = validateInt32(c, r0, r1, r2, r3);
    if (r) {
        r = validateInt32(ret, r0, r1, r2, r3);
    }
    return r;
}

result_t test_mm_packs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int8_t max = INT8_MAX;
    int8_t min = INT8_MIN;
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    int8_t d[16];
    for (int i = 0; i < 8; i++) {
        if (_a[i] > max)
            d[i] = max;
        else if (_a[i] < min)
            d[i] = min;
        else
            d[i] = (int8_t) _a[i];
    }
    for (int i = 0; i < 8; i++) {
        if (_b[i] > max)
            d[i + 8] = max;
        else if (_b[i] < min)
            d[i + 8] = min;
        else
            d[i + 8] = (int8_t) _b[i];
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_packs_epi16(a, b);

    return validateInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                        d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_packs_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int16_t max = INT16_MAX;
    int16_t min = INT16_MIN;
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int16_t d[8];
    for (int i = 0; i < 4; i++) {
        if (_a[i] > max)
            d[i] = max;
        else if (_a[i] < min)
            d[i] = min;
        else
            d[i] = (int16_t) _a[i];
    }
    for (int i = 0; i < 4; i++) {
        if (_b[i] > max)
            d[i + 4] = max;
        else if (_b[i] < min)
            d[i + 4] = min;
        else
            d[i + 4] = (int16_t) _b[i];
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_packs_epi32(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_packus_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint8_t max = UINT8_MAX;
    uint8_t min = 0;
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    uint8_t d[16];
    for (int i = 0; i < 8; i++) {
        if (_a[i] > (int16_t) max)
            d[i] = max;
        else if (_a[i] < (int16_t) min)
            d[i] = min;
        else
            d[i] = (uint8_t) _a[i];
    }
    for (int i = 0; i < 8; i++) {
        if (_b[i] > (int16_t) max)
            d[i + 8] = max;
        else if (_b[i] < (int16_t) min)
            d[i + 8] = min;
        else
            d[i + 8] = (uint8_t) _b[i];
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_packus_epi16(a, b);

    return validateUInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                         d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_pause(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    _mm_pause();
    return TEST_SUCCESS;
}

result_t test_mm_sad_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    uint16_t d0 = 0;
    uint16_t d1 = 0;
    for (int i = 0; i < 8; i++) {
        d0 += abs(_a[i] - _b[i]);
    }
    for (int i = 8; i < 16; i++) {
        d1 += abs(_a[i] - _b[i]);
    }

    const __m128i a = load_m128i(_a);
    const __m128i b = load_m128i(_b);
    __m128i c = _mm_sad_epu8(a, b);
    return validateUInt16(c, d0, 0, 0, 0, d1, 0, 0, 0);
}

result_t test_mm_set_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
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

result_t test_mm_set_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t x = impl.mTestInts[iter];
    int32_t y = impl.mTestInts[iter + 1];
    int32_t z = impl.mTestInts[iter + 2];
    int32_t w = impl.mTestInts[iter + 3];
    __m128i a = _mm_set_epi32(x, y, z, w);
    return validateInt32(a, w, z, y, x);
}

result_t test_mm_set_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_set_epi64((__m64) _a[1], (__m64) _a[0]);

    return validateInt64(ret, _a[0], _a[1]);
}

result_t test_mm_set_epi64x(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_set_epi64x(_a[1], _a[0]);

    return validateInt64(ret, _a[0], _a[1]);
}

result_t test_mm_set_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
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

result_t test_mm_set_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    double x = p[0];
    double y = p[1];
    __m128d a = _mm_set_pd(x, y);
    return validateDouble(a, y, x);
}

result_t test_mm_set_pd1(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double _a = impl.mTestFloats[iter];

    __m128d a = _mm_set_pd1(_a);

    return validateDouble(a, _a, _a);
}

result_t test_mm_set_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    double f0 = _a[0];
    double f1 = 0.0;

    __m128d a = _mm_set_sd(_a[0]);
    return validateDouble(a, f0, f1);
}

result_t test_mm_set1_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    int16_t d0 = _a[0];

    __m128i c = _mm_set1_epi16(d0);
    return validateInt16(c, d0, d0, d0, d0, d0, d0, d0, d0);
}

result_t test_mm_set1_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t x = impl.mTestInts[iter];
    __m128i a = _mm_set1_epi32(x);
    return validateInt32(a, x, x, x, x);
}

result_t test_mm_set1_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_set1_epi64((__m64) _a[0]);

    return validateInt64(ret, _a[0], _a[0]);
}

result_t test_mm_set1_epi64x(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_set1_epi64x(_a[0]);

    return validateInt64(ret, _a[0], _a[0]);
}

result_t test_mm_set1_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    int8_t d0 = _a[0];
    __m128i c = _mm_set1_epi8(d0);
    return validateInt8(c, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0,
                        d0, d0, d0);
}

result_t test_mm_set1_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    double d0 = _a[0];
    __m128d c = _mm_set1_pd(d0);
    return validateDouble(c, d0, d0);
}

result_t test_mm_setr_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;

    __m128i c =
        _mm_setr_epi16(_a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6], _a[7]);

    return validateInt16(c, _a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6],
                         _a[7]);
}

result_t test_mm_setr_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i c = _mm_setr_epi32(_a[0], _a[1], _a[2], _a[3]);
    return validateInt32(c, _a[0], _a[1], _a[2], _a[3]);
}

result_t test_mm_setr_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const __m64 *_a = (const __m64 *) impl.mTestIntPointer1;
    __m128i c = _mm_setr_epi64(_a[0], _a[1]);
    return validateInt64(c, (int64_t) _a[0], (int64_t) _a[1]);
}

result_t test_mm_setr_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    __m128i c = _mm_setr_epi8(_a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6],
                              _a[7], _a[8], _a[9], _a[10], _a[11], _a[12],
                              _a[13], _a[14], _a[15]);

    return validateInt8(c, _a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6],
                        _a[7], _a[8], _a[9], _a[10], _a[11], _a[12], _a[13],
                        _a[14], _a[15]);
}

result_t test_mm_setr_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = (const double *) impl.mTestFloatPointer1;

    double x = p[0];
    double y = p[1];

    __m128d a = _mm_setr_pd(x, y);

    return validateDouble(a, x, y);
}

result_t test_mm_setzero_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128d a = _mm_setzero_pd();
    return validateDouble(a, 0, 0);
}

result_t test_mm_setzero_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128i a = _mm_setzero_si128();
    return validateInt32(a, 0, 0, 0, 0);
}

result_t test_mm_shuffle_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int imm = 105;

    int32_t d0 = _a[((imm) &0x3)];
    int32_t d1 = _a[((imm >> 2) & 0x3)];
    int32_t d2 = _a[((imm >> 4) & 0x3)];
    int32_t d3 = _a[((imm >> 6) & 0x3)];

    __m128i a = load_m128i(_a);
    __m128i c = _mm_shuffle_epi32(a, imm);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_shuffle_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double d0 = _a[iter & 0x1];
    double d1 = _b[(iter & 0x2) >> 1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c;
    switch (iter & 0x3) {
    case 0:
        c = _mm_shuffle_pd(a, b, 0);
        break;
    case 1:
        c = _mm_shuffle_pd(a, b, 1);
        break;
    case 2:
        c = _mm_shuffle_pd(a, b, 2);
        break;
    case 3:
        c = _mm_shuffle_pd(a, b, 3);
        break;
    }

    return validateDouble(c, d0, d1);
}

result_t test_mm_shufflehi_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int imm = 112;

    int16_t d0 = _a[0];
    int16_t d1 = _a[1];
    int16_t d2 = _a[2];
    int16_t d3 = _a[3];
    int16_t d4 = ((const int64_t *) _a)[1] >> ((imm & 0x3) * 16);
    int16_t d5 = ((const int64_t *) _a)[1] >> (((imm >> 2) & 0x3) * 16);
    int16_t d6 = ((const int64_t *) _a)[1] >> (((imm >> 4) & 0x3) * 16);
    int16_t d7 = ((const int64_t *) _a)[1] >> (((imm >> 6) & 0x3) * 16);

    __m128i a = load_m128i(_a);
    __m128i c = _mm_shufflehi_epi16(a, imm);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_shufflelo_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int imm = 112;

    int16_t d0 = ((const int64_t *) _a)[0] >> ((imm & 0x3) * 16);
    int16_t d1 = ((const int64_t *) _a)[0] >> (((imm >> 2) & 0x3) * 16);
    int16_t d2 = ((const int64_t *) _a)[0] >> (((imm >> 4) & 0x3) * 16);
    int16_t d3 = ((const int64_t *) _a)[0] >> (((imm >> 6) & 0x3) * 16);
    int16_t d4 = _a[4];
    int16_t d5 = _a[5];
    int16_t d6 = _a[6];
    int16_t d7 = _a[7];

    __m128i a = load_m128i(_a);
    __m128i c = _mm_shufflelo_epi16(a, imm);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_sll_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) (iter % 18 - 1);  // range: -1 ~ 16

    uint16_t d0 = (count & ~15) ? 0 : _a[0] << count;
    uint16_t d1 = (count & ~15) ? 0 : _a[1] << count;
    uint16_t d2 = (count & ~15) ? 0 : _a[2] << count;
    uint16_t d3 = (count & ~15) ? 0 : _a[3] << count;
    uint16_t d4 = (count & ~15) ? 0 : _a[4] << count;
    uint16_t d5 = (count & ~15) ? 0 : _a[5] << count;
    uint16_t d6 = (count & ~15) ? 0 : _a[6] << count;
    uint16_t d7 = (count & ~15) ? 0 : _a[7] << count;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sll_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_sll_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) (iter % 34 - 1);  // range: -1 ~ 32

    uint32_t d0 = (count & ~31) ? 0 : _a[0] << count;
    uint32_t d1 = (count & ~31) ? 0 : _a[1] << count;
    uint32_t d2 = (count & ~31) ? 0 : _a[2] << count;
    uint32_t d3 = (count & ~31) ? 0 : _a[3] << count;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sll_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_sll_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) (iter % 66 - 1);  // range: -1 ~ 64

    uint64_t d0 = (count & ~63) ? 0 : _a[0] << count;
    uint64_t d1 = (count & ~63) ? 0 : _a[1] << count;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sll_epi64(a, b);

    return validateInt64(c, d0, d1);
}

result_t test_mm_slli_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int count = (int64_t) (iter % 18 - 1);  // range: -1 ~ 16

    int16_t d0 = (count & ~15) ? 0 : _a[0] << count;
    int16_t d1 = (count & ~15) ? 0 : _a[1] << count;
    int16_t d2 = (count & ~15) ? 0 : _a[2] << count;
    int16_t d3 = (count & ~15) ? 0 : _a[3] << count;
    int16_t d4 = (count & ~15) ? 0 : _a[4] << count;
    int16_t d5 = (count & ~15) ? 0 : _a[5] << count;
    int16_t d6 = (count & ~15) ? 0 : _a[6] << count;
    int16_t d7 = (count & ~15) ? 0 : _a[7] << count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_slli_epi16(a, count);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_slli_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
#if defined(__clang__)
    // Clang compiler does not allow the second argument of _mm_slli_epi32() to
    // be greater than 31.
    const int count = (int) (iter % 33 - 1);  // range: -1 ~ 31
#else
    const int count = (int) (iter % 34 - 1);  // range: -1 ~ 32
#endif

    int32_t d0 = (count & ~31) ? 0 : _a[0] << count;
    int32_t d1 = (count & ~31) ? 0 : _a[1] << count;
    int32_t d2 = (count & ~31) ? 0 : _a[2] << count;
    int32_t d3 = (count & ~31) ? 0 : _a[3] << count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_slli_epi32(a, count);
    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_slli_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
#if defined(__clang__)
    // Clang compiler does not allow the second argument of `_mm_slli_epi64()`
    // to be greater than 63.
    const int count = (int) (iter % 65 - 1);  // range: -1 ~ 63
#else
    const int count = (int) (iter % 66 - 1);  // range: -1 ~ 64
#endif
    int64_t d0 = (count & ~63) ? 0 : _a[0] << count;
    int64_t d1 = (count & ~63) ? 0 : _a[1] << count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_slli_epi64(a, count);
    return validateInt64(c, d0, d1);
}

result_t test_mm_slli_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;

    int8_t d[16];
    int count = (iter % 5) << 2;
    for (int i = 0; i < 16; i++) {
        if (i < count)
            d[i] = 0;
        else
            d[i] = ((const int8_t *) _a)[i - count];
    }

    __m128i a = load_m128i(_a);
    __m128i ret;
    switch (iter % 5) {
    case 0:
        ret = _mm_slli_si128(a, 0);
        break;
    case 1:
        ret = _mm_slli_si128(a, 4);
        break;
    case 2:
        ret = _mm_slli_si128(a, 8);
        break;
    case 3:
        ret = _mm_slli_si128(a, 12);
        break;
    case 4:
        ret = _mm_slli_si128(a, 16);
        break;
    }

    return validateInt8(ret, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                        d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_sqrt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    double f0 = sqrt(_a[0]);
    double f1 = sqrt(_a[1]);

    __m128d a = load_m128d(_a);
    __m128d c = _mm_sqrt_pd(a);

    return validateDouble(c, f0, f1);
}

result_t test_mm_sqrt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double f0 = sqrt(_b[0]);
    double f1 = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_sqrt_sd(a, b);

    return validateDouble(c, f0, f1);
}

result_t test_mm_sra_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) (iter % 18 - 1);  // range: -1 ~ 16

    int16_t d0 =
        (count & ~15) ? (_a[0] < 0 ? ~UINT16_C(0) : 0) : (_a[0] >> count);
    int16_t d1 =
        (count & ~15) ? (_a[1] < 0 ? ~UINT16_C(0) : 0) : (_a[1] >> count);
    int16_t d2 =
        (count & ~15) ? (_a[2] < 0 ? ~UINT16_C(0) : 0) : (_a[2] >> count);
    int16_t d3 =
        (count & ~15) ? (_a[3] < 0 ? ~UINT16_C(0) : 0) : (_a[3] >> count);
    int16_t d4 =
        (count & ~15) ? (_a[4] < 0 ? ~UINT16_C(0) : 0) : (_a[4] >> count);
    int16_t d5 =
        (count & ~15) ? (_a[5] < 0 ? ~UINT16_C(0) : 0) : (_a[5] >> count);
    int16_t d6 =
        (count & ~15) ? (_a[6] < 0 ? ~UINT16_C(0) : 0) : (_a[6] >> count);
    int16_t d7 =
        (count & ~15) ? (_a[7] < 0 ? ~UINT16_C(0) : 0) : (_a[7] >> count);

    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sra_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_sra_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) (iter % 34 - 1);  // range: -1 ~ 32

    int32_t d0 =
        (count & ~31) ? (_a[0] < 0 ? ~UINT32_C(0) : 0) : _a[0] >> count;
    int32_t d1 =
        (count & ~31) ? (_a[1] < 0 ? ~UINT32_C(0) : 0) : _a[1] >> count;
    int32_t d2 =
        (count & ~31) ? (_a[2] < 0 ? ~UINT32_C(0) : 0) : _a[2] >> count;
    int32_t d3 =
        (count & ~31) ? (_a[3] < 0 ? ~UINT32_C(0) : 0) : _a[3] >> count;

    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sra_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_srai_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int32_t b = (int32_t) (iter % 18 - 1);  // range: -1 ~ 16
    int16_t d[8];
    int count = (b & ~15) ? 15 : b;

    for (int i = 0; i < 8; i++) {
        d[i] = _a[i] >> count;
    }

    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i c = _mm_srai_epi16(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_srai_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t b = (int32_t) (iter % 34 - 1);  // range: -1 ~ 32

    int32_t d[4];
    int count = (b & ~31) ? 31 : b;
    for (int i = 0; i < 4; i++) {
        d[i] = _a[i] >> count;
    }

    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i c = _mm_srai_epi32(a, b);

    return validateInt32(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_srl_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) (iter % 18 - 1);  // range: -1 ~ 16

    uint16_t d0 = (count & ~15) ? 0 : (uint16_t) (_a[0]) >> count;
    uint16_t d1 = (count & ~15) ? 0 : (uint16_t) (_a[1]) >> count;
    uint16_t d2 = (count & ~15) ? 0 : (uint16_t) (_a[2]) >> count;
    uint16_t d3 = (count & ~15) ? 0 : (uint16_t) (_a[3]) >> count;
    uint16_t d4 = (count & ~15) ? 0 : (uint16_t) (_a[4]) >> count;
    uint16_t d5 = (count & ~15) ? 0 : (uint16_t) (_a[5]) >> count;
    uint16_t d6 = (count & ~15) ? 0 : (uint16_t) (_a[6]) >> count;
    uint16_t d7 = (count & ~15) ? 0 : (uint16_t) (_a[7]) >> count;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_srl_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) (iter % 34 - 1);  // range: -1 ~ 32

    uint32_t d0 = (count & ~31) ? 0 : (uint32_t) (_a[0]) >> count;
    uint32_t d1 = (count & ~31) ? 0 : (uint32_t) (_a[1]) >> count;
    uint32_t d2 = (count & ~31) ? 0 : (uint32_t) (_a[2]) >> count;
    uint32_t d3 = (count & ~31) ? 0 : (uint32_t) (_a[3]) >> count;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_srl_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) (iter % 66 - 1);  // range: -1 ~ 64

    uint64_t d0 = (count & ~63) ? 0 : (uint64_t) (_a[0]) >> count;
    uint64_t d1 = (count & ~63) ? 0 : (uint64_t) (_a[1]) >> count;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi64(a, b);

    return validateInt64(c, d0, d1);
}

result_t test_mm_srli_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int count = (int) (iter % 18 - 1);  // range: -1 ~ 16

    int16_t d0 = count & (~15) ? 0 : (uint16_t) (_a[0]) >> count;
    int16_t d1 = count & (~15) ? 0 : (uint16_t) (_a[1]) >> count;
    int16_t d2 = count & (~15) ? 0 : (uint16_t) (_a[2]) >> count;
    int16_t d3 = count & (~15) ? 0 : (uint16_t) (_a[3]) >> count;
    int16_t d4 = count & (~15) ? 0 : (uint16_t) (_a[4]) >> count;
    int16_t d5 = count & (~15) ? 0 : (uint16_t) (_a[5]) >> count;
    int16_t d6 = count & (~15) ? 0 : (uint16_t) (_a[6]) >> count;
    int16_t d7 = count & (~15) ? 0 : (uint16_t) (_a[7]) >> count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_srli_epi16(a, count);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_srli_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int count = (int) (iter % 34 - 1);  // range: -1 ~ 32

    int32_t d0 = count & (~31) ? 0 : (uint32_t) (_a[0]) >> count;
    int32_t d1 = count & (~31) ? 0 : (uint32_t) (_a[1]) >> count;
    int32_t d2 = count & (~31) ? 0 : (uint32_t) (_a[2]) >> count;
    int32_t d3 = count & (~31) ? 0 : (uint32_t) (_a[3]) >> count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_srli_epi32(a, count);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_srli_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int count = (int) (iter % 66 - 1);  // range: -1 ~ 64

    int64_t d0 = count & (~63) ? 0 : (uint64_t) (_a[0]) >> count;
    int64_t d1 = count & (~63) ? 0 : (uint64_t) (_a[1]) >> count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_srli_epi64(a, count);

    return validateInt64(c, d0, d1);
}

result_t test_mm_srli_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int count = (iter % 5) << 2;

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        if (i >= (16 - count))
            d[i] = 0;
        else
            d[i] = _a[i + count];
    }

    __m128i a = load_m128i(_a);
    __m128i ret;
    switch (iter % 5) {
    case 0:
        ret = _mm_srli_si128(a, 0);
        break;
    case 1:
        ret = _mm_srli_si128(a, 4);
        break;
    case 2:
        ret = _mm_srli_si128(a, 8);
        break;
    case 3:
        ret = _mm_srli_si128(a, 12);
        break;
    case 4:
        ret = _mm_srli_si128(a, 16);
        break;
    }

    return validateInt8(ret, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                        d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_store_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double x = impl.mTestFloats[iter + 4];
    double y = impl.mTestFloats[iter + 6];

    __m128d a = _mm_set_pd(x, y);
    _mm_store_pd(p, a);
    ASSERT_RETURN(p[0] == y);
    ASSERT_RETURN(p[1] == x);
    return TEST_SUCCESS;
}

result_t test_mm_store_pd1(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double _a[2] = {(double) impl.mTestFloats[iter],
                    (double) impl.mTestFloats[iter + 1]};

    __m128d a = load_m128d(_a);
    _mm_store_pd1(p, a);
    ASSERT_RETURN(p[0] == impl.mTestFloats[iter]);
    ASSERT_RETURN(p[1] == impl.mTestFloats[iter]);
    return TEST_SUCCESS;
}

result_t test_mm_store_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double _a[2] = {(double) impl.mTestFloats[iter],
                    (double) impl.mTestFloats[iter + 1]};

    __m128d a = load_m128d(_a);
    _mm_store_sd(p, a);
    ASSERT_RETURN(p[0] == impl.mTestFloats[iter]);
    return TEST_SUCCESS;
}

result_t test_mm_store_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    alignas(16) int32_t p[4];

    __m128i a = load_m128i(_a);
    _mm_store_si128((__m128i *) p, a);

    return validateInt32(a, p[0], p[1], p[2], p[3]);
}

result_t test_mm_store1_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_store_pd1(impl, iter);
}

result_t test_mm_storeh_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double mem;

    __m128d a = load_m128d(p);
    _mm_storeh_pd(&mem, a);

    ASSERT_RETURN(mem == p[1]);
    return TEST_SUCCESS;
}

result_t test_mm_storel_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int64_t *p = (int64_t *) impl.mTestIntPointer1;
    __m128i mem;

    __m128i a = load_m128i(p);
    _mm_storel_epi64(&mem, a);

    ASSERT_RETURN(mem[0] == p[0]);
    return TEST_SUCCESS;
}

result_t test_mm_storel_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double mem;

    __m128d a = load_m128d(p);
    _mm_storel_pd(&mem, a);

    ASSERT_RETURN(mem == p[0]);
    return TEST_SUCCESS;
}

result_t test_mm_storer_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double mem[2];

    __m128d a = load_m128d(p);
    _mm_storer_pd(mem, a);

    __m128d res = load_m128d(mem);
    return validateDouble(res, p[1], p[0]);
}

result_t test_mm_storeu_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double x = impl.mTestFloats[iter + 4];
    double y = impl.mTestFloats[iter + 6];

    __m128d a = _mm_set_pd(x, y);
    _mm_storeu_pd(p, a);
    ASSERT_RETURN(p[0] == y);
    ASSERT_RETURN(p[1] == x);
    return TEST_SUCCESS;
}

result_t test_mm_storeu_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i b;
    __m128i a = load_m128i(_a);
    _mm_storeu_si128(&b, a);
    int32_t *_b = (int32_t *) &b;
    return validateInt32(a, _b[0], _b[1], _b[2], _b[3]);
}

result_t test_mm_storeu_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_storeu_si32. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
    return TEST_UNIMPL;
#else
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i b;
    __m128i a = load_m128i(_a);
    _mm_storeu_si32(&b, a);
    int32_t *_b = (int32_t *) &b;
    return validateInt32(b, _a[0], _b[1], _b[2], _b[3]);
#endif
}

result_t test_mm_stream_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    double p[2];

    __m128d a = load_m128d(_a);
    _mm_stream_pd(p, a);

    return validateDouble(a, p[0], p[1]);
}

result_t test_mm_stream_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    alignas(16) int32_t p[4];

    __m128i a = load_m128i(_a);
    _mm_stream_si128((__m128i *) p, a);

    return validateInt32(a, p[0], p[1], p[2], p[3]);
}

result_t test_mm_stream_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t a = (const int32_t) impl.mTestInts[iter];
    int32_t p;

    _mm_stream_si32(&p, a);

    ASSERT_RETURN(a == p)
    return TEST_SUCCESS;
}

result_t test_mm_stream_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t a = (const int64_t) impl.mTestInts[iter];
    __int64 p[1];
    _mm_stream_si64(p, a);
    ASSERT_RETURN(p[0] == a);
    return TEST_SUCCESS;
}

result_t test_mm_sub_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d0 = _a[0] - _b[0];
    int16_t d1 = _a[1] - _b[1];
    int16_t d2 = _a[2] - _b[2];
    int16_t d3 = _a[3] - _b[3];
    int16_t d4 = _a[4] - _b[4];
    int16_t d5 = _a[5] - _b[5];
    int16_t d6 = _a[6] - _b[6];
    int16_t d7 = _a[7] - _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sub_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_sub_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t dx = _a[0] - _b[0];
    int32_t dy = _a[1] - _b[1];
    int32_t dz = _a[2] - _b[2];
    int32_t dw = _a[3] - _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sub_epi32(a, b);
    return validateInt32(c, dx, dy, dz, dw);
}

result_t test_mm_sub_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (int64_t *) impl.mTestIntPointer2;
    int64_t d0 = _a[0] - _b[0];
    int64_t d1 = _a[1] - _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sub_epi64(a, b);
    return validateInt64(c, d0, d1);
}

result_t test_mm_sub_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sub_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_sub_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] - _b[0];
    double d1 = _a[1] - _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_sub_pd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_sub_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] - _b[0];
    double d1 = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_sub_sd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_sub_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t d = _a[0] - _b[0];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_sub_si64(a, b);

    return validateInt64(c, d);
}

result_t test_mm_subs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t max = 32767;
    int32_t min = -32768;
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    int16_t d[8];
    for (int i = 0; i < 8; i++) {
        int32_t res = (int32_t) _a[i] - (int32_t) _b[i];
        if (res > max)
            d[i] = max;
        else if (res < min)
            d[i] = min;
        else
            d[i] = (int16_t) res;
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_subs_epi16(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_subs_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int16_t max = 127;
    int16_t min = -128;
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        int16_t res = (int16_t) _a[i] - (int16_t) _b[i];
        if (res > max)
            d[i] = max;
        else if (res < min)
            d[i] = min;
        else
            d[i] = (int8_t) res;
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_subs_epi8(a, b);

    return validateInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                        d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_subs_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_subs_epu16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_subs_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
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

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_subs_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_ucomieq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_comieq_sd(impl, iter);
}

result_t test_mm_ucomige_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_comige_sd(impl, iter);
}

result_t test_mm_ucomigt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_comigt_sd(impl, iter);
}

result_t test_mm_ucomile_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_comile_sd(impl, iter);
}

result_t test_mm_ucomilt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_comilt_sd(impl, iter);
}

result_t test_mm_ucomineq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_comineq_sd(impl, iter);
}

result_t test_mm_undefined_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128d a = _mm_undefined_pd();
    a = _mm_xor_pd(a, a);
    return validateDouble(a, 0, 0);
}

result_t test_mm_undefined_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128i a = _mm_undefined_si128();
    a = _mm_xor_si128(a, a);
    return validateInt64(a, 0, 0);
}

result_t test_mm_unpackhi_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    int16_t i0 = _a[4];
    int16_t i1 = _b[4];
    int16_t i2 = _a[5];
    int16_t i3 = _b[5];
    int16_t i4 = _a[6];
    int16_t i5 = _b[6];
    int16_t i6 = _a[7];
    int16_t i7 = _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpackhi_epi16(a, b);

    return validateInt16(ret, i0, i1, i2, i3, i4, i5, i6, i7);
}

result_t test_mm_unpackhi_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t i0 = _a[2];
    int32_t i1 = _b[2];
    int32_t i2 = _a[3];
    int32_t i3 = _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpackhi_epi32(a, b);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_unpackhi_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t i0 = _a[1];
    int64_t i1 = _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpackhi_epi64(a, b);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_unpackhi_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;

    int8_t i0 = _a[8];
    int8_t i1 = _b[8];
    int8_t i2 = _a[9];
    int8_t i3 = _b[9];
    int8_t i4 = _a[10];
    int8_t i5 = _b[10];
    int8_t i6 = _a[11];
    int8_t i7 = _b[11];
    int8_t i8 = _a[12];
    int8_t i9 = _b[12];
    int8_t i10 = _a[13];
    int8_t i11 = _b[13];
    int8_t i12 = _a[14];
    int8_t i13 = _b[14];
    int8_t i14 = _a[15];
    int8_t i15 = _b[15];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpackhi_epi8(a, b);

    return validateInt8(ret, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
                        i12, i13, i14, i15);
}

result_t test_mm_unpackhi_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_unpackhi_pd(a, b);

    return validateDouble(ret, _a[1], _b[1]);
}

result_t test_mm_unpacklo_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    int16_t i0 = _a[0];
    int16_t i1 = _b[0];
    int16_t i2 = _a[1];
    int16_t i3 = _b[1];
    int16_t i4 = _a[2];
    int16_t i5 = _b[2];
    int16_t i6 = _a[3];
    int16_t i7 = _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpacklo_epi16(a, b);

    return validateInt16(ret, i0, i1, i2, i3, i4, i5, i6, i7);
}

result_t test_mm_unpacklo_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t i0 = _a[0];
    int32_t i1 = _b[0];
    int32_t i2 = _a[1];
    int32_t i3 = _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpacklo_epi32(a, b);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_unpacklo_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t i0 = _a[0];
    int64_t i1 = _b[0];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpacklo_epi64(a, b);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_unpacklo_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;

    int8_t i0 = _a[0];
    int8_t i1 = _b[0];
    int8_t i2 = _a[1];
    int8_t i3 = _b[1];
    int8_t i4 = _a[2];
    int8_t i5 = _b[2];
    int8_t i6 = _a[3];
    int8_t i7 = _b[3];
    int8_t i8 = _a[4];
    int8_t i9 = _b[4];
    int8_t i10 = _a[5];
    int8_t i11 = _b[5];
    int8_t i12 = _a[6];
    int8_t i13 = _b[6];
    int8_t i14 = _a[7];
    int8_t i15 = _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpacklo_epi8(a, b);

    return validateInt8(ret, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
                        i12, i13, i14, i15);
}

result_t test_mm_unpacklo_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_unpacklo_pd(a, b);

    return validateDouble(ret, _a[0], _b[0]);
}

result_t test_mm_xor_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestFloatPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestFloatPointer2;

    int64_t d0 = _a[0] ^ _b[0];
    int64_t d1 = _a[1] ^ _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_xor_pd(a, b);

    return validateDouble(c, *((double *) &d0), *((double *) &d1));
}

result_t test_mm_xor_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t d0 = _a[0] ^ _b[0];
    int64_t d1 = _a[1] ^ _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_xor_si128(a, b);

    return validateInt64(c, d0, d1);
}

/* SSE3 */
result_t test_mm_addsub_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double d0 = _a[0] - _b[0];
    double d1 = _a[1] + _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_addsub_pd(a, b);

    return validateDouble(c, d0, d1);
}

result_t test_mm_addsub_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME: The rounding mode would affect the testing result on ARM platform.
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _a[0] - _b[0];
    float f1 = _a[1] + _b[1];
    float f2 = _a[2] - _b[2];
    float f3 = _a[3] + _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_addsub_ps(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_hadd_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double f0 = _a[0] + _a[1];
    double f1 = _b[0] + _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_hadd_pd(a, b);

    return validateDouble(c, f0, f1);
}

result_t test_mm_hadd_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME: The rounding mode would affect the testing result on ARM platform.
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _a[0] + _a[1];
    float f1 = _a[2] + _a[3];
    float f2 = _b[0] + _b[1];
    float f3 = _b[2] + _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_hadd_ps(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_hsub_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double f0 = _a[0] - _a[1];
    double f1 = _b[0] - _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_hsub_pd(a, b);

    return validateDouble(c, f0, f1);
}

result_t test_mm_hsub_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME: The rounding mode would affect the testing result on ARM platform.
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _a[0] - _a[1];
    float f1 = _a[2] - _a[3];
    float f2 = _b[0] - _b[1];
    float f3 = _b[2] - _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_hsub_ps(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_lddqu_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_loadu_si128(impl, iter);
}

result_t test_mm_loaddup_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *addr = (const double *) impl.mTestFloatPointer1;

    __m128d ret = _mm_loaddup_pd(addr);

    return validateDouble(ret, addr[0], addr[0]);
}

result_t test_mm_movedup_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = load_m128d(p);
    __m128d b = _mm_movedup_pd(a);

    return validateDouble(b, p[0], p[0]);
}

result_t test_mm_movehdup_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p = impl.mTestFloatPointer1;
    __m128 a = load_m128(p);
    return validateFloat(_mm_movehdup_ps(a), p[1], p[1], p[3], p[3]);
}

result_t test_mm_moveldup_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p = impl.mTestFloatPointer1;
    __m128 a = load_m128(p);
    return validateFloat(_mm_moveldup_ps(a), p[0], p[0], p[2], p[2]);
}

/* SSSE3 */
result_t test_mm_abs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);
    __m128i c = _mm_abs_epi16(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];
    uint32_t d2 = (_a[2] < 0) ? -_a[2] : _a[2];
    uint32_t d3 = (_a[3] < 0) ? -_a[3] : _a[3];
    uint32_t d4 = (_a[4] < 0) ? -_a[4] : _a[4];
    uint32_t d5 = (_a[5] < 0) ? -_a[5] : _a[5];
    uint32_t d6 = (_a[6] < 0) ? -_a[6] : _a[6];
    uint32_t d7 = (_a[7] < 0) ? -_a[7] : _a[7];

    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_abs_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);
    __m128i c = _mm_abs_epi32(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];
    uint32_t d2 = (_a[2] < 0) ? -_a[2] : _a[2];
    uint32_t d3 = (_a[3] < 0) ? -_a[3] : _a[3];

    return validateUInt32(c, d0, d1, d2, d3);
}

result_t test_mm_abs_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);
    __m128i c = _mm_abs_epi8(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];
    uint32_t d2 = (_a[2] < 0) ? -_a[2] : _a[2];
    uint32_t d3 = (_a[3] < 0) ? -_a[3] : _a[3];
    uint32_t d4 = (_a[4] < 0) ? -_a[4] : _a[4];
    uint32_t d5 = (_a[5] < 0) ? -_a[5] : _a[5];
    uint32_t d6 = (_a[6] < 0) ? -_a[6] : _a[6];
    uint32_t d7 = (_a[7] < 0) ? -_a[7] : _a[7];
    uint32_t d8 = (_a[8] < 0) ? -_a[8] : _a[8];
    uint32_t d9 = (_a[9] < 0) ? -_a[9] : _a[9];
    uint32_t d10 = (_a[10] < 0) ? -_a[10] : _a[10];
    uint32_t d11 = (_a[11] < 0) ? -_a[11] : _a[11];
    uint32_t d12 = (_a[12] < 0) ? -_a[12] : _a[12];
    uint32_t d13 = (_a[13] < 0) ? -_a[13] : _a[13];
    uint32_t d14 = (_a[14] < 0) ? -_a[14] : _a[14];
    uint32_t d15 = (_a[15] < 0) ? -_a[15] : _a[15];

    return validateUInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                         d12, d13, d14, d15);
}

result_t test_mm_abs_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    __m64 a = load_m64(_a);
    __m64 c = _mm_abs_pi16(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];
    uint32_t d2 = (_a[2] < 0) ? -_a[2] : _a[2];
    uint32_t d3 = (_a[3] < 0) ? -_a[3] : _a[3];

    return validateUInt16(c, d0, d1, d2, d3);
}

result_t test_mm_abs_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m64 a = load_m64(_a);
    __m64 c = _mm_abs_pi32(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];

    return validateUInt32(c, d0, d1);
}

result_t test_mm_abs_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    __m64 a = load_m64(_a);
    __m64 c = _mm_abs_pi8(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];
    uint32_t d2 = (_a[2] < 0) ? -_a[2] : _a[2];
    uint32_t d3 = (_a[3] < 0) ? -_a[3] : _a[3];
    uint32_t d4 = (_a[4] < 0) ? -_a[4] : _a[4];
    uint32_t d5 = (_a[5] < 0) ? -_a[5] : _a[5];
    uint32_t d6 = (_a[6] < 0) ? -_a[6] : _a[6];
    uint32_t d7 = (_a[7] < 0) ? -_a[7] : _a[7];

    return validateUInt8(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_alignr_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if defined(__clang__)
    return TEST_UNIMPL;
#else
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    unsigned int shift = (iter % 5) << 3;
    uint8_t d[32];

    if (shift >= 32) {
        memset((void *) d, 0, sizeof(d));
    } else {
        memcpy((void *) d, (const void *) _b, 16);
        memcpy((void *) (d + 16), (const void *) _a, 16);
        // shifting
        for (uint x = 0; x < sizeof(d); x++) {
            if (x + shift >= sizeof(d))
                d[x] = 0;
            else
                d[x] = d[x + shift];
        }
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret;
    switch (iter % 5) {
    case 0:
        ret = _mm_alignr_epi8(a, b, 0);
        break;
    case 1:
        ret = _mm_alignr_epi8(a, b, 8);
        break;
    case 2:
        ret = _mm_alignr_epi8(a, b, 16);
        break;
    case 3:
        ret = _mm_alignr_epi8(a, b, 24);
        break;
    case 4:
        ret = _mm_alignr_epi8(a, b, 32);
        break;
    }

    return validateUInt8(ret, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                         d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
#endif
}

result_t test_mm_alignr_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if defined(__clang__)
    return TEST_UNIMPL;
#else
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    unsigned int shift = (iter % 3) << 3;
    uint8_t d[16];

    if (shift >= 16) {
        memset((void *) d, 0, sizeof(d));
    } else {
        memcpy((void *) d, (const void *) _b, 8);
        memcpy((void *) (d + 8), (const void *) _a, 8);
        // shifting
        for (uint x = 0; x < sizeof(d); x++) {
            if (x + shift >= sizeof(d))
                d[x] = 0;
            else
                d[x] = d[x + shift];
        }
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret;
    switch (iter % 3) {
    case 0:
        ret = _mm_alignr_pi8(a, b, 0);
        break;
    case 1:
        ret = _mm_alignr_pi8(a, b, 8);
        break;
    case 2:
        ret = _mm_alignr_pi8(a, b, 16);
        break;
    }

    return validateUInt8(ret, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
#endif
}

result_t test_mm_hadd_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d0 = _a[0] + _a[1];
    int16_t d1 = _a[2] + _a[3];
    int16_t d2 = _a[4] + _a[5];
    int16_t d3 = _a[6] + _a[7];
    int16_t d4 = _b[0] + _b[1];
    int16_t d5 = _b[2] + _b[3];
    int16_t d6 = _b[4] + _b[5];
    int16_t d7 = _b[6] + _b[7];
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_hadd_epi16(a, b);
    return validateInt16(ret, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_hadd_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;
    int32_t d0 = _a[0] + _a[1];
    int32_t d1 = _a[2] + _a[3];
    int32_t d2 = _b[0] + _b[1];
    int32_t d3 = _b[2] + _b[3];
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_hadd_epi32(a, b);
    return validateInt32(ret, d0, d1, d2, d3);
}

result_t test_mm_hadd_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d0 = _a[0] + _a[1];
    int16_t d1 = _a[2] + _a[3];
    int16_t d2 = _b[0] + _b[1];
    int16_t d3 = _b[2] + _b[3];
    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_hadd_pi16(a, b);
    return validateInt16(ret, d0, d1, d2, d3);
}

result_t test_mm_hadd_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;
    int32_t d0 = _a[0] + _a[1];
    int32_t d1 = _b[0] + _b[1];
    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_hadd_pi32(a, b);
    return validateInt32(ret, d0, d1);
}

result_t test_mm_hadds_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer1;

    int16_t d16[8];
    int32_t d32[8];
    d32[0] = (int32_t) _a[0] + (int32_t) _a[1];
    d32[1] = (int32_t) _a[2] + (int32_t) _a[3];
    d32[2] = (int32_t) _a[4] + (int32_t) _a[5];
    d32[3] = (int32_t) _a[6] + (int32_t) _a[7];
    d32[4] = (int32_t) _b[0] + (int32_t) _b[1];
    d32[5] = (int32_t) _b[2] + (int32_t) _b[3];
    d32[6] = (int32_t) _b[4] + (int32_t) _b[5];
    d32[7] = (int32_t) _b[6] + (int32_t) _b[7];
    for (int i = 0; i < 8; i++) {
        if (d32[i] > (int32_t) INT16_MAX)
            d16[i] = INT16_MAX;
        else if (d32[i] < (int32_t) INT16_MIN)
            d16[i] = INT16_MIN;
        else
            d16[i] = (int16_t) d32[i];
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_hadds_epi16(a, b);

    return validateInt16(c, d16[0], d16[1], d16[2], d16[3], d16[4], d16[5],
                         d16[6], d16[7]);
}

result_t test_mm_hadds_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer1;

    int16_t d16[8];
    int32_t d32[8];
    d32[0] = (int32_t) _a[0] + (int32_t) _a[1];
    d32[1] = (int32_t) _a[2] + (int32_t) _a[3];
    d32[2] = (int32_t) _b[0] + (int32_t) _b[1];
    d32[3] = (int32_t) _b[2] + (int32_t) _b[3];
    for (int i = 0; i < 8; i++) {
        if (d32[i] > (int32_t) INT16_MAX)
            d16[i] = INT16_MAX;
        else if (d32[i] < (int32_t) INT16_MIN)
            d16[i] = INT16_MIN;
        else
            d16[i] = (int16_t) d32[i];
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_hadds_pi16(a, b);

    return validateInt16(c, d16[0], d16[1], d16[2], d16[3]);
}

result_t test_mm_hsub_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer1;

    int16_t d0 = _a[0] - _a[1];
    int16_t d1 = _a[2] - _a[3];
    int16_t d2 = _a[4] - _a[5];
    int16_t d3 = _a[6] - _a[7];
    int16_t d4 = _b[0] - _b[1];
    int16_t d5 = _b[2] - _b[3];
    int16_t d6 = _b[4] - _b[5];
    int16_t d7 = _b[6] - _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_hsub_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_hsub_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer1;

    int32_t d0 = _a[0] - _a[1];
    int32_t d1 = _a[2] - _a[3];
    int32_t d2 = _b[0] - _b[1];
    int32_t d3 = _b[2] - _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_hsub_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_hsub_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    int16_t d0 = _a[0] - _a[1];
    int16_t d1 = _a[2] - _a[3];
    int16_t d2 = _b[0] - _b[1];
    int16_t d3 = _b[2] - _b[3];
    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_hsub_pi16(a, b);

    return validateInt16(c, d0, d1, d2, d3);
}

result_t test_mm_hsub_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;

    int32_t d0 = _a[0] - _a[1];
    int32_t d1 = _b[0] - _b[1];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_hsub_pi32(a, b);

    return validateInt32(c, d0, d1);
}

result_t test_mm_hsubs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer1;

    int16_t d16[8];
    int32_t d32[8];
    d32[0] = (int32_t) _a[0] - (int32_t) _a[1];
    d32[1] = (int32_t) _a[2] - (int32_t) _a[3];
    d32[2] = (int32_t) _a[4] - (int32_t) _a[5];
    d32[3] = (int32_t) _a[6] - (int32_t) _a[7];
    d32[4] = (int32_t) _b[0] - (int32_t) _b[1];
    d32[5] = (int32_t) _b[2] - (int32_t) _b[3];
    d32[6] = (int32_t) _b[4] - (int32_t) _b[5];
    d32[7] = (int32_t) _b[6] - (int32_t) _b[7];
    for (int i = 0; i < 8; i++) {
        if (d32[i] > (int32_t) INT16_MAX)
            d16[i] = INT16_MAX;
        else if (d32[i] < (int32_t) INT16_MIN)
            d16[i] = INT16_MIN;
        else
            d16[i] = (int16_t) d32[i];
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_hsubs_epi16(a, b);

    return validateInt16(c, d16[0], d16[1], d16[2], d16[3], d16[4], d16[5],
                         d16[6], d16[7]);
}

result_t test_mm_hsubs_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer1;

    int32_t _d[4];
    _d[0] = (int32_t) _a[0] - (int32_t) _a[1];
    _d[1] = (int32_t) _a[2] - (int32_t) _a[3];
    _d[2] = (int32_t) _b[0] - (int32_t) _b[1];
    _d[3] = (int32_t) _b[2] - (int32_t) _b[3];

    for (int i = 0; i < 4; i++) {
        if (_d[i] > (int32_t) INT16_MAX) {
            _d[i] = INT16_MAX;
        } else if (_d[i] < (int32_t) INT16_MIN) {
            _d[i] = INT16_MIN;
        }
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_hsubs_pi16(a, b);

    return validateInt16(c, _d[0], _d[1], _d[2], _d[3]);
}

result_t test_mm_maddubs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    int32_t d0 = (int32_t) (_a[0] * _b[0]);
    int32_t d1 = (int32_t) (_a[1] * _b[1]);
    int32_t d2 = (int32_t) (_a[2] * _b[2]);
    int32_t d3 = (int32_t) (_a[3] * _b[3]);
    int32_t d4 = (int32_t) (_a[4] * _b[4]);
    int32_t d5 = (int32_t) (_a[5] * _b[5]);
    int32_t d6 = (int32_t) (_a[6] * _b[6]);
    int32_t d7 = (int32_t) (_a[7] * _b[7]);
    int32_t d8 = (int32_t) (_a[8] * _b[8]);
    int32_t d9 = (int32_t) (_a[9] * _b[9]);
    int32_t d10 = (int32_t) (_a[10] * _b[10]);
    int32_t d11 = (int32_t) (_a[11] * _b[11]);
    int32_t d12 = (int32_t) (_a[12] * _b[12]);
    int32_t d13 = (int32_t) (_a[13] * _b[13]);
    int32_t d14 = (int32_t) (_a[14] * _b[14]);
    int32_t d15 = (int32_t) (_a[15] * _b[15]);

    int16_t e0 = saturate_16(d0 + d1);
    int16_t e1 = saturate_16(d2 + d3);
    int16_t e2 = saturate_16(d4 + d5);
    int16_t e3 = saturate_16(d6 + d7);
    int16_t e4 = saturate_16(d8 + d9);
    int16_t e5 = saturate_16(d10 + d11);
    int16_t e6 = saturate_16(d12 + d13);
    int16_t e7 = saturate_16(d14 + d15);

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_maddubs_epi16(a, b);
    return validateInt16(c, e0, e1, e2, e3, e4, e5, e6, e7);
}

result_t test_mm_maddubs_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    int16_t d0 = (int16_t) (_a[0] * _b[0]);
    int16_t d1 = (int16_t) (_a[1] * _b[1]);
    int16_t d2 = (int16_t) (_a[2] * _b[2]);
    int16_t d3 = (int16_t) (_a[3] * _b[3]);
    int16_t d4 = (int16_t) (_a[4] * _b[4]);
    int16_t d5 = (int16_t) (_a[5] * _b[5]);
    int16_t d6 = (int16_t) (_a[6] * _b[6]);
    int16_t d7 = (int16_t) (_a[7] * _b[7]);

    int16_t e0 = saturate_16(d0 + d1);
    int16_t e1 = saturate_16(d2 + d3);
    int16_t e2 = saturate_16(d4 + d5);
    int16_t e3 = saturate_16(d6 + d7);

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_maddubs_pi16(a, b);

    return validateInt16(c, e0, e1, e2, e3);
}

result_t test_mm_mulhrs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    int32_t _c[8];
    for (int i = 0; i < 8; i++) {
        _c[i] =
            (((((int32_t) _a[i] * (int32_t) _b[i]) >> 14) + 1) & 0x1FFFE) >> 1;
    }
    __m128i c = _mm_mulhrs_epi16(a, b);

    return validateInt16(c, _c[0], _c[1], _c[2], _c[3], _c[4], _c[5], _c[6],
                         _c[7]);
}

result_t test_mm_mulhrs_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    int32_t _c[4];
    for (int i = 0; i < 4; i++) {
        _c[i] =
            (((((int32_t) _a[i] * (int32_t) _b[i]) >> 14) + 1) & 0x1FFFE) >> 1;
    }
    __m64 c = _mm_mulhrs_pi16(a, b);

    return validateInt16(c, _c[0], _c[1], _c[2], _c[3]);
}

result_t test_mm_shuffle_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    int8_t dst[16];

    for (int i = 0; i < 16; i++) {
        if (_b[i] & 0x80) {
            dst[i] = 0;
        } else {
            dst[i] = _a[_b[i] & 0x0F];
        }
    }
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_shuffle_epi8(a, b);

    return validateInt8(ret, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5],
                        dst[6], dst[7], dst[8], dst[9], dst[10], dst[11],
                        dst[12], dst[13], dst[14], dst[15]);
}

result_t test_mm_shuffle_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    int8_t dst[8];

    for (int i = 0; i < 8; i++) {
        if (_b[i] & 0x80) {
            dst[i] = 0;
        } else {
            dst[i] = _a[_b[i] & 0x07];
        }
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_shuffle_pi8(a, b);

    return validateInt8(ret, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5],
                        dst[6], dst[7]);
}

result_t test_mm_sign_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    int16_t d[8];
    for (int i = 0; i < 8; i++) {
        if (_b[i] < 0) {
            d[i] = -_a[i];
        } else if (_b[i] == 0) {
            d[i] = 0;
        } else {
            d[i] = _a[i];
        }
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sign_epi16(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_sign_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t d[4];
    for (int i = 0; i < 4; i++) {
        if (_b[i] < 0) {
            d[i] = -_a[i];
        } else if (_b[i] == 0) {
            d[i] = 0;
        } else {
            d[i] = _a[i];
        }
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sign_epi32(a, b);

    return validateInt32(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_sign_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        if (_b[i] < 0) {
            d[i] = -_a[i];
        } else if (_b[i] == 0) {
            d[i] = 0;
        } else {
            d[i] = _a[i];
        }
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sign_epi8(a, b);

    return validateInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                        d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_sign_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;

    int16_t d[4];
    for (int i = 0; i < 4; i++) {
        if (_b[i] < 0) {
            d[i] = -_a[i];
        } else if (_b[i] == 0) {
            d[i] = 0;
        } else {
            d[i] = _a[i];
        }
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_sign_pi16(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_sign_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t d[2];
    for (int i = 0; i < 2; i++) {
        if (_b[i] < 0) {
            d[i] = -_a[i];
        } else if (_b[i] == 0) {
            d[i] = 0;
        } else {
            d[i] = _a[i];
        }
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_sign_pi32(a, b);

    return validateInt32(c, d[0], d[1]);
}

result_t test_mm_sign_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;

    int8_t d[8];
    for (int i = 0; i < 8; i++) {
        if (_b[i] < 0) {
            d[i] = -_a[i];
        } else if (_b[i] == 0) {
            d[i] = 0;
        } else {
            d[i] = _a[i];
        }
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_sign_pi8(a, b);

    return validateInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

/* SSE4.1 */
result_t test_mm_blend_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    const int mask = 104;

    int16_t _c[8];
    for (int j = 0; j < 8; j++) {
        if ((mask >> j) & 0x1) {
            _c[j] = _b[j];
        } else {
            _c[j] = _a[j];
        }
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_blend_epi16(a, b, mask);

    return validateInt16(c, _c[0], _c[1], _c[2], _c[3], _c[4], _c[5], _c[6],
                         _c[7]);
}

result_t test_mm_blend_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    // the last argument must be a 2-bit immediate
    const int mask = 3;

    double _c[2];
    for (int j = 0; j < 2; j++) {
        if ((mask >> j) & 0x1) {
            _c[j] = _b[j];
        } else {
            _c[j] = _a[j];
        }
    }

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_blend_pd(a, b, mask);

    return validateDouble(c, _c[0], _c[1]);
}

result_t test_mm_blend_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    const char mask = (char) iter;

    float _c[4];
    for (int i = 0; i < 4; i++) {
        if (mask & (1 << i)) {
            _c[i] = _b[i];
        } else {
            _c[i] = _a[i];
        }
    }

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);

    // gcc and clang can't compile call to _mm_blend_ps with 3rd argument as
    // integer type due 4 bit size limitation and test framework doesn't support
    // compile time constant so for testing decided explicit define all 16
    // possible values
    __m128 c;
    switch (mask & 0xF) {
    case 0:
        c = _mm_blend_ps(a, b, 0);
        break;
    case 1:
        c = _mm_blend_ps(a, b, 1);
        break;
    case 2:
        c = _mm_blend_ps(a, b, 2);
        break;
    case 3:
        c = _mm_blend_ps(a, b, 3);
        break;

    case 4:
        c = _mm_blend_ps(a, b, 4);
        break;
    case 5:
        c = _mm_blend_ps(a, b, 5);
        break;
    case 6:
        c = _mm_blend_ps(a, b, 6);
        break;
    case 7:
        c = _mm_blend_ps(a, b, 7);
        break;

    case 8:
        c = _mm_blend_ps(a, b, 8);
        break;
    case 9:
        c = _mm_blend_ps(a, b, 9);
        break;
    case 10:
        c = _mm_blend_ps(a, b, 10);
        break;
    case 11:
        c = _mm_blend_ps(a, b, 11);
        break;

    case 12:
        c = _mm_blend_ps(a, b, 12);
        break;
    case 13:
        c = _mm_blend_ps(a, b, 13);
        break;
    case 14:
        c = _mm_blend_ps(a, b, 14);
        break;
    case 15:
        c = _mm_blend_ps(a, b, 15);
        break;
    }
    return validateFloat(c, _c[0], _c[1], _c[2], _c[3]);
}

result_t test_mm_blendv_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    const int8_t _mask[16] = {(const int8_t) impl.mTestInts[iter],
                              (const int8_t) impl.mTestInts[iter + 1],
                              (const int8_t) impl.mTestInts[iter + 2],
                              (const int8_t) impl.mTestInts[iter + 3],
                              (const int8_t) impl.mTestInts[iter + 4],
                              (const int8_t) impl.mTestInts[iter + 5],
                              (const int8_t) impl.mTestInts[iter + 6],
                              (const int8_t) impl.mTestInts[iter + 7]};

    int8_t _c[16];
    for (int i = 0; i < 16; i++) {
        if (_mask[i] >> 7) {
            _c[i] = _b[i];
        } else {
            _c[i] = _a[i];
        }
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i mask = load_m128i(_mask);
    __m128i c = _mm_blendv_epi8(a, b, mask);

    return validateInt8(c, _c[0], _c[1], _c[2], _c[3], _c[4], _c[5], _c[6],
                        _c[7], _c[8], _c[9], _c[10], _c[11], _c[12], _c[13],
                        _c[14], _c[15]);
}

result_t test_mm_blendv_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    const double _mask[] = {(double) impl.mTestFloats[iter],
                            (double) impl.mTestFloats[iter + 1]};

    double _c[2];
    for (int i = 0; i < 2; i++) {
        // signed shift right would return a result which is either all 1's from
        // negative numbers or all 0's from positive numbers
        if ((*(const int64_t *) (_mask + i)) >> 63) {
            _c[i] = _b[i];
        } else {
            _c[i] = _a[i];
        }
    }

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d mask = load_m128d(_mask);

    __m128d c = _mm_blendv_pd(a, b, mask);

    return validateDouble(c, _c[0], _c[1]);
}

result_t test_mm_blendv_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    const float _mask[] = {impl.mTestFloats[iter], impl.mTestFloats[iter + 1],
                           impl.mTestFloats[iter + 2],
                           impl.mTestFloats[iter + 3]};

    float _c[4];
    for (int i = 0; i < 4; i++) {
        // signed shift right would return a result which is either all 1's from
        // negative numbers or all 0's from positive numbers
        if ((*(const int32_t *) (_mask + i)) >> 31) {
            _c[i] = _b[i];
        } else {
            _c[i] = _a[i];
        }
    }

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 mask = load_m128(_mask);

    __m128 c = _mm_blendv_ps(a, b, mask);

    return validateFloat(c, _c[0], _c[1], _c[2], _c[3]);
}

result_t test_mm_ceil_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    double dx = ceil(_a[0]);
    double dy = ceil(_a[1]);

    __m128d a = load_m128d(_a);
    __m128d ret = _mm_ceil_pd(a);

    return validateDouble(ret, dx, dy);
}

result_t test_mm_ceil_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    float dx = ceilf(_a[0]);
    float dy = ceilf(_a[1]);
    float dz = ceilf(_a[2]);
    float dw = ceilf(_a[3]);

    __m128 a = _mm_load_ps(_a);
    __m128 c = _mm_ceil_ps(a);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_ceil_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double dx = ceil(_b[0]);
    double dy = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_ceil_sd(a, b);

    return validateDouble(ret, dx, dy);
}

result_t test_mm_ceil_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f0 = ceilf(_b[0]);

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_ceil_ss(a, b);

    return validateFloat(c, f0, _a[1], _a[2], _a[3]);
}

result_t test_mm_cmpeq_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;
    int64_t d0 = (_a[0] == _b[0]) ? 0xffffffffffffffff : 0x0;
    int64_t d1 = (_a[1] == _b[1]) ? 0xffffffffffffffff : 0x0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpeq_epi64(a, b);
    return validateInt64(c, d0, d1);
}

result_t test_mm_cvtepi16_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;

    int32_t i0 = (int32_t) _a[0];
    int32_t i1 = (int32_t) _a[1];
    int32_t i2 = (int32_t) _a[2];
    int32_t i3 = (int32_t) _a[3];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi16_epi32(a);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_cvtepi16_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi16_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepi32_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi32_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepi8_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    int16_t i0 = (int16_t) _a[0];
    int16_t i1 = (int16_t) _a[1];
    int16_t i2 = (int16_t) _a[2];
    int16_t i3 = (int16_t) _a[3];
    int16_t i4 = (int16_t) _a[4];
    int16_t i5 = (int16_t) _a[5];
    int16_t i6 = (int16_t) _a[6];
    int16_t i7 = (int16_t) _a[7];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi8_epi16(a);

    return validateInt16(ret, i0, i1, i2, i3, i4, i5, i6, i7);
}

result_t test_mm_cvtepi8_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    int32_t i0 = (int32_t) _a[0];
    int32_t i1 = (int32_t) _a[1];
    int32_t i2 = (int32_t) _a[2];
    int32_t i3 = (int32_t) _a[3];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi8_epi32(a);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_cvtepi8_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi8_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepu16_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;

    int32_t i0 = (int32_t) _a[0];
    int32_t i1 = (int32_t) _a[1];
    int32_t i2 = (int32_t) _a[2];
    int32_t i3 = (int32_t) _a[3];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu16_epi32(a);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_cvtepu16_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu16_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepu32_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu32_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepu8_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;

    int16_t i0 = (int16_t) _a[0];
    int16_t i1 = (int16_t) _a[1];
    int16_t i2 = (int16_t) _a[2];
    int16_t i3 = (int16_t) _a[3];
    int16_t i4 = (int16_t) _a[4];
    int16_t i5 = (int16_t) _a[5];
    int16_t i6 = (int16_t) _a[6];
    int16_t i7 = (int16_t) _a[7];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu8_epi16(a);

    return validateInt16(ret, i0, i1, i2, i3, i4, i5, i6, i7);
}

result_t test_mm_cvtepu8_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;

    int32_t i0 = (int32_t) _a[0];
    int32_t i1 = (int32_t) _a[1];
    int32_t i2 = (int32_t) _a[2];
    int32_t i3 = (int32_t) _a[3];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu8_epi32(a);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_cvtepu8_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu8_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_dp_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    // FIXME: Different immediate value should be tested
    const int imm = 0xFF;
    double d[2];
    double sum = 0;

    for (size_t i = 0; i < 2; i++)
        sum += ((imm) & (1 << (i + 4))) ? _a[i] * _b[i] : 0;
    for (size_t i = 0; i < 2; i++)
        d[i] = (imm & (1 << i)) ? sum : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_dp_pd(a, b, imm);

    return validateDouble(ret, d[0], d[1]);
}

result_t test_mm_dp_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    const int imm = 0xFF;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 out = _mm_dp_ps(a, b, imm);

    float r[4]; /* the reference */
    float sum = 0;

    for (size_t i = 0; i < 4; i++)
        sum += ((imm) & (1 << (i + 4))) ? _a[i] * _b[i] : 0;
    for (size_t i = 0; i < 4; i++)
        r[i] = (imm & (1 << i)) ? sum : 0;

    /* the epsilon has to be large enough, otherwise test suite fails. */
    return validateFloatEpsilon(out, r[0], r[1], r[2], r[3], 2050.0f);
}

result_t test_mm_extract_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t *_a = (int32_t *) impl.mTestIntPointer1;
    const int idx = iter & 0x3;
    __m128i a = load_m128i(_a);
    int c;
    switch (idx) {
    case 0:
        c = _mm_extract_epi32(a, 0);
        break;
    case 1:
        c = _mm_extract_epi32(a, 1);
        break;
    case 2:
        c = _mm_extract_epi32(a, 2);
        break;
    case 3:
        c = _mm_extract_epi32(a, 3);
        break;
    }

    ASSERT_RETURN(c == *(_a + idx));
    return TEST_SUCCESS;
}

result_t test_mm_extract_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int64_t *_a = (int64_t *) impl.mTestIntPointer1;

    __m128i a = load_m128i(_a);
    __int64_t c;

    switch (iter & 0x1) {
    case 0:
        c = _mm_extract_epi64(a, 0);
        break;
    case 1:
        c = _mm_extract_epi64(a, 1);
        break;
    }

    ASSERT_RETURN(c == *(_a + (iter & 1)));
    return TEST_SUCCESS;
}

result_t test_mm_extract_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint8_t *_a = (uint8_t *) impl.mTestIntPointer1;
    const int idx = iter & 0x7;

    __m128i a = load_m128i(_a);
    int c;
    switch (idx) {
    case 0:
        c = _mm_extract_epi8(a, 0);
        break;
    case 1:
        c = _mm_extract_epi8(a, 1);
        break;
    case 2:
        c = _mm_extract_epi8(a, 2);
        break;
    case 3:
        c = _mm_extract_epi8(a, 3);
        break;
    case 4:
        c = _mm_extract_epi8(a, 4);
        break;
    case 5:
        c = _mm_extract_epi8(a, 5);
        break;
    case 6:
        c = _mm_extract_epi8(a, 6);
        break;
    case 7:
        c = _mm_extract_epi8(a, 7);
        break;
    }

    ASSERT_RETURN(c == *(_a + idx));
    return TEST_SUCCESS;
}

result_t test_mm_extract_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    __m128 a = _mm_load_ps(_a);
    int32_t c;

    switch (iter & 0x3) {
    case 0:
        c = _mm_extract_ps(a, 0);
        break;
    case 1:
        c = _mm_extract_ps(a, 1);
        break;
    case 2:
        c = _mm_extract_ps(a, 2);
        break;
    case 3:
        c = _mm_extract_ps(a, 3);
        break;
    }

    ASSERT_RETURN(c == *(const int32_t *) (_a + (iter & 0x3)));
    return TEST_SUCCESS;
}

result_t test_mm_floor_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    double dx = floor(_a[0]);
    double dy = floor(_a[1]);

    __m128d a = load_m128d(_a);
    __m128d ret = _mm_floor_pd(a);

    return validateDouble(ret, dx, dy);
}

result_t test_mm_floor_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    float dx = floorf(_a[0]);
    float dy = floorf(_a[1]);
    float dz = floorf(_a[2]);
    float dw = floorf(_a[3]);

    __m128 a = load_m128(_a);
    __m128 c = _mm_floor_ps(a);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_floor_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double dx = floor(_b[0]);
    double dy = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_floor_sd(a, b);

    return validateDouble(ret, dx, dy);
}

result_t test_mm_floor_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f0 = floorf(_b[0]);

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_floor_ss(a, b);

    return validateFloat(c, f0, _a[1], _a[2], _a[3]);
}

result_t test_mm_insert_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t insert = (int32_t) *impl.mTestIntPointer2;
    const int imm8 = 2;

    int32_t d[4];
    for (int i = 0; i < 4; i++) {
        d[i] = _a[i];
    }
    d[imm8] = insert;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_insert_epi32(a, (int) insert, imm8);
    return validateInt32(b, d[0], d[1], d[2], d[3]);
}

result_t test_mm_insert_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    int64_t insert = (int64_t) *impl.mTestIntPointer2;
    const int imm8 = 1;

    int64_t d[2];

    d[0] = _a[0];
    d[1] = _a[1];
    d[imm8] = insert;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_insert_epi64(a, insert, imm8);
    return validateInt64(b, d[0], d[1]);
}

result_t test_mm_insert_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t insert = (int8_t) *impl.mTestIntPointer2;
    const int imm8 = 2;

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        d[i] = _a[i];
    }
    d[imm8] = insert;

    __m128i a = load_m128i(_a);
    __m128i b = _mm_insert_epi8(a, insert, imm8);
    return validateInt8(b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                        d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_insert_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    const uint8_t imm = 0x1 << 6 | 0x2 << 4 | 0x1 << 2;

    float d[4] = {_a[0], _a[1], _a[2], _a[3]};
    d[(imm >> 4) & 0x3] = _b[(imm >> 6) & 0x3];

    for (int j = 0; j < 4; j++) {
        if (imm & (1 << j)) {
            d[j] = 0;
        }
    }

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_insert_ps(a, b, imm);

    return validateFloat(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_max_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    int32_t d1 = _a[1] > _b[1] ? _a[1] : _b[1];
    int32_t d2 = _a[2] > _b[2] ? _a[2] : _b[2];
    int32_t d3 = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_max_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_max_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    int8_t d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    int8_t d1 = _a[1] > _b[1] ? _a[1] : _b[1];
    int8_t d2 = _a[2] > _b[2] ? _a[2] : _b[2];
    int8_t d3 = _a[3] > _b[3] ? _a[3] : _b[3];
    int8_t d4 = _a[4] > _b[4] ? _a[4] : _b[4];
    int8_t d5 = _a[5] > _b[5] ? _a[5] : _b[5];
    int8_t d6 = _a[6] > _b[6] ? _a[6] : _b[6];
    int8_t d7 = _a[7] > _b[7] ? _a[7] : _b[7];
    int8_t d8 = _a[8] > _b[8] ? _a[8] : _b[8];
    int8_t d9 = _a[9] > _b[9] ? _a[9] : _b[9];
    int8_t d10 = _a[10] > _b[10] ? _a[10] : _b[10];
    int8_t d11 = _a[11] > _b[11] ? _a[11] : _b[11];
    int8_t d12 = _a[12] > _b[12] ? _a[12] : _b[12];
    int8_t d13 = _a[13] > _b[13] ? _a[13] : _b[13];
    int8_t d14 = _a[14] > _b[14] ? _a[14] : _b[14];
    int8_t d15 = _a[15] > _b[15] ? _a[15] : _b[15];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_max_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_max_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;

    uint16_t d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    uint16_t d1 = _a[1] > _b[1] ? _a[1] : _b[1];
    uint16_t d2 = _a[2] > _b[2] ? _a[2] : _b[2];
    uint16_t d3 = _a[3] > _b[3] ? _a[3] : _b[3];
    uint16_t d4 = _a[4] > _b[4] ? _a[4] : _b[4];
    uint16_t d5 = _a[5] > _b[5] ? _a[5] : _b[5];
    uint16_t d6 = _a[6] > _b[6] ? _a[6] : _b[6];
    uint16_t d7 = _a[7] > _b[7] ? _a[7] : _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_max_epu16(a, b);

    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_max_epu32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;
    const uint32_t *_b = (const uint32_t *) impl.mTestIntPointer2;

    uint32_t d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    uint32_t d1 = _a[1] > _b[1] ? _a[1] : _b[1];
    uint32_t d2 = _a[2] > _b[2] ? _a[2] : _b[2];
    uint32_t d3 = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_max_epu32(a, b);

    return validateUInt32(c, d0, d1, d2, d3);
}

result_t test_mm_min_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    int32_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
    int32_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
    int32_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_min_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;

    int8_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    int8_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
    int8_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
    int8_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];
    int8_t d4 = _a[4] < _b[4] ? _a[4] : _b[4];
    int8_t d5 = _a[5] < _b[5] ? _a[5] : _b[5];
    int8_t d6 = _a[6] < _b[6] ? _a[6] : _b[6];
    int8_t d7 = _a[7] < _b[7] ? _a[7] : _b[7];
    int8_t d8 = _a[8] < _b[8] ? _a[8] : _b[8];
    int8_t d9 = _a[9] < _b[9] ? _a[9] : _b[9];
    int8_t d10 = _a[10] < _b[10] ? _a[10] : _b[10];
    int8_t d11 = _a[11] < _b[11] ? _a[11] : _b[11];
    int8_t d12 = _a[12] < _b[12] ? _a[12] : _b[12];
    int8_t d13 = _a[13] < _b[13] ? _a[13] : _b[13];
    int8_t d14 = _a[14] < _b[14] ? _a[14] : _b[14];
    int8_t d15 = _a[15] < _b[15] ? _a[15] : _b[15];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_min_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_min_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;

    uint16_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    uint16_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
    uint16_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
    uint16_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];
    uint16_t d4 = _a[4] < _b[4] ? _a[4] : _b[4];
    uint16_t d5 = _a[5] < _b[5] ? _a[5] : _b[5];
    uint16_t d6 = _a[6] < _b[6] ? _a[6] : _b[6];
    uint16_t d7 = _a[7] < _b[7] ? _a[7] : _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epu16(a, b);

    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_min_epu32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;
    const uint32_t *_b = (const uint32_t *) impl.mTestIntPointer2;

    uint32_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    uint32_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
    uint32_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
    uint32_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epu32(a, b);

    return validateUInt32(c, d0, d1, d2, d3);
}

result_t test_mm_minpos_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
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

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_minpos_epu16(a);
    return validateUInt16(ret, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_mpsadbw_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    const uint8_t imm = impl.mTestInts[iter];

    uint8_t a_offset = ((imm >> 2) & 0x1) * 4;
    uint8_t b_offset = (imm & 0x3) * 4;

    uint16_t d[8] = {};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            d[i] += abs(_a[(a_offset + i) + j] - _b[b_offset + j]);
        }
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c;
    switch (imm & 0x7) {
    case 0:
        c = _mm_mpsadbw_epu8(a, b, 0);
        break;
    case 1:
        c = _mm_mpsadbw_epu8(a, b, 1);
        break;
    case 2:
        c = _mm_mpsadbw_epu8(a, b, 2);
        break;
    case 3:
        c = _mm_mpsadbw_epu8(a, b, 3);
        break;
    case 4:
        c = _mm_mpsadbw_epu8(a, b, 4);
        break;
    case 5:
        c = _mm_mpsadbw_epu8(a, b, 5);
        break;
    case 6:
        c = _mm_mpsadbw_epu8(a, b, 6);
        break;
    case 7:
        c = _mm_mpsadbw_epu8(a, b, 7);
        break;

    default:
        return TEST_FAIL;
    }

    return validateUInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_mul_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int64_t dx = (int64_t) (_a[0]) * (int64_t) (_b[0]);
    int64_t dy = (int64_t) (_a[2]) * (int64_t) (_b[2]);

    __m128i a = _mm_loadu_si128((const __m128i *) _a);
    __m128i b = _mm_loadu_si128((const __m128i *) _b);
    __m128i r = _mm_mul_epi32(a, b);

    return validateInt64(r, dx, dy);
}

result_t test_mm_mullo_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t d[4];

    for (int i = 0; i < 4; i++) {
        d[i] = (int32_t) ((int64_t) _a[i] * (int64_t) _b[i]);
    }
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_mullo_epi32(a, b);
    return validateInt32(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_packus_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint16_t max = UINT16_MAX;
    uint16_t min = 0;
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    uint16_t d[8];
    for (int i = 0; i < 4; i++) {
        if (_a[i] > (int32_t) max)
            d[i] = max;
        else if (_a[i] < (int32_t) min)
            d[i] = min;
        else
            d[i] = (uint16_t) _a[i];
    }
    for (int i = 0; i < 4; i++) {
        if (_b[i] > (int32_t) max)
            d[i + 4] = max;
        else if (_b[i] < (int32_t) min)
            d[i + 4] = min;
        else
            d[i + 4] = (uint16_t) _b[i];
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_packus_epi32(a, b);

    return validateUInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_round_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (double *) impl.mTestFloatPointer1;
    double d[2];
    __m128d ret;

    __m128d a = load_m128d(_a);
    switch (iter & 0x7) {
    case 0:
        d[0] = bankersRounding(_a[0]);
        d[1] = bankersRounding(_a[1]);

        ret = _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        break;
    case 1:
        d[0] = floor(_a[0]);
        d[1] = floor(_a[1]);

        ret = _mm_round_pd(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        break;
    case 2:
        d[0] = ceil(_a[0]);
        d[1] = ceil(_a[1]);

        ret = _mm_round_pd(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        break;
    case 3:
        d[0] = _a[0] > 0 ? floor(_a[0]) : ceil(_a[0]);
        d[1] = _a[1] > 0 ? floor(_a[1]) : ceil(_a[1]);

        ret = _mm_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        break;
    case 4:
        d[0] = bankersRounding(_a[0]);
        d[1] = bankersRounding(_a[1]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        ret = _mm_round_pd(a, _MM_FROUND_CUR_DIRECTION);
        break;
    case 5:
        d[0] = floor(_a[0]);
        d[1] = floor(_a[1]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        ret = _mm_round_pd(a, _MM_FROUND_CUR_DIRECTION);
        break;
    case 6:
        d[0] = ceil(_a[0]);
        d[1] = ceil(_a[1]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        ret = _mm_round_pd(a, _MM_FROUND_CUR_DIRECTION);
        break;
    case 7:
        d[0] = _a[0] > 0 ? floor(_a[0]) : ceil(_a[0]);
        d[1] = _a[1] > 0 ? floor(_a[1]) : ceil(_a[1]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        ret = _mm_round_pd(a, _MM_FROUND_CUR_DIRECTION);
        break;
    }

    return validateDouble(ret, d[0], d[1]);
}

result_t test_mm_round_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    float f[4];
    __m128 ret;

    __m128 a = load_m128(_a);
    switch (iter & 0x7) {
    case 0:
        f[0] = bankersRounding(_a[0]);
        f[1] = bankersRounding(_a[1]);
        f[2] = bankersRounding(_a[2]);
        f[3] = bankersRounding(_a[3]);

        ret = _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        break;
    case 1:
        f[0] = floorf(_a[0]);
        f[1] = floorf(_a[1]);
        f[2] = floorf(_a[2]);
        f[3] = floorf(_a[3]);

        ret = _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        break;
    case 2:
        f[0] = ceilf(_a[0]);
        f[1] = ceilf(_a[1]);
        f[2] = ceilf(_a[2]);
        f[3] = ceilf(_a[3]);

        ret = _mm_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        break;
    case 3:
        f[0] = _a[0] > 0 ? floorf(_a[0]) : ceilf(_a[0]);
        f[1] = _a[1] > 0 ? floorf(_a[1]) : ceilf(_a[1]);
        f[2] = _a[2] > 0 ? floorf(_a[2]) : ceilf(_a[2]);
        f[3] = _a[3] > 0 ? floorf(_a[3]) : ceilf(_a[3]);

        ret = _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        break;
    case 4:
        f[0] = bankersRounding(_a[0]);
        f[1] = bankersRounding(_a[1]);
        f[2] = bankersRounding(_a[2]);
        f[3] = bankersRounding(_a[3]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        ret = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
        break;
    case 5:
        f[0] = floorf(_a[0]);
        f[1] = floorf(_a[1]);
        f[2] = floorf(_a[2]);
        f[3] = floorf(_a[3]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        ret = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
        break;
    case 6:
        f[0] = ceilf(_a[0]);
        f[1] = ceilf(_a[1]);
        f[2] = ceilf(_a[2]);
        f[3] = ceilf(_a[3]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        ret = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
        break;
    case 7:
        f[0] = _a[0] > 0 ? floorf(_a[0]) : ceilf(_a[0]);
        f[1] = _a[1] > 0 ? floorf(_a[1]) : ceilf(_a[1]);
        f[2] = _a[2] > 0 ? floorf(_a[2]) : ceilf(_a[2]);
        f[3] = _a[3] > 0 ? floorf(_a[3]) : ceilf(_a[3]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        ret = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
        break;
    }

    return validateFloat(ret, f[0], f[1], f[2], f[3]);
}

result_t test_mm_round_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = (double *) impl.mTestFloatPointer1;
    const double *_b = (double *) impl.mTestFloatPointer2;
    double d[2];
    __m128d ret;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    d[1] = _a[1];
    switch (iter & 0x7) {
    case 0:
        d[0] = bankersRounding(_b[0]);

        ret = _mm_round_sd(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        break;
    case 1:
        d[0] = floor(_b[0]);

        ret = _mm_round_sd(a, b, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        break;
    case 2:
        d[0] = ceil(_b[0]);

        ret = _mm_round_sd(a, b, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        break;
    case 3:
        d[0] = _b[0] > 0 ? floor(_b[0]) : ceil(_b[0]);

        ret = _mm_round_sd(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        break;
    case 4:
        d[0] = bankersRounding(_b[0]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        ret = _mm_round_sd(a, b, _MM_FROUND_CUR_DIRECTION);
        break;
    case 5:
        d[0] = floor(_b[0]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        ret = _mm_round_sd(a, b, _MM_FROUND_CUR_DIRECTION);
        break;
    case 6:
        d[0] = ceil(_b[0]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        ret = _mm_round_sd(a, b, _MM_FROUND_CUR_DIRECTION);
        break;
    case 7:
        d[0] = _b[0] > 0 ? floor(_b[0]) : ceil(_b[0]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        ret = _mm_round_sd(a, b, _MM_FROUND_CUR_DIRECTION);
        break;
    }

    return validateDouble(ret, d[0], d[1]);
}

result_t test_mm_round_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float f[4];
    __m128 ret;

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    switch (iter & 0x7) {
    case 0:
        f[0] = bankersRounding(_b[0]);

        ret = _mm_round_ss(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        break;
    case 1:
        f[0] = floorf(_b[0]);

        ret = _mm_round_ss(a, b, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        break;
    case 2:
        f[0] = ceilf(_b[0]);

        ret = _mm_round_ss(a, b, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        break;
    case 3:
        f[0] = _b[0] > 0 ? floorf(_b[0]) : ceilf(_b[0]);

        ret = _mm_round_ss(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        break;
    case 4:
        f[0] = bankersRounding(_b[0]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        ret = _mm_round_ss(a, b, _MM_FROUND_CUR_DIRECTION);
        break;
    case 5:
        f[0] = floorf(_b[0]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        ret = _mm_round_ss(a, b, _MM_FROUND_CUR_DIRECTION);
        break;
    case 6:
        f[0] = ceilf(_b[0]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        ret = _mm_round_ss(a, b, _MM_FROUND_CUR_DIRECTION);
        break;
    case 7:
        f[0] = _b[0] > 0 ? floorf(_b[0]) : ceilf(_b[0]);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        ret = _mm_round_ss(a, b, _MM_FROUND_CUR_DIRECTION);
        break;
    }
    f[1] = _a[1];
    f[2] = _a[2];
    f[3] = _a[3];


    return validateFloat(ret, f[0], f[1], f[2], f[3]);
}

result_t test_mm_stream_load_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t *addr = impl.mTestIntPointer1;

    __m128i ret = _mm_stream_load_si128((__m128i *) addr);

    return validateInt32(ret, addr[0], addr[1], addr[2], addr[3]);
}

result_t test_mm_test_all_ones(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);

    int32_t d0 = ~_a[0] & (~(uint32_t) 0);
    int32_t d1 = ~_a[1] & (~(uint32_t) 0);
    int32_t d2 = ~_a[2] & (~(uint32_t) 0);
    int32_t d3 = ~_a[3] & (~(uint32_t) 0);
    int32_t result = ((d0 | d1 | d2 | d3) == 0) ? 1 : 0;

    int32_t ret = _mm_test_all_ones(a);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_test_all_zeros(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_mask = (const int32_t *) impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i mask = load_m128i(_mask);

    int32_t d0 = _a[0] & _mask[0];
    int32_t d1 = _a[1] & _mask[1];
    int32_t d2 = _a[2] & _mask[2];
    int32_t d3 = _a[3] & _mask[3];
    int32_t result = ((d0 | d1 | d2 | d3) == 0) ? 1 : 0;

    int32_t ret = _mm_test_all_zeros(a, mask);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_test_mix_ones_zeros(const SSE2NEONTestImpl &impl,
                                     uint32_t iter)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_mask = (const int32_t *) impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i mask = load_m128i(_mask);

    int32_t d0 = !((_a[0]) & _mask[0]) & !((!_a[0]) & _mask[0]);
    int32_t d1 = !((_a[1]) & _mask[1]) & !((!_a[1]) & _mask[1]);
    int32_t d2 = !((_a[2]) & _mask[2]) & !((!_a[2]) & _mask[2]);
    int32_t d3 = !((_a[3]) & _mask[3]) & !((!_a[3]) & _mask[3]);
    int32_t result = ((d0 & d1 & d2 & d3) == 0) ? 1 : 0;

    int32_t ret = _mm_test_mix_ones_zeros(a, mask);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_testc_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i b = _mm_load_si128((const __m128i *) _b);
    int testc = 1;
    for (int i = 0; i < 2; i++) {
        if ((~(((SIMDVec *) &a)->m128_u64[i]) &
             ((SIMDVec *) &b)->m128_u64[i])) {
            testc = 0;
            break;
        }
    }
    return _mm_testc_si128(a, b) == testc ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_testnzc_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_test_mix_ones_zeros(impl, iter);
}

result_t test_mm_testz_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i b = _mm_load_si128((const __m128i *) _b);
    int testz = 1;
    for (int i = 0; i < 2; i++) {
        if ((((SIMDVec *) &a)->m128_u64[i] & ((SIMDVec *) &b)->m128_u64[i])) {
            testz = 0;
            break;
        }
    }
    return _mm_testz_si128(a, b) == testz ? TEST_SUCCESS : TEST_FAIL;
}

/* SSE4.2 */
#define IS_CMPESTRI 1

#define DEF_ENUM_MM_CMPESTRX_VARIANT(c, ...) c,

#define EVAL_MM_CMPESTRX_TEST_CASE(c, type, data_type, im, IM)             \
    do {                                                                   \
        data_type *a = test_mm_##im##_##type##_data[c].a,                  \
                  *b = test_mm_##im##_##type##_data[c].b;                  \
        int la = test_mm_##im##_##type##_data[c].la,                       \
            lb = test_mm_##im##_##type##_data[c].lb;                       \
        const int imm8 = IMM_##c;                                          \
        IIF(IM)                                                            \
        (int expect = test_mm_##im##_##type##_data[c].expect,              \
         data_type *expect = test_mm_##im##_##type##_data[c].expect);      \
        __m128i ma, mb;                                                    \
        memcpy(&ma, a, sizeof(ma));                                        \
        memcpy(&mb, b, sizeof(mb));                                        \
        IIF(IM)                                                            \
        (int res = _mm_##im(ma, la, mb, lb, imm8),                         \
         __m128i res = _mm_##im(ma, la, mb, lb, imm8));                    \
        if (IIF(IM)(res != expect, memcmp(expect, &res, sizeof(__m128i)))) \
            return TEST_FAIL;                                              \
    } while (0);

#define ENUM_MM_CMPESTRX_TEST_CASES(type, type_lower, data_type, func, FUNC, \
                                    IM)                                      \
    enum { MM_##FUNC##_##type##_TEST_CASES(DEF_ENUM_MM_CMPESTRX_VARIANT) };  \
    MM_##FUNC##_##type##_TEST_CASES(EVAL_MM_CMPESTRX_TEST_CASE, type_lower,  \
                                    data_type, func, IM)

#define IMM_UBYTE_EACH_LEAST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT)
#define IMM_UBYTE_EACH_LEAST_NEGATIVE                                   \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UBYTE_EACH_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UBYTE_EACH_MOST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT)
#define IMM_UBYTE_EACH_MOST_MASKED_NEGATIVE                            \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UBYTE_ANY_LEAST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT)
#define IMM_UBYTE_ANY_LEAST_NEGATIVE                                   \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UBYTE_ANY_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UBYTE_ANY_MOST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT)
#define IMM_UBYTE_ANY_MOST_NEGATIVE                                   \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UBYTE_ANY_MOST_MASKED_NEGATIVE                            \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UBYTE_RANGES_LEAST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT)
#define IMM_UBYTE_RANGES_MOST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT)
#define IMM_UBYTE_RANGES_LEAST_NEGATIVE                             \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UBYTE_RANGES_LEAST_MASKED_NEGATIVE                      \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UBYTE_RANGES_MOST_MASKED_NEGATIVE                      \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UBYTE_ORDERED_LEAST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT)
#define IMM_UBYTE_ORDERED_LEAST_NEGATIVE                                   \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UBYTE_ORDERED_MOST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT)
#define IMM_UBYTE_ORDERED_MOST_MASKED_NEGATIVE                            \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)

#define IMM_SBYTE_EACH_LEAST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SBYTE_EACH_LEAST_NEGATIVE                                   \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SBYTE_EACH_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SBYTE_EACH_MOST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT)
#define IMM_SBYTE_EACH_MOST_NEGATIVE                                   \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SBYTE_ANY_LEAST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SBYTE_ANY_MOST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT)
#define IMM_SBYTE_ANY_MOST_MASKED_NEGATIVE                             \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SBYTE_RANGES_LEAST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SBYTE_RANGES_LEAST_NEGATIVE                             \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SBYTE_RANGES_LEAST_MASKED_NEGATIVE                      \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SBYTE_RANGES_MOST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT)
#define IMM_SBYTE_RANGES_MOST_MASKED_NEGATIVE                      \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SBYTE_ORDERED_LEAST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SBYTE_ORDERED_LEAST_NEGATIVE                                   \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SBYTE_ORDERED_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SBYTE_ORDERED_MOST_NEGATIVE                                   \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SBYTE_ORDERED_MOST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT)
#define IMM_SBYTE_ORDERED_MOST_MASKED_NEGATIVE                            \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)

#define IMM_UWORD_RANGES_LEAST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT)
#define IMM_UWORD_RANGES_LEAST_NEGATIVE                             \
    (_SIDD_UWORD_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UWORD_RANGES_LEAST_MASKED_NEGATIVE                      \
    (_SIDD_UWORD_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UWORD_RANGES_MOST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT)
#define IMM_UWORD_RANGES_MOST_NEGATIVE                             \
    (_SIDD_UWORD_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UWORD_RANGES_MOST_MASKED_NEGATIVE                      \
    (_SIDD_UWORD_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UWORD_EACH_LEAST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT)
#define IMM_UWORD_EACH_MOST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT)
#define IMM_UWORD_EACH_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UWORD_EACH_MOST_MASKED_NEGATIVE                            \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UWORD_ANY_LEAST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT)
#define IMM_UWORD_ANY_MOST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT)
#define IMM_UWORD_ANY_MOST_NEGATIVE                                   \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UWORD_ANY_LEAST_NEGATIVE                                   \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UWORD_ANY_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UWORD_ORDERED_LEAST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT)
#define IMM_UWORD_ORDERED_LEAST_NEGATIVE                                   \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UWORD_ORDERED_MOST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT)
#define IMM_UWORD_ORDERED_MOST_NEGATIVE                                   \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)

#define IMM_SWORD_RANGES_LEAST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SWORD_RANGES_MOST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT)
#define IMM_SWORD_RANGES_LEAST_NEGATIVE                             \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SWORD_RANGES_MOST_MASKED_NEGATIVE                      \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SWORD_EACH_LEAST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SWORD_EACH_MOST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT)
#define IMM_SWORD_EACH_LEAST_NEGATIVE                                   \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SWORD_EACH_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SWORD_EACH_MOST_NEGATIVE                                   \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SWORD_EACH_MOST_MASKED_NEGATIVE                            \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SWORD_ANY_LEAST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SWORD_ANY_LEAST_NEGATIVE \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SWORD_ANY_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SWORD_ANY_MOST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT)
#define IMM_SWORD_ANY_MOST_MASKED_NEGATIVE                            \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SWORD_ORDERED_LEAST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SWORD_ORDERED_LEAST_NEGATIVE                                   \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SWORD_ORDERED_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SWORD_ORDERED_MOST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT)
#define IMM_SWORD_ORDERED_MOST_MASKED_NEGATIVE                            \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)

typedef struct {
    uint8_t a[16], b[16];
    int la, lb;
    const int imm8;
    int expect;
} test_mm_cmpestri_ubyte_data_t;
typedef struct {
    int8_t a[16], b[16];
    int la, lb;
    const int imm8;
    int expect;
} test_mm_cmpestri_sbyte_data_t;
typedef struct {
    uint16_t a[8], b[8];
    int la, lb;
    const int imm8;
    int expect;
} test_mm_cmpestri_uword_data_t;
typedef struct {
    int16_t a[8], b[8];
    int la, lb;
    const int imm8;
    int expect;
} test_mm_cmpestri_sword_data_t;

#define TEST_MM_CMPESTRA_UBYTE_DATA_LEN 3
static test_mm_cmpestri_ubyte_data_t
    test_mm_cmpestra_ubyte_data[TEST_MM_CMPESTRA_UBYTE_DATA_LEN] = {
        {{20, 10, 33, 56, 78},
         {20, 10, 34, 98, 127, 20, 10, 32, 20, 10, 32, 11, 3, 20, 10, 31},
         3,
         17,
         IMM_UBYTE_ORDERED_MOST,
         1},
        {{20, 127, 0, 45, 77, 1, 34, 43, 109},
         {2, 127, 0, 54, 6, 43, 12, 110, 100},
         9,
         20,
         IMM_UBYTE_EACH_LEAST_NEGATIVE,
         0},
        {{22, 33, 90, 1},
         {22, 33, 90, 1, 1, 5, 4, 7, 98, 34, 1, 12, 13, 14, 15, 16},
         4,
         11,
         IMM_UBYTE_ANY_LEAST_MASKED_NEGATIVE,
         0},
};

#define TEST_MM_CMPESTRA_SBYTE_DATA_LEN 3
static test_mm_cmpestri_sbyte_data_t
    test_mm_cmpestra_sbyte_data[TEST_MM_CMPESTRA_SBYTE_DATA_LEN] = {
        {{45, -94, 38, -11, 84, -123, -43, -49, 25, -55, -121, -6, 57, 108, -55,
          69},
         {-26, -61, -21, -96, 48, -112, 95, -56, 29, -55, -121, -6, 57, 108,
          -55, 69},
         23,
         28,
         IMM_SBYTE_RANGES_LEAST,
         0},
        {{-12, 8},
         {-12, 7, -12, 8, -13, 45, -12, 8},
         2,
         8,
         IMM_SBYTE_ORDERED_MOST_NEGATIVE,
         0},
        {{-100, -127, 56, 78, 21, -1, 9, 127, 45},
         {100, 126, 30, 65, 87, 54, 80, 81, -98, -101, 90, 1, 5, 60, -77, -65},
         10,
         20,
         IMM_SBYTE_ANY_LEAST,
         1},
};

#define TEST_MM_CMPESTRA_UWORD_DATA_LEN 3
static test_mm_cmpestri_uword_data_t
    test_mm_cmpestra_uword_data[TEST_MM_CMPESTRA_UWORD_DATA_LEN] = {
        {{10000, 20000, 30000, 40000, 50000},
         {40001, 50002, 10000, 20000, 30000, 40000, 50000},
         5,
         10,
         IMM_UWORD_ORDERED_LEAST,
         0},
        {{1001, 9487, 9487, 8000},
         {1001, 1002, 1003, 8709, 100, 1, 1000, 999},
         4,
         6,
         IMM_UWORD_RANGES_LEAST_MASKED_NEGATIVE,
         0},
        {{12, 21, 0, 45, 88, 10001, 10002, 65535},
         {22, 13, 3, 54, 888, 10003, 10000, 65530},
         13,
         13,
         IMM_UWORD_EACH_MOST,
         1},
};

#define TEST_MM_CMPESTRA_SWORD_DATA_LEN 3
static test_mm_cmpestri_sword_data_t
    test_mm_cmpestra_sword_data[TEST_MM_CMPESTRA_SWORD_DATA_LEN] = {
        {{-100, -80, -5, -1, 10, 1000},
         {-100, -99, -80, -2, 11, 789, 889, 999},
         6,
         12,
         IMM_SWORD_RANGES_LEAST_NEGATIVE,
         1},
        {{-30000, -90, -32766, 1200, 5},
         {-30001, 21, 10000, 1201, 888},
         5,
         5,
         IMM_SWORD_EACH_MOST,
         0},
        {{2001, -1928},
         {2000, 1928, 3000, 2289, 4000, 111, 2002, -1928},
         2,
         9,
         IMM_SWORD_ANY_LEAST_MASKED_NEGATIVE,
         0},
};


#define MM_CMPESTRA_UBYTE_TEST_CASES(_, ...)  \
    _(UBYTE_ORDERED_MOST, __VA_ARGS__)        \
    _(UBYTE_EACH_LEAST_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_ANY_LEAST_MASKED_NEGATIVE, __VA_ARGS__)

#define MM_CMPESTRA_SBYTE_TEST_CASES(_, ...)    \
    _(SBYTE_RANGES_LEAST, __VA_ARGS__)          \
    _(SBYTE_ORDERED_MOST_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ANY_LEAST, __VA_ARGS__)

#define MM_CMPESTRA_UWORD_TEST_CASES(_, ...)           \
    _(UWORD_ORDERED_LEAST, __VA_ARGS__)                \
    _(UWORD_RANGES_LEAST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(UWORD_EACH_MOST, __VA_ARGS__)

#define MM_CMPESTRA_SWORD_TEST_CASES(_, ...)    \
    _(SWORD_RANGES_LEAST_NEGATIVE, __VA_ARGS__) \
    _(SWORD_EACH_MOST, __VA_ARGS__)             \
    _(SWORD_ANY_LEAST_MASKED_NEGATIVE, __VA_ARGS__)

#define GENERATE_MM_CMPESTRA_TEST_CASES                                     \
    ENUM_MM_CMPESTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpestra, CMPESTRA,  \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpestra, CMPESTRA,   \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(UWORD, uword, uint16_t, cmpestra, CMPESTRA, \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SWORD, sword, int16_t, cmpestra, CMPESTRA,  \
                                IS_CMPESTRI)

result_t test_mm_cmpestra(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPESTRA_TEST_CASES
    return TEST_SUCCESS;
}

#define TEST_MM_CMPESTRC_UBYTE_DATA_LEN 4
static test_mm_cmpestri_ubyte_data_t
    test_mm_cmpestrc_ubyte_data[TEST_MM_CMPESTRC_UBYTE_DATA_LEN] = {
        {{66, 3, 3, 65},
         {66, 3, 3, 65, 67, 2, 2, 67, 56, 11, 1, 23, 66, 3, 3, 65},
         4,
         16,
         IMM_UBYTE_ORDERED_MOST_MASKED_NEGATIVE,
         1},
        {{1, 11, 2, 22, 3, 33, 4, 44, 5, 55, 6, 66, 7, 77, 8, 88},
         {2, 22, 3, 23, 5, 66, 255, 43, 6, 66, 7, 77, 9, 99, 10, 100},
         16,
         16,
         IMM_UBYTE_EACH_MOST,
         0},
        {{36, 72, 108}, {12, 24, 48, 96, 77, 84}, 3, 6, IMM_UBYTE_ANY_LEAST, 0},
        {{12, 24, 36, 48},
         {11, 49, 50, 56, 77, 15, 10},
         4,
         7,
         IMM_UBYTE_RANGES_LEAST_NEGATIVE,
         1},
};

#define TEST_MM_CMPESTRC_SBYTE_DATA_LEN 4
static test_mm_cmpestri_sbyte_data_t
    test_mm_cmpestrc_sbyte_data[TEST_MM_CMPESTRC_SBYTE_DATA_LEN] = {
        {{-22, -30, 40, 45},
         {-31, -32, 46, 77},
         4,
         4,
         IMM_SBYTE_RANGES_MOST,
         0},
        {{-12, -7, 33, 100, 12},
         {-12, -7, 33, 100, 11, -11, -7, 33, 100, 12},
         5,
         10,
         IMM_SBYTE_ORDERED_MOST_MASKED_NEGATIVE,
         1},
        {{1, 2, 3, 4, 5, -1, -2, -3, -4, -5},
         {1, 2, 3, 4, 5, -1, -2, -3, -5},
         10,
         9,
         IMM_SBYTE_ANY_MOST_MASKED_NEGATIVE,
         0},
        {{101, -128, -88, -76, 89, 109, 44, -12, -45, -100, 22, 1, 91},
         {102, -120, 88, -76, 98, 107, 33, 12, 45, -100, 22, 10, 19},
         13,
         13,
         IMM_SBYTE_EACH_MOST,
         1},
};

#define TEST_MM_CMPESTRC_UWORD_DATA_LEN 4
static test_mm_cmpestri_uword_data_t
    test_mm_cmpestrc_uword_data[TEST_MM_CMPESTRC_UWORD_DATA_LEN] = {
        {{1000, 2000, 4000, 8000, 16000},
         {40001, 1000, 2000, 40000, 8000, 16000},
         5,
         6,
         IMM_UWORD_ORDERED_LEAST_NEGATIVE,
         1},
        {{1111, 1212},
         {1110, 1213, 1110, 1214, 1100, 1220, 1000, 1233},
         2,
         8,
         IMM_UWORD_RANGES_MOST,
         0},
        {{10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000},
         {9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000},
         13,
         13,
         IMM_UWORD_EACH_LEAST_MASKED_NEGATIVE,
         1},
        {{12}, {11, 13, 14, 15, 10}, 1, 5, IMM_UWORD_ANY_MOST, 0},
};

#define TEST_MM_CMPESTRC_SWORD_DATA_LEN 4
static test_mm_cmpestri_sword_data_t
    test_mm_cmpestrc_sword_data[TEST_MM_CMPESTRC_SWORD_DATA_LEN] = {
        {{-100, -90, -80, -66, 1},
         {-101, -102, -1000, 2, 67, 10000},
         5,
         6,
         IMM_SWORD_RANGES_LEAST,
         0},
        {{12, 13, -700, 888, 44, -987, 19},
         {12, 13, -700, 888, 44, -987, 19},
         7,
         7,
         IMM_SWORD_EACH_MOST_NEGATIVE,
         0},
        {{2001, -1992, 1995, 10007, 2000},
         {2000, 1928, 3000, 9822, 5000, 1111, 2002, -1928},
         5,
         9,
         IMM_SWORD_ANY_LEAST_NEGATIVE,
         1},
        {{13, -26, 39},
         {12, -25, 33, 13, -26, 39},
         3,
         6,
         IMM_SWORD_ORDERED_MOST,
         1},
};


#define MM_CMPESTRC_UBYTE_TEST_CASES(_, ...)           \
    _(UBYTE_ORDERED_MOST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_EACH_MOST, __VA_ARGS__)                    \
    _(UBYTE_ANY_LEAST, __VA_ARGS__)                    \
    _(UBYTE_RANGES_LEAST_NEGATIVE, __VA_ARGS__)

#define MM_CMPESTRC_SBYTE_TEST_CASES(_, ...)           \
    _(SBYTE_RANGES_MOST, __VA_ARGS__)                  \
    _(SBYTE_ORDERED_MOST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ANY_MOST_MASKED_NEGATIVE, __VA_ARGS__)     \
    _(SBYTE_EACH_MOST, __VA_ARGS__)

#define MM_CMPESTRC_UWORD_TEST_CASES(_, ...)         \
    _(UWORD_ORDERED_LEAST_NEGATIVE, __VA_ARGS__)     \
    _(UWORD_RANGES_MOST, __VA_ARGS__)                \
    _(UWORD_EACH_LEAST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(UWORD_ANY_MOST, __VA_ARGS__)

#define MM_CMPESTRC_SWORD_TEST_CASES(_, ...) \
    _(SWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(SWORD_EACH_MOST_NEGATIVE, __VA_ARGS__) \
    _(SWORD_ANY_LEAST_NEGATIVE, __VA_ARGS__) \
    _(SWORD_ORDERED_MOST, __VA_ARGS__)

#define GENERATE_MM_CMPESTRC_TEST_CASES                                     \
    ENUM_MM_CMPESTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpestrc, CMPESTRC,  \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpestrc, CMPESTRC,   \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(UWORD, uword, uint16_t, cmpestrc, CMPESTRC, \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SWORD, sword, int16_t, cmpestrc, CMPESTRC,  \
                                IS_CMPESTRI)

result_t test_mm_cmpestrc(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPESTRC_TEST_CASES
    return TEST_SUCCESS;
}

#define TEST_MM_CMPESTRI_UBYTE_DATA_LEN 4
static test_mm_cmpestri_ubyte_data_t
    test_mm_cmpestri_ubyte_data[TEST_MM_CMPESTRI_UBYTE_DATA_LEN] = {
        {{23, 89, 255, 0, 90, 45, 67, 12, 1, 56, 200, 141, 3, 4, 2, 76},
         {32, 89, 255, 128, 9, 54, 78, 12, 1, 56, 100, 41, 42, 68, 32, 5},
         16,
         16,
         IMM_UBYTE_ANY_LEAST_NEGATIVE,
         0},
        {{0, 83, 112, 12, 221, 54, 76, 83, 112, 10},
         {0, 83, 112, 83, 122, 45, 67, 83, 112, 9},
         10,
         10,
         IMM_UBYTE_EACH_LEAST,
         0},
        {{34, 78, 12},
         {56, 100, 11, 67, 35, 79, 67, 255, 0, 43, 121, 234, 225, 91, 31, 23},
         3,
         16,
         IMM_UBYTE_RANGES_LEAST,
         0},
        {{13, 10, 9, 32, 105, 103, 110, 111, 114, 101, 32, 116, 104, 105, 115,
          32},
         {83, 112, 108, 105, 116, 32, 13, 10, 9, 32, 108, 105, 110, 101, 32,
          32},
         3,
         15,
         IMM_UBYTE_ORDERED_LEAST,
         6},
};

#define TEST_MM_CMPESTRI_SBYTE_DATA_LEN 4
static test_mm_cmpestri_sbyte_data_t
    test_mm_cmpestri_sbyte_data[TEST_MM_CMPESTRI_SBYTE_DATA_LEN] = {
        {{-12, -1, 90, -128, 43, 6, 87, 127},
         {-1, -1, 9, -127, 126, 6, 78, 23},
         8,
         8,
         IMM_SBYTE_EACH_LEAST,
         1},
        {{34, 67, -90, 33, 123, -100, 43, 56},
         {43, 76, -90, 44, 20, -100, 54, 56},
         8,
         8,
         IMM_SBYTE_ANY_LEAST,
         0},
        {{-43, 67, 89},
         {-44, -54, -30, -128, 127, 34, 10, -62},
         3,
         7,
         IMM_SBYTE_RANGES_LEAST,
         2},
        {{90, 34, -32, 0, 5},
         {19, 34, -32, 90, 34, -32, 45, 0, 5, 90, 34, -32, 0, 5, 19, 87},
         3,
         16,
         IMM_SBYTE_ORDERED_LEAST,
         3},
};

#define TEST_MM_CMPESTRI_UWORD_DATA_LEN 4
static test_mm_cmpestri_uword_data_t
    test_mm_cmpestri_uword_data[TEST_MM_CMPESTRI_UWORD_DATA_LEN] = {
        {{45, 65535, 0, 87, 1000, 10, 45, 26},
         {65534, 0, 0, 78, 1000, 10, 32, 26},
         8,
         8,
         IMM_UWORD_EACH_LEAST,
         2},
        {{45, 23, 10, 54, 88, 10000, 20000, 100},
         {544, 10000, 20000, 1, 0, 2897, 2330, 2892},
         8,
         8,
         IMM_UWORD_ANY_LEAST,
         1},
        {{10000, 15000},
         {12, 45, 67, 899, 10001, 32, 15001, 15000},
         2,
         8,
         IMM_UWORD_RANGES_LEAST,
         4},
        {{0, 1, 54, 89, 100},
         {101, 102, 65535, 0, 1, 54, 89, 100},
         5,
         8,
         IMM_UWORD_ORDERED_LEAST,
         3},
};

#define TEST_MM_CMPESTRI_SWORD_DATA_LEN 4
static test_mm_cmpestri_sword_data_t
    test_mm_cmpestri_sword_data[TEST_MM_CMPESTRI_SWORD_DATA_LEN] = {
        {{13, 6, 5, 4, 3, 2, 1, 3},
         {-7, 16, 5, 4, -1, 6, 1, 3},
         10,
         10,
         IMM_SWORD_RANGES_MOST,
         7},
        {{13, 6, 5, 4, 3, 2, 1, 3},
         {-7, 16, 5, 4, -1, 6, 1, 3},
         8,
         8,
         IMM_SWORD_EACH_LEAST,
         2},
        {{-32768, 90, 455, 67, -1000, -10000, 21, 12},
         {-7, 61, 455, 67, -32768, 32767, 11, 888},
         8,
         8,
         IMM_SWORD_ANY_LEAST,
         2},
        {{-12, -56},
         {-7, 16, 555, 554, -12, 61, -16, 3},
         2,
         8,
         IMM_SWORD_ORDERED_LEAST,
         8},
};

#define MM_CMPESTRI_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_ANY_LEAST_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(UBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(UBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPESTRI_SBYTE_TEST_CASES(_, ...) \
    _(SBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(SBYTE_ANY_LEAST, __VA_ARGS__)          \
    _(SBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(SBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPESTRI_UWORD_TEST_CASES(_, ...) \
    _(UWORD_EACH_LEAST, __VA_ARGS__)         \
    _(UWORD_ANY_LEAST, __VA_ARGS__)          \
    _(UWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(UWORD_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPESTRI_SWORD_TEST_CASES(_, ...) \
    _(SWORD_RANGES_MOST, __VA_ARGS__)        \
    _(SWORD_EACH_LEAST, __VA_ARGS__)         \
    _(SWORD_ANY_LEAST, __VA_ARGS__)          \
    _(SWORD_ORDERED_LEAST, __VA_ARGS__)

#define GENERATE_MM_CMPESTRI_TEST_CASES                                     \
    ENUM_MM_CMPESTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpestri, CMPESTRI,  \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpestri, CMPESTRI,   \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(UWORD, uword, uint16_t, cmpestri, CMPESTRI, \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SWORD, sword, int16_t, cmpestri, CMPESTRI,  \
                                IS_CMPESTRI)

result_t test_mm_cmpestri(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPESTRI_TEST_CASES
    return TEST_SUCCESS;
}

#define IS_CMPESTRM 0

typedef struct {
    uint8_t a[16], b[16];
    int la, lb;
    const int imm8;
    uint8_t expect[16];
} test_mm_cmpestrm_ubyte_data_t;
typedef struct {
    int8_t a[16], b[16];
    int la, lb;
    const int imm8;
    int8_t expect[16];
} test_mm_cmpestrm_sbyte_data_t;
typedef struct {
    uint16_t a[8], b[8];
    int la, lb;
    const int imm8;
    uint16_t expect[8];
} test_mm_cmpestrm_uword_data_t;
typedef struct {
    int16_t a[8], b[8];
    int la, lb;
    const int imm8;
    int16_t expect[8];
} test_mm_cmpestrm_sword_data_t;

#define IMM_UBYTE_EACH_UNIT \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK)
#define IMM_UBYTE_EACH_UNIT_NEGATIVE                            \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UBYTE_ANY_UNIT \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_UNIT_MASK)
#define IMM_UBYTE_ANY_BIT \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK)
#define IMM_UBYTE_RANGES_UNIT \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK)
#define IMM_UBYTE_ORDERED_UNIT \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_UNIT_MASK)

#define IMM_SBYTE_EACH_UNIT \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK)
#define IMM_SBYTE_EACH_BIT_MASKED_NEGATIVE                     \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_BIT_MASK | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SBYTE_ANY_UNIT \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_UNIT_MASK)
#define IMM_SBYTE_ANY_UNIT_MASKED_NEGATIVE                     \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_UNIT_MASK | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SBYTE_RANGES_UNIT \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK)
#define IMM_SBYTE_ORDERED_UNIT \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_UNIT_MASK)

#define IMM_UWORD_RANGES_UNIT \
    (_SIDD_UWORD_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK)
#define IMM_UWORD_EACH_UNIT \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK)
#define IMM_UWORD_ANY_UNIT \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_UNIT_MASK)
#define IMM_UWORD_ANY_BIT \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK)
#define IMM_UWORD_ORDERED_UNIT \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_UNIT_MASK)
#define IMM_UWORD_ORDERED_UNIT_NEGATIVE                            \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_UNIT_MASK | \
     _SIDD_NEGATIVE_POLARITY)

#define IMM_SWORD_RANGES_UNIT \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK)
#define IMM_SWORD_RANGES_BIT \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_BIT_MASK)
#define IMM_SWORD_EACH_UNIT \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK)
#define IMM_SWORD_ANY_UNIT \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_UNIT_MASK)
#define IMM_SWORD_ORDERED_UNIT \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_UNIT_MASK)

#define TEST_MM_CMPESTRM_UBYTE_DATA_LEN 4
static test_mm_cmpestrm_ubyte_data_t
    test_mm_cmpestrm_ubyte_data[TEST_MM_CMPESTRM_UBYTE_DATA_LEN] = {
        {{85, 115, 101, 70, 108, 97, 116, 65, 115, 115, 101, 109, 98, 108, 101,
          114},
         {85, 115, 105, 110, 103, 65, 110, 65, 115, 115, 101, 109, 98, 108, 101,
          114},
         16,
         16,
         IMM_UBYTE_EACH_UNIT_NEGATIVE,
         {0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{97, 101, 105, 111, 117, 121},
         {89, 111, 117, 32, 68, 114, 105, 118, 101, 32, 77, 101, 32, 77, 97,
          100},
         6,
         16,
         IMM_UBYTE_ANY_UNIT,
         {0, 255, 255, 0, 0, 0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0}},
        {{97, 122, 65, 90},
         {73, 39, 109, 32, 104, 101, 114, 101, 32, 98, 101, 99, 97, 117, 115,
          101},
         4,
         16,
         IMM_UBYTE_RANGES_UNIT,
         {255, 0, 255, 0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255,
          255}},
        {{87, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {87, 104, 101, 110, 87, 101, 87, 105, 108, 108, 66, 101, 87, 101, 100,
          33},
         2,
         16,
         IMM_UBYTE_ORDERED_UNIT,
         {0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0}},
};

#define TEST_MM_CMPESTRM_SBYTE_DATA_LEN 4
static test_mm_cmpestrm_sbyte_data_t
    test_mm_cmpestrm_sbyte_data[TEST_MM_CMPESTRM_SBYTE_DATA_LEN] = {
        {{-127, -127, 34, 88, 0, 1, -1, 78, 90, 9, 23, 34, 3, -128, 127, 0},
         {0, -127, 34, 88, 12, 43, -128, 78, 8, 9, 43, 32, 7, 126, 115, 0},
         16,
         16,
         IMM_SBYTE_EACH_UNIT,
         {0, -1, -1, -1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1}},
        {{0, 32, 7, 115, -128, 44, 33},
         {0, -127, 34, 88, 12, 43, -128, 78, 8, 9, 43, 32, 7, 126, 115, 0},
         7,
         10,
         IMM_SBYTE_ANY_UNIT_MASKED_NEGATIVE,
         {0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0}},
        {{-128, -80, -90, 10, 33},
         {-126, -93, -80, -77, -56, -23, -10, -1, 0, 3, 10, 12, 13, 33, 34, 56},
         5,
         16,
         IMM_SBYTE_RANGES_UNIT,
         {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0}},
        {{104, 9, -12},
         {0, 0, 87, 104, 9, -12, 89, -117, 9, 10, -11, 87, -114, 104, 9, -61},
         3,
         16,
         IMM_SBYTE_ORDERED_UNIT,
         {0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
};

#define TEST_MM_CMPESTRM_UWORD_DATA_LEN 4
static test_mm_cmpestrm_uword_data_t
    test_mm_cmpestrm_uword_data[TEST_MM_CMPESTRM_UWORD_DATA_LEN] = {
        {{1, 5, 13, 19, 22},
         {12, 60000, 5, 1, 100, 1000, 34, 20},
         5,
         8,
         IMM_UWORD_RANGES_UNIT,
         {0, 0, 65535, 65535, 0, 0, 0, 0}},
        {{65535, 12, 7, 9876, 3456, 12345, 10, 98},
         {65535, 0, 10, 9876, 3456, 0, 13, 32},
         8,
         8,
         IMM_UWORD_EACH_UNIT,
         {65535, 0, 0, 65535, 65535, 0, 0, 0}},
        {{100, 0},
         {12345, 6766, 234, 0, 1, 34, 89, 100},
         2,
         8,
         IMM_UWORD_ANY_BIT,
         {136, 0, 0, 0, 0, 0, 0, 0}},
        {{123, 67, 890},
         {123, 67, 890, 8900, 4, 0, 123, 67},
         3,
         8,
         IMM_UWORD_ORDERED_UNIT,
         {65535, 0, 0, 0, 0, 0, 65535, 0}},
};

#define TEST_MM_CMPESTRM_SWORD_DATA_LEN 4
static test_mm_cmpestrm_sword_data_t
    test_mm_cmpestrm_sword_data[TEST_MM_CMPESTRM_SWORD_DATA_LEN] = {
        {{13, 6, 5, 4, 3, 2, 1, 3},
         {-7, 16, 5, 4, -1, 6, 1, 3},
         10,
         10,
         IMM_SWORD_RANGES_UNIT,
         {0, 0, 0, 0, 0, 0, -1, -1}},
        {{85, 115, 101, 70, 108, 97, 116, 65},
         {85, 115, 105, 110, 103, 65, 110, 65},
         8,
         8,
         IMM_SWORD_EACH_UNIT,
         {-1, -1, 0, 0, 0, 0, 0, -1}},
        {{-32768, 10000, 10, -13},
         {-32767, 32767, -32768, 90, 0, -13, 23, 45},
         4,
         8,
         IMM_SWORD_ANY_UNIT,
         {0, 0, -1, 0, 0, -1, 0, 0}},
        {{10, 20, -10, 60},
         {0, 0, 0, 10, 20, -10, 60, 10},
         4,
         8,
         IMM_SWORD_ORDERED_UNIT,
         {0, 0, 0, -1, 0, 0, 0, -1}},
};

#define MM_CMPESTRM_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_EACH_UNIT_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_ANY_UNIT, __VA_ARGS__)           \
    _(UBYTE_RANGES_UNIT, __VA_ARGS__)        \
    _(UBYTE_ORDERED_UNIT, __VA_ARGS__)

#define MM_CMPESTRM_SBYTE_TEST_CASES(_, ...)       \
    _(SBYTE_EACH_UNIT, __VA_ARGS__)                \
    _(SBYTE_ANY_UNIT_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_RANGES_UNIT, __VA_ARGS__)              \
    _(SBYTE_ORDERED_UNIT, __VA_ARGS__)

#define MM_CMPESTRM_UWORD_TEST_CASES(_, ...) \
    _(UWORD_RANGES_UNIT, __VA_ARGS__)        \
    _(UWORD_EACH_UNIT, __VA_ARGS__)          \
    _(UWORD_ANY_BIT, __VA_ARGS__)            \
    _(UWORD_ORDERED_UNIT, __VA_ARGS__)

#define MM_CMPESTRM_SWORD_TEST_CASES(_, ...) \
    _(SWORD_RANGES_UNIT, __VA_ARGS__)        \
    _(SWORD_EACH_UNIT, __VA_ARGS__)          \
    _(SWORD_ANY_UNIT, __VA_ARGS__)           \
    _(SWORD_ORDERED_UNIT, __VA_ARGS__)

#define GENERATE_MM_CMPESTRM_TEST_CASES                                     \
    ENUM_MM_CMPESTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpestrm, CMPESTRM,  \
                                IS_CMPESTRM)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpestrm, CMPESTRM,   \
                                IS_CMPESTRM)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(UWORD, uword, uint16_t, cmpestrm, CMPESTRM, \
                                IS_CMPESTRM)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SWORD, sword, int16_t, cmpestrm, CMPESTRM,  \
                                IS_CMPESTRM)

result_t test_mm_cmpestrm(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPESTRM_TEST_CASES
    return TEST_SUCCESS;
}

#undef IS_CMPESTRM

#define TEST_MM_CMPESTRO_UBYTE_DATA_LEN 4
static test_mm_cmpestri_ubyte_data_t
    test_mm_cmpestro_ubyte_data[TEST_MM_CMPESTRO_UBYTE_DATA_LEN] = {
        {{56, 78, 255, 1, 9},
         {56, 78, 43, 255, 1, 6, 9},
         5,
         7,
         IMM_UBYTE_ANY_MOST_NEGATIVE,
         0},
        {{33, 44, 100, 24, 3, 89, 127, 254, 33, 45, 250},
         {33, 44, 100, 22, 3, 98, 125, 254, 33, 4, 243},
         11,
         11,
         IMM_UBYTE_EACH_LEAST_MASKED_NEGATIVE,
         0},
        {{34, 27, 18, 9}, {}, 4, 16, IMM_UBYTE_RANGES_LEAST_MASKED_NEGATIVE, 1},
        {{3, 18, 216},
         {3, 18, 222, 3, 17, 216, 3, 18, 216},
         3,
         9,
         IMM_UBYTE_ORDERED_LEAST_NEGATIVE,
         1},
};

#define TEST_MM_CMPESTRO_SBYTE_DATA_LEN 4
static test_mm_cmpestri_sbyte_data_t
    test_mm_cmpestro_sbyte_data[TEST_MM_CMPESTRO_SBYTE_DATA_LEN] = {
        {{23, -23, 24, -24, 25, -25, 26, -26, 27, -27, 28, -28, -29, 29, 30,
          31},
         {24, -23, 25, -24, 25, -25, 26, -26, 27, -27, 28, -28, -29, 29, 30,
          31},
         16,
         16,
         IMM_SBYTE_EACH_MOST_NEGATIVE,
         1},
        {{34, 33, 67, 72, -90, 127, 33, -128, 123, -90, -100, 34, 43, 15, 56,
          3},
         {3, 14, 15, 65, 90, -127, 100, 100},
         16,
         8,
         IMM_SBYTE_ANY_MOST,
         1},
        {{-13, 0, 34},
         {-12, -11, 1, 12, 56, 57, 3, 2, -17},
         6,
         9,
         IMM_SBYTE_RANGES_MOST_MASKED_NEGATIVE,
         0},
        {{1, 2, 3, 4, 5, 6, 7, 8},
         {-1, -2, -3, -4, -5, -6, -7, -8, 1, 2, 3, 4, 5, 6, 7, 8},
         8,
         16,
         IMM_SBYTE_ORDERED_MOST,
         0},
};

#define TEST_MM_CMPESTRO_UWORD_DATA_LEN 4
static test_mm_cmpestri_uword_data_t
    test_mm_cmpestro_uword_data[TEST_MM_CMPESTRO_UWORD_DATA_LEN] = {
        {{0, 0, 0, 4, 4, 4, 8, 8},
         {0, 0, 0, 3, 3, 16653, 3333, 222},
         8,
         8,
         IMM_UWORD_EACH_MOST_MASKED_NEGATIVE,
         0},
        {{12, 666, 9456, 10000, 32, 444, 57, 0},
         {11, 777, 9999, 32767, 23},
         8,
         5,
         IMM_UWORD_ANY_LEAST_MASKED_NEGATIVE,
         1},
        {{23, 32, 45, 67},
         {10022, 23, 32, 44, 66, 67, 12, 22},
         4,
         8,
         IMM_UWORD_RANGES_LEAST_NEGATIVE,
         1},
        {{222, 45, 8989},
         {221, 222, 45, 8989, 222, 45, 8989},
         3,
         7,
         IMM_UWORD_ORDERED_MOST,
         0},
};

#define TEST_MM_CMPESTRO_SWORD_DATA_LEN 4
static test_mm_cmpestri_sword_data_t
    test_mm_cmpestro_sword_data[TEST_MM_CMPESTRO_SWORD_DATA_LEN] = {
        {{-9999, -9487, -5000, -4433, -3000, -2999, -2000, -1087},
         {-32767, -30000, -4998},
         100,
         3,
         IMM_SWORD_RANGES_MOST_MASKED_NEGATIVE,
         1},
        {{-30, 89, 7777},
         {-30, 89, 7777},
         3,
         3,
         IMM_SWORD_EACH_MOST_MASKED_NEGATIVE,
         0},
        {{8, 9, -100, 1000, -5000, -32000, 32000, 7},
         {29999, 32001, 5, 555},
         8,
         4,
         IMM_SWORD_ANY_MOST_MASKED_NEGATIVE,
         1},
        {{-1, 56, -888, 9000, -23, 12, -1, -1},
         {-1, 56, -888, 9000, -23, 12, -1, -1},
         8,
         8,
         IMM_SWORD_ORDERED_MOST_MASKED_NEGATIVE,
         0},
};

#define MM_CMPESTRO_UBYTE_TEST_CASES(_, ...)           \
    _(UBYTE_ANY_MOST_NEGATIVE, __VA_ARGS__)            \
    _(UBYTE_EACH_LEAST_MASKED_NEGATIVE, __VA_ARGS__)   \
    _(UBYTE_RANGES_LEAST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_ORDERED_LEAST_NEGATIVE, __VA_ARGS__)

#define MM_CMPESTRO_SBYTE_TEST_CASES(_, ...)          \
    _(SBYTE_EACH_MOST_NEGATIVE, __VA_ARGS__)          \
    _(SBYTE_ANY_MOST, __VA_ARGS__)                    \
    _(SBYTE_RANGES_MOST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ORDERED_MOST, __VA_ARGS__)

#define MM_CMPESTRO_UWORD_TEST_CASES(_, ...)        \
    _(UWORD_EACH_MOST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(UWORD_ANY_LEAST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(UWORD_RANGES_LEAST_NEGATIVE, __VA_ARGS__)     \
    _(UWORD_ORDERED_MOST, __VA_ARGS__)

#define MM_CMPESTRO_SWORD_TEST_CASES(_, ...)          \
    _(SWORD_RANGES_MOST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SWORD_EACH_MOST_MASKED_NEGATIVE, __VA_ARGS__)   \
    _(SWORD_ANY_MOST_MASKED_NEGATIVE, __VA_ARGS__)    \
    _(SWORD_ORDERED_MOST_MASKED_NEGATIVE, __VA_ARGS__)

#define GENERATE_MM_CMPESTRO_TEST_CASES                                     \
    ENUM_MM_CMPESTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpestro, CMPESTRO,  \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpestro, CMPESTRO,   \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(UWORD, uword, uint16_t, cmpestro, CMPESTRO, \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SWORD, sword, int16_t, cmpestro, CMPESTRO,  \
                                IS_CMPESTRI)

result_t test_mm_cmpestro(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPESTRO_TEST_CASES
    return TEST_SUCCESS;
}

#define TEST_MM_CMPESTRS_UBYTE_DATA_LEN 2
static test_mm_cmpestri_ubyte_data_t
    test_mm_cmpestrs_ubyte_data[TEST_MM_CMPESTRS_UBYTE_DATA_LEN] = {
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         {0},
         16,
         0,
         IMM_UBYTE_ANY_MOST,
         0},
        {{1, 2, 3}, {1, 2, 3}, 3, 8, IMM_UBYTE_RANGES_MOST, 1},
};

#define TEST_MM_CMPESTRS_SBYTE_DATA_LEN 2
static test_mm_cmpestri_sbyte_data_t
    test_mm_cmpestrs_sbyte_data[TEST_MM_CMPESTRS_SBYTE_DATA_LEN] = {
        {{-1, -2, -3, -4, -100, 100, 1, 2, 3, 4},
         {-90, -80, 111, 67, 88},
         10,
         5,
         IMM_SBYTE_EACH_LEAST_MASKED_NEGATIVE,
         1},
        {{99, 100, 101, -99, -100, -101, 56, 7},
         {-128, -126, 100, 127},
         23,
         4,
         IMM_SBYTE_ORDERED_LEAST_MASKED_NEGATIVE,
         0},
};

#define TEST_MM_CMPESTRS_UWORD_DATA_LEN 2
static test_mm_cmpestri_uword_data_t
    test_mm_cmpestrs_uword_data[TEST_MM_CMPESTRS_UWORD_DATA_LEN] = {
        {{1},
         {90, 65535, 63355, 12, 8, 5, 34, 10000},
         100,
         7,
         IMM_UWORD_ANY_MOST_NEGATIVE,
         0},
        {{}, {0}, 0, 28, IMM_UWORD_RANGES_MOST_MASKED_NEGATIVE, 1},
};

#define TEST_MM_CMPESTRS_SWORD_DATA_LEN 2
static test_mm_cmpestri_sword_data_t
    test_mm_cmpestrs_sword_data[TEST_MM_CMPESTRS_SWORD_DATA_LEN] = {
        {{-30000, 2897, 1111, -4455},
         {30, 40, 500, 6000, 20, -10, -789, -29999},
         4,
         8,
         IMM_SWORD_ORDERED_LEAST_MASKED_NEGATIVE,
         1},
        {{34, 56, 789, 1024, 2048, 4096, 8192, -16384},
         {3, 9, -27, 81, -216, 1011},
         9,
         6,
         IMM_SWORD_EACH_LEAST_NEGATIVE,
         0},
};

#define MM_CMPESTRS_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_ANY_MOST, __VA_ARGS__)           \
    _(UBYTE_RANGES_MOST, __VA_ARGS__)

#define MM_CMPESTRS_SBYTE_TEST_CASES(_, ...)         \
    _(SBYTE_EACH_LEAST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ORDERED_LEAST_MASKED_NEGATIVE, __VA_ARGS__)

#define MM_CMPESTRS_UWORD_TEST_CASES(_, ...) \
    _(UWORD_ANY_MOST_NEGATIVE, __VA_ARGS__)  \
    _(UWORD_RANGES_MOST_MASKED_NEGATIVE, __VA_ARGS__)

#define MM_CMPESTRS_SWORD_TEST_CASES(_, ...)        \
    _(SWORD_ANY_LEAST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SWORD_EACH_LEAST_NEGATIVE, __VA_ARGS__)

#define GENERATE_MM_CMPESTRS_TEST_CASES                                     \
    ENUM_MM_CMPESTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpestrs, CMPESTRS,  \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpestrs, CMPESTRS,   \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(UWORD, uword, uint16_t, cmpestrs, CMPESTRS, \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SWORD, sword, int16_t, cmpestrs, CMPESTRS,  \
                                IS_CMPESTRI)

result_t test_mm_cmpestrs(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPESTRS_TEST_CASES
    return TEST_SUCCESS;
}

#define TEST_MM_CMPESTRZ_UBYTE_DATA_LEN 2
static test_mm_cmpestri_ubyte_data_t
    test_mm_cmpestrz_ubyte_data[TEST_MM_CMPESTRZ_UBYTE_DATA_LEN] = {
        {{0, 1, 2, 3, 4, 5, 6, 7},
         {12, 67, 0, 3},
         8,
         4,
         IMM_UBYTE_ANY_MOST_MASKED_NEGATIVE,
         1},
        {{255, 0, 127, 88},
         {1, 2, 4, 8, 16, 32, 64, 128, 254, 233, 209, 41, 66, 77, 90, 100},
         4,
         16,
         IMM_UBYTE_RANGES_MOST_MASKED_NEGATIVE,
         0},
};

#define TEST_MM_CMPESTRZ_SBYTE_DATA_LEN 2
static test_mm_cmpestri_sbyte_data_t
    test_mm_cmpestrz_sbyte_data[TEST_MM_CMPESTRZ_SBYTE_DATA_LEN] = {
        {{}, {-90, -80, 111, 67, 88}, 0, 18, IMM_SBYTE_EACH_LEAST_NEGATIVE, 0},
        {{9, 10, 10, -99, -100, -101, 56, 76},
         {-127, 127, -100, -120, 13, 108, 1, -66, -34, 89, -89, 123, 22, -19,
          -8},
         7,
         15,
         IMM_SBYTE_ORDERED_LEAST_NEGATIVE,
         1},
};

#define TEST_MM_CMPESTRZ_UWORD_DATA_LEN 2
static test_mm_cmpestri_uword_data_t
    test_mm_cmpestrz_uword_data[TEST_MM_CMPESTRZ_UWORD_DATA_LEN] = {
        {{1},
         {9000, 33333, 63333, 120, 8, 55, 34, 100},
         100,
         7,
         IMM_UWORD_ANY_LEAST_NEGATIVE,
         1},
        {{1, 2, 3},
         {1, 10000, 65535, 8964, 9487, 32, 451, 666},
         3,
         8,
         IMM_UWORD_RANGES_MOST_NEGATIVE,
         0},
};

#define TEST_MM_CMPESTRZ_SWORD_DATA_LEN 2
static test_mm_cmpestri_sword_data_t
    test_mm_cmpestrz_sword_data[TEST_MM_CMPESTRZ_SWORD_DATA_LEN] = {
        {{30000, 28997, 11111, 4455},
         {30, 40, 500, 6000, 20, -10, -789, -29999},
         4,
         8,
         IMM_SWORD_ORDERED_LEAST_MASKED_NEGATIVE,
         0},
        {{789, 1024, 2048, 4096, 8192},
         {-3, 9, -27, 18, -217, 10111, 22222},
         5,
         7,
         IMM_SWORD_EACH_LEAST_MASKED_NEGATIVE,
         1},
};

#define MM_CMPESTRZ_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_ANY_MOST, __VA_ARGS__)           \
    _(UBYTE_RANGES_MOST, __VA_ARGS__)

#define MM_CMPESTRZ_SBYTE_TEST_CASES(_, ...)  \
    _(SBYTE_EACH_LEAST_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ORDERED_LEAST_NEGATIVE, __VA_ARGS__)

#define MM_CMPESTRZ_UWORD_TEST_CASES(_, ...) \
    _(UWORD_ANY_LEAST_NEGATIVE, __VA_ARGS__) \
    _(UWORD_RANGES_MOST_NEGATIVE, __VA_ARGS__)

#define MM_CMPESTRZ_SWORD_TEST_CASES(_, ...)        \
    _(SWORD_ANY_LEAST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SWORD_EACH_LEAST_MASKED_NEGATIVE, __VA_ARGS__)

#define GENERATE_MM_CMPESTRZ_TEST_CASES                                     \
    ENUM_MM_CMPESTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpestrz, CMPESTRZ,  \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpestrz, CMPESTRZ,   \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(UWORD, uword, uint16_t, cmpestrz, CMPESTRZ, \
                                IS_CMPESTRI)                                \
    ENUM_MM_CMPESTRX_TEST_CASES(SWORD, sword, int16_t, cmpestrz, CMPESTRZ,  \
                                IS_CMPESTRI)

result_t test_mm_cmpestrz(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPESTRZ_TEST_CASES
    return TEST_SUCCESS;
}

#undef IS_CMPESTRI

result_t test_mm_cmpgt_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t result[2];
    result[0] = _a[0] > _b[0] ? -1 : 0;
    result[1] = _a[1] > _b[1] ? -1 : 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i iret = _mm_cmpgt_epi64(a, b);

    return validateInt64(iret, result[0], result[1]);
}

#define IS_CMPISTRI 1

#define DEF_ENUM_MM_CMPISTRX_VARIANT(c, ...) c,

#define EVAL_MM_CMPISTRX_TEST_CASE(c, type, data_type, im, IM)             \
    do {                                                                   \
        data_type *a = test_mm_##im##_##type##_data[c].a,                  \
                  *b = test_mm_##im##_##type##_data[c].b;                  \
        const int imm8 = IMM_##c;                                          \
        IIF(IM)                                                            \
        (int expect = test_mm_##im##_##type##_data[c].expect,              \
         data_type *expect = test_mm_##im##_##type##_data[c].expect);      \
        __m128i ma, mb;                                                    \
        memcpy(&ma, a, sizeof(ma));                                        \
        memcpy(&mb, b, sizeof(mb));                                        \
        IIF(IM)                                                            \
        (int res = _mm_##im(ma, mb, imm8),                                 \
         __m128i res = _mm_##im(ma, mb, imm8));                            \
        if (IIF(IM)(res != expect, memcmp(expect, &res, sizeof(__m128i)))) \
            return TEST_FAIL;                                              \
    } while (0);

#define ENUM_MM_CMPISTRX_TEST_CASES(type, type_lower, data_type, func, FUNC, \
                                    IM)                                      \
    enum { MM_##FUNC##_##type##_TEST_CASES(DEF_ENUM_MM_CMPISTRX_VARIANT) };  \
    MM_##FUNC##_##type##_TEST_CASES(EVAL_MM_CMPISTRX_TEST_CASE, type_lower,  \
                                    data_type, func, IM)

typedef struct {
    uint8_t a[16], b[16];
    const int imm8;
    int expect;
} test_mm_cmpistri_ubyte_data_t;
typedef struct {
    int8_t a[16], b[16];
    const int imm8;
    int expect;
} test_mm_cmpistri_sbyte_data_t;
typedef struct {
    uint16_t a[8], b[8];
    const int imm8;
    int expect;
} test_mm_cmpistri_uword_data_t;
typedef struct {
    int16_t a[8], b[8];
    const int imm8;
    int expect;
} test_mm_cmpistri_sword_data_t;

result_t test_mm_cmpistra(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistrc(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

#define TEST_MM_CMPISTRI_UBYTE_DATA_LEN 4
static test_mm_cmpistri_ubyte_data_t
    test_mm_cmpistri_ubyte_data[TEST_MM_CMPISTRI_UBYTE_DATA_LEN] = {
        {{104, 117, 110, 116, 114, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {33, 64, 35, 36, 37, 94, 38, 42, 40, 41, 91, 93, 58, 59, 60, 62},
         IMM_UBYTE_ANY_LEAST,
         16},
        {{4, 5, 6, 7, 8, 111, 34, 21, 0, 0, 0, 0, 0, 0, 0, 0},
         {5, 6, 7, 8, 8, 111, 43, 12, 0, 0, 0, 0, 0, 0, 0, 0},
         IMM_UBYTE_EACH_MOST_MASKED_NEGATIVE,
         15},
        {{65, 90, 97, 122, 48, 57, 0},
         {47, 46, 43, 44, 42, 43, 45, 41, 40, 123, 124, 125, 126, 127, 1, 2},
         IMM_UBYTE_RANGES_LEAST,
         16},
        {{111, 222, 22, 0},
         {33, 44, 55, 66, 77, 88, 99, 111, 222, 22, 11, 0},
         IMM_UBYTE_ORDERED_LEAST,
         7},
};

#define TEST_MM_CMPISTRI_SBYTE_DATA_LEN 4
static test_mm_cmpistri_sbyte_data_t
    test_mm_cmpistri_sbyte_data[TEST_MM_CMPISTRI_SBYTE_DATA_LEN] = {
        {{1, 2, 3, 4, 5, -99, -128, -100, -1, 49, 0},
         {2, 3, 3, 4, 5, -100, -128, -99, 1, 44, 0},
         IMM_SBYTE_EACH_LEAST,
         2},
        {{99, 100, 23, -90, 0},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 100, 23, -90, -90, 100},
         IMM_SBYTE_ANY_LEAST,
         10},
        {{-10, -2, 89, 97, 0},
         {-11, -12, -3, 1, 97, 0},
         IMM_SBYTE_RANGES_LEAST_NEGATIVE,
         0},
        {{-10, -90, -22, 30, 87, 127, 0}, {0}, IMM_SBYTE_ORDERED_LEAST, 16},
};

#define TEST_MM_CMPISTRI_UWORD_DATA_LEN 4
static test_mm_cmpistri_uword_data_t
    test_mm_cmpistri_uword_data[TEST_MM_CMPISTRI_UWORD_DATA_LEN] = {
        {{38767, 99, 1234, 65535, 2222, 1, 34456, 11},
         {38768, 999, 1235, 4444, 2222, 1, 34456, 12},
         IMM_UWORD_EACH_LEAST,
         4},
        {{22222, 33333, 44444, 55555, 6000, 600, 60, 6},
         {0},
         IMM_UWORD_ANY_LEAST,
         8},
        {{34, 777, 1000, 1004, 0},
         {33, 32, 889, 1003, 0},
         IMM_UWORD_RANGES_LEAST,
         3},
        {{44, 555, 44, 0},
         {44, 555, 44, 555, 44, 555, 44, 0},
         IMM_UWORD_ORDERED_MOST_NEGATIVE,
         7},
};

#define TEST_MM_CMPISTRI_SWORD_DATA_LEN 4
static test_mm_cmpistri_sword_data_t
    test_mm_cmpistri_sword_data[TEST_MM_CMPISTRI_SWORD_DATA_LEN] = {
        {{-1, -5, 10, 30, 40, 0},
         {13, -2, 7, 80, 11, 0},
         IMM_SWORD_RANGES_LEAST,
         0},
        {{-12, 12, 6666, 777, 0},
         {11, 12, 6666, 777, 0},
         IMM_SWORD_EACH_LEAST,
         1},
        {{23, 22, 33, 567, 9999, 12345, 0},
         {23, 22, 23, 22, 23, 22, 23, 12222},
         IMM_SWORD_ANY_MOST,
         6},
        {{12, -234, -567, 8888, 0},
         {13, -234, -567, 8888, 12, -234, -567, 8889},
         IMM_SWORD_ORDERED_LEAST,
         8},
};

#define MM_CMPISTRI_UBYTE_TEST_CASES(_, ...)        \
    _(UBYTE_ANY_LEAST, __VA_ARGS__)                 \
    _(UBYTE_EACH_MOST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_RANGES_LEAST, __VA_ARGS__)              \
    _(UBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRI_SBYTE_TEST_CASES(_, ...)    \
    _(SBYTE_EACH_LEAST, __VA_ARGS__)            \
    _(SBYTE_ANY_LEAST, __VA_ARGS__)             \
    _(SBYTE_RANGES_LEAST_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRI_UWORD_TEST_CASES(_, ...) \
    _(UWORD_EACH_LEAST, __VA_ARGS__)         \
    _(UWORD_ANY_LEAST, __VA_ARGS__)          \
    _(UWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(UWORD_ORDERED_MOST_NEGATIVE, __VA_ARGS__)

#define MM_CMPISTRI_SWORD_TEST_CASES(_, ...) \
    _(SWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(SWORD_EACH_LEAST, __VA_ARGS__)         \
    _(SWORD_ANY_MOST, __VA_ARGS__)           \
    _(SWORD_ORDERED_LEAST, __VA_ARGS__)

#define GENERATE_MM_CMPISTRI_TEST_CASES                                     \
    ENUM_MM_CMPISTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpistri, CMPISTRI,  \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpistri, CMPISTRI,   \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(UWORD, uword, uint16_t, cmpistri, CMPISTRI, \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SWORD, sword, int16_t, cmpistri, CMPISTRI,  \
                                IS_CMPISTRI)

result_t test_mm_cmpistri(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPISTRI_TEST_CASES
    return TEST_SUCCESS;
}

#define IS_CMPISTRM 0

typedef struct {
    uint8_t a[16], b[16];
    const int imm8;
    uint8_t expect[16];
} test_mm_cmpistrm_ubyte_data_t;
typedef struct {
    int8_t a[16], b[16];
    const int imm8;
    int8_t expect[16];
} test_mm_cmpistrm_sbyte_data_t;
typedef struct {
    uint16_t a[8], b[8];
    const int imm8;
    uint16_t expect[8];
} test_mm_cmpistrm_uword_data_t;
typedef struct {
    int16_t a[8], b[8];
    const int imm8;
    int16_t expect[8];
} test_mm_cmpistrm_sword_data_t;

#define TEST_MM_CMPISTRM_UBYTE_DATA_LEN 4
static test_mm_cmpistrm_ubyte_data_t
    test_mm_cmpistrm_ubyte_data[TEST_MM_CMPISTRM_UBYTE_DATA_LEN] = {
        {{88, 89, 90, 91, 92, 93, 0},
         {78, 88, 99, 127, 92, 93, 0},
         IMM_UBYTE_EACH_UNIT,
         {0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
          255}},
        {{30, 41, 52, 63, 74, 85, 0},
         {30, 42, 51, 63, 74, 85, 0},
         IMM_UBYTE_ANY_BIT,
         {57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{34, 32, 21, 16, 7, 0},
         {34, 33, 32, 31, 30, 29, 10, 6, 0},
         IMM_UBYTE_RANGES_UNIT,
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{33, 21, 123, 89, 76, 56, 0},
         {33, 21, 124, 33, 21, 123, 89, 76, 56, 33, 21, 123, 89, 76, 56, 22},
         IMM_UBYTE_ORDERED_UNIT,
         {0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0}},
};

#define TEST_MM_CMPISTRM_SBYTE_DATA_LEN 4
static test_mm_cmpistrm_sbyte_data_t
    test_mm_cmpistrm_sbyte_data[TEST_MM_CMPISTRM_SBYTE_DATA_LEN] = {
        {{-11, -90, -128, 127, 66, 45, 23, 32, 99, 10, 0},
         {-10, -90, -124, 33, 66, 45, 23, 22, 99, 100, 0},
         IMM_SBYTE_EACH_BIT_MASKED_NEGATIVE,
         {-115, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{13, 14, 55, 1, 32, 100, 101, 102, 103, 97, 23, 21, 45, 54, 55, 56},
         {22, 109, 87, 45, 1, 103, 22, 102, 43, 87, 78, 56, 65, 55, 44, 33},
         IMM_SBYTE_ANY_UNIT,
         {0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0}},
        {{-31, -28, -9, 10, 45, 67, 88, 0},
         {-30, -32, -33, -44, 93, 44, 9, 89, 0},
         IMM_SBYTE_RANGES_UNIT,
         {-1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{34, -10, 78, -99, -100, 100, 0},
         {34, 123, 88, 4, 34, -10, 78, -99, -100, 100, 34, -10, 78, -99, -100,
          -100},
         IMM_SBYTE_ORDERED_UNIT,
         {0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
};

#define TEST_MM_CMPISTRM_UWORD_DATA_LEN 4
static test_mm_cmpistrm_uword_data_t
    test_mm_cmpistrm_uword_data[TEST_MM_CMPISTRM_UWORD_DATA_LEN] = {
        {{1024, 2048, 4096, 5000, 0},
         {1023, 1000, 2047, 1596, 5566, 5666, 4477, 9487},
         IMM_UWORD_RANGES_UNIT,
         {0, 0, 65535, 65535, 0, 0, 65535, 0}},
        {{1, 2, 345, 7788, 10000, 0},
         {2, 1, 345, 7788, 10000, 0},
         IMM_UWORD_EACH_UNIT,
         {0, 0, 65535, 65535, 65535, 65535, 65535, 65535}},
        {{100, 0},
         {12345, 6766, 234, 0, 1, 34, 89, 100},
         IMM_UWORD_ANY_UNIT,
         {0, 0, 0, 0, 0, 0, 0, 0}},
        {{34, 122, 9000, 0},
         {34, 122, 9000, 34, 122, 9000, 34, 122},
         IMM_UWORD_ORDERED_UNIT_NEGATIVE,
         {0, 65535, 65535, 0, 65535, 65535, 0, 65535}},
};

#define TEST_MM_CMPISTRM_SWORD_DATA_LEN 4
static test_mm_cmpistrm_sword_data_t
    test_mm_cmpistrm_sword_data[TEST_MM_CMPISTRM_SWORD_DATA_LEN] = {
        {{-39, -10, 17, 89, 998, 1000, 1234, 4566},
         {-40, -52, -39, -29, 100, 1024, 4565, 4600},
         IMM_SWORD_RANGES_BIT,
         {0, 0, -1, -1, 0, 0, -1, 0}},
        {{345, -1900, -10000, -30000, 50, 6789, 0},
         {103, -1901, -10000, 32767, 50, 6780, 0},
         IMM_SWORD_EACH_UNIT,
         {0, 0, -1, 0, -1, 0, -1, -1}},
        {{677, 10001, 1001, 23, 0},
         {345, 677, 10001, 1003, 1001, 32, 23, 677},
         IMM_SWORD_ANY_UNIT,
         {0, -1, -1, 0, -1, 0, -1, -1}},
        {{1024, -2288, 3752, -4096, 0},
         {1024, 1024, -2288, 3752, -4096, 1024, -2288, 3752},
         IMM_SWORD_ORDERED_UNIT,
         {0, -1, 0, 0, 0, -1, 0, 0}},
};

#define MM_CMPISTRM_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_EACH_UNIT, __VA_ARGS__)          \
    _(UBYTE_ANY_BIT, __VA_ARGS__)            \
    _(UBYTE_RANGES_UNIT, __VA_ARGS__)        \
    _(UBYTE_ORDERED_UNIT, __VA_ARGS__)

#define MM_CMPISTRM_SBYTE_TEST_CASES(_, ...)       \
    _(SBYTE_EACH_BIT_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ANY_UNIT, __VA_ARGS__)                 \
    _(SBYTE_RANGES_UNIT, __VA_ARGS__)              \
    _(SBYTE_ORDERED_UNIT, __VA_ARGS__)

#define MM_CMPISTRM_UWORD_TEST_CASES(_, ...) \
    _(UWORD_RANGES_UNIT, __VA_ARGS__)        \
    _(UWORD_EACH_UNIT, __VA_ARGS__)          \
    _(UWORD_ANY_UNIT, __VA_ARGS__)           \
    _(UWORD_ORDERED_UNIT_NEGATIVE, __VA_ARGS__)

#define MM_CMPISTRM_SWORD_TEST_CASES(_, ...) \
    _(SWORD_RANGES_UNIT, __VA_ARGS__)        \
    _(SWORD_EACH_UNIT, __VA_ARGS__)          \
    _(SWORD_ANY_UNIT, __VA_ARGS__)           \
    _(SWORD_ORDERED_UNIT, __VA_ARGS__)

#define GENERATE_MM_CMPISTRM_TEST_CASES                                     \
    ENUM_MM_CMPISTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpistrm, CMPISTRM,  \
                                IS_CMPISTRM)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpistrm, CMPISTRM,   \
                                IS_CMPISTRM)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(UWORD, uword, uint16_t, cmpistrm, CMPISTRM, \
                                IS_CMPISTRM)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SWORD, sword, int16_t, cmpistrm, CMPISTRM,  \
                                IS_CMPISTRM)

result_t test_mm_cmpistrm(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPISTRM_TEST_CASES
    return TEST_SUCCESS;
}

#undef IS_CMPISTRM

result_t test_mm_cmpistro(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistrs(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistrz(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_UNIMPL;
}

result_t test_mm_crc32_u16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint32_t crc = *(const uint32_t *) impl.mTestIntPointer1;
    uint16_t v = iter;
    uint32_t result = _mm_crc32_u16(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u16(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint32_t crc = *(const uint32_t *) impl.mTestIntPointer1;
    uint32_t v = *(const uint32_t *) impl.mTestIntPointer2;
    uint32_t result = _mm_crc32_u32(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u32(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint64_t crc = *(const uint64_t *) impl.mTestIntPointer1;
    uint64_t v = *(const uint64_t *) impl.mTestIntPointer2;
    uint64_t result = _mm_crc32_u64(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u64(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint32_t crc = *(const uint32_t *) impl.mTestIntPointer1;
    uint8_t v = iter;
    uint32_t result = _mm_crc32_u8(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u8(crc, v));
    return TEST_SUCCESS;
}

/* AES */
result_t test_mm_aesenc_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *a = (int32_t *) impl.mTestIntPointer1;
    const int32_t *b = (int32_t *) impl.mTestIntPointer2;
    __m128i data = _mm_loadu_si128((const __m128i *) a);
    __m128i rk = _mm_loadu_si128((const __m128i *) b);

    __m128i resultReference = aesenc_128_reference(data, rk);
    __m128i resultIntrinsic = _mm_aesenc_si128(data, rk);

    return validate128(resultReference, resultIntrinsic);
}

result_t test_mm_aesenclast_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *b = (const int32_t *) impl.mTestIntPointer2;
    __m128i data = _mm_loadu_si128((const __m128i *) a);
    __m128i rk = _mm_loadu_si128((const __m128i *) b);

    __m128i resultReference = aesenclast_128_reference(data, rk);
    __m128i resultIntrinsic = _mm_aesenclast_si128(data, rk);

    return validate128(resultReference, resultIntrinsic);
}

// FIXME: improve the test case for AES-256 key expansion.
// Reference:
// https://github.com/randombit/botan/blob/master/src/lib/block/aes/aes_ni/aes_ni.cpp
result_t test_mm_aeskeygenassist_si128(const SSE2NEONTestImpl &impl,
                                       uint32_t iter)
{
    const int32_t *a = (int32_t *) impl.mTestIntPointer1;
    const int32_t *b = (int32_t *) impl.mTestIntPointer2;
    __m128i data = _mm_loadu_si128((const __m128i *) a);

    (void) b;  // parameter b is unused because we can only pass an 8-bit
               // immediate to _mm_aeskeygenassist_si128.
    const int8_t rcon = 0x40; /* an arbitrary 8-bit immediate */
    __m128i resultReference = aeskeygenassist_128_reference(data, rcon);
    __m128i resultIntrinsic = _mm_aeskeygenassist_si128(data, rcon);

    return validate128(resultReference, resultIntrinsic);
}

/* Others */
result_t test_mm_clmulepi64_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint64_t *_a = (const uint64_t *) impl.mTestIntPointer1;
    const uint64_t *_b = (const uint64_t *) impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    auto result = clmul_64(_a[0], _b[0]);
    if (!validateUInt64(_mm_clmulepi64_si128(a, b, 0x00), result.first,
                        result.second))
        return TEST_FAIL;
    result = clmul_64(_a[1], _b[0]);
    if (!validateUInt64(_mm_clmulepi64_si128(a, b, 0x01), result.first,
                        result.second))
        return TEST_FAIL;
    result = clmul_64(_a[0], _b[1]);
    if (!validateUInt64(_mm_clmulepi64_si128(a, b, 0x10), result.first,
                        result.second))
        return TEST_FAIL;
    result = clmul_64(_a[1], _b[1]);
    if (!validateUInt64(_mm_clmulepi64_si128(a, b, 0x11), result.first,
                        result.second))
        return TEST_FAIL;
    return TEST_SUCCESS;
}

result_t test_mm_get_denormals_zero_mode(const SSE2NEONTestImpl &impl,
                                         uint32_t iter)
{
    int res_denormals_zero_on, res_denormals_zero_off;

    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    res_denormals_zero_on =
        _MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON;

    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    res_denormals_zero_off =
        _MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_OFF;

    return (res_denormals_zero_on && res_denormals_zero_off) ? TEST_SUCCESS
                                                             : TEST_FAIL;
}

result_t test_mm_popcnt_u32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint64_t *a = (const uint64_t *) impl.mTestIntPointer1;
    ASSERT_RETURN(__builtin_popcount(a[0]) == _mm_popcnt_u32(a[0]));
    return TEST_SUCCESS;
}

result_t test_mm_popcnt_u64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint64_t *a = (const uint64_t *) impl.mTestIntPointer1;
    ASSERT_RETURN(__builtin_popcountll(a[0]) == _mm_popcnt_u64(a[0]));
    return TEST_SUCCESS;
}

result_t test_mm_set_denormals_zero_mode(const SSE2NEONTestImpl &impl,
                                         uint32_t iter)
{
    result_t res_set_denormals_zero_on, res_set_denormals_zero_off;
    float factor = 2;
    float denormal = FLT_MIN / factor;
    float denormals[4] = {denormal, denormal, denormal, denormal};
    float factors[4] = {factor, factor, factor, factor};
    __m128 ret;

    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    ret = _mm_mul_ps(load_m128(denormals), load_m128(factors));
    res_set_denormals_zero_on = validateFloat(ret, 0, 0, 0, 0);

    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    ret = _mm_mul_ps(load_m128(denormals), load_m128(factors));
#if defined(__arm__)
    // AArch32 Advanced SIMD arithmetic always uses the Flush-to-zero setting,
    // regardless of the value of the FZ bit.
    res_set_denormals_zero_off = validateFloat(ret, 0, 0, 0, 0);
#else
    res_set_denormals_zero_off =
        validateFloat(ret, FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN);
#endif

    if (res_set_denormals_zero_on == TEST_FAIL ||
        res_set_denormals_zero_off == TEST_FAIL)
        return TEST_FAIL;
    return TEST_SUCCESS;
}

result_t test_rdtsc(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint64_t start = _rdtsc();
    for (int i = 0; i < 100000; i++)
        __asm__ __volatile__("" ::: "memory");
    uint64_t end = _rdtsc();
    return end > start ? TEST_SUCCESS : TEST_FAIL;
}

SSE2NEONTestImpl::SSE2NEONTestImpl(void)
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

// Dummy function to match the case label in runSingleTest.
result_t test_last(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return TEST_SUCCESS;
}

result_t SSE2NEONTestImpl::loadTestFloatPointers(uint32_t i)
{
    result_t ret =
        do_mm_store_ps(mTestFloatPointer1, mTestFloats[i], mTestFloats[i + 1],
                       mTestFloats[i + 2], mTestFloats[i + 3]);
    if (ret == TEST_SUCCESS) {
        ret = do_mm_store_ps(mTestFloatPointer2, mTestFloats[i + 4],
                             mTestFloats[i + 5], mTestFloats[i + 6],
                             mTestFloats[i + 7]);
    }
    return ret;
}

result_t SSE2NEONTestImpl::loadTestIntPointers(uint32_t i)
{
    result_t ret =
        do_mm_store_ps(mTestIntPointer1, mTestInts[i], mTestInts[i + 1],
                       mTestInts[i + 2], mTestInts[i + 3]);
    if (ret == TEST_SUCCESS) {
        ret =
            do_mm_store_ps(mTestIntPointer2, mTestInts[i + 4], mTestInts[i + 5],
                           mTestInts[i + 6], mTestInts[i + 7]);
    }

    return ret;
}

result_t SSE2NEONTestImpl::runSingleTest(InstructionTest test, uint32_t i)
{
    result_t ret = TEST_SUCCESS;

    switch (test) {
        INTRIN_FOREACH(CASE)
    }

    return ret;
}

SSE2NEONTest *SSE2NEONTest::create(void)
{
    SSE2NEONTestImpl *st = new SSE2NEONTestImpl;
    return static_cast<SSE2NEONTest *>(st);
}

}  // namespace SSE2NEON
