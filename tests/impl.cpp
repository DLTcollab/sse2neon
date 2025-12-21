/* ARM64EC requires sse2neon.h to be included FIRST. See common.h for details.
 */
#if defined(_M_ARM64EC)
#include "sse2neon.h"
#endif

#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

#include "binding.h"
#include "impl.h"

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

// Some intrinsics operate on unaligned data types.
#if defined(__GNUC__) || defined(__clang__)
#define ALIGN_STRUCT(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#ifndef ALIGN_STRUCT
#define ALIGN_STRUCT(x) __declspec(align(x))
#endif
#endif

typedef int16_t ALIGN_STRUCT(1) unaligned_int16_t;
typedef int32_t ALIGN_STRUCT(1) unaligned_int32_t;
typedef int64_t ALIGN_STRUCT(1) unaligned_int64_t;

// This program a set of unit tests to ensure that each SSE call provide the
// output we expect.  If this fires an assert, then something didn't match up.
//
// Functions with "test_" prefix will be called in runSingleTest.
namespace SSE2NEON
{
// Inject a double-precision NaN into a float array for double-precision tests.
// The float arrays are reinterpreted as doubles via reinterpret_cast, where two
// adjacent floats (8 bytes) form one double: double[0] = float[0..1],
// double[1] = float[2..3].
static inline void inject_double_nan(float *arr, uint32_t double_idx)
{
    double nan_d = nan("");
    memcpy(&arr[double_idx * 2], &nan_d, sizeof(double));
}

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
    int8_t mTestUnalignedInts[32] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    };

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
            if (test == it_mm_rcp_ps || test == it_mm_rcp_ss ||
                test == it_mm_rsqrt_ps || test == it_mm_rsqrt_ss) {
                if ((rand() & 3) == 0) {
                    uint32_t r1 = rand() & 3;
                    uint32_t r2 = rand() & 3;
                    uint32_t r3 = rand() & 3;
                    uint32_t r4 = rand() & 3;
                    uint32_t r5 = rand() & 3;
                    uint32_t r6 = rand() & 3;
                    uint32_t r7 = rand() & 3;
                    uint32_t r8 = rand() & 3;
                    mTestFloatPointer1[r1] = 0.0f;
                    mTestFloatPointer1[r2] = 0.0f;
                    mTestFloatPointer1[r3] = 0.0f;
                    mTestFloatPointer1[r4] = 0.0f;
                    mTestFloatPointer1[r5] = -0.0f;
                    mTestFloatPointer1[r6] = -0.0f;
                    mTestFloatPointer1[r7] = -0.0f;
                    mTestFloatPointer1[r8] = -0.0f;
                }
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
                // Inject double NaN one out of four times.
                if ((rand() & 3) == 0) {
                    uint32_t nan_idx = rand() & 1;
                    inject_double_nan(mTestFloatPointer1, nan_idx);
                    inject_double_nan(mTestFloatPointer2, nan_idx);
                }
            }

            if (test == it_mm_max_pd || test == it_mm_max_sd ||
                test == it_mm_min_pd || test == it_mm_min_sd) {
                // Make sure the positive/negative inifinity values are included
                // in the testing one out of four times.
                if ((rand() & 3) == 0) {
                    uint32_t r1 = static_cast<uint32_t>((rand() & 1) << 1) + 1;
                    uint32_t r2 = static_cast<uint32_t>((rand() & 1) << 1) + 1;
                    uint32_t r3 = static_cast<uint32_t>((rand() & 1) << 1) + 1;
                    uint32_t r4 = static_cast<uint32_t>((rand() & 1) << 1) + 1;
                    mTestFloatPointer1[r1] = INFINITY;
                    mTestFloatPointer2[r2] = INFINITY;
                    mTestFloatPointer1[r3] = -INFINITY;
                    mTestFloatPointer1[r4] = -INFINITY;
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
                // Inject double NaN one out of four times.
                if ((rand() & 3) == 0) {
                    uint32_t nan_idx = rand() & 1;
                    inject_double_nan(mTestFloatPointer1, nan_idx);
                    inject_double_nan(mTestFloatPointer2, nan_idx);
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

const char *instructionString[] = {
#define _(x) #x,
    INTRIN_LIST
#undef _
};

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

// SplitMix64 PRNG by Sebastiano Vigna, see:
// <https://xoshiro.di.unimi.it/splitmix64.c>
static uint64_t state;  // the state of SplitMix64 PRNG
const double TWOPOWER64 = pow(2, 64);

#define SSE2NEON_INIT_RNG(seed) \
    do {                        \
        state = seed;           \
    } while (0)

static double next()
{
    uint64_t z = (state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return static_cast<double>(z ^ (z >> 31));
}

static float ranf()
{
    return static_cast<float>(next() / TWOPOWER64);
}

static float ranf(float low, float high)
{
    return ranf() * (high - low) + low;
}

// Enable the tests which are using the macro of another tests
result_t test_mm_slli_si128(const SSE2NEONTestImpl &impl, uint32_t iter);
result_t test_mm_srli_si128(const SSE2NEONTestImpl &impl, uint32_t iter);
result_t test_mm_shuffle_pi16(const SSE2NEONTestImpl &impl, uint32_t iter);

// This function is not called from "runSingleTest", but for other intrinsic
// tests that might need to call "_mm_set_epi32".
__m128i do_mm_set_epi32(int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    validateInt32(a, w, z, y, x);
    return a;
}

// This function is not called from "runSingleTest", but for other intrinsic
// tests that might need to load __m64 data.
template <class T>
__m64 load_m64(const T *p)
{
    return *(reinterpret_cast<const __m64 *>(p));
}

// This function is not called from "runSingleTest", but for other intrinsic
// tests that might need to call "_mm_load_ps".
template <class T>
__m128 load_m128(const T *p)
{
    return _mm_loadu_ps(reinterpret_cast<const float *>(p));
}

// This function is not called from "runSingleTest", but for other intrinsic
// tests that might need to call "_mm_load_ps".
template <class T>
__m128i load_m128i(const T *p)
{
    __m128 a = _mm_loadu_ps(reinterpret_cast<const float *>(p));
    __m128i ia = _mm_castps_si128(a);
    return ia;
}

// This function is not called from "runSingleTest", but for other intrinsic
// tests that might need to call "_mm_load_pd".
template <class T>
__m128d load_m128d(const T *p)
{
    return _mm_loadu_pd(reinterpret_cast<const double *>(p));
}

// This function is not called from "runSingleTest", but for other intrinsic
// tests that might need to call "_mm_store_ps".
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

// This function is not called from "runSingleTest", but for other intrinsic
// tests that might need to call "_mm_store_ps".
result_t do_mm_store_ps(int32_t *p, int32_t x, int32_t y, int32_t z, int32_t w)
{
    __m128i a = _mm_set_epi32(x, y, z, w);
    _mm_store_ps(reinterpret_cast<float *>(p), _mm_castsi128_ps(a));
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
    if (isnan(a) || isnan(b))
        return 0;
    return (a < b);
}

int32_t comigt_ss(float a, float b)
{
    if (isnan(a) || isnan(b))
        return 0;
    return (a > b);
}

int32_t comile_ss(float a, float b)
{
    if (isnan(a) || isnan(b))
        return 0;
    return (a <= b);
}

int32_t comige_ss(float a, float b)
{
    if (isnan(a) || isnan(b))
        return 0;
    return (a >= b);
}

int32_t comieq_ss(float a, float b)
{
    if (isnan(a) || isnan(b))
        return 0;
    return (a == b);
}

int32_t comineq_ss(float a, float b)
{
    if (isnan(a) || isnan(b))
        return 1;
    return (a != b);
}

static inline int16_t saturate_i16(int32_t a)
{
    if (a > INT16_MAX)
        return INT16_MAX;
    if (a < INT16_MIN)
        return INT16_MIN;
    return static_cast<int16_t>(a);
}

static inline uint16_t saturate_u16(uint32_t a)
{
    if (a > UINT16_MAX)
        return UINT16_MAX;
    return static_cast<uint16_t>(a);
}

static inline int8_t saturate_i8(int32_t a)
{
    if (a > INT8_MAX)
        return INT8_MAX;
    if (a < INT8_MIN)
        return INT8_MIN;
    return static_cast<int8_t>(a);
}

uint32_t canonical_crc32_u8(uint32_t crc, uint8_t v)
{
    crc ^= v;
    for (int bit = 0; bit < 8; bit++) {
        if (crc & 1)
            crc = (crc >> 1) ^ UINT32_C(0x82f63b78);
        else
            crc = (crc >> 1);
    }
    return crc;
}

uint32_t canonical_crc32_u16(uint32_t crc, uint16_t v)
{
    crc = canonical_crc32_u8(crc, static_cast<uint8_t>(v & 0xff));
    crc = canonical_crc32_u8(crc, static_cast<uint8_t>((v >> 8) & 0xff));
    return crc;
}

uint32_t canonical_crc32_u32(uint32_t crc, uint32_t v)
{
    crc = canonical_crc32_u16(crc, static_cast<uint16_t>(v & 0xffff));
    crc = canonical_crc32_u16(crc, static_cast<uint16_t>(v >> 16) & 0xffff);
    return crc;
}

uint64_t canonical_crc32_u64(uint64_t crc, uint64_t v)
{
    crc = canonical_crc32_u32(static_cast<uint32_t>(crc),
                              static_cast<uint32_t>(v & 0xffffffff));
    crc = canonical_crc32_u32(static_cast<uint32_t>(crc),
                              static_cast<uint32_t>((v >> 32) & 0xffffffff));
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

static const uint8_t crypto_aes_rsbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e,
    0x81, 0xf3, 0xd7, 0xfb, 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb, 0x54, 0x7b, 0x94, 0x32,
    0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49,
    0x6d, 0x8b, 0xd1, 0x25, 0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92, 0x6c, 0x70, 0x48, 0x50,
    0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05,
    0xb8, 0xb3, 0x45, 0x06, 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, 0x3a, 0x91, 0x11, 0x41,
    0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8,
    0x1c, 0x75, 0xdf, 0x6e, 0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, 0xfc, 0x56, 0x3e, 0x4b,
    0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59,
    0x27, 0x80, 0xec, 0x5f, 0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, 0xa0, 0xe0, 0x3b, 0x4d,
    0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63,
    0x55, 0x21, 0x0c, 0x7d,
};

// XT is x_time function that muliplies 'x' by 2 in GF(2^8)
#define XT(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))

inline __m128i aesenc_128_reference(__m128i a, __m128i b)
{
    uint8_t i, t, u, v[4][4];
    for (i = 0; i < 16; ++i) {
        v[((i / 4) + 4 - (i % 4)) % 4][i % 4] =
            crypto_aes_sbox[(reinterpret_cast<SIMDVec *>(&a))->m128_u8[i]];
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
        (reinterpret_cast<SIMDVec *>(&a))->m128_u8[i] =
            v[i / 4][i % 4] ^ (reinterpret_cast<SIMDVec *>(&b))->m128_u8[i];
    }

    return a;
}

#define MULTIPLY(x, y)                                                     \
    (((y & 1) * x) ^ ((y >> 1 & 1) * XT(x)) ^ ((y >> 2 & 1) * XT(XT(x))) ^ \
     ((y >> 3 & 1) * XT(XT(XT(x)))) ^ ((y >> 4 & 1) * XT(XT(XT(XT(x))))))

inline __m128i aesdec_128_reference(__m128i a, __m128i b)
{
    uint8_t i, e, f, g, h, v[4][4];
    for (i = 0; i < 16; ++i) {
        v[((i / 4) + (i % 4)) % 4][i % 4] =
            crypto_aes_rsbox[(reinterpret_cast<SIMDVec *>(&a))->m128_u8[i]];
    }

    for (i = 0; i < 4; ++i) {
        e = v[i][0];
        f = v[i][1];
        g = v[i][2];
        h = v[i][3];

        v[i][0] = static_cast<uint8_t>(MULTIPLY(e, 0x0e) ^ MULTIPLY(f, 0x0b) ^
                                       MULTIPLY(g, 0x0d) ^ MULTIPLY(h, 0x09));
        v[i][1] = static_cast<uint8_t>(MULTIPLY(e, 0x09) ^ MULTIPLY(f, 0x0e) ^
                                       MULTIPLY(g, 0x0b) ^ MULTIPLY(h, 0x0d));
        v[i][2] = static_cast<uint8_t>(MULTIPLY(e, 0x0d) ^ MULTIPLY(f, 0x09) ^
                                       MULTIPLY(g, 0x0e) ^ MULTIPLY(h, 0x0b));
        v[i][3] = static_cast<uint8_t>(MULTIPLY(e, 0x0b) ^ MULTIPLY(f, 0x0d) ^
                                       MULTIPLY(g, 0x09) ^ MULTIPLY(h, 0x0e));
    }

    for (i = 0; i < 16; ++i) {
        (reinterpret_cast<SIMDVec *>(&a))->m128_u8[i] =
            v[i / 4][i % 4] ^ (reinterpret_cast<SIMDVec *>(&b))->m128_u8[i];
    }
    return a;
}

inline __m128i aesenclast_128_reference(__m128i s, __m128i rk)
{
    uint8_t i, v[4][4];
    for (i = 0; i < 16; ++i)
        v[((i / 4) + 4 - (i % 4)) % 4][i % 4] =
            crypto_aes_sbox[(reinterpret_cast<SIMDVec *>(&s))->m128_u8[i]];
    for (i = 0; i < 16; ++i)
        (reinterpret_cast<SIMDVec *>(&s))->m128_u8[i] =
            v[i / 4][i % 4] ^ (reinterpret_cast<SIMDVec *>(&rk))->m128_u8[i];
    return s;
}

// Rotates right (circular right shift) value by "amount" positions
static inline uint32_t rotr(uint32_t value, uint32_t amount)
{
    return (value >> amount) | (value << ((32 - amount) & 31));
}

static inline uint64_t MUL(uint32_t a, uint32_t b)
{
    return static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
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

    x0 = x & static_cast<uint32_t>(0x11111111);
    x1 = x & static_cast<uint32_t>(0x22222222);
    x2 = x & static_cast<uint32_t>(0x44444444);
    x3 = x & static_cast<uint32_t>(0x88888888);
    y0 = y & static_cast<uint32_t>(0x11111111);
    y1 = y & static_cast<uint32_t>(0x22222222);
    y2 = y & static_cast<uint32_t>(0x44444444);
    y3 = y & static_cast<uint32_t>(0x88888888);
    z0 = MUL(x0, y0) ^ MUL(x1, y3) ^ MUL(x2, y2) ^ MUL(x3, y1);
    z1 = MUL(x0, y1) ^ MUL(x1, y0) ^ MUL(x2, y3) ^ MUL(x3, y2);
    z2 = MUL(x0, y2) ^ MUL(x1, y1) ^ MUL(x2, y0) ^ MUL(x3, y3);
    z3 = MUL(x0, y3) ^ MUL(x1, y2) ^ MUL(x2, y1) ^ MUL(x3, y0);
    z0 &= static_cast<uint64_t>(0x1111111111111111);
    z1 &= static_cast<uint64_t>(0x2222222222222222);
    z2 &= static_cast<uint64_t>(0x4444444444444444);
    z3 &= static_cast<uint64_t>(0x8888888888888888);
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
    uint32_t x0 = static_cast<uint32_t>(x & 0xffffffff);
    uint32_t x1 = static_cast<uint32_t>(x >> 32);
    // y = (y1 * B^m) + y0
    uint32_t y0 = static_cast<uint32_t>(y & 0xffffffff);
    uint32_t y1 = static_cast<uint32_t>(y >> 32);

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
    float d[4];
    d[0] = _a[0] + _b[0];
    d[1] = _a[1] + _b[1];
    d[2] = _a[2] + _b[2];
    d[3] = _a[3] + _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_add_ps(a, b);
    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_add_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f[4];
    f[0] = _a[0] + _b[0];
    f[1] = _a[1];
    f[2] = _a[2];
    f[3] = _a[3];

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_add_ss(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_and_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_and_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = reinterpret_cast<const uint32_t *>(&a);
    const uint32_t *ib = reinterpret_cast<const uint32_t *>(&b);
    uint32_t r[4];
    r[0] = ia[0] & ib[0];
    r[1] = ia[1] & ib[1];
    r[2] = ia[2] & ib[2];
    r[3] = ia[3] & ib[3];
    __m128i ret =
        do_mm_set_epi32(static_cast<int32_t>(r[3]), static_cast<int32_t>(r[2]),
                        static_cast<int32_t>(r[1]), static_cast<int32_t>(r[0]));
    result_t res = VALIDATE_UINT32_M128(_mm_castps_si128(c), r);
    if (res) {
        res = VALIDATE_UINT32_M128(ret, r);
    }
    return res;
}

// r0 := ~a0 & b0
// r1 := ~a1 & b1
// r2 := ~a2 & b2
// r3 := ~a3 & b3
result_t test_mm_andnot_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_andnot_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = reinterpret_cast<const uint32_t *>(&a);
    const uint32_t *ib = reinterpret_cast<const uint32_t *>(&b);
    uint32_t r[4];
    r[0] = ~ia[0] & ib[0];
    r[1] = ~ia[1] & ib[1];
    r[2] = ~ia[2] & ib[2];
    r[3] = ~ia[3] & ib[3];
    __m128i ret =
        do_mm_set_epi32(static_cast<int32_t>(r[3]), static_cast<int32_t>(r[2]),
                        static_cast<int32_t>(r[1]), static_cast<int32_t>(r[0]));
    result_t res = TEST_FAIL;
    res = VALIDATE_UINT32_M128(_mm_castps_si128(c), r);
    if (res) {
        res = VALIDATE_UINT32_M128(ret, r);
    }
    return res;
}

result_t test_mm_avg_pu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);
    const uint16_t *_b =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer2);
    uint16_t d[4];
    d[0] = static_cast<uint16_t>((_a[0] + _b[0] + 1) >> 1);
    d[1] = static_cast<uint16_t>((_a[1] + _b[1] + 1) >> 1);
    d[2] = static_cast<uint16_t>((_a[2] + _b[2] + 1) >> 1);
    d[3] = static_cast<uint16_t>((_a[3] + _b[3] + 1) >> 1);

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_avg_pu16(a, b);

    return VALIDATE_UINT16_M64(c, d);
}

result_t test_mm_avg_pu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    uint8_t d[8];
    d[0] = static_cast<uint8_t>((_a[0] + _b[0] + 1) >> 1);
    d[1] = static_cast<uint8_t>((_a[1] + _b[1] + 1) >> 1);
    d[2] = static_cast<uint8_t>((_a[2] + _b[2] + 1) >> 1);
    d[3] = static_cast<uint8_t>((_a[3] + _b[3] + 1) >> 1);
    d[4] = static_cast<uint8_t>((_a[4] + _b[4] + 1) >> 1);
    d[5] = static_cast<uint8_t>((_a[5] + _b[5] + 1) >> 1);
    d[6] = static_cast<uint8_t>((_a[6] + _b[6] + 1) >> 1);
    d[7] = static_cast<uint8_t>((_a[7] + _b[7] + 1) >> 1);

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_avg_pu8(a, b);

    return VALIDATE_UINT8_M64(c, d);
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
    __m128i iret = _mm_castps_si128(ret);
    return VALIDATE_INT32_M128(iret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    __m128i iret = _mm_castps_si128(ret);
    return VALIDATE_INT32_M128(iret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    __m128i iret = _mm_castps_si128(ret);
    return VALIDATE_INT32_M128(iret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    __m128i iret = _mm_castps_si128(ret);
    return VALIDATE_INT32_M128(iret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    __m128i iret = _mm_castps_si128(ret);
    return VALIDATE_INT32_M128(iret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    __m128i iret = _mm_castps_si128(ret);
    return VALIDATE_INT32_M128(iret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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
    return VALIDATE_FLOAT_M128(ret, result);
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

    return VALIDATE_FLOAT_M128(ret, result);
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

    return VALIDATE_FLOAT_M128(ret, result);
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

    return VALIDATE_FLOAT_M128(ret, result);
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

    return VALIDATE_FLOAT_M128(ret, result);
}

result_t test_mm_comieq_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comieq_ss correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
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
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
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
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
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
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
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

    float d[4];
    d[0] = static_cast<float>(_b[0]);
    d[1] = static_cast<float>(_b[1]);
    d[2] = _a[2];
    d[3] = _a[3];

    __m128 a = load_m128(_a);
    __m64 b = load_m64(_b);
    __m128 c = _mm_cvt_pi2ps(a, b);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvt_ps2pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    for (int idx = 0; idx < 2; idx++) {
        switch (iter & 0x3) {
        case 0:
            _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
            d[idx] = sse2neon_saturate_cast_int32(bankersRounding(_a[idx]));
            break;
        case 1:
            _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
            d[idx] = sse2neon_saturate_cast_int32(floorf(_a[idx]));
            break;
        case 2:
            _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
            d[idx] = sse2neon_saturate_cast_int32(ceilf(_a[idx]));
            break;
        case 3:
            _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
            d[idx] = sse2neon_saturate_cast_int32(_a[idx]);
            break;
        }
    }

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvt_ps2pi(a);

    return VALIDATE_INT32_M64(ret, d);
}

result_t test_mm_cvt_si2ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t b = *impl.mTestIntPointer2;

    float d[4];
    d[0] = static_cast<float>(b);
    d[1] = _a[1];
    d[2] = _a[2];
    d[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_cvt_si2ss(a, b);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvt_ss2si(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d0;

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d0 = sse2neon_saturate_cast_int32(bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d0 = sse2neon_saturate_cast_int32(floorf(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d0 = sse2neon_saturate_cast_int32(ceilf(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d0 = sse2neon_saturate_cast_int32(_a[0]);
        break;
    }

    __m128 a = load_m128(_a);
    int32_t ret = _mm_cvt_ss2si(a);
    return ret == d0 ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtpi16_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    float d[4];
    d[0] = static_cast<float>(_a[0]);
    d[1] = static_cast<float>(_a[1]);
    d[2] = static_cast<float>(_a[2]);
    d[3] = static_cast<float>(_a[3]);

    __m64 a = load_m64(_a);
    __m128 c = _mm_cvtpi16_ps(a);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvtpi32_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    float d[4];
    d[0] = static_cast<float>(_b[0]);
    d[1] = static_cast<float>(_b[1]);
    d[2] = _a[2];
    d[3] = _a[3];

    __m128 a = load_m128(_a);
    __m64 b = load_m64(_b);
    __m128 c = _mm_cvtpi32_ps(a, b);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvtpi32x2_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    float d[4];
    d[0] = static_cast<float>(_a[0]);
    d[1] = static_cast<float>(_a[1]);
    d[2] = static_cast<float>(_b[0]);
    d[3] = static_cast<float>(_b[1]);

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m128 c = _mm_cvtpi32x2_ps(a, b);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvtpi8_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);

    float d[4];
    d[0] = static_cast<float>(_a[0]);
    d[1] = static_cast<float>(_a[1]);
    d[2] = static_cast<float>(_a[2]);
    d[3] = static_cast<float>(_a[3]);

    __m64 a = load_m64(_a);
    __m128 c = _mm_cvtpi8_ps(a);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvtps_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int16_t rnd[4];

    for (int i = 0; i < 4; i++) {
        if (static_cast<float>(INT16_MAX) <= _a[i] &&
            _a[i] <= static_cast<float>(INT32_MAX)) {
            rnd[i] = INT16_MAX;
        } else if (INT16_MIN < _a[i] && _a[i] < INT16_MAX) {
            switch (iter & 0x3) {
            case 0:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
                rnd[i] = static_cast<int16_t>(bankersRounding(_a[i]));
                break;
            case 1:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
                rnd[i] = static_cast<int16_t>(floorf(_a[i]));
                break;
            case 2:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
                rnd[i] = static_cast<int16_t>(ceilf(_a[i]));
                break;
            case 3:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
                rnd[i] = static_cast<int16_t>(_a[i]);
                break;
            }
        } else {
            rnd[i] = INT16_MIN;
        }
    }

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvtps_pi16(a);
    return VALIDATE_INT16_M64(ret, rnd);
}

result_t test_mm_cvtps_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2] = {};

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d[0] = sse2neon_saturate_cast_int32(bankersRounding(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(bankersRounding(_a[1]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d[0] = sse2neon_saturate_cast_int32(floorf(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(floorf(_a[1]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d[0] = sse2neon_saturate_cast_int32(ceilf(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(ceilf(_a[1]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d[0] = sse2neon_saturate_cast_int32(_a[0]);
        d[1] = sse2neon_saturate_cast_int32(_a[1]);
        break;
    }

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvtps_pi32(a);

    return VALIDATE_INT32_M64(ret, d);
}

result_t test_mm_cvtps_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int8_t rnd[8] = {};

    for (int i = 0; i < 4; i++) {
        if (static_cast<float>(INT8_MAX) <= _a[i] &&
            _a[i] <= static_cast<float>(INT32_MAX)) {
            rnd[i] = INT8_MAX;
        } else if (INT8_MIN < _a[i] && _a[i] < INT8_MAX) {
            switch (iter & 0x3) {
            case 0:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
                rnd[i] = static_cast<int8_t>(bankersRounding(_a[i]));
                break;
            case 1:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
                rnd[i] = static_cast<int8_t>(floorf(_a[i]));
                break;
            case 2:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
                rnd[i] = static_cast<int8_t>(ceilf(_a[i]));
                break;
            case 3:
                _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
                rnd[i] = static_cast<int8_t>(_a[i]);
                break;
            }
        } else {
            rnd[i] = INT8_MIN;
        }
    }

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvtps_pi8(a);
    return VALIDATE_INT8_M64(ret, rnd);
}

result_t test_mm_cvtpu16_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);

    float d[4];
    d[0] = static_cast<float>(_a[0]);
    d[1] = static_cast<float>(_a[1]);
    d[2] = static_cast<float>(_a[2]);
    d[3] = static_cast<float>(_a[3]);

    __m64 a = load_m64(_a);
    __m128 c = _mm_cvtpu16_ps(a);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvtpu8_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);

    float d[4];
    d[0] = static_cast<float>(_a[0]);
    d[1] = static_cast<float>(_a[1]);
    d[2] = static_cast<float>(_a[2]);
    d[3] = static_cast<float>(_a[3]);

    __m64 a = load_m64(_a);
    __m128 c = _mm_cvtpu8_ps(a);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvtsi32_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t b = *impl.mTestIntPointer2;

    float d[4];
    d[0] = static_cast<float>(b);
    d[1] = _a[1];
    d[2] = _a[2];
    d[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_cvtsi32_ss(a, b);

    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_cvtsi64_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const int64_t b = *reinterpret_cast<int64_t *>(impl.mTestIntPointer2);

    float d[4];
    d[0] = static_cast<float>(b);
    d[1] = _a[1];
    d[2] = _a[2];
    d[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_cvtsi64_ss(a, b);

    return VALIDATE_FLOAT_M128(c, d);
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

    int32_t d0 = 0;
    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d0 = sse2neon_saturate_cast_int32(bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d0 = sse2neon_saturate_cast_int32(floorf(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d0 = sse2neon_saturate_cast_int32(ceilf(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d0 = sse2neon_saturate_cast_int32(_a[0]);
        break;
    }

    __m128 a = load_m128(_a);
    int32_t ret = _mm_cvtss_si32(a);

    return ret == d0 ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtss_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    int64_t d0 = 0;
    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d0 = sse2neon_saturate_cast_int64(bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d0 = sse2neon_saturate_cast_int64(floorf(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d0 = sse2neon_saturate_cast_int64(ceilf(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d0 = sse2neon_saturate_cast_int64(_a[0]);
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

    d[0] = sse2neon_saturate_cast_int32(_a[0]);
    d[1] = sse2neon_saturate_cast_int32(_a[1]);

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvtt_ps2pi(a);

    return VALIDATE_INT32_M64(ret, d);
}

result_t test_mm_cvtt_ss2si(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = load_m128(_a);
    int ret = _mm_cvtt_ss2si(a);

    return ret == sse2neon_saturate_cast_int32(_a[0]) ? TEST_SUCCESS
                                                      : TEST_FAIL;
}

result_t test_mm_cvttps_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    d[0] = sse2neon_saturate_cast_int32(_a[0]);
    d[1] = sse2neon_saturate_cast_int32(_a[1]);

    __m128 a = load_m128(_a);
    __m64 ret = _mm_cvttps_pi32(a);

    return VALIDATE_INT32_M64(ret, d);
}

result_t test_mm_cvttss_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = load_m128(_a);
    int ret = _mm_cvttss_si32(a);

    return ret == sse2neon_saturate_cast_int32(_a[0]) ? TEST_SUCCESS
                                                      : TEST_FAIL;
}

result_t test_mm_cvttss_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = load_m128(_a);
    int64_t ret = _mm_cvttss_si64(a);

    return ret == sse2neon_saturate_cast_int64(_a[0]) ? TEST_SUCCESS
                                                      : TEST_FAIL;
}

result_t test_mm_div_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float f[4];
    f[0] = _a[0] / _b[0];
    f[1] = _a[1] / _b[1];
    f[2] = _a[2] / _b[2];
    f[3] = _a[3] / _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_div_ps(a, b);

#if defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM64)
    // The implementation of "_mm_div_ps()" on ARM 32bit doesn't use "DIV"
    // instruction directly, instead it uses "FRECPE" instruction to approximate
    // it. Therefore, the precision is not as small as other architecture
    return validateFloatError(c, f[0], f[1], f[2], f[3], 0.00001f);
#else
    return VALIDATE_FLOAT_M128(c, f);
#endif
}

result_t test_mm_div_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float d[4];
    d[0] = _a[0] / _b[0];
    d[1] = _a[1];
    d[2] = _a[2];
    d[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_div_ss(a, b);

#if defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM64)
    // The implementation of "_mm_div_ps()" on ARM 32bit doesn't use "DIV"
    // instruction directly, instead it uses "FRECPE" instruction to approximate
    // it. Therefore, the precision is not as small as other architecture
    return validateFloatError(c, d[0], d[1], d[2], d[3], 0.00001f);
#else
    return VALIDATE_FLOAT_M128(c, d);
#endif
}

result_t test_mm_extract_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // GCC < 11 had a bug where _mm_extract_pi16 performed sign extension
    // instead of zero extension. Fixed in GCC 11.0. See:
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98495
#if defined(__clang__) || defined(_MSC_VER) || \
    (!defined(__clang__) && defined(__GNUC__) && (__GNUC__ >= 11))
    uint64_t *_a = reinterpret_cast<uint64_t *>(impl.mTestIntPointer1);
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

    ASSERT_RETURN(static_cast<uint64_t>(c) == ((*_a >> (idx * 16)) & 0xFFFF));
    ASSERT_RETURN(0 == (static_cast<uint64_t>(c) & 0xFFFF0000));
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

    // Test rounding modes
    unsigned int roundings[] = {_MM_ROUND_TOWARD_ZERO, _MM_ROUND_DOWN,
                                _MM_ROUND_UP, _MM_ROUND_NEAREST};
    for (size_t i = 0; i < sizeof(roundings) / sizeof(roundings[0]); i++) {
        _mm_setcsr(_mm_getcsr() | roundings[i]);
        if ((_mm_getcsr() & roundings[i]) != roundings[i]) {
            _mm_setcsr(originalCsr);
            return TEST_FAIL;
        }
    }

    // Test flush-to-zero mode via _mm_setcsr/_mm_getcsr
    // Note: On ARM, FPCR bit 24 controls both FZ and DAZ behavior. When either
    // is set, both will be reported as set. We must clear both masks together
    // to ensure bit 24 is truly cleared.
    const unsigned int fz_daz_mask =
        _MM_FLUSH_ZERO_MASK | _MM_DENORMALS_ZERO_MASK;
    unsigned int baseCsr = originalCsr & ~fz_daz_mask;

    _mm_setcsr(baseCsr | _MM_FLUSH_ZERO_ON);
    if ((_mm_getcsr() & _MM_FLUSH_ZERO_MASK) != _MM_FLUSH_ZERO_ON) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }
    _mm_setcsr(baseCsr);  // Clear both FZ and DAZ
    if ((_mm_getcsr() & _MM_FLUSH_ZERO_MASK) != _MM_FLUSH_ZERO_OFF) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test denormals-are-zero mode via _mm_setcsr/_mm_getcsr
    _mm_setcsr(baseCsr | _MM_DENORMALS_ZERO_ON);
    if ((_mm_getcsr() & _MM_DENORMALS_ZERO_MASK) != _MM_DENORMALS_ZERO_ON) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }
    _mm_setcsr(baseCsr);  // Clear both FZ and DAZ
    if ((_mm_getcsr() & _MM_DENORMALS_ZERO_MASK) != _MM_DENORMALS_ZERO_OFF) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test MXCSR exception flag macro values (API compatibility)
    // These values must match Intel spec on all platforms.
    if (_MM_EXCEPT_INVALID != 0x0001 || _MM_EXCEPT_DENORM != 0x0002 ||
        _MM_EXCEPT_DIV_ZERO != 0x0004 || _MM_EXCEPT_OVERFLOW != 0x0008 ||
        _MM_EXCEPT_UNDERFLOW != 0x0010 || _MM_EXCEPT_INEXACT != 0x0020) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }
    if (_MM_EXCEPT_MASK != 0x003F) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test MXCSR exception mask macro values (API compatibility)
    if (_MM_MASK_INVALID != 0x0080 || _MM_MASK_DENORM != 0x0100 ||
        _MM_MASK_DIV_ZERO != 0x0200 || _MM_MASK_OVERFLOW != 0x0400 ||
        _MM_MASK_UNDERFLOW != 0x0800 || _MM_MASK_INEXACT != 0x1000) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }
    if (_MM_MASK_MASK != 0x1F80) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || \
    defined(_M_ARM64EC)
    // ARM-specific tests: Exception flags and masks are NOT emulated on ARM.
    // See "MXCSR Exception Flags - NOT EMULATED" documentation in sse2neon.h.

    // Test exception state accessor macros (ARM: always returns 0)
    // On x86, this would return actual exception flags set by FP operations.
    unsigned int excState = _MM_GET_EXCEPTION_STATE();
    if (excState != 0) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test that _MM_SET_EXCEPTION_STATE is a no-op (state should still be 0)
    _MM_SET_EXCEPTION_STATE(_MM_EXCEPT_INVALID | _MM_EXCEPT_OVERFLOW);
    excState = _MM_GET_EXCEPTION_STATE();
    if (excState != 0) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test exception mask accessor macros (ARM: always returns _MM_MASK_MASK)
    // On x86, this reflects which exceptions are masked (defaults to all
    // masked).
    unsigned int excMask = _MM_GET_EXCEPTION_MASK();
    if (excMask != _MM_MASK_MASK) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test that _MM_SET_EXCEPTION_MASK is a no-op (mask should still be full)
    _MM_SET_EXCEPTION_MASK(0);  // Try to unmask all exceptions
    excMask = _MM_GET_EXCEPTION_MASK();
    if (excMask != _MM_MASK_MASK) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }
#endif  // ARM platform

    // restore original csr value for remaining tests
    _mm_setcsr(originalCsr);

    return TEST_SUCCESS;
}

result_t test_mm_insert_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t insert = static_cast<int16_t>(impl.mTestInts[iter]);
    __m64 a;
    __m64 b;

#define TEST_IMPL(IDX)                   \
    int16_t d##IDX[4];                   \
    for (int i = 0; i < 4; i++) {        \
        d##IDX[i] = _a[i];               \
    }                                    \
    d##IDX[IDX] = insert;                \
                                         \
    a = load_m64(_a);                    \
    b = _mm_insert_pi16(a, insert, IDX); \
    CHECK_RESULT(VALIDATE_INT16_M64(b, d##IDX))

    IMM_4_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_load_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_load_ps(addr);

    return VALIDATE_FLOAT_M128(ret, addr);
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
    const __m64 *b = reinterpret_cast<const __m64 *>(p2);
    __m128 a = _mm_load_ps(p1);
    __m128 c = _mm_loadh_pi(a, b);

    return validateFloat(c, p1[0], p1[1], p2[0], p2[1]);
}

result_t test_mm_loadl_pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *p1 = impl.mTestFloatPointer1;
    const float *p2 = impl.mTestFloatPointer2;
    __m128 a = _mm_load_ps(p1);
    const __m64 *b = reinterpret_cast<const __m64 *>(p2);
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

    return VALIDATE_FLOAT_M128(ret, addr);
}

result_t test_mm_loadu_si16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_loadu_si16. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
    return TEST_UNIMPL;
#else
    const unaligned_int16_t *addr = reinterpret_cast<const unaligned_int16_t *>(
        impl.mTestUnalignedInts + 1);

    __m128i ret = _mm_loadu_si16(static_cast<const void *>(addr));

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
    const unaligned_int64_t *addr = reinterpret_cast<const unaligned_int64_t *>(
        impl.mTestUnalignedInts + 1);

    __m128i ret = _mm_loadu_si64(static_cast<const void *>(addr));

    int64_t d[2] = {addr[0], 0};
    return VALIDATE_INT64_M128(ret, d);
#endif
}

result_t test_mm_malloc(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const size_t *a = reinterpret_cast<const size_t *>(impl.mTestIntPointer1);
    const size_t *b = reinterpret_cast<const size_t *>(impl.mTestIntPointer2);
    size_t size = *a % (1024 * 16) + 1;
    size_t align = static_cast<size_t>(2) << (*b % 5);

    void *p = _mm_malloc(size, align);
    if (!p)
        return TEST_FAIL;
    result_t res = ((reinterpret_cast<uintptr_t>(p) % align) == 0)
                       ? TEST_SUCCESS
                       : TEST_FAIL;
    _mm_free(p);
    return res;
}

result_t test_mm_maskmove_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_mask =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    char mem_addr[16];

    const __m64 *a = reinterpret_cast<const __m64 *>(_a);
    const __m64 *mask = reinterpret_cast<const __m64 *>(_mask);
    _mm_maskmove_si64(*a, *mask, static_cast<char *>(mem_addr));

    for (int i = 0; i < 8; i++) {
        if (_mask[i] >> 7) {
            ASSERT_RETURN(_a[i] == static_cast<uint8_t>(mem_addr[i]));
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
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t c[4];

    c[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] > _b[3] ? _a[3] : _b[3];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_max_pi16(a, b);
    return VALIDATE_INT16_M64(ret, c);
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
    return VALIDATE_FLOAT_M128(ret, c);
}

result_t test_mm_max_pu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
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
    return VALIDATE_UINT8_M64(ret, c);
}

result_t test_mm_max_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f[4];
    f[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    f[1] = _a[1];
    f[2] = _a[2];
    f[3] = _a[3];

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_max_ss(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_min_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t c[4];

    c[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] < _b[3] ? _a[3] : _b[3];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_min_pi16(a, b);
    return VALIDATE_INT16_M64(ret, c);
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
    return VALIDATE_FLOAT_M128(ret, c);
}

result_t test_mm_min_pu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
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
    return VALIDATE_UINT8_M64(ret, c);
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
    result[0] = _b[0];
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_move_ss(a, b);
    return VALIDATE_FLOAT_M128(ret, result);
}

result_t test_mm_movehl_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f[4];
    f[0] = _b[2];
    f[1] = _b[3];
    f[2] = _a[2];
    f[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 ret = _mm_movehl_ps(a, b);

    return VALIDATE_FLOAT_M128(ret, f);
}

result_t test_mm_movelh_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f[4];
    f[0] = _a[0];
    f[1] = _a[1];
    f[2] = _b[0];
    f[3] = _b[1];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 ret = _mm_movelh_ps(a, b);

    return VALIDATE_FLOAT_M128(ret, f);
}

result_t test_mm_movemask_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // Test 1: Random data from test framework
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    unsigned int _c = 0;
    for (int i = 0; i < 8; i++) {
        if (_a[i] & 0x80) {
            _c |= (1 << i);
        }
    }

    const __m64 *a = reinterpret_cast<const __m64 *>(_a);
    int c = _mm_movemask_pi8(*a);
    ASSERT_RETURN(static_cast<unsigned int>(c) == _c);

    // Test 2: Edge cases for 64-bit movemask
    // Use padding to ensure 8-byte alignment for each element
    struct {
        uint8_t bytes[8];
        int expected;
        int _pad;  // Padding to make struct size 16 (multiple of 8)
    } edge_cases[] = {
        // All zeros
        {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 0x00, 0},
        // All ones
        {{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 0xFF, 0},
        // Single bit positions
        {{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 0x01, 0},
        {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80}, 0x80, 0},
        // Alternating patterns
        {{0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00}, 0x55, 0},
        {{0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80}, 0xAA, 0},
        // Low/high nibbles
        {{0x80, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00}, 0x0F, 0},
        {{0x00, 0x00, 0x00, 0x00, 0x80, 0x80, 0x80, 0x80}, 0xF0, 0},
        // Boundary values
        {{0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F}, 0x00, 0},
        {{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}, 0xFF, 0},
    };

    for (size_t i = 0; i < sizeof(edge_cases) / sizeof(edge_cases[0]); i++) {
        __m64 vec = *reinterpret_cast<const __m64 *>(edge_cases[i].bytes);
        int result = _mm_movemask_pi8(vec);
        ASSERT_RETURN(result == edge_cases[i].expected);
    }

    return TEST_SUCCESS;
}

result_t test_mm_movemask_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // Test 1: Random data from test framework
    const float *p = impl.mTestFloatPointer1;
    int ret = 0;

    const uint32_t *ip = reinterpret_cast<const uint32_t *>(p);
    if (ip[0] & 0x80000000)
        ret |= 1;
    if (ip[1] & 0x80000000)
        ret |= 2;
    if (ip[2] & 0x80000000)
        ret |= 4;
    if (ip[3] & 0x80000000)
        ret |= 8;
    __m128 a = load_m128(p);
    int val = _mm_movemask_ps(a);
    ASSERT_RETURN(val == ret);

    // Test 2: Edge cases with specific float patterns
    // Using union to set exact bit patterns
    union {
        float f;
        uint32_t u;
    } conv;

    struct {
        uint32_t bits[4];
        int expected;
    } edge_cases[] = {
        // All positive zeros
        {{0x00000000, 0x00000000, 0x00000000, 0x00000000}, 0x0},
        // All negative zeros
        {{0x80000000, 0x80000000, 0x80000000, 0x80000000}, 0xF},
        // Mixed: negative zero at position 0
        {{0x80000000, 0x00000000, 0x00000000, 0x00000000}, 0x1},
        // Mixed: negative zero at position 3
        {{0x00000000, 0x00000000, 0x00000000, 0x80000000}, 0x8},
        // Alternating pattern
        {{0x80000000, 0x00000000, 0x80000000, 0x00000000}, 0x5},
        {{0x00000000, 0x80000000, 0x00000000, 0x80000000}, 0xA},
        // Positive infinity (MSB=0)
        {{0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000}, 0x0},
        // Negative infinity (MSB=1)
        {{0xFF800000, 0xFF800000, 0xFF800000, 0xFF800000}, 0xF},
        // Positive NaN (MSB=0)
        {{0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000}, 0x0},
        // Negative NaN (MSB=1)
        {{0xFFC00000, 0xFFC00000, 0xFFC00000, 0xFFC00000}, 0xF},
        // Max positive float
        {{0x7F7FFFFF, 0x7F7FFFFF, 0x7F7FFFFF, 0x7F7FFFFF}, 0x0},
        // Max negative float
        {{0xFF7FFFFF, 0xFF7FFFFF, 0xFF7FFFFF, 0xFF7FFFFF}, 0xF},
        // Smallest positive denormal
        {{0x00000001, 0x00000001, 0x00000001, 0x00000001}, 0x0},
        // Smallest negative denormal
        {{0x80000001, 0x80000001, 0x80000001, 0x80000001}, 0xF},
    };

    for (size_t i = 0; i < sizeof(edge_cases) / sizeof(edge_cases[0]); i++) {
        float floats[4];
        for (int j = 0; j < 4; j++) {
            conv.u = edge_cases[i].bits[j];
            floats[j] = conv.f;
        }
        __m128 vec = _mm_loadu_ps(floats);
        int result = _mm_movemask_ps(vec);
        ASSERT_RETURN(result == edge_cases[i].expected);
    }

    return TEST_SUCCESS;
}

result_t test_mm_mul_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float d[4];
    d[0] = _a[0] * _b[0];
    d[1] = _a[1] * _b[1];
    d[2] = _a[2] * _b[2];
    d[3] = _a[3] * _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_mul_ps(a, b);
    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_mul_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float d[4];
    d[0] = _a[0] * _b[0];
    d[1] = _a[1];
    d[2] = _a[2];
    d[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_mul_ss(a, b);
    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_mulhi_pu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);
    const uint16_t *_b =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer2);
    uint16_t d[4];
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t m =
            static_cast<uint32_t>(_a[i]) * static_cast<uint32_t>(_b[i]);
        d[i] = static_cast<uint16_t>(m >> 16);
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_mulhi_pu16(a, b);
    return VALIDATE_UINT16_M64(c, d);
}

result_t test_mm_or_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_or_ps(a, b);
    // now for the assertion...
    const uint32_t *ia = reinterpret_cast<const uint32_t *>(&a);
    const uint32_t *ib = reinterpret_cast<const uint32_t *>(&b);
    uint32_t r[4];
    r[0] = ia[0] | ib[0];
    r[1] = ia[1] | ib[1];
    r[2] = ia[2] | ib[2];
    r[3] = ia[3] | ib[3];
    __m128i ret =
        do_mm_set_epi32(static_cast<int32_t>(r[3]), static_cast<int32_t>(r[2]),
                        static_cast<int32_t>(r[1]), static_cast<int32_t>(r[0]));
    result_t res = VALIDATE_UINT32_M128(_mm_castps_si128(c), r);
    if (res) {
        res = VALIDATE_UINT32_M128(ret, r);
    }

    return res;
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
        _mm_prefetch((reinterpret_cast<const char *>(&test_vec[i].a)),
                     _MM_HINT_T0);
        _mm_prefetch((reinterpret_cast<const char *>(&test_vec[i].a)),
                     _MM_HINT_T1);
        _mm_prefetch((reinterpret_cast<const char *>(&test_vec[i].a)),
                     _MM_HINT_T2);
        _mm_prefetch((reinterpret_cast<const char *>(&test_vec[i].a)),
                     _MM_HINT_NTA);
    }

    return TEST_SUCCESS;
}

result_t test_m_psadbw(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    uint16_t d = 0;
    for (int i = 0; i < 8; i++) {
        d += static_cast<uint16_t>(abs(_a[i] - _b[i]));
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
    return validateFloatError(c, dx, dy, dz, dw, 0.001f);
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
    return validateFloatError(c, dx, dy, dz, dw, 0.001f);
}

result_t test_mm_rsqrt_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = reinterpret_cast<const float *>(impl.mTestFloatPointer1);

    float f0 = 1 / sqrtf(_a[0]);
    float f1 = 1 / sqrtf(_a[1]);
    float f2 = 1 / sqrtf(_a[2]);
    float f3 = 1 / sqrtf(_a[3]);

    __m128 a = load_m128(_a);
    __m128 c = _mm_rsqrt_ps(a);

    // Here, we ensure the error rate of "_mm_rsqrt_ps()" is under 0.1% compared
    // to the C implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.001f);
}

result_t test_mm_rsqrt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = reinterpret_cast<const float *>(impl.mTestFloatPointer1);

    float f0 = 1 / sqrtf(_a[0]);
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_rsqrt_ss(a);

    // Here, we ensure the error rate of "_mm_rsqrt_ps()" is under 0.1% compared
    // to the C implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.001f);
}

result_t test_mm_sad_pu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    uint16_t d = 0;
    for (int i = 0; i < 8; i++) {
        d += static_cast<uint16_t>(abs(_a[i] - _b[i]));
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_sad_pu8(a, b);
    return validateUInt16(c, d, 0, 0, 0);
}

OPTNONE result_t test_mm_set_flush_zero_mode(const SSE2NEONTestImpl &impl,
                                             uint32_t iter)
{
    // FTZ (Flush-to-Zero) affects OUTPUT denormals.
    // FLT_MIN * 0.5 produces a denormal result which should be flushed to 0
    // when FTZ is enabled.
    unsigned int originalCsr = _mm_getcsr();
    result_t res_flush_zero_on, res_flush_zero_off;
    float min_normals[4] = {FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN};
    float halves[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    __m128 ret;

    // On ARM, FPCR bit 24 controls both FZ and DAZ. Use atomic mask operations
    // to set/clear both flags together, matching test_mm_setcsr style.
    const unsigned int fz_daz_mask =
        _MM_FLUSH_ZERO_MASK | _MM_DENORMALS_ZERO_MASK;

    // Test FTZ ON: denormal output should be flushed to 0
    _mm_setcsr((originalCsr & ~fz_daz_mask) | _MM_FLUSH_ZERO_ON);
    ret = _mm_mul_ps(load_m128(min_normals), load_m128(halves));
    res_flush_zero_on = validateFloat(ret, 0, 0, 0, 0);
    if (res_flush_zero_on == TEST_FAIL) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test FTZ OFF: denormal output should be preserved
    _mm_setcsr(originalCsr & ~fz_daz_mask);
    ret = _mm_mul_ps(load_m128(min_normals), load_m128(halves));
#if defined(__arm__)
    // AArch32 Advanced SIMD arithmetic always uses the Flush-to-zero setting,
    // regardless of the value of the FZ bit.
    res_flush_zero_off = validateFloat(ret, 0, 0, 0, 0);
#else
    // Build expected denormal via bit pattern to avoid compiler
    // constant-folding FLT_MIN = 2^-126 = 0x00800000, so FLT_MIN/2 = 2^-127 =
    // 0x00400000
    float expected;
    uint32_t expected_bits = 0x00400000;
    memcpy(&expected, &expected_bits, sizeof(expected));
    res_flush_zero_off =
        validateFloat(ret, expected, expected, expected, expected);
#endif

    _mm_setcsr(originalCsr);
    if (res_flush_zero_off == TEST_FAIL)
        return TEST_FAIL;
    return TEST_SUCCESS;
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

OPTNONE result_t test_mm_set_rounding_mode(const SSE2NEONTestImpl &impl,
                                           uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    result_t res_toward_zero, res_to_neg_inf, res_to_pos_inf, res_nearest;

    __m128 a = load_m128(_a);
    __m128 b = _mm_setzero_ps(), c = _mm_setzero_ps();

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

OPTNONE result_t test_mm_setcsr(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // Test rounding modes
    if (test_mm_set_rounding_mode(impl, iter) != TEST_SUCCESS)
        return TEST_FAIL;

    // Test FTZ/DAZ behavior with denormal arithmetic
    unsigned int originalCsr = _mm_getcsr();
    __m128 ret;

    // Create denormal via bit pattern to avoid FTZ-dependent computation.
    // FLT_MIN = 2^-126 = 0x00800000, so FLT_MIN/2 = 2^-127 = 0x00400000
    float denormal;
    uint32_t denormal_bits = 0x00400000;
    memcpy(&denormal, &denormal_bits, sizeof(denormal));

    // Test flush-to-zero: FTZ affects OUTPUT denormals only.
    // FLT_MIN * 0.5 = denormal output, which should be flushed to 0.
    const unsigned int fz_daz_mask =
        _MM_FLUSH_ZERO_MASK | _MM_DENORMALS_ZERO_MASK;
    float min_normals[4] = {FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN};
    float halves[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    _mm_setcsr((originalCsr & ~fz_daz_mask) | _MM_FLUSH_ZERO_ON);
    ret = _mm_mul_ps(load_m128(min_normals), load_m128(halves));
    if (validateFloat(ret, 0, 0, 0, 0) != TEST_SUCCESS) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test denormals-are-zero: DAZ affects INPUT denormals.
    // denormal * 2 with DAZ: input treated as 0, so 0 * 2 = 0.
    float denormals[4] = {denormal, denormal, denormal, denormal};
    float factors[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    _mm_setcsr((originalCsr & ~fz_daz_mask) | _MM_DENORMALS_ZERO_ON);
    ret = _mm_mul_ps(load_m128(denormals), load_m128(factors));
    if (validateFloat(ret, 0, 0, 0, 0) != TEST_SUCCESS) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }

    // Test with FTZ/DAZ off: denormal * 2 = FLT_MIN (normal number)
    _mm_setcsr(originalCsr & ~fz_daz_mask);
    ret = _mm_mul_ps(load_m128(denormals), load_m128(factors));
#if defined(__arm__)
    // AArch32 Advanced SIMD always uses Flush-to-zero regardless of FZ bit
    if (validateFloat(ret, 0, 0, 0, 0) != TEST_SUCCESS) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }
#else
    if (validateFloat(ret, FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN) !=
        TEST_SUCCESS) {
        _mm_setcsr(originalCsr);
        return TEST_FAIL;
    }
#endif

    _mm_setcsr(originalCsr);
    return TEST_SUCCESS;
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
    /* FIXME: Assume that memory barriers always function as intended. */
    return TEST_SUCCESS;
}

result_t test_mm_shuffle_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if (__GNUC__ == 8) || (__GNUC__ == 9 && __GNUC_MINOR__ == 2)
#error Using older gcc versions can lead to an operand mismatch error. This issue affects all versions prior to gcc 10.
#else
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    __m64 a;
    __m64 d;
    int16_t _d[4];
#define TEST_IMPL(IDX)                               \
    a = load_m64(_a);                                \
    d = _mm_shuffle_pi16(a, IDX);                    \
                                                     \
    _d[0] = _a[IDX & 0x3];                           \
    _d[1] = _a[(IDX >> 2) & 0x3];                    \
    _d[2] = _a[(IDX >> 4) & 0x3];                    \
    _d[3] = _a[(IDX >> 6) & 0x3];                    \
    if (VALIDATE_INT16_M64(d, _d) != TEST_SUCCESS) { \
        return TEST_FAIL;                            \
    }

    IMM_256_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
#endif
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
    const float *_a = reinterpret_cast<const float *>(impl.mTestFloatPointer1);

    float f0 = sqrtf(_a[0]);
    float f1 = sqrtf(_a[1]);
    float f2 = sqrtf(_a[2]);
    float f3 = sqrtf(_a[3]);

    __m128 a = load_m128(_a);
    __m128 c = _mm_sqrt_ps(a);

#if defined(__arm__) && !defined(__arm64__) && !defined(_M_ARM64)
    // Here, we ensure the error rate of "_mm_sqrt_ps()" ARMv7-A implementation
    // is under 10^-4% compared to the C implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.0001f);
#else
    // Here, we ensure the error rate of "_mm_sqrt_ps()" is under 10^-6%
    // compared to the C implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.000001f);
#endif
}

result_t test_mm_sqrt_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = reinterpret_cast<const float *>(impl.mTestFloatPointer1);

    float f0 = sqrtf(_a[0]);
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = load_m128(_a);
    __m128 c = _mm_sqrt_ss(a);

#if defined(__arm__) && !defined(__arm64__) && !defined(_M_ARM64)
    // Here, we ensure the error rate of "_mm_sqrt_ps()" ARMv7-A implementation
    // is under 10^-4% compared to the C implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.0001f);
#else
    // Here, we ensure the error rate of "_mm_sqrt_ps()" is under 10^-6%
    // compared to the C implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.000001f);
#endif
}

result_t test_mm_store_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t *p = impl.mTestIntPointer1;
    int32_t x = impl.mTestInts[iter];
    int32_t y = impl.mTestInts[iter + 1];
    int32_t z = impl.mTestInts[iter + 2];
    int32_t w = impl.mTestInts[iter + 3];
    __m128i a = _mm_set_epi32(x, y, z, w);
    _mm_store_ps(reinterpret_cast<float *>(p), _mm_castsi128_ps(a));
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
    __m64 *b = reinterpret_cast<__m64 *>(d);

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
    __m64 *b = reinterpret_cast<__m64 *>(d);

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
    return VALIDATE_FLOAT_M128(a, f);
}

result_t test_mm_storeu_si16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_storeu_si16. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
    return TEST_UNIMPL;
#else
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m128i b = _mm_setzero_si128();
    __m128i a = load_m128i(_a);
    _mm_storeu_si16(&b, a);
    int16_t *_b = reinterpret_cast<int16_t *>(&b);
    int16_t *_c = reinterpret_cast<int16_t *>(&a);
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
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m128i b = _mm_setzero_si128();
    __m128i a = load_m128i(_a);
    _mm_storeu_si64(&b, a);
    int64_t *_b = reinterpret_cast<int64_t *>(&b);
    int64_t *_c = reinterpret_cast<int64_t *>(&a);
    int64_t d[2] = {_c[0], _b[1]};
    return VALIDATE_INT64_M128(b, d);
#endif
}

result_t test_mm_stream_pi(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
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
    float d[4];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1] - _b[1];
    d[2] = _a[2] - _b[2];
    d[3] = _a[3] - _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_sub_ps(a, b);
    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_sub_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float d[4];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1];
    d[2] = _a[2];
    d[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_sub_ss(a, b);
    return VALIDATE_FLOAT_M128(c, d);
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

    // When SSE2NEON_UNDEFINED_ZERO is set or on MSVC, the value should be zero.
    // On GCC/Clang without the flag, the value is truly undefined so we can
    // only test that XOR-with-self produces zero.
#if SSE2NEON_UNDEFINED_ZERO || (defined(_MSC_VER) && !defined(__clang__))
    result_t res = validateFloat(a, 0, 0, 0, 0);
    if (res != TEST_SUCCESS)
        return res;
#else
    // Use volatile barrier to prevent compiler from optimizing away the
    // undefined value. This forces materialization and avoids UB from
    // reading uninitialized memory directly.
    volatile __m128 barrier = a;
    a = barrier;
#endif

    a = _mm_xor_ps(a, a);
    return validateFloat(a, 0, 0, 0, 0);
}

result_t test_mm_unpackhi_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float *_a = impl.mTestFloatPointer1;
    float *_b = impl.mTestFloatPointer1;

    float f[4];
    f[0] = _a[2];
    f[1] = _b[2];
    f[2] = _a[3];
    f[3] = _b[3];

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_unpackhi_ps(a, b);
    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_unpacklo_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    float *_a = impl.mTestFloatPointer1;
    float *_b = impl.mTestFloatPointer1;

    float f[4];
    f[0] = _a[0];
    f[1] = _b[0];
    f[2] = _a[1];
    f[3] = _b[1];

    __m128 a = _mm_load_ps(_a);
    __m128 b = _mm_load_ps(_b);
    __m128 c = _mm_unpacklo_ps(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_xor_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestFloatPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestFloatPointer2);
    float d[4];
    d[0] = sse2neon_tool_recast_f32(_a[0] ^ _b[0]);
    d[1] = sse2neon_tool_recast_f32(_a[1] ^ _b[1]);
    d[2] = sse2neon_tool_recast_f32(_a[2] ^ _b[2]);
    d[3] = sse2neon_tool_recast_f32(_a[3] ^ _b[3]);

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_xor_ps(a, b);

    return VALIDATE_FLOAT_M128(c, d);
}

/* SSE2 */
result_t test_mm_add_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    int16_t d[8];
    d[0] = _a[0] + _b[0];
    d[1] = _a[1] + _b[1];
    d[2] = _a[2] + _b[2];
    d[3] = _a[3] + _b[3];
    d[4] = _a[4] + _b[4];
    d[5] = _a[5] + _b[5];
    d[6] = _a[6] + _b[6];
    d[7] = _a[7] + _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_add_epi16(a, b);

    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_add_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t d[4];
    d[0] = _a[0] + _b[0];
    d[1] = _a[1] + _b[1];
    d[2] = _a[2] + _b[2];
    d[3] = _a[3] + _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_add_epi32(a, b);
    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_add_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer2);

    int64_t d[2];
    d[0] = _a[0] + _b[0];
    d[1] = _a[1] + _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_add_epi64(a, b);

    return VALIDATE_INT64_M128(c, d);
}

result_t test_mm_add_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    int8_t d[16];
    d[0] = _a[0] + _b[0];
    d[1] = _a[1] + _b[1];
    d[2] = _a[2] + _b[2];
    d[3] = _a[3] + _b[3];
    d[4] = _a[4] + _b[4];
    d[5] = _a[5] + _b[5];
    d[6] = _a[6] + _b[6];
    d[7] = _a[7] + _b[7];
    d[8] = _a[8] + _b[8];
    d[9] = _a[9] + _b[9];
    d[10] = _a[10] + _b[10];
    d[11] = _a[11] + _b[11];
    d[12] = _a[12] + _b[12];
    d[13] = _a[13] + _b[13];
    d[14] = _a[14] + _b[14];
    d[15] = _a[15] + _b[15];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_add_epi8(a, b);
    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_add_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] + _b[0];
    d[1] = _a[1] + _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_add_pd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_add_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] + _b[0];
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_add_sd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_add_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer2);

    int64_t d0 = _a[0] + _b[0];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_add_si64(a, b);

    return validateInt64(c, d0);
}

result_t test_mm_adds_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] =
        saturate_i16(static_cast<int32_t>(_a[0]) + static_cast<int32_t>(_b[0]));
    d[1] =
        saturate_i16(static_cast<int32_t>(_a[1]) + static_cast<int32_t>(_b[1]));
    d[2] =
        saturate_i16(static_cast<int32_t>(_a[2]) + static_cast<int32_t>(_b[2]));
    d[3] =
        saturate_i16(static_cast<int32_t>(_a[3]) + static_cast<int32_t>(_b[3]));
    d[4] =
        saturate_i16(static_cast<int32_t>(_a[4]) + static_cast<int32_t>(_b[4]));
    d[5] =
        saturate_i16(static_cast<int32_t>(_a[5]) + static_cast<int32_t>(_b[5]));
    d[6] =
        saturate_i16(static_cast<int32_t>(_a[6]) + static_cast<int32_t>(_b[6]));
    d[7] =
        saturate_i16(static_cast<int32_t>(_a[7]) + static_cast<int32_t>(_b[7]));

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_adds_epi16(a, b);
    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_adds_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        d[i] = saturate_i8(static_cast<int16_t>(_a[i]) +
                           static_cast<int16_t>(_b[i]));
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_adds_epi8(a, b);

    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_adds_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);
    const uint16_t *_b =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer2);

    uint16_t d[8];
    d[0] = saturate_u16(static_cast<uint32_t>(_a[0]) +
                        static_cast<uint32_t>(_b[0]));
    d[1] = saturate_u16(static_cast<uint32_t>(_a[1]) +
                        static_cast<uint32_t>(_b[1]));
    d[2] = saturate_u16(static_cast<uint32_t>(_a[2]) +
                        static_cast<uint32_t>(_b[2]));
    d[3] = saturate_u16(static_cast<uint32_t>(_a[3]) +
                        static_cast<uint32_t>(_b[3]));
    d[4] = saturate_u16(static_cast<uint32_t>(_a[4]) +
                        static_cast<uint32_t>(_b[4]));
    d[5] = saturate_u16(static_cast<uint32_t>(_a[5]) +
                        static_cast<uint32_t>(_b[5]));
    d[6] = saturate_u16(static_cast<uint32_t>(_a[6]) +
                        static_cast<uint32_t>(_b[6]));
    d[7] = saturate_u16(static_cast<uint32_t>(_a[7]) +
                        static_cast<uint32_t>(_b[7]));

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_adds_epu16(a, b);

    return VALIDATE_UINT16_M128(c, d);
}

result_t test_mm_adds_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    uint8_t d[16];
    d[0] = static_cast<uint8_t>(_a[0]) + static_cast<uint8_t>(_b[0]);
    if (d[0] < static_cast<uint8_t>(_a[0]))
        d[0] = 255;
    d[1] = static_cast<uint8_t>(_a[1]) + static_cast<uint8_t>(_b[1]);
    if (d[1] < static_cast<uint8_t>(_a[1]))
        d[1] = 255;
    d[2] = static_cast<uint8_t>(_a[2]) + static_cast<uint8_t>(_b[2]);
    if (d[2] < static_cast<uint8_t>(_a[2]))
        d[2] = 255;
    d[3] = static_cast<uint8_t>(_a[3]) + static_cast<uint8_t>(_b[3]);
    if (d[3] < static_cast<uint8_t>(_a[3]))
        d[3] = 255;
    d[4] = static_cast<uint8_t>(_a[4]) + static_cast<uint8_t>(_b[4]);
    if (d[4] < static_cast<uint8_t>(_a[4]))
        d[4] = 255;
    d[5] = static_cast<uint8_t>(_a[5]) + static_cast<uint8_t>(_b[5]);
    if (d[5] < static_cast<uint8_t>(_a[5]))
        d[5] = 255;
    d[6] = static_cast<uint8_t>(_a[6]) + static_cast<uint8_t>(_b[6]);
    if (d[6] < static_cast<uint8_t>(_a[6]))
        d[6] = 255;
    d[7] = static_cast<uint8_t>(_a[7]) + static_cast<uint8_t>(_b[7]);
    if (d[7] < static_cast<uint8_t>(_a[7]))
        d[7] = 255;
    d[8] = static_cast<uint8_t>(_a[8]) + static_cast<uint8_t>(_b[8]);
    if (d[8] < static_cast<uint8_t>(_a[8]))
        d[8] = 255;
    d[9] = static_cast<uint8_t>(_a[9]) + static_cast<uint8_t>(_b[9]);
    if (d[9] < static_cast<uint8_t>(_a[9]))
        d[9] = 255;
    d[10] = static_cast<uint8_t>(_a[10]) + static_cast<uint8_t>(_b[10]);
    if (d[10] < static_cast<uint8_t>(_a[10]))
        d[10] = 255;
    d[11] = static_cast<uint8_t>(_a[11]) + static_cast<uint8_t>(_b[11]);
    if (d[11] < static_cast<uint8_t>(_a[11]))
        d[11] = 255;
    d[12] = static_cast<uint8_t>(_a[12]) + static_cast<uint8_t>(_b[12]);
    if (d[12] < static_cast<uint8_t>(_a[12]))
        d[12] = 255;
    d[13] = static_cast<uint8_t>(_a[13]) + static_cast<uint8_t>(_b[13]);
    if (d[13] < static_cast<uint8_t>(_a[13]))
        d[13] = 255;
    d[14] = static_cast<uint8_t>(_a[14]) + static_cast<uint8_t>(_b[14]);
    if (d[14] < static_cast<uint8_t>(_a[14]))
        d[14] = 255;
    d[15] = static_cast<uint8_t>(_a[15]) + static_cast<uint8_t>(_b[15]);
    if (d[15] < static_cast<uint8_t>(_a[15]))
        d[15] = 255;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_adds_epu8(a, b);
    return VALIDATE_UINT8_M128(c, d);
}

result_t test_mm_and_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestFloatPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = sse2neon_tool_recast_f64(static_cast<uint64_t>(_a[0] & _b[0]));
    d[1] = sse2neon_tool_recast_f64(static_cast<uint64_t>(_a[1] & _b[1]));

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_and_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_and_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128 fc = _mm_and_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b));
    __m128i c = _mm_castps_si128(fc);
    // now for the assertion...
    const uint32_t *ia = reinterpret_cast<const uint32_t *>(&a);
    const uint32_t *ib = reinterpret_cast<const uint32_t *>(&b);
    uint32_t r[4];
    r[0] = ia[0] & ib[0];
    r[1] = ia[1] & ib[1];
    r[2] = ia[2] & ib[2];
    r[3] = ia[3] & ib[3];
    __m128i ret =
        do_mm_set_epi32(static_cast<int32_t>(r[3]), static_cast<int32_t>(r[2]),
                        static_cast<int32_t>(r[1]), static_cast<int32_t>(r[0]));
    result_t res = VALIDATE_UINT32_M128(c, r);
    if (res) {
        res = VALIDATE_UINT32_M128(ret, r);
    }
    return res;
}

result_t test_mm_andnot_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_andnot_pd(a, b);

    // Take AND operation a complement of 'a' and 'b'. Bitwise operations are
    // not allowed on float/double datatype, so 'a' and 'b' are calculated in
    // uint64_t datatype.
    const uint64_t *ia = reinterpret_cast<const uint64_t *>(&a);
    const uint64_t *ib = reinterpret_cast<const uint64_t *>(&b);
    uint64_t d[2];
    d[0] = ~ia[0] & ib[0];
    d[1] = ~ia[1] & ib[1];
    return VALIDATE_UINT64_M128(_mm_castpd_si128(c), d);
}

result_t test_mm_andnot_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128 fc = _mm_andnot_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b));
    __m128i c = _mm_castps_si128(fc);
    // now for the assertion...
    const uint32_t *ia = reinterpret_cast<const uint32_t *>(&a);
    const uint32_t *ib = reinterpret_cast<const uint32_t *>(&b);
    uint32_t r[4];
    r[0] = ~ia[0] & ib[0];
    r[1] = ~ia[1] & ib[1];
    r[2] = ~ia[2] & ib[2];
    r[3] = ~ia[3] & ib[3];
    __m128i ret =
        do_mm_set_epi32(static_cast<int32_t>(r[3]), static_cast<int32_t>(r[2]),
                        static_cast<int32_t>(r[1]), static_cast<int32_t>(r[0]));
    result_t res = TEST_SUCCESS;
    res = VALIDATE_UINT32_M128(c, r);
    if (res) {
        res = VALIDATE_UINT32_M128(ret, r);
    }
    return res;
}

result_t test_mm_avg_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);
    const uint16_t *_b =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer2);
    uint16_t d[8];
    d[0] = static_cast<uint16_t>((_a[0] + _b[0] + 1) >> 1);
    d[1] = static_cast<uint16_t>((_a[1] + _b[1] + 1) >> 1);
    d[2] = static_cast<uint16_t>((_a[2] + _b[2] + 1) >> 1);
    d[3] = static_cast<uint16_t>((_a[3] + _b[3] + 1) >> 1);
    d[4] = static_cast<uint16_t>((_a[4] + _b[4] + 1) >> 1);
    d[5] = static_cast<uint16_t>((_a[5] + _b[5] + 1) >> 1);
    d[6] = static_cast<uint16_t>((_a[6] + _b[6] + 1) >> 1);
    d[7] = static_cast<uint16_t>((_a[7] + _b[7] + 1) >> 1);
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_avg_epu16(a, b);
    return VALIDATE_UINT16_M128(c, d);
}

result_t test_mm_avg_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    uint8_t d[16];
    d[0] = static_cast<uint8_t>((_a[0] + _b[0] + 1) >> 1);
    d[1] = static_cast<uint8_t>((_a[1] + _b[1] + 1) >> 1);
    d[2] = static_cast<uint8_t>((_a[2] + _b[2] + 1) >> 1);
    d[3] = static_cast<uint8_t>((_a[3] + _b[3] + 1) >> 1);
    d[4] = static_cast<uint8_t>((_a[4] + _b[4] + 1) >> 1);
    d[5] = static_cast<uint8_t>((_a[5] + _b[5] + 1) >> 1);
    d[6] = static_cast<uint8_t>((_a[6] + _b[6] + 1) >> 1);
    d[7] = static_cast<uint8_t>((_a[7] + _b[7] + 1) >> 1);
    d[8] = static_cast<uint8_t>((_a[8] + _b[8] + 1) >> 1);
    d[9] = static_cast<uint8_t>((_a[9] + _b[9] + 1) >> 1);
    d[10] = static_cast<uint8_t>((_a[10] + _b[10] + 1) >> 1);
    d[11] = static_cast<uint8_t>((_a[11] + _b[11] + 1) >> 1);
    d[12] = static_cast<uint8_t>((_a[12] + _b[12] + 1) >> 1);
    d[13] = static_cast<uint8_t>((_a[13] + _b[13] + 1) >> 1);
    d[14] = static_cast<uint8_t>((_a[14] + _b[14] + 1) >> 1);
    d[15] = static_cast<uint8_t>((_a[15] + _b[15] + 1) >> 1);
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_avg_epu8(a, b);
    return VALIDATE_UINT8_M128(c, d);
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
    const __m128i *_c = reinterpret_cast<const __m128i *>(_a);

    __m128i r = _mm_castpd_si128(a);

    return validate128(r, *_c);
}

result_t test_mm_castps_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const __m128 a = load_m128(_a);
    const __m128d *_c = reinterpret_cast<const __m128d *>(_a);

    __m128d r = _mm_castps_pd(a);

    return validate128(r, *_c);
}

result_t test_mm_castps_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;

    const __m128i *_c = reinterpret_cast<const __m128i *>(_a);

    const __m128 a = load_m128(_a);
    __m128i r = _mm_castps_si128(a);

    return validate128(r, *_c);
}

result_t test_mm_castsi128_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;

    const __m128d *_c = reinterpret_cast<const __m128d *>(_a);

    const __m128i a = load_m128i(_a);
    __m128d r = _mm_castsi128_pd(a);

    return validate128(r, *_c);
}

result_t test_mm_castsi128_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;

    const __m128 *_c = reinterpret_cast<const __m128 *>(_a);

    const __m128i a = load_m128i(_a);
    __m128 r = _mm_castsi128_ps(a);

    return validate128(r, *_c);
}

result_t test_mm_clflush(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    /* FIXME: Assume that we have portable mechanisms to flush cache. */
    return TEST_SUCCESS;
}

result_t test_mm_cmpeq_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] = (_a[0] == _b[0]) ? ~UINT16_C(0) : 0x0;
    d[1] = (_a[1] == _b[1]) ? ~UINT16_C(0) : 0x0;
    d[2] = (_a[2] == _b[2]) ? ~UINT16_C(0) : 0x0;
    d[3] = (_a[3] == _b[3]) ? ~UINT16_C(0) : 0x0;
    d[4] = (_a[4] == _b[4]) ? ~UINT16_C(0) : 0x0;
    d[5] = (_a[5] == _b[5]) ? ~UINT16_C(0) : 0x0;
    d[6] = (_a[6] == _b[6]) ? ~UINT16_C(0) : 0x0;
    d[7] = (_a[7] == _b[7]) ? ~UINT16_C(0) : 0x0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpeq_epi16(a, b);
    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_cmpeq_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;

    int32_t d[4];
    d[0] = (_a[0] == _b[0]) ? -1 : 0;
    d[1] = (_a[1] == _b[1]) ? -1 : 0;
    d[2] = (_a[2] == _b[2]) ? -1 : 0;
    d[3] = (_a[3] == _b[3]) ? -1 : 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpeq_epi32(a, b);

    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_cmpeq_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    int8_t d[16];
    d[0] = (_a[0] == _b[0]) ? ~UINT8_C(0) : 0x00;
    d[1] = (_a[1] == _b[1]) ? ~UINT8_C(0) : 0x00;
    d[2] = (_a[2] == _b[2]) ? ~UINT8_C(0) : 0x00;
    d[3] = (_a[3] == _b[3]) ? ~UINT8_C(0) : 0x00;
    d[4] = (_a[4] == _b[4]) ? ~UINT8_C(0) : 0x00;
    d[5] = (_a[5] == _b[5]) ? ~UINT8_C(0) : 0x00;
    d[6] = (_a[6] == _b[6]) ? ~UINT8_C(0) : 0x00;
    d[7] = (_a[7] == _b[7]) ? ~UINT8_C(0) : 0x00;
    d[8] = (_a[8] == _b[8]) ? ~UINT8_C(0) : 0x00;
    d[9] = (_a[9] == _b[9]) ? ~UINT8_C(0) : 0x00;
    d[10] = (_a[10] == _b[10]) ? ~UINT8_C(0) : 0x00;
    d[11] = (_a[11] == _b[11]) ? ~UINT8_C(0) : 0x00;
    d[12] = (_a[12] == _b[12]) ? ~UINT8_C(0) : 0x00;
    d[13] = (_a[13] == _b[13]) ? ~UINT8_C(0) : 0x00;
    d[14] = (_a[14] == _b[14]) ? ~UINT8_C(0) : 0x00;
    d[15] = (_a[15] == _b[15]) ? ~UINT8_C(0) : 0x00;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpeq_epi8(a, b);
    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_cmpeq_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] == _b[0]) ? sse2neon_tool_recast_f64(UINT64_MAX) : 0;
    d[1] = (_a[1] == _b[1]) ? sse2neon_tool_recast_f64(UINT64_MAX) : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpeq_pd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpeq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] == _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpeq_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpge_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] >= _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = (_a[1] >= _b[1]) ? ALL_BIT_1_64 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpge_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpge_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] >= _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpge_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpgt_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] = _a[0] > _b[0] ? -1 : 0;
    d[1] = _a[1] > _b[1] ? -1 : 0;
    d[2] = _a[2] > _b[2] ? -1 : 0;
    d[3] = _a[3] > _b[3] ? -1 : 0;
    d[4] = _a[4] > _b[4] ? -1 : 0;
    d[5] = _a[5] > _b[5] ? -1 : 0;
    d[6] = _a[6] > _b[6] ? -1 : 0;
    d[7] = _a[7] > _b[7] ? -1 : 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpgt_epi16(a, b);

    return VALIDATE_INT16_M128(c, d);
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
    return VALIDATE_INT32_M128(iret, result);
}

result_t test_mm_cmpgt_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    int8_t d[16];
    d[0] = (_a[0] > _b[0]) ? ~UINT8_C(0) : 0x00;
    d[1] = (_a[1] > _b[1]) ? ~UINT8_C(0) : 0x00;
    d[2] = (_a[2] > _b[2]) ? ~UINT8_C(0) : 0x00;
    d[3] = (_a[3] > _b[3]) ? ~UINT8_C(0) : 0x00;
    d[4] = (_a[4] > _b[4]) ? ~UINT8_C(0) : 0x00;
    d[5] = (_a[5] > _b[5]) ? ~UINT8_C(0) : 0x00;
    d[6] = (_a[6] > _b[6]) ? ~UINT8_C(0) : 0x00;
    d[7] = (_a[7] > _b[7]) ? ~UINT8_C(0) : 0x00;
    d[8] = (_a[8] > _b[8]) ? ~UINT8_C(0) : 0x00;
    d[9] = (_a[9] > _b[9]) ? ~UINT8_C(0) : 0x00;
    d[10] = (_a[10] > _b[10]) ? ~UINT8_C(0) : 0x00;
    d[11] = (_a[11] > _b[11]) ? ~UINT8_C(0) : 0x00;
    d[12] = (_a[12] > _b[12]) ? ~UINT8_C(0) : 0x00;
    d[13] = (_a[13] > _b[13]) ? ~UINT8_C(0) : 0x00;
    d[14] = (_a[14] > _b[14]) ? ~UINT8_C(0) : 0x00;
    d[15] = (_a[15] > _b[15]) ? ~UINT8_C(0) : 0x00;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpgt_epi8(a, b);
    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_cmpgt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] > _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = (_a[1] > _b[1]) ? ALL_BIT_1_64 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpgt_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpgt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] > _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpgt_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmple_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] <= _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = (_a[1] <= _b[1]) ? ALL_BIT_1_64 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmple_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmple_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] <= _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmple_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmplt_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] = _a[0] < _b[0] ? -1 : 0;
    d[1] = _a[1] < _b[1] ? -1 : 0;
    d[2] = _a[2] < _b[2] ? -1 : 0;
    d[3] = _a[3] < _b[3] ? -1 : 0;
    d[4] = _a[4] < _b[4] ? -1 : 0;
    d[5] = _a[5] < _b[5] ? -1 : 0;
    d[6] = _a[6] < _b[6] ? -1 : 0;
    d[7] = _a[7] < _b[7] ? -1 : 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmplt_epi16(a, b);

    return VALIDATE_INT16_M128(c, d);
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
    return VALIDATE_INT32_M128(iret, result);
}

result_t test_mm_cmplt_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    int8_t d[16];
    d[0] = (_a[0] < _b[0]) ? ~UINT8_C(0) : 0x00;
    d[1] = (_a[1] < _b[1]) ? ~UINT8_C(0) : 0x00;
    d[2] = (_a[2] < _b[2]) ? ~UINT8_C(0) : 0x00;
    d[3] = (_a[3] < _b[3]) ? ~UINT8_C(0) : 0x00;
    d[4] = (_a[4] < _b[4]) ? ~UINT8_C(0) : 0x00;
    d[5] = (_a[5] < _b[5]) ? ~UINT8_C(0) : 0x00;
    d[6] = (_a[6] < _b[6]) ? ~UINT8_C(0) : 0x00;
    d[7] = (_a[7] < _b[7]) ? ~UINT8_C(0) : 0x00;
    d[8] = (_a[8] < _b[8]) ? ~UINT8_C(0) : 0x00;
    d[9] = (_a[9] < _b[9]) ? ~UINT8_C(0) : 0x00;
    d[10] = (_a[10] < _b[10]) ? ~UINT8_C(0) : 0x00;
    d[11] = (_a[11] < _b[11]) ? ~UINT8_C(0) : 0x00;
    d[12] = (_a[12] < _b[12]) ? ~UINT8_C(0) : 0x00;
    d[13] = (_a[13] < _b[13]) ? ~UINT8_C(0) : 0x00;
    d[14] = (_a[14] < _b[14]) ? ~UINT8_C(0) : 0x00;
    d[15] = (_a[15] < _b[15]) ? ~UINT8_C(0) : 0x00;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmplt_epi8(a, b);
    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_cmplt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] < _b[0]) ? ALL_BIT_1_64 : UINT64_C(0);
    d[1] = (_a[1] < _b[1]) ? ALL_BIT_1_64 : UINT64_C(0);

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmplt_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmplt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] < _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmplt_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpneq_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] != _b[0]) ? ALL_BIT_1_64 : UINT64_C(0);
    d[1] = (_a[1] != _b[1]) ? ALL_BIT_1_64 : UINT64_C(0);

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpneq_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpneq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = (_a[0] != _b[0]) ? ALL_BIT_1_64 : UINT64_C(0);
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpneq_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpnge_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = !(_a[0] >= _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = !(_a[1] >= _b[1]) ? ALL_BIT_1_64 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnge_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpnge_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = !(_a[0] >= _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnge_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpngt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = !(_a[0] > _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = !(_a[1] > _b[1]) ? ALL_BIT_1_64 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpngt_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpngt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = !(_a[0] > _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpngt_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpnle_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = !(_a[0] <= _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = !(_a[1] <= _b[1]) ? ALL_BIT_1_64 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnle_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpnle_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = !(_a[0] <= _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnle_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpnlt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = !(_a[0] < _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = !(_a[1] < _b[1]) ? ALL_BIT_1_64 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnlt_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpnlt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = !(_a[0] < _b[0]) ? ALL_BIT_1_64 : 0;
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_cmpnlt_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cmpord_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);

    double d[2];

    for (uint32_t i = 0; i < 2; i++) {
        d[i] = cmp_noNaN(_a[i], _b[i]);
    }

    __m128d ret = _mm_cmpord_pd(a, b);

    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_cmpord_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);

    double d[2];
    d[0] = cmp_noNaN(_a[0], _b[0]);
    d[1] = _a[1];

    __m128d ret = _mm_cmpord_sd(a, b);
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_cmpunord_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);

    double d[2];
    d[0] = cmp_hasNaN(_a[0], _b[0]);
    d[1] = cmp_hasNaN(_a[1], _b[1]);

    __m128d ret = _mm_cmpunord_pd(a, b);
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_cmpunord_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);

    double d[2];
    d[0] = cmp_hasNaN(_a[0], _b[0]);
    d[1] = _a[1];

    __m128d ret = _mm_cmpunord_sd(a, b);
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_comieq_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // FIXME:
    // The GCC does not implement _mm_comieq_sd correctly.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98612 for more
    // information.
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
    return TEST_UNIMPL;
#else
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
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
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    int32_t _c = (_a[0] >= _b[0]) ? 1 : 0;

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    int32_t c = _mm_comige_sd(a, b);

    ASSERT_RETURN(c == _c);
    return TEST_SUCCESS;
}

result_t test_mm_comigt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
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
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
    return TEST_UNIMPL;
#else
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
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
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
    return TEST_UNIMPL;
#else
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
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
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
    return TEST_UNIMPL;
#else
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
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
    double d[2] = {static_cast<double>(_a[0]), static_cast<double>(_a[1])};

    __m128d ret = _mm_cvtepi32_pd(a);
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_cvtepi32_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);
    float trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = static_cast<float>(_a[i]);
    }

    __m128 ret = _mm_cvtepi32_ps(a);
    return VALIDATE_FLOAT_M128(ret, trun);
}

OPTNONE result_t test_mm_cvtpd_epi32(const SSE2NEONTestImpl &impl,
                                     uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    int32_t d[2] = {};

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d[0] = sse2neon_saturate_cast_int32(bankersRounding(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(bankersRounding(_a[1]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d[0] = sse2neon_saturate_cast_int32(floor(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(floor(_a[1]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d[0] = sse2neon_saturate_cast_int32(ceil(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(ceil(_a[1]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d[0] = sse2neon_saturate_cast_int32(_a[0]);
        d[1] = sse2neon_saturate_cast_int32(_a[1]);
        break;
    }

    __m128d a = load_m128d(_a);
    __m128i ret = _mm_cvtpd_epi32(a);

    return validateInt32(ret, d[0], d[1], 0, 0);
}

OPTNONE result_t test_mm_cvtpd_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    int32_t d[2] = {};

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d[0] = sse2neon_saturate_cast_int32(bankersRounding(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(bankersRounding(_a[1]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d[0] = sse2neon_saturate_cast_int32(floor(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(floor(_a[1]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d[0] = sse2neon_saturate_cast_int32(ceil(_a[0]));
        d[1] = sse2neon_saturate_cast_int32(ceil(_a[1]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d[0] = sse2neon_saturate_cast_int32(_a[0]);
        d[1] = sse2neon_saturate_cast_int32(_a[1]);
        break;
    }

    __m128d a = load_m128d(_a);
    __m64 ret = _mm_cvtpd_pi32(a);

    return VALIDATE_INT32_M64(ret, d);
}

result_t test_mm_cvtpd_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    float f0 = static_cast<float>(_a[0]);
    float f1 = static_cast<float>(_a[1]);
    const __m128d a = load_m128d(_a);

    __m128 r = _mm_cvtpd_ps(a);

    return validateFloat(r, f0, f1, 0, 0);
}

result_t test_mm_cvtpi32_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m64 a = load_m64(_a);

    double d[2] = {static_cast<double>(_a[0]), static_cast<double>(_a[1])};

    __m128d ret = _mm_cvtpi32_pd(a);

    return VALIDATE_DOUBLE_M128(ret, d);
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
            d[i] = sse2neon_saturate_cast_int32(bankersRounding(_a[i]));
        }
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        for (uint32_t i = 0; i < 4; i++) {
            d[i] = sse2neon_saturate_cast_int32(floorf(_a[i]));
        }
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        for (uint32_t i = 0; i < 4; i++) {
            d[i] = sse2neon_saturate_cast_int32(ceilf(_a[i]));
        }
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        for (uint32_t i = 0; i < 4; i++) {
            d[i] = sse2neon_saturate_cast_int32(_a[i]);
        }
        break;
    }

    __m128i ret = _mm_cvtps_epi32(a);
    return VALIDATE_INT32_M128(ret, d);
}

result_t test_mm_cvtps_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    double d[2];
    d[0] = static_cast<double>(_a[0]);
    d[1] = static_cast<double>(_a[1]);
    const __m128 a = load_m128(_a);

    __m128d r = _mm_cvtps_pd(a);

    return VALIDATE_DOUBLE_M128(r, d);
}

result_t test_mm_cvtsd_f64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    double d = _a[0];

    const __m128d *a = reinterpret_cast<const __m128d *>(_a);
    double r = _mm_cvtsd_f64(*a);

    return r == d ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsd_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    int32_t d;

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d = sse2neon_saturate_cast_int32(bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d = sse2neon_saturate_cast_int32(floor(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d = sse2neon_saturate_cast_int32(ceil(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d = sse2neon_saturate_cast_int32(_a[0]);
        break;
    }

    __m128d a = load_m128d(_a);
    int32_t ret = _mm_cvtsd_si32(a);

    return ret == d ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsd_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    int64_t d = 0;

    switch (iter & 0x3) {
    case 0:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        d = sse2neon_saturate_cast_int64(bankersRounding(_a[0]));
        break;
    case 1:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        d = sse2neon_saturate_cast_int64(floor(_a[0]));
        break;
    case 2:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        d = sse2neon_saturate_cast_int64(ceil(_a[0]));
        break;
    case 3:
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        d = sse2neon_saturate_cast_int64(_a[0]);
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
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    float f[4];
    f[0] = static_cast<float>(_b[0]);
    f[1] = static_cast<float>(_a[1]);
    f[2] = static_cast<float>(_a[2]);
    f[3] = static_cast<float>(_a[3]);

    __m128 a = load_m128(_a);
    __m128d b = load_m128d(_b);
    __m128 c = _mm_cvtsd_ss(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_cvtsi128_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);

    int32_t d = _a[0];

    __m128i a = load_m128i(_a);
    int c = _mm_cvtsi128_si32(a);

    return d == c ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsi128_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

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
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const int32_t b = static_cast<const int32_t>(impl.mTestInts[iter]);

    __m128d a = load_m128d(_a);
    __m128d c = _mm_cvtsi32_sd(a, b);

    double d[2] = {static_cast<double>(b), _a[1]};
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cvtsi32_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);

    int32_t d = _a[0];

    __m128i c = _mm_cvtsi32_si128(*_a);

    return validateInt32(c, d, 0, 0, 0);
}

result_t test_mm_cvtsi64_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const int64_t b = static_cast<const int64_t>(impl.mTestInts[iter]);

    __m128d a = load_m128d(_a);
    __m128d c = _mm_cvtsi64_sd(a, b);

    double d[2] = {static_cast<double>(b), _a[1]};
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cvtsi64_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    int64_t d[2] = {_a[0], 0};

    __m128i c = _mm_cvtsi64_si128(*_a);

    return VALIDATE_INT64_M128(c, d);
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
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const float *_b = impl.mTestFloatPointer2;
    double d[2];
    d[0] = static_cast<double>(_b[0]);
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128 b = load_m128(_b);
    __m128d c = _mm_cvtss_sd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_cvttpd_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    __m128d a = load_m128d(_a);
    int32_t d0 = sse2neon_saturate_cast_int32(_a[0]);
    int32_t d1 = sse2neon_saturate_cast_int32(_a[1]);

    __m128i ret = _mm_cvttpd_epi32(a);
    return validateInt32(ret, d0, d1, 0, 0);
}

OPTNONE result_t test_mm_cvttpd_pi32(const SSE2NEONTestImpl &impl,
                                     uint32_t iter)
{
    double v0, v1;

    /* Test edge cases for specific iterations, random values otherwise */
    switch (iter) {
    case 0: /* NaN in first element */
        v0 = std::nan("");
        v1 = 42.0;
        break;
    case 1: /* NaN in second element */
        v0 = -123.5;
        v1 = std::nan("");
        break;
    case 2: /* Both NaN */
        v0 = std::nan("");
        v1 = std::nan("");
        break;
    case 3: /* Positive infinity */
        v0 = INFINITY;
        v1 = 0.0;
        break;
    case 4: /* Negative infinity */
        v0 = -INFINITY;
        v1 = 100.0;
        break;
    case 5: /* Both infinity */
        v0 = INFINITY;
        v1 = -INFINITY;
        break;
    case 6: /* Overflow: value > INT32_MAX */
        v0 = 2147483648.0;
        v1 = 0.0;
        break;
    case 7: /* Overflow: value < INT32_MIN */
        v0 = -2147483649.0;
        v1 = 0.0;
        break;
    case 8: /* Large overflow */
        v0 = 1.0e10;
        v1 = -1.0e10;
        break;
    case 9: /* Boundary: exactly INT32_MAX */
        v0 = 2147483647.0;
        v1 = -2147483648.0;
        break;
    case 10: /* Near boundary */
        v0 = 2147483646.0;
        v1 = -2147483647.0;
        break;
    case 11: /* Truncation toward zero */
        v0 = 1.9;
        v1 = -1.9;
        break;
    case 12: /* Mixed: NaN and overflow */
        v0 = std::nan("");
        v1 = 2147483648.0;
        break;
    case 13: /* Mixed: infinity and overflow */
        v0 = -INFINITY;
        v1 = 3000000000.0;
        break;
    default: {
        /* Use random test values from the test framework */
        const double *_a =
            reinterpret_cast<const double *>(impl.mTestFloatPointer1);
        v0 = _a[0];
        v1 = _a[1];
        break;
    }
    }

    __m128d a = _mm_set_pd(v1, v0);
    int32_t d0 = sse2neon_saturate_cast_int32(v0);
    int32_t d1 = sse2neon_saturate_cast_int32(v1);

    __m64 ret = _mm_cvttpd_pi32(a);
    return validateInt32(ret, d0, d1);
}

result_t test_mm_cvttps_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    __m128 a = load_m128(_a);
    int32_t trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = sse2neon_saturate_cast_int32(_a[i]);
    }

    __m128i ret = _mm_cvttps_epi32(a);
    return VALIDATE_INT32_M128(ret, trun);
}

result_t test_mm_cvttsd_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    __m128d a = _mm_load_sd(_a);
    int32_t ret = _mm_cvttsd_si32(a);

    return ret == sse2neon_saturate_cast_int32(_a[0]) ? TEST_SUCCESS
                                                      : TEST_FAIL;
}

result_t test_mm_cvttsd_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    __m128d a = _mm_load_sd(_a);
    int64_t ret = _mm_cvttsd_si64(a);

    return ret == sse2neon_saturate_cast_int64(_a[0]) ? TEST_SUCCESS
                                                      : TEST_FAIL;
}

result_t test_mm_cvttsd_si64x(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_cvttsd_si64x is an alias defined only in sse2neon.h, not in native
    // x86 SSE headers. Only test on ARM where sse2neon provides this macro.
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || \
    defined(_M_ARM64EC)
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    __m128d a = _mm_load_sd(_a);
    int64_t ret = _mm_cvttsd_si64x(a);

    return ret == sse2neon_saturate_cast_int64(_a[0]) ? TEST_SUCCESS
                                                      : TEST_FAIL;
#else
    (void) impl;
    (void) iter;
    return TEST_UNIMPL;
#endif
}

result_t test_mm_div_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2] = {0.0, 0.0};

    if (_b[0] != 0.0)
        d[0] = _a[0] / _b[0];
    if (_b[1] != 0.0)
        d[1] = _a[1] / _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_div_pd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_div_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    double d[2];
    d[0] = _a[0] / _b[0];
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);

    __m128d c = _mm_div_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_extract_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint16_t *_a = reinterpret_cast<uint16_t *>(impl.mTestIntPointer1);
    const int idx = iter & 0x7;
    __m128i a = load_m128i(_a);
    int c = 0;
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
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t insert = static_cast<int16_t>(*impl.mTestIntPointer2);

#define TEST_IMPL(IDX)                                      \
    int16_t d##IDX[8];                                      \
    for (int i = 0; i < 8; i++) {                           \
        d##IDX[i] = _a[i];                                  \
    }                                                       \
    d##IDX[IDX] = insert;                                   \
                                                            \
    __m128i a##IDX = load_m128i(_a);                        \
    __m128i b##IDX = _mm_insert_epi16(a##IDX, insert, IDX); \
    CHECK_RESULT(VALIDATE_INT16_M128(b##IDX, d##IDX))

    IMM_8_ITER
#undef TEST_IMPL

    return TEST_SUCCESS;
}

result_t test_mm_lfence(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    /* FIXME: Assume that memory barriers always function as intended. */
    return TEST_SUCCESS;
}

result_t test_mm_load_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    __m128d a = _mm_load_pd(p);
    return VALIDATE_DOUBLE_M128(a, p);
}

result_t test_mm_load_pd1(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    __m128d a = _mm_load_pd1(p);
    double d[2] = {p[0], p[0]};
    return VALIDATE_DOUBLE_M128(a, d);
}

result_t test_mm_load_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    __m128d a = _mm_load_sd(p);
    double d[2] = {p[0], 0};
    return VALIDATE_DOUBLE_M128(a, d);
}

result_t test_mm_load_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *addr = impl.mTestIntPointer1;

    __m128i ret = _mm_load_si128(reinterpret_cast<const __m128i *>(addr));

    return VALIDATE_INT32_M128(ret, addr);
}

result_t test_mm_load1_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *addr =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    __m128d ret = _mm_load1_pd(addr);

    double d[2] = {addr[0], addr[0]};
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_loadh_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *addr =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    __m128d a = load_m128d(_a);
    __m128d ret = _mm_loadh_pd(a, addr);

    double d[2] = {_a[0], addr[0]};
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_loadl_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *addr =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    __m128i ret = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(addr));

    int64_t d[2] = {addr[0], 0};
    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_loadl_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *addr =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    __m128d a = load_m128d(_a);
    __m128d ret = _mm_loadl_pd(a, addr);

    double d[2] = {addr[0], _a[1]};
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_loadr_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *addr =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    __m128d ret = _mm_loadr_pd(addr);

    double d[2] = {addr[1], addr[0]};
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_loadu_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    __m128d a = _mm_loadu_pd(p);
    return VALIDATE_DOUBLE_M128(a, p);
}

result_t test_mm_loadu_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const unaligned_int32_t *_a = reinterpret_cast<const unaligned_int32_t *>(
        impl.mTestUnalignedInts + 1);
    __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(_a));
    return VALIDATE_INT32_M128(c, _a);
}

result_t test_mm_loadu_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_loadu_si32. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
    return TEST_UNIMPL;
#else
    const unaligned_int32_t *addr = reinterpret_cast<const unaligned_int32_t *>(
        impl.mTestUnalignedInts + 1);

    __m128i ret = _mm_loadu_si32(static_cast<const void *>(addr));

    return validateInt32(ret, addr[0], 0, 0, 0);
#endif
}

result_t test_mm_madd_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int32_t d0 = static_cast<int32_t>(_a[0]) * _b[0];
    int32_t d1 = static_cast<int32_t>(_a[1]) * _b[1];
    int32_t d2 = static_cast<int32_t>(_a[2]) * _b[2];
    int32_t d3 = static_cast<int32_t>(_a[3]) * _b[3];
    int32_t d4 = static_cast<int32_t>(_a[4]) * _b[4];
    int32_t d5 = static_cast<int32_t>(_a[5]) * _b[5];
    int32_t d6 = static_cast<int32_t>(_a[6]) * _b[6];
    int32_t d7 = static_cast<int32_t>(_a[7]) * _b[7];

    int32_t e[4];
    e[0] = d0 + d1;
    e[1] = d2 + d3;
    e[2] = d4 + d5;
    e[3] = d6 + d7;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_madd_epi16(a, b);
    return VALIDATE_INT32_M128(c, e);
}

result_t test_mm_maskmoveu_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_mask =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    char mem_addr[16];

    __m128i a = load_m128i(_a);
    __m128i mask = load_m128i(_mask);
    _mm_maskmoveu_si128(a, mask, mem_addr);

    for (int i = 0; i < 16; i++) {
        if (_mask[i] >> 7) {
            ASSERT_RETURN(_a[i] == static_cast<uint8_t>(mem_addr[i]));
        }
    }

    return TEST_SUCCESS;
}

result_t test_mm_max_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if (__GNUC__ == 8) || (__GNUC__ == 9 && __GNUC_MINOR__ == 2)
#error Using older gcc versions can lead to an operand mismatch error. This issue affects all versions prior to gcc 10.
#else
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] > _b[3] ? _a[3] : _b[3];
    d[4] = _a[4] > _b[4] ? _a[4] : _b[4];
    d[5] = _a[5] > _b[5] ? _a[5] : _b[5];
    d[6] = _a[6] > _b[6] ? _a[6] : _b[6];
    d[7] = _a[7] > _b[7] ? _a[7] : _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_max_epi16(a, b);
    return VALIDATE_INT16_M128(c, d);
#endif
}

result_t test_mm_max_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if (__GNUC__ == 8) || (__GNUC__ == 9 && __GNUC_MINOR__ == 2)
#error Using older gcc versions can lead to an operand mismatch error. This issue affects all versions prior to gcc 10.
#else
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    uint8_t d[16];
    d[0] = (static_cast<uint8_t>(_a[0]) > static_cast<uint8_t>(_b[0]))
               ? (static_cast<uint8_t>(_a[0]))
               : (static_cast<uint8_t>(_b[0]));
    d[1] = (static_cast<uint8_t>(_a[1]) > static_cast<uint8_t>(_b[1]))
               ? (static_cast<uint8_t>(_a[1]))
               : (static_cast<uint8_t>(_b[1]));
    d[2] = (static_cast<uint8_t>(_a[2]) > static_cast<uint8_t>(_b[2]))
               ? (static_cast<uint8_t>(_a[2]))
               : (static_cast<uint8_t>(_b[2]));
    d[3] = (static_cast<uint8_t>(_a[3]) > static_cast<uint8_t>(_b[3]))
               ? (static_cast<uint8_t>(_a[3]))
               : (static_cast<uint8_t>(_b[3]));
    d[4] = (static_cast<uint8_t>(_a[4]) > static_cast<uint8_t>(_b[4]))
               ? (static_cast<uint8_t>(_a[4]))
               : (static_cast<uint8_t>(_b[4]));
    d[5] = (static_cast<uint8_t>(_a[5]) > static_cast<uint8_t>(_b[5]))
               ? (static_cast<uint8_t>(_a[5]))
               : (static_cast<uint8_t>(_b[5]));
    d[6] = (static_cast<uint8_t>(_a[6]) > static_cast<uint8_t>(_b[6]))
               ? (static_cast<uint8_t>(_a[6]))
               : (static_cast<uint8_t>(_b[6]));
    d[7] = (static_cast<uint8_t>(_a[7]) > static_cast<uint8_t>(_b[7]))
               ? (static_cast<uint8_t>(_a[7]))
               : (static_cast<uint8_t>(_b[7]));
    d[8] = (static_cast<uint8_t>(_a[8]) > static_cast<uint8_t>(_b[8]))
               ? (static_cast<uint8_t>(_a[8]))
               : (static_cast<uint8_t>(_b[8]));
    d[9] = (static_cast<uint8_t>(_a[9]) > static_cast<uint8_t>(_b[9]))
               ? (static_cast<uint8_t>(_a[9]))
               : (static_cast<uint8_t>(_b[9]));
    d[10] = (static_cast<uint8_t>(_a[10]) > static_cast<uint8_t>(_b[10]))
                ? (static_cast<uint8_t>(_a[10]))
                : (static_cast<uint8_t>(_b[10]));
    d[11] = (static_cast<uint8_t>(_a[11]) > static_cast<uint8_t>(_b[11]))
                ? (static_cast<uint8_t>(_a[11]))
                : (static_cast<uint8_t>(_b[11]));
    d[12] = (static_cast<uint8_t>(_a[12]) > static_cast<uint8_t>(_b[12]))
                ? (static_cast<uint8_t>(_a[12]))
                : (static_cast<uint8_t>(_b[12]));
    d[13] = (static_cast<uint8_t>(_a[13]) > static_cast<uint8_t>(_b[13]))
                ? (static_cast<uint8_t>(_a[13]))
                : (static_cast<uint8_t>(_b[13]));
    d[14] = (static_cast<uint8_t>(_a[14]) > static_cast<uint8_t>(_b[14]))
                ? (static_cast<uint8_t>(_a[14]))
                : (static_cast<uint8_t>(_b[14]));
    d[15] = (static_cast<uint8_t>(_a[15]) > static_cast<uint8_t>(_b[15]))
                ? (static_cast<uint8_t>(_a[15]))
                : (static_cast<uint8_t>(_b[15]));

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_max_epu8(a, b);
    return VALIDATE_UINT8_M128(c, d);
#endif
}

result_t test_mm_max_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    double d[2];
    d[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] > _b[1] ? _a[1] : _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_max_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_max_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_max_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_mfence(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    /* FIXME: Assume that memory barriers always function as intended. */
    return TEST_SUCCESS;
}

result_t test_mm_min_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] < _b[3] ? _a[3] : _b[3];
    d[4] = _a[4] < _b[4] ? _a[4] : _b[4];
    d[5] = _a[5] < _b[5] ? _a[5] : _b[5];
    d[6] = _a[6] < _b[6] ? _a[6] : _b[6];
    d[7] = _a[7] < _b[7] ? _a[7] : _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epi16(a, b);
    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_min_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    uint8_t d[16];
    d[0] = (static_cast<uint8_t>(_a[0]) < static_cast<uint8_t>(_b[0]))
               ? static_cast<uint8_t>(_a[0])
               : static_cast<uint8_t>(_b[0]);
    d[1] = (static_cast<uint8_t>(_a[1]) < static_cast<uint8_t>(_b[1]))
               ? static_cast<uint8_t>(_a[1])
               : static_cast<uint8_t>(_b[1]);
    d[2] = (static_cast<uint8_t>(_a[2]) < static_cast<uint8_t>(_b[2]))
               ? static_cast<uint8_t>(_a[2])
               : static_cast<uint8_t>(_b[2]);
    d[3] = (static_cast<uint8_t>(_a[3]) < static_cast<uint8_t>(_b[3]))
               ? static_cast<uint8_t>(_a[3])
               : static_cast<uint8_t>(_b[3]);
    d[4] = (static_cast<uint8_t>(_a[4]) < static_cast<uint8_t>(_b[4]))
               ? static_cast<uint8_t>(_a[4])
               : static_cast<uint8_t>(_b[4]);
    d[5] = (static_cast<uint8_t>(_a[5]) < static_cast<uint8_t>(_b[5]))
               ? static_cast<uint8_t>(_a[5])
               : static_cast<uint8_t>(_b[5]);
    d[6] = (static_cast<uint8_t>(_a[6]) < static_cast<uint8_t>(_b[6]))
               ? static_cast<uint8_t>(_a[6])
               : static_cast<uint8_t>(_b[6]);
    d[7] = (static_cast<uint8_t>(_a[7]) < static_cast<uint8_t>(_b[7]))
               ? static_cast<uint8_t>(_a[7])
               : static_cast<uint8_t>(_b[7]);
    d[8] = (static_cast<uint8_t>(_a[8]) < static_cast<uint8_t>(_b[8]))
               ? static_cast<uint8_t>(_a[8])
               : static_cast<uint8_t>(_b[8]);
    d[9] = (static_cast<uint8_t>(_a[9]) < static_cast<uint8_t>(_b[9]))
               ? static_cast<uint8_t>(_a[9])
               : static_cast<uint8_t>(_b[9]);
    d[10] = (static_cast<uint8_t>(_a[10]) < static_cast<uint8_t>(_b[10]))
                ? static_cast<uint8_t>(_a[10])
                : static_cast<uint8_t>(_b[10]);
    d[11] = (static_cast<uint8_t>(_a[11]) < static_cast<uint8_t>(_b[11]))
                ? static_cast<uint8_t>(_a[11])
                : static_cast<uint8_t>(_b[11]);
    d[12] = (static_cast<uint8_t>(_a[12]) < static_cast<uint8_t>(_b[12]))
                ? static_cast<uint8_t>(_a[12])
                : static_cast<uint8_t>(_b[12]);
    d[13] = (static_cast<uint8_t>(_a[13]) < static_cast<uint8_t>(_b[13]))
                ? static_cast<uint8_t>(_a[13])
                : static_cast<uint8_t>(_b[13]);
    d[14] = (static_cast<uint8_t>(_a[14]) < static_cast<uint8_t>(_b[14]))
                ? static_cast<uint8_t>(_a[14])
                : static_cast<uint8_t>(_b[14]);
    d[15] = (static_cast<uint8_t>(_a[15]) < static_cast<uint8_t>(_b[15]))
                ? static_cast<uint8_t>(_a[15])
                : static_cast<uint8_t>(_b[15]);

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epu8(a, b);
    return VALIDATE_UINT8_M128(c, d);
}

result_t test_mm_min_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] < _b[1] ? _a[1] : _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);

    __m128d c = _mm_min_pd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_min_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_min_sd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_move_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    int64_t d[2];
    d[0] = _a[0];
    d[1] = 0;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_move_epi64(a);

    return VALIDATE_INT64_M128(c, d);
}

result_t test_mm_move_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);

    double d[2];
    d[0] = _b[0];
    d[1] = _a[1];

    __m128d ret = _mm_move_sd(a, b);
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_movemask_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // Test 1: Random data from test framework
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a = load_m128i(_a);

    const uint8_t *ip = reinterpret_cast<const uint8_t *>(_a);
    int ret = 0;
    for (int i = 0; i < 16; i++) {
        if (ip[i] & 0x80)
            ret |= (1 << i);
    }
    int test = _mm_movemask_epi8(a);
    ASSERT_RETURN(test == ret);

    // Test 2: Canonical edge cases (explicit patterns)
    // These test specific bit patterns that stress the implementation
    struct {
        uint8_t bytes[16];
        int expected;
    } edge_cases[] = {
        // All zeros - no MSBs set
        {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00},
         0x0000},
        // All ones - all MSBs set
        {{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
          0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
         0xFFFF},
        // Single bit at position 0
        {{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00},
         0x0001},
        // Single bit at position 15
        {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x80},
         0x8000},
        // Low 8 bits only (tests low half extraction)
        {{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00},
         0x00FF},
        // High 8 bits only (tests high half extraction)
        {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80},
         0xFF00},
        // Alternating pattern (0x5555) - tests even positions
        {{0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
          0x00, 0x80, 0x00, 0x80, 0x00},
         0x5555},
        // Alternating pattern (0xAAAA) - tests odd positions
        {{0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00,
          0x80, 0x00, 0x80, 0x00, 0x80},
         0xAAAA},
        // All 0x7F - MSB clear but other bits set
        {{0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F,
          0x7F, 0x7F, 0x7F, 0x7F, 0x7F},
         0x0000},
        // Mixed values with MSB set (0x81-0x90)
        {{0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B,
          0x8C, 0x8D, 0x8E, 0x8F, 0x90},
         0xFFFF},
        // Boundary: 0x80 exactly (minimum negative signed byte)
        {{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
          0x80, 0x80, 0x80, 0x80, 0x80},
         0xFFFF},
        // Walking bit at position 1
        {{0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00},
         0x0002},
        // Walking bit at position 7 (end of low byte)
        {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00},
         0x0080},
        // Walking bit at position 8 (start of high byte)
        {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00},
         0x0100},
        // Comparison result simulation (0xFF or 0x00 bytes)
        {{0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
          0x00, 0x00, 0x00, 0xFF, 0xFF},
         0xC735},
        // Nibble boundary test
        {{0x80, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00, 0x80, 0x80, 0x80,
          0x80, 0x00, 0x00, 0x00, 0x00},
         0x0F0F},
    };

    for (size_t i = 0; i < sizeof(edge_cases) / sizeof(edge_cases[0]); i++) {
        __m128i vec = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(edge_cases[i].bytes));
        int result = _mm_movemask_epi8(vec);
        ASSERT_RETURN(result == edge_cases[i].expected);
    }

    return TEST_SUCCESS;
}

result_t test_mm_movemask_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // Test 1: Random data from test framework
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    int _c = 0;
    _c |= static_cast<int>(((*reinterpret_cast<const uint64_t *>(_a)) >> 63) &
                           0x1);
    _c |= (((*reinterpret_cast<const uint64_t *>(_a + 1)) >> 62) & 0x2);

    __m128d a = load_m128d(_a);
    int c = _mm_movemask_pd(a);
    ASSERT_RETURN(c == _c);

    // Test 2: Edge cases with specific double patterns
    union {
        double d;
        uint64_t u;
    } conv;

    struct {
        uint64_t bits[2];
        int expected;
    } edge_cases[] = {
        // All positive zeros
        {{0x0000000000000000ULL, 0x0000000000000000ULL}, 0x0},
        // All negative zeros
        {{0x8000000000000000ULL, 0x8000000000000000ULL}, 0x3},
        // Mixed: negative zero at position 0 only
        {{0x8000000000000000ULL, 0x0000000000000000ULL}, 0x1},
        // Mixed: negative zero at position 1 only
        {{0x0000000000000000ULL, 0x8000000000000000ULL}, 0x2},
        // Positive infinity
        {{0x7FF0000000000000ULL, 0x7FF0000000000000ULL}, 0x0},
        // Negative infinity
        {{0xFFF0000000000000ULL, 0xFFF0000000000000ULL}, 0x3},
        // Positive NaN
        {{0x7FF8000000000000ULL, 0x7FF8000000000000ULL}, 0x0},
        // Negative NaN
        {{0xFFF8000000000000ULL, 0xFFF8000000000000ULL}, 0x3},
        // Max positive double
        {{0x7FEFFFFFFFFFFFFFULL, 0x7FEFFFFFFFFFFFFFULL}, 0x0},
        // Max negative double
        {{0xFFEFFFFFFFFFFFFFULL, 0xFFEFFFFFFFFFFFFFULL}, 0x3},
        // Smallest positive denormal
        {{0x0000000000000001ULL, 0x0000000000000001ULL}, 0x0},
        // Smallest negative denormal
        {{0x8000000000000001ULL, 0x8000000000000001ULL}, 0x3},
        // Mixed positive and negative
        {{0x3FF0000000000000ULL, 0xBFF0000000000000ULL}, 0x2},  // +1.0, -1.0
        {{0xBFF0000000000000ULL, 0x3FF0000000000000ULL}, 0x1},  // -1.0, +1.0
    };

    for (size_t i = 0; i < sizeof(edge_cases) / sizeof(edge_cases[0]); i++) {
        double doubles[2];
        for (int j = 0; j < 2; j++) {
            conv.u = edge_cases[i].bits[j];
            doubles[j] = conv.d;
        }
        __m128d vec = _mm_loadu_pd(doubles);
        int result = _mm_movemask_pd(vec);
        ASSERT_RETURN(result == edge_cases[i].expected);
    }

    return TEST_SUCCESS;
}

result_t test_mm_movepi64_pi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    int64_t d0 = _a[0];

    __m128i a = load_m128i(_a);
    __m64 c = _mm_movepi64_pi64(a);

    return validateInt64(c, d0);
}

result_t test_mm_movpi64_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    int64_t d[2] = {_a[0], 0};

    __m64 a = load_m64(_a);
    __m128i c = _mm_movpi64_epi64(a);

    return VALIDATE_INT64_M128(c, d);
}

result_t test_mm_mul_epu32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);
    const uint32_t *_b =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer2);
    uint64_t d[2];
    d[0] = static_cast<uint64_t>(_a[0]) * static_cast<uint64_t>(_b[0]);
    d[1] = static_cast<uint64_t>(_a[2]) * static_cast<uint64_t>(_b[2]);

    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(_a));
    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(_b));
    __m128i r = _mm_mul_epu32(a, b);
    return VALIDATE_UINT64_M128(r, d);
}

result_t test_mm_mul_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] * _b[0];
    d[1] = _a[1] * _b[1];

    __m128d a = _mm_load_pd(_a);
    __m128d b = _mm_load_pd(_b);
    __m128d c = _mm_mul_pd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_mul_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] * _b[0];
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_mul_sd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_mul_su32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);
    const uint32_t *_b =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer2);

    uint64_t u = static_cast<uint64_t>(_a[0]) * static_cast<uint64_t>(_b[0]);

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 r = _mm_mul_su32(a, b);

    return validateUInt64(r, u);
}

result_t test_mm_mulhi_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    for (uint32_t i = 0; i < 8; i++) {
        int32_t m = static_cast<int32_t>(_a[i]) * static_cast<int32_t>(_b[i]);
        d[i] = static_cast<int16_t>(m >> 16);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_mulhi_epi16(a, b);
    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_mulhi_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);
    const uint16_t *_b =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer2);
    uint16_t d[8];
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t m =
            static_cast<uint32_t>(_a[i]) * static_cast<uint32_t>(_b[i]);
        d[i] = static_cast<uint16_t>(m >> 16);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_mulhi_epu16(a, b);
    return VALIDATE_UINT16_M128(c, d);
}

result_t test_mm_mullo_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] = _a[0] * _b[0];
    d[1] = _a[1] * _b[1];
    d[2] = _a[2] * _b[2];
    d[3] = _a[3] * _b[3];
    d[4] = _a[4] * _b[4];
    d[5] = _a[5] * _b[5];
    d[6] = _a[6] * _b[6];
    d[7] = _a[7] * _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_mullo_epi16(a, b);
    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_or_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestFloatPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = sse2neon_tool_recast_f64(static_cast<uint64_t>(_a[0] | _b[0]));
    d[1] = sse2neon_tool_recast_f64(static_cast<uint64_t>(_a[1] | _b[1]));

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_or_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_or_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128 fc = _mm_or_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b));
    __m128i c = _mm_castps_si128(fc);
    // now for the assertion...
    const uint32_t *ia = reinterpret_cast<const uint32_t *>(&a);
    const uint32_t *ib = reinterpret_cast<const uint32_t *>(&b);
    uint32_t r[4];
    r[0] = ia[0] | ib[0];
    r[1] = ia[1] | ib[1];
    r[2] = ia[2] | ib[2];
    r[3] = ia[3] | ib[3];
    __m128i ret =
        do_mm_set_epi32(static_cast<int32_t>(r[3]), static_cast<int32_t>(r[2]),
                        static_cast<int32_t>(r[1]), static_cast<int32_t>(r[0]));
    result_t res = VALIDATE_UINT32_M128(c, r);
    if (res) {
        res = VALIDATE_UINT32_M128(ret, r);
    }
    return res;
}

result_t test_mm_packs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int8_t max = INT8_MAX;
    int8_t min = INT8_MIN;
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    int8_t d[16];
    for (int i = 0; i < 8; i++) {
        if (_a[i] > max)
            d[i] = max;
        else if (_a[i] < min)
            d[i] = min;
        else
            d[i] = static_cast<int8_t>(_a[i]);
    }
    for (int i = 0; i < 8; i++) {
        if (_b[i] > max)
            d[i + 8] = max;
        else if (_b[i] < min)
            d[i + 8] = min;
        else
            d[i + 8] = static_cast<int8_t>(_b[i]);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_packs_epi16(a, b);

    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_packs_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int16_t max = INT16_MAX;
    int16_t min = INT16_MIN;
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    int16_t d[8];
    for (int i = 0; i < 4; i++) {
        if (_a[i] > max)
            d[i] = max;
        else if (_a[i] < min)
            d[i] = min;
        else
            d[i] = static_cast<int16_t>(_a[i]);
    }
    for (int i = 0; i < 4; i++) {
        if (_b[i] > max)
            d[i + 4] = max;
        else if (_b[i] < min)
            d[i + 4] = min;
        else
            d[i + 4] = static_cast<int16_t>(_b[i]);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_packs_epi32(a, b);

    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_packus_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint8_t max = UINT8_MAX;
    uint8_t min = 0;
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    uint8_t d[16];
    for (int i = 0; i < 8; i++) {
        if (_a[i] > static_cast<int16_t>(max))
            d[i] = max;
        else if (_a[i] < static_cast<int16_t>(min))
            d[i] = min;
        else
            d[i] = static_cast<uint8_t>(_a[i]);
    }
    for (int i = 0; i < 8; i++) {
        if (_b[i] > static_cast<int16_t>(max))
            d[i + 8] = max;
        else if (_b[i] < static_cast<int16_t>(min))
            d[i + 8] = min;
        else
            d[i + 8] = static_cast<uint8_t>(_b[i]);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_packus_epi16(a, b);

    return VALIDATE_UINT8_M128(c, d);
}

result_t test_mm_pause(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    _mm_pause();
    return TEST_SUCCESS;
}

result_t test_mm_sad_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    uint16_t d0 = 0;
    uint16_t d1 = 0;
    for (int i = 0; i < 8; i++) {
        d0 += static_cast<uint16_t>(abs(_a[i] - _b[i]));
    }
    for (int i = 8; i < 16; i++) {
        d1 += static_cast<uint16_t>(abs(_a[i] - _b[i]));
    }

    const __m128i a = load_m128i(_a);
    const __m128i b = load_m128i(_b);
    __m128i c = _mm_sad_epu8(a, b);
    return validateUInt16(c, d0, 0, 0, 0, d1, 0, 0, 0);
}

result_t test_mm_set_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    int16_t d[8];
    d[0] = _a[0];
    d[1] = _a[1];
    d[2] = _a[2];
    d[3] = _a[3];
    d[4] = _a[4];
    d[5] = _a[5];
    d[6] = _a[6];
    d[7] = _a[7];

    __m128i c = _mm_set_epi16(d[7], d[6], d[5], d[4], d[3], d[2], d[1], d[0]);
    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_set_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t d[4];
    d[3] = impl.mTestInts[iter];
    d[2] = impl.mTestInts[iter + 1];
    d[1] = impl.mTestInts[iter + 2];
    d[0] = impl.mTestInts[iter + 3];
    __m128i a = _mm_set_epi32(d[3], d[2], d[1], d[0]);
    return VALIDATE_INT32_M128(a, d);
}

result_t test_mm_set_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    __m128i ret = _mm_set_epi64(load_m64(&_a[1]), load_m64(&_a[0]));

    return VALIDATE_INT64_M128(ret, _a);
}

result_t test_mm_set_epi64x(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    __m128i ret = _mm_set_epi64x(_a[1], _a[0]);

    return VALIDATE_INT64_M128(ret, _a);
}

result_t test_mm_set_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    int8_t d[16];
    d[0] = _a[0];
    d[1] = _a[1];
    d[2] = _a[2];
    d[3] = _a[3];
    d[4] = _a[4];
    d[5] = _a[5];
    d[6] = _a[6];
    d[7] = _a[7];
    d[8] = _a[8];
    d[9] = _a[9];
    d[10] = _a[10];
    d[11] = _a[11];
    d[12] = _a[12];
    d[13] = _a[13];
    d[14] = _a[14];
    d[15] = _a[15];

    __m128i c =
        _mm_set_epi8(d[15], d[14], d[13], d[12], d[11], d[10], d[9], d[8], d[7],
                     d[6], d[5], d[4], d[3], d[2], d[1], d[0]);
    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_set_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    double x = p[0];
    double y = p[1];
    __m128d a = _mm_set_pd(x, y);
    double d[2] = {y, x};
    return VALIDATE_DOUBLE_M128(a, d);
}

result_t test_mm_set_pd1(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double _a = impl.mTestFloats[iter];

    __m128d a = _mm_set_pd1(_a);

    double d[2] = {_a, _a};
    return VALIDATE_DOUBLE_M128(a, d);
}

result_t test_mm_set_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    double d[2];
    d[0] = _a[0];
    d[1] = 0.0;

    __m128d a = _mm_set_sd(_a[0]);
    return VALIDATE_DOUBLE_M128(a, d);
}

result_t test_mm_set1_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
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
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    __m128i ret = _mm_set1_epi64(load_m64(&_a[0]));

    int64_t d[2] = {_a[0], _a[0]};
    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_set1_epi64x(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);

    __m128i ret = _mm_set1_epi64x(_a[0]);

    int64_t d[2] = {_a[0], _a[0]};
    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_set1_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    int8_t d0 = _a[0];
    __m128i c = _mm_set1_epi8(d0);
    return validateInt8(c, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0,
                        d0, d0, d0);
}

result_t test_mm_set1_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    double d[2] = {_a[0], _a[0]};
    __m128d c = _mm_set1_pd(d[0]);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_setr_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    __m128i c =
        _mm_setr_epi16(_a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6], _a[7]);

    return VALIDATE_INT16_M128(c, _a);
}

result_t test_mm_setr_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m128i c = _mm_setr_epi32(_a[0], _a[1], _a[2], _a[3]);
    return VALIDATE_INT32_M128(c, _a);
}

result_t test_mm_setr_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    __m128i c = _mm_setr_epi64(load_m64(&_a[0]), load_m64(&_a[1]));
    return VALIDATE_INT64_M128(c, _a);
}

result_t test_mm_setr_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);

    __m128i c = _mm_setr_epi8(_a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6],
                              _a[7], _a[8], _a[9], _a[10], _a[11], _a[12],
                              _a[13], _a[14], _a[15]);

    return VALIDATE_INT8_M128(c, _a);
}

result_t test_mm_setr_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    double x = p[0];
    double y = p[1];

    __m128d a = _mm_setr_pd(x, y);

    double d[2] = {x, y};
    return VALIDATE_DOUBLE_M128(a, d);
}

result_t test_mm_setzero_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128d a = _mm_setzero_pd();
    double d[2] = {0, 0};
    return VALIDATE_DOUBLE_M128(a, d);
}

result_t test_mm_setzero_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128i a = _mm_setzero_si128();
    return validateInt32(a, 0, 0, 0, 0);
}

result_t test_mm_shuffle_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if (__GNUC__ == 8) || (__GNUC__ == 9 && __GNUC_MINOR__ == 2)
#error Using older gcc versions can lead to an operand mismatch error. This issue affects all versions prior to gcc 10.
#else
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a, c;
    int32_t _d[4];

#define TEST_IMPL(IDX)              \
    _d[0] = _a[((IDX) & 0x3)];      \
    _d[1] = _a[((IDX >> 2) & 0x3)]; \
    _d[2] = _a[((IDX >> 4) & 0x3)]; \
    _d[3] = _a[((IDX >> 6) & 0x3)]; \
                                    \
    a = load_m128i(_a);             \
    c = _mm_shuffle_epi32(a, IDX);  \
    CHECK_RESULT(VALIDATE_INT32_M128(c, _d))

    IMM_256_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
#endif
}

result_t test_mm_shuffle_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    __m128d a, b, c;

#define TEST_IMPL(IDX)                     \
    a = load_m128d(_a);                    \
    b = load_m128d(_b);                    \
    c = _mm_shuffle_pd(a, b, IDX);         \
                                           \
    double d0##IDX = _a[IDX & 0x1];        \
    double d1##IDX = _b[(IDX & 0x2) >> 1]; \
    CHECK_RESULT(validateDouble(c, d0##IDX, d1##IDX))

    IMM_4_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_shufflehi_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if (__GNUC__ == 8) || (__GNUC__ == 9 && __GNUC_MINOR__ == 2)
#error Using older gcc versions can lead to an operand mismatch error. This issue affects all versions prior to gcc 10.
#else
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    __m128i a, c;

    int16_t _d[8];
#define TEST_IMPL(IDX)                                                         \
    _d[0] = _a[0];                                                             \
    _d[1] = _a[1];                                                             \
    _d[2] = _a[2];                                                             \
    _d[3] = _a[3];                                                             \
    _d[4] = static_cast<int16_t>((reinterpret_cast<const int64_t *>(_a))[1] >> \
                                 ((IDX & 0x3) * 16));                          \
    _d[5] = static_cast<int16_t>((reinterpret_cast<const int64_t *>(_a))[1] >> \
                                 (((IDX >> 2) & 0x3) * 16));                   \
    _d[6] = static_cast<int16_t>((reinterpret_cast<const int64_t *>(_a))[1] >> \
                                 (((IDX >> 4) & 0x3) * 16));                   \
    _d[7] = static_cast<int16_t>((reinterpret_cast<const int64_t *>(_a))[1] >> \
                                 (((IDX >> 6) & 0x3) * 16));                   \
                                                                               \
    a = load_m128i(_a);                                                        \
    c = _mm_shufflehi_epi16(a, IDX);                                           \
                                                                               \
    CHECK_RESULT(VALIDATE_INT16_M128(c, _d))

    IMM_256_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
#endif
}

result_t test_mm_shufflelo_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
#if (__GNUC__ == 8) || (__GNUC__ == 9 && __GNUC_MINOR__ == 2)
#error Using older gcc versions can lead to an operand mismatch error. This issue affects all versions prior to gcc 10.
#else
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    __m128i a, c;
    int16_t _d[8];

#define TEST_IMPL(IDX)                                                         \
    _d[0] = static_cast<int16_t>((reinterpret_cast<const int64_t *>(_a))[0] >> \
                                 ((IDX & 0x3) * 16));                          \
    _d[1] = static_cast<int16_t>((reinterpret_cast<const int64_t *>(_a))[0] >> \
                                 (((IDX >> 2) & 0x3) * 16));                   \
    _d[2] = static_cast<int16_t>((reinterpret_cast<const int64_t *>(_a))[0] >> \
                                 (((IDX >> 4) & 0x3) * 16));                   \
    _d[3] = static_cast<int16_t>((reinterpret_cast<const int64_t *>(_a))[0] >> \
                                 (((IDX >> 6) & 0x3) * 16));                   \
    _d[4] = _a[4];                                                             \
    _d[5] = _a[5];                                                             \
    _d[6] = _a[6];                                                             \
    _d[7] = _a[7];                                                             \
                                                                               \
    a = load_m128i(_a);                                                        \
    c = _mm_shufflelo_epi16(a, IDX);                                           \
                                                                               \
    CHECK_RESULT(VALIDATE_INT16_M128(c, _d))

    IMM_256_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
#endif
}

result_t test_mm_sll_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    __m128i a, b, c;
    uint8_t idx;
#define TEST_IMPL(IDX)                                                \
    uint16_t d##IDX[8];                                               \
    idx = IDX;                                                        \
    d##IDX[0] = (idx > 15) ? 0 : static_cast<uint16_t>(_a[0] << idx); \
    d##IDX[1] = (idx > 15) ? 0 : static_cast<uint16_t>(_a[1] << idx); \
    d##IDX[2] = (idx > 15) ? 0 : static_cast<uint16_t>(_a[2] << idx); \
    d##IDX[3] = (idx > 15) ? 0 : static_cast<uint16_t>(_a[3] << idx); \
    d##IDX[4] = (idx > 15) ? 0 : static_cast<uint16_t>(_a[4] << idx); \
    d##IDX[5] = (idx > 15) ? 0 : static_cast<uint16_t>(_a[5] << idx); \
    d##IDX[6] = (idx > 15) ? 0 : static_cast<uint16_t>(_a[6] << idx); \
    d##IDX[7] = (idx > 15) ? 0 : static_cast<uint16_t>(_a[7] << idx); \
                                                                      \
    a = load_m128i(_a);                                               \
    b = _mm_set1_epi64x(IDX);                                         \
    c = _mm_sll_epi16(a, b);                                          \
    CHECK_RESULT(VALIDATE_UINT16_M128(c, d##IDX))

    IMM_64_ITER
#undef TEST_IMPL

    return TEST_SUCCESS;
}

result_t test_mm_sll_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m128i a, b, c;
    uint8_t idx;

#define TEST_IMPL(IDX)                                                \
    uint32_t d##IDX[4];                                               \
    idx = IDX;                                                        \
    d##IDX[0] = (idx > 31) ? 0 : static_cast<uint32_t>(_a[0] << idx); \
    d##IDX[1] = (idx > 31) ? 0 : static_cast<uint32_t>(_a[1] << idx); \
    d##IDX[2] = (idx > 31) ? 0 : static_cast<uint32_t>(_a[2] << idx); \
    d##IDX[3] = (idx > 31) ? 0 : static_cast<uint32_t>(_a[3] << idx); \
                                                                      \
    a = load_m128i(_a);                                               \
    b = _mm_set1_epi64x(IDX);                                         \
    c = _mm_sll_epi32(a, b);                                          \
    CHECK_RESULT(VALIDATE_UINT32_M128(c, d##IDX))

    IMM_64_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_sll_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    __m128i a, b, c;

#define TEST_IMPL(IDX)                                                        \
    uint64_t d0##IDX = (IDX & ~63) ? 0 : static_cast<uint64_t>(_a[0] << IDX); \
    uint64_t d1##IDX = (IDX & ~63) ? 0 : static_cast<uint64_t>(_a[1] << IDX); \
                                                                              \
    a = load_m128i(_a);                                                       \
    b = _mm_set1_epi64x(IDX);                                                 \
    c = _mm_sll_epi64(a, b);                                                  \
                                                                              \
    CHECK_RESULT(validateUInt64(c, d0##IDX, d1##IDX))

    IMM_64_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_slli_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    __m128i a, c;
    uint8_t idx;
#define TEST_IMPL(IDX)                                               \
    int16_t d##IDX[8];                                               \
    idx = IDX;                                                       \
    d##IDX[0] = (idx > 15) ? 0 : static_cast<int16_t>(_a[0] << idx); \
    d##IDX[1] = (idx > 15) ? 0 : static_cast<int16_t>(_a[1] << idx); \
    d##IDX[2] = (idx > 15) ? 0 : static_cast<int16_t>(_a[2] << idx); \
    d##IDX[3] = (idx > 15) ? 0 : static_cast<int16_t>(_a[3] << idx); \
    d##IDX[4] = (idx > 15) ? 0 : static_cast<int16_t>(_a[4] << idx); \
    d##IDX[5] = (idx > 15) ? 0 : static_cast<int16_t>(_a[5] << idx); \
    d##IDX[6] = (idx > 15) ? 0 : static_cast<int16_t>(_a[6] << idx); \
    d##IDX[7] = (idx > 15) ? 0 : static_cast<int16_t>(_a[7] << idx); \
                                                                     \
    a = load_m128i(_a);                                              \
    c = _mm_slli_epi16(a, IDX);                                      \
    CHECK_RESULT(VALIDATE_INT16_M128(c, d##IDX))

    IMM_64_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_slli_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
#if defined(__clang__)
    // Clang compiler does not allow the second argument of _mm_slli_epi32() to
    // be greater than 31.
    const int count = static_cast<int>(iter % 33 - 1);  // range: -1 ~ 31
#else
    const int count = static_cast<int>(iter % 34 - 1);  // range: -1 ~ 32
#endif

    int32_t d[4];
    d[0] = (count & ~31) ? 0 : _a[0] << count;
    d[1] = (count & ~31) ? 0 : _a[1] << count;
    d[2] = (count & ~31) ? 0 : _a[2] << count;
    d[3] = (count & ~31) ? 0 : _a[3] << count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_slli_epi32(a, count);
    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_slli_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
#if defined(__clang__)
    // Clang compiler does not allow the second argument of "_mm_slli_epi64()"
    // to be greater than 63.
    const int count = static_cast<int>(iter % 65 - 1);  // range: -1 ~ 63
#else
    const int count = static_cast<int>(iter % 66 - 1);  // range: -1 ~ 64
#endif
    int64_t d[2];
    d[0] = (count & ~63) ? 0 : _a[0] << count;
    d[1] = (count & ~63) ? 0 : _a[1] << count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_slli_epi64(a, count);
    return VALIDATE_INT64_M128(c, d);
}

result_t test_mm_slli_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;

    int8_t d[16];
    int count = static_cast<int>((iter % 5) << 2);
    for (int i = 0; i < 16; i++) {
        if (i < count)
            d[i] = 0;
        else
            d[i] = (reinterpret_cast<const int8_t *>(_a))[i - count];
    }

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_setzero_si128();
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

    return VALIDATE_INT8_M128(ret, d);
}

result_t test_mm_sqrt_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    double f0 = sqrt(_a[0]);
    double f1 = sqrt(_a[1]);

    __m128d a = load_m128d(_a);
    __m128d c = _mm_sqrt_pd(a);

    return validateFloatError(c, f0, f1, 1.0e-15);
}

result_t test_mm_sqrt_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    double f0 = sqrt(_b[0]);
    double f1 = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_sqrt_sd(a, b);

    return validateFloatError(c, f0, f1, 1.0e-15);
}

result_t test_mm_sra_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int64_t count =
        static_cast<int64_t>(iter % 18 - 1);  // range: -1 ~ 16

    int16_t d[8];
    d[0] = (count & ~15) ? (_a[0] < 0 ? ~UINT16_C(0) : 0) : (_a[0] >> count);
    d[1] = (count & ~15) ? (_a[1] < 0 ? ~UINT16_C(0) : 0) : (_a[1] >> count);
    d[2] = (count & ~15) ? (_a[2] < 0 ? ~UINT16_C(0) : 0) : (_a[2] >> count);
    d[3] = (count & ~15) ? (_a[3] < 0 ? ~UINT16_C(0) : 0) : (_a[3] >> count);
    d[4] = (count & ~15) ? (_a[4] < 0 ? ~UINT16_C(0) : 0) : (_a[4] >> count);
    d[5] = (count & ~15) ? (_a[5] < 0 ? ~UINT16_C(0) : 0) : (_a[5] >> count);
    d[6] = (count & ~15) ? (_a[6] < 0 ? ~UINT16_C(0) : 0) : (_a[6] >> count);
    d[7] = (count & ~15) ? (_a[7] < 0 ? ~UINT16_C(0) : 0) : (_a[7] >> count);

    __m128i a = _mm_load_si128(reinterpret_cast<const __m128i *>(_a));
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sra_epi16(a, b);

    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_sra_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int64_t count =
        static_cast<int64_t>(iter % 34 - 1);  // range: -1 ~ 32

    int32_t d[4];
    d[0] = (count & ~31) ? (_a[0] < 0 ? -1 : 0) : _a[0] >> count;
    d[1] = (count & ~31) ? (_a[1] < 0 ? -1 : 0) : _a[1] >> count;
    d[2] = (count & ~31) ? (_a[2] < 0 ? -1 : 0) : _a[2] >> count;
    d[3] = (count & ~31) ? (_a[3] < 0 ? -1 : 0) : _a[3] >> count;

    __m128i a = _mm_load_si128(reinterpret_cast<const __m128i *>(_a));
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sra_epi32(a, b);

    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_srai_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int32_t b = static_cast<int32_t>(iter % 18 - 1);  // range: -1 ~ 16
    int16_t d[8];
    int count = (b & ~15) ? 15 : b;

    for (int i = 0; i < 8; i++) {
        d[i] = _a[i] >> count;
    }

    __m128i a = _mm_load_si128(reinterpret_cast<const __m128i *>(_a));
    __m128i c = _mm_srai_epi16(a, b);

    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_srai_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t b = static_cast<int32_t>(iter % 34 - 1);  // range: -1 ~ 32

    int32_t d[4];
    int count = (b & ~31) ? 31 : b;
    for (int i = 0; i < 4; i++) {
        d[i] = _a[i] >> count;
    }

    __m128i a = _mm_load_si128(reinterpret_cast<const __m128i *>(_a));
    __m128i c = _mm_srai_epi32(a, b);

    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_srl_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int64_t count =
        static_cast<int64_t>(iter % 18 - 1);  // range: -1 ~ 16

    uint16_t d[8];
    d[0] = (count & ~15)
               ? 0
               : static_cast<uint16_t>(static_cast<uint16_t>(_a[0]) >> count);
    d[1] = (count & ~15)
               ? 0
               : static_cast<uint16_t>(static_cast<uint16_t>(_a[1]) >> count);
    d[2] = (count & ~15)
               ? 0
               : static_cast<uint16_t>(static_cast<uint16_t>(_a[2]) >> count);
    d[3] = (count & ~15)
               ? 0
               : static_cast<uint16_t>(static_cast<uint16_t>(_a[3]) >> count);
    d[4] = (count & ~15)
               ? 0
               : static_cast<uint16_t>(static_cast<uint16_t>(_a[4]) >> count);
    d[5] = (count & ~15)
               ? 0
               : static_cast<uint16_t>(static_cast<uint16_t>(_a[5]) >> count);
    d[6] = (count & ~15)
               ? 0
               : static_cast<uint16_t>(static_cast<uint16_t>(_a[6]) >> count);
    d[7] = (count & ~15)
               ? 0
               : static_cast<uint16_t>(static_cast<uint16_t>(_a[7]) >> count);

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi16(a, b);

    return VALIDATE_UINT16_M128(c, d);
}

result_t test_mm_srl_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int64_t count =
        static_cast<int64_t>(iter % 34 - 1);  // range: -1 ~ 32

    uint32_t d[4];
    d[0] = (count & ~31)
               ? 0
               : static_cast<uint32_t>(static_cast<uint32_t>(_a[0]) >> count);
    d[1] = (count & ~31)
               ? 0
               : static_cast<uint32_t>(static_cast<uint32_t>(_a[1]) >> count);
    d[2] = (count & ~31)
               ? 0
               : static_cast<uint32_t>(static_cast<uint32_t>(_a[2]) >> count);
    d[3] = (count & ~31)
               ? 0
               : static_cast<uint32_t>(static_cast<uint32_t>(_a[3]) >> count);

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi32(a, b);

    return VALIDATE_UINT32_M128(c, d);
}

result_t test_mm_srl_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t count =
        static_cast<int64_t>(iter % 66 - 1);  // range: -1 ~ 64

    uint64_t d[2];
    d[0] = (count & ~63)
               ? 0
               : static_cast<uint64_t>(static_cast<uint64_t>(_a[0]) >> count);
    d[1] = (count & ~63)
               ? 0
               : static_cast<uint64_t>(static_cast<uint64_t>(_a[1]) >> count);

    __m128i a = load_m128i(_a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi64(a, b);

    return VALIDATE_UINT64_M128(c, d);
}

result_t test_mm_srli_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);
    const int count = static_cast<int>(iter % 18 - 1);  // range: -1 ~ 16

    int16_t d[8];
    d[0] = count & (~15) ? 0 : static_cast<int16_t>(_a[0] >> count);
    d[1] = count & (~15) ? 0 : static_cast<int16_t>(_a[1] >> count);
    d[2] = count & (~15) ? 0 : static_cast<int16_t>(_a[2] >> count);
    d[3] = count & (~15) ? 0 : static_cast<int16_t>(_a[3] >> count);
    d[4] = count & (~15) ? 0 : static_cast<int16_t>(_a[4] >> count);
    d[5] = count & (~15) ? 0 : static_cast<int16_t>(_a[5] >> count);
    d[6] = count & (~15) ? 0 : static_cast<int16_t>(_a[6] >> count);
    d[7] = count & (~15) ? 0 : static_cast<int16_t>(_a[7] >> count);

    __m128i a = load_m128i(_a);
    __m128i c = _mm_srli_epi16(a, count);

    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_srli_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);
    const int count = static_cast<int>(iter % 34 - 1);  // range: -1 ~ 32

    int32_t d[4];
    d[0] = count & (~31) ? 0 : static_cast<int32_t>(_a[0] >> count);
    d[1] = count & (~31) ? 0 : static_cast<int32_t>(_a[1] >> count);
    d[2] = count & (~31) ? 0 : static_cast<int32_t>(_a[2] >> count);
    d[3] = count & (~31) ? 0 : static_cast<int32_t>(_a[3] >> count);

    __m128i a = load_m128i(_a);
    __m128i c = _mm_srli_epi32(a, count);

    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_srli_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int count = static_cast<int>(iter % 66 - 1);  // range: -1 ~ 64

    uint64_t d[2];
    d[0] = count & (~63) ? 0 : static_cast<uint64_t>(_a[0]) >> count;
    d[1] = count & (~63) ? 0 : static_cast<uint64_t>(_a[1]) >> count;

    __m128i a = load_m128i(_a);
    __m128i c = _mm_srli_epi64(a, count);

    return VALIDATE_UINT64_M128(c, d);
}

result_t test_mm_srli_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int count = static_cast<int>((iter % 5) << 2);

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        if (i >= (16 - count))
            d[i] = 0;
        else
            d[i] = _a[i + count];
    }

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_setzero_si128();
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

    return VALIDATE_INT8_M128(ret, d);
}

result_t test_mm_store_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = reinterpret_cast<double *>(impl.mTestFloatPointer1);
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
    double *p = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double _a[2] = {static_cast<double>(impl.mTestFloats[iter]),
                    static_cast<double>(impl.mTestFloats[iter + 1])};

    __m128d a = load_m128d(_a);
    _mm_store_pd1(p, a);
    ASSERT_RETURN(p[0] == impl.mTestFloats[iter]);
    ASSERT_RETURN(p[1] == impl.mTestFloats[iter]);
    return TEST_SUCCESS;
}

result_t test_mm_store_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double _a[2] = {static_cast<double>(impl.mTestFloats[iter]),
                    static_cast<double>(impl.mTestFloats[iter + 1])};

    __m128d a = load_m128d(_a);
    _mm_store_sd(p, a);
    ASSERT_RETURN(p[0] == impl.mTestFloats[iter]);
    return TEST_SUCCESS;
}

result_t test_mm_store_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    alignas(16) int32_t p[4];

    __m128i a = load_m128i(_a);
    _mm_store_si128(reinterpret_cast<__m128i *>(p), a);

    return VALIDATE_INT32_M128(a, p);
}

result_t test_mm_store1_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_store_pd1(impl, iter);
}

result_t test_mm_storeh_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double mem;

    __m128d a = load_m128d(p);
    _mm_storeh_pd(&mem, a);

    ASSERT_RETURN(mem == p[1]);
    return TEST_SUCCESS;
}

result_t test_mm_storel_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int64_t *p = reinterpret_cast<int64_t *>(impl.mTestIntPointer1);
    __m128i mem;

    __m128i a = load_m128i(p);
    _mm_storel_epi64(&mem, a);

    ASSERT_RETURN((reinterpret_cast<SIMDVec *>(&mem))->m128_u64[0] ==
                  static_cast<uint64_t>(p[0]));
    return TEST_SUCCESS;
}

result_t test_mm_storel_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double mem;

    __m128d a = load_m128d(p);
    _mm_storel_pd(&mem, a);

    ASSERT_RETURN(mem == p[0]);
    return TEST_SUCCESS;
}

result_t test_mm_storer_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double mem[2];

    __m128d a = load_m128d(p);
    _mm_storer_pd(mem, a);

    __m128d res = load_m128d(mem);
    double d[2] = {p[1], p[0]};
    return VALIDATE_DOUBLE_M128(res, d);
}

result_t test_mm_storeu_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    double *p = reinterpret_cast<double *>(impl.mTestFloatPointer1);
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
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m128i b;
    __m128i a = load_m128i(_a);
    _mm_storeu_si128(&b, a);
    int32_t *_b = reinterpret_cast<int32_t *>(&b);
    return VALIDATE_INT32_M128(a, _b);
}

result_t test_mm_storeu_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_storeu_si32. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
    return TEST_UNIMPL;
#else
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m128i b = _mm_setzero_si128();
    __m128i a = load_m128i(_a);
    _mm_storeu_si32(&b, a);
    int32_t *_b = reinterpret_cast<int32_t *>(&b);
    return validateInt32(b, _a[0], _b[1], _b[2], _b[3]);
#endif
}

result_t test_mm_stream_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    double p[2];

    __m128d a = load_m128d(_a);
    _mm_stream_pd(p, a);

    return VALIDATE_DOUBLE_M128(a, p);
}

result_t test_mm_stream_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    alignas(16) int32_t p[4];

    __m128i a = load_m128i(_a);
    _mm_stream_si128(reinterpret_cast<__m128i *>(p), a);

    return VALIDATE_INT32_M128(a, p);
}

result_t test_mm_stream_si32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t a = static_cast<const int32_t>(impl.mTestInts[iter]);
    int32_t p;

    _mm_stream_si32(&p, a);

    ASSERT_RETURN(a == p)
    return TEST_SUCCESS;
}

result_t test_mm_stream_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t a = static_cast<const int64_t>(impl.mTestInts[iter]);
    __int64 p[1];
    _mm_stream_si64(p, a);
    ASSERT_RETURN(p[0] == a);
    return TEST_SUCCESS;
}

result_t test_mm_sub_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1] - _b[1];
    d[2] = _a[2] - _b[2];
    d[3] = _a[3] - _b[3];
    d[4] = _a[4] - _b[4];
    d[5] = _a[5] - _b[5];
    d[6] = _a[6] - _b[6];
    d[7] = _a[7] - _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sub_epi16(a, b);
    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_sub_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t d[4];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1] - _b[1];
    d[2] = _a[2] - _b[2];
    d[3] = _a[3] - _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sub_epi32(a, b);
    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_sub_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a = reinterpret_cast<int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b = reinterpret_cast<int64_t *>(impl.mTestIntPointer2);
    int64_t d[2];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1] - _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sub_epi64(a, b);
    return VALIDATE_INT64_M128(c, d);
}

result_t test_mm_sub_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    int8_t d[16];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1] - _b[1];
    d[2] = _a[2] - _b[2];
    d[3] = _a[3] - _b[3];
    d[4] = _a[4] - _b[4];
    d[5] = _a[5] - _b[5];
    d[6] = _a[6] - _b[6];
    d[7] = _a[7] - _b[7];
    d[8] = _a[8] - _b[8];
    d[9] = _a[9] - _b[9];
    d[10] = _a[10] - _b[10];
    d[11] = _a[11] - _b[11];
    d[12] = _a[12] - _b[12];
    d[13] = _a[13] - _b[13];
    d[14] = _a[14] - _b[14];
    d[15] = _a[15] - _b[15];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_sub_epi8(a, b);
    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_sub_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1] - _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_sub_pd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_sub_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_sub_sd(a, b);
    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_sub_si64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer2);

    int64_t d = _a[0] - _b[0];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_sub_si64(a, b);

    return validateInt64(c, d);
}

result_t test_mm_subs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    int16_t d[8];
    for (int i = 0; i < 8; i++) {
        d[i] = saturate_i16(static_cast<int32_t>(_a[i]) -
                            static_cast<int32_t>(_b[i]));
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_subs_epi16(a, b);

    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_subs_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        d[i] = saturate_i8(static_cast<int16_t>(_a[i]) -
                           static_cast<int16_t>(_b[i]));
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_subs_epi8(a, b);

    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_subs_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    uint16_t d[8];
    d[0] = static_cast<uint16_t>(_a[0]) - static_cast<uint16_t>(_b[0]);
    if (d[0] > static_cast<uint16_t>(_a[0]))
        d[0] = 0;
    d[1] = static_cast<uint16_t>(_a[1]) - static_cast<uint16_t>(_b[1]);
    if (d[1] > static_cast<uint16_t>(_a[1]))
        d[1] = 0;
    d[2] = static_cast<uint16_t>(_a[2]) - static_cast<uint16_t>(_b[2]);
    if (d[2] > static_cast<uint16_t>(_a[2]))
        d[2] = 0;
    d[3] = static_cast<uint16_t>(_a[3]) - static_cast<uint16_t>(_b[3]);
    if (d[3] > static_cast<uint16_t>(_a[3]))
        d[3] = 0;
    d[4] = static_cast<uint16_t>(_a[4]) - static_cast<uint16_t>(_b[4]);
    if (d[4] > static_cast<uint16_t>(_a[4]))
        d[4] = 0;
    d[5] = static_cast<uint16_t>(_a[5]) - static_cast<uint16_t>(_b[5]);
    if (d[5] > static_cast<uint16_t>(_a[5]))
        d[5] = 0;
    d[6] = static_cast<uint16_t>(_a[6]) - static_cast<uint16_t>(_b[6]);
    if (d[6] > static_cast<uint16_t>(_a[6]))
        d[6] = 0;
    d[7] = static_cast<uint16_t>(_a[7]) - static_cast<uint16_t>(_b[7]);
    if (d[7] > static_cast<uint16_t>(_a[7]))
        d[7] = 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_subs_epu16(a, b);
    return VALIDATE_UINT16_M128(c, d);
}

result_t test_mm_subs_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    uint8_t d[16];
    d[0] = static_cast<uint8_t>(_a[0]) - static_cast<uint8_t>(_b[0]);
    if (d[0] > static_cast<uint8_t>(_a[0]))
        d[0] = 0;
    d[1] = static_cast<uint8_t>(_a[1]) - static_cast<uint8_t>(_b[1]);
    if (d[1] > static_cast<uint8_t>(_a[1]))
        d[1] = 0;
    d[2] = static_cast<uint8_t>(_a[2]) - static_cast<uint8_t>(_b[2]);
    if (d[2] > static_cast<uint8_t>(_a[2]))
        d[2] = 0;
    d[3] = static_cast<uint8_t>(_a[3]) - static_cast<uint8_t>(_b[3]);
    if (d[3] > static_cast<uint8_t>(_a[3]))
        d[3] = 0;
    d[4] = static_cast<uint8_t>(_a[4]) - static_cast<uint8_t>(_b[4]);
    if (d[4] > static_cast<uint8_t>(_a[4]))
        d[4] = 0;
    d[5] = static_cast<uint8_t>(_a[5]) - static_cast<uint8_t>(_b[5]);
    if (d[5] > static_cast<uint8_t>(_a[5]))
        d[5] = 0;
    d[6] = static_cast<uint8_t>(_a[6]) - static_cast<uint8_t>(_b[6]);
    if (d[6] > static_cast<uint8_t>(_a[6]))
        d[6] = 0;
    d[7] = static_cast<uint8_t>(_a[7]) - static_cast<uint8_t>(_b[7]);
    if (d[7] > static_cast<uint8_t>(_a[7]))
        d[7] = 0;
    d[8] = static_cast<uint8_t>(_a[8]) - static_cast<uint8_t>(_b[8]);
    if (d[8] > static_cast<uint8_t>(_a[8]))
        d[8] = 0;
    d[9] = static_cast<uint8_t>(_a[9]) - static_cast<uint8_t>(_b[9]);
    if (d[9] > static_cast<uint8_t>(_a[9]))
        d[9] = 0;
    d[10] = static_cast<uint8_t>(_a[10]) - static_cast<uint8_t>(_b[10]);
    if (d[10] > static_cast<uint8_t>(_a[10]))
        d[10] = 0;
    d[11] = static_cast<uint8_t>(_a[11]) - static_cast<uint8_t>(_b[11]);
    if (d[11] > static_cast<uint8_t>(_a[11]))
        d[11] = 0;
    d[12] = static_cast<uint8_t>(_a[12]) - static_cast<uint8_t>(_b[12]);
    if (d[12] > static_cast<uint8_t>(_a[12]))
        d[12] = 0;
    d[13] = static_cast<uint8_t>(_a[13]) - static_cast<uint8_t>(_b[13]);
    if (d[13] > static_cast<uint8_t>(_a[13]))
        d[13] = 0;
    d[14] = static_cast<uint8_t>(_a[14]) - static_cast<uint8_t>(_b[14]);
    if (d[14] > static_cast<uint8_t>(_a[14]))
        d[14] = 0;
    d[15] = static_cast<uint8_t>(_a[15]) - static_cast<uint8_t>(_b[15]);
    if (d[15] > static_cast<uint8_t>(_a[15]))
        d[15] = 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_subs_epu8(a, b);
    return VALIDATE_UINT8_M128(c, d);
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

    // When SSE2NEON_UNDEFINED_ZERO is set or on MSVC, the value should be zero.
    // On GCC/Clang without the flag, the value is truly undefined so we can
    // only test that XOR-with-self produces zero.
#if SSE2NEON_UNDEFINED_ZERO || (defined(_MSC_VER) && !defined(__clang__))
    double zero[2] = {0, 0};
    result_t res = VALIDATE_DOUBLE_M128(a, zero);
    if (res != TEST_SUCCESS)
        return res;
#else
    // Use volatile barrier to prevent compiler from optimizing away the
    // undefined value. This forces materialization and avoids UB from
    // reading uninitialized memory directly.
    volatile __m128d barrier = a;
    a = barrier;
#endif

    a = _mm_xor_pd(a, a);
    double d[2] = {0, 0};
    return VALIDATE_DOUBLE_M128(a, d);
}

result_t test_mm_undefined_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    __m128i a = _mm_undefined_si128();

    // When SSE2NEON_UNDEFINED_ZERO is set or on MSVC, the value should be zero.
    // On GCC/Clang without the flag, the value is truly undefined so we can
    // only test that XOR-with-self produces zero.
#if SSE2NEON_UNDEFINED_ZERO || (defined(_MSC_VER) && !defined(__clang__))
    int64_t zero[2] = {0, 0};
    result_t res = VALIDATE_INT64_M128(a, zero);
    if (res != TEST_SUCCESS)
        return res;
#else
    // Use volatile barrier to prevent compiler from optimizing away the
    // undefined value. This forces materialization and avoids UB from
    // reading uninitialized memory directly.
    volatile __m128i barrier = a;
    a = barrier;
#endif

    a = _mm_xor_si128(a, a);
    int64_t d[2] = {0, 0};
    return VALIDATE_INT64_M128(a, d);
}

result_t test_mm_unpackhi_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    int16_t d[8];
    d[0] = _a[4];
    d[1] = _b[4];
    d[2] = _a[5];
    d[3] = _b[5];
    d[4] = _a[6];
    d[5] = _b[6];
    d[6] = _a[7];
    d[7] = _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpackhi_epi16(a, b);

    return VALIDATE_INT16_M128(ret, d);
}

result_t test_mm_unpackhi_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    int32_t d[4];
    d[0] = _a[2];
    d[1] = _b[2];
    d[2] = _a[3];
    d[3] = _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpackhi_epi32(a, b);

    return VALIDATE_INT32_M128(ret, d);
}

result_t test_mm_unpackhi_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer2);

    int64_t d[2];
    d[0] = _a[1];
    d[1] = _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpackhi_epi64(a, b);

    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_unpackhi_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);

    int8_t d[16];
    d[0] = _a[8];
    d[1] = _b[8];
    d[2] = _a[9];
    d[3] = _b[9];
    d[4] = _a[10];
    d[5] = _b[10];
    d[6] = _a[11];
    d[7] = _b[11];
    d[8] = _a[12];
    d[9] = _b[12];
    d[10] = _a[13];
    d[11] = _b[13];
    d[12] = _a[14];
    d[13] = _b[14];
    d[14] = _a[15];
    d[15] = _b[15];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpackhi_epi8(a, b);

    return VALIDATE_INT8_M128(ret, d);
}

result_t test_mm_unpackhi_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_unpackhi_pd(a, b);

    double d[2] = {_a[1], _b[1]};
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_unpacklo_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    int16_t d[8];
    d[0] = _a[0];
    d[1] = _b[0];
    d[2] = _a[1];
    d[3] = _b[1];
    d[4] = _a[2];
    d[5] = _b[2];
    d[6] = _a[3];
    d[7] = _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpacklo_epi16(a, b);

    return VALIDATE_INT16_M128(ret, d);
}

result_t test_mm_unpacklo_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    int32_t d[4];
    d[0] = _a[0];
    d[1] = _b[0];
    d[2] = _a[1];
    d[3] = _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpacklo_epi32(a, b);

    return VALIDATE_INT32_M128(ret, d);
}

result_t test_mm_unpacklo_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer2);

    int64_t d[2];
    d[0] = _a[0];
    d[1] = _b[0];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpacklo_epi64(a, b);

    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_unpacklo_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);

    int8_t d[16];
    d[0] = _a[0];
    d[1] = _b[0];
    d[2] = _a[1];
    d[3] = _b[1];
    d[4] = _a[2];
    d[5] = _b[2];
    d[6] = _a[3];
    d[7] = _b[3];
    d[8] = _a[4];
    d[9] = _b[4];
    d[10] = _a[5];
    d[11] = _b[5];
    d[12] = _a[6];
    d[13] = _b[6];
    d[14] = _a[7];
    d[15] = _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_unpacklo_epi8(a, b);

    return VALIDATE_INT8_M128(ret, d);
}

result_t test_mm_unpacklo_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_unpacklo_pd(a, b);

    double d[2] = {_a[0], _b[0]};
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_xor_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestFloatPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestFloatPointer2);
    double d[2];
    d[0] = sse2neon_tool_recast_f64(static_cast<uint64_t>(_a[0] ^ _b[0]));
    d[1] = sse2neon_tool_recast_f64(static_cast<uint64_t>(_a[1] ^ _b[1]));

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_xor_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_xor_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer2);

    int64_t d[2];
    d[0] = _a[0] ^ _b[0];
    d[1] = _a[1] ^ _b[1];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_xor_si128(a, b);

    return VALIDATE_INT64_M128(c, d);
}

/* SSE3 */
result_t test_mm_addsub_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    double d[2];
    d[0] = _a[0] - _b[0];
    d[1] = _a[1] + _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_addsub_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_addsub_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // See SSE3 rounding mode note in sse2neon.h
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f[4];
    f[0] = _a[0] - _b[0];
    f[1] = _a[1] + _b[1];
    f[2] = _a[2] - _b[2];
    f[3] = _a[3] + _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_addsub_ps(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_hadd_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    double d[2];
    d[0] = _a[0] + _a[1];
    d[1] = _b[0] + _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_hadd_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_hadd_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // See SSE3 rounding mode note in sse2neon.h
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f[4];
    f[0] = _a[0] + _a[1];
    f[1] = _a[2] + _a[3];
    f[2] = _b[0] + _b[1];
    f[3] = _b[2] + _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_hadd_ps(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_hsub_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    double d[2];
    d[0] = _a[0] - _a[1];
    d[1] = _b[0] - _b[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d c = _mm_hsub_pd(a, b);

    return VALIDATE_DOUBLE_M128(c, d);
}

result_t test_mm_hsub_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // See SSE3 rounding mode note in sse2neon.h
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f[4];
    f[0] = _a[0] - _a[1];
    f[1] = _a[2] - _a[3];
    f[2] = _b[0] - _b[1];
    f[3] = _b[2] - _b[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_hsub_ps(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_lddqu_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    return test_mm_loadu_si128(impl, iter);
}

result_t test_mm_loaddup_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *addr =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    __m128d ret = _mm_loaddup_pd(addr);

    double d[2] = {addr[0], addr[0]};
    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_monitor(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_monitor is a hint instruction with no visible side effects.
    // On ARM, it's implemented as a no-op since there's no equivalent
    // userspace address-monitoring mechanism.
    // On x86, MONITOR requires CPL0 (kernel mode) and traps in user space.
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || \
    defined(_M_ARM)
    int dummy = 42;
    _mm_monitor(&dummy, 0, 0);
    return TEST_SUCCESS;
#else
    return TEST_UNIMPL;
#endif
}

result_t test_mm_movedup_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *p = reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    __m128d a = load_m128d(p);
    __m128d b = _mm_movedup_pd(a);

    double d[2] = {p[0], p[0]};
    return VALIDATE_DOUBLE_M128(b, d);
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

result_t test_mm_mwait(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    // _mm_mwait is a hint instruction for power-saving wait states.
    // On ARM, it's implemented as yield (default), wfe, or wfi depending
    // on SSE2NEON_MWAIT_POLICY. This test validates API presence.
    // On x86, MWAIT requires CPL0 (kernel mode) and traps in user space.
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || \
    defined(_M_ARM)
    _mm_mwait(0, 0);
    return TEST_SUCCESS;
#else
    return TEST_UNIMPL;
#endif
}

/* SSSE3 */
result_t test_mm_abs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    __m128i a = load_m128i(_a);
    __m128i c = _mm_abs_epi16(a);

    uint16_t d[8];
    d[0] = static_cast<uint16_t>((_a[0] < 0) ? -_a[0] : _a[0]);
    d[1] = static_cast<uint16_t>((_a[1] < 0) ? -_a[1] : _a[1]);
    d[2] = static_cast<uint16_t>((_a[2] < 0) ? -_a[2] : _a[2]);
    d[3] = static_cast<uint16_t>((_a[3] < 0) ? -_a[3] : _a[3]);
    d[4] = static_cast<uint16_t>((_a[4] < 0) ? -_a[4] : _a[4]);
    d[5] = static_cast<uint16_t>((_a[5] < 0) ? -_a[5] : _a[5]);
    d[6] = static_cast<uint16_t>((_a[6] < 0) ? -_a[6] : _a[6]);
    d[7] = static_cast<uint16_t>((_a[7] < 0) ? -_a[7] : _a[7]);

    return VALIDATE_UINT16_M128(c, d);
}

result_t test_mm_abs_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m128i a = load_m128i(_a);
    __m128i c = _mm_abs_epi32(a);

    uint32_t d[4];
    d[0] = static_cast<uint32_t>((_a[0] < 0) ? -_a[0] : _a[0]);
    d[1] = static_cast<uint32_t>((_a[1] < 0) ? -_a[1] : _a[1]);
    d[2] = static_cast<uint32_t>((_a[2] < 0) ? -_a[2] : _a[2]);
    d[3] = static_cast<uint32_t>((_a[3] < 0) ? -_a[3] : _a[3]);

    return VALIDATE_UINT32_M128(c, d);
}

result_t test_mm_abs_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    __m128i a = load_m128i(_a);
    __m128i c = _mm_abs_epi8(a);

    uint8_t d[16];
    for (int i = 0; i < 16; i++) {
        d[i] = static_cast<uint8_t>((_a[i] < 0) ? -_a[i] : _a[i]);
    }

    return VALIDATE_UINT8_M128(c, d);
}

result_t test_mm_abs_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    __m64 a = load_m64(_a);
    __m64 c = _mm_abs_pi16(a);

    uint16_t d[4];
    d[0] = static_cast<uint16_t>((_a[0] < 0) ? -_a[0] : _a[0]);
    d[1] = static_cast<uint16_t>((_a[1] < 0) ? -_a[1] : _a[1]);
    d[2] = static_cast<uint16_t>((_a[2] < 0) ? -_a[2] : _a[2]);
    d[3] = static_cast<uint16_t>((_a[3] < 0) ? -_a[3] : _a[3]);

    return VALIDATE_UINT16_M64(c, d);
}

result_t test_mm_abs_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m64 a = load_m64(_a);
    __m64 c = _mm_abs_pi32(a);

    uint32_t d[2];
    d[0] = static_cast<uint32_t>((_a[0] < 0) ? -_a[0] : _a[0]);
    d[1] = static_cast<uint32_t>((_a[1] < 0) ? -_a[1] : _a[1]);

    return VALIDATE_UINT32_M64(c, d);
}

result_t test_mm_abs_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    __m64 a = load_m64(_a);
    __m64 c = _mm_abs_pi8(a);

    uint8_t d[8];
    d[0] = static_cast<uint8_t>((_a[0] < 0) ? -_a[0] : _a[0]);
    d[1] = static_cast<uint8_t>((_a[1] < 0) ? -_a[1] : _a[1]);
    d[2] = static_cast<uint8_t>((_a[2] < 0) ? -_a[2] : _a[2]);
    d[3] = static_cast<uint8_t>((_a[3] < 0) ? -_a[3] : _a[3]);
    d[4] = static_cast<uint8_t>((_a[4] < 0) ? -_a[4] : _a[4]);
    d[5] = static_cast<uint8_t>((_a[5] < 0) ? -_a[5] : _a[5]);
    d[6] = static_cast<uint8_t>((_a[6] < 0) ? -_a[6] : _a[6]);
    d[7] = static_cast<uint8_t>((_a[7] < 0) ? -_a[7] : _a[7]);

    return VALIDATE_UINT8_M64(c, d);
}

result_t test_mm_alignr_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    unsigned int shift = (iter % 5) << 3;
    uint8_t d[32];

    if (shift >= 32) {
        memset(static_cast<void *>(d), 0, sizeof(d));
    } else {
        memcpy(static_cast<void *>(d), static_cast<const void *>(_b), 16);
        memcpy(static_cast<void *>(d + 16), static_cast<const void *>(_a), 16);
        // shifting
        for (size_t x = 0; x < sizeof(d); x++) {
            if (x + shift >= sizeof(d))
                d[x] = 0;
            else
                d[x] = d[x + shift];
        }
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_setzero_si128();
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

    return VALIDATE_UINT8_M128(ret, d);
}

result_t test_mm_alignr_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);
    unsigned int shift = (iter % 3) << 3;
    uint8_t d[16];

    if (shift >= 16) {
        memset(static_cast<void *>(d), 0, sizeof(d));
    } else {
        memcpy(static_cast<void *>(d), static_cast<const void *>(_b), 8);
        memcpy(static_cast<void *>(d + 8), static_cast<const void *>(_a), 8);
        // shifting
        for (size_t x = 0; x < sizeof(d); x++) {
            if (x + shift >= sizeof(d))
                d[x] = 0;
            else
                d[x] = d[x + shift];
        }
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    uint8_t zeros[] = {0, 0, 0, 0, 0, 0, 0, 0};
    __m64 ret = load_m64(zeros);
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

    return VALIDATE_UINT8_M64(ret, d);
}

result_t test_mm_hadd_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[8];
    d[0] = _a[0] + _a[1];
    d[1] = _a[2] + _a[3];
    d[2] = _a[4] + _a[5];
    d[3] = _a[6] + _a[7];
    d[4] = _b[0] + _b[1];
    d[5] = _b[2] + _b[3];
    d[6] = _b[4] + _b[5];
    d[7] = _b[6] + _b[7];
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_hadd_epi16(a, b);
    return VALIDATE_INT16_M128(ret, d);
}

result_t test_mm_hadd_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);
    int32_t d[4];
    d[0] = _a[0] + _a[1];
    d[1] = _a[2] + _a[3];
    d[2] = _b[0] + _b[1];
    d[3] = _b[2] + _b[3];
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i ret = _mm_hadd_epi32(a, b);
    return VALIDATE_INT32_M128(ret, d);
}

result_t test_mm_hadd_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t d[4];
    d[0] = _a[0] + _a[1];
    d[1] = _a[2] + _a[3];
    d[2] = _b[0] + _b[1];
    d[3] = _b[2] + _b[3];
    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_hadd_pi16(a, b);
    return VALIDATE_INT16_M64(ret, d);
}

result_t test_mm_hadd_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);
    int32_t d[2];
    d[0] = _a[0] + _a[1];
    d[1] = _b[0] + _b[1];
    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 ret = _mm_hadd_pi32(a, b);
    return VALIDATE_INT32_M64(ret, d);
}

result_t test_mm_hadds_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    int16_t d16[8];
    int32_t d32[8];
    d32[0] = static_cast<int32_t>(_a[0]) + static_cast<int32_t>(_a[1]);
    d32[1] = static_cast<int32_t>(_a[2]) + static_cast<int32_t>(_a[3]);
    d32[2] = static_cast<int32_t>(_a[4]) + static_cast<int32_t>(_a[5]);
    d32[3] = static_cast<int32_t>(_a[6]) + static_cast<int32_t>(_a[7]);
    d32[4] = static_cast<int32_t>(_b[0]) + static_cast<int32_t>(_b[1]);
    d32[5] = static_cast<int32_t>(_b[2]) + static_cast<int32_t>(_b[3]);
    d32[6] = static_cast<int32_t>(_b[4]) + static_cast<int32_t>(_b[5]);
    d32[7] = static_cast<int32_t>(_b[6]) + static_cast<int32_t>(_b[7]);
    for (int i = 0; i < 8; i++) {
        if (d32[i] > static_cast<int32_t>(INT16_MAX))
            d16[i] = INT16_MAX;
        else if (d32[i] < static_cast<int32_t>(INT16_MIN))
            d16[i] = INT16_MIN;
        else
            d16[i] = static_cast<int16_t>(d32[i]);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_hadds_epi16(a, b);

    return VALIDATE_INT16_M128(c, d16);
}

result_t test_mm_hadds_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    int16_t d16[8];
    int32_t d32[8];
    d32[0] = static_cast<int32_t>(_a[0]) + static_cast<int32_t>(_a[1]);
    d32[1] = static_cast<int32_t>(_a[2]) + static_cast<int32_t>(_a[3]);
    d32[2] = static_cast<int32_t>(_b[0]) + static_cast<int32_t>(_b[1]);
    d32[3] = static_cast<int32_t>(_b[2]) + static_cast<int32_t>(_b[3]);
    for (int i = 0; i < 8; i++) {
        if (d32[i] > static_cast<int32_t>(INT16_MAX))
            d16[i] = INT16_MAX;
        else if (d32[i] < static_cast<int32_t>(INT16_MIN))
            d16[i] = INT16_MIN;
        else
            d16[i] = static_cast<int16_t>(d32[i]);
    }

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_hadds_pi16(a, b);

    return VALIDATE_INT16_M64(c, d16);
}

result_t test_mm_hsub_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    int16_t d[8];
    d[0] = _a[0] - _a[1];
    d[1] = _a[2] - _a[3];
    d[2] = _a[4] - _a[5];
    d[3] = _a[6] - _a[7];
    d[4] = _b[0] - _b[1];
    d[5] = _b[2] - _b[3];
    d[6] = _b[4] - _b[5];
    d[7] = _b[6] - _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_hsub_epi16(a, b);

    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_hsub_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer1;

    int32_t d[4];
    d[0] = _a[0] - _a[1];
    d[1] = _a[2] - _a[3];
    d[2] = _b[0] - _b[1];
    d[3] = _b[2] - _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_hsub_epi32(a, b);

    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_hsub_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    int16_t d[4];
    d[0] = _a[0] - _a[1];
    d[1] = _a[2] - _a[3];
    d[2] = _b[0] - _b[1];
    d[3] = _b[2] - _b[3];
    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_hsub_pi16(a, b);

    return VALIDATE_INT16_M64(c, d);
}

result_t test_mm_hsub_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;

    int32_t d[2];
    d[0] = _a[0] - _a[1];
    d[1] = _b[0] - _b[1];

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_hsub_pi32(a, b);

    return VALIDATE_INT32_M64(c, d);
}

result_t test_mm_hsubs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    int16_t d16[8];
    d16[0] =
        saturate_i16(static_cast<int32_t>(_a[0]) - static_cast<int32_t>(_a[1]));
    d16[1] =
        saturate_i16(static_cast<int32_t>(_a[2]) - static_cast<int32_t>(_a[3]));
    d16[2] =
        saturate_i16(static_cast<int32_t>(_a[4]) - static_cast<int32_t>(_a[5]));
    d16[3] =
        saturate_i16(static_cast<int32_t>(_a[6]) - static_cast<int32_t>(_a[7]));
    d16[4] =
        saturate_i16(static_cast<int32_t>(_b[0]) - static_cast<int32_t>(_b[1]));
    d16[5] =
        saturate_i16(static_cast<int32_t>(_b[2]) - static_cast<int32_t>(_b[3]));
    d16[6] =
        saturate_i16(static_cast<int32_t>(_b[4]) - static_cast<int32_t>(_b[5]));
    d16[7] =
        saturate_i16(static_cast<int32_t>(_b[6]) - static_cast<int32_t>(_b[7]));

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_hsubs_epi16(a, b);

    return VALIDATE_INT16_M128(c, d16);
}

result_t test_mm_hsubs_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    int16_t _d[4];
    _d[0] =
        saturate_i16(static_cast<int32_t>(_a[0]) - static_cast<int32_t>(_a[1]));
    _d[1] =
        saturate_i16(static_cast<int32_t>(_a[2]) - static_cast<int32_t>(_a[3]));
    _d[2] =
        saturate_i16(static_cast<int32_t>(_b[0]) - static_cast<int32_t>(_b[1]));
    _d[3] =
        saturate_i16(static_cast<int32_t>(_b[2]) - static_cast<int32_t>(_b[3]));

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_hsubs_pi16(a, b);

    return VALIDATE_INT16_M64(c, _d);
}

result_t test_mm_maddubs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    int32_t d0 = static_cast<int32_t>(_a[0] * _b[0]);
    int32_t d1 = static_cast<int32_t>(_a[1] * _b[1]);
    int32_t d2 = static_cast<int32_t>(_a[2] * _b[2]);
    int32_t d3 = static_cast<int32_t>(_a[3] * _b[3]);
    int32_t d4 = static_cast<int32_t>(_a[4] * _b[4]);
    int32_t d5 = static_cast<int32_t>(_a[5] * _b[5]);
    int32_t d6 = static_cast<int32_t>(_a[6] * _b[6]);
    int32_t d7 = static_cast<int32_t>(_a[7] * _b[7]);
    int32_t d8 = static_cast<int32_t>(_a[8] * _b[8]);
    int32_t d9 = static_cast<int32_t>(_a[9] * _b[9]);
    int32_t d10 = static_cast<int32_t>(_a[10] * _b[10]);
    int32_t d11 = static_cast<int32_t>(_a[11] * _b[11]);
    int32_t d12 = static_cast<int32_t>(_a[12] * _b[12]);
    int32_t d13 = static_cast<int32_t>(_a[13] * _b[13]);
    int32_t d14 = static_cast<int32_t>(_a[14] * _b[14]);
    int32_t d15 = static_cast<int32_t>(_a[15] * _b[15]);

    int16_t e[8];
    e[0] = saturate_i16(d0 + d1);
    e[1] = saturate_i16(d2 + d3);
    e[2] = saturate_i16(d4 + d5);
    e[3] = saturate_i16(d6 + d7);
    e[4] = saturate_i16(d8 + d9);
    e[5] = saturate_i16(d10 + d11);
    e[6] = saturate_i16(d12 + d13);
    e[7] = saturate_i16(d14 + d15);

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_maddubs_epi16(a, b);
    return VALIDATE_INT16_M128(c, e);
}

result_t test_mm_maddubs_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    int16_t d0 = static_cast<int16_t>(_a[0] * _b[0]);
    int16_t d1 = static_cast<int16_t>(_a[1] * _b[1]);
    int16_t d2 = static_cast<int16_t>(_a[2] * _b[2]);
    int16_t d3 = static_cast<int16_t>(_a[3] * _b[3]);
    int16_t d4 = static_cast<int16_t>(_a[4] * _b[4]);
    int16_t d5 = static_cast<int16_t>(_a[5] * _b[5]);
    int16_t d6 = static_cast<int16_t>(_a[6] * _b[6]);
    int16_t d7 = static_cast<int16_t>(_a[7] * _b[7]);

    int16_t e[4];
    e[0] = saturate_i16(d0 + d1);
    e[1] = saturate_i16(d2 + d3);
    e[2] = saturate_i16(d4 + d5);
    e[3] = saturate_i16(d6 + d7);

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    __m64 c = _mm_maddubs_pi16(a, b);

    return VALIDATE_INT16_M64(c, e);
}

result_t test_mm_mulhrs_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    int16_t _c[8];
    for (int i = 0; i < 8; i++) {
        _c[i] = static_cast<int16_t>(
            ((((static_cast<int32_t>(_a[i]) * static_cast<int32_t>(_b[i])) >>
               14) +
              1) &
             0x1FFFE) >>
            1);
    }
    __m128i c = _mm_mulhrs_epi16(a, b);

    return VALIDATE_INT16_M128(c, _c);
}

result_t test_mm_mulhrs_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

    __m64 a = load_m64(_a);
    __m64 b = load_m64(_b);
    int16_t _c[4];
    for (int i = 0; i < 4; i++) {
        _c[i] = static_cast<int16_t>(
            ((((static_cast<int32_t>(_a[i]) * static_cast<int32_t>(_b[i])) >>
               14) +
              1) &
             0x1FFFE) >>
            1);
    }
    __m64 c = _mm_mulhrs_pi16(a, b);

    return VALIDATE_INT16_M64(c, _c);
}

result_t test_mm_shuffle_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
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

    return VALIDATE_INT8_M128(ret, dst);
}

result_t test_mm_shuffle_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
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

    return VALIDATE_INT8_M64(ret, dst);
}

result_t test_mm_sign_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

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

    return VALIDATE_INT16_M128(c, d);
}

result_t test_mm_sign_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

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

    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_sign_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);

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

    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_sign_pi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);

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

    return VALIDATE_INT16_M64(c, d);
}

result_t test_mm_sign_pi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

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

    return VALIDATE_INT32_M64(c, d);
}

result_t test_mm_sign_pi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);

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

    return VALIDATE_INT8_M64(c, d);
}

/* SSE4.1 */
result_t test_mm_blend_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    const int16_t *_b =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer2);
    int16_t _c[8];
    __m128i a, b, c;

#define TEST_IMPL(IDX)              \
    for (int j = 0; j < 8; j++) {   \
        if ((IDX >> j) & 0x1) {     \
            _c[j] = _b[j];          \
        } else {                    \
            _c[j] = _a[j];          \
        }                           \
    }                               \
    a = load_m128i(_a);             \
    b = load_m128i(_b);             \
    c = _mm_blend_epi16(a, b, IDX); \
    CHECK_RESULT(VALIDATE_INT16_M128(c, _c));

    IMM_256_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_blend_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    __m128d a, b, c;

#define TEST_IMPL(IDX)            \
    double _c##IDX[2];            \
    for (int j = 0; j < 2; j++) { \
        if ((IDX >> j) & 0x1) {   \
            _c##IDX[j] = _b[j];   \
        } else {                  \
            _c##IDX[j] = _a[j];   \
        }                         \
    }                             \
                                  \
    a = load_m128d(_a);           \
    b = load_m128d(_b);           \
    c = _mm_blend_pd(a, b, IDX);  \
    CHECK_RESULT(validateDouble(c, _c##IDX[0], _c##IDX[1]))

    IMM_4_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_blend_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c;

    // gcc and clang can't compile call to _mm_blend_ps with 3rd argument as
    // integer type due 4 bit size limitation.
#define TEST_IMPL(IDX)            \
    float _c##IDX[4];             \
    for (int i = 0; i < 4; i++) { \
        if (IDX & (1 << i)) {     \
            _c##IDX[i] = _b[i];   \
        } else {                  \
            _c##IDX[i] = _a[i];   \
        }                         \
    }                             \
                                  \
    c = _mm_blend_ps(a, b, IDX);  \
    CHECK_RESULT(                 \
        validateFloat(c, _c##IDX[0], _c##IDX[1], _c##IDX[2], _c##IDX[3]))

    IMM_4_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_blendv_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    const int8_t _mask[16] = {
        static_cast<const int8_t>(impl.mTestInts[iter]),
        static_cast<const int8_t>(impl.mTestInts[iter + 1]),
        static_cast<const int8_t>(impl.mTestInts[iter + 2]),
        static_cast<const int8_t>(impl.mTestInts[iter + 3]),
        static_cast<const int8_t>(impl.mTestInts[iter + 4]),
        static_cast<const int8_t>(impl.mTestInts[iter + 5]),
        static_cast<const int8_t>(impl.mTestInts[iter + 6]),
        static_cast<const int8_t>(impl.mTestInts[iter + 7])};

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

    return VALIDATE_INT8_M128(c, _c);
}

result_t test_mm_blendv_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);
    const double _mask[] = {static_cast<double>(impl.mTestFloats[iter]),
                            static_cast<double>(impl.mTestFloats[iter + 1])};

    double _c[2];
    for (int i = 0; i < 2; i++) {
        // signed shift right would return a result which is either all 1's from
        // negative numbers or all 0's from positive numbers
        int64_t m = sse2neon_tool_recast_i64(_mask[i]);
        if (m >> 63) {
            _c[i] = _b[i];
        } else {
            _c[i] = _a[i];
        }
    }

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d mask = load_m128d(_mask);

    __m128d c = _mm_blendv_pd(a, b, mask);

    return VALIDATE_DOUBLE_M128(c, _c);
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
        if ((*reinterpret_cast<const int32_t *>(_mask + i)) >> 31) {
            _c[i] = _b[i];
        } else {
            _c[i] = _a[i];
        }
    }

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 mask = load_m128(_mask);

    __m128 c = _mm_blendv_ps(a, b, mask);

    return VALIDATE_FLOAT_M128(c, _c);
}

result_t test_mm_ceil_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    double d[2];
    d[0] = ceil(_a[0]);
    d[1] = ceil(_a[1]);

    __m128d a = load_m128d(_a);
    __m128d ret = _mm_ceil_pd(a);

    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_ceil_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    float d[4];
    d[0] = ceilf(_a[0]);
    d[1] = ceilf(_a[1]);
    d[2] = ceilf(_a[2]);
    d[3] = ceilf(_a[3]);

    __m128 a = _mm_load_ps(_a);
    __m128 c = _mm_ceil_ps(a);
    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_ceil_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    double d[2];
    d[0] = ceil(_b[0]);
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_ceil_sd(a, b);

    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_ceil_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f[4];
    f[0] = ceilf(_b[0]);
    f[1] = _a[1];
    f[2] = _a[2];
    f[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_ceil_ss(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_cmpeq_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer2);
    int64_t d[2];
    d[0] = (_a[0] == _b[0]) ? -1 : 0;
    d[1] = (_a[1] == _b[1]) ? -1 : 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_cmpeq_epi64(a, b);
    return VALIDATE_INT64_M128(c, d);
}

result_t test_mm_cvtepi16_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    int32_t d[4];
    d[0] = static_cast<int32_t>(_a[0]);
    d[1] = static_cast<int32_t>(_a[1]);
    d[2] = static_cast<int32_t>(_a[2]);
    d[3] = static_cast<int32_t>(_a[3]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi16_epi32(a);

    return VALIDATE_INT32_M128(ret, d);
}

result_t test_mm_cvtepi16_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);

    int64_t d[2];
    d[0] = static_cast<int64_t>(_a[0]);
    d[1] = static_cast<int64_t>(_a[1]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi16_epi64(a);

    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_cvtepi32_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);

    int64_t d[2];
    d[0] = static_cast<int64_t>(_a[0]);
    d[1] = static_cast<int64_t>(_a[1]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi32_epi64(a);

    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_cvtepi8_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);

    int16_t d[8];
    d[0] = static_cast<int16_t>(_a[0]);
    d[1] = static_cast<int16_t>(_a[1]);
    d[2] = static_cast<int16_t>(_a[2]);
    d[3] = static_cast<int16_t>(_a[3]);
    d[4] = static_cast<int16_t>(_a[4]);
    d[5] = static_cast<int16_t>(_a[5]);
    d[6] = static_cast<int16_t>(_a[6]);
    d[7] = static_cast<int16_t>(_a[7]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi8_epi16(a);

    return VALIDATE_INT16_M128(ret, d);
}

result_t test_mm_cvtepi8_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);

    int32_t d[4];
    d[0] = static_cast<int32_t>(_a[0]);
    d[1] = static_cast<int32_t>(_a[1]);
    d[2] = static_cast<int32_t>(_a[2]);
    d[3] = static_cast<int32_t>(_a[3]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi8_epi32(a);

    return VALIDATE_INT32_M128(ret, d);
}

result_t test_mm_cvtepi8_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);

    int64_t d[2];
    d[0] = static_cast<int64_t>(_a[0]);
    d[1] = static_cast<int64_t>(_a[1]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepi8_epi64(a);

    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_cvtepu16_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);

    int32_t d[4];
    d[0] = static_cast<int32_t>(_a[0]);
    d[1] = static_cast<int32_t>(_a[1]);
    d[2] = static_cast<int32_t>(_a[2]);
    d[3] = static_cast<int32_t>(_a[3]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu16_epi32(a);

    return VALIDATE_INT32_M128(ret, d);
}

result_t test_mm_cvtepu16_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);

    int64_t d[2];
    d[0] = static_cast<int64_t>(_a[0]);
    d[1] = static_cast<int64_t>(_a[1]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu16_epi64(a);

    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_cvtepu32_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);

    int64_t d[2];
    d[0] = static_cast<int64_t>(_a[0]);
    d[1] = static_cast<int64_t>(_a[1]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu32_epi64(a);

    return VALIDATE_INT64_M128(ret, d);
}

result_t test_mm_cvtepu8_epi16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);

    int16_t d[8];
    d[0] = static_cast<int16_t>(_a[0]);
    d[1] = static_cast<int16_t>(_a[1]);
    d[2] = static_cast<int16_t>(_a[2]);
    d[3] = static_cast<int16_t>(_a[3]);
    d[4] = static_cast<int16_t>(_a[4]);
    d[5] = static_cast<int16_t>(_a[5]);
    d[6] = static_cast<int16_t>(_a[6]);
    d[7] = static_cast<int16_t>(_a[7]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu8_epi16(a);

    return VALIDATE_INT16_M128(ret, d);
}

result_t test_mm_cvtepu8_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);

    int32_t d[4];
    d[0] = static_cast<int32_t>(_a[0]);
    d[1] = static_cast<int32_t>(_a[1]);
    d[2] = static_cast<int32_t>(_a[2]);
    d[3] = static_cast<int32_t>(_a[3]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu8_epi32(a);

    return VALIDATE_INT32_M128(ret, d);
}

result_t test_mm_cvtepu8_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);

    int64_t d[2];
    d[0] = static_cast<int64_t>(_a[0]);
    d[1] = static_cast<int64_t>(_a[1]);

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_cvtepu8_epi64(a);

    return VALIDATE_INT64_M128(ret, d);
}

#define MM_DP_PD_TEST_CASE_WITH(imm8)                            \
    do {                                                         \
        const double _a[] = {impl.mTestFloatPointer1[0],         \
                             impl.mTestFloatPointer1[1]};        \
        const double _b[] = {impl.mTestFloatPointer2[0],         \
                             impl.mTestFloatPointer2[1]};        \
        const int imm = imm8;                                    \
        double d[2] = {0};                                       \
        double sum = 0;                                          \
        for (size_t i = 0; i < 2; i++)                           \
            sum += ((imm) & (1 << (i + 4))) ? _a[i] * _b[i] : 0; \
        for (size_t i = 0; i < 2; i++)                           \
            d[i] = (imm & (1 << i)) ? sum : 0;                   \
        __m128d a = load_m128d(_a);                              \
        __m128d b = load_m128d(_b);                              \
        __m128d ret = _mm_dp_pd(a, b, imm);                      \
        if (validateDouble(ret, d[0], d[1]) != TEST_SUCCESS)     \
            return TEST_FAIL;                                    \
    } while (0)

#define GENERATE_MM_DP_PD_TEST_CASES \
    MM_DP_PD_TEST_CASE_WITH(0xF0);   \
    MM_DP_PD_TEST_CASE_WITH(0xF1);   \
    MM_DP_PD_TEST_CASE_WITH(0xF2);   \
    MM_DP_PD_TEST_CASE_WITH(0xFF);   \
    MM_DP_PD_TEST_CASE_WITH(0x10);   \
    MM_DP_PD_TEST_CASE_WITH(0x11);   \
    MM_DP_PD_TEST_CASE_WITH(0x12);   \
    MM_DP_PD_TEST_CASE_WITH(0x13);   \
    MM_DP_PD_TEST_CASE_WITH(0x00);   \
    MM_DP_PD_TEST_CASE_WITH(0x01);   \
    MM_DP_PD_TEST_CASE_WITH(0x02);   \
    MM_DP_PD_TEST_CASE_WITH(0x03);   \
    MM_DP_PD_TEST_CASE_WITH(0x20);   \
    MM_DP_PD_TEST_CASE_WITH(0x21);   \
    MM_DP_PD_TEST_CASE_WITH(0x22);   \
    MM_DP_PD_TEST_CASE_WITH(0x23);

OPTNONE result_t test_mm_dp_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_DP_PD_TEST_CASES
    return TEST_SUCCESS;
}

#define MM_DP_PS_TEST_CASE_WITH(IMM)                                          \
    do {                                                                      \
        const float *_a = impl.mTestFloatPointer1;                            \
        const float *_b = impl.mTestFloatPointer2;                            \
        const int imm = IMM;                                                  \
        __m128 a = load_m128(_a);                                             \
        __m128 b = load_m128(_b);                                             \
        __m128 out = _mm_dp_ps(a, b, imm);                                    \
        float r[4]; /* the reference */                                       \
        float sum = 0;                                                        \
        for (size_t i = 0; i < 4; i++)                                        \
            sum += ((imm) & (1 << (i + 4))) ? _a[i] * _b[i] : 0;              \
        for (size_t i = 0; i < 4; i++)                                        \
            r[i] = (imm & (1 << i)) ? sum : 0;                                \
        /* the epsilon has to be large enough, otherwise test suite fails. */ \
        if (validateFloatEpsilon(out, r[0], r[1], r[2], r[3], 2050.0f) !=     \
            TEST_SUCCESS)                                                     \
            return TEST_FAIL;                                                 \
    } while (0)

// Test cases organized by category for comprehensive coverage:
// - Fast path cases (0xFF, 0x7F)
// - Single input lane cases (0x1*, 0x2*, 0x4*, 0x8*)
// - Single output lane cases (0xF1, 0xF2, 0xF4, 0xF8)
// - Mixed input/output patterns
// - Edge cases (0x00, 0x0F, 0xF0)
#define GENERATE_MM_DP_PS_TEST_CASES     \
    /* Fast paths */                     \
    MM_DP_PS_TEST_CASE_WITH(0xFF);       \
    MM_DP_PS_TEST_CASE_WITH(0x7F);       \
    /* All inputs, single output lane */ \
    MM_DP_PS_TEST_CASE_WITH(0xF1);       \
    MM_DP_PS_TEST_CASE_WITH(0xF2);       \
    MM_DP_PS_TEST_CASE_WITH(0xF4);       \
    MM_DP_PS_TEST_CASE_WITH(0xF8);       \
    /* Single input lane, all outputs */ \
    MM_DP_PS_TEST_CASE_WITH(0x1F);       \
    MM_DP_PS_TEST_CASE_WITH(0x2F);       \
    MM_DP_PS_TEST_CASE_WITH(0x4F);       \
    MM_DP_PS_TEST_CASE_WITH(0x8F);       \
    /* Two input lanes */                \
    MM_DP_PS_TEST_CASE_WITH(0x3F);       \
    MM_DP_PS_TEST_CASE_WITH(0x5F);       \
    MM_DP_PS_TEST_CASE_WITH(0x6F);       \
    MM_DP_PS_TEST_CASE_WITH(0x9F);       \
    MM_DP_PS_TEST_CASE_WITH(0xAF);       \
    MM_DP_PS_TEST_CASE_WITH(0xCF);       \
    /* Three input lanes (not 0x7F) */   \
    MM_DP_PS_TEST_CASE_WITH(0xBF);       \
    MM_DP_PS_TEST_CASE_WITH(0xDF);       \
    MM_DP_PS_TEST_CASE_WITH(0xEF);       \
    /* Mixed input/output patterns */    \
    MM_DP_PS_TEST_CASE_WITH(0x23);       \
    MM_DP_PS_TEST_CASE_WITH(0x31);       \
    MM_DP_PS_TEST_CASE_WITH(0x42);       \
    MM_DP_PS_TEST_CASE_WITH(0x54);       \
    MM_DP_PS_TEST_CASE_WITH(0x68);       \
    MM_DP_PS_TEST_CASE_WITH(0x71);       \
    MM_DP_PS_TEST_CASE_WITH(0x8C);       \
    MM_DP_PS_TEST_CASE_WITH(0x99);       \
    MM_DP_PS_TEST_CASE_WITH(0xA5);       \
    MM_DP_PS_TEST_CASE_WITH(0xB5);       \
    MM_DP_PS_TEST_CASE_WITH(0xC3);       \
    MM_DP_PS_TEST_CASE_WITH(0xD2);       \
    MM_DP_PS_TEST_CASE_WITH(0xE1);       \
    /* Edge cases */                     \
    MM_DP_PS_TEST_CASE_WITH(0x00);       \
    MM_DP_PS_TEST_CASE_WITH(0x0F);       \
    MM_DP_PS_TEST_CASE_WITH(0xF0);

OPTNONE result_t test_mm_dp_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_DP_PS_TEST_CASES
    return TEST_SUCCESS;
}

result_t test_mm_extract_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t *_a = reinterpret_cast<int32_t *>(impl.mTestIntPointer1);
    __m128i a = load_m128i(_a);
    int c;

#define TEST_IMPL(IDX)             \
    c = _mm_extract_epi32(a, IDX); \
    ASSERT_RETURN(c == *(_a + IDX));

    IMM_4_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_extract_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int64_t *_a = reinterpret_cast<int64_t *>(impl.mTestIntPointer1);
    __m128i a = load_m128i(_a);
    __int64 c;

#define TEST_IMPL(IDX)             \
    c = _mm_extract_epi64(a, IDX); \
    ASSERT_RETURN(c == *(_a + IDX));

    IMM_2_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_extract_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint8_t *_a = reinterpret_cast<uint8_t *>(impl.mTestIntPointer1);
    __m128i a = load_m128i(_a);
    int c;

#define TEST_IMPL(IDX)            \
    c = _mm_extract_epi8(a, IDX); \
    ASSERT_RETURN(c == *(_a + IDX));

    IMM_8_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_extract_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = reinterpret_cast<const float *>(impl.mTestFloatPointer1);

    __m128 a = _mm_load_ps(_a);
    int32_t c;

#define TEST_IMPL(IDX)          \
    c = _mm_extract_ps(a, IDX); \
    ASSERT_RETURN(c == *reinterpret_cast<const int32_t *>(_a + IDX));

    IMM_4_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_floor_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);

    double d[2];
    d[0] = floor(_a[0]);
    d[1] = floor(_a[1]);

    __m128d a = load_m128d(_a);
    __m128d ret = _mm_floor_pd(a);

    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_floor_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    float d[4];
    d[0] = floorf(_a[0]);
    d[1] = floorf(_a[1]);
    d[2] = floorf(_a[2]);
    d[3] = floorf(_a[3]);

    __m128 a = load_m128(_a);
    __m128 c = _mm_floor_ps(a);
    return VALIDATE_FLOAT_M128(c, d);
}

result_t test_mm_floor_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a =
        reinterpret_cast<const double *>(impl.mTestFloatPointer1);
    const double *_b =
        reinterpret_cast<const double *>(impl.mTestFloatPointer2);

    double d[2];
    d[0] = floor(_b[0]);
    d[1] = _a[1];

    __m128d a = load_m128d(_a);
    __m128d b = load_m128d(_b);
    __m128d ret = _mm_floor_sd(a, b);

    return VALIDATE_DOUBLE_M128(ret, d);
}

result_t test_mm_floor_ss(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f[4];
    f[0] = floorf(_b[0]);
    f[1] = _a[1];
    f[2] = _a[2];
    f[3] = _a[3];

    __m128 a = load_m128(_a);
    __m128 b = load_m128(_b);
    __m128 c = _mm_floor_ss(a, b);

    return VALIDATE_FLOAT_M128(c, f);
}

result_t test_mm_insert_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t insert = static_cast<int32_t>(*impl.mTestIntPointer2);
    __m128i a, b;

#define TEST_IMPL(IDX)                                      \
    int32_t d##IDX[4];                                      \
    for (int i = 0; i < 4; i++) {                           \
        d##IDX[i] = _a[i];                                  \
    }                                                       \
    d##IDX[IDX] = insert;                                   \
                                                            \
    a = load_m128i(_a);                                     \
    b = _mm_insert_epi32(a, static_cast<int>(insert), IDX); \
    CHECK_RESULT(VALIDATE_INT32_M128(b, d##IDX));

    IMM_4_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_insert_epi64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    int64_t insert = static_cast<int64_t>(*impl.mTestIntPointer2);

    __m128i a, b;
    int64_t d[2];
#define TEST_IMPL(IDX)                    \
    d[0] = _a[0];                         \
    d[1] = _a[1];                         \
    d[IDX] = insert;                      \
    a = load_m128i(_a);                   \
    b = _mm_insert_epi64(a, insert, IDX); \
    CHECK_RESULT(validateInt64(b, d[0], d[1]));

    IMM_2_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_insert_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t insert = static_cast<int8_t>(*impl.mTestIntPointer2);
    __m128i a, b;
    int8_t d[16];

#define TEST_IMPL(IDX)                   \
    for (int i = 0; i < 16; i++) {       \
        d[i] = _a[i];                    \
    }                                    \
    d[IDX] = insert;                     \
    a = load_m128i(_a);                  \
    b = _mm_insert_epi8(a, insert, IDX); \
    CHECK_RESULT(VALIDATE_INT8_M128(b, d));

    IMM_16_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_insert_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    __m128 a, b, c;
#define TEST_IMPL(IDX)                               \
    float d##IDX[4] = {_a[0], _a[1], _a[2], _a[3]};  \
    d##IDX[(IDX >> 4) & 0x3] = _b[(IDX >> 6) & 0x3]; \
                                                     \
    for (int j = 0; j < 4; j++) {                    \
        if (IDX & (1 << j)) {                        \
            d##IDX[j] = 0;                           \
        }                                            \
    }                                                \
                                                     \
    a = _mm_load_ps(_a);                             \
    b = _mm_load_ps(_b);                             \
    c = _mm_insert_ps(a, b, IDX);                    \
    CHECK_RESULT(validateFloat(c, d##IDX[0], d##IDX[1], d##IDX[2], d##IDX[3]));

    IMM_256_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_max_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    int32_t d[4];
    d[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_max_epi32(a, b);

    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_max_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);
    int8_t d[16];
    d[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] > _b[3] ? _a[3] : _b[3];
    d[4] = _a[4] > _b[4] ? _a[4] : _b[4];
    d[5] = _a[5] > _b[5] ? _a[5] : _b[5];
    d[6] = _a[6] > _b[6] ? _a[6] : _b[6];
    d[7] = _a[7] > _b[7] ? _a[7] : _b[7];
    d[8] = _a[8] > _b[8] ? _a[8] : _b[8];
    d[9] = _a[9] > _b[9] ? _a[9] : _b[9];
    d[10] = _a[10] > _b[10] ? _a[10] : _b[10];
    d[11] = _a[11] > _b[11] ? _a[11] : _b[11];
    d[12] = _a[12] > _b[12] ? _a[12] : _b[12];
    d[13] = _a[13] > _b[13] ? _a[13] : _b[13];
    d[14] = _a[14] > _b[14] ? _a[14] : _b[14];
    d[15] = _a[15] > _b[15] ? _a[15] : _b[15];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_max_epi8(a, b);
    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_max_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);
    const uint16_t *_b =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer2);

    uint16_t d[8];
    d[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] > _b[3] ? _a[3] : _b[3];
    d[4] = _a[4] > _b[4] ? _a[4] : _b[4];
    d[5] = _a[5] > _b[5] ? _a[5] : _b[5];
    d[6] = _a[6] > _b[6] ? _a[6] : _b[6];
    d[7] = _a[7] > _b[7] ? _a[7] : _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_max_epu16(a, b);

    return VALIDATE_UINT16_M128(c, d);
}

result_t test_mm_max_epu32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);
    const uint32_t *_b =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer2);

    uint32_t d[4];
    d[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_max_epu32(a, b);

    return VALIDATE_UINT32_M128(c, d);
}

result_t test_mm_min_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    int32_t d[4];
    d[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epi32(a, b);

    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_min_epi8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int8_t *_a = reinterpret_cast<const int8_t *>(impl.mTestIntPointer1);
    const int8_t *_b = reinterpret_cast<const int8_t *>(impl.mTestIntPointer2);

    int8_t d[16];
    d[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] < _b[3] ? _a[3] : _b[3];
    d[4] = _a[4] < _b[4] ? _a[4] : _b[4];
    d[5] = _a[5] < _b[5] ? _a[5] : _b[5];
    d[6] = _a[6] < _b[6] ? _a[6] : _b[6];
    d[7] = _a[7] < _b[7] ? _a[7] : _b[7];
    d[8] = _a[8] < _b[8] ? _a[8] : _b[8];
    d[9] = _a[9] < _b[9] ? _a[9] : _b[9];
    d[10] = _a[10] < _b[10] ? _a[10] : _b[10];
    d[11] = _a[11] < _b[11] ? _a[11] : _b[11];
    d[12] = _a[12] < _b[12] ? _a[12] : _b[12];
    d[13] = _a[13] < _b[13] ? _a[13] : _b[13];
    d[14] = _a[14] < _b[14] ? _a[14] : _b[14];
    d[15] = _a[15] < _b[15] ? _a[15] : _b[15];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);

    __m128i c = _mm_min_epi8(a, b);
    return VALIDATE_INT8_M128(c, d);
}

result_t test_mm_min_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint16_t *_a =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer1);
    const uint16_t *_b =
        reinterpret_cast<const uint16_t *>(impl.mTestIntPointer2);

    uint16_t d[8];
    d[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] < _b[3] ? _a[3] : _b[3];
    d[4] = _a[4] < _b[4] ? _a[4] : _b[4];
    d[5] = _a[5] < _b[5] ? _a[5] : _b[5];
    d[6] = _a[6] < _b[6] ? _a[6] : _b[6];
    d[7] = _a[7] < _b[7] ? _a[7] : _b[7];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epu16(a, b);

    return VALIDATE_UINT16_M128(c, d);
}

result_t test_mm_min_epu32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint32_t *_a =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);
    const uint32_t *_b =
        reinterpret_cast<const uint32_t *>(impl.mTestIntPointer2);

    uint32_t d[4];
    d[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    d[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    d[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    d[3] = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_min_epu32(a, b);

    return VALIDATE_UINT32_M128(c, d);
}

result_t test_mm_minpos_epu16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int16_t *_a =
        reinterpret_cast<const int16_t *>(impl.mTestIntPointer1);
    uint16_t index = 0, min = static_cast<uint16_t>(_a[0]);
    for (int i = 0; i < 8; i++) {
        if (static_cast<uint16_t>(_a[i]) < min) {
            index = static_cast<uint16_t>(i);
            min = static_cast<uint16_t>(_a[i]);
        }
    }

    uint16_t d[8] = {min, index, 0, 0, 0, 0, 0, 0};

    __m128i a = load_m128i(_a);
    __m128i ret = _mm_minpos_epu16(a);
    return VALIDATE_UINT16_M128(ret, d);
}

result_t test_mm_mpsadbw_epu8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *_a =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *_b =
        reinterpret_cast<const uint8_t *>(impl.mTestIntPointer2);

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c;
#define TEST_IMPL(IDX)                                                     \
    uint8_t a_offset##IDX = ((IDX >> 2) & 0x1) * 4;                        \
    uint8_t b_offset##IDX = (IDX & 0x3) * 4;                               \
                                                                           \
    uint16_t d##IDX[8] = {};                                               \
    for (int i = 0; i < 8; i++) {                                          \
        for (int j = 0; j < 4; j++) {                                      \
            d##IDX[i] += static_cast<uint16_t>(                            \
                abs(_a[(a_offset##IDX + i) + j] - _b[b_offset##IDX + j])); \
        }                                                                  \
    }                                                                      \
    c = _mm_mpsadbw_epu8(a, b, IDX);                                       \
    CHECK_RESULT(VALIDATE_UINT16_M128(c, d##IDX));

    IMM_8_ITER
#undef TEST_IMPL
    return TEST_SUCCESS;
}

result_t test_mm_mul_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    int64_t d[2];
    d[0] = static_cast<int64_t>(_a[0]) * static_cast<int64_t>(_b[0]);
    d[1] = static_cast<int64_t>(_a[2]) * static_cast<int64_t>(_b[2]);

    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(_a));
    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(_b));
    __m128i r = _mm_mul_epi32(a, b);

    return VALIDATE_INT64_M128(r, d);
}

result_t test_mm_mullo_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t d[4];

    for (int i = 0; i < 4; i++) {
        d[i] = static_cast<int32_t>(static_cast<int64_t>(_a[i]) *
                                    static_cast<int64_t>(_b[i]));
    }
    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_mullo_epi32(a, b);
    return VALIDATE_INT32_M128(c, d);
}

result_t test_mm_packus_epi32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint16_t max = UINT16_MAX;
    uint16_t min = 0;
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_b =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);

    uint16_t d[8];
    for (int i = 0; i < 4; i++) {
        if (_a[i] > static_cast<int32_t>(max))
            d[i] = max;
        else if (_a[i] < static_cast<int32_t>(min))
            d[i] = min;
        else
            d[i] = static_cast<uint16_t>(_a[i]);
    }
    for (int i = 0; i < 4; i++) {
        if (_b[i] > static_cast<int32_t>(max))
            d[i + 4] = max;
        else if (_b[i] < static_cast<int32_t>(min))
            d[i + 4] = min;
        else
            d[i + 4] = static_cast<uint16_t>(_b[i]);
    }

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i c = _mm_packus_epi32(a, b);

    return VALIDATE_UINT16_M128(c, d);
}

OPTNONE result_t test_mm_round_pd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    double d[2] = {};
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

    return VALIDATE_DOUBLE_M128(ret, d);
}

OPTNONE result_t test_mm_round_ps(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const float *_a = impl.mTestFloatPointer1;
    float f[4] = {};
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

    return VALIDATE_FLOAT_M128(ret, f);
}

result_t test_mm_round_sd(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const double *_a = reinterpret_cast<double *>(impl.mTestFloatPointer1);
    const double *_b = reinterpret_cast<double *>(impl.mTestFloatPointer2);
    double d[2] = {};
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

    return VALIDATE_DOUBLE_M128(ret, d);
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


    return VALIDATE_FLOAT_M128(ret, f);
}

result_t test_mm_stream_load_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    int32_t *addr = impl.mTestIntPointer1;

    __m128i ret = _mm_stream_load_si128(reinterpret_cast<__m128i *>(addr));

    return VALIDATE_INT32_M128(ret, addr);
}

result_t test_mm_test_all_ones(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    __m128i a = load_m128i(_a);

    int32_t d0 = ~_a[0] & -1;
    int32_t d1 = ~_a[1] & -1;
    int32_t d2 = ~_a[2] & -1;
    int32_t d3 = ~_a[3] & -1;
    int32_t result = ((d0 | d1 | d2 | d3) == 0) ? 1 : 0;

    int32_t ret = _mm_test_all_ones(a);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_test_all_zeros(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_mask =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);
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
    const int32_t *_a =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *_mask =
        reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);
    __m128i a = load_m128i(_a);
    __m128i mask = load_m128i(_mask);

    int32_t ZF = 1;
    int32_t CF = 1;
    for (int i = 0; i < 4; i++) {
        ZF &= ((_a[i] & _mask[i]) == 0);
        CF &= ((~_a[i] & _mask[i]) == 0);
    }
    int32_t result = (ZF == 0 && CF == 0);

    int32_t ret = _mm_test_mix_ones_zeros(a, mask);
    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_testc_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = _mm_load_si128(reinterpret_cast<const __m128i *>(_a));
    __m128i b = _mm_load_si128(reinterpret_cast<const __m128i *>(_b));
    int testc = 1;
    for (int i = 0; i < 2; i++) {
        if ((~((reinterpret_cast<SIMDVec *>(&a))->m128_u64[i]) &
             (reinterpret_cast<SIMDVec *>(&b))->m128_u64[i])) {
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
    __m128i a = _mm_load_si128(reinterpret_cast<const __m128i *>(_a));
    __m128i b = _mm_load_si128(reinterpret_cast<const __m128i *>(_b));
    int testz = 1;
    for (int i = 0; i < 2; i++) {
        if (((reinterpret_cast<SIMDVec *>(&a))->m128_u64[i] &
             (reinterpret_cast<SIMDVec *>(&b))->m128_u64[i])) {
            testz = 0;
            break;
        }
    }
    return _mm_testz_si128(a, b) == testz ? TEST_SUCCESS : TEST_FAIL;
}

/* SSE4.2 */
#define IS_CMPESTRI 1

#define DEF_ENUM_MM_CMPESTRX_VARIANT(c, ...) c,

#define EVAL_MM_CMPESTRX_TEST_CASE(c, type, data_type, im, IM)               \
    do {                                                                     \
        data_type *a = test_mm_##im##_##type##_data[c].a,                    \
                  *b = test_mm_##im##_##type##_data[c].b;                    \
        int la = test_mm_##im##_##type##_data[c].la,                         \
            lb = test_mm_##im##_##type##_data[c].lb;                         \
        const int imm8 = IMM_##c;                                            \
        IIF(IM)(int expect = test_mm_##im##_##type##_data[c].expect,         \
                data_type *expect = test_mm_##im##_##type##_data[c].expect); \
        __m128i ma, mb;                                                      \
        memcpy(&ma, a, sizeof(ma));                                          \
        memcpy(&mb, b, sizeof(mb));                                          \
        IIF(IM)(int res = _mm_##im(ma, la, mb, lb, imm8),                    \
                __m128i res = _mm_##im(ma, la, mb, lb, imm8));               \
        if (IIF(IM)(res != expect, memcmp(expect, &res, sizeof(__m128i))))   \
            return TEST_FAIL;                                                \
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
#define IMM_UBYTE_EACH_LEAST_MASKED_POSITIVE                            \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_POSITIVE_POLARITY)
#define IMM_UBYTE_EACH_MOST \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT)
#define IMM_UBYTE_EACH_MOST_NEGATIVE                                   \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
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
#define IMM_UBYTE_ANY_LEAST_MASKED_POSITIVE                            \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_POSITIVE_POLARITY)
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
#define IMM_UBYTE_RANGES_MOST_NEGATIVE                             \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT | \
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
#define IMM_UBYTE_ORDERED_MOST_NEGATIVE                                   \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
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
#define IMM_SBYTE_EACH_MOST_MASKED_NEGATIVE                            \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_SBYTE_ANY_LEAST \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SBYTE_ANY_LEAST_NEGATIVE                                   \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
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
#define IMM_SBYTE_RANGES_MOST_NEGATIVE                             \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
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
#define IMM_UWORD_EACH_LEAST_NEGATIVE                                   \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UWORD_EACH_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UWORD_EACH_LEAST_MASKED_POSITIVE                            \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_POSITIVE_POLARITY)
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
#define IMM_UWORD_ORDERED_LEAST_MASKED_NEGATIVE                            \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
#define IMM_UWORD_ORDERED_MOST \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT)
#define IMM_UWORD_ORDERED_MOST_NEGATIVE                                   \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_UWORD_ORDERED_MOST_MASKED_NEGATIVE                            \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)

#define IMM_SWORD_RANGES_LEAST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT)
#define IMM_SWORD_RANGES_MOST \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_MOST_SIGNIFICANT)
#define IMM_SWORD_RANGES_LEAST_NEGATIVE                             \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SWORD_RANGES_LEAST_MASKED_NEGATIVE                      \
    (_SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_LEAST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
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
#define IMM_SWORD_ANY_MOST_NEGATIVE                                   \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_NEGATIVE_POLARITY)
#define IMM_SWORD_ANY_MOST_MASKED_NEGATIVE                            \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT | \
     _SIDD_MASKED_NEGATIVE_POLARITY)
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

#define TEST_MM_CMPESTRI_UBYTE_DATA_LEN 5
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
        /* MASKED_POSITIVE_POLARITY test: a={1,2,3,4,...}, b={1,2,99,4,...}
         * la=16, lb=4. EQUAL_EACH compares element-wise for valid positions:
         *   b[0]=1==a[0]=1 (match), b[1]=2==a[1]=2 (match),
         *   b[2]=99!=a[2]=3 (no match), b[3]=4==a[3]=4 (match).
         * IntRes1 = 0b1011 (bits 0,1,3 set for matches).
         * MASKED_POSITIVE keeps bits < lb=4, zeros bits >= 4.
         * IntRes2 = 0b1011. First set bit is at index 0. */
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         {1, 2, 99, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         16,
         4,
         IMM_UBYTE_EACH_LEAST_MASKED_POSITIVE,
         0},
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

#define TEST_MM_CMPESTRI_UWORD_DATA_LEN 5
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
        /* MASKED_POSITIVE_POLARITY test for UWORD:
         * a={100,200,300,400}, b={100,200,999,400,500,600,700,800}
         * lb=3, so only first 3 elements of b are considered.
         * EQUAL_EACH: b[0]==a[0], b[1]==a[1], b[2]!=a[2]. Result: 0b011.
         * MASKED_POSITIVE keeps bits < lb=3, zeros bits >= 3.
         * Result: 0b011. First set bit at index 0. */
        {{100, 200, 300, 400, 500, 600, 700, 800},
         {100, 200, 999, 400, 500, 600, 700, 800},
         8,
         3,
         IMM_UWORD_EACH_LEAST_MASKED_POSITIVE,
         0},
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
    _(UBYTE_ORDERED_LEAST, __VA_ARGS__)      \
    _(UBYTE_EACH_LEAST_MASKED_POSITIVE, __VA_ARGS__)

#define MM_CMPESTRI_SBYTE_TEST_CASES(_, ...) \
    _(SBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(SBYTE_ANY_LEAST, __VA_ARGS__)          \
    _(SBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(SBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPESTRI_UWORD_TEST_CASES(_, ...) \
    _(UWORD_EACH_LEAST, __VA_ARGS__)         \
    _(UWORD_ANY_LEAST, __VA_ARGS__)          \
    _(UWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(UWORD_ORDERED_LEAST, __VA_ARGS__)      \
    _(UWORD_EACH_LEAST_MASKED_POSITIVE, __VA_ARGS__)

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
#define IMM_UBYTE_EACH_UNIT_MASKED_POSITIVE                     \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK | \
     _SIDD_MASKED_POSITIVE_POLARITY)
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
#define IMM_SBYTE_EACH_UNIT_MASKED_POSITIVE                     \
    (_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK | \
     _SIDD_MASKED_POSITIVE_POLARITY)
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
#define IMM_UWORD_EACH_UNIT_MASKED_POSITIVE                     \
    (_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK | \
     _SIDD_MASKED_POSITIVE_POLARITY)
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
#define IMM_SWORD_EACH_UNIT_MASKED_POSITIVE                     \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK | \
     _SIDD_MASKED_POSITIVE_POLARITY)
#define IMM_SWORD_ANY_UNIT \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_UNIT_MASK)
#define IMM_SWORD_ORDERED_UNIT \
    (_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_UNIT_MASK)

#define TEST_MM_CMPESTRM_UBYTE_DATA_LEN 5
static test_mm_cmpestrm_ubyte_data_t
    test_mm_cmpestrm_ubyte_data[TEST_MM_CMPESTRM_UBYTE_DATA_LEN] = {
        {
            {85, 115, 101, 70, 108, 97, 116, 65, 115, 115, 101, 109, 98, 108,
             101, 114},
            {85, 115, 105, 110, 103, 65, 110, 65, 115, 115, 101, 109, 98, 108,
             101, 114},
            16,
            16,
            IMM_UBYTE_EACH_UNIT_NEGATIVE,
            {0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
        {
            {97, 101, 105, 111, 117, 121},
            {89, 111, 117, 32, 68, 114, 105, 118, 101, 32, 77, 101, 32, 77, 97,
             100},
            6,
            16,
            IMM_UBYTE_ANY_UNIT,
            {0, 255, 255, 0, 0, 0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0},
        },
        {
            {97, 122, 65, 90},
            {73, 39, 109, 32, 104, 101, 114, 101, 32, 98, 101, 99, 97, 117, 115,
             101},
            4,
            16,
            IMM_UBYTE_RANGES_UNIT,
            {255, 0, 255, 0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255,
             255, 255},
        },
        {
            {87, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {87, 104, 101, 110, 87, 101, 87, 105, 108, 108, 66, 101, 87, 101,
             100, 33},
            2,
            16,
            IMM_UBYTE_ORDERED_UNIT,
            {0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0},
        },
        /* MASKED_POSITIVE_POLARITY test for mask output:
         * a={1,2,3,...}, b={1,2,99,4,...}, la=16, lb=5
         * EQUAL_EACH compares element-wise for positions 0 to lb-1:
         *   b[0]=1==a[0]=1 → match, b[1]=2==a[1]=2 → match,
         *   b[2]=99!=a[2]=3 → no match, b[3]=4==a[3]=4 → match,
         *   b[4]=5==a[4]=5 → match
         * IntRes1 = 0b11011 (positions 0,1,3,4 match).
         * MASKED_POSITIVE keeps bits < lb=5, zeros bits >= 5.
         * IntRes2 = 0b11011 (no change since lb=5 covers all set bits).
         * With UNIT_MASK, output byte[i] = 0xFF if bit i is set.
         * Expected: {255,255,0,255,255, 0,0,0,0,0,0,0,0,0,0,0} */
        {
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
            {1, 2, 99, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
            16,
            5,
            IMM_UBYTE_EACH_UNIT_MASKED_POSITIVE,
            {255, 255, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
};

#define TEST_MM_CMPESTRM_SBYTE_DATA_LEN 4
static test_mm_cmpestrm_sbyte_data_t
    test_mm_cmpestrm_sbyte_data[TEST_MM_CMPESTRM_SBYTE_DATA_LEN] = {
        {
            {-127, -127, 34, 88, 0, 1, -1, 78, 90, 9, 23, 34, 3, -128, 127, 0},
            {0, -127, 34, 88, 12, 43, -128, 78, 8, 9, 43, 32, 7, 126, 115, 0},
            16,
            16,
            IMM_SBYTE_EACH_UNIT,
            {0, -1, -1, -1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1},
        },
        {
            {0, 32, 7, 115, -128, 44, 33},
            {0, -127, 34, 88, 12, 43, -128, 78, 8, 9, 43, 32, 7, 126, 115, 0},
            7,
            10,
            IMM_SBYTE_ANY_UNIT_MASKED_NEGATIVE,
            {0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0},
        },
        {
            {-128, -80, -90, 10, 33},
            {-126, -93, -80, -77, -56, -23, -10, -1, 0, 3, 10, 12, 13, 33, 34,
             56},
            5,
            16,
            IMM_SBYTE_RANGES_UNIT,
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0},
        },
        {
            {104, 9, -12},
            {0, 0, 87, 104, 9, -12, 89, -117, 9, 10, -11, 87, -114, 104, 9,
             -61},
            3,
            16,
            IMM_SBYTE_ORDERED_UNIT,
            {0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
};

#define TEST_MM_CMPESTRM_UWORD_DATA_LEN 4
static test_mm_cmpestrm_uword_data_t
    test_mm_cmpestrm_uword_data[TEST_MM_CMPESTRM_UWORD_DATA_LEN] = {
        {
            {1, 5, 13, 19, 22},
            {12, 60000, 5, 1, 100, 1000, 34, 20},
            5,
            8,
            IMM_UWORD_RANGES_UNIT,
            {0, 0, 65535, 65535, 0, 0, 0, 0},
        },
        {
            {65535, 12, 7, 9876, 3456, 12345, 10, 98},
            {65535, 0, 10, 9876, 3456, 0, 13, 32},
            8,
            8,
            IMM_UWORD_EACH_UNIT,
            {65535, 0, 0, 65535, 65535, 0, 0, 0},
        },
        {
            {100, 0},
            {12345, 6766, 234, 0, 1, 34, 89, 100},
            2,
            8,
            IMM_UWORD_ANY_BIT,
            {136, 0, 0, 0, 0, 0, 0, 0},
        },
        {
            {123, 67, 890},
            {123, 67, 890, 8900, 4, 0, 123, 67},
            3,
            8,
            IMM_UWORD_ORDERED_UNIT,
            {65535, 0, 0, 0, 0, 0, 65535, 0},
        },
};

#define TEST_MM_CMPESTRM_SWORD_DATA_LEN 4
static test_mm_cmpestrm_sword_data_t
    test_mm_cmpestrm_sword_data[TEST_MM_CMPESTRM_SWORD_DATA_LEN] = {
        {
            {13, 6, 5, 4, 3, 2, 1, 3},
            {-7, 16, 5, 4, -1, 6, 1, 3},
            10,
            10,
            IMM_SWORD_RANGES_UNIT,
            {0, 0, 0, 0, 0, 0, -1, -1},
        },
        {
            {85, 115, 101, 70, 108, 97, 116, 65},
            {85, 115, 105, 110, 103, 65, 110, 65},
            8,
            8,
            IMM_SWORD_EACH_UNIT,
            {-1, -1, 0, 0, 0, 0, 0, -1},
        },
        {
            {-32768, 10000, 10, -13},
            {-32767, 32767, -32768, 90, 0, -13, 23, 45},
            4,
            8,
            IMM_SWORD_ANY_UNIT,
            {0, 0, -1, 0, 0, -1, 0, 0},
        },
        {
            {10, 20, -10, 60},
            {0, 0, 0, 10, 20, -10, 60, 10},
            4,
            8,
            IMM_SWORD_ORDERED_UNIT,
            {0, 0, 0, -1, 0, 0, 0, -1},
        },
};

#define MM_CMPESTRM_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_EACH_UNIT_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_ANY_UNIT, __VA_ARGS__)           \
    _(UBYTE_RANGES_UNIT, __VA_ARGS__)        \
    _(UBYTE_ORDERED_UNIT, __VA_ARGS__)       \
    _(UBYTE_EACH_UNIT_MASKED_POSITIVE, __VA_ARGS__)

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
    const int64_t *_a =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer1);
    const int64_t *_b =
        reinterpret_cast<const int64_t *>(impl.mTestIntPointer2);

    int64_t d[2];
    d[0] = _a[0] > _b[0] ? -1 : 0;
    d[1] = _a[1] > _b[1] ? -1 : 0;

    __m128i a = load_m128i(_a);
    __m128i b = load_m128i(_b);
    __m128i iret = _mm_cmpgt_epi64(a, b);

    return VALIDATE_INT64_M128(iret, d);
}

#define IS_CMPISTRI 1

#define DEF_ENUM_MM_CMPISTRX_VARIANT(c, ...) c,

#define EVAL_MM_CMPISTRX_TEST_CASE(c, type, data_type, im, IM)               \
    do {                                                                     \
        data_type *a = test_mm_##im##_##type##_data[c].a,                    \
                  *b = test_mm_##im##_##type##_data[c].b;                    \
        const int imm8 = IMM_##c;                                            \
        IIF(IM)(int expect = test_mm_##im##_##type##_data[c].expect,         \
                data_type *expect = test_mm_##im##_##type##_data[c].expect); \
        __m128i ma, mb;                                                      \
        memcpy(&ma, a, sizeof(ma));                                          \
        memcpy(&mb, b, sizeof(mb));                                          \
        IIF(IM)(int res = _mm_##im(ma, mb, imm8),                            \
                __m128i res = _mm_##im(ma, mb, imm8));                       \
        if (IIF(IM)(res != expect, memcmp(expect, &res, sizeof(__m128i))))   \
            return TEST_FAIL;                                                \
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

#define TEST_MM_CMPISTRA_UBYTE_DATA_LEN 4
static test_mm_cmpistri_ubyte_data_t
    test_mm_cmpistra_ubyte_data[TEST_MM_CMPISTRA_UBYTE_DATA_LEN] = {
        {{10, 11, 12, 13, 14, 15, 16, 17, 18, 9, 20, 98, 97, 96, 95, 127},
         {1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 101, 102, 103, 104, 105, 106},
         IMM_UBYTE_ANY_LEAST,
         1},
        {{1, 22, 33, 44, 5, 66, 7, 88, 9, 10, 111, 0},
         {2, 23, 34, 21, 6, 65, 8, 84, 99, 100, 11, 112, 123, 14, 15, 6},
         IMM_UBYTE_EACH_LEAST,
         1},
        {{5, 15, 25, 35, 45, 55, 65, 75, 0},
         {4, 6, 14, 16, 24, 26, 34, 36, 44, 46, 54, 56, 74, 76},
         IMM_UBYTE_RANGES_LEAST,
         0},
        {{4, 14, 64, 84, 0},
         {4, 14, 64, 84, 0, 4, 14, 65, 84, 0, 4, 14, 64, 84, 0, 1},
         IMM_UBYTE_ORDERED_MOST_NEGATIVE,
         0},
};

#define TEST_MM_CMPISTRA_SBYTE_DATA_LEN 4
static test_mm_cmpistri_sbyte_data_t
    test_mm_cmpistra_sbyte_data[TEST_MM_CMPISTRA_SBYTE_DATA_LEN] = {
        {{-11, -13, -43, -50, 66, 77, 87, 98, -128, 127, 126, 99, 1, 2, 3, -5},
         {-12, -13, -43, -56, 66, 78, 88, 98, -125, 127, 120, 9, 100, 22, 54,
          -10},
         IMM_SBYTE_EACH_LEAST,
         0},
        {{10, 11, 100, -90, 0},
         {8, 9, 10, 11, 0, 8, 9, 10, -90, 0},
         IMM_SBYTE_ANY_LEAST_NEGATIVE,
         0},
        {{-90, -60, -34, -25, 34, 56, 70, 79, 0},
         {-100, -59, -35, -24, -101, 33, 57, 69, 80, 81, -128, 100, 101, 102,
          -101, -102},
         IMM_SBYTE_RANGES_LEAST,
         1},
        {{1, 1, 1, 1, -1, -1, -1, -1, -10, 10, -10, 10, 44, -44, 44, -44},
         {1, 1, -1, 1, -1, -1, -1, -1, -10, 10, -10, 10, 44, -44, 44, -44},
         IMM_SBYTE_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRA_UWORD_DATA_LEN 4
static test_mm_cmpistri_uword_data_t
    test_mm_cmpistra_uword_data[TEST_MM_CMPISTRA_UWORD_DATA_LEN] = {
        {{88, 888, 8888, 31888, 10888, 18088, 10880, 28888},
         {888, 88, 8888, 32000, 10888, 18000, 10888, 28888},
         IMM_UWORD_EACH_LEAST_NEGATIVE,
         0},
        {{3, 4, 555, 6666, 7777, 888, 9, 100},
         {1, 2, 333, 4444, 5555, 666, 7, 8},
         IMM_UWORD_ANY_LEAST,
         1},
        {{1000, 2000, 2002, 3000, 3002, 4000, 5000, 5999},
         {999, 2001, 3001, 4001, 4002, 4999, 6000, 6001},
         IMM_UWORD_RANGES_LEAST,
         1},
        {{55, 66, 77, 888, 0},
         {55, 66, 77, 888, 0, 33, 2, 10000},
         IMM_UWORD_ORDERED_LEAST,
         0},
};

#define TEST_MM_CMPISTRA_SWORD_DATA_LEN 4
static test_mm_cmpistri_sword_data_t
    test_mm_cmpistra_sword_data[TEST_MM_CMPISTRA_SWORD_DATA_LEN] = {
        {{-32000, -28000, 0},
         {-32001, -29999, -28001, -28000, -27999, -26000, -32768},
         IMM_SWORD_RANGES_LEAST_MASKED_NEGATIVE,
         0},
        {{-12, -11, -10, -9, -8, -7, 90, 1000},
         {-13, -10, 9, -8, -7, 1000, 1000, 90},
         IMM_SWORD_EACH_LEAST,
         1},
        {{33, 44, 787, 23, 0},
         {32, 43, 788, 0, 32, 0, 43, 0},
         IMM_SWORD_ANY_LEAST,
         0},
        {{18, 78, 999, -56, 0},
         {18, 78, 999, 56, 18, 78, 999, 4},
         IMM_SWORD_ORDERED_LEAST,
         1},
};

#define MM_CMPISTRA_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_ANY_LEAST, __VA_ARGS__)          \
    _(UBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(UBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(UBYTE_ORDERED_MOST_NEGATIVE, __VA_ARGS__)

#define MM_CMPISTRA_SBYTE_TEST_CASES(_, ...) \
    _(SBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(SBYTE_ANY_LEAST_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(SBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRA_UWORD_TEST_CASES(_, ...)  \
    _(UWORD_EACH_LEAST_NEGATIVE, __VA_ARGS__) \
    _(UWORD_ANY_LEAST, __VA_ARGS__)           \
    _(UWORD_RANGES_LEAST, __VA_ARGS__)        \
    _(UWORD_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRA_SWORD_TEST_CASES(_, ...)           \
    _(SWORD_RANGES_LEAST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SWORD_EACH_LEAST, __VA_ARGS__)                   \
    _(SWORD_ANY_LEAST, __VA_ARGS__)                    \
    _(SWORD_ORDERED_LEAST, __VA_ARGS__)

#define GENERATE_MM_CMPISTRA_TEST_CASES                                     \
    ENUM_MM_CMPISTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpistra, CMPISTRA,  \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpistra, CMPISTRA,   \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(UWORD, uword, uint16_t, cmpistra, CMPISTRA, \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SWORD, sword, int16_t, cmpistra, CMPISTRA,  \
                                IS_CMPISTRI)

result_t test_mm_cmpistra(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPISTRA_TEST_CASES
    return TEST_SUCCESS;
}

#define TEST_MM_CMPISTRC_UBYTE_DATA_LEN 4
static test_mm_cmpistri_ubyte_data_t
    test_mm_cmpistrc_ubyte_data[TEST_MM_CMPISTRC_UBYTE_DATA_LEN] = {
        {{89, 64, 88, 23, 11, 109, 34, 55, 0},
         {2, 64, 87, 32, 1, 110, 43, 66, 0},
         IMM_UBYTE_ANY_LEAST,
         1},
        {{99, 67, 2, 127, 125, 3, 24, 77, 32, 68, 96, 74, 70, 110, 111, 5},
         {98, 88, 67, 125, 111, 4, 56, 88, 33, 69, 99, 79, 123, 11, 10, 6},
         IMM_UBYTE_EACH_LEAST,
         0},
        {{2, 3, 74, 78, 81, 83, 85, 87, 89, 90, 0},
         {86, 90, 74, 85, 87, 81, 2, 3, 3, 3, 75, 76, 77, 78, 82, 85},
         IMM_UBYTE_RANGES_MOST_NEGATIVE,
         0},
        {{45, 67, 8, 9, 0},
         {67, 45, 67, 8, 9, 45, 67, 8, 9, 45, 67, 8, 9, 45, 67, 8},
         IMM_UBYTE_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRC_SBYTE_DATA_LEN 4
static test_mm_cmpistri_sbyte_data_t
    test_mm_cmpistrc_sbyte_data[TEST_MM_CMPISTRC_SBYTE_DATA_LEN] = {
        {{35, -35, 67, -66, 34, 55, 12, -100, 34, -34, 66, -67, 52, 100, 127,
          -128},
         {35, -35, 67, -66, 0, 55, 12, -100, 0, -34, 66, -67, 0, 100, 127,
          -128},
         IMM_SBYTE_EACH_MOST_MASKED_NEGATIVE,
         0},
        {{-119, 112, 105, 104, 0},
         {119, -112, 105, -104, 104, -34, 112, -119, 0},
         IMM_SBYTE_ANY_LEAST,
         1},
        {{-79, -69, -40, -35, 34, 45, 67, 88, 0},
         {1, 2, 3, 4, 5, 6, 7, 8, 0},
         IMM_SBYTE_RANGES_LEAST,
         0},
        {{22, -109, 123, 115, -12, 0},
         {22, -109, 12, 115, 22, -109, 123, 115, -12, 0},
         IMM_SBYTE_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRC_UWORD_DATA_LEN 4
static test_mm_cmpistri_uword_data_t
    test_mm_cmpistrc_uword_data[TEST_MM_CMPISTRC_UWORD_DATA_LEN] = {
        {{23, 45, 67, 89, 102, 121, 23, 45},
         {23, 45, 67, 89, 102, 121, 23, 44},
         IMM_UWORD_EACH_LEAST,
         1},
        {{1, 11, 55, 75}, {13, 14, 56, 77, 0}, IMM_UWORD_ANY_LEAST, 0},
        {{1, 9, 11, 19, 21, 29, 91, 99},
         {10, 29, 30, 40, 50, 60, 70, 80},
         IMM_UWORD_RANGES_LEAST,
         1},
        {{3, 4, 5, 0},
         {0, 3, 4, 5, 3, 4, 5, 0},
         IMM_UWORD_ORDERED_LEAST_MASKED_NEGATIVE,
         0},
};

#define TEST_MM_CMPISTRC_SWORD_DATA_LEN 4
static test_mm_cmpistri_sword_data_t
    test_mm_cmpistrc_sword_data[TEST_MM_CMPISTRC_SWORD_DATA_LEN] = {
        {{-78, -56, 1000, 1002},
         {-79, -55, -12, -13, 999, 1003, -80, 10000},
         IMM_SWORD_RANGES_LEAST,
         0},
        {{45, 32767, -30000, 2345, -23450, 0},
         {45, 32767, -30000, 2346, -23456, 0, 45, 333},
         IMM_SWORD_EACH_LEAST,
         1},
        {{-10000, -20000, -30000, 10000, 20000, 30000, 0},
         {10000, 20000, 30000, -10000, -20000, 20000, -30000, 12},
         IMM_SWORD_ANY_MOST_NEGATIVE,
         1},
        {{1, 2, -3, -55, -666, -7777, 8888},
         {2, -3, -55, -666, -7777, 8888, 1},
         IMM_SWORD_ORDERED_LEAST,
         0},
};

#define MM_CMPISTRC_UBYTE_TEST_CASES(_, ...)   \
    _(UBYTE_ANY_LEAST, __VA_ARGS__)            \
    _(UBYTE_EACH_LEAST, __VA_ARGS__)           \
    _(UBYTE_RANGES_MOST_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRC_SBYTE_TEST_CASES(_, ...)        \
    _(SBYTE_EACH_MOST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ANY_LEAST, __VA_ARGS__)                 \
    _(SBYTE_RANGES_LEAST, __VA_ARGS__)              \
    _(SBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRC_UWORD_TEST_CASES(_, ...) \
    _(UWORD_EACH_LEAST, __VA_ARGS__)         \
    _(UWORD_ANY_LEAST, __VA_ARGS__)          \
    _(UWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(UWORD_ORDERED_LEAST_MASKED_NEGATIVE, __VA_ARGS__)

#define MM_CMPISTRC_SWORD_TEST_CASES(_, ...) \
    _(SWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(SWORD_EACH_LEAST, __VA_ARGS__)         \
    _(SWORD_ANY_MOST_NEGATIVE, __VA_ARGS__)  \
    _(SWORD_ORDERED_LEAST, __VA_ARGS__)

#define GENERATE_MM_CMPISTRC_TEST_CASES                                     \
    ENUM_MM_CMPISTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpistrc, CMPISTRC,  \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpistrc, CMPISTRC,   \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(UWORD, uword, uint16_t, cmpistrc, CMPISTRC, \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SWORD, sword, int16_t, cmpistrc, CMPISTRC,  \
                                IS_CMPISTRI)

result_t test_mm_cmpistrc(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPISTRC_TEST_CASES
    return TEST_SUCCESS;
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
        {
            {88, 89, 90, 91, 92, 93, 0},
            {78, 88, 99, 127, 92, 93, 0},
            IMM_UBYTE_EACH_UNIT,
            {0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255},
        },
        {
            {30, 41, 52, 63, 74, 85, 0},
            {30, 42, 51, 63, 74, 85, 0},
            IMM_UBYTE_ANY_BIT,
            {57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
        {
            {34, 32, 21, 16, 7, 0},
            {34, 33, 32, 31, 30, 29, 10, 6, 0},
            IMM_UBYTE_RANGES_UNIT,
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
        {
            {33, 21, 123, 89, 76, 56, 0},
            {33, 21, 124, 33, 21, 123, 89, 76, 56, 33, 21, 123, 89, 76, 56, 22},
            IMM_UBYTE_ORDERED_UNIT,
            {0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0},
        },
};

#define TEST_MM_CMPISTRM_SBYTE_DATA_LEN 4
static test_mm_cmpistrm_sbyte_data_t
    test_mm_cmpistrm_sbyte_data[TEST_MM_CMPISTRM_SBYTE_DATA_LEN] = {
        {
            {-11, -90, -128, 127, 66, 45, 23, 32, 99, 10, 0},
            {-10, -90, -124, 33, 66, 45, 23, 22, 99, 100, 0},
            IMM_SBYTE_EACH_BIT_MASKED_NEGATIVE,
            {-115, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
        {
            {13, 14, 55, 1, 32, 100, 101, 102, 103, 97, 23, 21, 45, 54, 55, 56},
            {22, 109, 87, 45, 1, 103, 22, 102, 43, 87, 78, 56, 65, 55, 44, 33},
            IMM_SBYTE_ANY_UNIT,
            {0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0},
        },
        {
            {-31, -28, -9, 10, 45, 67, 88, 0},
            {-30, -32, -33, -44, 93, 44, 9, 89, 0},
            IMM_SBYTE_RANGES_UNIT,
            {-1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
        {
            {34, -10, 78, -99, -100, 100, 0},
            {34, 123, 88, 4, 34, -10, 78, -99, -100, 100, 34, -10, 78, -99,
             -100, -100},
            IMM_SBYTE_ORDERED_UNIT,
            {0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        },
};

#define TEST_MM_CMPISTRM_UWORD_DATA_LEN 4
static test_mm_cmpistrm_uword_data_t
    test_mm_cmpistrm_uword_data[TEST_MM_CMPISTRM_UWORD_DATA_LEN] = {
        {
            {1024, 2048, 4096, 5000, 0},
            {1023, 1000, 2047, 1596, 5566, 5666, 4477, 9487},
            IMM_UWORD_RANGES_UNIT,
            {0, 0, 65535, 65535, 0, 0, 65535, 0},
        },
        {
            {1, 2, 345, 7788, 10000, 0},
            {2, 1, 345, 7788, 10000, 0},
            IMM_UWORD_EACH_UNIT,
            {0, 0, 65535, 65535, 65535, 65535, 65535, 65535},
        },
        {
            {100, 0},
            {12345, 6766, 234, 0, 1, 34, 89, 100},
            IMM_UWORD_ANY_UNIT,
            {0, 0, 0, 0, 0, 0, 0, 0},
        },
        {
            {34, 122, 9000, 0},
            {34, 122, 9000, 34, 122, 9000, 34, 122},
            IMM_UWORD_ORDERED_UNIT_NEGATIVE,
            {0, 65535, 65535, 0, 65535, 65535, 0, 65535},
        },
};

#define TEST_MM_CMPISTRM_SWORD_DATA_LEN 4
static test_mm_cmpistrm_sword_data_t
    test_mm_cmpistrm_sword_data[TEST_MM_CMPISTRM_SWORD_DATA_LEN] = {
        {
            {-39, -10, 17, 89, 998, 1000, 1234, 4566},
            {-40, -52, -39, -29, 100, 1024, 4565, 4600},
            IMM_SWORD_RANGES_BIT,
            {0, 0, -1, -1, 0, 0, -1, 0},
        },
        {
            {345, -1900, -10000, -30000, 50, 6789, 0},
            {103, -1901, -10000, 32767, 50, 6780, 0},
            IMM_SWORD_EACH_UNIT,
            {0, 0, -1, 0, -1, 0, -1, -1},
        },
        {
            {677, 10001, 1001, 23, 0},
            {345, 677, 10001, 1003, 1001, 32, 23, 677},
            IMM_SWORD_ANY_UNIT,
            {0, -1, -1, 0, -1, 0, -1, -1},
        },
        {
            {1024, -2288, 3752, -4096, 0},
            {1024, 1024, -2288, 3752, -4096, 1024, -2288, 3752},
            IMM_SWORD_ORDERED_UNIT,
            {0, -1, 0, 0, 0, -1, 0, 0},
        },
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

#define TEST_MM_CMPISTRO_UBYTE_DATA_LEN 4
static test_mm_cmpistri_ubyte_data_t
    test_mm_cmpistro_ubyte_data[TEST_MM_CMPISTRO_UBYTE_DATA_LEN] = {
        {{3, 4, 5, 0}, {5, 5, 5, 4, 3, 0}, IMM_UBYTE_ANY_LEAST, 1},
        {{23, 127, 88, 3, 45, 6, 7, 2, 0},
         {32, 127, 87, 2, 44, 32, 1, 2, 0},
         IMM_UBYTE_EACH_MOST_NEGATIVE,
         1},
        {{3, 4, 55, 56, 0},
         {2, 3, 4, 5, 43, 54, 55, 56, 0},
         IMM_UBYTE_RANGES_LEAST,
         0},
        {{55, 66, 77, 11, 23, 0},
         {55, 55, 66, 77, 11, 23, 55, 66, 77, 11, 23, 33, 123, 18, 0},
         IMM_UBYTE_ORDERED_LEAST,
         0},
};

#define TEST_MM_CMPISTRO_SBYTE_DATA_LEN 4
static test_mm_cmpistri_sbyte_data_t
    test_mm_cmpistro_sbyte_data[TEST_MM_CMPISTRO_SBYTE_DATA_LEN] = {
        {{33, -33, 23, -32, -1, -1, 23, 46, 78, 34, 54, 100, 90, 91, 92, 101},
         {32, 33, 23, -33, -2, -3, 23, 46, -78, 43, 56, 10, 9, 91, 90, 126},
         IMM_SBYTE_EACH_LEAST,
         0},
        {{-1, -2, -3, -4, -5, -6, -7, -8, 87, 86, 85, 84, 83, 82, 81, 80},
         {87, 79, 0},
         IMM_SBYTE_ANY_LEAST,
         1},
        {{3, 4, 2, 0},
         {3, 3, 4, 5, 6, 2, 0},
         IMM_SBYTE_RANGES_MOST_NEGATIVE,
         0},
        {{23, 66, 1, 13, 17, 1, 13, 17, 0},
         {23, 66, 1, 13, 17, 1, 13, 17, 32, 23, 66, 1, 13, 17, 1, 13},
         IMM_SBYTE_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRO_UWORD_DATA_LEN 4
static test_mm_cmpistri_uword_data_t
    test_mm_cmpistro_uword_data[TEST_MM_CMPISTRO_UWORD_DATA_LEN] = {
        {{3333, 4444, 10000, 20000, 40000, 50000, 65535, 0},
         {3332, 4443, 10000, 20001, 40000, 50000, 65534, 0},
         IMM_UWORD_EACH_LEAST,
         0},
        {{1, 2, 333, 4444, 55555, 7777, 23, 347},
         {4444, 7777, 55555, 23, 347, 2, 1, 0},
         IMM_UWORD_ANY_LEAST,
         1},
        {{356, 380, 320, 456, 0},
         {455, 379, 333, 319, 300, 299, 0},
         IMM_UWORD_RANGES_LEAST,
         1},
        {{3, 1001, 235, 0},
         {3, 1001, 235, 0, 3, 1001, 235, 0},
         IMM_UWORD_ORDERED_MOST_MASKED_NEGATIVE,
         0},
};

#define TEST_MM_CMPISTRO_SWORD_DATA_LEN 4
static test_mm_cmpistri_sword_data_t
    test_mm_cmpistro_sword_data[TEST_MM_CMPISTRO_SWORD_DATA_LEN] = {
        {{-10, -5, -100, -90, 45, 56, 1000, 1009},
         {54, -1, -5, -6, 1001, 10001, 1009, 1009},
         IMM_SWORD_RANGES_LEAST,
         1},
        {{456, -32768, 32767, 13, 0},
         {455, -32768, 32767, 31, 0},
         IMM_SWORD_EACH_LEAST,
         0},
        {{23, 46, -44, 32000, 0},
         {23, 66, -44, 678, 32000, 0},
         IMM_SWORD_ANY_MOST_MASKED_NEGATIVE,
         0},
        {{-7900, -101, -34, 666, 345, 0},
         {-7900, -101, -34, 666, 345, -7900, -191, -34},
         IMM_SWORD_ORDERED_LEAST,
         1},
};

#define MM_CMPISTRO_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_ANY_LEAST, __VA_ARGS__)          \
    _(UBYTE_EACH_MOST_NEGATIVE, __VA_ARGS__) \
    _(UBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(UBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRO_SBYTE_TEST_CASES(_, ...)   \
    _(SBYTE_EACH_LEAST, __VA_ARGS__)           \
    _(SBYTE_ANY_LEAST, __VA_ARGS__)            \
    _(SBYTE_RANGES_MOST_NEGATIVE, __VA_ARGS__) \
    _(SBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRO_UWORD_TEST_CASES(_, ...) \
    _(UWORD_EACH_LEAST, __VA_ARGS__)         \
    _(UWORD_ANY_LEAST, __VA_ARGS__)          \
    _(UWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(UWORD_ORDERED_MOST_MASKED_NEGATIVE, __VA_ARGS__)

#define MM_CMPISTRO_SWORD_TEST_CASES(_, ...)       \
    _(SWORD_RANGES_LEAST, __VA_ARGS__)             \
    _(SWORD_EACH_LEAST, __VA_ARGS__)               \
    _(SWORD_ANY_MOST_MASKED_NEGATIVE, __VA_ARGS__) \
    _(SWORD_ORDERED_LEAST, __VA_ARGS__)

#define GENERATE_MM_CMPISTRO_TEST_CASES                                     \
    ENUM_MM_CMPISTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpistro, CMPISTRO,  \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpistro, CMPISTRO,   \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(UWORD, uword, uint16_t, cmpistro, CMPISTRO, \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SWORD, sword, int16_t, cmpistro, CMPISTRO,  \
                                IS_CMPISTRI)

result_t test_mm_cmpistro(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPISTRO_TEST_CASES
    return TEST_SUCCESS;
}

#define TEST_MM_CMPISTRS_UBYTE_DATA_LEN 4
static test_mm_cmpistri_ubyte_data_t
    test_mm_cmpistrs_ubyte_data[TEST_MM_CMPISTRS_UBYTE_DATA_LEN] = {
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         {1, 2, 3, 4, 5, 0},
         IMM_UBYTE_ANY_LEAST,
         0},
        {{127, 126, 125, 124, 0},
         {127, 1, 34, 43, 54, 0},
         IMM_UBYTE_EACH_LEAST,
         1},
        {{127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
          127, 127},
         {56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 0},
         IMM_UBYTE_RANGES_LEAST,
         0},
        {{33, 44, 55, 78, 99, 100, 101, 102, 0},
         {0},
         IMM_UBYTE_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRS_SBYTE_DATA_LEN 4
static test_mm_cmpistri_sbyte_data_t
    test_mm_cmpistrs_sbyte_data[TEST_MM_CMPISTRS_SBYTE_DATA_LEN] = {
        {{100, 99, 98, 97, -67, -4, -5, -6, -7, -1, -2, -3, -128, -128, -128,
          -128},
         {0},
         IMM_SBYTE_EACH_LEAST,
         0},
        {{-128, -128, -128, -128, 127, 127, 127, 127, -128, -128, -128, -128,
          127, 127, 127, 127},
         {-1, -2, -11, -98, -12, 0},
         IMM_SBYTE_ANY_LEAST,
         0},
        {{0, 1, 2, 3, 4, 5, -6, -7},
         {0, 1, 2, 3, 4, 5, 6, 7},
         IMM_SBYTE_RANGES_LEAST,
         1},
        {{0, 1, 0, -1, 0, -2, 0, 0, -3, 4, 0, 0, 5, 6, 7, 8},
         {0},
         IMM_SBYTE_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRS_UWORD_DATA_LEN 4
static test_mm_cmpistri_uword_data_t
    test_mm_cmpistrs_uword_data[TEST_MM_CMPISTRS_UWORD_DATA_LEN] = {
        {{0, 1, 2, 3, 65535, 0, 0, 0},
         {9, 8, 7, 6, 5, 4, 3, 2},
         IMM_UWORD_EACH_LEAST,
         1},
        {{4, 567, 65535, 32, 34, 0}, {0}, IMM_UWORD_ANY_LEAST, 1},
        {{65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535},
         {1, 2, 3, 4, 900, 7890, 6767, 0},
         IMM_UWORD_RANGES_LEAST,
         0},
        {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 0}, IMM_UWORD_ORDERED_LEAST, 0},
};

#define TEST_MM_CMPISTRS_SWORD_DATA_LEN 4
static test_mm_cmpistri_sword_data_t
    test_mm_cmpistrs_sword_data[TEST_MM_CMPISTRS_SWORD_DATA_LEN] = {
        {{-32768, -32768, -32768, -32768, -32768, -32768, -32768, -3276},
         {34, 45, 6, 7, 9, 8, 7, 6},
         IMM_SWORD_RANGES_LEAST,
         0},
        {{1000, 2000, 4000, 8000, 16000, 32000, 32767, 0},
         {3, 4, 56, 23, 0},
         IMM_SWORD_EACH_LEAST,
         1},
        {{0, 1, 3, 4, -32768, 9, 0, 1},
         {56, 47, 43, 999, 1111, 0},
         IMM_SWORD_ANY_LEAST,
         1},
        {{1111, 1212, 831, 2345, 32767, 32767, -32768, 32767},
         {0},
         IMM_SWORD_ORDERED_LEAST,
         0},
};

#define MM_CMPISTRS_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_ANY_LEAST, __VA_ARGS__)          \
    _(UBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(UBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(UBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRS_SBYTE_TEST_CASES(_, ...) \
    _(SBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(SBYTE_ANY_LEAST, __VA_ARGS__)          \
    _(SBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(SBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRS_UWORD_TEST_CASES(_, ...) \
    _(UWORD_EACH_LEAST, __VA_ARGS__)         \
    _(UWORD_ANY_LEAST, __VA_ARGS__)          \
    _(UWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(UWORD_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRS_SWORD_TEST_CASES(_, ...) \
    _(SWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(SWORD_EACH_LEAST, __VA_ARGS__)         \
    _(SWORD_ANY_LEAST, __VA_ARGS__)          \
    _(SWORD_ORDERED_LEAST, __VA_ARGS__)

#define GENERATE_MM_CMPISTRS_TEST_CASES                                     \
    ENUM_MM_CMPISTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpistrs, CMPISTRS,  \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpistrs, CMPISTRS,   \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(UWORD, uword, uint16_t, cmpistrs, CMPISTRS, \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SWORD, sword, int16_t, cmpistrs, CMPISTRS,  \
                                IS_CMPISTRI)

result_t test_mm_cmpistrs(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPISTRS_TEST_CASES
    return TEST_SUCCESS;
}

#define TEST_MM_CMPISTRZ_UBYTE_DATA_LEN 4
static test_mm_cmpistri_ubyte_data_t
    test_mm_cmpistrz_ubyte_data[TEST_MM_CMPISTRZ_UBYTE_DATA_LEN] = {
        {{0},
         {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
          255, 255},
         IMM_UBYTE_ANY_LEAST,
         0},
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         {1, 1, 1, 1, 2, 2, 2, 2, 4, 5, 6, 7, 89, 89, 89, 89},
         IMM_UBYTE_EACH_LEAST,
         0},
        {{1, 2, 3, 4, 0}, {}, IMM_UBYTE_RANGES_LEAST, 1},
        {{127, 126, 125, 124, 124, 0},
         {100, 101, 123, 100, 111, 122, 0},
         IMM_UBYTE_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRZ_SBYTE_DATA_LEN 4
static test_mm_cmpistri_sbyte_data_t
    test_mm_cmpistrz_sbyte_data[TEST_MM_CMPISTRZ_SBYTE_DATA_LEN] = {
        {{127, 126, 99, -100, 0},
         {-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
          -128, -128, -128, -128, -128},
         IMM_SBYTE_EACH_LEAST,
         0},
        {{120, 66, 54, 0}, {3, 4, 5, -99, -6, 0}, IMM_SBYTE_ANY_LEAST, 1},
        {{0},
         {127, 127, 127, 127, 126, 126, 126, 126, -127, -127, -127, -127, -1,
          -1, -1, -1},
         IMM_SBYTE_RANGES_LEAST,
         0},
        {{12, 3, 4, 5, 6, 7, 8, 0},
         {-1, -2, -3, -4, -6, 75, 0},
         IMM_SBYTE_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRZ_UWORD_DATA_LEN 4
static test_mm_cmpistri_uword_data_t
    test_mm_cmpistrz_uword_data[TEST_MM_CMPISTRZ_UWORD_DATA_LEN] = {
        {{10000, 20000, 50000, 40000, 0},
         {65535, 65533, 60000, 60000, 50000, 123, 1, 2},
         IMM_UWORD_EACH_LEAST,
         0},
        {{0},
         {65528, 65529, 65530, 65531, 65532, 65533, 65534, 65535},
         IMM_UWORD_ANY_LEAST,
         0},
        {{3, 333, 3333, 33333, 0}, {0}, IMM_UWORD_RANGES_LEAST, 1},
        {{123, 456, 7, 890, 0},
         {123, 456, 7, 900, 0},
         IMM_UWORD_ORDERED_LEAST,
         1},
};

#define TEST_MM_CMPISTRZ_SWORD_DATA_LEN 4
static test_mm_cmpistri_sword_data_t
    test_mm_cmpistrz_sword_data[TEST_MM_CMPISTRZ_SWORD_DATA_LEN] = {
        {{2, 22, 222, 2222, 22222, -2222, -222, -22},
         {-32768, 32767, -32767, 32766, -32766, 32765, -32768, 32767},
         IMM_SWORD_RANGES_LEAST,
         0},
        {{345, 10000, -10000, -30000, 0},
         {1, 2, 3, 4, 5, 6, 7, 0},
         IMM_SWORD_EACH_LEAST,
         1},
        {{}, {0}, IMM_SWORD_ANY_LEAST, 1},
        {{1, 2, -789, -1, -90, 0},
         {1, 10, 100, 1000, 10000, -10000, -1000, 1000},
         IMM_SWORD_ORDERED_LEAST,
         0},
};

#define MM_CMPISTRZ_UBYTE_TEST_CASES(_, ...) \
    _(UBYTE_ANY_LEAST, __VA_ARGS__)          \
    _(UBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(UBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(UBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRZ_SBYTE_TEST_CASES(_, ...) \
    _(SBYTE_EACH_LEAST, __VA_ARGS__)         \
    _(SBYTE_ANY_LEAST, __VA_ARGS__)          \
    _(SBYTE_RANGES_LEAST, __VA_ARGS__)       \
    _(SBYTE_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRZ_UWORD_TEST_CASES(_, ...) \
    _(UWORD_EACH_LEAST, __VA_ARGS__)         \
    _(UWORD_ANY_LEAST, __VA_ARGS__)          \
    _(UWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(UWORD_ORDERED_LEAST, __VA_ARGS__)

#define MM_CMPISTRZ_SWORD_TEST_CASES(_, ...) \
    _(SWORD_RANGES_LEAST, __VA_ARGS__)       \
    _(SWORD_EACH_LEAST, __VA_ARGS__)         \
    _(SWORD_ANY_LEAST, __VA_ARGS__)          \
    _(SWORD_ORDERED_LEAST, __VA_ARGS__)

#define GENERATE_MM_CMPISTRZ_TEST_CASES                                     \
    ENUM_MM_CMPISTRX_TEST_CASES(UBYTE, ubyte, uint8_t, cmpistrz, CMPISTRZ,  \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SBYTE, sbyte, int8_t, cmpistrz, CMPISTRZ,   \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(UWORD, uword, uint16_t, cmpistrz, CMPISTRZ, \
                                IS_CMPISTRI)                                \
    ENUM_MM_CMPISTRX_TEST_CASES(SWORD, sword, int16_t, cmpistrz, CMPISTRZ,  \
                                IS_CMPISTRI)

result_t test_mm_cmpistrz(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    GENERATE_MM_CMPISTRZ_TEST_CASES
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u16(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint32_t crc = *reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);
    uint16_t v = static_cast<uint16_t>(iter);
    uint32_t result = _mm_crc32_u16(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u16(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint32_t crc = *reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);
    uint32_t v = *reinterpret_cast<const uint32_t *>(impl.mTestIntPointer2);
    uint32_t result = _mm_crc32_u32(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u32(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint64_t crc = *reinterpret_cast<const uint64_t *>(impl.mTestIntPointer1);
    uint64_t v = *reinterpret_cast<const uint64_t *>(impl.mTestIntPointer2);
    uint64_t result = _mm_crc32_u64(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u64(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u8(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    uint32_t crc = *reinterpret_cast<const uint32_t *>(impl.mTestIntPointer1);
    uint8_t v = static_cast<uint8_t>(iter);
    uint32_t result = _mm_crc32_u8(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u8(crc, v));
    return TEST_SUCCESS;
}

/* AES */
result_t test_mm_aesenc_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *a = reinterpret_cast<int32_t *>(impl.mTestIntPointer1);
    const int32_t *b = reinterpret_cast<int32_t *>(impl.mTestIntPointer2);
    __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a));
    __m128i rk = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b));

    __m128i resultReference = aesenc_128_reference(data, rk);
    __m128i resultIntrinsic = _mm_aesenc_si128(data, rk);

    return validate128(resultReference, resultIntrinsic);
}

result_t test_mm_aesdec_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *a = reinterpret_cast<int32_t *>(impl.mTestIntPointer1);
    const int32_t *b = reinterpret_cast<int32_t *>(impl.mTestIntPointer2);
    __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a));
    __m128i rk = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b));

    __m128i resultReference = aesdec_128_reference(data, rk);
    __m128i resultIntrinsic = _mm_aesdec_si128(data, rk);

    return validate128(resultReference, resultIntrinsic);
}

result_t test_mm_aesenclast_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const int32_t *a = reinterpret_cast<const int32_t *>(impl.mTestIntPointer1);
    const int32_t *b = reinterpret_cast<const int32_t *>(impl.mTestIntPointer2);
    __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a));
    __m128i rk = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b));

    __m128i resultReference = aesenclast_128_reference(data, rk);
    __m128i resultIntrinsic = _mm_aesenclast_si128(data, rk);

    return validate128(resultReference, resultIntrinsic);
}

result_t test_mm_aesdeclast_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *a = reinterpret_cast<uint8_t *>(impl.mTestIntPointer1);
    const uint8_t *rk = reinterpret_cast<uint8_t *>(impl.mTestIntPointer2);
    __m128i _a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a));
    __m128i _rk = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rk));
    uint8_t c[16] = {};

    uint8_t v[4][4];
    for (int i = 0; i < 16; ++i) {
        v[((i / 4) + (i % 4)) % 4][i % 4] = crypto_aes_rsbox[a[i]];
    }
    for (int i = 0; i < 16; ++i) {
        c[i] = v[i / 4][i % 4] ^ rk[i];
    }

    __m128i result_reference =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(c));
    __m128i result_intrinsic = _mm_aesdeclast_si128(_a, _rk);

    return validate128(result_reference, result_intrinsic);
}

result_t test_mm_aesimc_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint8_t *a = reinterpret_cast<uint8_t *>(impl.mTestIntPointer1);
    __m128i _a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a));

    uint8_t e, f, g, h, v[4][4];
    for (int i = 0; i < 16; ++i) {
        (reinterpret_cast<uint8_t *>(v))[i] = a[i];
    }
    for (int i = 0; i < 4; ++i) {
        e = v[i][0];
        f = v[i][1];
        g = v[i][2];
        h = v[i][3];

        v[i][0] = static_cast<uint8_t>(MULTIPLY(e, 0x0e) ^ MULTIPLY(f, 0x0b) ^
                                       MULTIPLY(g, 0x0d) ^ MULTIPLY(h, 0x09));
        v[i][1] = static_cast<uint8_t>(MULTIPLY(e, 0x09) ^ MULTIPLY(f, 0x0e) ^
                                       MULTIPLY(g, 0x0b) ^ MULTIPLY(h, 0x0d));
        v[i][2] = static_cast<uint8_t>(MULTIPLY(e, 0x0d) ^ MULTIPLY(f, 0x09) ^
                                       MULTIPLY(g, 0x0e) ^ MULTIPLY(h, 0x0b));
        v[i][3] = static_cast<uint8_t>(MULTIPLY(e, 0x0b) ^ MULTIPLY(f, 0x0d) ^
                                       MULTIPLY(g, 0x09) ^ MULTIPLY(h, 0x0e));
    }

    __m128i result_reference =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(v));
    __m128i result_intrinsic = _mm_aesimc_si128(_a);

    return validate128(result_reference, result_intrinsic);
}

static inline uint32_t sub_word(uint32_t in)
{
    return (static_cast<uint32_t>(crypto_aes_sbox[(in >> 24) & 0xff]) << 24) |
           (static_cast<uint32_t>(crypto_aes_sbox[(in >> 16) & 0xff]) << 16) |
           (static_cast<uint32_t>(crypto_aes_sbox[(in >> 8) & 0xff]) << 8) |
           (static_cast<uint32_t>(crypto_aes_sbox[in & 0xff]));
}

// AES-256 key expansion helper: expands key using aeskeygenassist result
// For even rounds (uses RotWord+SubWord+Rcon from word[3])
// Reference: Botan AES-NI implementation
static inline __m128i aes_mix_key(__m128i key)
{
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return key;
}

static inline __m128i aes256_key_exp_even(__m128i key, __m128i assist)
{
    // Broadcast word[3] which contains SubWord(RotWord(X[3])) ^ Rcon
    __m128i tmp = _mm_shuffle_epi32(assist, _MM_SHUFFLE(3, 3, 3, 3));
    return _mm_xor_si128(aes_mix_key(key), tmp);
}

// For odd rounds (uses SubWord only from word[2], no Rcon)
static inline __m128i aes256_key_exp_odd(__m128i key, __m128i assist)
{
    // Broadcast word[2] which contains SubWord(X[3])
    __m128i tmp = _mm_shuffle_epi32(assist, _MM_SHUFFLE(2, 2, 2, 2));
    return _mm_xor_si128(aes_mix_key(key), tmp);
}

// Validate __m128i against expected 16-byte array
static inline bool validate_round_key_bytes(__m128i key,
                                            const uint8_t *expected)
{
    alignas(16) uint8_t result[16];
    _mm_store_si128(reinterpret_cast<__m128i *>(result), key);
    return memcmp(result, expected, 16) == 0;
}

result_t test_mm_aeskeygenassist_si128(const SSE2NEONTestImpl &impl,
                                       uint32_t iter)
{
    // Part 1: Basic SubWord/RotWord validation for all Rcon values (0-255)
    const uint32_t *a = reinterpret_cast<uint32_t *>(impl.mTestIntPointer1);
    __m128i data = load_m128i(a);
    uint32_t sub_x1 = sub_word(a[1]);
    uint32_t sub_x3 = sub_word(a[3]);
    __m128i result_reference;
    __m128i result_intrinsic;
#define TEST_IMPL(IDX)                                       \
    uint32_t res##IDX[4] = {                                 \
        sub_x1,                                              \
        rotr(sub_x1, 8) ^ IDX,                               \
        sub_x3,                                              \
        rotr(sub_x3, 8) ^ IDX,                               \
    };                                                       \
    result_reference = load_m128i(res##IDX);                 \
    result_intrinsic = _mm_aeskeygenassist_si128(data, IDX); \
    CHECK_RESULT(validate128(result_reference, result_intrinsic));

    IMM_256_ITER
#undef TEST_IMPL

    // Part 2: NIST FIPS 197 Appendix A.3 AES-256 Key Expansion Test
    // Reference: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf
    //
    // Cipher Key (256-bit, byte array):
    //   60 3d eb 10 15 ca 71 be 2b 73 ae f0 85 7d 77 81
    //   1f 35 2c 07 3b 61 08 d7 2d 98 10 a3 09 14 df f4
    //
    // This test verifies full AES-256 key expansion using _mm_aeskeygenassist
    // following the Botan AES-NI implementation pattern.

    // NIST FIPS 197 A.3: AES-256 cipher key (32 bytes)
    const uint8_t nist_key[32] = {
        0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe,  // K0 bytes 0-7
        0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,  // K0 bytes 8-15
        0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7,  // K1 bytes 0-7
        0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4   // K1 bytes 8-15
    };

    // NIST FIPS 197 A.3: Expected round keys (15 x 16 bytes)
    // Byte arrays in memory order (as loaded by _mm_loadu_si128)
    const uint8_t nist_round_keys[15][16] = {
        // K0: w[0-3]
        {0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe, 0x2b, 0x73, 0xae, 0xf0,
         0x85, 0x7d, 0x77, 0x81},
        // K1: w[4-7]
        {0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7, 0x2d, 0x98, 0x10, 0xa3,
         0x09, 0x14, 0xdf, 0xf4},
        // K2: w[8-11]
        {0x9b, 0xa3, 0x54, 0x11, 0x8e, 0x69, 0x25, 0xaf, 0xa5, 0x1a, 0x8b, 0x5f,
         0x20, 0x67, 0xfc, 0xde},
        // K3: w[12-15]
        {0xa8, 0xb0, 0x9c, 0x1a, 0x93, 0xd1, 0x94, 0xcd, 0xbe, 0x49, 0x84, 0x6e,
         0xb7, 0x5d, 0x5b, 0x9a},
        // K4: w[16-19]
        {0xd5, 0x9a, 0xec, 0xb8, 0x5b, 0xf3, 0xc9, 0x17, 0xfe, 0xe9, 0x42, 0x48,
         0xde, 0x8e, 0xbe, 0x96},
        // K5: w[20-23]
        {0xb5, 0xa9, 0x32, 0x8a, 0x26, 0x78, 0xa6, 0x47, 0x98, 0x31, 0x22, 0x29,
         0x2f, 0x6c, 0x79, 0xb3},
        // K6: w[24-27]
        {0x81, 0x2c, 0x81, 0xad, 0xda, 0xdf, 0x48, 0xba, 0x24, 0x36, 0x0a, 0xf2,
         0xfa, 0xb8, 0xb4, 0x64},
        // K7: w[28-31]
        {0x98, 0xc5, 0xbf, 0xc9, 0xbe, 0xbd, 0x19, 0x8e, 0x26, 0x8c, 0x3b, 0xa7,
         0x09, 0xe0, 0x42, 0x14},
        // K8: w[32-35]
        {0x68, 0x00, 0x7b, 0xac, 0xb2, 0xdf, 0x33, 0x16, 0x96, 0xe9, 0x39, 0xe4,
         0x6c, 0x51, 0x8d, 0x80},
        // K9: w[36-39]
        {0xc8, 0x14, 0xe2, 0x04, 0x76, 0xa9, 0xfb, 0x8a, 0x50, 0x25, 0xc0, 0x2d,
         0x59, 0xc5, 0x82, 0x39},
        // K10: w[40-43]
        {0xde, 0x13, 0x69, 0x67, 0x6c, 0xcc, 0x5a, 0x71, 0xfa, 0x25, 0x63, 0x95,
         0x96, 0x74, 0xee, 0x15},
        // K11: w[44-47]
        {0x58, 0x86, 0xca, 0x5d, 0x2e, 0x2f, 0x31, 0xd7, 0x7e, 0x0a, 0xf1, 0xfa,
         0x27, 0xcf, 0x73, 0xc3},
        // K12: w[48-51]
        {0x74, 0x9c, 0x47, 0xab, 0x18, 0x50, 0x1d, 0xda, 0xe2, 0x75, 0x7e, 0x4f,
         0x74, 0x01, 0x90, 0x5a},
        // K13: w[52-55]
        {0xca, 0xfa, 0xaa, 0xe3, 0xe4, 0xd5, 0x9b, 0x34, 0x9a, 0xdf, 0x6a, 0xce,
         0xbd, 0x10, 0x19, 0x0d},
        // K14: w[56-59]
        {0xfe, 0x48, 0x90, 0xd1, 0xe6, 0x18, 0x8d, 0x0b, 0x04, 0x6d, 0xf3, 0x44,
         0x70, 0x6c, 0x63, 0x1e},
    };

    // Load initial key halves (K0 and K1)
    __m128i K0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&nist_key[0]));
    __m128i K1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&nist_key[16]));

    // Validate K0 and K1 match expected values
    if (!validate_round_key_bytes(K0, nist_round_keys[0]))
        return TEST_FAIL;
    if (!validate_round_key_bytes(K1, nist_round_keys[1]))
        return TEST_FAIL;

    // AES-256 key expansion using _mm_aeskeygenassist_si128
    // Pattern: K[2i] uses Rcon, K[2i+1] uses no Rcon
    // Rcon values: 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40
    __m128i K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14;

    // Round 2 (Rcon = 0x01)
    K2 = aes256_key_exp_even(K0, _mm_aeskeygenassist_si128(K1, 0x01));
    if (!validate_round_key_bytes(K2, nist_round_keys[2]))
        return TEST_FAIL;

    // Round 3 (no Rcon, SubWord only)
    K3 = aes256_key_exp_odd(K1, _mm_aeskeygenassist_si128(K2, 0x00));
    if (!validate_round_key_bytes(K3, nist_round_keys[3]))
        return TEST_FAIL;

    // Round 4 (Rcon = 0x02)
    K4 = aes256_key_exp_even(K2, _mm_aeskeygenassist_si128(K3, 0x02));
    if (!validate_round_key_bytes(K4, nist_round_keys[4]))
        return TEST_FAIL;

    // Round 5 (no Rcon)
    K5 = aes256_key_exp_odd(K3, _mm_aeskeygenassist_si128(K4, 0x00));
    if (!validate_round_key_bytes(K5, nist_round_keys[5]))
        return TEST_FAIL;

    // Round 6 (Rcon = 0x04)
    K6 = aes256_key_exp_even(K4, _mm_aeskeygenassist_si128(K5, 0x04));
    if (!validate_round_key_bytes(K6, nist_round_keys[6]))
        return TEST_FAIL;

    // Round 7 (no Rcon)
    K7 = aes256_key_exp_odd(K5, _mm_aeskeygenassist_si128(K6, 0x00));
    if (!validate_round_key_bytes(K7, nist_round_keys[7]))
        return TEST_FAIL;

    // Round 8 (Rcon = 0x08)
    K8 = aes256_key_exp_even(K6, _mm_aeskeygenassist_si128(K7, 0x08));
    if (!validate_round_key_bytes(K8, nist_round_keys[8]))
        return TEST_FAIL;

    // Round 9 (no Rcon)
    K9 = aes256_key_exp_odd(K7, _mm_aeskeygenassist_si128(K8, 0x00));
    if (!validate_round_key_bytes(K9, nist_round_keys[9]))
        return TEST_FAIL;

    // Round 10 (Rcon = 0x10)
    K10 = aes256_key_exp_even(K8, _mm_aeskeygenassist_si128(K9, 0x10));
    if (!validate_round_key_bytes(K10, nist_round_keys[10]))
        return TEST_FAIL;

    // Round 11 (no Rcon)
    K11 = aes256_key_exp_odd(K9, _mm_aeskeygenassist_si128(K10, 0x00));
    if (!validate_round_key_bytes(K11, nist_round_keys[11]))
        return TEST_FAIL;

    // Round 12 (Rcon = 0x20)
    K12 = aes256_key_exp_even(K10, _mm_aeskeygenassist_si128(K11, 0x20));
    if (!validate_round_key_bytes(K12, nist_round_keys[12]))
        return TEST_FAIL;

    // Round 13 (no Rcon)
    K13 = aes256_key_exp_odd(K11, _mm_aeskeygenassist_si128(K12, 0x00));
    if (!validate_round_key_bytes(K13, nist_round_keys[13]))
        return TEST_FAIL;

    // Round 14 (Rcon = 0x40)
    K14 = aes256_key_exp_even(K12, _mm_aeskeygenassist_si128(K13, 0x40));
    if (!validate_round_key_bytes(K14, nist_round_keys[14]))
        return TEST_FAIL;

    // Part 3: AES-256 Encryption Test with NIST FIPS 197 Appendix C.3
    // Plaintext:  00112233445566778899aabbccddeeff
    // Key: 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
    // Ciphertext: 8ea2b7ca516745bfeafc49904b496089

    // NIST FIPS 197 C.3 test key
    const uint8_t enc_key[32] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
        0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
        0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    };

    // NIST plaintext and expected ciphertext
    const uint8_t plaintext[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    };
    const uint8_t expected_cipher[16] = {
        0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf,
        0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89,
    };

    // Expand the C.3 key
    __m128i EK0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&enc_key[0]));
    __m128i EK1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&enc_key[16]));

    __m128i EK[15];
    EK[0] = EK0;
    EK[1] = EK1;
    EK[2] = aes256_key_exp_even(EK[0], _mm_aeskeygenassist_si128(EK[1], 0x01));
    EK[3] = aes256_key_exp_odd(EK[1], _mm_aeskeygenassist_si128(EK[2], 0x00));
    EK[4] = aes256_key_exp_even(EK[2], _mm_aeskeygenassist_si128(EK[3], 0x02));
    EK[5] = aes256_key_exp_odd(EK[3], _mm_aeskeygenassist_si128(EK[4], 0x00));
    EK[6] = aes256_key_exp_even(EK[4], _mm_aeskeygenassist_si128(EK[5], 0x04));
    EK[7] = aes256_key_exp_odd(EK[5], _mm_aeskeygenassist_si128(EK[6], 0x00));
    EK[8] = aes256_key_exp_even(EK[6], _mm_aeskeygenassist_si128(EK[7], 0x08));
    EK[9] = aes256_key_exp_odd(EK[7], _mm_aeskeygenassist_si128(EK[8], 0x00));
    EK[10] = aes256_key_exp_even(EK[8], _mm_aeskeygenassist_si128(EK[9], 0x10));
    EK[11] = aes256_key_exp_odd(EK[9], _mm_aeskeygenassist_si128(EK[10], 0x00));
    EK[12] =
        aes256_key_exp_even(EK[10], _mm_aeskeygenassist_si128(EK[11], 0x20));
    EK[13] =
        aes256_key_exp_odd(EK[11], _mm_aeskeygenassist_si128(EK[12], 0x00));
    EK[14] =
        aes256_key_exp_even(EK[12], _mm_aeskeygenassist_si128(EK[13], 0x40));

    // Encrypt plaintext using AES-256 (14 rounds)
    __m128i state =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(plaintext));

    // Initial round: AddRoundKey
    state = _mm_xor_si128(state, EK[0]);

    // Rounds 1-13: SubBytes, ShiftRows, MixColumns, AddRoundKey
    state = _mm_aesenc_si128(state, EK[1]);
    state = _mm_aesenc_si128(state, EK[2]);
    state = _mm_aesenc_si128(state, EK[3]);
    state = _mm_aesenc_si128(state, EK[4]);
    state = _mm_aesenc_si128(state, EK[5]);
    state = _mm_aesenc_si128(state, EK[6]);
    state = _mm_aesenc_si128(state, EK[7]);
    state = _mm_aesenc_si128(state, EK[8]);
    state = _mm_aesenc_si128(state, EK[9]);
    state = _mm_aesenc_si128(state, EK[10]);
    state = _mm_aesenc_si128(state, EK[11]);
    state = _mm_aesenc_si128(state, EK[12]);
    state = _mm_aesenc_si128(state, EK[13]);

    // Final round: SubBytes, ShiftRows, AddRoundKey (no MixColumns)
    state = _mm_aesenclast_si128(state, EK[14]);

    // Verify ciphertext
    uint8_t result_cipher[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result_cipher), state);
    for (int i = 0; i < 16; i++) {
        if (result_cipher[i] != expected_cipher[i])
            return TEST_FAIL;
    }

    // Part 4: AES-256 Decryption Test (verify roundtrip)

    // Generate decryption keys using InvMixColumns
    __m128i DK[15];
    DK[0] = EK[14];  // First decryption key = last encryption key
    for (int i = 1; i < 14; i++)
        DK[i] = _mm_aesimc_si128(EK[14 - i]);
    DK[14] = EK[0];  // Last decryption key = first encryption key

    // Decrypt ciphertext
    state = _mm_loadu_si128(reinterpret_cast<const __m128i *>(expected_cipher));

    // Initial round: AddRoundKey
    state = _mm_xor_si128(state, DK[0]);

    // Rounds 1-13: InvSubBytes, InvShiftRows, InvMixColumns, AddRoundKey
    state = _mm_aesdec_si128(state, DK[1]);
    state = _mm_aesdec_si128(state, DK[2]);
    state = _mm_aesdec_si128(state, DK[3]);
    state = _mm_aesdec_si128(state, DK[4]);
    state = _mm_aesdec_si128(state, DK[5]);
    state = _mm_aesdec_si128(state, DK[6]);
    state = _mm_aesdec_si128(state, DK[7]);
    state = _mm_aesdec_si128(state, DK[8]);
    state = _mm_aesdec_si128(state, DK[9]);
    state = _mm_aesdec_si128(state, DK[10]);
    state = _mm_aesdec_si128(state, DK[11]);
    state = _mm_aesdec_si128(state, DK[12]);
    state = _mm_aesdec_si128(state, DK[13]);

    // Final round: InvSubBytes, InvShiftRows, AddRoundKey (no InvMixColumns)
    state = _mm_aesdeclast_si128(state, DK[14]);

    // Verify decryption matches original plaintext
    uint8_t result_plain[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result_plain), state);
    for (int i = 0; i < 16; i++) {
        if (result_plain[i] != plaintext[i])
            return TEST_FAIL;
    }

    return TEST_SUCCESS;
}

/* Others */
result_t test_mm_clmulepi64_si128(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint64_t *_a =
        reinterpret_cast<const uint64_t *>(impl.mTestIntPointer1);
    const uint64_t *_b =
        reinterpret_cast<const uint64_t *>(impl.mTestIntPointer2);
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

static int popcnt_reference(uint64_t a)
{
    int count = 0;
    while (a != 0) {
        count += a & 1;
        a >>= 1;
    }
    return count;
}

result_t test_mm_popcnt_u32(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint64_t *a =
        reinterpret_cast<const uint64_t *>(impl.mTestIntPointer1);
    ASSERT_RETURN(popcnt_reference(static_cast<uint32_t>(a[0])) ==
                  _mm_popcnt_u32(static_cast<unsigned int>(a[0])));
    return TEST_SUCCESS;
}

result_t test_mm_popcnt_u64(const SSE2NEONTestImpl &impl, uint32_t iter)
{
    const uint64_t *a =
        reinterpret_cast<const uint64_t *>(impl.mTestIntPointer1);
    ASSERT_RETURN(popcnt_reference(a[0]) == _mm_popcnt_u64(a[0]));
    return TEST_SUCCESS;
}

OPTNONE result_t test_mm_set_denormals_zero_mode(const SSE2NEONTestImpl &impl,
                                                 uint32_t iter)
{
    result_t res_set_denormals_zero_on, res_set_denormals_zero_off;
    float factor = 2;
    float denormal = FLT_MIN / factor;
    float denormals[4] = {denormal, denormal, denormal, denormal};
    float factors[4] = {factor, factor, factor, factor};
    __m128 ret = _mm_setzero_ps();

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
    for (int i = 0; i < 100000; i++) {
#if defined(_MSC_VER)
        _ReadWriteBarrier();
#else
        __asm__ __volatile__("" ::: "memory");
#endif
    }
    uint64_t end = _rdtsc();
    return end > start ? TEST_SUCCESS : TEST_FAIL;
}

SSE2NEONTestImpl::SSE2NEONTestImpl(void)
{
    mTestFloatPointer1 =
        static_cast<float *>(platformAlignedAlloc(sizeof(__m128)));
    mTestFloatPointer2 =
        static_cast<float *>(platformAlignedAlloc(sizeof(__m128)));
    mTestIntPointer1 =
        static_cast<int32_t *>(platformAlignedAlloc(sizeof(__m128i)));
    mTestIntPointer2 =
        static_cast<int32_t *>(platformAlignedAlloc(sizeof(__m128i)));
    SSE2NEON_INIT_RNG(123456);
    for (uint32_t i = 0; i < MAX_TEST_VALUE; i++) {
        mTestFloats[i] = ranf(-100000, 100000);
        mTestInts[i] = static_cast<int32_t>(ranf(-100000, 100000));
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
#define _(x)                      \
    case it_##x:                  \
        ret = test_##x(*this, i); \
        break;
        INTRIN_LIST
#undef _
    }

    return ret;
}

SSE2NEONTest *SSE2NEONTest::create(void)
{
    SSE2NEONTestImpl *st = new SSE2NEONTestImpl;
    return static_cast<SSE2NEONTest *>(st);
}

}  // namespace SSE2NEON
