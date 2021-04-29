#include "impl.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include "binding.h"

// Try 10,000 random floating point values for each test we run
#define MAX_TEST_VALUE 10000

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

            if (test == it_mm_cmpord_ps || test == it_mm_comilt_ss ||
                test == it_mm_comile_ss || test == it_mm_comige_ss ||
                test == it_mm_comieq_ss || test == it_mm_comineq_ss ||
                test == it_mm_comigt_ss) {  // if testing for NAN's make sure we
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
                result_t ok = test_mm_comilt_ss(mTestFloatPointer1, mTestFloatPointer1);
                if (ok == TEST_FAIL) {
                    printf("Debug me");
                }
            }
#endif
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
result_t test_mm_slli_si128(const SSE2NEONTestImpl &impl, uint32_t i);
result_t test_mm_srli_si128(const SSE2NEONTestImpl &impl, uint32_t i);

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
__m64 do_mm_load_m64(const int64_t *p)
{
    __m64 a = *((const __m64 *) p);
    validateInt64(a, p[0]);
    return a;
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_load_ps`.
__m128 do_mm_load_ps(const float *p)
{
    __m128 a = _mm_load_ps(p);
    validateFloat(a, p[0], p[1], p[2], p[3]);
    return a;
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_load_ps`.
__m128i do_mm_load_ps(const int32_t *p)
{
    __m128 a = _mm_load_ps((const float *) p);
    __m128i ia = *(const __m128i *) &a;
    validateInt32(ia, p[0], p[1], p[2], p[3]);
    return ia;
}

// This function is not called from `runSingleTest`, but for other intrinsic
// tests that might need to call `_mm_load_pd`.
__m128d do_mm_load_pd(const double *p)
{
    __m128d a = _mm_load_pd(p);
    validateDouble(a, p[0], p[1]);
    return a;
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

float compord(float a, float b)
{
    float ret;

    bool isNANA = isNAN(a);
    bool isNANB = isNAN(b);
    ret = (!isNANA && !isNANB) ? getNAN() : 0.0f;
    return ret;
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
    crc = canonical_crc32_u32((uint32_t)(crc), v & 0xffffffff);
    crc = canonical_crc32_u32((uint32_t)(crc), (v >> 32) & 0xffffffff);
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

/* SSE */
result_t test_mm_add_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float dx = _a[0] + _b[0];
    float dy = _a[1] + _b[1];
    float dz = _a[2] + _b[2];
    float dw = _a[3] + _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_add_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_add_ss(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_and_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
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
result_t test_mm_andnot_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    result_t r = TEST_FAIL;
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
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

result_t test_mm_avg_pu16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;
    uint16_t d0 = (_a[0] + _b[0] + 1) >> 1;
    uint16_t d1 = (_a[1] + _b[1] + 1) >> 1;
    uint16_t d2 = (_a[2] + _b[2] + 1) >> 1;
    uint16_t d3 = (_a[3] + _b[3] + 1) >> 1;

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_avg_pu16(a, b);

    return validateUInt16(c, d0, d1, d2, d3);
}

result_t test_mm_avg_pu8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_avg_pu8(a, b);

    return validateUInt8(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_cmpeq_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] == _b[0] ? -1 : 0;
    result[1] = _a[1] == _b[1] ? -1 : 0;
    result[2] = _a[2] == _b[2] ? -1 : 0;
    result[3] = _a[3] == _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpeq_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpeq_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = _a[0] == _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpeq_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpge_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] >= _b[0] ? -1 : 0;
    result[1] = _a[1] >= _b[1] ? -1 : 0;
    result[2] = _a[2] >= _b[2] ? -1 : 0;
    result[3] = _a[3] >= _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpge_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpge_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = _a[0] >= _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpge_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpgt_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] > _b[0] ? -1 : 0;
    result[1] = _a[1] > _b[1] ? -1 : 0;
    result[2] = _a[2] > _b[2] ? -1 : 0;
    result[3] = _a[3] > _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpgt_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpgt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = _a[0] > _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpgt_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmple_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] <= _b[0] ? -1 : 0;
    result[1] = _a[1] <= _b[1] ? -1 : 0;
    result[2] = _a[2] <= _b[2] ? -1 : 0;
    result[3] = _a[3] <= _b[3] ? -1 : 0;

    __m128 ret = _mm_cmple_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmple_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = _a[0] <= _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmple_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmplt_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] < _b[0] ? -1 : 0;
    result[1] = _a[1] < _b[1] ? -1 : 0;
    result[2] = _a[2] < _b[2] ? -1 : 0;
    result[3] = _a[3] < _b[3] ? -1 : 0;

    __m128 ret = _mm_cmplt_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmplt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = _a[0] < _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmplt_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpneq_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] != _b[0] ? -1 : 0;
    result[1] = _a[1] != _b[1] ? -1 : 0;
    result[2] = _a[2] != _b[2] ? -1 : 0;
    result[3] = _a[3] != _b[3] ? -1 : 0;

    __m128 ret = _mm_cmpneq_ps(a, b);
    __m128i iret = *(const __m128i *) &ret;
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpneq_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = _a[0] != _b[0] ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpneq_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnge_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = !(_a[0] >= _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = !(_a[1] >= _b[1]) ? ALL_BIT_1_32 : 0;
    result[2] = !(_a[2] >= _b[2]) ? ALL_BIT_1_32 : 0;
    result[3] = !(_a[3] >= _b[3]) ? ALL_BIT_1_32 : 0;

    __m128 ret = _mm_cmpnge_ps(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnge_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = !(_a[0] >= _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpnge_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpngt_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = !(_a[0] > _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = !(_a[1] > _b[1]) ? ALL_BIT_1_32 : 0;
    result[2] = !(_a[2] > _b[2]) ? ALL_BIT_1_32 : 0;
    result[3] = !(_a[3] > _b[3]) ? ALL_BIT_1_32 : 0;

    __m128 ret = _mm_cmpngt_ps(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpngt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = !(_a[0] > _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpngt_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnle_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = !(_a[0] <= _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = !(_a[1] <= _b[1]) ? ALL_BIT_1_32 : 0;
    result[2] = !(_a[2] <= _b[2]) ? ALL_BIT_1_32 : 0;
    result[3] = !(_a[3] <= _b[3]) ? ALL_BIT_1_32 : 0;

    __m128 ret = _mm_cmpnle_ps(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnle_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = !(_a[0] <= _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpnle_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnlt_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = !(_a[0] < _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = !(_a[1] < _b[1]) ? ALL_BIT_1_32 : 0;
    result[2] = !(_a[2] < _b[2]) ? ALL_BIT_1_32 : 0;
    result[3] = !(_a[3] < _b[3]) ? ALL_BIT_1_32 : 0;

    __m128 ret = _mm_cmpnlt_ps(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpnlt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = !(_a[0] < _b[0]) ? ALL_BIT_1_32 : 0;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpnlt_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpord_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];

    for (uint32_t i = 0; i < 4; i++) {
        result[i] = compord(_a[i], _b[i]);
    }

    __m128 ret = _mm_cmpord_ps(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpord_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = compord(_a[0], _b[0]);
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpord_ss(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpunord_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];

    for (uint32_t i = 0; i < 4; i++) {
        result[i] = (isNAN(_a[i]) || isNAN(_b[i])) ? getNAN() : 0.0f;
    }

    __m128 ret = _mm_cmpunord_ps(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpunord_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = (isNAN(_a[0]) || isNAN(_b[0])) ? getNAN() : 0.0f;
    result[1] = _a[1];
    result[2] = _a[2];
    result[3] = _a[3];

    __m128 ret = _mm_cmpunord_ss(a, b);

    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_comieq_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    if (isNAN(_a[0]) || isNAN(_b[0]))
        // Test disabled: GCC and Clang on x86_64 return different values.
        return TEST_SUCCESS;

    int32_t result = comieq_ss(_a[0], _b[0]);
    int32_t ret = _mm_comieq_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_comige_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    int32_t result = comige_ss(_a[0], _b[0]);
    int32_t ret = _mm_comige_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_comigt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    int32_t result = comigt_ss(_a[0], _b[0]);
    int32_t ret = _mm_comigt_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_comile_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    if (isNAN(_a[0]) || isNAN(_b[0]))
        // Test disabled: GCC and Clang on x86_64 return different values.
        return TEST_SUCCESS;

    int32_t result = comile_ss(_a[0], _b[0]);
    int32_t ret = _mm_comile_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_comilt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    if (isNAN(_a[0]) || isNAN(_b[0]))
        // Test disabled: GCC and Clang on x86_64 return different values.
        return TEST_SUCCESS;

    int32_t result = comilt_ss(_a[0], _b[0]);

    int32_t ret = _mm_comilt_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_comineq_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    if (isNAN(_a[0]) || isNAN(_b[0]))
        // Test disabled: GCC and Clang on x86_64 return different values.
        return TEST_SUCCESS;

    int32_t result = comineq_ss(_a[0], _b[0]);
    int32_t ret = _mm_comineq_ss(a, b);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvt_pi2ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t *_b = impl.mTestIntPointer2;

    float dx = (float) _b[0];
    float dy = (float) _b[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m128 c = _mm_cvt_pi2ps(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvt_ps2pi(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    for (int i = 0; i < 2; i++) {
        int32_t f = (int32_t) floor(_a[i]);
        int32_t c = (int32_t) ceil(_a[i]);
        float diff = _a[i] - floor(_a[i]);
        // Round to nearest, ties to even
        if (diff > 0.5)
            d[i] = c;
        else if (diff == 0.5)
            d[i] = c & 1 ? f : c;
        else
            d[i] = f;
    }

    __m128 a = do_mm_load_ps(_a);
    __m64 ret = _mm_cvt_ps2pi(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvt_si2ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t b = *impl.mTestIntPointer2;

    float dx = (float) b;
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_cvt_si2ss(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvt_ss2si(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d0;
    int32_t f = (int32_t) floor(_a[0]);
    int32_t c = (int32_t) ceil(_a[0]);
    float diff = _a[0] - floor(_a[0]);
    // Round to nearest, ties to even
    if (diff > 0.5)
        d0 = c;
    else if (diff == 0.5)
        d0 = c & 1 ? f : c;
    else
        d0 = f;

    __m128 a = do_mm_load_ps(_a);
    int32_t ret = _mm_cvt_ss2si(a);
    return ret == d0 ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtpi16_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _a[2];
    float dw = (float) _a[3];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m128 c = _mm_cvtpi16_ps(a);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtpi32_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    float dx = (float) _b[0];
    float dy = (float) _b[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m128 c = _mm_cvtpi32_ps(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtpi32x2_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _b[0];
    float dw = (float) _b[1];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m128 c = _mm_cvtpi32x2_ps(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtpi8_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _a[2];
    float dw = (float) _a[3];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m128 c = _mm_cvtpi8_ps(a);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtps_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    float _b[4];
    int16_t trun[4];

    // Beyond int16_t range _mm_cvtps_pi16 function (both native and arm)
    // do not behave the same as BankersRounding.
    // Forcing the float input values to be in the int16_t range
    // Dividing by 10.0f ensures (with the current data set) it,
    // without forcing a saturation.
    for (int j = 0; j < 4; j++) {
        _b[j] = fabsf(_a[j]) > 32767.0f ? _a[j] / 10.0f : _a[j];
        trun[j] = (int16_t)(bankersRounding(_b[j]));
    }

    __m128 b = do_mm_load_ps(_b);
    __m64 ret = _mm_cvtps_pi16(b);
    return validateInt16(ret, trun[0], trun[1], trun[2], trun[3]);
}

result_t test_mm_cvtps_pi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    for (int i = 0; i < 2; i++) {
        int32_t f = (int32_t) floor(_a[i]);
        int32_t c = (int32_t) ceil(_a[i]);
        float diff = _a[i] - floor(_a[i]);
        // Round to nearest, ties to even
        if (diff > 0.5)
            d[i] = c;
        else if (diff == 0.5)
            d[i] = c & 1 ? f : c;
        else
            d[i] = f;
    }

    __m128 a = do_mm_load_ps(_a);
    __m64 ret = _mm_cvtps_pi32(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvtps_pi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtpu16_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _a[2];
    float dw = (float) _a[3];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m128 c = _mm_cvtpu16_ps(a);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtpu8_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;

    float dx = (float) _a[0];
    float dy = (float) _a[1];
    float dz = (float) _a[2];
    float dw = (float) _a[3];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m128 c = _mm_cvtpu8_ps(a);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtsi32_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const int32_t b = *impl.mTestIntPointer2;

    float dx = (float) b;
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_cvtsi32_ss(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtsi64_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const int64_t b = *(int64_t *) impl.mTestIntPointer2;

    float dx = (float) b;
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_cvtsi64_ss(a, b);

    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_cvtss_f32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;

    float f = _a[0];

    __m128 a = do_mm_load_ps(_a);
    float c = _mm_cvtss_f32(a);

    return f == c ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtss_si32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;

    int32_t d0;
    int32_t f = (int32_t) floor(_a[0]);
    int32_t c = (int32_t) ceil(_a[0]);
    float diff = _a[0] - floor(_a[0]);
    // Round to nearest, ties to even
    if (diff > 0.5)
        d0 = c;
    else if (diff == 0.5)
        d0 = c & 1 ? f : c;
    else
        d0 = f;

    __m128 a = do_mm_load_ps(_a);
    int32_t ret = _mm_cvtss_si32(a);

    return ret == d0 ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtss_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;

    int64_t d0;
    int64_t f = (int64_t) floor(_a[0]);
    int64_t c = (int64_t) ceil(_a[0]);
    float diff = _a[0] - floor(_a[0]);
    // Round to nearest, ties to even
    if (diff > 0.5)
        d0 = c;
    else if (diff == 0.5)
        d0 = c & 1 ? f : c;
    else
        d0 = f;

    __m128 a = do_mm_load_ps(_a);
    int64_t ret = _mm_cvtss_si64(a);

    return ret == d0 ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtt_ps2pi(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    d[0] = (int32_t) _a[0];
    d[1] = (int32_t) _a[1];

    __m128 a = do_mm_load_ps(_a);
    __m64 ret = _mm_cvtt_ps2pi(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvtt_ss2si(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = do_mm_load_ps(_a);
    int ret = _mm_cvtt_ss2si(a);

    return ret == (int32_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvttps_pi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    int32_t d[2];

    d[0] = (int32_t) _a[0];
    d[1] = (int32_t) _a[1];

    __m128 a = do_mm_load_ps(_a);
    __m64 ret = _mm_cvttps_pi32(a);

    return validateInt32(ret, d[0], d[1]);
}

result_t test_mm_cvttss_si32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = do_mm_load_ps(_a);
    int ret = _mm_cvttss_si32(a);

    return ret == (int32_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvttss_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;

    __m128 a = do_mm_load_ps(_a);
    int64_t ret = _mm_cvttss_si64(a);

    return ret == (int64_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_div_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float f0 = _a[0] / _b[0];
    float f1 = _a[1] / _b[1];
    float f2 = _a[2] / _b[2];
    float f3 = _a[3] / _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
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

result_t test_mm_div_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float d0 = _a[0] / _b[0];
    float d1 = _a[1];
    float d2 = _a[2];
    float d3 = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
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

result_t test_mm_extract_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // FIXME GCC has bug on `_mm_extract_pi16` intrinsics. We will enable this
    // test when GCC fix this bug.
    // see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98495 for more
    // information
#if defined(__clang__)
    uint64_t *_a = (uint64_t *) impl.mTestIntPointer1;
    const int imm = 1;

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    int32_t c = _mm_extract_pi16(a, imm);
    ASSERT_RETURN((uint64_t) c == ((*_a >> ((imm & 0x3) * 16)) & 0xFFFF));
    ASSERT_RETURN(0 == ((uint64_t) c & 0xFFFF0000));
    return TEST_SUCCESS;
#else
    return TEST_UNIMPL;
#endif
}

result_t test_mm_free(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_getcsr(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_insert_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t insert = (int16_t) impl.mTestInts[i];
    const int imm8 = 2;

    int16_t d[4];
    for (int i = 0; i < 4; i++) {
        d[i] = _a[i];
    }
    d[imm8] = insert;

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = _mm_insert_pi16(a, insert, imm8);

    return validateInt16(b, d[0], d[1], d[2], d[3]);
}

result_t test_mm_load_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_load_ps(addr);

    return validateFloat(ret, addr[0], addr[1], addr[2], addr[3]);
}

result_t test_mm_load_ps1(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_load_ps1(addr);

    return validateFloat(ret, addr[0], addr[0], addr[0], addr[0]);
}

result_t test_mm_load_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_load_ss(addr);

    return validateFloat(ret, addr[0], 0, 0, 0);
}

result_t test_mm_load1_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *p = impl.mTestFloatPointer1;
    __m128 a = _mm_load1_ps(p);
    return validateFloat(a, p[0], p[0], p[0], p[0]);
}

result_t test_mm_loadh_pi(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *p1 = impl.mTestFloatPointer1;
    const float *p2 = impl.mTestFloatPointer2;
    const __m64 *b = (const __m64 *) p2;
    __m128 a = _mm_load_ps(p1);
    __m128 c = _mm_loadh_pi(a, b);

    return validateFloat(c, p1[0], p1[1], p2[0], p2[1]);
}

result_t test_mm_loadl_pi(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *p1 = impl.mTestFloatPointer1;
    const float *p2 = impl.mTestFloatPointer2;
    __m128 a = _mm_load_ps(p1);
    const __m64 *b = (const __m64 *) p2;
    __m128 c = _mm_loadl_pi(a, b);

    return validateFloat(c, p2[0], p2[1], p1[2], p1[3]);
}

result_t test_mm_loadr_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_loadr_ps(addr);

    return validateFloat(ret, addr[3], addr[2], addr[1], addr[0]);
}

result_t test_mm_loadu_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *addr = impl.mTestFloatPointer1;

    __m128 ret = _mm_loadu_ps(addr);

    return validateFloat(ret, addr[0], addr[1], addr[2], addr[3]);
}

result_t test_mm_loadu_si16(const SSE2NEONTestImpl &impl, uint32_t i)
{
#if defined(__clang__)
    const int16_t *addr = (const int16_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_loadu_si16((const void *) addr);

    return validateInt16(ret, addr[0], 0, 0, 0, 0, 0, 0, 0);
#else
    // The intrinsic _mm_loadu_si16() does not exist in GCC
    return TEST_UNIMPL;
#endif
}

result_t test_mm_loadu_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *addr = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_loadu_si64((const void *) addr);

    return validateInt64(ret, addr[0], 0);
}

result_t test_mm_malloc(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_maskmove_si64(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_m_maskmovq(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_maskmove_si64(impl, i);
}

result_t test_mm_max_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t c[4];

    c[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] > _b[3] ? _a[3] : _b[3];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 ret = _mm_max_pi16(a, b);
    return validateInt16(ret, c[0], c[1], c[2], c[3]);
}

result_t test_mm_max_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float c[4];

    c[0] = _a[0] > _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] > _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] > _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 ret = _mm_max_ps(a, b);
    return validateFloat(ret, c[0], c[1], c[2], c[3]);
}

result_t test_mm_max_pu8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 ret = _mm_max_pu8(a, b);
    return validateUInt8(ret, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}

result_t test_mm_max_ss(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_min_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t c[4];

    c[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] < _b[3] ? _a[3] : _b[3];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 ret = _mm_min_pi16(a, b);
    return validateInt16(ret, c[0], c[1], c[2], c[3]);
}

result_t test_mm_min_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float c[4];

    c[0] = _a[0] < _b[0] ? _a[0] : _b[0];
    c[1] = _a[1] < _b[1] ? _a[1] : _b[1];
    c[2] = _a[2] < _b[2] ? _a[2] : _b[2];
    c[3] = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 ret = _mm_min_ps(a, b);
    return validateFloat(ret, c[0], c[1], c[2], c[3]);
}

result_t test_mm_min_pu8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 ret = _mm_min_pu8(a, b);
    return validateUInt8(ret, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}

result_t test_mm_min_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float c;

    c = _a[0] < _b[0] ? _a[0] : _b[0];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 ret = _mm_min_ss(a, b);

    return validateFloat(ret, c, _a[1], _a[2], _a[3]);
}

result_t test_mm_move_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

    float result[4];
    result[0] = b[0];
    result[1] = a[1];
    result[2] = a[2];
    result[3] = a[3];

    __m128 ret = _mm_move_ss(a, b);
    return validateFloat(ret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_movehl_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _b[2];
    float f1 = _b[3];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 ret = _mm_movehl_ps(a, b);

    return validateFloat(ret, f0, f1, f2, f3);
}

result_t test_mm_movelh_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _a[0];
    float f1 = _a[1];
    float f2 = _b[0];
    float f3 = _b[1];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 ret = _mm_movelh_ps(a, b);

    return validateFloat(ret, f0, f1, f2, f3);
}

result_t test_mm_movemask_pi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_movemask_ps(const SSE2NEONTestImpl &impl, uint32_t i)
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
    __m128 a = do_mm_load_ps(p);
    int val = _mm_movemask_ps(a);
    return val == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_mul_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float dx = _a[0] * _b[0];
    float dy = _a[1] * _b[1];
    float dz = _a[2] * _b[2];
    float dw = _a[3] * _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_mul_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_mul_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float dx = _a[0] * _b[0];
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_mul_ss(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_mulhi_pu16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;
    uint16_t d[4];
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t m = (uint32_t) _a[i] * (uint32_t) _b[i];
        d[i] = (uint16_t)(m >> 16);
    }

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_mulhi_pu16(a, b);
    return validateUInt16(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_or_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
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

result_t test_m_pavgb(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_avg_pu8(impl, i);
}

result_t test_m_pavgw(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_avg_pu16(impl, i);
}

result_t test_m_pextrw(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_extract_pi16(impl, i);
}

result_t test_m_pinsrw(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_insert_pi16(impl, i);
}

result_t test_m_pmaxsw(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_max_pi16(impl, i);
}

result_t test_m_pmaxub(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_max_pu8(impl, i);
}

result_t test_m_pminsw(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_min_pi16(impl, i);
}

result_t test_m_pminub(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_min_pu8(impl, i);
}

result_t test_m_pmovmskb(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_movemask_pi8(impl, i);
}

result_t test_m_pmulhuw(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_mulhi_pu16(impl, i);
}

result_t test_mm_prefetch(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_m_psadbw(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    uint16_t d = 0;
    for (int i = 0; i < 8; i++) {
        d += abs(_a[i] - _b[i]);
    }

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_sad_pu8(a, b);
    return validateUInt16(c, d, 0, 0, 0);
}

result_t test_m_pshufw(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_rcp_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    float dx = 1.0f / _a[0];
    float dy = 1.0f / _a[1];
    float dz = 1.0f / _a[2];
    float dw = 1.0f / _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_rcp_ps(a);
    return validateFloatEpsilon(c, dx, dy, dz, dw, 300.0f);
}

result_t test_mm_rcp_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;

    float dx = 1.0f / _a[0];
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_rcp_ss(a);
    return validateFloatEpsilon(c, dx, dy, dz, dw, 300.0f);
}

result_t test_mm_rsqrt_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    float f0 = 1 / sqrt(_a[0]);
    float f1 = 1 / sqrt(_a[1]);
    float f2 = 1 / sqrt(_a[2]);
    float f3 = 1 / sqrt(_a[3]);

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_rsqrt_ps(a);

    // Here, we ensure `_mm_rsqrt_ps()`'s error is under 1% compares to the C
    // implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.01f);
}

result_t test_mm_rsqrt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    float f0 = 1 / sqrt(_a[0]);
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_rsqrt_ss(a);

    // Here, we ensure `_mm_rsqrt_ps()`'s error is under 1% compares to the C
    // implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.01f);
}

result_t test_mm_sad_pu8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    uint16_t d = 0;
    for (int i = 0; i < 8; i++) {
        d += abs(_a[i] - _b[i]);
    }

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_sad_pu8(a, b);
    return validateUInt16(c, d, 0, 0, 0);
}

result_t test_mm_set_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float x = impl.mTestFloats[i];
    float y = impl.mTestFloats[i + 1];
    float z = impl.mTestFloats[i + 2];
    float w = impl.mTestFloats[i + 3];
    __m128 a = _mm_set_ps(x, y, z, w);
    return validateFloat(a, w, z, y, x);
}

result_t test_mm_set_ps1(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float a = impl.mTestFloats[i];

    __m128 ret = _mm_set_ps1(a);

    return validateFloat(ret, a, a, a, a);
}

result_t test_mm_set_rounding_mode(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    result_t res_toward_zero, res_to_neg_inf, res_to_pos_inf, res_nearest;

    __m128 a = do_mm_load_ps(_a);
    __m128 b, c;

// _MM_ROUND_TOWARD_ZERO has not the expected behavior on aarch32
#if defined(__arm__)
    res_toward_zero = TEST_SUCCESS;
#else
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
    b = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    c = _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    res_toward_zero =
        validateFloatEpsilon(c, ((float *) &b)[0], ((float *) &b)[1],
                             ((float *) &b)[2], ((float *) &b)[3], 5.0f);
#endif

    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    b = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    c = _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    res_to_neg_inf =
        validateFloatEpsilon(c, ((float *) &b)[0], ((float *) &b)[1],
                             ((float *) &b)[2], ((float *) &b)[3], 5.0f);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    b = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    c = _mm_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    res_to_pos_inf =
        validateFloatEpsilon(c, ((float *) &b)[0], ((float *) &b)[1],
                             ((float *) &b)[2], ((float *) &b)[3], 5.0f);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    b = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    c = _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    res_nearest =
        validateFloatEpsilon(c, ((float *) &b)[0], ((float *) &b)[1],
                             ((float *) &b)[2], ((float *) &b)[3], 5.0f);

    if (res_toward_zero == TEST_SUCCESS && res_to_neg_inf == TEST_SUCCESS &&
        res_to_pos_inf == TEST_SUCCESS && res_nearest == TEST_SUCCESS) {
        return TEST_SUCCESS;
    } else {
        return TEST_FAIL;
    }
}

result_t test_mm_set_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float a = impl.mTestFloats[i];
    __m128 c = _mm_set_ss(a);
    return validateFloat(c, a, 0, 0, 0);
}

result_t test_mm_set1_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float w = impl.mTestFloats[i];
    __m128 a = _mm_set1_ps(w);
    return validateFloat(a, w, w, w, w);
}

result_t test_mm_setcsr(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_set_rounding_mode(impl, i);
}

result_t test_mm_setr_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float x = impl.mTestFloats[i];
    float y = impl.mTestFloats[i + 1];
    float z = impl.mTestFloats[i + 2];
    float w = impl.mTestFloats[i + 3];

    __m128 ret = _mm_setr_ps(w, z, y, x);

    return validateFloat(ret, w, z, y, x);
}

result_t test_mm_setzero_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    __m128 a = _mm_setzero_ps();
    return validateFloat(a, 0, 0, 0, 0);
}

result_t test_mm_sfence(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_shuffle_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

// Note, NEON does not have a general purpose shuffled command like SSE.
// When invoking this method, there is special code for a number of the most
// common shuffle permutations
result_t test_mm_shuffle_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    result_t isValid = TEST_SUCCESS;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
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

result_t test_mm_sqrt_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    float f0 = sqrt(_a[0]);
    float f1 = sqrt(_a[1]);
    float f2 = sqrt(_a[2]);
    float f3 = sqrt(_a[3]);

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_sqrt_ps(a);

    // Here, we ensure `_mm_sqrt_ps()`'s error is under 1% compares to the C
    // implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.01f);
}

result_t test_mm_sqrt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    float f0 = sqrt(_a[0]);
    float f1 = _a[1];
    float f2 = _a[2];
    float f3 = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_sqrt_ss(a);

    // Here, we ensure `_mm_sqrt_ps()`'s error is under 1% compares to the C
    // implementation.
    return validateFloatError(c, f0, f1, f2, f3, 0.01f);
}

result_t test_mm_store_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    int32_t *p = impl.mTestIntPointer1;
    int32_t x = impl.mTestInts[i];
    int32_t y = impl.mTestInts[i + 1];
    int32_t z = impl.mTestInts[i + 2];
    int32_t w = impl.mTestInts[i + 3];
    __m128i a = _mm_set_epi32(x, y, z, w);
    _mm_store_ps((float *) p, *(const __m128 *) &a);
    ASSERT_RETURN(p[0] == w);
    ASSERT_RETURN(p[1] == z);
    ASSERT_RETURN(p[2] == y);
    ASSERT_RETURN(p[3] == x);
    return TEST_SUCCESS;
}

result_t test_mm_store_ps1(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float *p = impl.mTestFloatPointer1;
    float d[4];

    __m128 a = do_mm_load_ps(p);
    _mm_store_ps1(d, a);

    ASSERT_RETURN(d[0] == *p);
    ASSERT_RETURN(d[1] == *p);
    ASSERT_RETURN(d[2] == *p);
    ASSERT_RETURN(d[3] == *p);
    return TEST_SUCCESS;
}

result_t test_mm_store_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float x = impl.mTestFloats[i];
    float p[4];

    __m128 a = _mm_set_ss(x);
    _mm_store_ss(p, a);
    ASSERT_RETURN(p[0] == x);
    return TEST_SUCCESS;
}

result_t test_mm_store1_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float *p = impl.mTestFloatPointer1;
    float d[4];

    __m128 a = do_mm_load_ps(p);
    _mm_store1_ps(d, a);

    ASSERT_RETURN(d[0] == *p);
    ASSERT_RETURN(d[1] == *p);
    ASSERT_RETURN(d[2] == *p);
    ASSERT_RETURN(d[3] == *p);
    return TEST_SUCCESS;
}

result_t test_mm_storeh_pi(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_storel_pi(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_storer_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float *p = impl.mTestFloatPointer1;
    float d[4];

    __m128 a = do_mm_load_ps(p);
    _mm_storer_ps(d, a);

    ASSERT_RETURN(d[0] == p[3]);
    ASSERT_RETURN(d[1] == p[2]);
    ASSERT_RETURN(d[2] == p[1]);
    ASSERT_RETURN(d[3] == p[0]);
    return TEST_SUCCESS;
}

result_t test_mm_storeu_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    float *_a = impl.mTestFloatPointer1;
    float f[4];
    __m128 a = _mm_load_ps(_a);

    _mm_storeu_ps(f, a);
    return validateFloat(a, f[0], f[1], f[2], f[3]);
}

result_t test_mm_storeu_si16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_storeu_si16. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if defined(__GNUC__) && __GNUC__ <= 10
    return TEST_UNIMPL;
#else
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i b;
    __m128i a = do_mm_load_ps(_a);
    _mm_storeu_si16(&b, a);
    int16_t *_b = (int16_t *) &b;
    int16_t *_c = (int16_t *) &a;
    return validateInt16(b, _c[0], _b[1], _b[2], _b[3], _b[4], _b[5], _b[6],
                         _b[7]);
#endif
}

result_t test_mm_storeu_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i b;
    __m128i a = do_mm_load_ps(_a);
    _mm_storeu_si64(&b, a);
    int64_t *_b = (int64_t *) &b;
    int64_t *_c = (int64_t *) &a;
    return validateInt64(b, _c[0], _b[1]);
}

result_t test_mm_stream_pi(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    __m64 a = do_mm_load_m64(_a);
    __m64 p;

    _mm_stream_pi(&p, a);
    return validateInt64(p, _a[0]);
}

result_t test_mm_stream_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    __m128 a = do_mm_load_ps(_a);
    float p[4];

    _mm_stream_ps(p, a);
    ASSERT_RETURN(p[0] == _a[0]);
    ASSERT_RETURN(p[1] == _a[1]);
    ASSERT_RETURN(p[2] == _a[2]);
    ASSERT_RETURN(p[3] == _a[3]);
    return TEST_SUCCESS;
}

result_t test_mm_sub_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float dx = _a[0] - _b[0];
    float dy = _a[1] - _b[1];
    float dz = _a[2] - _b[2];
    float dw = _a[3] - _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_sub_ps(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_sub_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    float dx = _a[0] - _b[0];
    float dy = _a[1];
    float dz = _a[2];
    float dw = _a[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_sub_ss(a, b);
    return validateFloat(c, dx, dy, dz, dw);
}

result_t test_mm_ucomieq_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // _mm_ucomieq_ss is equal to _mm_comieq_ss
    return test_mm_comieq_ss(impl, i);
}

result_t test_mm_ucomige_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // _mm_ucomige_ss is equal to _mm_comige_ss
    return test_mm_comige_ss(impl, i);
}

result_t test_mm_ucomigt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // _mm_ucomigt_ss is equal to _mm_comigt_ss
    return test_mm_comigt_ss(impl, i);
}

result_t test_mm_ucomile_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // _mm_ucomile_ss is equal to _mm_comile_ss
    return test_mm_comile_ss(impl, i);
}

result_t test_mm_ucomilt_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // _mm_ucomilt_ss is equal to _mm_comilt_ss
    return test_mm_comilt_ss(impl, i);
}

result_t test_mm_ucomineq_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // _mm_ucomineq_ss is equal to _mm_comineq_ss
    return test_mm_comineq_ss(impl, i);
}

result_t test_mm_undefined_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_unpackhi_ps(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_unpacklo_ps(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_xor_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestFloatPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestFloatPointer2;

    int32_t d0 = _a[0] ^ _b[0];
    int32_t d1 = _a[1] ^ _b[1];
    int32_t d2 = _a[2] ^ _b[2];
    int32_t d3 = _a[3] ^ _b[3];

    __m128 a = do_mm_load_ps((const float *) _a);
    __m128 b = do_mm_load_ps((const float *) _b);
    __m128 c = _mm_xor_ps(a, b);

    return validateFloat(c, *((float *) &d0), *((float *) &d1),
                         *((float *) &d2), *((float *) &d3));
}

/* SSE2 */
result_t test_mm_add_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_add_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_add_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t dx = _a[0] + _b[0];
    int32_t dy = _a[1] + _b[1];
    int32_t dz = _a[2] + _b[2];
    int32_t dw = _a[3] + _b[3];

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i c = _mm_add_epi32(a, b);
    return validateInt32(c, dx, dy, dz, dw);
}

result_t test_mm_add_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t d0 = _a[0] + _b[0];
    int64_t d1 = _a[1] + _b[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_add_epi64(a, b);

    return validateInt64(c, d0, d1);
}

result_t test_mm_add_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_add_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_add_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] + _b[0];
    double d1 = _a[1] + _b[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_add_pd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_add_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] + _b[0];
    double d1 = _a[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_add_sd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_add_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t d0 = _a[0] + _b[0];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_add_si64(a, b);

    return validateInt64(c, d0);
}

result_t test_mm_adds_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);

    __m128i c = _mm_adds_epi16(a, b);
    return validateInt16(c, (int16_t) d0, (int16_t) d1, (int16_t) d2,
                         (int16_t) d3, (int16_t) d4, (int16_t) d5, (int16_t) d6,
                         (int16_t) d7);
}

result_t test_mm_adds_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_adds_epi8(a, b);

    return validateInt8(
        c, (int8_t) d[0], (int8_t) d[1], (int8_t) d[2], (int8_t) d[3],
        (int8_t) d[4], (int8_t) d[5], (int8_t) d[6], (int8_t) d[7],
        (int8_t) d[8], (int8_t) d[9], (int8_t) d[10], (int8_t) d[11],
        (int8_t) d[12], (int8_t) d[13], (int8_t) d[14], (int8_t) d[15]);
}

result_t test_mm_adds_epu16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_adds_epu16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_adds_epu8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_adds_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_and_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestFloatPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestFloatPointer2;

    int64_t d0 = _a[0] & _b[0];
    int64_t d1 = _a[1] & _b[1];

    __m128d a = do_mm_load_pd((const double *) _a);
    __m128d b = do_mm_load_pd((const double *) _b);
    __m128d c = _mm_and_pd(a, b);

    return validateDouble(c, *((double *) &d0), *((double *) &d1));
}

result_t test_mm_and_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
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

result_t test_mm_andnot_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
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

result_t test_mm_andnot_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    result_t r = TEST_SUCCESS;
    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
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

result_t test_mm_avg_epu16(const SSE2NEONTestImpl &impl, uint32_t i)
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
    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_avg_epu16(a, b);
    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_avg_epu8(const SSE2NEONTestImpl &impl, uint32_t i)
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
    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_avg_epu8(a, b);
    return validateUInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                         d12, d13, d14, d15);
}

result_t test_mm_bslli_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_slli_si128(impl, i);
}

result_t test_mm_bsrli_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_srli_si128(impl, i);
}

result_t test_mm_castpd_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const __m128d a = do_mm_load_pd((const double *) _a);
    const __m128 _c = do_mm_load_ps(_a);

    __m128 r = _mm_castpd_ps(a);

    return validate128(r, _c);
}

result_t test_mm_castpd_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const __m128d a = do_mm_load_pd((const double *) _a);
    const __m128i *_c = (const __m128i *) _a;

    __m128i r = _mm_castpd_si128(a);

    return validate128(r, *_c);
}

result_t test_mm_castps_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const __m128 a = do_mm_load_ps(_a);
    const __m128d *_c = (const __m128d *) _a;

    __m128d r = _mm_castps_pd(a);

    return validate128(r, *_c);
}

result_t test_mm_castps_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;

    const __m128i *_c = (const __m128i *) _a;

    const __m128 a = do_mm_load_ps(_a);
    __m128i r = _mm_castps_si128(a);

    return validate128(r, *_c);
}

result_t test_mm_castsi128_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;

    const __m128d *_c = (const __m128d *) _a;

    const __m128i a = do_mm_load_ps(_a);
    __m128d r = _mm_castsi128_pd(a);

    return validate128(r, *_c);
}

result_t test_mm_castsi128_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;

    const __m128 *_c = (const __m128 *) _a;

    const __m128i a = do_mm_load_ps(_a);
    __m128 r = _mm_castsi128_ps(a);

    return validate128(r, *_c);
}

result_t test_mm_clflush(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpeq_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpeq_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_cmpeq_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;

    int32_t d0 = (_a[0] == _b[0]) ? ~UINT64_C(0) : 0x0;
    int32_t d1 = (_a[1] == _b[1]) ? ~UINT64_C(0) : 0x0;
    int32_t d2 = (_a[2] == _b[2]) ? ~UINT64_C(0) : 0x0;
    int32_t d3 = (_a[3] == _b[3]) ? ~UINT64_C(0) : 0x0;

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i c = _mm_cmpeq_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_cmpeq_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpeq_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_cmpeq_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] == _b[0]) ? 0xffffffffffffffff : 0;
    uint64_t d1 = (_a[1] == _b[1]) ? 0xffffffffffffffff : 0;

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpeq_pd(a, b);
    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpeq_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    const uint64_t d0 = (_a[0] == _b[0]) ? ~UINT64_C(0) : 0;
    const uint64_t d1 = ((const uint64_t *) _a)[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpeq_sd(a, b);

    return validateDouble(c, *(const double *) &d0, *(const double *) &d1);
}

result_t test_mm_cmpge_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] >= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = (_a[1] >= _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpge_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpge_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] >= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpge_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpgt_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpgt_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_cmpgt_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);

    int32_t result[4];

    result[0] = _a[0] > _b[0] ? -1 : 0;
    result[1] = _a[1] > _b[1] ? -1 : 0;
    result[2] = _a[2] > _b[2] ? -1 : 0;
    result[3] = _a[3] > _b[3] ? -1 : 0;

    __m128i iret = _mm_cmpgt_epi32(a, b);
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmpgt_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpgt_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_cmpgt_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] > _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = (_a[1] > _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpgt_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpgt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] > _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpgt_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmple_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = (_a[1] <= _b[1]) ? ~UINT64_C(0) : 0;

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmple_pd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmple_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmple_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmplt_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmplt_epi16(a, b);

    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_cmplt_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);

    int32_t result[4];
    result[0] = _a[0] < _b[0] ? -1 : 0;
    result[1] = _a[1] < _b[1] ? -1 : 0;
    result[2] = _a[2] < _b[2] ? -1 : 0;
    result[3] = _a[3] < _b[3] ? -1 : 0;

    __m128i iret = _mm_cmplt_epi32(a, b);
    return validateInt32(iret, result[0], result[1], result[2], result[3]);
}

result_t test_mm_cmplt_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmplt_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_cmplt_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    int64_t f0 = (_a[0] < _b[0]) ? ~UINT64_C(0) : UINT64_C(0);
    int64_t f1 = (_a[1] < _b[1]) ? ~UINT64_C(0) : UINT64_C(0);

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmplt_pd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmplt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;
    uint64_t d0 = (_a[0] <= _b[0]) ? ~UINT64_C(0) : 0;
    uint64_t d1 = ((uint64_t *) _a)[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmplt_sd(a, b);

    return validateDouble(c, *(double *) &d0, *(double *) &d1);
}

result_t test_mm_cmpneq_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    int64_t f0 = (_a[0] != _b[0]) ? ~UINT64_C(0) : UINT64_C(0);
    int64_t f1 = (_a[1] != _b[1]) ? ~UINT64_C(0) : UINT64_C(0);

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpneq_pd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmpneq_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;

    int64_t f0 = (_a[0] != _b[0]) ? ~UINT64_C(0) : UINT64_C(0);
    int64_t f1 = ((int64_t *) _a)[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpneq_sd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmpnge_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    int64_t f0 = (_a[0] >= _b[0]) ? UINT64_C(0) : ~UINT64_C(0);
    int64_t f1 = (_a[1] >= _b[1]) ? UINT64_C(0) : ~UINT64_C(0);

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpnge_pd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmpnge_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *_a = (double *) impl.mTestFloatPointer1;
    double *_b = (double *) impl.mTestFloatPointer2;

    int64_t f0 = (_a[0] >= _b[0]) ? UINT64_C(0) : ~UINT64_C(0);
    int64_t f1 = ((int64_t *) _a)[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpnge_sd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmpngt_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    int64_t f0 = !(_a[0] > _b[0]) ? ~UINT64_C(0) : UINT64_C(0);
    int64_t f1 = !(_a[1] > _b[1]) ? ~UINT64_C(0) : UINT64_C(0);

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_cmpngt_pd(a, b);

    return validateDouble(c, *(double *) &f0, *(double *) &f1);
}

result_t test_mm_cmpngt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpnle_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpnle_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpnlt_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpnlt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpord_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpord_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpunord_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpunord_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_comieq_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    int32_t _c = (_a[0] == _b[0]) ? 1 : 0;

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    int32_t c = _mm_comieq_sd(a, b);

    printf("c == %d, _c == %d\n", c, _c);
    ASSERT_RETURN(c == _c);
    return TEST_SUCCESS;
}

result_t test_mm_comige_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_comigt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_comile_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_comilt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_comineq_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtepi32_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a = do_mm_load_ps(_a);
    double trun[2] = {(double) _a[0], (double) _a[1]};

    __m128d ret = _mm_cvtepi32_pd(a);
    return validateDouble(ret, trun[0], trun[1]);
}

result_t test_mm_cvtepi32_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a = do_mm_load_ps(_a);
    float trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (float) _a[i];
    }

    __m128 ret = _mm_cvtepi32_ps(a);
    return validateFloat(ret, trun[0], trun[1], trun[2], trun[3]);
}

result_t test_mm_cvtpd_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtpd_pi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtpd_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    float f0 = (float) _a[0];
    float f1 = (float) _a[1];
    const __m128d a = do_mm_load_pd(_a);

    __m128 r = _mm_cvtpd_ps(a);

    return validateFloat(r, f0, f1, 0, 0);
}

result_t test_mm_cvtpi32_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m64 a = do_mm_load_m64((const int64_t *) _a);

    double trun[2] = {(double) _a[0], (double) _a[1]};

    __m128d ret = _mm_cvtpi32_pd(a);

    return validateDouble(ret, trun[0], trun[1]);
}

// https://msdn.microsoft.com/en-us/library/xdc42k5e%28v=vs.90%29.aspx?f=255&MSPPError=-2147217396
result_t test_mm_cvtps_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    __m128 a = do_mm_load_ps(_a);
    int32_t trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (int32_t)(bankersRounding(_a[i]));
    }

    __m128i ret = _mm_cvtps_epi32(a);
    return validateInt32(ret, trun[0], trun[1], trun[2], trun[3]);
}

result_t test_mm_cvtps_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    double d0 = (double) _a[0];
    double d1 = (double) _a[1];
    const __m128 a = do_mm_load_ps(_a);

    __m128d r = _mm_cvtps_pd(a);

    return validateDouble(r, d0, d1);
}

result_t test_mm_cvtsd_f64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    double d = _a[0];

    const __m128d *a = (const __m128d *) _a;
    double r = _mm_cvtsd_f64(*a);

    return r == d ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsd_si32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtsd_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtsd_si64x(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtsd_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtsi128_si32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;

    int32_t d = _a[0];

    __m128i a = do_mm_load_ps(_a);
    int c = _mm_cvtsi128_si32(a);

    return d == c ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsi128_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d = _a[0];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    int64_t c = _mm_cvtsi128_si64(a);

    return d == c ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvtsi128_si64x(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtsi32_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtsi32_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;

    int32_t d = _a[0];

    __m128i c = _mm_cvtsi32_si128(*_a);

    return validateInt32(c, d, 0, 0, 0);
}

result_t test_mm_cvtsi64_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtsi64_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d = _a[0];

    __m128i c = _mm_cvtsi64_si128(*_a);

    return validateInt64(c, d, 0);
}

result_t test_mm_cvtsi64x_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtsi64x_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvtss_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    double d0 = double(_b[0]);
    double d1 = _a[1];

    __m128d a = do_mm_load_pd(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128d c = _mm_cvtss_sd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_cvttpd_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvttpd_pi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvttps_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    __m128 a = do_mm_load_ps(_a);
    int32_t trun[4];
    for (uint32_t i = 0; i < 4; i++) {
        trun[i] = (int32_t) _a[i];
    }

    __m128i ret = _mm_cvttps_epi32(a);
    return validateInt32(ret, trun[0], trun[1], trun[2], trun[3]);
}

result_t test_mm_cvttsd_si32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cvttsd_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    __m128d a = _mm_load_sd(_a);
    int64_t ret = _mm_cvttsd_si64(a);

    return ret == (int64_t) _a[0] ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_cvttsd_si64x(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_div_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = 0.0, d1 = 0.0;

    if (_b[0] != 0.0)
        d0 = _a[0] / _b[0];
    if (_b[1] != 0.0)
        d1 = _a[1] / _b[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_div_pd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_div_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double d0 = _a[0] / _b[0];
    double d1 = _a[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);

    __m128d c = _mm_div_sd(a, b);

    return validateDouble(c, d0, d1);
}

result_t test_mm_extract_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    uint16_t *_a = (uint16_t *) impl.mTestIntPointer1;
    const int imm = 1;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    int c = _mm_extract_epi16(a, imm);
    ASSERT_RETURN(c == *(_a + imm));
    return TEST_SUCCESS;
}

result_t test_mm_insert_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t insert = (int16_t) *impl.mTestIntPointer2;
    const int imm8 = 2;

    int16_t d[8];
    for (int i = 0; i < 8; i++) {
        d[i] = _a[i];
    }
    d[imm8] = insert;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_insert_epi16(a, insert, imm8);
    return validateInt16(b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_lfence(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_load_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = _mm_load_pd(p);
    return validateDouble(a, p[0], p[1]);
}

result_t test_mm_load_pd1(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = _mm_load_pd1(p);
    return validateDouble(a, p[0], p[0]);
}

result_t test_mm_load_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = _mm_load_sd(p);
    return validateDouble(a, p[0], 0);
}

result_t test_mm_load_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *addr = impl.mTestIntPointer1;

    __m128i ret = _mm_load_si128((const __m128i *) addr);

    return validateInt32(ret, addr[0], addr[1], addr[2], addr[3]);
}

result_t test_mm_load1_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *addr = (const double *) impl.mTestFloatPointer1;

    __m128d ret = _mm_load1_pd(addr);

    return validateDouble(ret, addr[0], addr[0]);
}

result_t test_mm_loadh_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *addr = (const double *) impl.mTestFloatPointer2;

    __m128d a = do_mm_load_pd(_a);
    __m128d ret = _mm_loadh_pd(a, addr);

    return validateDouble(ret, _a[0], addr[0]);
}

result_t test_mm_loadl_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *addr = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_loadl_epi64((const __m128i *) addr);

    return validateInt64(ret, addr[0], 0);
}

result_t test_mm_loadl_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *addr = (const double *) impl.mTestFloatPointer2;

    __m128d a = do_mm_load_pd(_a);
    __m128d ret = _mm_loadl_pd(a, addr);

    return validateDouble(ret, addr[0], _a[1]);
}

result_t test_mm_loadr_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *addr = (const double *) impl.mTestFloatPointer1;

    __m128d ret = _mm_loadr_pd(addr);

    return validateDouble(ret, addr[1], addr[0]);
}

result_t test_mm_loadu_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = _mm_loadu_pd(p);
    return validateDouble(a, p[0], p[1]);
}

result_t test_mm_loadu_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i c = _mm_loadu_si128((const __m128i *) _a);
    return validateInt32(c, _a[0], _a[1], _a[2], _a[3]);
}

result_t test_mm_loadu_si32(const SSE2NEONTestImpl &impl, uint32_t i)
{
#if defined(__clang__)
    const int32_t *addr = (const int32_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_loadu_si32((const void *) addr);

    return validateInt32(ret, addr[0], 0, 0, 0);
#else
    // The intrinsic _mm_loadu_si32() does not exist in GCC
    return TEST_UNIMPL;
#endif
}

result_t test_mm_madd_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_madd_epi16(a, b);
    return validateInt32(c, e0, e1, e2, e3);
}

result_t test_mm_maskmoveu_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_mask = (const uint8_t *) impl.mTestIntPointer2;
    char mem_addr[16];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i mask = do_mm_load_ps((const int32_t *) _mask);
    _mm_maskmoveu_si128(a, mask, mem_addr);

    for (int i = 0; i < 16; i++) {
        if (_mask[i] >> 7) {
            ASSERT_RETURN(_a[i] == (uint8_t) mem_addr[i]);
        }
    }

    return TEST_SUCCESS;
}

result_t test_mm_max_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);

    __m128i c = _mm_max_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_max_epu8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_max_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_max_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double f0 = _a[0] > _b[0] ? _a[0] : _b[0];
    double f1 = _a[1] > _b[1] ? _a[1] : _b[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_max_pd(a, b);

    return validateDouble(c, f0, f1);
}

result_t test_mm_max_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = fmax(_a[0], _b[0]);
    double d1 = _a[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_max_sd(a, b);

    return validateDouble(c, d0, d1);
}

result_t test_mm_mfence(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_min_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_min_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_min_epu8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_min_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_min_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double f0 = fmin(_a[0], _b[0]);
    double f1 = fmin(_a[1], _b[1]);

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);

    __m128d c = _mm_min_pd(a, b);
    return validateDouble(c, f0, f1);
}

result_t test_mm_min_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = fmin(_a[0], _b[0]);
    double d1 = _a[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_min_sd(a, b);

    return validateDouble(c, d0, d1);
}

result_t test_mm_move_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d0 = _a[0];
    int64_t d1 = 0;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_move_epi64(a);

    return validateInt64(c, d0, d1);
}

result_t test_mm_move_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);

    double result[2];
    result[0] = _b[0];
    result[1] = _a[1];

    __m128d ret = _mm_move_sd(a, b);
    return validateDouble(ret, result[0], result[1]);
}

result_t test_mm_movemask_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    __m128i a = do_mm_load_ps(_a);

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

result_t test_mm_movemask_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    unsigned int _c = 0;
    _c |= ((*(const uint64_t *) _a) >> 63) & 0x1;
    _c |= (((*(const uint64_t *) (_a + 1)) >> 62) & 0x2);

    __m128d a = do_mm_load_pd(_a);
    int c = _mm_movemask_pd(a);

    ASSERT_RETURN((unsigned int) c == _c);
    return TEST_SUCCESS;
}

result_t test_mm_movepi64_pi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d0 = _a[0];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m64 c = _mm_movepi64_pi64(a);

    return validateInt64(c, d0);
}

result_t test_mm_movpi64_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    int64_t d0 = _a[0];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m128i c = _mm_movpi64_epi64(a);

    return validateInt64(c, d0, 0);
}

result_t test_mm_mul_epu32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;
    const uint32_t *_b = (const uint32_t *) impl.mTestIntPointer2;
    uint64_t dx = (uint64_t)(_a[0]) * (uint64_t)(_b[0]);
    uint64_t dy = (uint64_t)(_a[2]) * (uint64_t)(_b[2]);

    __m128i a = _mm_loadu_si128((const __m128i *) _a);
    __m128i b = _mm_loadu_si128((const __m128i *) _b);
    __m128i r = _mm_mul_epu32(a, b);
    return validateUInt64(r, dx, dy);
}

result_t test_mm_mul_pd(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_mul_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double dx = _a[0] * _b[0];
    double dy = _a[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_mul_sd(a, b);
    return validateDouble(c, dx, dy);
}

result_t test_mm_mul_su32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;
    const uint32_t *_b = (const uint32_t *) impl.mTestIntPointer2;

    uint64_t u = (uint64_t)(_a[0]) * (uint64_t)(_b[0]);

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 r = _mm_mul_su32(a, b);

    return validateUInt64(r, u);
}

result_t test_mm_mulhi_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d[8];
    for (uint32_t i = 0; i < 8; i++) {
        int32_t m = (int32_t) _a[i] * (int32_t) _b[i];
        d[i] = (int16_t)(m >> 16);
    }

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_mulhi_epi16(a, b);
    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_mulhi_epu16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;
    const uint16_t *_b = (const uint16_t *) impl.mTestIntPointer2;
    uint16_t d[8];
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t m = (uint32_t) _a[i] * (uint32_t) _b[i];
        d[i] = (uint16_t)(m >> 16);
    }

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_mulhi_epu16(a, b);
    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_mullo_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_mullo_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_or_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestFloatPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestFloatPointer2;

    int64_t d0 = _a[0] | _b[0];
    int64_t d1 = _a[1] | _b[1];

    __m128d a = do_mm_load_pd((const double *) _a);
    __m128d b = do_mm_load_pd((const double *) _b);
    __m128d c = _mm_or_pd(a, b);

    return validateDouble(c, *((double *) &d0), *((double *) &d1));
}

result_t test_mm_or_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
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

result_t test_mm_packs_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_packs_epi16(a, b);

    return validateInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                        d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_packs_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_packs_epi32(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_packus_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_packus_epi16(a, b);

    return validateUInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                         d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_pause(const SSE2NEONTestImpl &impl, uint32_t i)
{
    _mm_pause();
    return TEST_SUCCESS;
}

result_t test_mm_sad_epu8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    const __m128i a = do_mm_load_ps((const int32_t *) _a);
    const __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sad_epu8(a, b);
    return validateUInt16(c, d0, 0, 0, 0, d1, 0, 0, 0);
}

result_t test_mm_set_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_set_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    int32_t x = impl.mTestInts[i];
    int32_t y = impl.mTestInts[i + 1];
    int32_t z = impl.mTestInts[i + 2];
    int32_t w = impl.mTestInts[i + 3];
    __m128i a = _mm_set_epi32(x, y, z, w);
    return validateInt32(a, w, z, y, x);
}

result_t test_mm_set_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_set_epi64((__m64) _a[1], (__m64) _a[0]);

    return validateInt64(ret, _a[0], _a[1]);
}

result_t test_mm_set_epi64x(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_set_epi64x(_a[1], _a[0]);

    return validateInt64(ret, _a[0], _a[1]);
}

result_t test_mm_set_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_set_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    double x = p[0];
    double y = p[1];
    __m128d a = _mm_set_pd(x, y);
    return validateDouble(a, y, x);
}

result_t test_mm_set_pd1(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double _a = impl.mTestFloats[i];

    __m128d a = _mm_set_pd1(_a);

    return validateDouble(a, _a, _a);
}

result_t test_mm_set_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    double f0 = _a[0];
    double f1 = 0.0;

    __m128d a = _mm_set_sd(_a[0]);
    return validateDouble(a, f0, f1);
}

result_t test_mm_set1_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    int16_t d0 = _a[0];

    __m128i c = _mm_set1_epi16(d0);
    return validateInt16(c, d0, d0, d0, d0, d0, d0, d0, d0);
}

result_t test_mm_set1_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    int32_t x = impl.mTestInts[i];
    __m128i a = _mm_set1_epi32(x);
    return validateInt32(a, x, x, x, x);
}

result_t test_mm_set1_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_set1_epi64((__m64) _a[0]);

    return validateInt64(ret, _a[0], _a[0]);
}

result_t test_mm_set1_epi64x(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;

    __m128i ret = _mm_set1_epi64x(_a[0]);

    return validateInt64(ret, _a[0], _a[0]);
}

result_t test_mm_set1_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    int8_t d0 = _a[0];
    __m128i c = _mm_set1_epi8(d0);
    return validateInt8(c, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0,
                        d0, d0, d0);
}

result_t test_mm_set1_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    double d0 = _a[0];
    __m128d c = _mm_set1_pd(d0);
    return validateDouble(c, d0, d0);
}

result_t test_mm_setr_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;

    __m128i c =
        _mm_setr_epi16(_a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6], _a[7]);

    return validateInt16(c, _a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6],
                         _a[7]);
}

result_t test_mm_setr_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i c = _mm_setr_epi32(_a[0], _a[1], _a[2], _a[3]);
    return validateInt32(c, _a[0], _a[1], _a[2], _a[3]);
}

result_t test_mm_setr_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const __m64 *_a = (const __m64 *) impl.mTestIntPointer1;
    __m128i c = _mm_setr_epi64(_a[0], _a[1]);
    return validateInt64(c, (int64_t) _a[0], (int64_t) _a[1]);
}

result_t test_mm_setr_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    __m128i c = _mm_setr_epi8(_a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6],
                              _a[7], _a[8], _a[9], _a[10], _a[11], _a[12],
                              _a[13], _a[14], _a[15]);

    return validateInt8(c, _a[0], _a[1], _a[2], _a[3], _a[4], _a[5], _a[6],
                        _a[7], _a[8], _a[9], _a[10], _a[11], _a[12], _a[13],
                        _a[14], _a[15]);
}

result_t test_mm_setr_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *p = (const double *) impl.mTestFloatPointer1;

    double x = p[0];
    double y = p[1];

    __m128d a = _mm_setr_pd(x, y);

    return validateDouble(a, x, y);
}

result_t test_mm_setzero_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    __m128d a = _mm_setzero_pd();
    return validateDouble(a, 0, 0);
}

result_t test_mm_setzero_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    __m128i a = _mm_setzero_si128();
    return validateInt32(a, 0, 0, 0, 0);
}

result_t test_mm_shuffle_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int imm = 105;

    int32_t d0 = _a[((imm) &0x3)];
    int32_t d1 = _a[((imm >> 2) & 0x3)];
    int32_t d2 = _a[((imm >> 4) & 0x3)];
    int32_t d3 = _a[((imm >> 6) & 0x3)];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_shuffle_epi32(a, imm);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_shuffle_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double d0 = _a[i & 0x1];
    double d1 = _b[(i & 0x2) >> 1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c;
    switch (i & 0x3) {
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

result_t test_mm_shufflehi_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_shufflehi_epi16(a, imm);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_shufflelo_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_shufflelo_epi16(a, imm);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_sll_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) i;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
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

result_t test_mm_sll_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) i;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
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

result_t test_mm_sll_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) i;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sll_epi64(a, b);
    if (count < 0 || count > 63)
        return validateInt64(c, 0, 0);

    uint64_t d0 = _a[0] << count;
    uint64_t d1 = _a[1] << count;
    return validateInt64(c, d0, d1);
}

result_t test_mm_slli_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int count = 3;

    int16_t d0 = _a[0] << count;
    int16_t d1 = _a[1] << count;
    int16_t d2 = _a[2] << count;
    int16_t d3 = _a[3] << count;
    int16_t d4 = _a[4] << count;
    int16_t d5 = _a[5] << count;
    int16_t d6 = _a[6] << count;
    int16_t d7 = _a[7] << count;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_slli_epi16(a, count);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_slli_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;
#if defined(__clang__)
    // Clang compiler does not allow the second argument of _mm_slli_epi32() to
    // be greater than 31.
    int count = (uint32_t) _b[0] % 32;
#else
    int count = (uint32_t) _b[0] % 64;
    // The value for doing the modulo should be greater
    // than 32. Using 64 would provide more equal
    // distribution for both under 32 and above 32 input value.
#endif

    int32_t d0 = (count > 31) ? 0 : _a[0] << count;
    int32_t d1 = (count > 31) ? 0 : _a[1] << count;
    int32_t d2 = (count > 31) ? 0 : _a[2] << count;
    int32_t d3 = (count > 31) ? 0 : _a[3] << count;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_slli_epi32(a, count);
    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_slli_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;
#if defined(__clang__)
    // Clang compiler does not allow the second argument of `_mm_slli_epi64()`
    // to be greater than 63.
    int count = (uint64_t) _b[0] % 64;
#else
    int count =
        (uint64_t) _b[0] %
        128;  // The value for doing the modulo should be greater
              // than 64. Using 128 would provide more equal
              // distribution for both under 64 and above 64 input value.
#endif
    int64_t d0 = (count > 63) ? 0 : _a[0] << count;
    int64_t d1 = (count > 63) ? 0 : _a[1] << count;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_slli_epi64(a, count);
    return validateInt64(c, d0, d1);
}

result_t test_mm_slli_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // FIXME:
    // The shift value should be tested with random constant immediate value.
    const int32_t *_a = impl.mTestIntPointer1;

    int8_t d[16];
    int count = 5;
    for (int i = 0; i < 16; i++) {
        if (i < count)
            d[i] = 0;
        else
            d[i] = ((const int8_t *) _a)[i - count];
    }

    __m128i a = do_mm_load_ps(_a);
    __m128i ret = _mm_slli_si128(a, 5);

    return validateInt8(ret, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                        d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_sqrt_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;

    double f0 = sqrt(_a[0]);
    double f1 = sqrt(_a[1]);

    __m128d a = do_mm_load_pd(_a);
    __m128d c = _mm_sqrt_pd(a);

    return validateDouble(c, f0, f1);
}

result_t test_mm_sqrt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double f0 = sqrt(_b[0]);
    double f1 = _a[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_sqrt_sd(a, b);

    return validateDouble(c, f0, f1);
}

result_t test_mm_sra_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) i;
    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sra_epi16(a, b);
    if (count > 15) {
        int16_t d0 = _a[0] < 0 ? ~UINT16_C(0) : 0;
        int16_t d1 = _a[1] < 0 ? ~UINT16_C(0) : 0;
        int16_t d2 = _a[2] < 0 ? ~UINT16_C(0) : 0;
        int16_t d3 = _a[3] < 0 ? ~UINT16_C(0) : 0;
        int16_t d4 = _a[4] < 0 ? ~UINT16_C(0) : 0;
        int16_t d5 = _a[5] < 0 ? ~UINT16_C(0) : 0;
        int16_t d6 = _a[6] < 0 ? ~UINT16_C(0) : 0;
        int16_t d7 = _a[7] < 0 ? ~UINT16_C(0) : 0;

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

result_t test_mm_sra_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) i;
    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_sra_epi32(a, b);
    if (count > 31) {
        int32_t d0 = _a[0] < 0 ? ~UINT64_C(0) : 0;
        int32_t d1 = _a[1] < 0 ? ~UINT64_C(0) : 0;
        int32_t d2 = _a[2] < 0 ? ~UINT64_C(0) : 0;
        int32_t d3 = _a[3] < 0 ? ~UINT64_C(0) : 0;

        return validateInt32(c, d0, d1, d2, d3);
    }

    int32_t d0 = _a[0] >> count;
    int32_t d1 = _a[1] >> count;
    int32_t d2 = _a[2] >> count;
    int32_t d3 = _a[3] >> count;

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_srai_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    int64_t _b = (int64_t) i;
    const int b = _b;
    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i c = _mm_srai_epi16(a, b);

    __m128i ret;
    int count = (b & ~15) ? 15 : b;
    for (size_t i = 0; i < 8; i++) {
        ((SIMDVec *) &ret)->m128_i16[i] =
            ((SIMDVec *) &a)->m128_i16[i] >> count;
    }
    return validate128(c, ret);
}

result_t test_mm_srai_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t b = (const int32_t) impl.mTestInts[i];

    int32_t d[4];
    int count = (b & ~31) ? 31 : b;
    for (int i = 0; i < 4; i++) {
        d[i] = _a[i] >> count;
    }

    __m128i a = _mm_load_si128((const __m128i *) _a);
    __m128i c = _mm_srai_epi32(a, b);

    return validateInt32(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_srl_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) i;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
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

result_t test_mm_srl_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) i;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
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

result_t test_mm_srl_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t count = (int64_t) i;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_set1_epi64x(count);
    __m128i c = _mm_srl_epi64(a, b);
    if (count < 0 || count > 63)
        return validateInt64(c, 0, 0);

    uint64_t d0 = (uint64_t)(_a[0]) >> count;
    uint64_t d1 = (uint64_t)(_a[1]) >> count;
    return validateInt64(c, d0, d1);
}

result_t test_mm_srli_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int count = impl.mTestInts[i];

    int16_t d0 = count & (~15) ? 0 : (uint16_t)(_a[0]) >> count;
    int16_t d1 = count & (~15) ? 0 : (uint16_t)(_a[1]) >> count;
    int16_t d2 = count & (~15) ? 0 : (uint16_t)(_a[2]) >> count;
    int16_t d3 = count & (~15) ? 0 : (uint16_t)(_a[3]) >> count;
    int16_t d4 = count & (~15) ? 0 : (uint16_t)(_a[4]) >> count;
    int16_t d5 = count & (~15) ? 0 : (uint16_t)(_a[5]) >> count;
    int16_t d6 = count & (~15) ? 0 : (uint16_t)(_a[6]) >> count;
    int16_t d7 = count & (~15) ? 0 : (uint16_t)(_a[7]) >> count;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_srli_epi16(a, count);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_srli_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int count = impl.mTestInts[i];

    int32_t d0 = count & (~31) ? 0 : (uint32_t)(_a[0]) >> count;
    int32_t d1 = count & (~31) ? 0 : (uint32_t)(_a[1]) >> count;
    int32_t d2 = count & (~31) ? 0 : (uint32_t)(_a[2]) >> count;
    int32_t d3 = count & (~31) ? 0 : (uint32_t)(_a[3]) >> count;

    __m128i a = do_mm_load_ps(_a);
    __m128i c = _mm_srli_epi32(a, count);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_srli_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int count = impl.mTestInts[i];

    int64_t d0 = count & (~63) ? 0 : (uint64_t)(_a[0]) >> count;
    int64_t d1 = count & (~63) ? 0 : (uint64_t)(_a[1]) >> count;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i c = _mm_srli_epi64(a, count);

    return validateInt64(c, d0, d1);
}

result_t test_mm_srli_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // FIXME:
    // The shift value should be tested with random constant immediate value.
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    int count = 5;

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        if (i >= (16 - count))
            d[i] = 0;
        else
            d[i] = _a[i + count];
    }

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_srli_si128(a, 5);

    return validateInt8(ret, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                        d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_store_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double x = impl.mTestFloats[i + 4];
    double y = impl.mTestFloats[i + 6];

    __m128d a = _mm_set_pd(x, y);
    _mm_store_pd(p, a);
    ASSERT_RETURN(p[0] == y);
    ASSERT_RETURN(p[1] == x);
    return TEST_SUCCESS;
}

result_t test_mm_store_pd1(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double _a[2] = {(double) impl.mTestFloats[i],
                    (double) impl.mTestFloats[i + 1]};

    __m128d a = do_mm_load_pd((const double *) _a);
    _mm_store_pd1(p, a);
    ASSERT_RETURN(p[0] == impl.mTestFloats[i]);
    ASSERT_RETURN(p[1] == impl.mTestFloats[i]);
    return TEST_SUCCESS;
}

result_t test_mm_store_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double _a[2] = {(double) impl.mTestFloats[i],
                    (double) impl.mTestFloats[i + 1]};

    __m128d a = do_mm_load_pd((const double *) _a);
    _mm_store_sd(p, a);
    ASSERT_RETURN(p[0] == impl.mTestFloats[i]);
    return TEST_SUCCESS;
}

result_t test_mm_store_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    int32_t p[4];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    _mm_store_si128((__m128i *) p, a);

    return validateInt32(a, p[0], p[1], p[2], p[3]);
}

result_t test_mm_store1_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_store_pd1(impl, i);
}

result_t test_mm_storeh_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double mem;

    __m128d a = do_mm_load_pd(p);
    _mm_storeh_pd(&mem, a);

    ASSERT_RETURN(mem == p[1]);
    return TEST_SUCCESS;
}

result_t test_mm_storel_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    int64_t *p = (int64_t *) impl.mTestIntPointer1;
    __m128i mem;

    __m128i a = do_mm_load_ps((const int32_t *) p);
    _mm_storel_epi64(&mem, a);

    ASSERT_RETURN(mem[0] == p[0]);
    return TEST_SUCCESS;
}

result_t test_mm_storel_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double mem;

    __m128d a = do_mm_load_pd(p);
    _mm_storel_pd(&mem, a);

    ASSERT_RETURN(mem == p[0]);
    return TEST_SUCCESS;
}

result_t test_mm_storer_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double mem[2];

    __m128d a = do_mm_load_pd(p);
    _mm_storer_pd(mem, a);

    __m128d res = do_mm_load_pd(mem);
    return validateDouble(res, p[1], p[0]);
}

result_t test_mm_storeu_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    double *p = (double *) impl.mTestFloatPointer1;
    double x = impl.mTestFloats[i + 4];
    double y = impl.mTestFloats[i + 6];

    __m128d a = _mm_set_pd(x, y);
    _mm_storeu_pd(p, a);
    ASSERT_RETURN(p[0] == y);
    ASSERT_RETURN(p[1] == x);
    return TEST_SUCCESS;
}

result_t test_mm_storeu_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i b;
    __m128i a = do_mm_load_ps(_a);
    _mm_storeu_si128(&b, a);
    int32_t *_b = (int32_t *) &b;
    return validateInt32(a, _b[0], _b[1], _b[2], _b[3]);
}

result_t test_mm_storeu_si32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    // The GCC version before 11 does not implement intrinsic function
    // _mm_storeu_si32. Check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    // for more information.
#if defined(__GNUC__) && __GNUC__ <= 10
    return TEST_UNIMPL;
#else
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i b;
    __m128i a = do_mm_load_ps(_a);
    _mm_storeu_si32(&b, a);
    int32_t *_b = (int32_t *) &b;
    return validateInt32(b, _a[0], _b[1], _b[2], _b[3]);
#endif
}

result_t test_mm_stream_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    double p[2];

    __m128d a = do_mm_load_pd(_a);
    _mm_stream_pd(p, a);

    return validateDouble(a, p[0], p[1]);
}

result_t test_mm_stream_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    int32_t p[4];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    _mm_stream_si128((__m128i *) p, a);

    return validateInt32(a, p[0], p[1], p[2], p[3]);
}

result_t test_mm_stream_si32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t a = (const int32_t) impl.mTestInts[i];
    int32_t p;

    _mm_stream_si32(&p, a);

    ASSERT_RETURN(a == p)
    return TEST_SUCCESS;
}

result_t test_mm_stream_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_sub_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sub_epi16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_sub_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t dx = _a[0] - _b[0];
    int32_t dy = _a[1] - _b[1];
    int32_t dz = _a[2] - _b[2];
    int32_t dw = _a[3] - _b[3];

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i c = _mm_sub_epi32(a, b);
    return validateInt32(c, dx, dy, dz, dw);
}

result_t test_mm_sub_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (int64_t *) impl.mTestIntPointer2;
    int64_t d0 = _a[0] - _b[0];
    int64_t d1 = _a[1] - _b[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sub_epi64(a, b);
    return validateInt64(c, d0, d1);
}

result_t test_mm_sub_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sub_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_sub_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] - _b[0];
    double d1 = _a[1] - _b[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_sub_pd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_sub_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    double d0 = _a[0] - _b[0];
    double d1 = _a[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_sub_sd(a, b);
    return validateDouble(c, d0, d1);
}

result_t test_mm_sub_si64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t d = _a[0] - _b[0];

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_sub_si64(a, b);

    return validateInt64(c, d);
}

result_t test_mm_subs_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_subs_epi16(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_subs_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_subs_epi8(a, b);

    return validateInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                        d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_subs_epu16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);

    __m128i c = _mm_subs_epu16(a, b);
    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_subs_epu8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_subs_epu8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_ucomieq_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_comieq_sd(impl, i);
}

result_t test_mm_ucomige_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_comige_sd(impl, i);
}

result_t test_mm_ucomigt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_comigt_sd(impl, i);
}

result_t test_mm_ucomile_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_comile_sd(impl, i);
}

result_t test_mm_ucomilt_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_comilt_sd(impl, i);
}

result_t test_mm_ucomineq_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_comineq_sd(impl, i);
}

result_t test_mm_undefined_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_undefined_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_unpackhi_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_unpackhi_epi16(a, b);

    return validateInt16(ret, i0, i1, i2, i3, i4, i5, i6, i7);
}

result_t test_mm_unpackhi_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t i0 = _a[2];
    int32_t i1 = _b[2];
    int32_t i2 = _a[3];
    int32_t i3 = _b[3];

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i ret = _mm_unpackhi_epi32(a, b);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_unpackhi_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t i0 = _a[1];
    int64_t i1 = _b[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_unpackhi_epi64(a, b);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_unpackhi_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_unpackhi_epi8(a, b);

    return validateInt8(ret, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
                        i12, i13, i14, i15);
}

result_t test_mm_unpackhi_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d ret = _mm_unpackhi_pd(a, b);

    return validateDouble(ret, _a[1], _b[1]);
}

result_t test_mm_unpacklo_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_unpacklo_epi16(a, b);

    return validateInt16(ret, i0, i1, i2, i3, i4, i5, i6, i7);
}

result_t test_mm_unpacklo_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t i0 = _a[0];
    int32_t i1 = _b[0];
    int32_t i2 = _a[1];
    int32_t i3 = _b[1];

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i ret = _mm_unpacklo_epi32(a, b);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_unpacklo_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t i0 = _a[0];
    int64_t i1 = _b[0];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_unpacklo_epi64(a, b);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_unpacklo_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_unpacklo_epi8(a, b);

    return validateInt8(ret, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
                        i12, i13, i14, i15);
}

result_t test_mm_unpacklo_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d ret = _mm_unpacklo_pd(a, b);

    return validateDouble(ret, _a[0], _b[0]);
}

result_t test_mm_xor_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestFloatPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestFloatPointer2;

    int64_t d0 = _a[0] ^ _b[0];
    int64_t d1 = _a[1] ^ _b[1];

    __m128d a = do_mm_load_pd((const double *) _a);
    __m128d b = do_mm_load_pd((const double *) _b);
    __m128d c = _mm_xor_pd(a, b);

    return validateDouble(c, *((double *) &d0), *((double *) &d1));
}

result_t test_mm_xor_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t d0 = _a[0] ^ _b[0];
    int64_t d1 = _a[1] ^ _b[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_xor_si128(a, b);

    return validateInt64(c, d0, d1);
}

/* SSE3 */
result_t test_mm_addsub_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double d0 = _a[0] - _b[0];
    double d1 = _a[1] + _b[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_addsub_pd(a, b);

    return validateDouble(c, d0, d1);
}

result_t test_mm_addsub_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _a[0] - _b[0];
    float f1 = _a[1] + _b[1];
    float f2 = _a[2] - _b[2];
    float f3 = _a[3] + _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_addsub_ps(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_hadd_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double f0 = _a[0] + _a[1];
    double f1 = _b[0] + _b[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_hadd_pd(a, b);

    return validateDouble(c, f0, f1);
}

result_t test_mm_hadd_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _a[0] + _a[1];
    float f1 = _a[2] + _a[3];
    float f2 = _b[0] + _b[1];
    float f3 = _b[2] + _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_hadd_ps(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_hsub_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;

    double f0 = _a[0] - _a[1];
    double f1 = _b[0] - _b[1];

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d c = _mm_hsub_pd(a, b);

    return validateDouble(c, f0, f1);
}

result_t test_mm_hsub_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    float f0 = _a[0] - _a[1];
    float f1 = _a[2] - _a[3];
    float f2 = _b[0] - _b[1];
    float f3 = _b[2] - _b[3];

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_hsub_ps(a, b);

    return validateFloat(c, f0, f1, f2, f3);
}

result_t test_mm_lddqu_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return test_mm_loadu_si128(impl, i);
}

result_t test_mm_loaddup_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *addr = (const double *) impl.mTestFloatPointer1;

    __m128d ret = _mm_loaddup_pd(addr);

    return validateDouble(ret, addr[0], addr[0]);
}

result_t test_mm_movedup_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *p = (const double *) impl.mTestFloatPointer1;
    __m128d a = do_mm_load_pd(p);
    __m128d b = _mm_movedup_pd(a);

    return validateDouble(b, p[0], p[0]);
}

result_t test_mm_movehdup_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *p = impl.mTestFloatPointer1;
    __m128 a = do_mm_load_ps(p);
    return validateFloat(_mm_movehdup_ps(a), p[1], p[1], p[3], p[3]);
}

result_t test_mm_moveldup_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *p = impl.mTestFloatPointer1;
    __m128 a = do_mm_load_ps(p);
    return validateFloat(_mm_moveldup_ps(a), p[0], p[0], p[2], p[2]);
}

/* SSSE3 */
result_t test_mm_abs_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
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

result_t test_mm_abs_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i a = do_mm_load_ps(_a);
    __m128i c = _mm_abs_epi32(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];
    uint32_t d2 = (_a[2] < 0) ? -_a[2] : _a[2];
    uint32_t d3 = (_a[3] < 0) ? -_a[3] : _a[3];

    return validateUInt32(c, d0, d1, d2, d3);
}

result_t test_mm_abs_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
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

result_t test_mm_abs_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 c = _mm_abs_pi16(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];
    uint32_t d2 = (_a[2] < 0) ? -_a[2] : _a[2];
    uint32_t d3 = (_a[3] < 0) ? -_a[3] : _a[3];

    return validateUInt16(c, d0, d1, d2, d3);
}

result_t test_mm_abs_pi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 c = _mm_abs_pi32(a);

    uint32_t d0 = (_a[0] < 0) ? -_a[0] : _a[0];
    uint32_t d1 = (_a[1] < 0) ? -_a[1] : _a[1];

    return validateUInt32(c, d0, d1);
}

result_t test_mm_abs_pi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    __m64 a = do_mm_load_m64((const int64_t *) _a);
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

result_t test_mm_alignr_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
#if defined(__clang__)
    return TEST_UNIMPL;
#else
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    // FIXME: The different immediate value should be tested in the future
    const int shift = 18;
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_alignr_epi8(a, b, shift);

    return validateUInt8(ret, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                         d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
#endif
}

result_t test_mm_alignr_pi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
#if defined(__clang__)
    return TEST_UNIMPL;
#else
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const uint8_t *_b = (const uint8_t *) impl.mTestIntPointer2;
    // FIXME: The different immediate value should be tested in the future
    const int shift = 10;
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

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 ret = _mm_alignr_pi8(a, b, shift);

    return validateUInt8(ret, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
#endif
}

result_t test_mm_hadd_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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
    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_hadd_epi16(a, b);
    return validateInt16(ret, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_hadd_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;
    int32_t d0 = _a[0] + _a[1];
    int32_t d1 = _a[2] + _a[3];
    int32_t d2 = _b[0] + _b[1];
    int32_t d3 = _b[2] + _b[3];
    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i ret = _mm_hadd_epi32(a, b);
    return validateInt32(ret, d0, d1, d2, d3);
}

result_t test_mm_hadd_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;
    const int16_t *_b = (const int16_t *) impl.mTestIntPointer2;
    int16_t d0 = _a[0] + _a[1];
    int16_t d1 = _a[2] + _a[3];
    int16_t d2 = _b[0] + _b[1];
    int16_t d3 = _b[2] + _b[3];
    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 ret = _mm_hadd_pi16(a, b);
    return validateInt16(ret, d0, d1, d2, d3);
}

result_t test_mm_hadd_pi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;
    int32_t d0 = _a[0] + _a[1];
    int32_t d1 = _b[0] + _b[1];
    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 ret = _mm_hadd_pi32(a, b);
    return validateInt32(ret, d0, d1);
}

result_t test_mm_hadds_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_hadds_epi16(a, b);

    return validateInt16(c, d16[0], d16[1], d16[2], d16[3], d16[4], d16[5],
                         d16[6], d16[7]);
}

result_t test_mm_hadds_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_hsub_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_hsub_epi16(a, b);

    return validateInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_hsub_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer1;

    int32_t d0 = _a[0] - _a[1];
    int32_t d1 = _a[2] - _a[3];
    int32_t d2 = _b[0] - _b[1];
    int32_t d3 = _b[2] - _b[3];

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i c = _mm_hsub_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_hsub_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_hsub_pi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_hsubs_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_hsubs_epi16(a, b);

    return validateInt16(c, d16[0], d16[1], d16[2], d16[3], d16[4], d16[5],
                         d16[6], d16[7]);
}

result_t test_mm_hsubs_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_maddubs_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_maddubs_epi16(a, b);
    return validateInt16(c, e0, e1, e2, e3, e4, e5, e6, e7);
}

result_t test_mm_maddubs_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_mulhrs_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_mulhrs_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_shuffle_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *a = impl.mTestIntPointer1;
    const int32_t *b = impl.mTestIntPointer2;
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

    __m128i ret = _mm_shuffle_epi8(do_mm_load_ps(a), do_mm_load_ps(b));

    return validateInt32(ret, r[0], r[1], r[2], r[3]);
}

result_t test_mm_shuffle_pi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_sign_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sign_epi16(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_sign_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i c = _mm_sign_epi32(a, b);

    return validateInt32(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_sign_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_sign_epi8(a, b);

    return validateInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                        d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_sign_pi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_sign_pi16(a, b);

    return validateInt16(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_sign_pi32(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_sign_pi32(a, b);

    return validateInt32(c, d[0], d[1]);
}

result_t test_mm_sign_pi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m64 a = do_mm_load_m64((const int64_t *) _a);
    __m64 b = do_mm_load_m64((const int64_t *) _b);
    __m64 c = _mm_sign_pi8(a, b);

    return validateInt8(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

/* SSE4.1 */
result_t test_mm_blend_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_blend_epi16(a, b, mask);

    return validateInt16(c, _c[0], _c[1], _c[2], _c[3], _c[4], _c[5], _c[6],
                         _c[7]);
}

result_t test_mm_blend_pd(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128d a = do_mm_load_pd((const double *) _a);
    __m128d b = do_mm_load_pd((const double *) _b);
    __m128d c = _mm_blend_pd(a, b, mask);

    return validateDouble(c, _c[0], _c[1]);
}

result_t test_mm_blend_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;

    const char mask = (char) i;

    float _c[4];
    for (int i = 0; i < 4; i++) {
        if (mask & (1 << i)) {
            _c[i] = _b[i];
        } else {
            _c[i] = _a[i];
        }
    }

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);

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

result_t test_mm_blendv_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t *_b = (const int8_t *) impl.mTestIntPointer2;
    const int8_t _mask[16] = {(const int8_t) impl.mTestInts[i],
                              (const int8_t) impl.mTestInts[i + 1],
                              (const int8_t) impl.mTestInts[i + 2],
                              (const int8_t) impl.mTestInts[i + 3],
                              (const int8_t) impl.mTestInts[i + 4],
                              (const int8_t) impl.mTestInts[i + 5],
                              (const int8_t) impl.mTestInts[i + 6],
                              (const int8_t) impl.mTestInts[i + 7]};

    int8_t _c[16];
    for (int i = 0; i < 16; i++) {
        if (_mask[i] >> 7) {
            _c[i] = _b[i];
        } else {
            _c[i] = _a[i];
        }
    }

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i mask = do_mm_load_ps((const int32_t *) _mask);
    __m128i c = _mm_blendv_epi8(a, b, mask);

    return validateInt8(c, _c[0], _c[1], _c[2], _c[3], _c[4], _c[5], _c[6],
                        _c[7], _c[8], _c[9], _c[10], _c[11], _c[12], _c[13],
                        _c[14], _c[15]);
}

result_t test_mm_blendv_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const double *_a = (const double *) impl.mTestFloatPointer1;
    const double *_b = (const double *) impl.mTestFloatPointer2;
    const double _mask[] = {(double) impl.mTestFloats[i],
                            (double) impl.mTestFloats[i + 1]};

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

    __m128d a = do_mm_load_pd(_a);
    __m128d b = do_mm_load_pd(_b);
    __m128d mask = do_mm_load_pd(_mask);

    __m128d c = _mm_blendv_pd(a, b, mask);

    return validateDouble(c, _c[0], _c[1]);
}

result_t test_mm_blendv_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    const float _mask[] = {impl.mTestFloats[i], impl.mTestFloats[i + 1],
                           impl.mTestFloats[i + 2], impl.mTestFloats[i + 3]};

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

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 mask = do_mm_load_ps(_mask);

    __m128 c = _mm_blendv_ps(a, b, mask);

    return validateFloat(c, _c[0], _c[1], _c[2], _c[3]);
}

result_t test_mm_ceil_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_ceil_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    float dx = ceilf(_a[0]);
    float dy = ceilf(_a[1]);
    float dz = ceilf(_a[2]);
    float dw = ceilf(_a[3]);

    __m128 a = _mm_load_ps(_a);
    __m128 c = _mm_ceil_ps(a);
    return validateFloatEpsilon(c, dx, dy, dz, dw, 5.0f);
}

result_t test_mm_ceil_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_ceil_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f0 = ceilf(_b[0]);

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_ceil_ss(a, b);

    return validateFloat(c, f0, _a[1], _a[2], _a[3]);
}

result_t test_mm_cmpeq_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;
    int64_t d0 = (_a[0] == _b[0]) ? 0xffffffffffffffff : 0x0;
    int64_t d1 = (_a[1] == _b[1]) ? 0xffffffffffffffff : 0x0;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_cmpeq_epi64(a, b);
    return validateInt64(c, d0, d1);
}

result_t test_mm_cvtepi16_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;

    int32_t i0 = (int32_t) _a[0];
    int32_t i1 = (int32_t) _a[1];
    int32_t i2 = (int32_t) _a[2];
    int32_t i3 = (int32_t) _a[3];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepi16_epi32(a);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_cvtepi16_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int16_t *_a = (const int16_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepi16_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepi32_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = do_mm_load_ps(_a);
    __m128i ret = _mm_cvtepi32_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepi8_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepi8_epi16(a);

    return validateInt16(ret, i0, i1, i2, i3, i4, i5, i6, i7);
}

result_t test_mm_cvtepi8_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    int32_t i0 = (int32_t) _a[0];
    int32_t i1 = (int32_t) _a[1];
    int32_t i2 = (int32_t) _a[2];
    int32_t i3 = (int32_t) _a[3];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepi8_epi32(a);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_cvtepi8_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepi8_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepu16_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;

    int32_t i0 = (int32_t) _a[0];
    int32_t i1 = (int32_t) _a[1];
    int32_t i2 = (int32_t) _a[2];
    int32_t i3 = (int32_t) _a[3];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepu16_epi32(a);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_cvtepu16_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint16_t *_a = (const uint16_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepu16_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepu32_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepu32_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_cvtepu8_epi16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepu8_epi16(a);

    return validateInt16(ret, i0, i1, i2, i3, i4, i5, i6, i7);
}

result_t test_mm_cvtepu8_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;

    int32_t i0 = (int32_t) _a[0];
    int32_t i1 = (int32_t) _a[1];
    int32_t i2 = (int32_t) _a[2];
    int32_t i3 = (int32_t) _a[3];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepu8_epi32(a);

    return validateInt32(ret, i0, i1, i2, i3);
}

result_t test_mm_cvtepu8_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint8_t *_a = (const uint8_t *) impl.mTestIntPointer1;

    int64_t i0 = (int64_t) _a[0];
    int64_t i1 = (int64_t) _a[1];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_cvtepu8_epi64(a);

    return validateInt64(ret, i0, i1);
}

result_t test_mm_dp_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_dp_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer2;
    const int imm = 0xFF;
    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
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

result_t test_mm_extract_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    int32_t *_a = (int32_t *) impl.mTestIntPointer1;
    const int imm = 1;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    int c = _mm_extract_epi32(a, imm);

    ASSERT_RETURN(c == *(_a + imm));
    return TEST_SUCCESS;
}

result_t test_mm_extract_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    int64_t *_a = (int64_t *) impl.mTestIntPointer1;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __int64_t c;

    switch (i & 0x1) {
    case 0:
        c = _mm_extract_epi64(a, 0);
        break;
    case 1:
        c = _mm_extract_epi64(a, 1);
        break;
    }

    ASSERT_RETURN(c == *(_a + (i & 1)));
    return TEST_SUCCESS;
}

result_t test_mm_extract_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_extract_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = (const float *) impl.mTestFloatPointer1;

    __m128 a = _mm_load_ps(_a);
    int32_t c;

    switch (i & 0x3) {
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

    ASSERT_RETURN(c == *(const int32_t *) (_a + (i & 0x3)));
    return TEST_SUCCESS;
}

result_t test_mm_floor_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_floor_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    float dx = floorf(_a[0]);
    float dy = floorf(_a[1]);
    float dz = floorf(_a[2]);
    float dw = floorf(_a[3]);

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_floor_ps(a);
    return validateFloatEpsilon(c, dx, dy, dz, dw, 5.0f);
}

result_t test_mm_floor_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_floor_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    const float *_b = impl.mTestFloatPointer1;

    float f0 = floorf(_b[0]);

    __m128 a = do_mm_load_ps(_a);
    __m128 b = do_mm_load_ps(_b);
    __m128 c = _mm_floor_ss(a, b);

    return validateFloat(c, f0, _a[1], _a[2], _a[3]);
}

result_t test_mm_insert_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t insert = (int32_t) *impl.mTestIntPointer2;
    const int imm8 = 2;

    int32_t d[4];
    for (int i = 0; i < 4; i++) {
        d[i] = _a[i];
    }
    d[imm8] = insert;

    __m128i a = do_mm_load_ps(_a);
    __m128i b = _mm_insert_epi32(a, (int) insert, imm8);
    return validateInt32(b, d[0], d[1], d[2], d[3]);
}

result_t test_mm_insert_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    int64_t insert = (int64_t) *impl.mTestIntPointer2;
    const int imm8 = 1;

    int64_t d[2];

    d[0] = _a[0];
    d[1] = _a[1];
    d[imm8] = insert;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_insert_epi64(a, insert, imm8);
    return validateInt64(b, d[0], d[1]);
}

result_t test_mm_insert_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int8_t *_a = (const int8_t *) impl.mTestIntPointer1;
    const int8_t insert = (int8_t) *impl.mTestIntPointer2;
    const int imm8 = 2;

    int8_t d[16];
    for (int i = 0; i < 16; i++) {
        d[i] = _a[i];
    }
    d[imm8] = insert;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = _mm_insert_epi8(a, insert, imm8);
    return validateInt8(b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
                        d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

result_t test_mm_insert_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_max_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    int32_t d1 = _a[1] > _b[1] ? _a[1] : _b[1];
    int32_t d2 = _a[2] > _b[2] ? _a[2] : _b[2];
    int32_t d3 = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i c = _mm_max_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_max_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);

    __m128i c = _mm_max_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_max_epu16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_max_epu16(a, b);

    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_max_epu32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;
    const uint32_t *_b = (const uint32_t *) impl.mTestIntPointer2;

    uint32_t d0 = _a[0] > _b[0] ? _a[0] : _b[0];
    uint32_t d1 = _a[1] > _b[1] ? _a[1] : _b[1];
    uint32_t d2 = _a[2] > _b[2] ? _a[2] : _b[2];
    uint32_t d3 = _a[3] > _b[3] ? _a[3] : _b[3];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_max_epu32(a, b);

    return validateUInt32(c, d0, d1, d2, d3);
}

result_t test_mm_min_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int32_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    int32_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
    int32_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
    int32_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i c = _mm_min_epi32(a, b);

    return validateInt32(c, d0, d1, d2, d3);
}

result_t test_mm_min_epi8(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);

    __m128i c = _mm_min_epi8(a, b);
    return validateInt8(c, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,
                        d12, d13, d14, d15);
}

result_t test_mm_min_epu16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_min_epu16(a, b);

    return validateUInt16(c, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_min_epu32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint32_t *_a = (const uint32_t *) impl.mTestIntPointer1;
    const uint32_t *_b = (const uint32_t *) impl.mTestIntPointer2;

    uint32_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
    uint32_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
    uint32_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
    uint32_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_min_epu32(a, b);

    return validateUInt32(c, d0, d1, d2, d3);
}

result_t test_mm_minpos_epu16(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i ret = _mm_minpos_epu16(a);
    return validateUInt16(ret, d0, d1, d2, d3, d4, d5, d6, d7);
}

result_t test_mm_mpsadbw_epu8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_mul_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_b = (const int32_t *) impl.mTestIntPointer2;

    int64_t dx = (int64_t)(_a[0]) * (int64_t)(_b[0]);
    int64_t dy = (int64_t)(_a[2]) * (int64_t)(_b[2]);

    __m128i a = _mm_loadu_si128((const __m128i *) _a);
    __m128i b = _mm_loadu_si128((const __m128i *) _b);
    __m128i r = _mm_mul_epi32(a, b);

    return validateInt64(r, dx, dy);
}

result_t test_mm_mullo_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = impl.mTestIntPointer1;
    const int32_t *_b = impl.mTestIntPointer2;
    int32_t d[4];

    for (int i = 0; i < 4; i++) {
        d[i] = (int32_t)((int64_t) _a[i] * (int64_t) _b[i]);
    }
    __m128i a = do_mm_load_ps(_a);
    __m128i b = do_mm_load_ps(_b);
    __m128i c = _mm_mullo_epi32(a, b);
    return validateInt32(c, d[0], d[1], d[2], d[3]);
}

result_t test_mm_packus_epi32(const SSE2NEONTestImpl &impl, uint32_t i)
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

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i c = _mm_packus_epi32(a, b);

    return validateUInt16(c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

result_t test_mm_round_pd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_round_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const float *_a = impl.mTestFloatPointer1;
    float dx = roundf(_a[0]);
    float dy = roundf(_a[1]);
    float dz = roundf(_a[2]);
    float dw = roundf(_a[3]);

    __m128 a = do_mm_load_ps(_a);
    __m128 c = _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    return validateFloatEpsilon(c, dx, dy, dz, dw, 5.0f);
}

result_t test_mm_round_sd(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_round_ss(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_stream_load_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    int32_t *addr = impl.mTestIntPointer1;

    __m128i ret = _mm_stream_load_si128((__m128i *) addr);

    return validateInt32(ret, addr[0], addr[1], addr[2], addr[3]);
}

result_t test_mm_test_all_ones(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    __m128i a = do_mm_load_ps(_a);

    int32_t d0 = ~_a[0] & (~(uint32_t) 0);
    int32_t d1 = ~_a[1] & (~(uint32_t) 0);
    int32_t d2 = ~_a[2] & (~(uint32_t) 0);
    int32_t d3 = ~_a[3] & (~(uint32_t) 0);
    int32_t result = ((d0 | d1 | d2 | d3) == 0) ? 1 : 0;

    int32_t ret = _mm_test_all_ones(a);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_test_all_zeros(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *_a = (const int32_t *) impl.mTestIntPointer1;
    const int32_t *_mask = (const int32_t *) impl.mTestIntPointer2;
    __m128i a = do_mm_load_ps(_a);
    __m128i mask = do_mm_load_ps(_mask);

    int32_t d0 = _a[0] & _mask[0];
    int32_t d1 = _a[1] & _mask[1];
    int32_t d2 = _a[2] & _mask[2];
    int32_t d3 = _a[3] & _mask[3];
    int32_t result = ((d0 | d1 | d2 | d3) == 0) ? 1 : 0;

    int32_t ret = _mm_test_all_zeros(a, mask);

    return result == ret ? TEST_SUCCESS : TEST_FAIL;
}

result_t test_mm_test_mix_ones_zeros(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_testc_si128(const SSE2NEONTestImpl &impl, uint32_t i)
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

result_t test_mm_testnzc_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_testz_si128(const SSE2NEONTestImpl &impl, uint32_t i)
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
result_t test_mm_cmpestra(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpestrc(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpestri(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpestrm(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpestro(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpestrs(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpestrz(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpgt_epi64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int64_t *_a = (const int64_t *) impl.mTestIntPointer1;
    const int64_t *_b = (const int64_t *) impl.mTestIntPointer2;

    int64_t result[2];
    result[0] = _a[0] > _b[0] ? -1 : 0;
    result[1] = _a[1] > _b[1] ? -1 : 0;

    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
    __m128i iret = _mm_cmpgt_epi64(a, b);

    return validateInt64(iret, result[0], result[1]);
}

result_t test_mm_cmpistra(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistrc(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistri(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistrm(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistro(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistrs(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_cmpistrz(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

result_t test_mm_crc32_u16(const SSE2NEONTestImpl &impl, uint32_t i)
{
    uint32_t crc = *(const uint32_t *) impl.mTestIntPointer1;
    uint16_t v = i;
    uint32_t result = _mm_crc32_u16(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u16(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    uint32_t crc = *(const uint32_t *) impl.mTestIntPointer1;
    uint32_t v = *(const uint32_t *) impl.mTestIntPointer2;
    uint32_t result = _mm_crc32_u32(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u32(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    uint64_t crc = *(const uint64_t *) impl.mTestIntPointer1;
    uint64_t v = *(const uint64_t *) impl.mTestIntPointer2;
    uint64_t result = _mm_crc32_u64(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u64(crc, v));
    return TEST_SUCCESS;
}

result_t test_mm_crc32_u8(const SSE2NEONTestImpl &impl, uint32_t i)
{
    uint32_t crc = *(const uint32_t *) impl.mTestIntPointer1;
    uint8_t v = i;
    uint32_t result = _mm_crc32_u8(crc, v);
    ASSERT_RETURN(result == canonical_crc32_u8(crc, v));
    return TEST_SUCCESS;
}

/* AES */
result_t test_mm_aesenc_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const int32_t *a = (int32_t *) impl.mTestIntPointer1;
    const int32_t *b = (int32_t *) impl.mTestIntPointer2;
    __m128i data = _mm_loadu_si128((const __m128i *) a);
    __m128i rk = _mm_loadu_si128((const __m128i *) b);

    __m128i resultReference = aesenc_128_reference(data, rk);
    __m128i resultIntrinsic = _mm_aesenc_si128(data, rk);

    return validate128(resultReference, resultIntrinsic);
}

result_t test_mm_aesenclast_si128(const SSE2NEONTestImpl &impl, uint32_t i)
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
result_t test_mm_aeskeygenassist_si128(const SSE2NEONTestImpl &impl, uint32_t i)
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

/* FMA */
result_t test_mm_fmadd_ps(const SSE2NEONTestImpl &impl, uint32_t i)
{
    return TEST_UNIMPL;
}

/* Others */
result_t test_mm_clmulepi64_si128(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint64_t *_a = (const uint64_t *) impl.mTestIntPointer1;
    const uint64_t *_b = (const uint64_t *) impl.mTestIntPointer2;
    __m128i a = do_mm_load_ps((const int32_t *) _a);
    __m128i b = do_mm_load_ps((const int32_t *) _b);
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

result_t test_mm_popcnt_u32(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint64_t *a = (const uint64_t *) impl.mTestIntPointer1;
    ASSERT_RETURN(__builtin_popcount(a[0]) == _mm_popcnt_u32(a[0]));
    return TEST_SUCCESS;
}

result_t test_mm_popcnt_u64(const SSE2NEONTestImpl &impl, uint32_t i)
{
    const uint64_t *a = (const uint64_t *) impl.mTestIntPointer1;
    ASSERT_RETURN(__builtin_popcountll(a[0]) == _mm_popcnt_u64(a[0]));
    return TEST_SUCCESS;
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
result_t test_last(const SSE2NEONTestImpl &impl, uint32_t i)
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
