/**
 * nan.cpp - Comprehensive NaN Propagation and Handling Tests
 *
 * This test file provides comprehensive coverage for NaN behavior in sse2neon.
 *
 * Coverage:
 *   - NaN Propagation: All FP arithmetic operations
 *   - NaN Comparisons: All comparison intrinsics
 *   - NaN Payload: Preservation through operations
 *   - Signaling NaN: sNaN → qNaN conversion
 *   - Special Functions: sqrt, rsqrt, rcp with NaN inputs
 *   - Double Precision: All _pd and _sd variants
 *   - Min/Max Edge Cases: SSE vs NEON NaN handling differences
 *
 * NaN Behavior Differences (SSE vs NEON):
 *   - Signaling vs Quiet NaN: NEON may not preserve the distinction
 *   - NaN Payload: SSE preserves NaN payload bits; NEON may canonicalize
 *   - Propagation Order: When both operands are NaN, which is returned varies
 *   - Min/Max: SSE returns second operand if either is NaN; NEON differs
 *
 * Reference: IEEE 754-2008, Intel Intrinsics Guide
 */

#if defined(_M_ARM64EC)
#include "sse2neon.h"
#endif

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "common.h"

using namespace SSE2NEON;

/* Test result counters */
static int g_pass_count = 0;
static int g_fail_count = 0;
static int g_skip_count = 0;

/* Helper macros */
#define TEST_CASE(name) static result_t test_##name(void)

#define RUN_TEST(name)                                                        \
    do {                                                                      \
        result_t r = test_##name();                                           \
        if (r == TEST_SUCCESS) {                                              \
            g_pass_count++;                                                   \
            printf("Test %-40s [ " COLOR_GREEN "OK" COLOR_RESET " ]\n",       \
                   #name);                                                    \
        } else if (r == TEST_UNIMPL) {                                        \
            g_skip_count++;                                                   \
            printf("Test %-40s [SKIP]\n", #name);                             \
        } else {                                                              \
            g_fail_count++;                                                   \
            printf("Test %-40s [" COLOR_RED "FAIL" COLOR_RESET "]\n", #name); \
        }                                                                     \
    } while (0)

#define EXPECT_TRUE(cond)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            printf("    FAILED: %s (line %d)\n", #cond, __LINE__); \
            return TEST_FAIL;                                      \
        }                                                          \
    } while (0)

#define EXPECT_NAN_F(a) EXPECT_TRUE(std::isnan(a))
#define EXPECT_NAN_D(a) EXPECT_TRUE(std::isnan(a))
#define EXPECT_NOT_NAN_F(a) EXPECT_TRUE(!std::isnan(a))

/* All-lane validation macros - reduces boilerplate for vector tests */
#define EXPECT_ALL_NAN_PS(v)                                           \
    do {                                                               \
        float _buf[4];                                                 \
        _mm_storeu_ps(_buf, (v));                                      \
        for (int _i = 0; _i < 4; _i++) {                               \
            if (!std::isnan(_buf[_i])) {                               \
                printf("    FAIL: Lane %d is not NaN (line %d)\n", _i, \
                       __LINE__);                                      \
                return TEST_FAIL;                                      \
            }                                                          \
        }                                                              \
    } while (0)

#define EXPECT_ALL_NAN_PD(v)                                           \
    do {                                                               \
        double _buf[2];                                                \
        _mm_storeu_pd(_buf, (v));                                      \
        for (int _i = 0; _i < 2; _i++) {                               \
            if (!std::isnan(_buf[_i])) {                               \
                printf("    FAIL: Lane %d is not NaN (line %d)\n", _i, \
                       __LINE__);                                      \
                return TEST_FAIL;                                      \
            }                                                          \
        }                                                              \
    } while (0)

#define EXPECT_ALL_BITS_PS(v, expected)                                       \
    do {                                                                      \
        uint32_t _buf[4];                                                     \
        _mm_storeu_ps(reinterpret_cast<float *>(_buf), (v));                  \
        for (int _i = 0; _i < 4; _i++) {                                      \
            if (_buf[_i] != (expected)) {                                     \
                printf("    FAIL: Lane %d bits=0x%08X expected=0x%08X\n", _i, \
                       _buf[_i], (expected));                                 \
                return TEST_FAIL;                                             \
            }                                                                 \
        }                                                                     \
    } while (0)

#define EXPECT_ALL_BITS_PD(v, expected)                          \
    do {                                                         \
        uint64_t _buf[2];                                        \
        _mm_storeu_pd(reinterpret_cast<double *>(_buf), (v));    \
        for (int _i = 0; _i < 2; _i++) {                         \
            if (_buf[_i] != (expected)) {                        \
                printf("    FAIL: Lane %d bits mismatch\n", _i); \
                return TEST_FAIL;                                \
            }                                                    \
        }                                                        \
    } while (0)

/* IEEE-754 NaN Value Generators */

/* Quiet NaN (qNaN): exponent all 1s, mantissa MSB = 1
 * Standard quiet NaN with canonical payload
 */
static inline float make_qnan_f(void)
{
    uint32_t bits = 0x7FC00000;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Signaling NaN (sNaN): exponent all 1s, mantissa MSB = 0, at least one other
 * bit set Intel x86 will convert sNaN to qNaN on most operations
 */
static inline float make_snan_f(void)
{
    uint32_t bits = 0x7F800001;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* qNaN with custom payload (bits 0-21 of mantissa) */
static inline float make_qnan_payload_f(uint32_t payload)
{
    uint32_t bits = 0x7FC00000 | (payload & 0x003FFFFF);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Negative qNaN */
static inline float make_neg_qnan_f(void)
{
    uint32_t bits = 0xFFC00000;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Double precision qNaN */
static inline double make_qnan_d(void)
{
    uint64_t bits = 0x7FF8000000000000ULL;
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

/* Double precision sNaN */
static inline double make_snan_d(void)
{
    uint64_t bits = 0x7FF0000000000001ULL;
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

/* Double precision qNaN with custom payload */
static inline double make_qnan_payload_d(uint64_t payload)
{
    uint64_t bits = 0x7FF8000000000000ULL | (payload & 0x0007FFFFFFFFFFFFULL);
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

/* Negative double qNaN */
__attribute__((unused)) static inline double make_neg_qnan_d(void)
{
    uint64_t bits = 0xFFF8000000000000ULL;
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

/* Extract float from __m128 at index */
static inline float extract_ps(__m128 v, int idx)
{
    float result[4];
    _mm_storeu_ps(result, v);
    return result[idx];
}

/* Extract double from __m128d at index */
static inline double extract_pd(__m128d v, int idx)
{
    double result[2];
    _mm_storeu_pd(result, v);
    return result[idx];
}

/* Extract uint32 from __m128 at index (for bit-level comparison) */
static inline uint32_t extract_ps_bits(__m128 v, int idx)
{
    uint32_t result[4];
    _mm_storeu_ps(reinterpret_cast<float *>(result), v);
    return result[idx];
}

/* Extract uint64 from __m128d at index (for bit-level comparison) */
static inline uint64_t extract_pd_bits(__m128d v, int idx)
{
    uint64_t result[2];
    _mm_storeu_pd(reinterpret_cast<double *>(result), v);
    return result[idx];
}

/* Check if a float is a quiet NaN (MSB of mantissa = 1) */
__attribute__((unused)) static inline bool is_qnan_f(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return ((bits & 0x7F800000) == 0x7F800000) && ((bits & 0x00400000) != 0);
}

/* Check if a float is a signaling NaN (MSB of mantissa = 0, other mantissa
 * bits != 0)
 */
__attribute__((unused)) static inline bool is_snan_f(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return ((bits & 0x7F800000) == 0x7F800000) && ((bits & 0x00400000) == 0) &&
           ((bits & 0x003FFFFF) != 0);
}

/* Check if a double is a quiet NaN */
__attribute__((unused)) static inline bool is_qnan_d(double d)
{
    uint64_t bits;
    memcpy(&bits, &d, sizeof(bits));
    return ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) &&
           ((bits & 0x0008000000000000ULL) != 0);
}

/* NaN Propagation Tests - Packed Single Precision
 * IEEE-754: Any operation with NaN operand produces NaN result
 */

TEST_CASE(add_ps_nan_first_operand)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
    __m128 c = _mm_add_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(add_ps_nan_second_operand)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
    __m128 b = _mm_set1_ps(nan);
    __m128 c = _mm_add_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(add_ps_nan_both_operands)
{
    float nan1 = make_qnan_payload_f(0x123);
    float nan2 = make_qnan_payload_f(0x456);
    __m128 a = _mm_set1_ps(nan1);
    __m128 b = _mm_set1_ps(nan2);
    __m128 c = _mm_add_ps(a, b);

    /* Result should be NaN (propagation order may vary) */
    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(sub_ps_nan_propagation)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ps(nan, 1.0f, nan, 1.0f);
    __m128 b = _mm_set_ps(1.0f, nan, 1.0f, nan);
    __m128 c = _mm_sub_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(mul_ps_nan_propagation)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ps(nan, 2.0f, nan, 2.0f);
    __m128 b = _mm_set_ps(3.0f, nan, 3.0f, nan);
    __m128 c = _mm_mul_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(mul_ps_nan_times_zero)
{
    /* IEEE-754: NaN * 0 = NaN (not 0) */
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(0.0f);
    __m128 c = _mm_mul_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(div_ps_nan_propagation)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ps(nan, 6.0f, nan, 6.0f);
    __m128 b = _mm_set_ps(2.0f, nan, 2.0f, nan);
    __m128 c = _mm_div_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(div_ps_nan_by_zero)
{
    /* IEEE-754: NaN / 0 = NaN */
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(0.0f);
    __m128 c = _mm_div_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

/* NaN Propagation Tests - Scalar Single Precision */

TEST_CASE(add_ss_nan_propagation)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);
    __m128 c = _mm_add_ss(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));

    /* Test second operand NaN */
    a = _mm_set_ss(1.0f);
    b = _mm_set_ss(nan);
    c = _mm_add_ss(a, b);
    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(sub_ss_nan_propagation)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);
    __m128 c = _mm_sub_ss(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(mul_ss_nan_propagation)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(2.0f);
    __m128 c = _mm_mul_ss(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(div_ss_nan_propagation)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(2.0f);
    __m128 c = _mm_div_ss(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

/* NaN Propagation Tests - Special Functions */

TEST_CASE(sqrt_ps_nan_input)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 c = _mm_sqrt_ps(a);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(sqrt_ss_nan_input)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 c = _mm_sqrt_ss(a);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(rsqrt_ps_nan_input)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 c = _mm_rsqrt_ps(a);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(rsqrt_ss_nan_input)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 c = _mm_rsqrt_ss(a);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(rcp_ps_nan_input)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 c = _mm_rcp_ps(a);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(rcp_ss_nan_input)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 c = _mm_rcp_ss(a);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

/* NaN Propagation Tests - Double Precision */

TEST_CASE(add_pd_nan_propagation)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_pd(nan, 1.0);
    __m128d b = _mm_set_pd(2.0, nan);
    __m128d c = _mm_add_pd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));
    EXPECT_NAN_D(extract_pd(c, 1));

    return TEST_SUCCESS;
}

TEST_CASE(sub_pd_nan_propagation)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_pd(nan, 1.0);
    __m128d b = _mm_set_pd(2.0, nan);
    __m128d c = _mm_sub_pd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));
    EXPECT_NAN_D(extract_pd(c, 1));

    return TEST_SUCCESS;
}

TEST_CASE(mul_pd_nan_propagation)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_pd(nan, 2.0);
    __m128d b = _mm_set_pd(3.0, nan);
    __m128d c = _mm_mul_pd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));
    EXPECT_NAN_D(extract_pd(c, 1));

    return TEST_SUCCESS;
}

TEST_CASE(div_pd_nan_propagation)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_pd(nan, 6.0);
    __m128d b = _mm_set_pd(2.0, nan);
    __m128d c = _mm_div_pd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));
    EXPECT_NAN_D(extract_pd(c, 1));

    return TEST_SUCCESS;
}

TEST_CASE(sqrt_pd_nan_input)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d c = _mm_sqrt_pd(a);

    EXPECT_NAN_D(extract_pd(c, 0));
    EXPECT_NAN_D(extract_pd(c, 1));

    return TEST_SUCCESS;
}

/* Scalar double precision */
TEST_CASE(add_sd_nan_propagation)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);
    __m128d c = _mm_add_sd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(sub_sd_nan_propagation)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);
    __m128d c = _mm_sub_sd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(mul_sd_nan_propagation)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(2.0);
    __m128d c = _mm_mul_sd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(div_sd_nan_propagation)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(2.0);
    __m128d c = _mm_div_sd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(sqrt_sd_nan_input)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d c = _mm_sqrt_sd(a, a);

    EXPECT_NAN_D(extract_pd(c, 0));

    return TEST_SUCCESS;
}

/* NaN Comparison Tests - Single Precision
 * IEEE-754: All comparisons with NaN return false, except !=
 */

TEST_CASE(cmpeq_ps_nan)
{
    float nan = make_qnan_f();
    float normal = 1.0f;

    /* NaN == normal should be false (all zeros) */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_cmpeq_ps(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    /* NaN == NaN should also be false */
    c = _mm_cmpeq_ps(a, a);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpneq_ps_nan)
{
    float nan = make_qnan_f();
    float normal = 1.0f;

    /* NaN != normal should be true (all ones) */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_cmpneq_ps(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0xFFFFFFFF);

    /* NaN != NaN should also be true */
    c = _mm_cmpneq_ps(a, a);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0xFFFFFFFF);

    return TEST_SUCCESS;
}

TEST_CASE(cmplt_ps_nan)
{
    float nan = make_qnan_f();
    float normal = 1.0f;

    /* NaN < normal should be false */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_cmplt_ps(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    /* normal < NaN should also be false */
    c = _mm_cmplt_ps(b, a);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmple_ps_nan)
{
    float nan = make_qnan_f();
    float normal = 1.0f;

    /* NaN <= normal should be false */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_cmple_ps(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpgt_ps_nan)
{
    float nan = make_qnan_f();
    float normal = 1.0f;

    /* NaN > normal should be false */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_cmpgt_ps(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpge_ps_nan)
{
    float nan = make_qnan_f();
    float normal = 1.0f;

    /* NaN >= normal should be false */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_cmpge_ps(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpord_ps_nan)
{
    float nan = make_qnan_f();
    float normal = 1.0f;

    /* cmpord(NaN, normal) should be false (unordered) */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_cmpord_ps(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    /* cmpord(normal, normal) should be true */
    __m128 d = _mm_set1_ps(2.0f);
    c = _mm_cmpord_ps(b, d);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0xFFFFFFFF);

    return TEST_SUCCESS;
}

TEST_CASE(cmpunord_ps_nan)
{
    float nan = make_qnan_f();
    float normal = 1.0f;

    /* cmpunord(NaN, normal) should be true */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_cmpunord_ps(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0xFFFFFFFF);

    /* cmpunord(normal, normal) should be false */
    c = _mm_cmpunord_ps(b, b);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    return TEST_SUCCESS;
}

/* Scalar comparisons */
TEST_CASE(cmpeq_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);
    __m128 c = _mm_cmpeq_ss(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpneq_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);
    __m128 c = _mm_cmpneq_ss(a, b);

    EXPECT_TRUE(extract_ps_bits(c, 0) == 0xFFFFFFFF);

    return TEST_SUCCESS;
}

/* NaN Comparison Tests - Double Precision */

TEST_CASE(cmpeq_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_cmpeq_pd(a, b);

    EXPECT_TRUE(extract_pd_bits(c, 0) == 0);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpneq_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_cmpneq_pd(a, b);

    EXPECT_TRUE(extract_pd_bits(c, 0) == 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0xFFFFFFFFFFFFFFFFULL);

    return TEST_SUCCESS;
}

TEST_CASE(cmplt_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_cmplt_pd(a, b);

    EXPECT_TRUE(extract_pd_bits(c, 0) == 0);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0);

    /* normal < NaN should also be false */
    c = _mm_cmplt_pd(b, a);
    EXPECT_TRUE(extract_pd_bits(c, 0) == 0);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpord_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_cmpord_pd(a, b);

    EXPECT_TRUE(extract_pd_bits(c, 0) == 0);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0);

    /* cmpord(normal, normal) should be true */
    __m128d d = _mm_set1_pd(2.0);
    c = _mm_cmpord_pd(b, d);
    EXPECT_TRUE(extract_pd_bits(c, 0) == 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0xFFFFFFFFFFFFFFFFULL);

    return TEST_SUCCESS;
}

TEST_CASE(cmpunord_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_cmpunord_pd(a, b);

    EXPECT_TRUE(extract_pd_bits(c, 0) == 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0xFFFFFFFFFFFFFFFFULL);

    /* cmpunord(normal, normal) should be false */
    c = _mm_cmpunord_pd(b, b);
    EXPECT_TRUE(extract_pd_bits(c, 0) == 0);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmple_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_cmple_pd(a, b);

    /* NaN <= normal should be false */
    EXPECT_TRUE(extract_pd_bits(c, 0) == 0);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpgt_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_cmpgt_pd(a, b);

    /* NaN > normal should be false */
    EXPECT_TRUE(extract_pd_bits(c, 0) == 0);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(cmpge_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_cmpge_pd(a, b);

    /* NaN >= normal should be false */
    EXPECT_TRUE(extract_pd_bits(c, 0) == 0);
    EXPECT_TRUE(extract_pd_bits(c, 1) == 0);

    return TEST_SUCCESS;
}

/* Min/Max NaN Behavior Tests
 * SSE behavior: If either operand is NaN, return the SECOND operand
 * NEON behavior: May differ, handled by SSE2NEON_PRECISE_MINMAX flag
 */

TEST_CASE(min_ps_nan_first)
{
    /* SSE: min(NaN, x) = x (returns second operand) */
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(2.0f);
    __m128 c = _mm_min_ps(a, b);

#if SSE2NEON_PRECISE_MINMAX
    /* Precise mode: Should match SSE behavior */
    EXPECT_TRUE(extract_ps(c, 0) == 2.0f);
#else
    /* Default mode: Result may be NaN or 2.0f depending on NEON behavior */
    (void) c;
#endif

    return TEST_SUCCESS;
}

TEST_CASE(min_ps_nan_second)
{
    /* SSE: min(x, NaN) = NaN (returns second operand) */
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(2.0f);
    __m128 b = _mm_set1_ps(nan);
    __m128 c = _mm_min_ps(a, b);

#if SSE2NEON_PRECISE_MINMAX
    /* Precise mode: Should return NaN (second operand) */
    EXPECT_NAN_F(extract_ps(c, 0));
#else
    (void) c;
#endif

    return TEST_SUCCESS;
}

TEST_CASE(max_ps_nan_first)
{
    /* SSE: max(NaN, x) = x (returns second operand) */
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(2.0f);
    __m128 c = _mm_max_ps(a, b);

#if SSE2NEON_PRECISE_MINMAX
    EXPECT_TRUE(extract_ps(c, 0) == 2.0f);
#else
    (void) c;
#endif

    return TEST_SUCCESS;
}

TEST_CASE(max_ps_nan_second)
{
    /* SSE: max(x, NaN) = NaN (returns second operand) */
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(2.0f);
    __m128 b = _mm_set1_ps(nan);
    __m128 c = _mm_max_ps(a, b);

#if SSE2NEON_PRECISE_MINMAX
    EXPECT_NAN_F(extract_ps(c, 0));
#else
    (void) c;
#endif

    return TEST_SUCCESS;
}

TEST_CASE(min_pd_nan_behavior)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(2.0);
    __m128d c = _mm_min_pd(a, b);

#if SSE2NEON_PRECISE_MINMAX
    EXPECT_TRUE(extract_pd(c, 0) == 2.0);
#else
    (void) c;
#endif

    return TEST_SUCCESS;
}

TEST_CASE(max_pd_nan_behavior)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(2.0);
    __m128d b = _mm_set1_pd(nan);
    __m128d c = _mm_max_pd(a, b);

#if SSE2NEON_PRECISE_MINMAX
    EXPECT_NAN_D(extract_pd(c, 0));
#else
    (void) c;
#endif

    return TEST_SUCCESS;
}

/* NaN Payload Preservation Tests
 * SSE preserves NaN payload bits through operations
 * NEON may canonicalize NaN to a default quiet NaN
 */

/* Helper to extract payload from float NaN */
static inline uint32_t get_nan_payload_f(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits & 0x003FFFFF;
}

/* Helper to extract sign bit from float */
static inline uint32_t get_sign_bit_f(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (bits >> 31) & 1;
}

/* Helper to extract payload from double NaN */
static inline uint64_t get_nan_payload_d(double d)
{
    uint64_t bits;
    memcpy(&bits, &d, sizeof(bits));
    return bits & 0x0007FFFFFFFFFFFFULL;
}

/* Helper to extract sign bit from double */
__attribute__((unused)) static inline uint64_t get_sign_bit_d(double d)
{
    uint64_t bits;
    memcpy(&bits, &d, sizeof(bits));
    return (bits >> 63) & 1;
}

TEST_CASE(nan_payload_through_copy)
{
    /* Test that NaN payload is preserved through simple copy operations */
    uint32_t payload = 0x12345;
    float nan = make_qnan_payload_f(payload);
    __m128 a = _mm_set1_ps(nan);

    float result[4];
    _mm_storeu_ps(result, a);

    /* All lanes should contain the original NaN with preserved payload */
    EXPECT_NAN_F(result[0]);
    EXPECT_NAN_F(result[1]);
    EXPECT_NAN_F(result[2]);
    EXPECT_NAN_F(result[3]);

    /* Verify payload is preserved through copy */
    EXPECT_TRUE(get_nan_payload_f(result[0]) == payload);
    EXPECT_TRUE(get_nan_payload_f(result[1]) == payload);
    EXPECT_TRUE(get_nan_payload_f(result[2]) == payload);
    EXPECT_TRUE(get_nan_payload_f(result[3]) == payload);

    return TEST_SUCCESS;
}

TEST_CASE(nan_payload_through_arithmetic)
{
    /* Test NaN propagation through arithmetic (payload may or may not be
     * preserved)
     */
    uint32_t payload = 0xABCDE;
    float nan = make_qnan_payload_f(payload);
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(1.0f);
    __m128 c = _mm_add_ps(a, b);

    /* Result must be NaN */
    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    /* On x86, payload should be preserved through arithmetic */
#if defined(__x86_64__) || defined(__i386__)
    EXPECT_TRUE(get_nan_payload_f(extract_ps(c, 0)) == payload);
#endif
    /* On ARM/NEON, payload may be canonicalized - just verify still NaN */

    return TEST_SUCCESS;
}

TEST_CASE(nan_sign_preservation)
{
    /* Test that negative NaN sign is handled correctly */
    float neg_nan = make_neg_qnan_f();
    __m128 a = _mm_set1_ps(neg_nan);

    float result[4];
    _mm_storeu_ps(result, a);

    /* All results should be NaN with sign bit = 1 */
    EXPECT_NAN_F(result[0]);
    EXPECT_NAN_F(result[1]);
    EXPECT_NAN_F(result[2]);
    EXPECT_NAN_F(result[3]);

    /* Verify sign bit is preserved through copy */
    EXPECT_TRUE(get_sign_bit_f(result[0]) == 1);
    EXPECT_TRUE(get_sign_bit_f(result[1]) == 1);
    EXPECT_TRUE(get_sign_bit_f(result[2]) == 1);
    EXPECT_TRUE(get_sign_bit_f(result[3]) == 1);

    return TEST_SUCCESS;
}

TEST_CASE(nan_payload_double)
{
    /* Test double precision NaN payload preservation */
    uint64_t payload = 0x123456789ABCULL;
    double nan = make_qnan_payload_d(payload);
    __m128d a = _mm_set1_pd(nan);

    double result[2];
    _mm_storeu_pd(result, a);

    EXPECT_NAN_D(result[0]);
    EXPECT_NAN_D(result[1]);

    /* Verify payload is preserved */
    EXPECT_TRUE(get_nan_payload_d(result[0]) == payload);
    EXPECT_TRUE(get_nan_payload_d(result[1]) == payload);

    return TEST_SUCCESS;
}

/* Bitwise Operations NaN Payload Tests
 * Bitwise ops should preserve NaN bits exactly (no arithmetic modification)
 */

TEST_CASE(and_ps_nan_payload)
{
    /* AND with all-ones should preserve NaN exactly */
    uint32_t payload = 0x12345;
    float nan = make_qnan_payload_f(payload);
    __m128 a = _mm_set1_ps(nan);
    __m128 ones = _mm_castsi128_ps(_mm_set1_epi32(-1));
    __m128 c = _mm_and_ps(a, ones);

    float result[4];
    _mm_storeu_ps(result, c);

    for (int i = 0; i < 4; i++) {
        EXPECT_NAN_F(result[i]);
        EXPECT_TRUE(get_nan_payload_f(result[i]) == payload);
    }

    return TEST_SUCCESS;
}

TEST_CASE(or_ps_nan_payload)
{
    /* OR with zeros should preserve NaN exactly */
    uint32_t payload = 0x23456;
    float nan = make_qnan_payload_f(payload);
    __m128 a = _mm_set1_ps(nan);
    __m128 zeros = _mm_setzero_ps();
    __m128 c = _mm_or_ps(a, zeros);

    float result[4];
    _mm_storeu_ps(result, c);

    for (int i = 0; i < 4; i++) {
        EXPECT_NAN_F(result[i]);
        EXPECT_TRUE(get_nan_payload_f(result[i]) == payload);
    }

    return TEST_SUCCESS;
}

TEST_CASE(xor_ps_nan_with_zero)
{
    /* XOR with zeros should preserve NaN exactly */
    uint32_t payload = 0x34567;
    float nan = make_qnan_payload_f(payload);
    __m128 a = _mm_set1_ps(nan);
    __m128 zeros = _mm_setzero_ps();
    __m128 c = _mm_xor_ps(a, zeros);

    float result[4];
    _mm_storeu_ps(result, c);

    for (int i = 0; i < 4; i++) {
        EXPECT_NAN_F(result[i]);
        EXPECT_TRUE(get_nan_payload_f(result[i]) == payload);
    }

    return TEST_SUCCESS;
}

TEST_CASE(andnot_ps_nan)
{
    /* ANDNOT: (~a) & b - NaN in second operand should be preserved */
    uint32_t payload = 0x45678;
    float nan = make_qnan_payload_f(payload);
    __m128 zeros = _mm_setzero_ps();
    __m128 b = _mm_set1_ps(nan);
    __m128 c = _mm_andnot_ps(zeros, b);

    float result[4];
    _mm_storeu_ps(result, c);

    for (int i = 0; i < 4; i++) {
        EXPECT_NAN_F(result[i]);
        EXPECT_TRUE(get_nan_payload_f(result[i]) == payload);
    }

    return TEST_SUCCESS;
}

/* Shuffle NaN Payload Tests */

TEST_CASE(shuffle_ps_nan_payload)
{
    /* Shuffle should preserve NaN payloads */
    uint32_t p0 = 0x11111, p1 = 0x22222, p2 = 0x33333, p3 = 0x44444;
    float n0 = make_qnan_payload_f(p0);
    float n1 = make_qnan_payload_f(p1);
    float n2 = make_qnan_payload_f(p2);
    float n3 = make_qnan_payload_f(p3);

    __m128 a = _mm_set_ps(n3, n2, n1, n0);
    /* Reverse the order: 0,1,2,3 -> 3,2,1,0 */
    __m128 c = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));

    float result[4];
    _mm_storeu_ps(result, c);

    /* After shuffle: result[0]=n3, result[1]=n2, result[2]=n1, result[3]=n0 */
    EXPECT_NAN_F(result[0]);
    EXPECT_NAN_F(result[1]);
    EXPECT_NAN_F(result[2]);
    EXPECT_NAN_F(result[3]);

    EXPECT_TRUE(get_nan_payload_f(result[0]) == p3);
    EXPECT_TRUE(get_nan_payload_f(result[1]) == p2);
    EXPECT_TRUE(get_nan_payload_f(result[2]) == p1);
    EXPECT_TRUE(get_nan_payload_f(result[3]) == p0);

    return TEST_SUCCESS;
}

TEST_CASE(unpacklo_ps_nan_payload)
{
    /* Unpack should preserve NaN payloads */
    uint32_t p0 = 0xAAAAA, p1 = 0xBBBBB;
    float n0 = make_qnan_payload_f(p0);
    float n1 = make_qnan_payload_f(p1);

    __m128 a = _mm_set_ps(1.0f, 2.0f, n1, n0);
    __m128 b = _mm_set_ps(3.0f, 4.0f, 5.0f, 6.0f);
    __m128 c = _mm_unpacklo_ps(a, b);

    /* Result: {a[0], b[0], a[1], b[1]} = {n0, 6.0, n1, 5.0} */
    float result[4];
    _mm_storeu_ps(result, c);

    EXPECT_NAN_F(result[0]);
    EXPECT_TRUE(get_nan_payload_f(result[0]) == p0);
    EXPECT_NAN_F(result[2]);
    EXPECT_TRUE(get_nan_payload_f(result[2]) == p1);

    return TEST_SUCCESS;
}

TEST_CASE(movehl_ps_nan_payload)
{
    /* Move high-to-low should preserve NaN payloads */
    uint32_t p2 = 0xCCCCC, p3 = 0xDDDDD;
    float n2 = make_qnan_payload_f(p2);
    float n3 = make_qnan_payload_f(p3);

    __m128 a = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
    __m128 b = _mm_set_ps(n3, n2, 5.0f, 6.0f);
    __m128 c = _mm_movehl_ps(a, b);

    /* Result: {b[2], b[3], a[2], a[3]} = {n2, n3, 2.0, 1.0} */
    float result[4];
    _mm_storeu_ps(result, c);

    EXPECT_NAN_F(result[0]);
    EXPECT_TRUE(get_nan_payload_f(result[0]) == p2);
    EXPECT_NAN_F(result[1]);
    EXPECT_TRUE(get_nan_payload_f(result[1]) == p3);

    return TEST_SUCCESS;
}

/* Signaling NaN (sNaN) to Quiet NaN (qNaN) Conversion Tests
 * IEEE-754: Most operations on sNaN should produce qNaN and raise exception
 */

TEST_CASE(snan_to_qnan_add)
{
    /* Adding with sNaN should convert it to qNaN */
    float snan = make_snan_f();
    __m128 a = _mm_set1_ps(snan);
    __m128 b = _mm_set1_ps(1.0f);
    __m128 c = _mm_add_ps(a, b);

    /* Result should be NaN (quiet) */
    EXPECT_NAN_F(extract_ps(c, 0));

    /* On x86, the result should be qNaN (MSB of mantissa = 1) */
#if defined(__x86_64__) || defined(__i386__)
    EXPECT_TRUE(is_qnan_f(extract_ps(c, 0)));
#endif

    return TEST_SUCCESS;
}

TEST_CASE(snan_to_qnan_mul)
{
    float snan = make_snan_f();
    __m128 a = _mm_set1_ps(snan);
    __m128 b = _mm_set1_ps(2.0f);
    __m128 c = _mm_mul_ps(a, b);

    /* All lanes should produce NaN */
    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(snan_to_qnan_sub)
{
    float snan = make_snan_f();
    __m128 a = _mm_set1_ps(snan);
    __m128 b = _mm_set1_ps(1.0f);
    __m128 c = _mm_sub_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(snan_to_qnan_div)
{
    float snan = make_snan_f();
    __m128 a = _mm_set1_ps(snan);
    __m128 b = _mm_set1_ps(2.0f);
    __m128 c = _mm_div_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(snan_to_qnan_sqrt)
{
    float snan = make_snan_f();
    __m128 a = _mm_set1_ps(snan);
    __m128 c = _mm_sqrt_ps(a);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(snan_to_qnan_rsqrt)
{
    float snan = make_snan_f();
    __m128 a = _mm_set1_ps(snan);
    __m128 c = _mm_rsqrt_ps(a);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(snan_to_qnan_rcp)
{
    float snan = make_snan_f();
    __m128 a = _mm_set1_ps(snan);
    __m128 c = _mm_rcp_ps(a);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(snan_comparison)
{
    /* Comparison with sNaN should work like qNaN (return false for ordered) */
    float snan = make_snan_f();
    __m128 a = _mm_set1_ps(snan);
    __m128 b = _mm_set1_ps(1.0f);

    /* sNaN == normal should be false (all lanes) */
    __m128 c = _mm_cmpeq_ps(a, b);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0);

    /* sNaN is unordered with everything (all lanes) */
    c = _mm_cmpunord_ps(a, b);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 1) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 2) == 0xFFFFFFFF);
    EXPECT_TRUE(extract_ps_bits(c, 3) == 0xFFFFFFFF);

    /* sNaN < normal should be false */
    c = _mm_cmplt_ps(a, b);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);

    /* sNaN > normal should be false */
    c = _mm_cmpgt_ps(a, b);
    EXPECT_TRUE(extract_ps_bits(c, 0) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(snan_double_to_qnan)
{
    double snan = make_snan_d();
    __m128d a = _mm_set1_pd(snan);
    __m128d b = _mm_set1_pd(1.0);
    __m128d c = _mm_add_pd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));

    return TEST_SUCCESS;
}

/* NaN Generation Tests
 * Operations that generate NaN from non-NaN inputs
 */

TEST_CASE(nan_generation_zero_div_zero)
{
    /* 0.0 / 0.0 = NaN */
    __m128 a = _mm_set1_ps(0.0f);
    __m128 b = _mm_set1_ps(0.0f);
    __m128 c = _mm_div_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(nan_generation_inf_minus_inf)
{
    /* inf - inf = NaN */
    __m128 a = _mm_set1_ps(INFINITY);
    __m128 b = _mm_set1_ps(INFINITY);
    __m128 c = _mm_sub_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(nan_generation_inf_times_zero)
{
    /* inf * 0 = NaN */
    __m128 a = _mm_set1_ps(INFINITY);
    __m128 b = _mm_set1_ps(0.0f);
    __m128 c = _mm_mul_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(nan_generation_sqrt_negative)
{
    /* sqrt(-1) = NaN */
    __m128 a = _mm_set1_ps(-1.0f);
    __m128 c = _mm_sqrt_ps(a);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(nan_generation_inf_div_inf)
{
    /* inf / inf = NaN */
    __m128 a = _mm_set1_ps(INFINITY);
    __m128 b = _mm_set1_ps(INFINITY);
    __m128 c = _mm_div_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(nan_generation_double)
{
    /* Test NaN generation for double precision */
    __m128d a = _mm_set1_pd(0.0);
    __m128d b = _mm_set1_pd(0.0);
    __m128d c = _mm_div_pd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));

    /* sqrt(-1) = NaN */
    a = _mm_set1_pd(-1.0);
    c = _mm_sqrt_pd(a);
    EXPECT_NAN_D(extract_pd(c, 0));

    return TEST_SUCCESS;
}

/* NaN to Integer Conversion Tests
 * x86 SSE: Converting NaN to integer returns "Integer Indefinite" (0x80000000)
 * ARM NEON: May return 0 or saturate differently
 */

TEST_CASE(cvtps_epi32_nan)
{
    /* x86 SSE: NaN converts to 0x80000000 (Integer Indefinite) */
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128i res = _mm_cvtps_epi32(a);

    int32_t r0, r1, r2, r3;
    r0 = _mm_extract_epi32(res, 0);
    r1 = _mm_extract_epi32(res, 1);
    r2 = _mm_extract_epi32(res, 2);
    r3 = _mm_extract_epi32(res, 3);

    /* All lanes should be Integer Indefinite */
    EXPECT_TRUE(r0 == static_cast<int32_t>(0x80000000));
    EXPECT_TRUE(r1 == static_cast<int32_t>(0x80000000));
    EXPECT_TRUE(r2 == static_cast<int32_t>(0x80000000));
    EXPECT_TRUE(r3 == static_cast<int32_t>(0x80000000));

    return TEST_SUCCESS;
}

TEST_CASE(cvttps_epi32_nan)
{
    /* Truncating conversion: NaN also returns Integer Indefinite */
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128i res = _mm_cvttps_epi32(a);

    int32_t r0 = _mm_extract_epi32(res, 0);
    EXPECT_TRUE(r0 == static_cast<int32_t>(0x80000000));

    return TEST_SUCCESS;
}

TEST_CASE(cvtss_si32_nan)
{
    /* Scalar conversion: NaN returns Integer Indefinite on x86
     * Note: sse2neon may return 0 on some ARM implementations due to
     * differences in fcvtns instruction behavior. This is a known divergence.
     */
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    int32_t res = _mm_cvtss_si32(a);

    /* x86 SSE returns 0x80000000 (Integer Indefinite)
     * ARM NEON fcvtns may return 0 for NaN
     */
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
    /* Accept either 0x80000000 (emulated) or 0 (native ARM behavior) */
    EXPECT_TRUE(res == static_cast<int32_t>(0x80000000) || res == 0);
#else
    EXPECT_TRUE(res == static_cast<int32_t>(0x80000000));
#endif

    return TEST_SUCCESS;
}

TEST_CASE(cvttss_si32_nan)
{
    /* Scalar truncating conversion */
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    int32_t res = _mm_cvttss_si32(a);

    EXPECT_TRUE(res == static_cast<int32_t>(0x80000000));

    return TEST_SUCCESS;
}

TEST_CASE(cvtpd_epi32_nan)
{
    /* Double to int32 conversion */
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128i res = _mm_cvtpd_epi32(a);

    int32_t r0 = _mm_extract_epi32(res, 0);
    int32_t r1 = _mm_extract_epi32(res, 1);

    EXPECT_TRUE(r0 == static_cast<int32_t>(0x80000000));
    EXPECT_TRUE(r1 == static_cast<int32_t>(0x80000000));

    return TEST_SUCCESS;
}

TEST_CASE(cvttpd_epi32_nan)
{
    /* Double truncating conversion */
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128i res = _mm_cvttpd_epi32(a);

    int32_t r0 = _mm_extract_epi32(res, 0);
    EXPECT_TRUE(r0 == static_cast<int32_t>(0x80000000));

    return TEST_SUCCESS;
}

TEST_CASE(cvtsd_si32_nan)
{
    /* Scalar double to int32 */
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    int32_t res = _mm_cvtsd_si32(a);

    EXPECT_TRUE(res == static_cast<int32_t>(0x80000000));

    return TEST_SUCCESS;
}

/* 64-bit Integer Conversion Tests
 * Only available on 64-bit platforms
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
    defined(_M_ARM64)

TEST_CASE(cvtss_si64_nan)
{
    /* Scalar float to int64 */
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    int64_t res = _mm_cvtss_si64(a);

    /* x86 returns 0x8000000000000000 (Integer Indefinite for 64-bit)
     * ARM may return 0
     */
    int64_t indefinite = static_cast<int64_t>(0x8000000000000000ULL);
#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_TRUE(res == indefinite || res == 0);
#else
    EXPECT_TRUE(res == indefinite);
#endif

    return TEST_SUCCESS;
}

TEST_CASE(cvttss_si64_nan)
{
    /* Scalar float to int64 with truncation */
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    int64_t res = _mm_cvttss_si64(a);

    int64_t indefinite = static_cast<int64_t>(0x8000000000000000ULL);
    EXPECT_TRUE(res == indefinite);

    return TEST_SUCCESS;
}

TEST_CASE(cvtsd_si64_nan)
{
    /* Scalar double to int64 */
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    int64_t res = _mm_cvtsd_si64(a);

    int64_t indefinite = static_cast<int64_t>(0x8000000000000000ULL);
#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_TRUE(res == indefinite || res == 0);
#else
    EXPECT_TRUE(res == indefinite);
#endif

    return TEST_SUCCESS;
}

TEST_CASE(cvttsd_si64_nan)
{
    /* Scalar double to int64 with truncation */
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    int64_t res = _mm_cvttsd_si64(a);

    int64_t indefinite = static_cast<int64_t>(0x8000000000000000ULL);
    EXPECT_TRUE(res == indefinite);

    return TEST_SUCCESS;
}

#endif /* 64-bit platforms */

/* Scalar Comparison (COMI/UCOMI) NaN Tests
 * comieq/ucomieq: Ordered/unordered scalar comparison setting flags
 */

TEST_CASE(comieq_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);

    /* NaN == normal should return 0 (not equal) */
    EXPECT_TRUE(_mm_comieq_ss(a, b) == 0);
    /* NaN == NaN should also return 0 */
    EXPECT_TRUE(_mm_comieq_ss(a, a) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comineq_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);

    /* NaN != normal should return 1 (not equal) */
    EXPECT_TRUE(_mm_comineq_ss(a, b) == 1);
    /* NaN != NaN should also return 1 */
    EXPECT_TRUE(_mm_comineq_ss(a, a) == 1);

    return TEST_SUCCESS;
}

TEST_CASE(comilt_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);

    /* NaN < normal should return 0 */
    EXPECT_TRUE(_mm_comilt_ss(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comile_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);

    /* NaN <= normal should return 0 */
    EXPECT_TRUE(_mm_comile_ss(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comigt_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);

    /* NaN > normal should return 0 */
    EXPECT_TRUE(_mm_comigt_ss(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comige_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);

    /* NaN >= normal should return 0 */
    EXPECT_TRUE(_mm_comige_ss(a, b) == 0);

    return TEST_SUCCESS;
}

/* Unordered comparisons (ucomi) - quiet, don't signal on QNaN */
TEST_CASE(ucomieq_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);

    /* NaN == normal should return 0 */
    EXPECT_TRUE(_mm_ucomieq_ss(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(ucomineq_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ss(nan);
    __m128 b = _mm_set_ss(1.0f);

    /* NaN != normal should return 1 */
    EXPECT_TRUE(_mm_ucomineq_ss(a, b) == 1);

    return TEST_SUCCESS;
}

/* Double precision scalar comparisons */
TEST_CASE(comieq_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_comieq_sd(a, b) == 0);
    EXPECT_TRUE(_mm_comieq_sd(a, a) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comineq_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_comineq_sd(a, b) == 1);

    return TEST_SUCCESS;
}

TEST_CASE(ucomieq_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_ucomieq_sd(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comilt_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    /* NaN < normal should return 0 */
    EXPECT_TRUE(_mm_comilt_sd(a, b) == 0);
    /* normal < NaN should also return 0 */
    EXPECT_TRUE(_mm_comilt_sd(b, a) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comile_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_comile_sd(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comigt_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_comigt_sd(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(comige_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_comige_sd(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(ucomineq_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    /* NaN != normal should return 1 */
    EXPECT_TRUE(_mm_ucomineq_sd(a, b) == 1);

    return TEST_SUCCESS;
}

TEST_CASE(ucomilt_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_ucomilt_sd(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(ucomile_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_ucomile_sd(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(ucomigt_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_ucomigt_sd(a, b) == 0);

    return TEST_SUCCESS;
}

TEST_CASE(ucomige_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_sd(nan);
    __m128d b = _mm_set_sd(1.0);

    EXPECT_TRUE(_mm_ucomige_sd(a, b) == 0);

    return TEST_SUCCESS;
}

/* Rounding with NaN Tests (SSE4.1)
 * round(NaN) should return NaN
 */

TEST_CASE(round_ps_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);

    /* All rounding modes should preserve NaN */
    __m128 r_nearest = _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
    __m128 r_floor = _mm_round_ps(a, _MM_FROUND_TO_NEG_INF);
    __m128 r_ceil = _mm_round_ps(a, _MM_FROUND_TO_POS_INF);
    __m128 r_trunc = _mm_round_ps(a, _MM_FROUND_TO_ZERO);

    /* Check all lanes for each rounding mode */
    EXPECT_NAN_F(extract_ps(r_nearest, 0));
    EXPECT_NAN_F(extract_ps(r_nearest, 1));
    EXPECT_NAN_F(extract_ps(r_nearest, 2));
    EXPECT_NAN_F(extract_ps(r_nearest, 3));

    EXPECT_NAN_F(extract_ps(r_floor, 0));
    EXPECT_NAN_F(extract_ps(r_floor, 1));
    EXPECT_NAN_F(extract_ps(r_floor, 2));
    EXPECT_NAN_F(extract_ps(r_floor, 3));

    EXPECT_NAN_F(extract_ps(r_ceil, 0));
    EXPECT_NAN_F(extract_ps(r_ceil, 1));
    EXPECT_NAN_F(extract_ps(r_ceil, 2));
    EXPECT_NAN_F(extract_ps(r_ceil, 3));

    EXPECT_NAN_F(extract_ps(r_trunc, 0));
    EXPECT_NAN_F(extract_ps(r_trunc, 1));
    EXPECT_NAN_F(extract_ps(r_trunc, 2));
    EXPECT_NAN_F(extract_ps(r_trunc, 3));

    return TEST_SUCCESS;
}

TEST_CASE(round_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);

    __m128d r_nearest = _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
    __m128d r_floor = _mm_round_pd(a, _MM_FROUND_TO_NEG_INF);
    __m128d r_ceil = _mm_round_pd(a, _MM_FROUND_TO_POS_INF);
    __m128d r_trunc = _mm_round_pd(a, _MM_FROUND_TO_ZERO);

    EXPECT_NAN_D(extract_pd(r_nearest, 0));
    EXPECT_NAN_D(extract_pd(r_nearest, 1));
    EXPECT_NAN_D(extract_pd(r_floor, 0));
    EXPECT_NAN_D(extract_pd(r_floor, 1));
    EXPECT_NAN_D(extract_pd(r_ceil, 0));
    EXPECT_NAN_D(extract_pd(r_ceil, 1));
    EXPECT_NAN_D(extract_pd(r_trunc, 0));
    EXPECT_NAN_D(extract_pd(r_trunc, 1));

    return TEST_SUCCESS;
}

TEST_CASE(floor_ps_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 r = _mm_floor_ps(a);

    EXPECT_NAN_F(extract_ps(r, 0));
    EXPECT_NAN_F(extract_ps(r, 1));
    EXPECT_NAN_F(extract_ps(r, 2));
    EXPECT_NAN_F(extract_ps(r, 3));

    return TEST_SUCCESS;
}

TEST_CASE(ceil_ps_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set1_ps(nan);
    __m128 r = _mm_ceil_ps(a);

    EXPECT_NAN_F(extract_ps(r, 0));
    EXPECT_NAN_F(extract_ps(r, 1));
    EXPECT_NAN_F(extract_ps(r, 2));
    EXPECT_NAN_F(extract_ps(r, 3));

    return TEST_SUCCESS;
}

/* Horizontal Operation NaN Tests */

TEST_CASE(hadd_ps_nan_propagation)
{
    /* hadd result: [a0+a1, a2+a3, b0+b1, b2+b3]
     * _mm_set_ps(d,c,b,a) creates {a,b,c,d} where index 0=a, index 1=b, etc.
     */
    float nan = make_qnan_f();

    /* Put NaN in position that will be added */
    __m128 a = _mm_set_ps(1.0f, 2.0f, nan, 4.0f); /* a = {4, nan, 2, 1} */
    __m128 b = _mm_set_ps(5.0f, 6.0f, 7.0f, 8.0f);
    __m128 c = _mm_hadd_ps(a, b);

    /* c[0] = a[0]+a[1] = 4+nan = NaN */
    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(hsub_ps_nan_propagation)
{
    /* hsub result: [a0-a1, a2-a3, b0-b1, b2-b3]
     * _mm_set_ps(d,c,b,a) creates {a,b,c,d} where index 0=a, index 1=b, etc.
     */
    float nan = make_qnan_f();

    /* Put NaN in position 1 (will be subtracted from position 0) */
    __m128 a = _mm_set_ps(1.0f, 2.0f, nan, 4.0f); /* a = {4, nan, 2, 1} */
    __m128 b = _mm_set_ps(5.0f, 6.0f, 7.0f, 8.0f);
    __m128 c = _mm_hsub_ps(a, b);

    /* c[0] = a[0]-a[1] = 4-nan = NaN */
    EXPECT_NAN_F(extract_ps(c, 0));

    return TEST_SUCCESS;
}

/* Blend/Select with NaN Tests */

TEST_CASE(blendv_ps_nan_selector)
{
    /* When selector contains NaN, its sign bit determines selection
     * NaN with sign bit 0 = select from a
     * NaN with sign bit 1 = select from b
     */
    float pos_nan = make_qnan_f();
    float neg_nan = make_neg_qnan_f();

    __m128 a = _mm_set1_ps(1.0f);
    __m128 b = _mm_set1_ps(2.0f);
    __m128 mask = _mm_set_ps(neg_nan, pos_nan, neg_nan, pos_nan);

    __m128 c = _mm_blendv_ps(a, b, mask);

    /* Positive NaN (sign=0) selects from a,
     * negative NaN (sign=1) selects from b
     */
    EXPECT_TRUE(extract_ps(c, 0) == 1.0f); /* pos_nan -> a */
    EXPECT_TRUE(extract_ps(c, 1) == 2.0f); /* neg_nan -> b */
    EXPECT_TRUE(extract_ps(c, 2) == 1.0f); /* pos_nan -> a */
    EXPECT_TRUE(extract_ps(c, 3) == 2.0f); /* neg_nan -> b */

    return TEST_SUCCESS;
}

TEST_CASE(blendv_ps_nan_operands)
{
    /* Blending NaN values - NaN should be preserved in output */
    float nan = make_qnan_f();

    __m128 a = _mm_set_ps(nan, 1.0f, nan, 1.0f);
    __m128 b = _mm_set_ps(2.0f, nan, 2.0f, nan);
    __m128 mask = _mm_set_ps(-1.0f, -1.0f, 0.0f, 0.0f);

    __m128 c = _mm_blendv_ps(a, b, mask);

    /* mask high bits: select from b, else from a */
    EXPECT_TRUE(extract_ps(c, 0) == 1.0f);
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_TRUE(extract_ps(c, 3) == 2.0f);

    return TEST_SUCCESS;
}

TEST_CASE(blendv_pd_nan_selector)
{
    /* Double precision blend with NaN selector */
    double pos_nan = make_qnan_d();
    double neg_nan = make_neg_qnan_d();

    __m128d a = _mm_set1_pd(1.0);
    __m128d b = _mm_set1_pd(2.0);
    __m128d mask = _mm_set_pd(neg_nan, pos_nan);

    __m128d c = _mm_blendv_pd(a, b, mask);

    /* Positive NaN (sign=0) selects from a,
     * negative NaN (sign=1) selects from b
     */
    EXPECT_TRUE(extract_pd(c, 0) == 1.0); /* pos_nan -> a */
    EXPECT_TRUE(extract_pd(c, 1) == 2.0); /* neg_nan -> b */

    return TEST_SUCCESS;
}

TEST_CASE(blendv_pd_nan_operands)
{
    /* Blending double NaN values */
    double nan = make_qnan_d();

    __m128d a = _mm_set_pd(nan, 1.0);
    __m128d b = _mm_set_pd(2.0, nan);
    __m128d mask = _mm_set_pd(-1.0, 0.0);

    __m128d c = _mm_blendv_pd(a, b, mask);

    /* mask[0] sign=0 -> select a[0]=1.0
     * mask[1] sign=1 -> select b[1]=2.0
     */
    EXPECT_TRUE(extract_pd(c, 0) == 1.0);
    EXPECT_TRUE(extract_pd(c, 1) == 2.0);

    return TEST_SUCCESS;
}

/* Unconditional Min/Max NaN Tests
 * These tests verify NaN behavior unconditionally (regardless of
 * SSE2NEON_PRECISE_MINMAX)
 * to document the actual platform behavior
 */

TEST_CASE(min_ps_nan_both_nan)
{
    /* When both operands are NaN, result should be NaN */
    float nan1 = make_qnan_payload_f(0x123);
    float nan2 = make_qnan_payload_f(0x456);

    __m128 a = _mm_set1_ps(nan1);
    __m128 b = _mm_set1_ps(nan2);
    __m128 c = _mm_min_ps(a, b);

    /* Result must be NaN regardless of platform */
    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(max_ps_nan_both_nan)
{
    /* When both operands are NaN, result should be NaN */
    float nan1 = make_qnan_payload_f(0x123);
    float nan2 = make_qnan_payload_f(0x456);

    __m128 a = _mm_set1_ps(nan1);
    __m128 b = _mm_set1_ps(nan2);
    __m128 c = _mm_max_ps(a, b);

    EXPECT_NAN_F(extract_ps(c, 0));
    EXPECT_NAN_F(extract_ps(c, 1));
    EXPECT_NAN_F(extract_ps(c, 2));
    EXPECT_NAN_F(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(min_pd_nan_both_nan)
{
    double nan1 = make_qnan_payload_d(0x123);
    double nan2 = make_qnan_payload_d(0x456);

    __m128d a = _mm_set1_pd(nan1);
    __m128d b = _mm_set1_pd(nan2);
    __m128d c = _mm_min_pd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));
    EXPECT_NAN_D(extract_pd(c, 1));

    return TEST_SUCCESS;
}

TEST_CASE(max_pd_nan_both_nan)
{
    double nan1 = make_qnan_payload_d(0x123);
    double nan2 = make_qnan_payload_d(0x456);

    __m128d a = _mm_set1_pd(nan1);
    __m128d b = _mm_set1_pd(nan2);
    __m128d c = _mm_max_pd(a, b);

    EXPECT_NAN_D(extract_pd(c, 0));
    EXPECT_NAN_D(extract_pd(c, 1));

    return TEST_SUCCESS;
}

/* Scalar Rounding with NaN Tests (SSE4.1) */

TEST_CASE(round_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ps(1.0f, 2.0f, 3.0f, nan);

    __m128 r = _mm_round_ss(a, a, _MM_FROUND_TO_NEAREST_INT);

    /* Only lane 0 is rounded, others preserved */
    EXPECT_NAN_F(extract_ps(r, 0));

    return TEST_SUCCESS;
}

TEST_CASE(floor_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ps(1.0f, 2.0f, 3.0f, nan);

    __m128 r = _mm_floor_ss(a, a);
    EXPECT_NAN_F(extract_ps(r, 0));

    return TEST_SUCCESS;
}

TEST_CASE(ceil_ss_nan)
{
    float nan = make_qnan_f();
    __m128 a = _mm_set_ps(1.0f, 2.0f, 3.0f, nan);

    __m128 r = _mm_ceil_ss(a, a);
    EXPECT_NAN_F(extract_ps(r, 0));

    return TEST_SUCCESS;
}

TEST_CASE(round_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_pd(1.0, nan);

    __m128d r = _mm_round_sd(a, a, _MM_FROUND_TO_NEAREST_INT);
    EXPECT_NAN_D(extract_pd(r, 0));

    return TEST_SUCCESS;
}

TEST_CASE(floor_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_pd(1.0, nan);

    __m128d r = _mm_floor_sd(a, a);
    EXPECT_NAN_D(extract_pd(r, 0));

    return TEST_SUCCESS;
}

TEST_CASE(ceil_sd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set_pd(1.0, nan);

    __m128d r = _mm_ceil_sd(a, a);
    EXPECT_NAN_D(extract_pd(r, 0));

    return TEST_SUCCESS;
}

/* Double Precision Floor/Ceil with NaN */

TEST_CASE(floor_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d r = _mm_floor_pd(a);

    EXPECT_NAN_D(extract_pd(r, 0));
    EXPECT_NAN_D(extract_pd(r, 1));

    return TEST_SUCCESS;
}

TEST_CASE(ceil_pd_nan)
{
    double nan = make_qnan_d();
    __m128d a = _mm_set1_pd(nan);
    __m128d r = _mm_ceil_pd(a);

    EXPECT_NAN_D(extract_pd(r, 0));
    EXPECT_NAN_D(extract_pd(r, 1));

    return TEST_SUCCESS;
}

/* Print Configuration and Run Tests */

static void print_nan_test_config(void)
{
    printf("NaN Test Configuration:\n");
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
    printf("  Platform: ARM (using sse2neon)\n");
#else
    printf("  Platform: x86 (native SSE)\n");
#endif
#ifdef SSE2NEON_PRECISE_MINMAX
    printf("  SSE2NEON_PRECISE_MINMAX = %d\n", SSE2NEON_PRECISE_MINMAX);
#else
    printf("  SSE2NEON_PRECISE_MINMAX = 0 (default)\n");
#endif
    printf("\n");
}

int main(void)
{
    printf("===========================================\n");
    printf("NaN Propagation and Handling Tests\n");
    printf("===========================================\n\n");

    print_nan_test_config();

    printf("--- NaN Propagation: Packed Single Precision ---\n");
    RUN_TEST(add_ps_nan_first_operand);
    RUN_TEST(add_ps_nan_second_operand);
    RUN_TEST(add_ps_nan_both_operands);
    RUN_TEST(sub_ps_nan_propagation);
    RUN_TEST(mul_ps_nan_propagation);
    RUN_TEST(mul_ps_nan_times_zero);
    RUN_TEST(div_ps_nan_propagation);
    RUN_TEST(div_ps_nan_by_zero);

    printf("\n--- NaN Propagation: Scalar Single Precision ---\n");
    RUN_TEST(add_ss_nan_propagation);
    RUN_TEST(sub_ss_nan_propagation);
    RUN_TEST(mul_ss_nan_propagation);
    RUN_TEST(div_ss_nan_propagation);

    printf("\n--- NaN Propagation: Special Functions ---\n");
    RUN_TEST(sqrt_ps_nan_input);
    RUN_TEST(sqrt_ss_nan_input);
    RUN_TEST(rsqrt_ps_nan_input);
    RUN_TEST(rsqrt_ss_nan_input);
    RUN_TEST(rcp_ps_nan_input);
    RUN_TEST(rcp_ss_nan_input);

    printf("\n--- NaN Propagation: Double Precision ---\n");
    RUN_TEST(add_pd_nan_propagation);
    RUN_TEST(sub_pd_nan_propagation);
    RUN_TEST(mul_pd_nan_propagation);
    RUN_TEST(div_pd_nan_propagation);
    RUN_TEST(sqrt_pd_nan_input);

    printf("\n--- NaN Propagation: Scalar Double Precision ---\n");
    RUN_TEST(add_sd_nan_propagation);
    RUN_TEST(sub_sd_nan_propagation);
    RUN_TEST(mul_sd_nan_propagation);
    RUN_TEST(div_sd_nan_propagation);
    RUN_TEST(sqrt_sd_nan_input);

    printf("\n--- NaN Comparison: Single Precision ---\n");
    RUN_TEST(cmpeq_ps_nan);
    RUN_TEST(cmpneq_ps_nan);
    RUN_TEST(cmplt_ps_nan);
    RUN_TEST(cmple_ps_nan);
    RUN_TEST(cmpgt_ps_nan);
    RUN_TEST(cmpge_ps_nan);
    RUN_TEST(cmpord_ps_nan);
    RUN_TEST(cmpunord_ps_nan);
    RUN_TEST(cmpeq_ss_nan);
    RUN_TEST(cmpneq_ss_nan);

    printf("\n--- NaN Comparison: Double Precision ---\n");
    RUN_TEST(cmpeq_pd_nan);
    RUN_TEST(cmpneq_pd_nan);
    RUN_TEST(cmplt_pd_nan);
    RUN_TEST(cmple_pd_nan);
    RUN_TEST(cmpgt_pd_nan);
    RUN_TEST(cmpge_pd_nan);
    RUN_TEST(cmpord_pd_nan);
    RUN_TEST(cmpunord_pd_nan);

    printf("\n--- Min/Max NaN Behavior ---\n");
    RUN_TEST(min_ps_nan_first);
    RUN_TEST(min_ps_nan_second);
    RUN_TEST(max_ps_nan_first);
    RUN_TEST(max_ps_nan_second);
    RUN_TEST(min_pd_nan_behavior);
    RUN_TEST(max_pd_nan_behavior);

    printf("\n--- NaN Payload Preservation ---\n");
    RUN_TEST(nan_payload_through_copy);
    RUN_TEST(nan_payload_through_arithmetic);
    RUN_TEST(nan_sign_preservation);
    RUN_TEST(nan_payload_double);

    printf("\n--- Bitwise Operations NaN Payload Tests ---\n");
    RUN_TEST(and_ps_nan_payload);
    RUN_TEST(or_ps_nan_payload);
    RUN_TEST(xor_ps_nan_with_zero);
    RUN_TEST(andnot_ps_nan);

    printf("\n--- Shuffle NaN Payload Tests ---\n");
    RUN_TEST(shuffle_ps_nan_payload);
    RUN_TEST(unpacklo_ps_nan_payload);
    RUN_TEST(movehl_ps_nan_payload);

    printf("\n--- Signaling NaN to Quiet NaN Conversion ---\n");
    RUN_TEST(snan_to_qnan_add);
    RUN_TEST(snan_to_qnan_mul);
    RUN_TEST(snan_to_qnan_sub);
    RUN_TEST(snan_to_qnan_div);
    RUN_TEST(snan_to_qnan_sqrt);
    RUN_TEST(snan_to_qnan_rsqrt);
    RUN_TEST(snan_to_qnan_rcp);
    RUN_TEST(snan_comparison);
    RUN_TEST(snan_double_to_qnan);

    printf("\n--- NaN Generation Tests ---\n");
    RUN_TEST(nan_generation_zero_div_zero);
    RUN_TEST(nan_generation_inf_minus_inf);
    RUN_TEST(nan_generation_inf_times_zero);
    RUN_TEST(nan_generation_sqrt_negative);
    RUN_TEST(nan_generation_inf_div_inf);
    RUN_TEST(nan_generation_double);

    printf("\n--- NaN to Integer Conversion Tests ---\n");
    RUN_TEST(cvtps_epi32_nan);
    RUN_TEST(cvttps_epi32_nan);
    RUN_TEST(cvtss_si32_nan);
    RUN_TEST(cvttss_si32_nan);
    RUN_TEST(cvtpd_epi32_nan);
    RUN_TEST(cvttpd_epi32_nan);
    RUN_TEST(cvtsd_si32_nan);

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
    defined(_M_ARM64)
    printf("\n--- NaN to 64-bit Integer Conversion Tests ---\n");
    RUN_TEST(cvtss_si64_nan);
    RUN_TEST(cvttss_si64_nan);
    RUN_TEST(cvtsd_si64_nan);
    RUN_TEST(cvttsd_si64_nan);
#endif

    printf("\n--- Scalar Comparison (COMI/UCOMI) NaN Tests ---\n");
    RUN_TEST(comieq_ss_nan);
    RUN_TEST(comineq_ss_nan);
    RUN_TEST(comilt_ss_nan);
    RUN_TEST(comile_ss_nan);
    RUN_TEST(comigt_ss_nan);
    RUN_TEST(comige_ss_nan);
    RUN_TEST(ucomieq_ss_nan);
    RUN_TEST(ucomineq_ss_nan);
    RUN_TEST(comieq_sd_nan);
    RUN_TEST(comineq_sd_nan);
    RUN_TEST(ucomieq_sd_nan);
    RUN_TEST(comilt_sd_nan);
    RUN_TEST(comile_sd_nan);
    RUN_TEST(comigt_sd_nan);
    RUN_TEST(comige_sd_nan);
    RUN_TEST(ucomineq_sd_nan);
    RUN_TEST(ucomilt_sd_nan);
    RUN_TEST(ucomile_sd_nan);
    RUN_TEST(ucomigt_sd_nan);
    RUN_TEST(ucomige_sd_nan);

    printf("\n--- Rounding with NaN Tests ---\n");
    RUN_TEST(round_ps_nan);
    RUN_TEST(round_pd_nan);
    RUN_TEST(floor_ps_nan);
    RUN_TEST(ceil_ps_nan);
    RUN_TEST(floor_pd_nan);
    RUN_TEST(ceil_pd_nan);

    printf("\n--- Scalar Rounding with NaN Tests ---\n");
    RUN_TEST(round_ss_nan);
    RUN_TEST(floor_ss_nan);
    RUN_TEST(ceil_ss_nan);
    RUN_TEST(round_sd_nan);
    RUN_TEST(floor_sd_nan);
    RUN_TEST(ceil_sd_nan);

    printf("\n--- Horizontal Operations with NaN ---\n");
    RUN_TEST(hadd_ps_nan_propagation);
    RUN_TEST(hsub_ps_nan_propagation);

    printf("\n--- Blend/Select with NaN ---\n");
    RUN_TEST(blendv_ps_nan_selector);
    RUN_TEST(blendv_ps_nan_operands);
    RUN_TEST(blendv_pd_nan_selector);
    RUN_TEST(blendv_pd_nan_operands);

    printf("\n--- Unconditional Min/Max NaN Tests ---\n");
    RUN_TEST(min_ps_nan_both_nan);
    RUN_TEST(max_ps_nan_both_nan);
    RUN_TEST(min_pd_nan_both_nan);
    RUN_TEST(max_pd_nan_both_nan);

    printf("\n===========================================\n");
    printf("Results: %d passed, %d failed, %d skipped\n", g_pass_count,
           g_fail_count, g_skip_count);
    printf("===========================================\n");

    return g_fail_count > 0 ? 1 : 0;
}
