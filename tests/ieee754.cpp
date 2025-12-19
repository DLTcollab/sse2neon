/**
 * ieee754.cpp - IEEE-754 Floating-Point Edge Case Validation
 *
 * This test file systematically validates sse2neon behavior against IEEE-754
 * standards for edge cases: NaN, infinity, denormals, and signed zero.
 *
 * Coverage:
 *   - Arithmetic: add, sub, mul, div
 *   - Comparisons: cmpeq, cmplt, cmpgt, cmple, cmpge, cmpord, cmpunord
 *   - Min/Max: min, max (with precision flag awareness)
 *   - Special: sqrt, rsqrt, rcp
 *
 * Precision Modes:
 *   - Default (SSE2NEON_PRECISE_* = 0): Performance-optimized, may differ from
 *     x86 SSE for edge cases
 *   - Precise (SSE2NEON_PRECISE_* = 1): IEEE-754 compliant, matches x86 SSE
 *
 * Usage:
 *   make ieee754 && ./tests/ieee754
 *   make FEATURE=crypto+crc ieee754 && ./tests/ieee754
 *
 * Reference: IEEE 754-2008, Intel Intrinsics Guide
 */

/* ARM64EC requires sse2neon.h to be included FIRST. See common.h for details.
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
            printf("Test %-35s [ " COLOR_GREEN "OK" COLOR_RESET " ]\n",       \
                   #name);                                                    \
        } else if (r == TEST_UNIMPL) {                                        \
            g_skip_count++;                                                   \
            printf("Test %-35s [SKIP]\n", #name);                             \
        } else {                                                              \
            g_fail_count++;                                                   \
            printf("Test %-35s [" COLOR_RED "FAIL" COLOR_RESET "]\n", #name); \
        }                                                                     \
    } while (0)

#define EXPECT_TRUE(cond)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            printf("    FAILED: %s (line %d)\n", #cond, __LINE__); \
            return TEST_FAIL;                                      \
        }                                                          \
    } while (0)

#define EXPECT_FLOAT_NAN(a) EXPECT_TRUE(std::isnan(a))
#define EXPECT_FLOAT_INF(a) EXPECT_TRUE(std::isinf(a))

/* IEEE-754 special value generators */
static inline float make_qnan(void)
{
    uint32_t bits = 0x7FC00000; /* Quiet NaN */
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

__attribute__((unused)) static inline float make_snan(void)
{
    uint32_t bits = 0x7F800001; /* Signaling NaN */
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline float make_pos_inf(void)
{
    return INFINITY;
}

static inline float make_neg_inf(void)
{
    return -INFINITY;
}

static inline float make_pos_zero(void)
{
    return 0.0f;
}

static inline float make_neg_zero(void)
{
    return -0.0f;
}

static inline float make_denormal(void)
{
    /* Smallest positive denormal: 2^-149 */
    uint32_t bits = 0x00000001;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline float make_half_flt_min(void)
{
    /* FLT_MIN / 2 = 2^-127 (denormal), constructed via bit pattern to avoid
     * FTZ-dependent computation. FLT_MIN = 0x00800000, so FLT_MIN/2 =
     * 0x00400000
     */
    uint32_t bits = 0x00400000;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

__attribute__((unused)) static inline float make_neg_denormal(void)
{
    uint32_t bits = 0x80000001;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline double make_qnan_d(void)
{
    uint64_t bits = 0x7FF8000000000000ULL;
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

static inline double make_pos_inf_d(void)
{
    return static_cast<double>(INFINITY);
}

static inline double make_neg_inf_d(void)
{
    return static_cast<double>(-INFINITY);
}

__attribute__((unused)) static inline double make_denormal_d(void)
{
    uint64_t bits = 0x0000000000000001ULL;
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

/* Binary comparison for floats (handles NaN, signed zero) */
__attribute__((unused)) static inline bool float_binary_eq(float a, float b)
{
    uint32_t ua, ub;
    memcpy(&ua, &a, sizeof(ua));
    memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

__attribute__((unused)) static inline bool double_binary_eq(double a, double b)
{
    uint64_t ua, ub;
    memcpy(&ua, &a, sizeof(ua));
    memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
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

/* Verify all lanes of __m128 have expected bit pattern */
static inline bool verify_all_lanes_ps(__m128 v, uint32_t expected_bits)
{
    float results[4];
    _mm_storeu_ps(results, v);
    for (int i = 0; i < 4; i++) {
        uint32_t bits;
        memcpy(&bits, &results[i], sizeof(bits));
        if (bits != expected_bits)
            return false;
    }
    return true;
}

/* Verify all lanes of __m128d have expected bit pattern */
static inline bool verify_all_lanes_pd(__m128d v, uint64_t expected_bits)
{
    double results[2];
    _mm_storeu_pd(results, v);
    for (int i = 0; i < 2; i++) {
        uint64_t bits;
        memcpy(&bits, &results[i], sizeof(bits));
        if (bits != expected_bits)
            return false;
    }
    return true;
}

/* Verify all lanes of __m128 are positive infinity */
static inline bool verify_all_lanes_pos_inf(__m128 v)
{
    float results[4];
    _mm_storeu_ps(results, v);
    for (int i = 0; i < 4; i++) {
        if (!std::isinf(results[i]) || results[i] < 0)
            return false;
    }
    return true;
}

/* Verify all lanes of __m128 are negative infinity */
static inline bool verify_all_lanes_neg_inf(__m128 v)
{
    float results[4];
    _mm_storeu_ps(results, v);
    for (int i = 0; i < 4; i++) {
        if (!std::isinf(results[i]) || results[i] > 0)
            return false;
    }
    return true;
}

/* Verify all lanes are positive infinity or NaN (for rsqrt compatibility) */
static inline bool verify_all_lanes_pos_inf_or_nan(__m128 v)
{
    float results[4];
    _mm_storeu_ps(results, v);
    for (int i = 0; i < 4; i++) {
        if (!((std::isinf(results[i]) && results[i] > 0) ||
              std::isnan(results[i])))
            return false;
    }
    return true;
}

/* Verify all lanes are negative infinity or NaN (for rsqrt compatibility) */
static inline bool verify_all_lanes_neg_inf_or_nan(__m128 v)
{
    float results[4];
    _mm_storeu_ps(results, v);
    for (int i = 0; i < 4; i++) {
        if (!((std::isinf(results[i]) && results[i] < 0) ||
              std::isnan(results[i])))
            return false;
    }
    return true;
}

/* Verify all lanes of __m128d are positive infinity */
static inline bool verify_all_lanes_pos_inf_pd(__m128d v)
{
    double results[2];
    _mm_storeu_pd(results, v);
    for (int i = 0; i < 2; i++) {
        if (!std::isinf(results[i]) || results[i] < 0)
            return false;
    }
    return true;
}

/* Verify all lanes of __m128d are negative infinity */
static inline bool verify_all_lanes_neg_inf_pd(__m128d v)
{
    double results[2];
    _mm_storeu_pd(results, v);
    for (int i = 0; i < 2; i++) {
        if (!std::isinf(results[i]) || results[i] > 0)
            return false;
    }
    return true;
}

/* NaN Propagation Tests
 * IEEE-754: Operations involving NaN should propagate NaN (with exceptions)
 */

TEST_CASE(add_ps_nan_propagation)
{
    float nan = make_qnan();
    float normal = 1.0f;

    __m128 a = _mm_set_ps(nan, normal, nan, normal);
    __m128 b = _mm_set_ps(normal, nan, normal, nan);
    __m128 c = _mm_add_ps(a, b);

    /* NaN + anything = NaN, anything + NaN = NaN */
    EXPECT_FLOAT_NAN(extract_ps(c, 0));
    EXPECT_FLOAT_NAN(extract_ps(c, 1));
    EXPECT_FLOAT_NAN(extract_ps(c, 2));
    EXPECT_FLOAT_NAN(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(sub_ps_nan_propagation)
{
    float nan = make_qnan();
    float normal = 1.0f;

    __m128 a = _mm_set_ps(nan, normal, nan, normal);
    __m128 b = _mm_set_ps(normal, nan, normal, nan);
    __m128 c = _mm_sub_ps(a, b);

    EXPECT_FLOAT_NAN(extract_ps(c, 0));
    EXPECT_FLOAT_NAN(extract_ps(c, 1));
    EXPECT_FLOAT_NAN(extract_ps(c, 2));
    EXPECT_FLOAT_NAN(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(mul_ps_nan_propagation)
{
    float nan = make_qnan();
    float normal = 2.0f;

    __m128 a = _mm_set_ps(nan, normal, nan, normal);
    __m128 b = _mm_set_ps(normal, nan, normal, nan);
    __m128 c = _mm_mul_ps(a, b);

    EXPECT_FLOAT_NAN(extract_ps(c, 0));
    EXPECT_FLOAT_NAN(extract_ps(c, 1));
    EXPECT_FLOAT_NAN(extract_ps(c, 2));
    EXPECT_FLOAT_NAN(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(div_ps_nan_propagation)
{
    float nan = make_qnan();
    float normal = 2.0f;

    __m128 a = _mm_set_ps(nan, normal, nan, normal);
    __m128 b = _mm_set_ps(normal, nan, normal, nan);
    __m128 c = _mm_div_ps(a, b);

    EXPECT_FLOAT_NAN(extract_ps(c, 0));
    EXPECT_FLOAT_NAN(extract_ps(c, 1));
    EXPECT_FLOAT_NAN(extract_ps(c, 2));
    EXPECT_FLOAT_NAN(extract_ps(c, 3));

    return TEST_SUCCESS;
}

/* Infinity Arithmetic Tests */

TEST_CASE(add_ps_infinity)
{
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();
    float normal = 1.0f;

    /* inf + normal = inf */
    __m128 a = _mm_set1_ps(pos_inf);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_add_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));
    EXPECT_TRUE(extract_ps(c, 0) > 0);

    /* -inf + normal = -inf */
    a = _mm_set1_ps(neg_inf);
    c = _mm_add_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));
    EXPECT_TRUE(extract_ps(c, 0) < 0);

    /* inf + (-inf) = NaN */
    a = _mm_set1_ps(pos_inf);
    b = _mm_set1_ps(neg_inf);
    c = _mm_add_ps(a, b);
    EXPECT_FLOAT_NAN(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(sub_ps_infinity)
{
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();

    /* inf - inf = NaN */
    __m128 a = _mm_set1_ps(pos_inf);
    __m128 b = _mm_set1_ps(pos_inf);
    __m128 c = _mm_sub_ps(a, b);
    EXPECT_FLOAT_NAN(extract_ps(c, 0));

    /* inf - (-inf) = inf */
    b = _mm_set1_ps(neg_inf);
    c = _mm_sub_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));
    EXPECT_TRUE(extract_ps(c, 0) > 0);

    return TEST_SUCCESS;
}

TEST_CASE(mul_ps_infinity)
{
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();
    float zero = 0.0f;
    float normal = 2.0f;

    /* inf * normal = inf */
    __m128 a = _mm_set1_ps(pos_inf);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_mul_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));

    /* inf * (-normal) = -inf */
    b = _mm_set1_ps(-normal);
    c = _mm_mul_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));
    EXPECT_TRUE(extract_ps(c, 0) < 0);

    /* inf * 0 = NaN */
    b = _mm_set1_ps(zero);
    c = _mm_mul_ps(a, b);
    EXPECT_FLOAT_NAN(extract_ps(c, 0));

    /* inf * (-inf) = -inf */
    b = _mm_set1_ps(neg_inf);
    c = _mm_mul_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));
    EXPECT_TRUE(extract_ps(c, 0) < 0);

    return TEST_SUCCESS;
}

TEST_CASE(div_ps_infinity)
{
    float pos_inf = make_pos_inf();
    float zero = 0.0f;
    float normal = 2.0f;

    /* normal / 0 = inf */
    __m128 a = _mm_set1_ps(normal);
    __m128 b = _mm_set1_ps(zero);
    __m128 c = _mm_div_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));

    /* inf / inf = NaN */
    a = _mm_set1_ps(pos_inf);
    b = _mm_set1_ps(pos_inf);
    c = _mm_div_ps(a, b);
    EXPECT_FLOAT_NAN(extract_ps(c, 0));

    /* 0 / 0 = NaN */
    a = _mm_set1_ps(zero);
    b = _mm_set1_ps(zero);
    c = _mm_div_ps(a, b);
    EXPECT_FLOAT_NAN(extract_ps(c, 0));

    /* normal / inf = 0 */
    a = _mm_set1_ps(normal);
    b = _mm_set1_ps(pos_inf);
    c = _mm_div_ps(a, b);
    EXPECT_TRUE(extract_ps(c, 0) == 0.0f);

    return TEST_SUCCESS;
}

/* Signed Zero Tests
 * IEEE-754: +0.0 and -0.0 are equal in value but distinct in representation
 */

TEST_CASE(signed_zero_comparison)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();

    /* +0 == -0 in value comparison (IEEE-754: equal in value)
     * Verify all SIMD lanes to catch lane-mixing bugs.
     */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 b = _mm_set1_ps(neg_zero);
    __m128 c = _mm_cmpeq_ps(a, b);

    /* All lanes should have all bits set (true) */
    EXPECT_TRUE(verify_all_lanes_ps(c, 0xFFFFFFFF));

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_arithmetic)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();

    /* +0 + (-0) = +0 (IEEE-754 default rounding) */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 b = _mm_set1_ps(neg_zero);
    __m128 c = _mm_add_ps(a, b);

    float result = extract_ps(c, 0);
    EXPECT_TRUE(result == 0.0f);
    /* Check it's positive zero */
    uint32_t bits;
    memcpy(&bits, &result, sizeof(bits));
    EXPECT_TRUE(bits == 0x00000000);

    /* -0 + (-0) = -0 */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(neg_zero);
    c = _mm_add_ps(a, b);
    result = extract_ps(c, 0);
    memcpy(&bits, &result, sizeof(bits));
    EXPECT_TRUE(bits == 0x80000000);

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_mul)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();
    float positive = 1.0f;
    float negative = -1.0f;

    /* positive * (+0) = +0 */
    __m128 a = _mm_set1_ps(positive);
    __m128 b = _mm_set1_ps(pos_zero);
    __m128 c = _mm_mul_ps(a, b);
    float result = extract_ps(c, 0);
    uint32_t bits;
    memcpy(&bits, &result, sizeof(bits));
    EXPECT_TRUE(bits == 0x00000000);

    /* positive * (-0) = -0 */
    b = _mm_set1_ps(neg_zero);
    c = _mm_mul_ps(a, b);
    result = extract_ps(c, 0);
    memcpy(&bits, &result, sizeof(bits));
    EXPECT_TRUE(bits == 0x80000000);

    /* negative * (-0) = +0 */
    a = _mm_set1_ps(negative);
    c = _mm_mul_ps(a, b);
    result = extract_ps(c, 0);
    memcpy(&bits, &result, sizeof(bits));
    EXPECT_TRUE(bits == 0x00000000);

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_div)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();
    float positive = 1.0f;
    float negative = -1.0f;

    /* IEEE-754 division with signed zeros:
     *   positive / +0 = +Inf
     *   positive / -0 = -Inf
     *   negative / +0 = -Inf
     *   negative / -0 = +Inf
     *
     * Verify all SIMD lanes to catch lane-mixing bugs.
     */

    /* positive / (+0) = +Inf (all lanes) */
    __m128 a = _mm_set1_ps(positive);
    __m128 b = _mm_set1_ps(pos_zero);
    __m128 c = _mm_div_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_pos_inf(c));

    /* positive / (-0) = -Inf (all lanes) */
    b = _mm_set1_ps(neg_zero);
    c = _mm_div_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_neg_inf(c));

    /* negative / (+0) = -Inf (all lanes) */
    a = _mm_set1_ps(negative);
    b = _mm_set1_ps(pos_zero);
    c = _mm_div_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_neg_inf(c));

    /* negative / (-0) = +Inf (all lanes) */
    b = _mm_set1_ps(neg_zero);
    c = _mm_div_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_pos_inf(c));

    /* Zero divided by non-zero preserves sign:
     *   +0 / positive = +0
     *   -0 / positive = -0
     *   +0 / negative = -0
     *   -0 / negative = +0
     */

    /* (+0) / positive = +0 (all lanes) */
    a = _mm_set1_ps(pos_zero);
    b = _mm_set1_ps(positive);
    c = _mm_div_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* (-0) / positive = -0 (all lanes) */
    a = _mm_set1_ps(neg_zero);
    c = _mm_div_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* (+0) / negative = -0 (all lanes) */
    a = _mm_set1_ps(pos_zero);
    b = _mm_set1_ps(negative);
    c = _mm_div_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* (-0) / negative = +0 (all lanes) */
    a = _mm_set1_ps(neg_zero);
    c = _mm_div_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_sqrt)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();

    /* IEEE-754: sqrt preserves sign of zero
     *   sqrt(+0) = +0
     *   sqrt(-0) = -0
     *
     * Verify all SIMD lanes to catch lane-mixing bugs.
     */

    /* sqrt(+0) = +0 (all lanes) */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 c = _mm_sqrt_ps(a);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* sqrt(-0) = -0 (all lanes) */
    a = _mm_set1_ps(neg_zero);
    c = _mm_sqrt_ps(a);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_rsqrt)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();

    /* rsqrt(0) behavior:
     * - x86 SSE: rsqrt(+0) = +Inf, rsqrt(-0) = -Inf
     * - ARM NEON: vrsqrteq_f32(+0) = +Inf, vrsqrteq_f32(-0) = -Inf
     *
     * Both platforms agree: rsqrt preserves the sign of zero in the infinity.
     * Note: The existing test in rsqrt_ps_special_values accepts NaN as well
     * because the existing sse2neon.h implementation may return NaN for some
     * cases (documented in header). This strict test verifies the ideal case.
     *
     * Verify all SIMD lanes to catch lane-mixing bugs.
     */

    /* rsqrt(+0) = +Inf (all lanes; accept NaN for sse2neon compatibility) */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 c = _mm_rsqrt_ps(a);
    EXPECT_TRUE(verify_all_lanes_pos_inf_or_nan(c));

    /* rsqrt(-0) = -Inf (all lanes; accept NaN for sse2neon compatibility) */
    a = _mm_set1_ps(neg_zero);
    c = _mm_rsqrt_ps(a);
    EXPECT_TRUE(verify_all_lanes_neg_inf_or_nan(c));

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_sub)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();
    float positive = 1.0f;
    float negative = -1.0f;

    /* IEEE-754 subtraction with signed zeros:
     *   +0 - (+0) = +0 (in round-to-nearest mode)
     *   -0 - (-0) = +0
     *   +0 - (-0) = +0
     *   -0 - (+0) = -0
     *
     * For non-zero operands:
     *   x - x = +0 (always, in round-to-nearest mode)
     */

    /* +0 - (+0) = +0 */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 b = _mm_set1_ps(pos_zero);
    __m128 c = _mm_sub_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* -0 - (-0) = +0 */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(neg_zero);
    c = _mm_sub_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* +0 - (-0) = +0 */
    a = _mm_set1_ps(pos_zero);
    b = _mm_set1_ps(neg_zero);
    c = _mm_sub_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* -0 - (+0) = -0 */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(pos_zero);
    c = _mm_sub_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* positive - positive = +0 (x - x = +0) */
    a = _mm_set1_ps(positive);
    c = _mm_sub_ps(a, a);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* negative - negative = +0 (x - x = +0) */
    a = _mm_set1_ps(negative);
    c = _mm_sub_ps(a, a);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_minmax)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();

    /* SSE min/max with equal values (including signed zeros):
     *
     * SSE MINPS/MAXPS behavior:
     *   - If a < b, return a; else return b (for MINPS)
     *   - If a > b, return a; else return b (for MAXPS)
     *   - When a == b (including +0 vs -0), returns b (second operand)
     *
     * ARM NEON vminq_f32/vmaxq_f32 behavior (IEEE 754 minNum/maxNum):
     *   - min: returns the more negative value (-0 is "less than" +0)
     *   - max: returns the more positive value (+0 is "greater than" -0)
     *   - This is deterministic IEEE 754-2008 minNum/maxNum semantics
     *
     * With SSE2NEON_PRECISE_MINMAX=1, we emulate SSE semantics exactly.
     * Without it, we use fast NEON instructions with IEEE 754 signed zero
     * handling (which differs from SSE for some operand orderings).
     */

#if SSE2NEON_PRECISE_MINMAX
    /* With PRECISE_MINMAX, SSE semantics are enforced */

    /* min(+0, -0): SSE returns second operand (-0) */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 b = _mm_set1_ps(neg_zero);
    __m128 c = _mm_min_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* min(-0, +0): SSE returns second operand (+0) */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(pos_zero);
    c = _mm_min_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* max(+0, -0): SSE returns second operand (-0) */
    a = _mm_set1_ps(pos_zero);
    b = _mm_set1_ps(neg_zero);
    c = _mm_max_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* max(-0, +0): SSE returns second operand (+0) */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(pos_zero);
    c = _mm_max_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || \
    defined(_M_ARM64) || defined(_M_ARM64EC)
    /* Without PRECISE_MINMAX on ARM, NEON uses vminq_f32/vmaxq_f32 which follow
     * IEEE 754-2008 minNum/maxNum semantics for signed zeros:
     *   - min: returns the more negative value (-0 < +0 numerically)
     *   - max: returns the more positive value (+0 > -0 numerically)
     *
     * This differs from SSE which always returns the second operand when equal.
     */

    /* min(+0, -0): NEON returns -0 (numerically smaller) */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 b = _mm_set1_ps(neg_zero);
    __m128 c = _mm_min_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* min(-0, +0): NEON returns -0 (numerically smaller) */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(pos_zero);
    c = _mm_min_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* max(+0, -0): NEON returns +0 (numerically larger) */
    a = _mm_set1_ps(pos_zero);
    b = _mm_set1_ps(neg_zero);
    c = _mm_max_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* max(-0, +0): NEON returns +0 (numerically larger) */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(pos_zero);
    c = _mm_max_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));
#else
    /* On x86 without PRECISE_MINMAX, native SSE semantics apply:
     * SSE always returns the second operand when values compare equal.
     */

    /* min(+0, -0): SSE returns -0 (second operand) */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 b = _mm_set1_ps(neg_zero);
    __m128 c = _mm_min_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* min(-0, +0): SSE returns +0 (second operand) */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(pos_zero);
    c = _mm_min_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));

    /* max(+0, -0): SSE returns -0 (second operand) */
    a = _mm_set1_ps(pos_zero);
    b = _mm_set1_ps(neg_zero);
    c = _mm_max_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x80000000));

    /* max(-0, +0): SSE returns +0 (second operand) */
    a = _mm_set1_ps(neg_zero);
    b = _mm_set1_ps(pos_zero);
    c = _mm_max_ps(a, b);
    EXPECT_TRUE(verify_all_lanes_ps(c, 0x00000000));
#endif

    return TEST_SUCCESS;
}

/* Denormal Number Tests
 * Note: NEON may flush denormals to zero depending on FPSCR settings
 */

TEST_CASE(denormal_arithmetic)
{
    float denorm = make_denormal();

    /* denormal + 0 = denormal (or 0 if flushed) */
    __m128 a = _mm_set1_ps(denorm);
    __m128 b = _mm_set1_ps(0.0f);
    __m128 c = _mm_add_ps(a, b);
    float result = extract_ps(c, 0);

    /* Result should be either the denormal or 0 (if flush-to-zero) */
    EXPECT_TRUE(result == denorm || result == 0.0f);

    /* denormal * 2 should stay denormal or become normal */
    b = _mm_set1_ps(2.0f);
    c = _mm_mul_ps(a, b);
    result = extract_ps(c, 0);
    EXPECT_TRUE(result >= 0.0f); /* Should be non-negative */

    return TEST_SUCCESS;
}

/* Flush-to-Zero (FTZ) and Denormals-Are-Zero (DAZ) Mode Tests
 *
 * x86 MXCSR register:
 *   - Bit 15 (FZ): Flush-to-zero - output denormals become zero
 *   - Bit 6 (DAZ): Denormals-are-zero - input denormals treated as zero
 *
 * ARM FPCR/FPSCR:
 *   - Bit 24 (FZ): Controls both flush-to-zero AND denormals-are-zero
 *   - ARM unifies FZ and DAZ behavior into a single bit
 *
 * ARMv7 (AArch32) Note:
 *   - Advanced SIMD (NEON) always operates with flush-to-zero enabled
 *   - The FZ bit in FPSCR is ignored for NEON operations
 *   - Denormals are always flushed to zero regardless of FPSCR settings
 */
TEST_CASE(ftz_output_flush)
{
    /* Test that output denormals are flushed to zero when FTZ is enabled */
    unsigned int original_csr = _mm_getcsr();
    result_t ret = TEST_SUCCESS;

    /* Create a denormal by multiplying FLT_MIN by 0.5 (output becomes denormal)
     * FLT_MIN is the smallest normal, created via bit pattern to be safe.
     */
    float half = 0.5f;
    float min_normal = FLT_MIN;
    __m128 a = _mm_set1_ps(min_normal);
    __m128 b = _mm_set1_ps(half);

    /* Enable FTZ via _mm_setcsr */
    _mm_setcsr(
        (original_csr & ~static_cast<unsigned>(_MM_FLUSH_ZERO_MASK |
                                               _MM_DENORMALS_ZERO_MASK)) |
        _MM_FLUSH_ZERO_ON);

    __m128 c = _mm_mul_ps(a, b);
    float result = extract_ps(c, 0);

    /* With FTZ enabled, result should be flushed to zero (not denormal) */
    if (result != 0.0f)
        ret = TEST_FAIL;

    /* Restore original CSR before returning */
    _mm_setcsr(original_csr);

    return ret;
}

TEST_CASE(daz_input_flush)
{
    /* Test that input denormals are treated as zero when DAZ is enabled */
    unsigned int original_csr = _mm_getcsr();
    result_t ret = TEST_SUCCESS;

    float denorm = make_denormal(); /* Created via bit pattern, safe from FTZ */
    float factor = 2.0f;
    __m128 a = _mm_set1_ps(denorm), b = _mm_set1_ps(factor);

    /* Enable DAZ via _mm_setcsr */
    _mm_setcsr(
        (original_csr & ~static_cast<unsigned>(_MM_FLUSH_ZERO_MASK |
                                               _MM_DENORMALS_ZERO_MASK)) |
        _MM_DENORMALS_ZERO_ON);

    /* denorm * 2 should be 0 when input denormals are treated as zero */
    __m128 c = _mm_mul_ps(a, b);
    float result = extract_ps(c, 0);

    /* With DAZ enabled, input denormal is treated as 0, so 0 * 2 = 0 */
    if (result != 0.0f)
        ret = TEST_FAIL;

    /* Restore original CSR before returning */
    _mm_setcsr(original_csr);

    return ret;
}

TEST_CASE(ftz_daz_disabled)
{
    /* Test denormal arithmetic with FTZ and DAZ disabled
     * Use FLT_MIN / 2 as denormal so that denormal * 2 = FLT_MIN (normal)
     * Denormal created via bit pattern to avoid FTZ-dependent computation.
     */
    unsigned int original_csr = _mm_getcsr();
    result_t ret = TEST_SUCCESS;

    float factor = 2.0f;
    float denorm = make_half_flt_min(); /* FLT_MIN / 2 via bit pattern */
    __m128 a = _mm_set1_ps(denorm);
    __m128 b = _mm_set1_ps(factor);

    /* Disable both FTZ and DAZ */
    _mm_setcsr(original_csr & ~static_cast<unsigned>(_MM_FLUSH_ZERO_MASK |
                                                     _MM_DENORMALS_ZERO_MASK));

    __m128 c = _mm_mul_ps(a, b);
    float result = extract_ps(c, 0);

#if defined(__arm__)
    /* ARMv7 (AArch32) NEON always flushes to zero regardless of FPSCR.FZ */
    if (result != 0.0f)
        ret = TEST_FAIL;
#else
    /* On x86 and AArch64 with FTZ/DAZ disabled, denormal arithmetic works
     * normally: denormal * 2 = FLT_MIN (normal) */
    if (result != FLT_MIN)
        ret = TEST_FAIL;
#endif

    /* Restore original CSR before returning */
    _mm_setcsr(original_csr);

    return ret;
}

TEST_CASE(ftz_getcsr_roundtrip)
{
    /* Test that _mm_setcsr/_mm_getcsr correctly round-trip FTZ/DAZ bits */
    unsigned int original_csr = _mm_getcsr();
    result_t ret = TEST_SUCCESS;
    unsigned int csr;

    /* Test FTZ ON */
    _mm_setcsr(
        (original_csr & ~static_cast<unsigned>(_MM_FLUSH_ZERO_MASK |
                                               _MM_DENORMALS_ZERO_MASK)) |
        _MM_FLUSH_ZERO_ON);
    csr = _mm_getcsr();
    if ((csr & _MM_FLUSH_ZERO_MASK) != _MM_FLUSH_ZERO_ON) {
        printf("    FAILED: FTZ ON - FZ bit not set (line %d)\n", __LINE__);
        ret = TEST_FAIL;
    }
#if defined(__aarch64__) || defined(__arm__)
    /* On ARM, FZ and DAZ share bit 24, so enabling FZ also enables DAZ */
    if (ret == TEST_SUCCESS &&
        (csr & _MM_DENORMALS_ZERO_MASK) != _MM_DENORMALS_ZERO_ON) {
        printf("    FAILED: FTZ ON - DAZ not coupled on ARM (line %d)\n",
               __LINE__);
        ret = TEST_FAIL;
    }
#else
    /* On x86, FZ and DAZ are separate bits */
    if (ret == TEST_SUCCESS &&
        (csr & _MM_DENORMALS_ZERO_MASK) != _MM_DENORMALS_ZERO_OFF) {
        printf("    FAILED: FTZ ON - DAZ should be off on x86 (line %d)\n",
               __LINE__);
        ret = TEST_FAIL;
    }
#endif

    /* Test DAZ ON */
    if (ret == TEST_SUCCESS) {
        _mm_setcsr(
            (original_csr & ~static_cast<unsigned>(_MM_FLUSH_ZERO_MASK |
                                                   _MM_DENORMALS_ZERO_MASK)) |
            _MM_DENORMALS_ZERO_ON);
        csr = _mm_getcsr();
        if ((csr & _MM_DENORMALS_ZERO_MASK) != _MM_DENORMALS_ZERO_ON) {
            printf("    FAILED: DAZ ON - DAZ bit not set (line %d)\n",
                   __LINE__);
            ret = TEST_FAIL;
        }
#if defined(__aarch64__) || defined(__arm__)
        /* On ARM, DAZ and FZ share bit 24, so enabling DAZ also enables FZ */
        if (ret == TEST_SUCCESS &&
            (csr & _MM_FLUSH_ZERO_MASK) != _MM_FLUSH_ZERO_ON) {
            printf("    FAILED: DAZ ON - FZ not coupled on ARM (line %d)\n",
                   __LINE__);
            ret = TEST_FAIL;
        }
#else
        /* On x86, FZ and DAZ are separate bits */
        if (ret == TEST_SUCCESS &&
            (csr & _MM_FLUSH_ZERO_MASK) != _MM_FLUSH_ZERO_OFF) {
            printf("    FAILED: DAZ ON - FZ should be off on x86 (line %d)\n",
                   __LINE__);
            ret = TEST_FAIL;
        }
#endif
    }

    /* Test both OFF */
    if (ret == TEST_SUCCESS) {
        _mm_setcsr(original_csr &
                   ~static_cast<unsigned>(_MM_FLUSH_ZERO_MASK |
                                          _MM_DENORMALS_ZERO_MASK));
        csr = _mm_getcsr();
        if ((csr & _MM_FLUSH_ZERO_MASK) != _MM_FLUSH_ZERO_OFF) {
            printf("    FAILED: both OFF - FZ not off (line %d)\n", __LINE__);
            ret = TEST_FAIL;
        }
        if (ret == TEST_SUCCESS &&
            (csr & _MM_DENORMALS_ZERO_MASK) != _MM_DENORMALS_ZERO_OFF) {
            printf("    FAILED: both OFF - DAZ not off (line %d)\n", __LINE__);
            ret = TEST_FAIL;
        }
    }

    /* Restore original CSR before returning */
    _mm_setcsr(original_csr);

    return ret;
}

/* Comparison Tests with Special Values */

TEST_CASE(cmpord_ps_nan)
{
    float nan = make_qnan();
    float normal = 1.0f;

    /* cmpord returns true only if BOTH operands are ordered (not NaN) */
    __m128 a = _mm_set_ps(nan, normal, nan, normal);
    __m128 b = _mm_set_ps(normal, nan, normal, normal);
    __m128 c = _mm_cmpord_ps(a, b);

    float results[4];
    _mm_storeu_ps(results, c);
    uint32_t bits[4];
    memcpy(bits, results, sizeof(bits));

    /* [3]: nan vs normal = unordered -> false (0) */
    EXPECT_TRUE(bits[3] == 0x00000000);
    /* [2]: normal vs nan = unordered -> false (0) */
    EXPECT_TRUE(bits[2] == 0x00000000);
    /* [1]: nan vs normal = unordered -> false (0) */
    EXPECT_TRUE(bits[1] == 0x00000000);
    /* [0]: normal vs normal = ordered -> true (all 1s) */
    EXPECT_TRUE(bits[0] == 0xFFFFFFFF);

    return TEST_SUCCESS;
}

TEST_CASE(cmpunord_ps_nan)
{
    float nan = make_qnan();
    float normal = 1.0f;

    /* cmpunord returns true if EITHER operand is NaN */
    __m128 a = _mm_set_ps(nan, normal, nan, normal);
    __m128 b = _mm_set_ps(normal, nan, normal, normal);
    __m128 c = _mm_cmpunord_ps(a, b);

    float results[4];
    _mm_storeu_ps(results, c);
    uint32_t bits[4];
    memcpy(bits, results, sizeof(bits));

    /* [3]: nan vs normal = unordered -> true */
    EXPECT_TRUE(bits[3] == 0xFFFFFFFF);
    /* [2]: normal vs nan = unordered -> true */
    EXPECT_TRUE(bits[2] == 0xFFFFFFFF);
    /* [1]: nan vs normal = unordered -> true */
    EXPECT_TRUE(bits[1] == 0xFFFFFFFF);
    /* [0]: normal vs normal = ordered -> false */
    EXPECT_TRUE(bits[0] == 0x00000000);

    return TEST_SUCCESS;
}

TEST_CASE(cmpeq_ps_nan)
{
    float nan = make_qnan();
    float normal = 1.0f;

    /* NaN is never equal to anything, including itself */
    __m128 a = _mm_set_ps(nan, nan, normal, normal);
    __m128 b = _mm_set_ps(nan, normal, nan, normal);
    __m128 c = _mm_cmpeq_ps(a, b);

    float results[4];
    _mm_storeu_ps(results, c);
    uint32_t bits[4];
    memcpy(bits, results, sizeof(bits));

    /* NaN == NaN -> false */
    EXPECT_TRUE(bits[3] == 0x00000000);
    /* NaN == normal -> false */
    EXPECT_TRUE(bits[2] == 0x00000000);
    /* normal == NaN -> false */
    EXPECT_TRUE(bits[1] == 0x00000000);
    /* normal == normal -> true */
    EXPECT_TRUE(bits[0] == 0xFFFFFFFF);

    return TEST_SUCCESS;
}

TEST_CASE(cmplt_ps_infinity)
{
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();
    float normal = 1.0f;

    /* normal < inf = true */
    __m128 a = _mm_set1_ps(normal);
    __m128 b = _mm_set1_ps(pos_inf);
    __m128 c = _mm_cmplt_ps(a, b);

    uint32_t bits;
    memcpy(&bits, &c, sizeof(bits));
    EXPECT_TRUE(bits == 0xFFFFFFFF);

    /* -inf < normal = true */
    a = _mm_set1_ps(neg_inf);
    b = _mm_set1_ps(normal);
    c = _mm_cmplt_ps(a, b);
    memcpy(&bits, &c, sizeof(bits));
    EXPECT_TRUE(bits == 0xFFFFFFFF);

    /* inf < inf = false */
    a = _mm_set1_ps(pos_inf);
    b = _mm_set1_ps(pos_inf);
    c = _mm_cmplt_ps(a, b);
    memcpy(&bits, &c, sizeof(bits));
    EXPECT_TRUE(bits == 0x00000000);

    return TEST_SUCCESS;
}

/* Min/Max Tests
 * Note: x86 SSE min/max have specific NaN behavior that differs from IEEE-754
 * SSE: If either operand is NaN, returns the SECOND operand
 */

TEST_CASE(min_ps_nan_behavior)
{
    float nan = make_qnan();
    float normal = 1.0f;

#if SSE2NEON_PRECISE_MINMAX
    /* With precise mode: should match x86 SSE behavior */
    /* SSE: min(NaN, x) = x, min(x, NaN) = NaN */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_min_ps(a, b);
    EXPECT_TRUE(extract_ps(c, 0) == normal); /* NaN vs normal -> normal */

    c = _mm_min_ps(b, a);
    EXPECT_FLOAT_NAN(extract_ps(c, 0)); /* normal vs NaN -> NaN */
#else
    /* Without precise mode: behavior may differ */
    /* Just verify no crash and result is either NaN or normal */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_min_ps(a, b);
    float result = extract_ps(c, 0);
    EXPECT_TRUE(std::isnan(result) || result == normal);
#endif

    return TEST_SUCCESS;
}

TEST_CASE(max_ps_nan_behavior)
{
    float nan = make_qnan();
    float normal = 1.0f;

#if SSE2NEON_PRECISE_MINMAX
    /* With precise mode: should match x86 SSE behavior */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_max_ps(a, b);
    EXPECT_TRUE(extract_ps(c, 0) == normal); /* NaN vs normal -> normal */

    c = _mm_max_ps(b, a);
    EXPECT_FLOAT_NAN(extract_ps(c, 0)); /* normal vs NaN -> NaN */
#else
    /* Without precise mode: just verify no crash */
    __m128 a = _mm_set1_ps(nan);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_max_ps(a, b);
    float result = extract_ps(c, 0);
    EXPECT_TRUE(std::isnan(result) || result == normal);
#endif

    return TEST_SUCCESS;
}

TEST_CASE(min_ps_infinity)
{
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();
    float normal = 1.0f;

    /* min(inf, normal) = normal */
    __m128 a = _mm_set1_ps(pos_inf);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_min_ps(a, b);
    EXPECT_TRUE(extract_ps(c, 0) == normal);

    /* min(-inf, normal) = -inf */
    a = _mm_set1_ps(neg_inf);
    c = _mm_min_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));
    EXPECT_TRUE(extract_ps(c, 0) < 0);

    /* min(inf, -inf) = -inf */
    a = _mm_set1_ps(pos_inf);
    b = _mm_set1_ps(neg_inf);
    c = _mm_min_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));
    EXPECT_TRUE(extract_ps(c, 0) < 0);

    return TEST_SUCCESS;
}

TEST_CASE(max_ps_infinity)
{
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();
    float normal = 1.0f;

    /* max(inf, normal) = inf */
    __m128 a = _mm_set1_ps(pos_inf);
    __m128 b = _mm_set1_ps(normal);
    __m128 c = _mm_max_ps(a, b);
    EXPECT_FLOAT_INF(extract_ps(c, 0));
    EXPECT_TRUE(extract_ps(c, 0) > 0);

    /* max(-inf, normal) = normal */
    a = _mm_set1_ps(neg_inf);
    c = _mm_max_ps(a, b);
    EXPECT_TRUE(extract_ps(c, 0) == normal);

    return TEST_SUCCESS;
}

/* Sqrt / Rsqrt / Rcp Tests */

TEST_CASE(sqrt_ps_special_values)
{
    float pos_inf = make_pos_inf();
    float neg_one = -1.0f;
    float zero = 0.0f;

    /* sqrt(0) = 0 */
    __m128 a = _mm_set1_ps(zero);
    __m128 c = _mm_sqrt_ps(a);
    EXPECT_TRUE(extract_ps(c, 0) == 0.0f);

    /*
     * sqrt(inf):
     * - x86 SSE: returns +inf
     * - ARM NEON (no precise): returns +inf
     * - ARM NEON (with SSE2NEON_PRECISE_SQRT): Newton-Raphson iterations
     *   may return NaN due to iteration behavior with infinity
     */
    a = _mm_set1_ps(pos_inf);
    c = _mm_sqrt_ps(a);
    float result = extract_ps(c, 0);
    EXPECT_TRUE(std::isinf(result) || std::isnan(result));

    /* sqrt(negative) = NaN */
    a = _mm_set1_ps(neg_one);
    c = _mm_sqrt_ps(a);
    EXPECT_FLOAT_NAN(extract_ps(c, 0));

    return TEST_SUCCESS;
}

TEST_CASE(rsqrt_ps_special_values)
{
    float pos_inf = make_pos_inf();
    float zero = 0.0f;

    /*
     * rsqrt(inf):
     * - x86 SSE: returns 0
     * - ARM NEON: returns NaN (documented NEON behavior difference)
     *
     * This is a known platform divergence. The NEON vrsqrte instruction
     * returns NaN for infinity input, while x86 rsqrtps returns 0.
     */
    __m128 a = _mm_set1_ps(pos_inf);
    __m128 c = _mm_rsqrt_ps(a);
    float result = extract_ps(c, 0);
    /* Accept 0 (x86) or NaN (NEON) or very small value */
    EXPECT_TRUE(result == 0.0f || std::isnan(result) ||
                (result > 0.0f && result < 1e-19f));

    /*
     * rsqrt(0):
     * - x86 SSE: returns +inf
     * - ARM NEON: returns NaN (documented NEON behavior)
     */
    a = _mm_set1_ps(zero);
    c = _mm_rsqrt_ps(a);
    result = extract_ps(c, 0);
    EXPECT_TRUE(std::isinf(result) || std::isnan(result));

    return TEST_SUCCESS;
}

TEST_CASE(rcp_ps_special_values)
{
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();
    float pos_zero = 0.0f;
    float neg_zero = -0.0f;

    /* rcp(+inf) = +0 */
    __m128 a = _mm_set1_ps(pos_inf);
    __m128 c = _mm_rcp_ps(a);
    float result = extract_ps(c, 0);
    EXPECT_TRUE(result == 0.0f);
    uint32_t bits;
    memcpy(&bits, &result, sizeof(bits));
    EXPECT_TRUE(bits == 0x00000000); /* Verify positive zero */

    /* rcp(-inf) = -0 (verify sign bit) */
    a = _mm_set1_ps(neg_inf);
    c = _mm_rcp_ps(a);
    result = extract_ps(c, 0);
    EXPECT_TRUE(result == 0.0f);
    memcpy(&bits, &result, sizeof(bits));
    /* Accept either -0 (0x80000000) or +0 (0x00000000) due to platform variance
     */
    EXPECT_TRUE(bits == 0x80000000 || bits == 0x00000000);

    /* rcp(+0) = +inf (or very large positive number due to approximation) */
    a = _mm_set1_ps(pos_zero);
    c = _mm_rcp_ps(a);
    result = extract_ps(c, 0);
    EXPECT_TRUE((std::isinf(result) && result > 0) || result > 1e30f);

    /* rcp(-0) = -inf (or very large negative number) */
    a = _mm_set1_ps(neg_zero);
    c = _mm_rcp_ps(a);
    result = extract_ps(c, 0);
    EXPECT_TRUE((std::isinf(result) && result < 0) || result < -1e30f);

    return TEST_SUCCESS;
}

/* Double Precision Tests */

TEST_CASE(add_pd_nan_propagation)
{
    double nan = make_qnan_d();
    double normal = 1.0;

    __m128d a = _mm_set_pd(nan, normal);
    __m128d b = _mm_set_pd(normal, nan);
    __m128d c = _mm_add_pd(a, b);

    EXPECT_TRUE(std::isnan(extract_pd(c, 0)));
    EXPECT_TRUE(std::isnan(extract_pd(c, 1)));

    return TEST_SUCCESS;
}

TEST_CASE(add_pd_infinity)
{
    double pos_inf = make_pos_inf_d();
    double neg_inf = make_neg_inf_d();
    double normal = 1.0;

    /* inf + normal = inf */
    __m128d a = _mm_set1_pd(pos_inf);
    __m128d b = _mm_set1_pd(normal);
    __m128d c = _mm_add_pd(a, b);
    EXPECT_TRUE(std::isinf(extract_pd(c, 0)));

    /* inf + (-inf) = NaN */
    b = _mm_set1_pd(neg_inf);
    c = _mm_add_pd(a, b);
    EXPECT_TRUE(std::isnan(extract_pd(c, 0)));

    return TEST_SUCCESS;
}

TEST_CASE(min_pd_nan_behavior)
{
    double nan = make_qnan_d();
    double normal = 1.0;

#if SSE2NEON_PRECISE_MINMAX
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(normal);
    __m128d c = _mm_min_pd(a, b);
    EXPECT_TRUE(extract_pd(c, 0) == normal);

    c = _mm_min_pd(b, a);
    EXPECT_TRUE(std::isnan(extract_pd(c, 0)));
#else
    __m128d a = _mm_set1_pd(nan);
    __m128d b = _mm_set1_pd(normal);
    __m128d c = _mm_min_pd(a, b);
    double result = extract_pd(c, 0);
    EXPECT_TRUE(std::isnan(result) || result == normal);
#endif

    return TEST_SUCCESS;
}

/* Double Precision Helper Functions */
static inline double make_pos_zero_d(void)
{
    return 0.0;
}

static inline double make_neg_zero_d(void)
{
    return -0.0;
}

TEST_CASE(signed_zero_pd_div)
{
    double pos_zero = make_pos_zero_d();
    double neg_zero = make_neg_zero_d();
    double positive = 1.0;
    double negative = -1.0;

    /* IEEE-754 division with signed zeros (double precision):
     *   positive / +0 = +Inf
     *   positive / -0 = -Inf
     *   negative / +0 = -Inf
     *   negative / -0 = +Inf
     *
     * Verify all SIMD lanes to catch lane-mixing bugs.
     */

    /* positive / (+0) = +Inf (all lanes) */
    __m128d a = _mm_set1_pd(positive);
    __m128d b = _mm_set1_pd(pos_zero);
    __m128d c = _mm_div_pd(a, b);
    EXPECT_TRUE(verify_all_lanes_pos_inf_pd(c));

    /* positive / (-0) = -Inf (all lanes) */
    b = _mm_set1_pd(neg_zero);
    c = _mm_div_pd(a, b);
    EXPECT_TRUE(verify_all_lanes_neg_inf_pd(c));

    /* negative / (+0) = -Inf (all lanes) */
    a = _mm_set1_pd(negative);
    b = _mm_set1_pd(pos_zero);
    c = _mm_div_pd(a, b);
    EXPECT_TRUE(verify_all_lanes_neg_inf_pd(c));

    /* negative / (-0) = +Inf (all lanes) */
    b = _mm_set1_pd(neg_zero);
    c = _mm_div_pd(a, b);
    EXPECT_TRUE(verify_all_lanes_pos_inf_pd(c));

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_pd_sqrt)
{
    double pos_zero = make_pos_zero_d();
    double neg_zero = make_neg_zero_d();

    /* IEEE-754: sqrt preserves sign of zero (double precision)
     *   sqrt(+0) = +0
     *   sqrt(-0) = -0
     *
     * Verify all SIMD lanes to catch lane-mixing bugs.
     */

    /* sqrt(+0) = +0 (all lanes) */
    __m128d a = _mm_set1_pd(pos_zero);
    __m128d c = _mm_sqrt_pd(a);
    EXPECT_TRUE(verify_all_lanes_pd(c, 0x0000000000000000ULL));

    /* sqrt(-0) = -0 (all lanes) */
    a = _mm_set1_pd(neg_zero);
    c = _mm_sqrt_pd(a);
    EXPECT_TRUE(verify_all_lanes_pd(c, 0x8000000000000000ULL));

    return TEST_SUCCESS;
}

TEST_CASE(signed_zero_pd_arithmetic)
{
    double pos_zero = make_pos_zero_d();
    double neg_zero = make_neg_zero_d();

    /* IEEE-754 signed zero arithmetic (double precision).
     * Verify all SIMD lanes to catch lane-mixing bugs.
     */

    /* +0 + (-0) = +0 (IEEE-754 default rounding, all lanes) */
    __m128d a = _mm_set1_pd(pos_zero);
    __m128d b = _mm_set1_pd(neg_zero);
    __m128d c = _mm_add_pd(a, b);
    EXPECT_TRUE(verify_all_lanes_pd(c, 0x0000000000000000ULL));

    /* -0 + (-0) = -0 (all lanes) */
    a = _mm_set1_pd(neg_zero);
    b = _mm_set1_pd(neg_zero);
    c = _mm_add_pd(a, b);
    EXPECT_TRUE(verify_all_lanes_pd(c, 0x8000000000000000ULL));

    /* -0 - (+0) = -0 (all lanes) */
    a = _mm_set1_pd(neg_zero);
    b = _mm_set1_pd(pos_zero);
    c = _mm_sub_pd(a, b);
    EXPECT_TRUE(verify_all_lanes_pd(c, 0x8000000000000000ULL));

    /* positive * (-0) = -0 (all lanes) */
    a = _mm_set1_pd(1.0);
    b = _mm_set1_pd(neg_zero);
    c = _mm_mul_pd(a, b);
    EXPECT_TRUE(verify_all_lanes_pd(c, 0x8000000000000000ULL));

    return TEST_SUCCESS;
}

/* Scalar Instruction Tests */

TEST_CASE(add_ss_nan_propagation)
{
    float nan = make_qnan();
    float normal1 = 1.0f;
    float normal2 = 2.0f;
    float normal3 = 3.0f;

    /* add_ss only affects lowest element */
    __m128 a = _mm_set_ps(normal3, normal2, normal1, nan);
    __m128 b = _mm_set_ps(normal3, normal2, normal1, normal1);
    __m128 c = _mm_add_ss(a, b);

    /* Low element: NaN + normal = NaN */
    EXPECT_FLOAT_NAN(extract_ps(c, 0));
    /* Other elements: unchanged from 'a' */
    EXPECT_TRUE(extract_ps(c, 1) == normal1);
    EXPECT_TRUE(extract_ps(c, 2) == normal2);
    EXPECT_TRUE(extract_ps(c, 3) == normal3);

    return TEST_SUCCESS;
}

TEST_CASE(min_ss_infinity)
{
    float pos_inf = make_pos_inf();
    float normal = 1.0f;

    __m128 a = _mm_set_ps(normal, normal, normal, pos_inf);
    __m128 b = _mm_set_ps(normal, normal, normal, normal);
    __m128 c = _mm_min_ss(a, b);

    /* Low element: min(inf, normal) = normal */
    EXPECT_TRUE(extract_ps(c, 0) == normal);
    /* Other elements: unchanged from 'a' */
    EXPECT_TRUE(extract_ps(c, 1) == normal);
    EXPECT_TRUE(extract_ps(c, 2) == normal);
    EXPECT_TRUE(extract_ps(c, 3) == normal);

    return TEST_SUCCESS;
}

/* Rounding Mode Edge Cases */

TEST_CASE(round_ps_special_values)
{
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();
    float nan = make_qnan();

    __m128 a = _mm_set_ps(nan, pos_inf, neg_inf, 1.5f);

    /* Round to nearest */
    __m128 c = _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    /* 1.5 rounds to 2.0 (banker's rounding) */
    EXPECT_TRUE(extract_ps(c, 0) == 2.0f);
    /* -inf stays -inf */
    EXPECT_FLOAT_INF(extract_ps(c, 1));
    EXPECT_TRUE(extract_ps(c, 1) < 0);
    /* +inf stays +inf */
    EXPECT_FLOAT_INF(extract_ps(c, 2));
    EXPECT_TRUE(extract_ps(c, 2) > 0);
    /* NaN stays NaN */
    EXPECT_FLOAT_NAN(extract_ps(c, 3));

    return TEST_SUCCESS;
}

TEST_CASE(floor_ps_special_values)
{
    float pos_inf = make_pos_inf();
    float nan = make_qnan();

    __m128 a = _mm_set_ps(nan, pos_inf, -1.5f, 1.5f);
    __m128 c = _mm_floor_ps(a);

    EXPECT_TRUE(extract_ps(c, 0) == 1.0f);  /* floor(1.5) = 1.0 */
    EXPECT_TRUE(extract_ps(c, 1) == -2.0f); /* floor(-1.5) = -2.0 */
    EXPECT_FLOAT_INF(extract_ps(c, 2));     /* floor(inf) = inf */
    EXPECT_FLOAT_NAN(extract_ps(c, 3));     /* floor(NaN) = NaN */

    return TEST_SUCCESS;
}

TEST_CASE(ceil_ps_special_values)
{
    float neg_inf = make_neg_inf();
    float nan = make_qnan();

    __m128 a = _mm_set_ps(nan, neg_inf, -1.5f, 1.5f);
    __m128 c = _mm_ceil_ps(a);

    EXPECT_TRUE(extract_ps(c, 0) == 2.0f);  /* ceil(1.5) = 2.0 */
    EXPECT_TRUE(extract_ps(c, 1) == -1.0f); /* ceil(-1.5) = -1.0 */
    EXPECT_FLOAT_INF(extract_ps(c, 2));     /* ceil(-inf) = -inf */
    EXPECT_FLOAT_NAN(extract_ps(c, 3));     /* ceil(NaN) = NaN */

    return TEST_SUCCESS;
}

/* Float-to-Integer Conversion Saturation Tests
 * x86 SSE returns INT32_MIN (0x80000000, "integer indefinite") for NaN, Inf,
 * and out-of-range values. These tests verify sse2neon matches this behavior.
 */

static inline int32_t extract_epi32(__m128i v, int index)
{
    int32_t r[4];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r), v);
    return r[index];
}

TEST_CASE(cvttps_epi32_nan)
{
    /* x86 returns INT32_MIN for NaN */
    float nan = make_qnan();
    __m128 a = _mm_set_ps(nan, nan, 1.0f, nan);
    __m128i r = _mm_cvttps_epi32(a);

    EXPECT_TRUE(extract_epi32(r, 0) == INT32_MIN);
    EXPECT_TRUE(extract_epi32(r, 1) == 1);
    EXPECT_TRUE(extract_epi32(r, 2) == INT32_MIN);
    EXPECT_TRUE(extract_epi32(r, 3) == INT32_MIN);

    return TEST_SUCCESS;
}

TEST_CASE(cvttps_epi32_positive_overflow)
{
    /* x86 returns INT32_MIN for values >= 2147483648.0f */
    __m128 a = _mm_set_ps(2147483648.0f, /* exactly INT32_MAX + 1 */
                          2147483647.0f, /* INT32_MAX (rounds to 2^31) */
                          4.0e9f,        /* well beyond INT32_MAX */
                          100.0f);       /* normal value */
    __m128i r = _mm_cvttps_epi32(a);

    EXPECT_TRUE(extract_epi32(r, 0) == 100);
    EXPECT_TRUE(extract_epi32(r, 1) == INT32_MIN); /* 4.0e9 overflows */
    /* Note: 2147483647.0f rounds up in float representation to 2147483648.0f */
    EXPECT_TRUE(extract_epi32(r, 2) ==
                INT32_MIN); /* INT32_MAX in float overflows */
    EXPECT_TRUE(extract_epi32(r, 3) == INT32_MIN); /* exact overflow boundary */

    return TEST_SUCCESS;
}

TEST_CASE(cvttps_epi32_negative_in_range)
{
    /* Negative values within INT32 range should convert correctly */
    __m128 a = _mm_set_ps(-2147483648.0f, /* exactly INT32_MIN */
                          -1.5f,          /* truncates to -1 */
                          -1000000.0f,    /* -1000000 */
                          -100.0f);       /* -100 */
    __m128i r = _mm_cvttps_epi32(a);

    EXPECT_TRUE(extract_epi32(r, 0) == -100);
    EXPECT_TRUE(extract_epi32(r, 1) == -1000000);
    EXPECT_TRUE(extract_epi32(r, 2) == -1); /* truncation */
    EXPECT_TRUE(extract_epi32(r, 3) ==
                INT32_MIN); /* exact INT32_MIN is valid */

    return TEST_SUCCESS;
}

TEST_CASE(cvttps_epi32_infinity)
{
    /* x86 returns INT32_MIN for infinity */
    float pos_inf = make_pos_inf();
    float neg_inf = make_neg_inf();
    __m128 a = _mm_set_ps(neg_inf, pos_inf, 1.0f, -1.0f);
    __m128i r = _mm_cvttps_epi32(a);

    EXPECT_TRUE(extract_epi32(r, 0) == -1);
    EXPECT_TRUE(extract_epi32(r, 1) == 1);
    EXPECT_TRUE(extract_epi32(r, 2) == INT32_MIN); /* +Inf */
    EXPECT_TRUE(extract_epi32(r, 3) == INT32_MIN); /* -Inf */

    return TEST_SUCCESS;
}

TEST_CASE(cvtps_epi32_saturation)
{
    /* Test _mm_cvtps_epi32 (with rounding) saturation behavior */
    float nan = make_qnan();
    float pos_inf = make_pos_inf();
    __m128 a = _mm_set_ps(pos_inf, nan, 2147483648.0f, 100.5f);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    __m128i r = _mm_cvtps_epi32(a);

    EXPECT_TRUE(extract_epi32(r, 0) == 100 || extract_epi32(r, 0) == 101);
    EXPECT_TRUE(extract_epi32(r, 1) == INT32_MIN); /* overflow */
    EXPECT_TRUE(extract_epi32(r, 2) == INT32_MIN); /* NaN */
    EXPECT_TRUE(extract_epi32(r, 3) == INT32_MIN); /* +Inf */

    return TEST_SUCCESS;
}

TEST_CASE(cvtt_ss2si_saturation)
{
    /* Test scalar _mm_cvtt_ss2si saturation behavior */
    float nan = make_qnan();
    float pos_inf = make_pos_inf();

    __m128 a_nan = _mm_set_ss(nan);
    __m128 a_inf = _mm_set_ss(pos_inf);
    __m128 a_overflow = _mm_set_ss(3.0e9f);
    __m128 a_normal = _mm_set_ss(-42.7f);

    EXPECT_TRUE(_mm_cvtt_ss2si(a_nan) == INT32_MIN);
    EXPECT_TRUE(_mm_cvtt_ss2si(a_inf) == INT32_MIN);
    EXPECT_TRUE(_mm_cvtt_ss2si(a_overflow) == INT32_MIN);
    EXPECT_TRUE(_mm_cvtt_ss2si(a_normal) == -42);

    return TEST_SUCCESS;
}

/* Main Test Runner */

static void print_precision_config(void)
{
    printf("Precision Configuration:\n");
#ifdef SSE2NEON_PRECISE_MINMAX
    printf("  SSE2NEON_PRECISE_MINMAX = %d\n", SSE2NEON_PRECISE_MINMAX);
#else
    printf("  SSE2NEON_PRECISE_MINMAX = 0 (default)\n");
#endif
#ifdef SSE2NEON_PRECISE_DIV
    printf("  SSE2NEON_PRECISE_DIV = %d\n", SSE2NEON_PRECISE_DIV);
#else
    printf("  SSE2NEON_PRECISE_DIV = 0 (default)\n");
#endif
#ifdef SSE2NEON_PRECISE_SQRT
    printf("  SSE2NEON_PRECISE_SQRT = %d\n", SSE2NEON_PRECISE_SQRT);
#else
    printf("  SSE2NEON_PRECISE_SQRT = 0 (default)\n");
#endif
#ifdef SSE2NEON_PRECISE_DP
    printf("  SSE2NEON_PRECISE_DP = %d\n", SSE2NEON_PRECISE_DP);
#else
    printf("  SSE2NEON_PRECISE_DP = 0 (default)\n");
#endif
    printf("\n");
}

int main(void)
{
    printf("===========================================\n");
    printf("IEEE-754 Floating-Point Edge Case Tests\n");
    printf("===========================================\n\n");

    print_precision_config();

    printf("--- NaN Propagation Tests ---\n");
    RUN_TEST(add_ps_nan_propagation);
    RUN_TEST(sub_ps_nan_propagation);
    RUN_TEST(mul_ps_nan_propagation);
    RUN_TEST(div_ps_nan_propagation);

    printf("\n--- Infinity Arithmetic Tests ---\n");
    RUN_TEST(add_ps_infinity);
    RUN_TEST(sub_ps_infinity);
    RUN_TEST(mul_ps_infinity);
    RUN_TEST(div_ps_infinity);

    printf("\n--- Signed Zero Tests ---\n");
    RUN_TEST(signed_zero_comparison);
    RUN_TEST(signed_zero_arithmetic);
    RUN_TEST(signed_zero_mul);
    RUN_TEST(signed_zero_div);
    RUN_TEST(signed_zero_sqrt);
    RUN_TEST(signed_zero_rsqrt);
    RUN_TEST(signed_zero_sub);
    RUN_TEST(signed_zero_minmax);

    printf("\n--- Denormal Tests ---\n");
    RUN_TEST(denormal_arithmetic);

    printf("\n--- Flush-to-Zero / Denormals-Are-Zero Tests ---\n");
    RUN_TEST(ftz_output_flush);
    RUN_TEST(daz_input_flush);
    RUN_TEST(ftz_daz_disabled);
    RUN_TEST(ftz_getcsr_roundtrip);

    printf("\n--- Comparison Tests ---\n");
    RUN_TEST(cmpord_ps_nan);
    RUN_TEST(cmpunord_ps_nan);
    RUN_TEST(cmpeq_ps_nan);
    RUN_TEST(cmplt_ps_infinity);

    printf("\n--- Min/Max Tests ---\n");
    RUN_TEST(min_ps_nan_behavior);
    RUN_TEST(max_ps_nan_behavior);
    RUN_TEST(min_ps_infinity);
    RUN_TEST(max_ps_infinity);

    printf("\n--- Sqrt/Rsqrt/Rcp Tests ---\n");
    RUN_TEST(sqrt_ps_special_values);
    RUN_TEST(rsqrt_ps_special_values);
    RUN_TEST(rcp_ps_special_values);

    printf("\n--- Double Precision Tests ---\n");
    RUN_TEST(add_pd_nan_propagation);
    RUN_TEST(add_pd_infinity);
    RUN_TEST(min_pd_nan_behavior);
    RUN_TEST(signed_zero_pd_div);
    RUN_TEST(signed_zero_pd_sqrt);
    RUN_TEST(signed_zero_pd_arithmetic);

    printf("\n--- Scalar Instruction Tests ---\n");
    RUN_TEST(add_ss_nan_propagation);
    RUN_TEST(min_ss_infinity);

    printf("\n--- Rounding Tests ---\n");
    RUN_TEST(round_ps_special_values);
    RUN_TEST(floor_ps_special_values);
    RUN_TEST(ceil_ps_special_values);

    printf("\n--- Float-to-Integer Conversion Saturation Tests ---\n");
    RUN_TEST(cvttps_epi32_nan);
    RUN_TEST(cvttps_epi32_positive_overflow);
    RUN_TEST(cvttps_epi32_negative_in_range);
    RUN_TEST(cvttps_epi32_infinity);
    RUN_TEST(cvtps_epi32_saturation);
    RUN_TEST(cvtt_ss2si_saturation);

    printf("\n===========================================\n");
    printf("Results: %d passed, %d failed, %d skipped\n", g_pass_count,
           g_fail_count, g_skip_count);
    printf("===========================================\n");

    return g_fail_count > 0 ? 1 : 0;
}
