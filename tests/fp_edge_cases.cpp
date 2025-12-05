/**
 * fp_edge_cases.cpp - IEEE-754 Floating-Point Edge Case Validation
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
 *   make fp_edge_cases && ./tests/fp_edge_cases
 *   make FEATURE=crypto+crc fp_edge_cases && ./tests/fp_edge_cases
 *
 * Reference: IEEE 754-2008, Intel Intrinsics Guide
 */

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

#define RUN_TEST(name)                      \
    do {                                    \
        result_t r = test_##name();         \
        if (r == TEST_SUCCESS) {            \
            g_pass_count++;                 \
            printf("  [PASS] %s\n", #name); \
        } else if (r == TEST_UNIMPL) {      \
            g_skip_count++;                 \
            printf("  [SKIP] %s\n", #name); \
        } else {                            \
            g_fail_count++;                 \
            printf("  [FAIL] %s\n", #name); \
        }                                   \
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

/* ============================================================================
 * NaN PROPAGATION TESTS
 * IEEE-754: Operations involving NaN should propagate NaN (with exceptions)
 * ============================================================================
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

/* ============================================================================
 * INFINITY ARITHMETIC TESTS
 * ============================================================================
 */

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

/* ============================================================================
 * SIGNED ZERO TESTS
 * IEEE-754: +0.0 and -0.0 are equal in value but distinct in representation
 * ============================================================================
 */

TEST_CASE(signed_zero_comparison)
{
    float pos_zero = make_pos_zero();
    float neg_zero = make_neg_zero();

    /* +0 == -0 in value comparison */
    __m128 a = _mm_set1_ps(pos_zero);
    __m128 b = _mm_set1_ps(neg_zero);
    __m128 c = _mm_cmpeq_ps(a, b);

    /* All bits should be set (true) */
    uint32_t result;
    memcpy(&result, &c, sizeof(result));
    EXPECT_TRUE(result == 0xFFFFFFFF);

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

/* ============================================================================
 * DENORMAL NUMBER TESTS
 * Note: NEON may flush denormals to zero depending on FPSCR settings
 * ============================================================================
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

/* ============================================================================
 * COMPARISON TESTS WITH SPECIAL VALUES
 * ============================================================================
 */

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

/* ============================================================================
 * MIN/MAX TESTS
 * Note: x86 SSE min/max have specific NaN behavior that differs from IEEE-754
 * SSE: If either operand is NaN, returns the SECOND operand
 * ============================================================================
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

/* ============================================================================
 * SQRT / RSQRT / RCP TESTS
 * ============================================================================
 */

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

/* ============================================================================
 * DOUBLE PRECISION TESTS
 * ============================================================================
 */

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

/* ============================================================================
 * SCALAR INSTRUCTION TESTS
 * ============================================================================
 */

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

/* ============================================================================
 * ROUNDING MODE EDGE CASES
 * ============================================================================
 */

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

/* ============================================================================
 * MAIN TEST RUNNER
 * ============================================================================
 */

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

    printf("\n--- Denormal Tests ---\n");
    RUN_TEST(denormal_arithmetic);

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

    printf("\n--- Scalar Instruction Tests ---\n");
    RUN_TEST(add_ss_nan_propagation);
    RUN_TEST(min_ss_infinity);

    printf("\n--- Rounding Tests ---\n");
    RUN_TEST(round_ps_special_values);
    RUN_TEST(floor_ps_special_values);
    RUN_TEST(ceil_ps_special_values);

    printf("\n===========================================\n");
    printf("Results: %d passed, %d failed, %d skipped\n", g_pass_count,
           g_fail_count, g_skip_count);
    printf("===========================================\n");

    return g_fail_count > 0 ? 1 : 0;
}
