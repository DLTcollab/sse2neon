/*
 * Differential Testing Harness for sse2neon
 *
 * Purpose: Verify NEON implementations match x86 SSE semantics by comparing
 * outputs against golden reference data generated on x86.
 *
 * Usage:
 *   1. On x86: ./tests/differential --generate golden/
 *   2. On ARM: ./tests/differential --verify golden/
 *
 * The harness uses deterministic PRNG (seed=123456) to ensure identical
 * inputs on both platforms.
 */

#include <dirent.h>
#include <sys/stat.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "binding.h"
#include "common.h"
#include "impl.h"

namespace SSE2NEON
{
/* Define instructionString array - must match INTRIN_LIST in impl.h */
const char *instructionString[] = {
#define _(x) #x,
    INTRIN_LIST
#undef _
};

/* Golden data file format:
 * - Header: magic (4), version (4), intrinsic_id (4), iteration_count (4)
 * - Per iteration: output_data (16 bytes for __m128)
 */
constexpr uint32_t GOLDEN_MAGIC = 0x474F4C44;  // "GOLD"
constexpr uint32_t GOLDEN_VERSION = 1;

/* Reduced iterations for differential testing (100 vs 10000 for main tests) */
constexpr uint32_t DIFFERENTIAL_ITERATIONS = 100;

struct GoldenHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t intrinsic_id;
    uint32_t iteration_count;
};

/* SplitMix64 PRNG - must match impl.cpp exactly */
static uint64_t prng_state;
static const double TWOPOWER64 = 18446744073709551616.0;

static void init_prng(uint64_t seed)
{
    prng_state = seed;
}

static double prng_next()
{
    uint64_t z = (prng_state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return static_cast<double>(z ^ (z >> 31));
}

static float ranf()
{
    return static_cast<float>(prng_next() / TWOPOWER64);
}

static float ranf(float low, float high)
{
    return ranf() * (high - low) + low;
}

/* Test data structure matching impl.cpp */
class DifferentialTestData
{
public:
    float *mTestFloatPointer1;
    float *mTestFloatPointer2;
    int32_t *mTestIntPointer1;
    int32_t *mTestIntPointer2;
    float mTestFloats[10000];
    int32_t mTestInts[10000];

    DifferentialTestData()
    {
        mTestFloatPointer1 =
            static_cast<float *>(platformAlignedAlloc(sizeof(__m128)));
        mTestFloatPointer2 =
            static_cast<float *>(platformAlignedAlloc(sizeof(__m128)));
        mTestIntPointer1 =
            static_cast<int32_t *>(platformAlignedAlloc(sizeof(__m128i)));
        mTestIntPointer2 =
            static_cast<int32_t *>(platformAlignedAlloc(sizeof(__m128i)));

        /* Initialize with same PRNG seed as main tests */
        init_prng(123456);
        for (uint32_t i = 0; i < 10000; i++) {
            mTestFloats[i] = ranf(-100000, 100000);
            mTestInts[i] = static_cast<int32_t>(ranf(-100000, 100000));
        }
    }

    ~DifferentialTestData()
    {
        platformAlignedFree(mTestFloatPointer1);
        platformAlignedFree(mTestFloatPointer2);
        platformAlignedFree(mTestIntPointer1);
        platformAlignedFree(mTestIntPointer2);
    }

    void loadIteration(uint32_t i)
    {
        mTestFloatPointer1[0] = mTestFloats[i];
        mTestFloatPointer1[1] = mTestFloats[i + 1];
        mTestFloatPointer1[2] = mTestFloats[i + 2];
        mTestFloatPointer1[3] = mTestFloats[i + 3];

        mTestFloatPointer2[0] = mTestFloats[i + 4];
        mTestFloatPointer2[1] = mTestFloats[i + 5];
        mTestFloatPointer2[2] = mTestFloats[i + 6];
        mTestFloatPointer2[3] = mTestFloats[i + 7];

        mTestIntPointer1[0] = mTestInts[i];
        mTestIntPointer1[1] = mTestInts[i + 1];
        mTestIntPointer1[2] = mTestInts[i + 2];
        mTestIntPointer1[3] = mTestInts[i + 3];

        mTestIntPointer2[0] = mTestInts[i + 4];
        mTestIntPointer2[1] = mTestInts[i + 5];
        mTestIntPointer2[2] = mTestInts[i + 6];
        mTestIntPointer2[3] = mTestInts[i + 7];
    }
};

/* Output union for capturing intrinsic results */
union OutputData {
    __m128 ps;
    __m128i si;
    __m128d pd;
    __m64 m64;
    uint8_t bytes[16];
    uint32_t u32[4];
    uint64_t u64[2];
    int32_t i32[4];
    int64_t i64[2];
    float f32[4];
    double f64[2];
};

/* Known differences between x86 and ARM
 *
 * These intrinsics have documented semantic differences that require
 * special comparison logic or tolerance.
 */
enum DiffCategory {
    DIFF_NONE = 0,        /* Exact match required */
    DIFF_FLOAT_TOLERANCE, /* FP tolerance allowed (rsqrt, rcp approximations) */
    DIFF_NAN_HANDLING,    /* NaN propagation differs without PRECISE_MINMAX */
    DIFF_DENORMAL,        /* Denormal handling may differ (FTZ/DAZ) */
    DIFF_SCALAR_FALLBACK, /* ARMv7 scalar fallback (f32 tolerance) */
    DIFF_SCALAR_FALLBACK_F64, /* ARMv7 scalar fallback (f64 tolerance) */
    DIFF_SKIP,                /* Skip comparison (memory ops, side effects) */
};

/* Scalar Fallback Intrinsics Documentation
 *
 * The following intrinsics have ARMv7 scalar fallback paths that execute
 * differently from AArch64 NEON implementations:
 *
 * Category: Horizontal Operations (no vaddv on ARMv7)
 *   - _mm_dp_ps, _mm_dp_pd: Dot product uses scalar accumulation
 *   - _mm_hadd_ps, _mm_hsub_ps: Horizontal add/sub uses NEON pairwise ops
 *   - _mm_hadd_pd, _mm_hsub_pd: Double-precision horizontal ops
 *   - _mm_hadd_epi16/32, _mm_hsub_epi16/32: Integer horizontal ops
 *
 * Category: Double-Precision (no f64 SIMD on ARMv7)
 *   - _mm_cvttpd_pi32, _mm_cvttpd_epi32: Scalar f64→i32 conversion loop
 *
 * Category: Movemask (no vceqq_u64 on ARMv7)
 *   - _mm_movemask_epi8: Different extraction path
 *
 * Category: String Comparison (no vmaxvq on ARMv7)
 *   - _mm_cmpistri, _mm_cmpistrm: Sequential loop (16 iterations)
 *   - _mm_cmpestri, _mm_cmpestrm: Scalar masking + extraction
 *   - _mm_cmpistra/c/o/s/z, _mm_cmpestra/c/o/s/z: Flag variants
 *
 * Category: AES (no __ARM_FEATURE_CRYPTO on some platforms)
 *   - _mm_aesenc_si128, _mm_aesenclast_si128: Vectorized S-box on AArch64,
 *     scalar T-table on ARMv7 (cache-timing vulnerable)
 *   - _mm_aesdec_si128, _mm_aesdeclast_si128: Inverse operations
 *
 * Category: CRC32 (no __ARM_FEATURE_CRC32)
 *   - _mm_crc32_u8/16/32/64: Nibble-table lookup fallback
 *
 * To test scalar fallback paths on AArch64, compile with:
 *   -DSSE2NEON_FORCE_SCALAR_FALLBACK=1
 */

struct IntrinsicInfo {
    InstructionTest id;
    const char *name;
    DiffCategory category;
    float tolerance; /* For DIFF_FLOAT_TOLERANCE */
};

/* Intrinsics with known x86/ARM differences
 *
 * Reference: IMPROVE.md Migration Guide section
 */
// clang-format off
static const IntrinsicInfo known_differences[] = {
    /* Approximation intrinsics - relaxed precision */
    {it_mm_rcp_ps, "mm_rcp_ps", DIFF_FLOAT_TOLERANCE, 1.5e-4f},
    {it_mm_rcp_ss, "mm_rcp_ss", DIFF_FLOAT_TOLERANCE, 1.5e-4f},
    {it_mm_rsqrt_ps, "mm_rsqrt_ps", DIFF_FLOAT_TOLERANCE, 1.5e-4f},
    {it_mm_rsqrt_ss, "mm_rsqrt_ss", DIFF_FLOAT_TOLERANCE, 1.5e-4f},

    /* Min/max with NaN - requires SSE2NEON_PRECISE_MINMAX for exact match */
#if !SSE2NEON_PRECISE_MINMAX
    {it_mm_min_ps, "mm_min_ps", DIFF_NAN_HANDLING, 0},
    {it_mm_min_ss, "mm_min_ss", DIFF_NAN_HANDLING, 0},
    {it_mm_max_ps, "mm_max_ps", DIFF_NAN_HANDLING, 0},
    {it_mm_max_ss, "mm_max_ss", DIFF_NAN_HANDLING, 0},
    {it_mm_min_pd, "mm_min_pd", DIFF_NAN_HANDLING, 0},
    {it_mm_min_sd, "mm_min_sd", DIFF_NAN_HANDLING, 0},
    {it_mm_max_pd, "mm_max_pd", DIFF_NAN_HANDLING, 0},
    {it_mm_max_sd, "mm_max_sd", DIFF_NAN_HANDLING, 0},
#endif

    /* Memory/side-effect intrinsics - skip comparison */
    {it_mm_malloc, "mm_malloc", DIFF_SKIP, 0},
    {it_mm_free, "mm_free", DIFF_SKIP, 0},
    {it_mm_empty, "mm_empty", DIFF_SKIP, 0},
    {it_mm_sfence, "mm_sfence", DIFF_SKIP, 0},
    {it_mm_lfence, "mm_lfence", DIFF_SKIP, 0},
    {it_mm_mfence, "mm_mfence", DIFF_SKIP, 0},
    {it_mm_pause, "mm_pause", DIFF_SKIP, 0},
    {it_mm_clflush, "mm_clflush", DIFF_SKIP, 0},
    {it_mm_prefetch, "mm_prefetch", DIFF_SKIP, 0},
    {it_mm_getcsr, "mm_getcsr", DIFF_SKIP, 0},
    {it_mm_setcsr, "mm_setcsr", DIFF_SKIP, 0},
    {it_mm_stream_pi, "mm_stream_pi", DIFF_SKIP, 0},
    {it_mm_stream_ps, "mm_stream_ps", DIFF_SKIP, 0},
    {it_mm_stream_pd, "mm_stream_pd", DIFF_SKIP, 0},
    {it_mm_stream_si32, "mm_stream_si32", DIFF_SKIP, 0},
    {it_mm_stream_si64, "mm_stream_si64", DIFF_SKIP, 0},
    {it_mm_stream_si128, "mm_stream_si128", DIFF_SKIP, 0},
    {it_mm_maskmove_si64, "mm_maskmove_si64", DIFF_SKIP, 0},
    {it_m_maskmovq, "m_maskmovq", DIFF_SKIP, 0},
    {it_mm_maskmoveu_si128, "mm_maskmoveu_si128", DIFF_SKIP, 0},
    {it_rdtsc, "rdtsc", DIFF_SKIP, 0},

    /* Flush-to-zero mode intrinsics */
    {it_mm_get_flush_zero_mode, "mm_get_flush_zero_mode", DIFF_SKIP, 0},
    {it_mm_set_flush_zero_mode, "mm_set_flush_zero_mode", DIFF_SKIP, 0},
    {it_mm_get_denormals_zero_mode, "mm_get_denormals_zero_mode", DIFF_SKIP, 0},
    {it_mm_set_denormals_zero_mode, "mm_set_denormals_zero_mode", DIFF_SKIP, 0},
    {it_mm_get_rounding_mode, "mm_get_rounding_mode", DIFF_SKIP, 0},
    {it_mm_set_rounding_mode, "mm_set_rounding_mode", DIFF_SKIP, 0},

    /* Scalar fallback intrinsics - tolerance for ARMv7 differences
     * These may have minor FP rounding differences due to different
     * instruction sequences on ARMv7 vs AArch64. */

    /* Horizontal operations (ARMv7 uses NEON pairwise ops) */
    {it_mm_hadd_ps, "mm_hadd_ps", DIFF_SCALAR_FALLBACK, 1e-6f},
    {it_mm_hsub_ps, "mm_hsub_ps", DIFF_SCALAR_FALLBACK, 1e-6f},
    {it_mm_hadd_pd, "mm_hadd_pd", DIFF_SCALAR_FALLBACK_F64, 1e-12f},
    {it_mm_hsub_pd, "mm_hsub_pd", DIFF_SCALAR_FALLBACK_F64, 1e-12f},

    /* Dot product (ARMv7 uses scalar accumulation) */
    {it_mm_dp_ps, "mm_dp_ps", DIFF_SCALAR_FALLBACK, 1e-5f},
    {it_mm_dp_pd, "mm_dp_pd", DIFF_SCALAR_FALLBACK_F64, 1e-12f},

    /* Sentinel */
    {it_last, nullptr, DIFF_NONE, 0},
};
// clang-format on

static DiffCategory get_diff_category(InstructionTest id, float *tolerance)
{
    for (size_t i = 0; known_differences[i].name != nullptr; i++) {
        if (known_differences[i].id == id) {
            if (tolerance)
                *tolerance = known_differences[i].tolerance;
            return known_differences[i].category;
        }
    }
    if (tolerance)
        *tolerance = 0;
    return DIFF_NONE;
}

/* Helper to create directory if it doesn't exist */
static bool ensure_directory(const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0)
        return S_ISDIR(st.st_mode);
    return mkdir(path, 0755) == 0;
}

/* Compare two outputs with optional tolerance */
static bool compare_output(const OutputData &golden,
                           const OutputData &actual,
                           DiffCategory cat,
                           float tolerance)
{
    switch (cat) {
    case DIFF_SKIP:
        return true;

    case DIFF_NAN_HANDLING:
        /* For NaN handling differences:
         * - Golden NaN (any actual): pass (ARM may not propagate NaN)
         * - Golden not NaN, actual NaN: FAIL (ARM shouldn't create NaN)
         * - Neither NaN: require exact match */
        for (int i = 0; i < 4; i++) {
            if (std::isnan(golden.f32[i]))
                continue; /* x86 NaN - any ARM result OK */
            if (std::isnan(actual.f32[i]))
                return false; /* ARM produced unexpected NaN */
            if (golden.u32[i] != actual.u32[i])
                return false;
        }
        return true;

    case DIFF_FLOAT_TOLERANCE:
        for (int i = 0; i < 4; i++) {
            bool g_nan = std::isnan(golden.f32[i]);
            bool a_nan = std::isnan(actual.f32[i]);
            if (g_nan && a_nan)
                continue;
            if (g_nan != a_nan)
                return false; /* NaN mismatch */
            bool g_inf = std::isinf(golden.f32[i]);
            bool a_inf = std::isinf(actual.f32[i]);
            if (g_inf && a_inf) {
                if ((golden.f32[i] > 0) != (actual.f32[i] > 0))
                    return false;
                continue;
            }
            if (g_inf != a_inf)
                return false; /* Inf mismatch */
            float diff = std::fabs(golden.f32[i] - actual.f32[i]);
            float rel = std::fabs(golden.f32[i]) > 1e-6f
                            ? diff / std::fabs(golden.f32[i])
                            : diff;
            if (rel > tolerance && diff > tolerance)
                return false;
        }
        return true;

    case DIFF_SCALAR_FALLBACK:
        /* Scalar fallback paths may have minor FP differences (f32) */
        for (int i = 0; i < 4; i++) {
            bool g_nan = std::isnan(golden.f32[i]);
            bool a_nan = std::isnan(actual.f32[i]);
            if (g_nan && a_nan)
                continue;
            if (g_nan != a_nan)
                return false; /* NaN mismatch */
            bool g_inf = std::isinf(golden.f32[i]);
            bool a_inf = std::isinf(actual.f32[i]);
            if (g_inf && a_inf) {
                if ((golden.f32[i] > 0) != (actual.f32[i] > 0))
                    return false;
                continue;
            }
            if (g_inf != a_inf)
                return false; /* Inf mismatch */
            /* Allow both exact match and tolerance-based match */
            if (golden.u32[i] == actual.u32[i])
                continue;
            float diff = std::fabs(golden.f32[i] - actual.f32[i]);
            float rel = std::fabs(golden.f32[i]) > 1e-6f
                            ? diff / std::fabs(golden.f32[i])
                            : diff;
            if (rel > tolerance && diff > tolerance)
                return false;
        }
        return true;

    case DIFF_SCALAR_FALLBACK_F64:
        /* Scalar fallback paths for double-precision intrinsics */
        for (int i = 0; i < 2; i++) {
            bool g_nan = std::isnan(golden.f64[i]);
            bool a_nan = std::isnan(actual.f64[i]);
            if (g_nan && a_nan)
                continue;
            if (g_nan != a_nan)
                return false; /* NaN mismatch */
            bool g_inf = std::isinf(golden.f64[i]);
            bool a_inf = std::isinf(actual.f64[i]);
            if (g_inf && a_inf) {
                if ((golden.f64[i] > 0) != (actual.f64[i] > 0))
                    return false;
                continue;
            }
            if (g_inf != a_inf)
                return false; /* Inf mismatch */
            /* Allow both exact match and tolerance-based match */
            if (golden.u64[i] == actual.u64[i])
                continue;
            double diff = std::fabs(golden.f64[i] - actual.f64[i]);
            double rel = std::fabs(golden.f64[i]) > 1e-12
                             ? diff / std::fabs(golden.f64[i])
                             : diff;
            if (rel > static_cast<double>(tolerance) &&
                diff > static_cast<double>(tolerance))
                return false;
        }
        return true;

    case DIFF_DENORMAL:
    case DIFF_NONE:
    default:
        /* Exact bit match required */
        return memcmp(golden.bytes, actual.bytes, 16) == 0;
    }
}

/* Execute a single intrinsic and capture output
 *
 * This is a simplified version - full implementation would need to handle
 * each intrinsic's specific signature and output type.
 */
static bool execute_intrinsic(InstructionTest id,
                              DifferentialTestData &data,
                              OutputData &out)
{
    memset(&out, 0, sizeof(out));

    const float *fp1 = data.mTestFloatPointer1;
    const float *fp2 = data.mTestFloatPointer2;
    const int32_t *ip1 = data.mTestIntPointer1;
    const int32_t *ip2 = data.mTestIntPointer2;

    /* Load standard inputs */
    __m128 a_ps = _mm_loadu_ps(fp1);
    __m128 b_ps = _mm_loadu_ps(fp2);
    __m128i a_si = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ip1));
    __m128i b_si = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ip2));
    __m128d a_pd =
        _mm_loadu_pd(reinterpret_cast<const double *>(data.mTestFloatPointer1));
    __m128d b_pd =
        _mm_loadu_pd(reinterpret_cast<const double *>(data.mTestFloatPointer2));

    (void) a_pd;
    (void) b_pd;
    (void) a_si;
    (void) b_si;

    /* Representative intrinsics for differential testing
     *
     * This covers the major categories; full coverage would enumerate all 544.
     * Priority: intrinsics with known semantic complexity or past issues.
     */
    switch (id) {
    /* Arithmetic - float */
    case it_mm_add_ps:
        out.ps = _mm_add_ps(a_ps, b_ps);
        return true;
    case it_mm_sub_ps:
        out.ps = _mm_sub_ps(a_ps, b_ps);
        return true;
    case it_mm_mul_ps:
        out.ps = _mm_mul_ps(a_ps, b_ps);
        return true;
    case it_mm_div_ps:
        out.ps = _mm_div_ps(a_ps, b_ps);
        return true;
    case it_mm_add_ss:
        out.ps = _mm_add_ss(a_ps, b_ps);
        return true;
    case it_mm_sub_ss:
        out.ps = _mm_sub_ss(a_ps, b_ps);
        return true;
    case it_mm_mul_ss:
        out.ps = _mm_mul_ss(a_ps, b_ps);
        return true;
    case it_mm_div_ss:
        out.ps = _mm_div_ss(a_ps, b_ps);
        return true;

    /* Min/Max - critical for NaN handling */
    case it_mm_min_ps:
        out.ps = _mm_min_ps(a_ps, b_ps);
        return true;
    case it_mm_max_ps:
        out.ps = _mm_max_ps(a_ps, b_ps);
        return true;
    case it_mm_min_ss:
        out.ps = _mm_min_ss(a_ps, b_ps);
        return true;
    case it_mm_max_ss:
        out.ps = _mm_max_ss(a_ps, b_ps);
        return true;

    /* Approximations - tolerance required */
    case it_mm_rcp_ps:
        out.ps = _mm_rcp_ps(a_ps);
        return true;
    case it_mm_rcp_ss:
        out.ps = _mm_rcp_ss(a_ps);
        return true;
    case it_mm_rsqrt_ps:
        /* Use absolute values to avoid NaN from negative sqrt */
        out.ps = _mm_rsqrt_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), a_ps));
        return true;
    case it_mm_rsqrt_ss:
        out.ps = _mm_rsqrt_ss(_mm_andnot_ps(_mm_set1_ps(-0.0f), a_ps));
        return true;
    case it_mm_sqrt_ps:
        out.ps = _mm_sqrt_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), a_ps));
        return true;
    case it_mm_sqrt_ss:
        out.ps = _mm_sqrt_ss(_mm_andnot_ps(_mm_set1_ps(-0.0f), a_ps));
        return true;

    /* Comparisons */
    case it_mm_cmpeq_ps:
        out.ps = _mm_cmpeq_ps(a_ps, b_ps);
        return true;
    case it_mm_cmplt_ps:
        out.ps = _mm_cmplt_ps(a_ps, b_ps);
        return true;
    case it_mm_cmple_ps:
        out.ps = _mm_cmple_ps(a_ps, b_ps);
        return true;
    case it_mm_cmpgt_ps:
        out.ps = _mm_cmpgt_ps(a_ps, b_ps);
        return true;
    case it_mm_cmpge_ps:
        out.ps = _mm_cmpge_ps(a_ps, b_ps);
        return true;

    /* Integer arithmetic */
    case it_mm_add_epi8:
        out.si = _mm_add_epi8(a_si, b_si);
        return true;
    case it_mm_add_epi16:
        out.si = _mm_add_epi16(a_si, b_si);
        return true;
    case it_mm_add_epi32:
        out.si = _mm_add_epi32(a_si, b_si);
        return true;
    case it_mm_add_epi64:
        out.si = _mm_add_epi64(a_si, b_si);
        return true;
    case it_mm_sub_epi8:
        out.si = _mm_sub_epi8(a_si, b_si);
        return true;
    case it_mm_sub_epi16:
        out.si = _mm_sub_epi16(a_si, b_si);
        return true;
    case it_mm_sub_epi32:
        out.si = _mm_sub_epi32(a_si, b_si);
        return true;
    case it_mm_sub_epi64:
        out.si = _mm_sub_epi64(a_si, b_si);
        return true;

    /* Saturating arithmetic */
    case it_mm_adds_epi8:
        out.si = _mm_adds_epi8(a_si, b_si);
        return true;
    case it_mm_adds_epi16:
        out.si = _mm_adds_epi16(a_si, b_si);
        return true;
    case it_mm_adds_epu8:
        out.si = _mm_adds_epu8(a_si, b_si);
        return true;
    case it_mm_adds_epu16:
        out.si = _mm_adds_epu16(a_si, b_si);
        return true;
    case it_mm_subs_epi8:
        out.si = _mm_subs_epi8(a_si, b_si);
        return true;
    case it_mm_subs_epi16:
        out.si = _mm_subs_epi16(a_si, b_si);
        return true;
    case it_mm_subs_epu8:
        out.si = _mm_subs_epu8(a_si, b_si);
        return true;
    case it_mm_subs_epu16:
        out.si = _mm_subs_epu16(a_si, b_si);
        return true;

    /* Multiply */
    case it_mm_mullo_epi16:
        out.si = _mm_mullo_epi16(a_si, b_si);
        return true;
    case it_mm_mulhi_epi16:
        out.si = _mm_mulhi_epi16(a_si, b_si);
        return true;
    case it_mm_mulhi_epu16:
        out.si = _mm_mulhi_epu16(a_si, b_si);
        return true;
    case it_mm_mul_epu32:
        out.si = _mm_mul_epu32(a_si, b_si);
        return true;

    /* Shifts */
    case it_mm_slli_epi16:
        out.si = _mm_slli_epi16(a_si, 4);
        return true;
    case it_mm_slli_epi32:
        out.si = _mm_slli_epi32(a_si, 8);
        return true;
    case it_mm_slli_epi64:
        out.si = _mm_slli_epi64(a_si, 16);
        return true;
    case it_mm_srli_epi16:
        out.si = _mm_srli_epi16(a_si, 4);
        return true;
    case it_mm_srli_epi32:
        out.si = _mm_srli_epi32(a_si, 8);
        return true;
    case it_mm_srli_epi64:
        out.si = _mm_srli_epi64(a_si, 16);
        return true;
    case it_mm_srai_epi16:
        out.si = _mm_srai_epi16(a_si, 4);
        return true;
    case it_mm_srai_epi32:
        out.si = _mm_srai_epi32(a_si, 8);
        return true;

    /* Logical */
    case it_mm_and_ps:
        out.ps = _mm_and_ps(a_ps, b_ps);
        return true;
    case it_mm_andnot_ps:
        out.ps = _mm_andnot_ps(a_ps, b_ps);
        return true;
    case it_mm_or_ps:
        out.ps = _mm_or_ps(a_ps, b_ps);
        return true;
    case it_mm_xor_ps:
        out.ps = _mm_xor_ps(a_ps, b_ps);
        return true;
    case it_mm_and_si128:
        out.si = _mm_and_si128(a_si, b_si);
        return true;
    case it_mm_andnot_si128:
        out.si = _mm_andnot_si128(a_si, b_si);
        return true;
    case it_mm_or_si128:
        out.si = _mm_or_si128(a_si, b_si);
        return true;
    case it_mm_xor_si128:
        out.si = _mm_xor_si128(a_si, b_si);
        return true;

    /* Shuffle/permute */
    case it_mm_shuffle_epi32:
        out.si = _mm_shuffle_epi32(a_si, 0x1B);  // reverse
        return true;
    case it_mm_shufflelo_epi16:
        out.si = _mm_shufflelo_epi16(a_si, 0x1B);
        return true;
    case it_mm_shufflehi_epi16:
        out.si = _mm_shufflehi_epi16(a_si, 0x1B);
        return true;
    case it_mm_shuffle_ps:
        out.ps = _mm_shuffle_ps(a_ps, b_ps, _MM_SHUFFLE(3, 2, 1, 0));
        return true;

    /* Pack/unpack */
    case it_mm_packs_epi16:
        out.si = _mm_packs_epi16(a_si, b_si);
        return true;
    case it_mm_packs_epi32:
        out.si = _mm_packs_epi32(a_si, b_si);
        return true;
    case it_mm_packus_epi16:
        out.si = _mm_packus_epi16(a_si, b_si);
        return true;
    case it_mm_unpackhi_epi8:
        out.si = _mm_unpackhi_epi8(a_si, b_si);
        return true;
    case it_mm_unpackhi_epi16:
        out.si = _mm_unpackhi_epi16(a_si, b_si);
        return true;
    case it_mm_unpackhi_epi32:
        out.si = _mm_unpackhi_epi32(a_si, b_si);
        return true;
    case it_mm_unpackhi_epi64:
        out.si = _mm_unpackhi_epi64(a_si, b_si);
        return true;
    case it_mm_unpacklo_epi8:
        out.si = _mm_unpacklo_epi8(a_si, b_si);
        return true;
    case it_mm_unpacklo_epi16:
        out.si = _mm_unpacklo_epi16(a_si, b_si);
        return true;
    case it_mm_unpacklo_epi32:
        out.si = _mm_unpacklo_epi32(a_si, b_si);
        return true;
    case it_mm_unpacklo_epi64:
        out.si = _mm_unpacklo_epi64(a_si, b_si);
        return true;

    /* Conversion */
    case it_mm_cvtepi32_ps:
        out.ps = _mm_cvtepi32_ps(a_si);
        return true;
    case it_mm_cvtps_epi32:
        out.si = _mm_cvtps_epi32(a_ps);
        return true;
    case it_mm_cvttps_epi32:
        out.si = _mm_cvttps_epi32(a_ps);
        return true;

    /* Movemask - critical for string/search ops */
    case it_mm_movemask_epi8:
        out.i32[0] = _mm_movemask_epi8(a_si);
        return true;
    case it_mm_movemask_ps:
        out.i32[0] = _mm_movemask_ps(a_ps);
        return true;

    /* SSE4.1 */
    case it_mm_blend_ps:
        out.ps = _mm_blend_ps(a_ps, b_ps, 0x5);
        return true;
    case it_mm_blendv_ps:
        out.ps = _mm_blendv_ps(a_ps, b_ps, _mm_cmplt_ps(a_ps, b_ps));
        return true;
    case it_mm_blend_epi16:
        out.si = _mm_blend_epi16(a_si, b_si, 0x55);
        return true;
    case it_mm_blendv_epi8:
        out.si = _mm_blendv_epi8(a_si, b_si, _mm_cmplt_epi8(a_si, b_si));
        return true;
    case it_mm_min_epi8:
        out.si = _mm_min_epi8(a_si, b_si);
        return true;
    case it_mm_max_epi8:
        out.si = _mm_max_epi8(a_si, b_si);
        return true;
    case it_mm_min_epu16:
        out.si = _mm_min_epu16(a_si, b_si);
        return true;
    case it_mm_max_epu16:
        out.si = _mm_max_epu16(a_si, b_si);
        return true;
    case it_mm_min_epi32:
        out.si = _mm_min_epi32(a_si, b_si);
        return true;
    case it_mm_max_epi32:
        out.si = _mm_max_epi32(a_si, b_si);
        return true;
    case it_mm_min_epu32:
        out.si = _mm_min_epu32(a_si, b_si);
        return true;
    case it_mm_max_epu32:
        out.si = _mm_max_epu32(a_si, b_si);
        return true;
    case it_mm_mullo_epi32:
        out.si = _mm_mullo_epi32(a_si, b_si);
        return true;

    /* ----- Scalar fallback intrinsics -----
     * These intrinsics have ARMv7 scalar fallback paths.
     * Testing them ensures both AArch64 NEON and ARMv7 scalar paths work.
     */

    /* SSE3 Horizontal operations - uses NEON pairwise ops */
    case it_mm_hadd_ps:
        out.ps = _mm_hadd_ps(a_ps, b_ps);
        return true;
    case it_mm_hsub_ps:
        out.ps = _mm_hsub_ps(a_ps, b_ps);
        return true;
    case it_mm_hadd_pd:
        out.pd = _mm_hadd_pd(a_pd, b_pd);
        return true;
    case it_mm_hsub_pd:
        out.pd = _mm_hsub_pd(a_pd, b_pd);
        return true;

    /* SSE3 Add/Sub */
    case it_mm_addsub_ps:
        out.ps = _mm_addsub_ps(a_ps, b_ps);
        return true;
    case it_mm_addsub_pd:
        out.pd = _mm_addsub_pd(a_pd, b_pd);
        return true;

    /* SSSE3 Horizontal integer ops - ARMv7 uses different reduction */
    case it_mm_hadd_epi16:
        out.si = _mm_hadd_epi16(a_si, b_si);
        return true;
    case it_mm_hadd_epi32:
        out.si = _mm_hadd_epi32(a_si, b_si);
        return true;
    case it_mm_hsub_epi16:
        out.si = _mm_hsub_epi16(a_si, b_si);
        return true;
    case it_mm_hsub_epi32:
        out.si = _mm_hsub_epi32(a_si, b_si);
        return true;
    case it_mm_hadds_epi16:
        out.si = _mm_hadds_epi16(a_si, b_si);
        return true;
    case it_mm_hsubs_epi16:
        out.si = _mm_hsubs_epi16(a_si, b_si);
        return true;

    /* SSE4.1 Dot product - ARMv7 uses scalar accumulation (no vaddvq_f32) */
    case it_mm_dp_ps:
        out.ps = _mm_dp_ps(a_ps, b_ps, 0xFF); /* All lanes multiply and sum */
        return true;
    case it_mm_dp_pd:
        out.pd = _mm_dp_pd(a_pd, b_pd, 0x33); /* Both lanes multiply and sum */
        return true;

    /* Double-precision conversion - ARMv7 scalar fallback (no f64 SIMD) */
    case it_mm_cvttpd_epi32:
        out.si = _mm_cvttpd_epi32(a_pd);
        return true;
    case it_mm_cvttpd_pi32:
        /* Returns __m64 (64 bits) - zero upper half for consistent comparison
         */
        memset(&out, 0, sizeof(out));
        out.m64 = _mm_cvttpd_pi32(a_pd);
        return true;

    /* SSE4.1 more intrinsics */
    case it_mm_minpos_epu16:
        out.si = _mm_minpos_epu16(a_si);
        return true;
    case it_mm_mpsadbw_epu8:
        out.si = _mm_mpsadbw_epu8(a_si, b_si, 0);
        return true;

    /* SSE4.2 String comparison - ARMv7 uses sequential loops */
    case it_mm_cmpistrm:
        out.si =
            _mm_cmpistrm(a_si, b_si, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpistri:
        out.i32[0] =
            _mm_cmpistri(a_si, b_si, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpistra:
        out.i32[0] =
            _mm_cmpistra(a_si, b_si, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpistrc:
        out.i32[0] =
            _mm_cmpistrc(a_si, b_si, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpistro:
        out.i32[0] =
            _mm_cmpistro(a_si, b_si, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpistrs:
        out.i32[0] =
            _mm_cmpistrs(a_si, b_si, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpistrz:
        out.i32[0] =
            _mm_cmpistrz(a_si, b_si, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;

    case it_mm_cmpestrm:
        out.si = _mm_cmpestrm(a_si, 16, b_si, 16,
                              _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpestri:
        out.i32[0] = _mm_cmpestri(a_si, 16, b_si, 16,
                                  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpestra:
        out.i32[0] = _mm_cmpestra(a_si, 16, b_si, 16,
                                  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpestrc:
        out.i32[0] = _mm_cmpestrc(a_si, 16, b_si, 16,
                                  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpestro:
        out.i32[0] = _mm_cmpestro(a_si, 16, b_si, 16,
                                  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpestrs:
        out.i32[0] = _mm_cmpestrs(a_si, 16, b_si, 16,
                                  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;
    case it_mm_cmpestrz:
        out.i32[0] = _mm_cmpestrz(a_si, 16, b_si, 16,
                                  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
        return true;

    /* SSE4.2 CRC32 - software fallback when no __ARM_FEATURE_CRC32 */
    case it_mm_crc32_u8:
        out.u32[0] =
            _mm_crc32_u8(static_cast<uint32_t>(static_cast<unsigned>(ip1[0])),
                         static_cast<uint8_t>(ip2[0]));
        return true;
    case it_mm_crc32_u16:
        out.u32[0] =
            _mm_crc32_u16(static_cast<uint32_t>(static_cast<unsigned>(ip1[0])),
                          static_cast<uint16_t>(ip2[0]));
        return true;
    case it_mm_crc32_u32:
        out.u32[0] =
            _mm_crc32_u32(static_cast<uint32_t>(static_cast<unsigned>(ip1[0])),
                          static_cast<uint32_t>(static_cast<unsigned>(ip2[0])));
        return true;
    case it_mm_crc32_u64:
        out.u64[0] = _mm_crc32_u64(
            static_cast<uint64_t>(static_cast<unsigned>(ip1[0])) |
                (static_cast<uint64_t>(static_cast<unsigned>(ip1[1])) << 32),
            static_cast<uint64_t>(static_cast<unsigned>(ip2[0])) |
                (static_cast<uint64_t>(static_cast<unsigned>(ip2[1])) << 32));
        return true;

        /* AES - if available */
#if defined(__ARM_FEATURE_CRYPTO) || (defined(__x86_64__) || defined(__i386__))
    case it_mm_aesenc_si128:
        out.si = _mm_aesenc_si128(a_si, b_si);
        return true;
    case it_mm_aesenclast_si128:
        out.si = _mm_aesenclast_si128(a_si, b_si);
        return true;
    case it_mm_aesdec_si128:
        out.si = _mm_aesdec_si128(a_si, b_si);
        return true;
    case it_mm_aesdeclast_si128:
        out.si = _mm_aesdeclast_si128(a_si, b_si);
        return true;
#endif

    /* Skip intrinsics that require special handling */
    default:
        return false;
    }
}

/* Generate golden data for a single intrinsic */
static bool generate_golden_file(InstructionTest id,
                                 const char *output_dir,
                                 DifferentialTestData &data)
{
    float tolerance;
    DiffCategory cat = get_diff_category(id, &tolerance);
    if (cat == DIFF_SKIP)
        return true;

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/%s.golden", output_dir,
             instructionString[id]);

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to create: %s\n", filepath);
        return false;
    }

    GoldenHeader header = {GOLDEN_MAGIC, GOLDEN_VERSION,
                           static_cast<uint32_t>(id), DIFFERENTIAL_ITERATIONS};
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fprintf(stderr, "Failed to write header: %s\n", filepath);
        fclose(fp);
        return false;
    }

    OutputData out;
    uint32_t valid_iterations = 0;
    bool write_error = false;
    for (uint32_t i = 0; i < DIFFERENTIAL_ITERATIONS; i++) {
        data.loadIteration(
            i * 8);  // Offset by 8 to use different data each iteration
        if (execute_intrinsic(id, data, out)) {
            if (fwrite(out.bytes, 16, 1, fp) != 1) {
                write_error = true;
                break;
            }
            valid_iterations++;
        }
    }

    if (write_error) {
        fprintf(stderr, "Failed to write data: %s\n", filepath);
        fclose(fp);
        remove(filepath);
        return false;
    }

    /* Update header with actual iteration count */
    if (valid_iterations != DIFFERENTIAL_ITERATIONS) {
        fseek(fp, 0, SEEK_SET);
        header.iteration_count = valid_iterations;
        if (fwrite(&header, sizeof(header), 1, fp) != 1) {
            fprintf(stderr, "Failed to update header: %s\n", filepath);
            fclose(fp);
            remove(filepath);
            return false;
        }
    }

    fclose(fp);

    if (valid_iterations == 0) {
        /* Remove empty file */
        remove(filepath);
        return true;
    }

    printf("Generated: %s (%u iterations)\n", instructionString[id],
           valid_iterations);
    return true;
}

/* Verify against golden data */
static bool verify_golden_file(InstructionTest id,
                               const char *golden_dir,
                               DifferentialTestData &data,
                               uint32_t *pass_count,
                               uint32_t *fail_count)
{
    float tolerance;
    DiffCategory cat = get_diff_category(id, &tolerance);
    if (cat == DIFF_SKIP) {
        return true;
    }

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/%s.golden", golden_dir,
             instructionString[id]);

    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        /* No golden file - intrinsic not covered */
        return true;
    }

    GoldenHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return false;
    }

    if (header.magic != GOLDEN_MAGIC || header.version != GOLDEN_VERSION) {
        fprintf(stderr, "Invalid golden file: %s\n", filepath);
        fclose(fp);
        return false;
    }

    OutputData golden, actual;
    bool all_pass = true;

    for (uint32_t i = 0; i < header.iteration_count; i++) {
        if (fread(golden.bytes, 16, 1, fp) != 1) {
            fprintf(stderr, "Truncated golden file: %s\n", filepath);
            fclose(fp);
            return false;
        }

        data.loadIteration(i * 8);
        if (!execute_intrinsic(id, data, actual)) {
            continue;  // Intrinsic not implemented in execute_intrinsic
        }

        if (!compare_output(golden, actual, cat, tolerance)) {
            if (all_pass) {
                fprintf(stderr, "\n%s MISMATCH at iteration %u:\n",
                        instructionString[id], i);
                fprintf(stderr, "  Golden: %08x %08x %08x %08x\n",
                        golden.u32[0], golden.u32[1], golden.u32[2],
                        golden.u32[3]);
                fprintf(stderr, "  Actual: %08x %08x %08x %08x\n",
                        actual.u32[0], actual.u32[1], actual.u32[2],
                        actual.u32[3]);
                all_pass = false;
            }
        }
    }

    fclose(fp);

    if (all_pass) {
        (*pass_count)++;
        printf("PASS: %s\n", instructionString[id]);
    } else {
        (*fail_count)++;
        printf("FAIL: %s\n", instructionString[id]);
    }

    return all_pass;
}

}  // namespace SSE2NEON

static void print_usage(const char *prog)
{
    printf("Usage: %s --generate <output_dir>\n", prog);
    printf("       %s --verify <golden_dir>\n", prog);
    printf("\n");
    printf("Differential Testing Harness for sse2neon\n");
    printf("\n");
    printf("Options:\n");
    printf("  --generate <dir>  Generate golden reference data (run on x86)\n");
    printf("  --verify <dir>    Verify against golden data (run on ARM)\n");
    printf("\n");
    printf("Known x86/ARM differences are documented in IMPROVE.md.\n");
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    bool generate_mode = (strcmp(argv[1], "--generate") == 0);
    bool verify_mode = (strcmp(argv[1], "--verify") == 0);

    if (!generate_mode && !verify_mode) {
        print_usage(argv[0]);
        return 1;
    }

    const char *dir = argv[2];

    if (generate_mode) {
        if (!SSE2NEON::ensure_directory(dir)) {
            fprintf(stderr, "Failed to create directory: %s\n", dir);
            return 1;
        }
    }

    SSE2NEON::DifferentialTestData data;

    printf("sse2neon Differential Testing\n");
    printf("==============================\n");
#if defined(__x86_64__) || defined(__i386__)
    printf("Platform: x86 (native SSE)\n");
#elif defined(__aarch64__) || defined(__arm__)
    printf("Platform: ARM (sse2neon)\n");
#else
    printf("Platform: unknown\n");
#endif
    printf("Mode: %s\n", generate_mode ? "generate" : "verify");
    printf("Directory: %s\n", dir);
    printf("\n");

    uint32_t pass_count = 0;
    uint32_t fail_count = 0;
    uint32_t skip_count = 0;

    for (uint32_t i = 0; i < SSE2NEON::it_last; i++) {
        SSE2NEON::InstructionTest id =
            static_cast<SSE2NEON::InstructionTest>(i);

        float tolerance;
        SSE2NEON::DiffCategory cat =
            SSE2NEON::get_diff_category(id, &tolerance);
        if (cat == SSE2NEON::DIFF_SKIP) {
            skip_count++;
            continue;
        }

        if (generate_mode) {
            SSE2NEON::generate_golden_file(id, dir, data);
        } else {
            SSE2NEON::verify_golden_file(id, dir, data, &pass_count,
                                         &fail_count);
        }
    }

    printf("\n");
    printf("==============================\n");
    if (verify_mode) {
        printf("Results: %u passed, %u failed, %u skipped\n", pass_count,
               fail_count, skip_count);
    } else {
        printf("Golden data generated in: %s\n", dir);
    }

    return fail_count > 0 ? 1 : 0;
}
