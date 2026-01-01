/*
 * Comprehensive fuzzer covering SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AES,
 * and CLMUL extensions. Tests sse2neon intrinsics with 23 strategies:
 *
 * 1. Out-of-range shuffle indices
 * 2. Alignment edge cases
 * 3. Integer overflow/saturation behavior
 * 4. Floating-point edge cases (NaN, Inf, denormals)
 * 5. Scalar operations (upper lane preservation)
 * 6. NaN operand ordering differences (x86 vs ARM)
 * 7. Ordered vs unordered comparisons
 * 8. Type conversion boundary cases
 * 9. SSE4.2 string comparison operations
 * 10. AES encryption/decryption rounds
 * 11. CLMUL carry-less multiplication
 * 12. Non-temporal (streaming) stores
 * 13. SSE4.1 rounding modes
 * 14. Masked memory operations
 * 15. Advanced integer ops (minpos, mulhrs, packus, 64-bit insert/extract)
 *
 * Build with:
 *   clang++ -g -O1 -fsanitize=fuzzer,address,undefined \
 *           -march=armv8-a+fp+simd+crypto tests/fuzz.cpp -o tests/fuzz
 *
 * Run:
 *   ./tests/fuzz -max_total_time=90
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "sse2neon.h"

/* Static buffers to avoid malloc overhead */
static uint8_t aligned_buf[256] __attribute__((aligned(64)));
static uint8_t unaligned_buf[256 + 15];

/* Volatile sink to prevent dead code elimination.
 * Without this, the compiler may optimize away intrinsic calls at -O1+
 * since their results are unused, defeating the purpose of fuzzing. */
static volatile uint64_t g_sink;

/* Sink a 128-bit vector to prevent DCE */
#define SINK_I128(v)                            \
    do {                                        \
        uint64_t tmp[2];                        \
        _mm_storeu_si128((__m128i *) tmp, (v)); \
        g_sink ^= tmp[0] ^ tmp[1];              \
    } while (0)

#define SINK_PS(v) SINK_I128(_mm_castps_si128(v))
#define SINK_PD(v) SINK_I128(_mm_castpd_si128(v))
#define SINK_I32(v) g_sink ^= (uint32_t) (v)
#define SINK_I64(v) g_sink ^= (uint64_t) (v)

/* Extract values from fuzz input with bounds checking */
static inline uint8_t get_u8(const uint8_t *data, size_t size, size_t idx)
{
    return (idx < size) ? data[idx] : 0;
}

static inline uint32_t get_u32(const uint8_t *data, size_t size, size_t idx)
{
    uint32_t v = 0;
    for (size_t i = 0; i < 4 && (idx + i) < size; i++)
        v |= (uint32_t) data[idx + i] << (i * 8);
    return v;
}

static inline int32_t get_i32(const uint8_t *data, size_t size, size_t idx)
{
    return (int32_t) get_u32(data, size, idx);
}

/* Strategy 1: Shuffle Index Testing
 *
 * Tests shuffle intrinsics with various index patterns including:
 * - Valid indices (0-15 for epi8, 0-3 for ps/epi32)
 * - Boundary indices (exactly at limits)
 * - High-bit set indices (negative in signed interpretation)
 * - All-zero and all-ones patterns
 */
static void test_shuffle_indices(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    /* Prepare input vectors from fuzz data */
    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i b = _mm_loadu_si128((const __m128i *) (data + 16));

    /* _mm_shuffle_epi8: Tests byte shuffle with arbitrary indices
     * The high bit of each index byte controls zeroing (SSE behavior).
     * Invalid indices (with high bit set) should produce zeros. */
    SINK_I128(_mm_shuffle_epi8(a, b));

    /* _mm_shuffle_epi32: Tests dword shuffle with compile-time immediates
     * Indices are 2-bit fields in the immediate, always valid (0-3). */
#define TEST_SHUFFLE_EPI32(IMM) SINK_I128(_mm_shuffle_epi32(a, IMM))

    /* Test a sampling of immediate values */
    TEST_SHUFFLE_EPI32(0x00);
    TEST_SHUFFLE_EPI32(0x55);
    TEST_SHUFFLE_EPI32(0xAA);
    TEST_SHUFFLE_EPI32(0xFF);
    TEST_SHUFFLE_EPI32(0x1B); /* Reverse */
    TEST_SHUFFLE_EPI32(0xE4); /* Identity */
#undef TEST_SHUFFLE_EPI32

    /* _mm_shufflelo_epi16 / _mm_shufflehi_epi16: Tests word shuffles */
#define TEST_SHUFFLELO(IMM) SINK_I128(_mm_shufflelo_epi16(a, IMM))
#define TEST_SHUFFLEHI(IMM) SINK_I128(_mm_shufflehi_epi16(a, IMM))

    TEST_SHUFFLELO(0x00);
    TEST_SHUFFLELO(0xFF);
    TEST_SHUFFLEHI(0x00);
    TEST_SHUFFLEHI(0xFF);
#undef TEST_SHUFFLELO
#undef TEST_SHUFFLEHI

    /* _mm_shuffle_ps: Tests float shuffle between two vectors */
    __m128 fa = _mm_loadu_ps((const float *) data);
    __m128 fb = _mm_loadu_ps((const float *) (data + 16));

#define TEST_SHUFFLE_PS(IMM) SINK_PS(_mm_shuffle_ps(fa, fb, IMM))

    TEST_SHUFFLE_PS(0x00);
    TEST_SHUFFLE_PS(0xFF);
    TEST_SHUFFLE_PS(0x4E); /* Swap pairs */
    TEST_SHUFFLE_PS(0xB1); /* Interleave */
#undef TEST_SHUFFLE_PS

    /* _mm_alignr_epi8: Tests byte alignment with variable shift count
     * Valid shift range is 0-32 for 128-bit vectors. */
#define TEST_ALIGNR(IMM) SINK_I128(_mm_alignr_epi8(a, b, IMM))

    /* Test boundary shift counts */
    TEST_ALIGNR(0);
    TEST_ALIGNR(1);
    TEST_ALIGNR(15);
    TEST_ALIGNR(16);
    TEST_ALIGNR(17);
    TEST_ALIGNR(31);
    /* Out-of-range values to test masking/zeroing behavior */
    TEST_ALIGNR(32);
    TEST_ALIGNR(63);
    TEST_ALIGNR(255);
#undef TEST_ALIGNR
}

/* Strategy 2: Alignment Testing
 *
 * Tests memory load/store intrinsics with various alignments:
 * - Properly aligned (16-byte boundary)
 * - Misaligned by various offsets (1-15 bytes)
 * - Edge cases near page boundaries
 */
static void test_alignment(const uint8_t *data, size_t size)
{
    if (size < 64)
        return;

    /* Copy fuzz data into aligned and unaligned buffers */
    size_t copy_len = (size < 64) ? size : 64;
    memcpy(aligned_buf, data, copy_len);

    /* Test aligned loads (should always work) */
    SINK_I128(_mm_load_si128((const __m128i *) aligned_buf));
    SINK_PS(_mm_load_ps((const float *) aligned_buf));
    SINK_PD(_mm_load_pd((const double *) aligned_buf));

    /* Test unaligned loads with various offsets */
    for (int offset = 0; offset < 16 && offset + 16 <= (int) size; offset++) {
        memcpy(unaligned_buf + offset, data, 16);

        /* _mm_loadu_* intrinsics should handle any alignment */
        SINK_I128(_mm_loadu_si128((const __m128i *) (unaligned_buf + offset)));
        SINK_PS(_mm_loadu_ps((const float *) (unaligned_buf + offset)));
        SINK_PD(_mm_loadu_pd((const double *) (unaligned_buf + offset)));
    }

    /* Test aligned stores */
    __m128i store_data = _mm_loadu_si128((const __m128i *) data);
    _mm_store_si128((__m128i *) aligned_buf, store_data);

    __m128 store_ps = _mm_loadu_ps((const float *) data);
    _mm_store_ps((float *) aligned_buf, store_ps);

    /* Test unaligned stores */
    for (int offset = 1; offset < 16; offset++) {
        _mm_storeu_si128((__m128i *) (unaligned_buf + offset), store_data);
        _mm_storeu_ps((float *) (unaligned_buf + offset), store_ps);
    }

    /* Partial loads/stores */
    if (size >= 8) {
        SINK_I128(_mm_loadl_epi64((const __m128i *) data));
        _mm_storel_epi64((__m128i *) aligned_buf, store_data);
    }

    if (size >= 4) {
        SINK_I128(_mm_loadu_si32(data));
        _mm_storeu_si32(aligned_buf, store_data);
    }
}

/* Strategy 3: Integer Overflow Testing
 *
 * Tests arithmetic intrinsics with values designed to trigger:
 * - Signed overflow (INT_MAX + 1, INT_MIN - 1)
 * - Unsigned overflow (wrap-around)
 * - Saturation behavior (adds/addu, subs/subu)
 * - Multiplication overflow
 */
static void test_integer_overflow(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    /* Load vectors from fuzz data */
    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i b = _mm_loadu_si128((const __m128i *) (data + 16));

    /* Create extreme value vectors for overflow testing */
    __m128i max_i8 = _mm_set1_epi8(INT8_MAX);
    __m128i min_i8 = _mm_set1_epi8(INT8_MIN);
    __m128i max_i16 = _mm_set1_epi16(INT16_MAX);
    __m128i min_i16 = _mm_set1_epi16(INT16_MIN);
    __m128i max_i32 = _mm_set1_epi32(INT32_MAX);
    __m128i min_i32 = _mm_set1_epi32(INT32_MIN);
    __m128i max_u8 = _mm_set1_epi8((char) UINT8_MAX);
    __m128i max_u16 = _mm_set1_epi16((short) UINT16_MAX);
    __m128i zero = _mm_setzero_si128();

    /* 8-bit arithmetic with potential overflow */
    SINK_I128(_mm_add_epi8(a, b));
    SINK_I128(_mm_sub_epi8(a, b));
    SINK_I128(_mm_adds_epi8(a, max_i8)); /* Saturating add positive */
    SINK_I128(_mm_subs_epi8(a, max_i8)); /* Saturating sub */
    SINK_I128(_mm_adds_epu8(a, max_u8)); /* Unsigned saturating */
    SINK_I128(_mm_subs_epu8(a, max_u8));
    SINK_I128(_mm_adds_epi8(min_i8, min_i8)); /* Saturate to INT8_MIN */
    SINK_I128(_mm_subs_epi8(min_i8, max_i8)); /* Saturate to INT8_MIN */
    SINK_I128(_mm_subs_epu8(zero, a)); /* Unsigned underflow clamps to 0 */

    /* 16-bit arithmetic */
    SINK_I128(_mm_add_epi16(a, b));
    SINK_I128(_mm_sub_epi16(a, b));
    SINK_I128(_mm_adds_epi16(a, max_i16));
    SINK_I128(_mm_subs_epi16(a, min_i16));
    SINK_I128(_mm_adds_epu16(a, max_u16));
    SINK_I128(_mm_subs_epu16(a, max_u16));
    SINK_I128(_mm_adds_epi16(min_i16, min_i16)); /* Saturate to INT16_MIN */
    SINK_I128(_mm_subs_epi16(min_i16, max_i16)); /* Saturate to INT16_MIN */
    SINK_I128(_mm_subs_epu16(zero, a));          /* Unsigned underflow */

    /* 32-bit arithmetic */
    SINK_I128(_mm_add_epi32(a, b));
    SINK_I128(_mm_sub_epi32(a, b));
    SINK_I128(_mm_add_epi32(max_i32, _mm_set1_epi32(1)));
    SINK_I128(_mm_sub_epi32(min_i32, _mm_set1_epi32(1)));

    /* 64-bit arithmetic */
    SINK_I128(_mm_add_epi64(a, b));
    SINK_I128(_mm_sub_epi64(a, b));

    /* Multiplication (can produce large results) */
    SINK_I128(_mm_mullo_epi16(a, b));
    SINK_I128(_mm_mulhi_epi16(a, b));
    SINK_I128(_mm_mulhi_epu16(a, b));
    SINK_I128(_mm_mullo_epi32(a, b));
    SINK_I128(_mm_mul_epi32(a, b));
    SINK_I128(_mm_mul_epu32(a, b));

    /* Multiply-add (madd) - can overflow */
    SINK_I128(_mm_madd_epi16(a, b));
    SINK_I128(_mm_maddubs_epi16(a, b));

    /* Horizontal operations */
    SINK_I128(_mm_hadd_epi16(a, b));
    SINK_I128(_mm_hadd_epi32(a, b));
    SINK_I128(_mm_hsub_epi16(a, b));
    SINK_I128(_mm_hsub_epi32(a, b));
    SINK_I128(_mm_hadds_epi16(a, b));
    SINK_I128(_mm_hsubs_epi16(a, b));

    /* Pack with saturation */
    SINK_I128(_mm_packs_epi16(a, b));
    SINK_I128(_mm_packs_epi32(a, b));
    SINK_I128(_mm_packus_epi16(a, b));
    SINK_I128(_mm_packus_epi32(a, b));

    /* SAD (sum of absolute differences) */
    SINK_I128(_mm_sad_epu8(a, b));

    /* Average (can have rounding edge cases) */
    SINK_I128(_mm_avg_epu8(a, b));
    SINK_I128(_mm_avg_epu16(a, b));
}

/* Strategy 4: Common SSE Instruction Robustness
 *
 * Tests frequently-used intrinsics with various input patterns:
 * - Comparison operations
 * - Logical operations
 * - Conversion operations
 * - Min/Max operations
 * - Movemask operations
 */
static void test_common_instructions(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    /* Integer vectors */
    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i b = _mm_loadu_si128((const __m128i *) (data + 16));

    /* Float vectors */
    __m128 fa = _mm_loadu_ps((const float *) data);
    __m128 fb = _mm_loadu_ps((const float *) (data + 16));

    /* Double vectors */
    __m128d da = _mm_loadu_pd((const double *) data);
    __m128d db = _mm_loadu_pd((const double *) (data + 16));

    /* Comparison operations - integer */
    SINK_I128(_mm_cmpeq_epi8(a, b));
    SINK_I128(_mm_cmpeq_epi16(a, b));
    SINK_I128(_mm_cmpeq_epi32(a, b));
    SINK_I128(_mm_cmpeq_epi64(a, b));
    SINK_I128(_mm_cmpgt_epi8(a, b));
    SINK_I128(_mm_cmpgt_epi16(a, b));
    SINK_I128(_mm_cmpgt_epi32(a, b));
    SINK_I128(_mm_cmpgt_epi64(a, b));
    SINK_I128(_mm_cmplt_epi8(a, b));
    SINK_I128(_mm_cmplt_epi16(a, b));
    SINK_I128(_mm_cmplt_epi32(a, b));

    /* Comparison operations - float */
    SINK_PS(_mm_cmpeq_ps(fa, fb));
    SINK_PS(_mm_cmplt_ps(fa, fb));
    SINK_PS(_mm_cmple_ps(fa, fb));
    SINK_PS(_mm_cmpgt_ps(fa, fb));
    SINK_PS(_mm_cmpge_ps(fa, fb));
    SINK_PS(_mm_cmpord_ps(fa, fb));
    SINK_PS(_mm_cmpunord_ps(fa, fb));

    /* Comparison operations - double */
    SINK_PD(_mm_cmpeq_pd(da, db));
    SINK_PD(_mm_cmplt_pd(da, db));
    SINK_PD(_mm_cmpord_pd(da, db));

    /* Logical operations */
    SINK_I128(_mm_and_si128(a, b));
    SINK_I128(_mm_or_si128(a, b));
    SINK_I128(_mm_xor_si128(a, b));
    SINK_I128(_mm_andnot_si128(a, b));
    SINK_PS(_mm_and_ps(fa, fb));
    SINK_PS(_mm_or_ps(fa, fb));
    SINK_PS(_mm_xor_ps(fa, fb));
    SINK_PD(_mm_and_pd(da, db));
    SINK_PD(_mm_or_pd(da, db));
    SINK_PD(_mm_xor_pd(da, db));

    /* Shift operations - variable count */
    __m128i shift_count = _mm_set_epi64x(0, get_u8(data, size, 0) % 64);
    SINK_I128(_mm_sll_epi16(a, shift_count));
    SINK_I128(_mm_sll_epi32(a, shift_count));
    SINK_I128(_mm_sll_epi64(a, shift_count));
    SINK_I128(_mm_srl_epi16(a, shift_count));
    SINK_I128(_mm_srl_epi32(a, shift_count));
    SINK_I128(_mm_srl_epi64(a, shift_count));
    SINK_I128(_mm_sra_epi16(a, shift_count));
    SINK_I128(_mm_sra_epi32(a, shift_count));

    /* Immediate shifts with boundary values */
#define TEST_SHIFT_IMM(OP, VEC, IMM) SINK_I128(OP(VEC, IMM))
    /* Valid boundary shift counts */
    TEST_SHIFT_IMM(_mm_slli_epi16, a, 0);
    TEST_SHIFT_IMM(_mm_slli_epi16, a, 15);
    TEST_SHIFT_IMM(_mm_slli_epi32, a, 0);
    TEST_SHIFT_IMM(_mm_slli_epi32, a, 31);
    TEST_SHIFT_IMM(_mm_slli_epi64, a, 0);
    TEST_SHIFT_IMM(_mm_slli_epi64, a, 63);
    TEST_SHIFT_IMM(_mm_srli_epi16, a, 0);
    TEST_SHIFT_IMM(_mm_srli_epi16, a, 15);
    TEST_SHIFT_IMM(_mm_srli_epi32, a, 0);
    TEST_SHIFT_IMM(_mm_srli_epi32, a, 31);
    TEST_SHIFT_IMM(_mm_srli_epi64, a, 0);
    TEST_SHIFT_IMM(_mm_srli_epi64, a, 63);
    TEST_SHIFT_IMM(_mm_srai_epi16, a, 0);
    TEST_SHIFT_IMM(_mm_srai_epi16, a, 15);
    TEST_SHIFT_IMM(_mm_srai_epi32, a, 0);
    TEST_SHIFT_IMM(_mm_srai_epi32, a, 31);

    /* Out-of-range shift counts (should zero result or fill with sign) */
    TEST_SHIFT_IMM(_mm_slli_epi16, a, 16);  /* Beyond 16-bit */
    TEST_SHIFT_IMM(_mm_slli_epi16, a, 255); /* Maximum immediate */
    TEST_SHIFT_IMM(_mm_slli_epi32, a, 32);  /* Beyond 32-bit */
    TEST_SHIFT_IMM(_mm_slli_epi64, a, 64);  /* Beyond 64-bit */
    TEST_SHIFT_IMM(_mm_srli_epi16, a, 16);
    TEST_SHIFT_IMM(_mm_srli_epi32, a, 32);
    TEST_SHIFT_IMM(_mm_srli_epi64, a, 64);
    TEST_SHIFT_IMM(_mm_srai_epi16, a, 16);
    TEST_SHIFT_IMM(_mm_srai_epi32, a, 32);
#undef TEST_SHIFT_IMM

    /* Byte shift (slli_si128 / srli_si128) */
#define TEST_BYTE_SHIFT(OP, VEC, IMM) SINK_I128(OP(VEC, IMM))
    /* Valid range: 0-15 */
    TEST_BYTE_SHIFT(_mm_slli_si128, a, 0);
    TEST_BYTE_SHIFT(_mm_slli_si128, a, 8);
    TEST_BYTE_SHIFT(_mm_slli_si128, a, 15);
    TEST_BYTE_SHIFT(_mm_srli_si128, a, 0);
    TEST_BYTE_SHIFT(_mm_srli_si128, a, 8);
    TEST_BYTE_SHIFT(_mm_srli_si128, a, 15);
    /* Out-of-range (>= 16 should zero entire vector) */
    TEST_BYTE_SHIFT(_mm_slli_si128, a, 16);
    TEST_BYTE_SHIFT(_mm_slli_si128, a, 255);
    TEST_BYTE_SHIFT(_mm_srli_si128, a, 16);
    TEST_BYTE_SHIFT(_mm_srli_si128, a, 255);
#undef TEST_BYTE_SHIFT

    /* Min/Max operations */
    SINK_I128(_mm_min_epi8(a, b));
    SINK_I128(_mm_max_epi8(a, b));
    SINK_I128(_mm_min_epu8(a, b));
    SINK_I128(_mm_max_epu8(a, b));
    SINK_I128(_mm_min_epi16(a, b));
    SINK_I128(_mm_max_epi16(a, b));
    SINK_I128(_mm_min_epu16(a, b));
    SINK_I128(_mm_max_epu16(a, b));
    SINK_I128(_mm_min_epi32(a, b));
    SINK_I128(_mm_max_epi32(a, b));
    SINK_I128(_mm_min_epu32(a, b));
    SINK_I128(_mm_max_epu32(a, b));
    SINK_PS(_mm_min_ps(fa, fb));
    SINK_PS(_mm_max_ps(fa, fb));
    SINK_PD(_mm_min_pd(da, db));
    SINK_PD(_mm_max_pd(da, db));

    /* Movemask operations */
    SINK_I32(_mm_movemask_epi8(a));
    SINK_I32(_mm_movemask_ps(fa));
    SINK_I32(_mm_movemask_pd(da));

    /* Conversion operations */
    SINK_PS(_mm_cvtepi32_ps(a));
    SINK_I128(_mm_cvtps_epi32(fa));
    SINK_I128(_mm_cvttps_epi32(fa));
    SINK_PD(_mm_cvtepi32_pd(a));
    SINK_PD(_mm_cvtps_pd(fa));
    SINK_PS(_mm_cvtpd_ps(da));

    /* Sign extension operations */
    SINK_I128(_mm_cvtepi8_epi16(a));
    SINK_I128(_mm_cvtepi8_epi32(a));
    SINK_I128(_mm_cvtepi8_epi64(a));
    SINK_I128(_mm_cvtepi16_epi32(a));
    SINK_I128(_mm_cvtepi16_epi64(a));
    SINK_I128(_mm_cvtepi32_epi64(a));
    SINK_I128(_mm_cvtepu8_epi16(a));
    SINK_I128(_mm_cvtepu8_epi32(a));
    SINK_I128(_mm_cvtepu8_epi64(a));
    SINK_I128(_mm_cvtepu16_epi32(a));
    SINK_I128(_mm_cvtepu16_epi64(a));
    SINK_I128(_mm_cvtepu32_epi64(a));

    /* Unpack operations */
    SINK_I128(_mm_unpacklo_epi8(a, b));
    SINK_I128(_mm_unpackhi_epi8(a, b));
    SINK_I128(_mm_unpacklo_epi16(a, b));
    SINK_I128(_mm_unpackhi_epi16(a, b));
    SINK_I128(_mm_unpacklo_epi32(a, b));
    SINK_I128(_mm_unpackhi_epi32(a, b));
    SINK_I128(_mm_unpacklo_epi64(a, b));
    SINK_I128(_mm_unpackhi_epi64(a, b));

    /* Blend operations - variable mask */
    SINK_I128(_mm_blendv_epi8(a, b, a));
    SINK_PS(_mm_blendv_ps(fa, fb, fa));
    SINK_PD(_mm_blendv_pd(da, db, da));

    /* Blend operations - immediate mask (SSE4.1) */
    SINK_I128(_mm_blend_epi16(a, b, 0x00));
    SINK_I128(_mm_blend_epi16(a, b, 0xAA));
    SINK_I128(_mm_blend_epi16(a, b, 0xFF));
    SINK_PS(_mm_blend_ps(fa, fb, 0x5));
    SINK_PS(_mm_blend_ps(fa, fb, 0xA));
    SINK_PD(_mm_blend_pd(da, db, 0x1));
    SINK_PD(_mm_blend_pd(da, db, 0x2));

    /* Insert float with immediate control (SSE4.1) */
    SINK_PS(_mm_insert_ps(fa, fb, 0x00));
    SINK_PS(_mm_insert_ps(fa, fb, 0x1D));
    SINK_PS(_mm_insert_ps(fa, fb, 0xF0));

    /* Test operations */
    SINK_I32(_mm_testz_si128(a, b));
    SINK_I32(_mm_testc_si128(a, b));
    SINK_I32(_mm_testnzc_si128(a, b));
}

/* Strategy 5: Floating-Point Edge Cases
 *
 * Tests float/double intrinsics with special values:
 * - NaN, Inf, -Inf
 * - Denormals
 * - Zero (positive and negative)
 * - Values near limits
 */
static void test_float_edge_cases(const uint8_t *data, size_t size)
{
    if (size < 16)
        return;

    /* Load fuzz data as floats */
    __m128 fa = _mm_loadu_ps((const float *) data);
    __m128d da = _mm_loadu_pd((const double *) data);

    /* Create special value vectors */
    __m128 nan_ps = _mm_set1_ps(__builtin_nanf(""));
    __m128 inf_ps = _mm_set1_ps(__builtin_inff());
    __m128 ninf_ps = _mm_set1_ps(-__builtin_inff());
    __m128 zero_ps = _mm_setzero_ps();
    __m128 nzero_ps = _mm_set1_ps(-0.0f);

    __m128d nan_pd = _mm_set1_pd(__builtin_nan(""));

    /* Arithmetic with special values */
    SINK_PS(_mm_add_ps(fa, nan_ps));
    SINK_PS(_mm_add_ps(fa, inf_ps));
    SINK_PS(_mm_sub_ps(fa, inf_ps));
    SINK_PS(_mm_mul_ps(fa, inf_ps));
    SINK_PS(_mm_div_ps(fa, zero_ps));
    SINK_PS(_mm_div_ps(fa, nzero_ps));

    /* Comparisons with NaN */
    SINK_PS(_mm_cmpeq_ps(fa, nan_ps));
    SINK_PS(_mm_cmpord_ps(fa, nan_ps));
    SINK_PS(_mm_cmpunord_ps(fa, nan_ps));

    /* Min/Max with NaN (SSE has specific NaN propagation rules) */
    SINK_PS(_mm_min_ps(fa, nan_ps));
    SINK_PS(_mm_max_ps(fa, nan_ps));
    SINK_PS(_mm_min_ps(fa, inf_ps));
    SINK_PS(_mm_max_ps(fa, inf_ps));

    /* Square root edge cases */
    SINK_PS(_mm_sqrt_ps(_mm_set1_ps(-1.0f))); /* Should produce NaN */
    SINK_PS(_mm_sqrt_ps(zero_ps));
    SINK_PS(_mm_sqrt_ps(nzero_ps));
    SINK_PS(_mm_sqrt_ps(inf_ps));

    /* Reciprocal and rsqrt edge cases */
    SINK_PS(_mm_rcp_ps(zero_ps));
    SINK_PS(_mm_rcp_ps(nzero_ps));
    SINK_PS(_mm_rcp_ps(inf_ps));
    SINK_PS(_mm_rsqrt_ps(zero_ps));
    SINK_PS(_mm_rsqrt_ps(_mm_set1_ps(-1.0f)));

    /* Double precision edge cases */
    SINK_PD(_mm_add_pd(da, nan_pd));
    SINK_PD(_mm_sqrt_pd(_mm_set1_pd(-1.0)));
    SINK_PD(_mm_min_pd(da, nan_pd));
    SINK_PD(_mm_max_pd(da, nan_pd));

    /* Rounding operations */
    SINK_PS(_mm_floor_ps(fa));
    SINK_PS(_mm_ceil_ps(fa));
    SINK_PD(_mm_floor_pd(da));
    SINK_PD(_mm_ceil_pd(da));

    /* Round with special values */
    SINK_PS(_mm_floor_ps(inf_ps));
    SINK_PS(_mm_ceil_ps(ninf_ps));
    SINK_PS(_mm_floor_ps(nan_ps));
}

/* Strategy 6: Extract/Insert Operations
 *
 * Tests extract and insert intrinsics with boundary indices.
 */
static void test_extract_insert(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128 fa = _mm_loadu_ps((const float *) data);

    /* Extract operations with various indices */
#define TEST_EXTRACT_EPI8(IDX) SINK_I32(_mm_extract_epi8(a, IDX))
#define TEST_EXTRACT_EPI16(IDX) SINK_I32(_mm_extract_epi16(a, IDX))
#define TEST_EXTRACT_EPI32(IDX) SINK_I32(_mm_extract_epi32(a, IDX))
#define TEST_EXTRACT_PS(IDX) SINK_I32(_mm_extract_ps(fa, IDX))

    /* Test boundary indices */
    TEST_EXTRACT_EPI8(0);
    TEST_EXTRACT_EPI8(7);
    TEST_EXTRACT_EPI8(15);
    TEST_EXTRACT_EPI16(0);
    TEST_EXTRACT_EPI16(3);
    TEST_EXTRACT_EPI16(7);
    TEST_EXTRACT_EPI32(0);
    TEST_EXTRACT_EPI32(1);
    TEST_EXTRACT_EPI32(2);
    TEST_EXTRACT_EPI32(3);
    TEST_EXTRACT_PS(0);
    TEST_EXTRACT_PS(1);
    TEST_EXTRACT_PS(2);
    TEST_EXTRACT_PS(3);

#undef TEST_EXTRACT_EPI8
#undef TEST_EXTRACT_EPI16
#undef TEST_EXTRACT_EPI32
#undef TEST_EXTRACT_PS

    /* Insert operations */
    int32_t val32 = get_i32(data, size, 16);
    int16_t val16 = (int16_t) get_u32(data, size, 20);
    int8_t val8 = (int8_t) get_u8(data, size, 24);

#define TEST_INSERT_EPI8(IDX) SINK_I128(_mm_insert_epi8(a, val8, IDX))
#define TEST_INSERT_EPI16(IDX) SINK_I128(_mm_insert_epi16(a, val16, IDX))
#define TEST_INSERT_EPI32(IDX) SINK_I128(_mm_insert_epi32(a, val32, IDX))

    TEST_INSERT_EPI8(0);
    TEST_INSERT_EPI8(7);
    TEST_INSERT_EPI8(15);
    TEST_INSERT_EPI16(0);
    TEST_INSERT_EPI16(3);
    TEST_INSERT_EPI16(7);
    TEST_INSERT_EPI32(0);
    TEST_INSERT_EPI32(1);
    TEST_INSERT_EPI32(2);
    TEST_INSERT_EPI32(3);

#undef TEST_INSERT_EPI8
#undef TEST_INSERT_EPI16
#undef TEST_INSERT_EPI32
}

/* Strategy 7: CRC32 Operations
 *
 * Tests CRC32 intrinsics with various input patterns.
 */
static void test_crc32(const uint8_t *data, size_t size)
{
    if (size < 16)
        return;

    uint32_t crc = get_u32(data, size, 0);

    /* CRC32 with various data widths */
    for (size_t i = 4; i < size; i++) {
        crc = _mm_crc32_u8(crc, data[i]);
    }

    for (size_t i = 4; i + 2 <= size; i += 2) {
        uint16_t v16;
        memcpy(&v16, data + i, 2);
        crc = _mm_crc32_u16(crc, v16);
    }

    for (size_t i = 4; i + 4 <= size; i += 4) {
        uint32_t v32;
        memcpy(&v32, data + i, 4);
        crc = _mm_crc32_u32(crc, v32);
    }

#if defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64)
    for (size_t i = 4; i + 8 <= size; i += 8) {
        uint64_t v64;
        memcpy(&v64, data + i, 8);
        crc = (uint32_t) _mm_crc32_u64(crc, v64);
    }
#endif

    /* Sink CRC result */
    SINK_I32(crc);
}

/* Strategy 8: ABS and Sign Operations
 *
 * Tests absolute value and sign manipulation intrinsics.
 */
static void test_abs_sign(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i b = _mm_loadu_si128((const __m128i *) (data + 16));

    /* Absolute value operations */
    SINK_I128(_mm_abs_epi8(a));
    SINK_I128(_mm_abs_epi16(a));
    SINK_I128(_mm_abs_epi32(a));

    /* Test with INT_MIN (abs of INT_MIN is still INT_MIN due to overflow) */
    __m128i min8 = _mm_set1_epi8(INT8_MIN);
    __m128i min16 = _mm_set1_epi16(INT16_MIN);
    __m128i min32 = _mm_set1_epi32(INT32_MIN);
    SINK_I128(_mm_abs_epi8(min8));
    SINK_I128(_mm_abs_epi16(min16));
    SINK_I128(_mm_abs_epi32(min32));

    /* Sign operations */
    SINK_I128(_mm_sign_epi8(a, b));
    SINK_I128(_mm_sign_epi16(a, b));
    SINK_I128(_mm_sign_epi32(a, b));

    /* Sign with zero mask (should zero elements where mask is zero) */
    __m128i zero = _mm_setzero_si128();
    SINK_I128(_mm_sign_epi8(a, zero));
    SINK_I128(_mm_sign_epi16(a, zero));
    SINK_I128(_mm_sign_epi32(a, zero));
}

/* Strategy 9: Scalar Floating-Point Operations
 *
 * Tests scalar operations that must preserve upper lanes.
 * Common bug: Overwriting upper 3 lanes instead of preserving them.
 */
static void test_scalar_float(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128 a = _mm_loadu_ps((const float *) data);
    __m128 b = _mm_loadu_ps((const float *) (data + 16));
    __m128d da = _mm_loadu_pd((const double *) data);
    __m128d db = _mm_loadu_pd((const double *) (data + 16));

    /* Scalar arithmetic - must preserve upper lanes */
    SINK_PS(_mm_add_ss(a, b));
    SINK_PS(_mm_sub_ss(a, b));
    SINK_PS(_mm_mul_ss(a, b));
    SINK_PS(_mm_div_ss(a, b));
    SINK_PS(_mm_min_ss(a, b));
    SINK_PS(_mm_max_ss(a, b));
    SINK_PS(_mm_sqrt_ss(a));
    SINK_PS(_mm_rcp_ss(a));
    SINK_PS(_mm_rsqrt_ss(a));

    /* Double scalar operations */
    SINK_PD(_mm_add_sd(da, db));
    SINK_PD(_mm_sub_sd(da, db));
    SINK_PD(_mm_mul_sd(da, db));
    SINK_PD(_mm_div_sd(da, db));
    SINK_PD(_mm_min_sd(da, db));
    SINK_PD(_mm_max_sd(da, db));
    SINK_PD(_mm_sqrt_sd(da, db));

    /* Scalar comparisons */
    SINK_PS(_mm_cmpeq_ss(a, b));
    SINK_PS(_mm_cmplt_ss(a, b));
    SINK_PS(_mm_cmple_ss(a, b));
    SINK_PS(_mm_cmpgt_ss(a, b));
    SINK_PS(_mm_cmpge_ss(a, b));
    SINK_PS(_mm_cmpord_ss(a, b));
    SINK_PS(_mm_cmpunord_ss(a, b));

    /* move_ss: Scalar move preserving upper lanes */
    SINK_PS(_mm_move_ss(a, b));
}

/* Strategy 10: NaN Operand Order Testing
 *
 * x86 and ARM handle min/max with NaN differently based on operand order.
 * This tests SSE2NEON_PRECISE_MINMAX behavior.
 */
static void test_nan_operand_order(const uint8_t *data, size_t size)
{
    if (size < 16)
        return;

    __m128 fa = _mm_loadu_ps((const float *) data);
    __m128d da = _mm_loadu_pd((const double *) data);

    __m128 nan_ps = _mm_set1_ps(__builtin_nanf(""));
    __m128d nan_pd = _mm_set1_pd(__builtin_nan(""));

    /* Test NaN in first operand vs second operand
     * SSE returns second operand when first is NaN */
    SINK_PS(_mm_min_ps(nan_ps, fa));
    SINK_PS(_mm_min_ps(fa, nan_ps));
    SINK_PS(_mm_max_ps(nan_ps, fa));
    SINK_PS(_mm_max_ps(fa, nan_ps));

    /* Scalar versions */
    SINK_PS(_mm_min_ss(nan_ps, fa));
    SINK_PS(_mm_min_ss(fa, nan_ps));
    SINK_PS(_mm_max_ss(nan_ps, fa));
    SINK_PS(_mm_max_ss(fa, nan_ps));

    /* Double precision */
    SINK_PD(_mm_min_pd(nan_pd, da));
    SINK_PD(_mm_min_pd(da, nan_pd));
    SINK_PD(_mm_max_pd(nan_pd, da));
    SINK_PD(_mm_max_pd(da, nan_pd));

    /* Both operands NaN */
    SINK_PS(_mm_min_ps(nan_ps, nan_ps));
    SINK_PS(_mm_max_ps(nan_ps, nan_ps));
}

/* Strategy 11: Ordered vs Unordered Comparisons
 *
 * Tests comieq/ucomieq differences in NaN handling and flag setting.
 */
static void test_ordered_unordered_cmp(const uint8_t *data, size_t size)
{
    if (size < 16)
        return;

    __m128 a = _mm_loadu_ps((const float *) data);
    __m128 b = _mm_loadu_ps((const float *) (data + (size >= 32 ? 16 : 0)));
    __m128d da = _mm_loadu_pd((const double *) data);
    __m128d db = _mm_loadu_pd((const double *) (data + (size >= 32 ? 16 : 0)));

    __m128 nan_ps = _mm_set1_ps(__builtin_nanf(""));
    __m128d nan_pd = _mm_set1_pd(__builtin_nan(""));

    /* Ordered comparisons (signal on QNaN) */
    SINK_I32(_mm_comieq_ss(a, b));
    SINK_I32(_mm_comilt_ss(a, b));
    SINK_I32(_mm_comile_ss(a, b));
    SINK_I32(_mm_comigt_ss(a, b));
    SINK_I32(_mm_comige_ss(a, b));
    SINK_I32(_mm_comineq_ss(a, b));

    /* Unordered comparisons (quiet on QNaN) */
    SINK_I32(_mm_ucomieq_ss(a, b));
    SINK_I32(_mm_ucomilt_ss(a, b));
    SINK_I32(_mm_ucomile_ss(a, b));
    SINK_I32(_mm_ucomigt_ss(a, b));
    SINK_I32(_mm_ucomige_ss(a, b));
    SINK_I32(_mm_ucomineq_ss(a, b));

    /* Compare with NaN */
    SINK_I32(_mm_comieq_ss(a, nan_ps));
    SINK_I32(_mm_ucomieq_ss(a, nan_ps));
    SINK_I32(_mm_comilt_ss(nan_ps, a));
    SINK_I32(_mm_ucomilt_ss(nan_ps, a));

    /* Double precision */
    SINK_I32(_mm_comieq_sd(da, db));
    SINK_I32(_mm_ucomieq_sd(da, db));
    SINK_I32(_mm_comieq_sd(da, nan_pd));
    SINK_I32(_mm_ucomieq_sd(da, nan_pd));
}

/* Strategy 12: Dot Product Operations
 *
 * Tests _mm_dp_ps/_mm_dp_pd with various mask combinations.
 * The mask controls which elements are multiplied and where results go.
 */
static void test_dot_product(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128 a = _mm_loadu_ps((const float *) data);
    __m128 b = _mm_loadu_ps((const float *) (data + 16));
    __m128d da = _mm_loadu_pd((const double *) data);
    __m128d db = _mm_loadu_pd((const double *) (data + 16));

    /* Test various mask combinations for dp_ps */
#define TEST_DP_PS(MASK) SINK_PS(_mm_dp_ps(a, b, MASK))

    TEST_DP_PS(0xFF); /* All elements, broadcast to all */
    TEST_DP_PS(0xF1); /* All elements, result in lane 0 only */
    TEST_DP_PS(0x7F); /* First 3 elements, broadcast to all */
    TEST_DP_PS(0x31); /* Elements 0,1 only, result in lane 0 */
    TEST_DP_PS(0x11); /* Element 0 only */
    TEST_DP_PS(0x00); /* No elements (edge case) */
#undef TEST_DP_PS

    /* Test dp_pd */
#define TEST_DP_PD(MASK) SINK_PD(_mm_dp_pd(da, db, MASK))

    TEST_DP_PD(0xFF);
    TEST_DP_PD(0x31);
    TEST_DP_PD(0x11);
#undef TEST_DP_PD

    /* Dot product with special values */
    __m128 nan_ps = _mm_set1_ps(__builtin_nanf(""));
    __m128 inf_ps = _mm_set1_ps(__builtin_inff());
    __m128 zero_ps = _mm_setzero_ps();

    SINK_PS(_mm_dp_ps(a, nan_ps, 0xFF));
    SINK_PS(_mm_dp_ps(a, inf_ps, 0xFF));
    SINK_PS(_mm_dp_ps(a, zero_ps, 0xFF));
}

/* Strategy 13: Double Precision Arithmetic
 *
 * Expanded coverage for __m128d operations.
 */
static void test_double_precision(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128d a = _mm_loadu_pd((const double *) data);
    __m128d b = _mm_loadu_pd((const double *) (data + 16));

    /* Basic arithmetic (SSE2) */
    SINK_PD(_mm_add_pd(a, b));
    SINK_PD(_mm_sub_pd(a, b));
    SINK_PD(_mm_mul_pd(a, b));
    SINK_PD(_mm_div_pd(a, b));
    SINK_PD(_mm_sqrt_pd(a));

    /* Shuffle */
#define TEST_SHUFFLE_PD(IMM) SINK_PD(_mm_shuffle_pd(a, b, IMM))

    TEST_SHUFFLE_PD(0x0);
    TEST_SHUFFLE_PD(0x1);
    TEST_SHUFFLE_PD(0x2);
    TEST_SHUFFLE_PD(0x3);
#undef TEST_SHUFFLE_PD

    /* Unpack */
    SINK_PD(_mm_unpacklo_pd(a, b));
    SINK_PD(_mm_unpackhi_pd(a, b));

    /* Move operations (SSE2) */
    SINK_PD(_mm_move_sd(a, b));
    /* Note: _mm_movedup_pd is SSE3, tested in test_sse3_ops */

    /* All comparisons (SSE2) */
    SINK_PD(_mm_cmpeq_pd(a, b));
    SINK_PD(_mm_cmplt_pd(a, b));
    SINK_PD(_mm_cmple_pd(a, b));
    SINK_PD(_mm_cmpgt_pd(a, b));
    SINK_PD(_mm_cmpge_pd(a, b));
    SINK_PD(_mm_cmpneq_pd(a, b));
    SINK_PD(_mm_cmpnlt_pd(a, b));
    SINK_PD(_mm_cmpnle_pd(a, b));
    SINK_PD(_mm_cmpngt_pd(a, b));
    SINK_PD(_mm_cmpnge_pd(a, b));

    /* Scalar double comparisons (return vector, preserve upper lane) */
    SINK_PD(_mm_cmpeq_sd(a, b));
    SINK_PD(_mm_cmplt_sd(a, b));
    SINK_PD(_mm_cmple_sd(a, b));
    SINK_PD(_mm_cmpgt_sd(a, b));
    SINK_PD(_mm_cmpge_sd(a, b));
    SINK_PD(_mm_cmpneq_sd(a, b));
    SINK_PD(_mm_cmpord_sd(a, b));
    SINK_PD(_mm_cmpunord_sd(a, b));
    /* Note: SSE4.1 rounding tested in test_rounding_ops */
}

/* Strategy 14: Conversion Edge Cases
 *
 * Tests type conversions with boundary values where x86/ARM may differ.
 */
static void test_conversion_edge_cases(const uint8_t *data, size_t size)
{
    if (size < 16)
        return;

    __m128 fa = _mm_loadu_ps((const float *) data);
    __m128d da = _mm_loadu_pd((const double *) data);
    __m128i ia = _mm_loadu_si128((const __m128i *) data);

    /* Float to int conversions with edge cases */
    __m128 large_f = _mm_set1_ps(2147483648.0f);  /* INT32_MAX + 1 */
    __m128 small_f = _mm_set1_ps(-2147483904.0f); /* Below INT32_MIN */
    __m128 nan_f = _mm_set1_ps(__builtin_nanf(""));
    __m128 inf_f = _mm_set1_ps(__builtin_inff());

    /* cvt (round to nearest) vs cvtt (truncate) */
    SINK_I128(_mm_cvtps_epi32(fa));
    SINK_I128(_mm_cvttps_epi32(fa));
    SINK_I128(_mm_cvtps_epi32(large_f));
    SINK_I128(_mm_cvttps_epi32(large_f));
    SINK_I128(_mm_cvtps_epi32(small_f));
    SINK_I128(_mm_cvtps_epi32(nan_f));
    SINK_I128(_mm_cvtps_epi32(inf_f));

    /* Double to int conversions */
    __m128d large_d = _mm_set1_pd(2147483648.0);
    __m128d nan_d = _mm_set1_pd(__builtin_nan(""));

    SINK_I128(_mm_cvtpd_epi32(da));
    SINK_I128(_mm_cvttpd_epi32(da));
    SINK_I128(_mm_cvtpd_epi32(large_d));
    SINK_I128(_mm_cvtpd_epi32(nan_d));

    /* Int to float (may lose precision for large ints) */
    __m128i large_i = _mm_set1_epi32(0x7FFFFFFF);
    SINK_PS(_mm_cvtepi32_ps(large_i));
    SINK_PS(_mm_cvtepi32_ps(ia));

    /* Float <-> double precision changes */
    SINK_PS(_mm_cvtpd_ps(da));
    SINK_PD(_mm_cvtps_pd(fa));

    /* Scalar conversions - use clamped values to avoid UB
     * (out-of-range float-to-int conversion is undefined behavior)
     */
    float f_scalar;
    memcpy(&f_scalar, data, sizeof(float));
    /* Clamp to int32 representable range; also handle NaN/Inf */
    if (!(f_scalar >= -2147483520.0f && f_scalar <= 2147483520.0f))
        f_scalar = 0.0f;
    __m128 fa_safe = _mm_set_ss(f_scalar);

    double d_scalar;
    memcpy(&d_scalar, data, sizeof(double));
    if (!(d_scalar >= -2147483520.0 && d_scalar <= 2147483520.0))
        d_scalar = 0.0;
    __m128d da_safe = _mm_set_sd(d_scalar);

    SINK_I32(_mm_cvtss_si32(fa_safe));
    SINK_I32(_mm_cvttss_si32(fa_safe));
    SINK_I32(_mm_cvtsd_si32(da_safe));
    SINK_I32(_mm_cvttsd_si32(da_safe));

#if defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64)
    /* For 64-bit, use larger range but still need NaN/Inf check */
    memcpy(&f_scalar, data, sizeof(float));
    if (!(f_scalar >= -9223371487098961920.0f &&
          f_scalar <= 9223371487098961920.0f))
        f_scalar = 0.0f;
    fa_safe = _mm_set_ss(f_scalar);

    memcpy(&d_scalar, data, sizeof(double));
    if (!(d_scalar >= -9223372036854774784.0 &&
          d_scalar <= 9223372036854774784.0))
        d_scalar = 0.0;
    da_safe = _mm_set_sd(d_scalar);

    SINK_I64(_mm_cvtss_si64(fa_safe));
    SINK_I64(_mm_cvttss_si64(fa_safe));
    SINK_I64(_mm_cvtsd_si64(da_safe));
    SINK_I64(_mm_cvttsd_si64(da_safe));
#endif
}

/* Strategy 15: SSE3 Additional Operations
 *
 * Tests SSE3 intrinsics not covered elsewhere.
 */
static void test_sse3_ops(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128 fa = _mm_loadu_ps((const float *) data);
    __m128 fb = _mm_loadu_ps((const float *) (data + 16));
    __m128d da = _mm_loadu_pd((const double *) data);
    __m128d db = _mm_loadu_pd((const double *) (data + 16));

    /* Duplicate operations */
    SINK_PS(_mm_movehdup_ps(fa));
    SINK_PS(_mm_moveldup_ps(fa));
    SINK_PD(_mm_movedup_pd(da));

    /* Unaligned load (lddqu is optimized for misaligned access) */
    for (int offset = 0; offset < 16 && offset + 16 <= (int) size; offset++) {
        memcpy(unaligned_buf + offset, data, 16);
        SINK_I128(_mm_lddqu_si128((const __m128i *) (unaligned_buf + offset)));
    }

    /* Horizontal add/sub for floats (also in test_double_precision) */
    SINK_PS(_mm_hadd_ps(fa, fb));
    SINK_PS(_mm_hsub_ps(fa, fb));
    SINK_PS(_mm_addsub_ps(fa, fb));

    /* Test with special values */
    __m128 nan_ps = _mm_set1_ps(__builtin_nanf(""));
    SINK_PS(_mm_hadd_ps(fa, nan_ps));
    SINK_PS(_mm_hsub_ps(nan_ps, fa));

    /* Double precision horizontal ops */
    SINK_PD(_mm_hadd_pd(da, db));
    SINK_PD(_mm_hsub_pd(da, db));
    SINK_PD(_mm_addsub_pd(da, db));
}

/* Strategy 16: SSE4.2 String Comparison Operations
 *
 * Tests PCMPISTRI/PCMPISTRM/PCMPESTRI/PCMPESTRM string comparison intrinsics.
 * These are commonly used for string processing and pattern matching.
 */
static void test_sse42_string(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i b = _mm_loadu_si128((const __m128i *) (data + 16));

    /* Implicit length string comparisons (null-terminated) */
    /* _SIDD_UBYTE_OPS: unsigned byte comparison */
    /* _SIDD_CMP_EQUAL_ANY: find any matching character */
#define TEST_CMPISTR(MODE)                   \
    do {                                     \
        SINK_I32(_mm_cmpistri(a, b, MODE));  \
        SINK_I32(_mm_cmpistrc(a, b, MODE));  \
        SINK_I32(_mm_cmpistrs(a, b, MODE));  \
        SINK_I32(_mm_cmpistrz(a, b, MODE));  \
        SINK_I128(_mm_cmpistrm(a, b, MODE)); \
    } while (0)

    /* Test comparison modes with various data types */
    /* Unsigned byte operations */
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY);
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED);
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES);

    /* Signed byte operations */
    TEST_CMPISTR(_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY);
    TEST_CMPISTR(_SIDD_SBYTE_OPS | _SIDD_CMP_RANGES);

    /* Word operations (signed and unsigned) */
    TEST_CMPISTR(_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_ANY);
    TEST_CMPISTR(_SIDD_SWORD_OPS | _SIDD_CMP_EQUAL_EACH);
    TEST_CMPISTR(_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH);
    TEST_CMPISTR(_SIDD_UWORD_OPS | _SIDD_CMP_RANGES);

    /* Polarity and output control flags */
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
                 _SIDD_NEGATIVE_POLARITY);
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH |
                 _SIDD_MASKED_POSITIVE_POLARITY);
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH |
                 _SIDD_MASKED_NEGATIVE_POLARITY);
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
    TEST_CMPISTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_UNIT_MASK);
#undef TEST_CMPISTR

    /* Explicit length string comparisons */
    int la = get_u8(data, size, 0) % 17; /* Length 0-16 */
    int lb = get_u8(data, size, 1) % 17;

#define TEST_CMPESTR(MODE)                           \
    do {                                             \
        SINK_I32(_mm_cmpestri(a, la, b, lb, MODE));  \
        SINK_I32(_mm_cmpestrc(a, la, b, lb, MODE));  \
        SINK_I32(_mm_cmpestrs(a, la, b, lb, MODE));  \
        SINK_I32(_mm_cmpestrz(a, la, b, lb, MODE));  \
        SINK_I128(_mm_cmpestrm(a, la, b, lb, MODE)); \
    } while (0)

    /* Test comparison modes with various data types */
    TEST_CMPESTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY);
    TEST_CMPESTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH);
    TEST_CMPESTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED);
    TEST_CMPESTR(_SIDD_UBYTE_OPS | _SIDD_CMP_RANGES);
    TEST_CMPESTR(_SIDD_SBYTE_OPS | _SIDD_CMP_EQUAL_ANY);
    TEST_CMPESTR(_SIDD_SWORD_OPS | _SIDD_CMP_RANGES);
    TEST_CMPESTR(_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH);

    /* With polarity flags */
    TEST_CMPESTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
                 _SIDD_NEGATIVE_POLARITY);
    TEST_CMPESTR(_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_BIT_MASK);
#undef TEST_CMPESTR

    /* Edge cases: empty strings, full strings, negative lengths clamped */
    SINK_I32(_mm_cmpestri(a, 0, b, 0, _SIDD_UBYTE_OPS));
    SINK_I32(_mm_cmpestri(a, 16, b, 16, _SIDD_UBYTE_OPS));
    SINK_I32(
        _mm_cmpestri(a, -1, b, 5, _SIDD_UBYTE_OPS)); /* Negative length test */
}

/* Strategy 17: AES Encryption Operations
 *
 * Tests AES-NI intrinsics for encryption/decryption.
 * These are critical for cryptographic applications.
 */
static void test_aes_ops(const uint8_t *data, size_t size)
{
    if (size < 48)
        return;

    /* Use fuzz data as state and round keys */
    __m128i state = _mm_loadu_si128((const __m128i *) data);
    __m128i round_key1 = _mm_loadu_si128((const __m128i *) (data + 16));
    __m128i round_key2 = _mm_loadu_si128((const __m128i *) (data + 32));

    /* Single round AES encryption */
    __m128i enc1 = _mm_aesenc_si128(state, round_key1);
    SINK_I128(_mm_aesenc_si128(enc1, round_key2));

    /* Last round AES encryption (no MixColumns) */
    SINK_I128(_mm_aesenclast_si128(state, round_key1));

    /* Single round AES decryption */
    __m128i dec1 = _mm_aesdec_si128(state, round_key1);
    SINK_I128(_mm_aesdec_si128(dec1, round_key2));

    /* Last round AES decryption */
    SINK_I128(_mm_aesdeclast_si128(state, round_key1));

    /* Inverse MixColumns for decryption key expansion */
    SINK_I128(_mm_aesimc_si128(round_key1));

    /* Key generation assist with various rcon values */
#define TEST_KEYGEN(RCON) SINK_I128(_mm_aeskeygenassist_si128(round_key1, RCON))

    TEST_KEYGEN(0x01);
    TEST_KEYGEN(0x02);
    TEST_KEYGEN(0x04);
    TEST_KEYGEN(0x08);
    TEST_KEYGEN(0x10);
    TEST_KEYGEN(0x20);
    TEST_KEYGEN(0x40);
    TEST_KEYGEN(0x80);
    TEST_KEYGEN(0x1B);
    TEST_KEYGEN(0x36);
#undef TEST_KEYGEN

    /* Chain multiple rounds like real AES-128 */
    __m128i aes_state = state;
    aes_state = _mm_aesenc_si128(aes_state, round_key1);
    aes_state = _mm_aesenc_si128(aes_state, round_key2);
    aes_state = _mm_aesenc_si128(aes_state, round_key1);
    SINK_I128(_mm_aesenclast_si128(aes_state, round_key2));

    /* Decryption chain */
    __m128i dec_state = state;
    dec_state = _mm_aesdec_si128(dec_state, _mm_aesimc_si128(round_key1));
    dec_state = _mm_aesdec_si128(dec_state, _mm_aesimc_si128(round_key2));
    SINK_I128(_mm_aesdeclast_si128(dec_state, round_key1));
}

/* Strategy 18: CLMUL Carry-less Multiplication
 *
 * Tests PCLMULQDQ for carry-less (polynomial) multiplication.
 * Used in GCM mode, CRC calculations, and finite field arithmetic.
 */
static void test_clmul_ops(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i b = _mm_loadu_si128((const __m128i *) (data + 16));

    /* Carry-less multiply with different quadword selections
     * imm8[0] selects which 64-bit half of a
     * imm8[4] selects which 64-bit half of b */
#define TEST_CLMUL(IMM) SINK_I128(_mm_clmulepi64_si128(a, b, IMM))

    TEST_CLMUL(0x00); /* a[63:0] * b[63:0] */
    TEST_CLMUL(0x01); /* a[127:64] * b[63:0] */
    TEST_CLMUL(0x10); /* a[63:0] * b[127:64] */
    TEST_CLMUL(0x11); /* a[127:64] * b[127:64] */
#undef TEST_CLMUL

    /* Edge cases: multiply by zero, multiply by one */
    __m128i zero = _mm_setzero_si128();
    __m128i one = _mm_set_epi64x(0, 1);

    SINK_I128(_mm_clmulepi64_si128(a, zero, 0x00));
    SINK_I128(_mm_clmulepi64_si128(a, one, 0x00));
    SINK_I128(_mm_clmulepi64_si128(a, a, 0x00)); /* Square */

    /* High bits set (tests full polynomial multiplication) */
    __m128i high = _mm_set_epi64x((int64_t) 0x8000000000000000ULL,
                                  (int64_t) 0x8000000000000000ULL);
    SINK_I128(_mm_clmulepi64_si128(high, high, 0x00));
}

/* Strategy 19: Fuzz-Controlled Parameters
 *
 * Uses fuzz input to control operation parameters for better coverage.
 */
static void test_fuzz_controlled(const uint8_t *data, size_t size)
{
    if (size < 48)
        return;

    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i b = _mm_loadu_si128((const __m128i *) (data + 16));

    /* Fuzz-controlled shift amounts */
    uint8_t shift16 = get_u8(data, size, 32);
    uint8_t shift32 = get_u8(data, size, 33);
    uint8_t shift64 = get_u8(data, size, 34);

    __m128i count16 = _mm_set_epi64x(0, shift16);
    __m128i count32 = _mm_set_epi64x(0, shift32);
    __m128i count64 = _mm_set_epi64x(0, shift64);

    SINK_I128(_mm_sll_epi16(a, count16));
    SINK_I128(_mm_sll_epi32(a, count32));
    SINK_I128(_mm_sll_epi64(a, count64));
    SINK_I128(_mm_srl_epi16(a, count16));
    SINK_I128(_mm_srl_epi32(a, count32));
    SINK_I128(_mm_srl_epi64(a, count64));
    SINK_I128(_mm_sra_epi16(a, count16));
    SINK_I128(_mm_sra_epi32(a, count32));

    /* Fuzz-controlled alignment offset */
    uint8_t align_offset = get_u8(data, size, 35) % 16;
    memcpy(unaligned_buf + align_offset, data, 16);
    SINK_I128(
        _mm_loadu_si128((const __m128i *) (unaligned_buf + align_offset)));

    /* Fuzz-controlled blend mask */
    __m128i mask = _mm_loadu_si128((const __m128i *) (data + 32));
    SINK_I128(_mm_blendv_epi8(a, b, mask));

    /* Fuzz-controlled mpsadbw block offsets */
#define TEST_MPSADBW(IMM) SINK_I128(_mm_mpsadbw_epu8(a, b, IMM))

    TEST_MPSADBW(0x00);
    TEST_MPSADBW(0x05);
    TEST_MPSADBW(0x22);
    TEST_MPSADBW(0x37);
#undef TEST_MPSADBW
}

/* Strategy 20: Streaming Store Operations
 *
 * Tests non-temporal (streaming) store intrinsics.
 * These bypass the cache and are critical for memory subsystem testing.
 */
static void test_streaming_stores(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128i ia = _mm_loadu_si128((const __m128i *) data);
    __m128 fa = _mm_loadu_ps((const float *) data);
    __m128d da = _mm_loadu_pd((const double *) data);

    /* Non-temporal stores to aligned buffer */
    _mm_stream_si128((__m128i *) aligned_buf, ia);
    _mm_stream_ps((float *) aligned_buf, fa);
    _mm_stream_pd((double *) aligned_buf, da);

    /* Memory fence after streaming stores */
    _mm_sfence();

    /* Verify data can be read back */
    SINK_I128(_mm_load_si128((const __m128i *) aligned_buf));
    SINK_PS(_mm_load_ps((const float *) aligned_buf));
    SINK_PD(_mm_load_pd((const double *) aligned_buf));

    /* Stream load (SSE4.1) */
    SINK_I128(_mm_stream_load_si128((__m128i *) aligned_buf));
}

/* Strategy 21: SSE4.1 Rounding Operations
 *
 * Tests explicit rounding control intrinsics with all rounding modes:
 * - _MM_FROUND_TO_NEAREST_INT (round to nearest)
 * - _MM_FROUND_TO_NEG_INF (round down/floor)
 * - _MM_FROUND_TO_POS_INF (round up/ceil)
 * - _MM_FROUND_TO_ZERO (truncate)
 */
static void test_rounding_ops(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128 fa = _mm_loadu_ps((const float *) data);
    __m128 fb = _mm_loadu_ps((const float *) (data + 16));
    __m128d da = _mm_loadu_pd((const double *) data);
    __m128d db = _mm_loadu_pd((const double *) (data + 16));

    /* Packed single-precision rounding */
    SINK_PS(_mm_round_ps(fa, _MM_FROUND_TO_NEAREST_INT));
    SINK_PS(_mm_round_ps(fa, _MM_FROUND_TO_NEG_INF));
    SINK_PS(_mm_round_ps(fa, _MM_FROUND_TO_POS_INF));
    SINK_PS(_mm_round_ps(fa, _MM_FROUND_TO_ZERO));

    /* With NO_EXC flag (suppress exceptions) */
    SINK_PS(_mm_round_ps(fa, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    /* Packed double-precision rounding */
    SINK_PD(_mm_round_pd(da, _MM_FROUND_TO_NEAREST_INT));
    SINK_PD(_mm_round_pd(da, _MM_FROUND_TO_NEG_INF));
    SINK_PD(_mm_round_pd(da, _MM_FROUND_TO_POS_INF));
    SINK_PD(_mm_round_pd(da, _MM_FROUND_TO_ZERO));

    /* Scalar rounding (preserves upper lanes) */
    SINK_PS(_mm_round_ss(fa, fb, _MM_FROUND_TO_NEAREST_INT));
    SINK_PS(_mm_round_ss(fa, fb, _MM_FROUND_TO_NEG_INF));
    SINK_PD(_mm_round_sd(da, db, _MM_FROUND_TO_NEAREST_INT));
    SINK_PD(_mm_round_sd(da, db, _MM_FROUND_TO_NEG_INF));

    /* Edge cases: NaN, Inf */
    __m128 nan_ps = _mm_set1_ps(__builtin_nanf(""));
    __m128 inf_ps = _mm_set1_ps(__builtin_inff());
    SINK_PS(_mm_round_ps(nan_ps, _MM_FROUND_TO_NEAREST_INT));
    SINK_PS(_mm_round_ps(inf_ps, _MM_FROUND_TO_NEAREST_INT));
}

/* Strategy 22: Masked Move Operations
 *
 * Tests _mm_maskmoveu_si128 which selectively stores bytes based on mask.
 * High-risk operation for memory safety.
 */
static void test_masked_moves(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i mask = _mm_loadu_si128((const __m128i *) (data + 16));

    /* Clear destination buffer */
    memset(unaligned_buf, 0, 32);

    /* Masked store: only bytes with high bit set in mask are written */
    _mm_maskmoveu_si128(a, mask, (char *) unaligned_buf);

    /* Test with various mask patterns */
    __m128i all_ones = _mm_set1_epi8((char) 0xFF);
    __m128i all_zeros = _mm_setzero_si128();
    __m128i alternating = _mm_set_epi8(
        (char) 0x80, 0, (char) 0x80, 0, (char) 0x80, 0, (char) 0x80, 0,
        (char) 0x80, 0, (char) 0x80, 0, (char) 0x80, 0, (char) 0x80, 0);

    _mm_maskmoveu_si128(a, all_ones, (char *) (unaligned_buf + 16));
    _mm_maskmoveu_si128(a, all_zeros, (char *) unaligned_buf);
    _mm_maskmoveu_si128(a, alternating, (char *) unaligned_buf);

    /* Verify buffer is modified */
    SINK_I128(_mm_loadu_si128((const __m128i *) unaligned_buf));
}

/* Strategy 23: Advanced Integer Operations
 *
 * Tests SSSE3/SSE4.1 advanced integer intrinsics:
 * - _mm_minpos_epu16: horizontal minimum with position
 * - _mm_mulhrs_epi16: high-rounding multiply
 * - _mm_packus_epi32: 32-to-16 unsigned packing
 * - 64-bit insert/extract
 */
static void test_advanced_integer(const uint8_t *data, size_t size)
{
    if (size < 32)
        return;

    __m128i a = _mm_loadu_si128((const __m128i *) data);
    __m128i b = _mm_loadu_si128((const __m128i *) (data + 16));

    /* Horizontal minimum with position (SSE4.1) */
    SINK_I128(_mm_minpos_epu16(a));

    /* High-rounding multiply (SSSE3) */
    SINK_I128(_mm_mulhrs_epi16(a, b));

    /* 32-to-16 unsigned pack (SSE4.1) */
    SINK_I128(_mm_packus_epi32(a, b));

    /* 64-bit extract/insert (SSE4.1, 64-bit only) */
#if defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64)
    SINK_I64(_mm_extract_epi64(a, 0));
    SINK_I64(_mm_extract_epi64(a, 1));

    int64_t insert_val =
        get_u32(data, size, 0) | ((int64_t) get_u32(data, size, 4) << 32);
    SINK_I128(_mm_insert_epi64(a, insert_val, 0));
    SINK_I128(_mm_insert_epi64(a, insert_val, 1));
#endif

    /* Edge cases for minpos */
    __m128i all_max = _mm_set1_epi16((short) 0xFFFF);
    __m128i all_zero = _mm_setzero_si128();
    SINK_I128(_mm_minpos_epu16(all_max));
    SINK_I128(_mm_minpos_epu16(all_zero));

    /* mulhrs edge cases */
    __m128i max16 = _mm_set1_epi16((short) 0x7FFF);
    __m128i min16 = _mm_set1_epi16((short) 0x8000);
    SINK_I128(_mm_mulhrs_epi16(max16, max16));
    SINK_I128(_mm_mulhrs_epi16(min16, min16));
    SINK_I128(_mm_mulhrs_epi16(max16, min16));
}

/* Strategy 24: Scalar Upper Lane Preservation
 *
 * UB-Invariant check: Scalar operations (_mm_add_ss, etc.) must preserve
 * upper lanes regardless of input values (NaN, Inf, denormals, etc.).
 * This is a semantic requirement from Intel that must hold for ANY input.
 */
static void test_scalar_lane_preservation(const uint8_t *data, size_t size)
{
    if (size < 4)
        return;

    /* Use fuzzed data for the scalar value being operated on */
    float fuzz_val;
    memcpy(&fuzz_val, data, sizeof(float));

    /* Sentinel values for upper lanes - must be preserved */
    const float sentinel1 = 1.5f;
    const float sentinel2 = 2.5f;
    const float sentinel3 = 3.5f;
    const double d_sentinel = 7.5;

    /* Build vector with fuzzed lower lane, known upper lanes */
    alignas(16) float vec_a[4] = {fuzz_val, sentinel1, sentinel2, sentinel3};
    alignas(16) float vec_b[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    __m128 a = _mm_load_ps(vec_a);
    __m128 b = _mm_load_ps(vec_b);

    /* Macro to verify upper lanes preserved after scalar op */
#define CHECK_SCALAR_PS(op)                                          \
    do {                                                             \
        __m128 result = op;                                          \
        alignas(16) float out[4];                                    \
        _mm_store_ps(out, result);                                   \
        /* Upper 3 lanes must equal original 'a' upper lanes */      \
        if (out[1] != sentinel1 || out[2] != sentinel2 ||            \
            out[3] != sentinel3) {                                   \
            /* Force observable side effect on failure */            \
            volatile int *null_ptr = NULL;                           \
            *null_ptr = 1; /* Crash to signal invariant violation */ \
        }                                                            \
        SINK_PS(result);                                             \
    } while (0)

    /* Test all scalar float operations */
    CHECK_SCALAR_PS(_mm_add_ss(a, b));
    CHECK_SCALAR_PS(_mm_sub_ss(a, b));
    CHECK_SCALAR_PS(_mm_mul_ss(a, b));
    CHECK_SCALAR_PS(_mm_min_ss(a, b));
    CHECK_SCALAR_PS(_mm_max_ss(a, b));
    CHECK_SCALAR_PS(_mm_sqrt_ss(a));
    CHECK_SCALAR_PS(_mm_rcp_ss(a));
    CHECK_SCALAR_PS(_mm_rsqrt_ss(a));
    CHECK_SCALAR_PS(_mm_cmpeq_ss(a, b));
    CHECK_SCALAR_PS(_mm_cmplt_ss(a, b));
    CHECK_SCALAR_PS(_mm_cmple_ss(a, b));
    CHECK_SCALAR_PS(_mm_move_ss(a, b));
    CHECK_SCALAR_PS(_mm_div_ss(a, b)); /* vec_b[0]=1.0f, no div-by-zero */

#undef CHECK_SCALAR_PS

    /* Double precision scalar upper lane test */
    if (size >= 8) {
        double d_fuzz;
        memcpy(&d_fuzz, data, sizeof(double));

        alignas(16) double dvec_a[2] = {d_fuzz, d_sentinel};
        alignas(16) double dvec_b[2] = {1.0, 0.0};
        __m128d da = _mm_load_pd(dvec_a);
        __m128d db = _mm_load_pd(dvec_b);

#define CHECK_SCALAR_PD(op)                \
    do {                                   \
        __m128d result = op;               \
        alignas(16) double out[2];         \
        _mm_store_pd(out, result);         \
        if (out[1] != d_sentinel) {        \
            volatile int *null_ptr = NULL; \
            *null_ptr = 1;                 \
        }                                  \
        SINK_PD(result);                   \
    } while (0)

        CHECK_SCALAR_PD(_mm_add_sd(da, db));
        CHECK_SCALAR_PD(_mm_sub_sd(da, db));
        CHECK_SCALAR_PD(_mm_mul_sd(da, db));
        CHECK_SCALAR_PD(_mm_min_sd(da, db));
        CHECK_SCALAR_PD(_mm_max_sd(da, db));
        CHECK_SCALAR_PD(_mm_sqrt_sd(da, db));
        CHECK_SCALAR_PD(_mm_cmpeq_sd(da, db));
        CHECK_SCALAR_PD(_mm_cmplt_sd(da, db));
        CHECK_SCALAR_PD(_mm_div_sd(da, db)); /* dvec_b[0]=1.0, no div-by-zero */

#undef CHECK_SCALAR_PD
    }
}

/* Strategy 25: System Intrinsics
 *
 * Tests memory barrier, prefetch, and control register intrinsics.
 * These are critical for cache behavior and synchronization.
 */
static void test_system_intrinsics(const uint8_t *data, size_t size)
{
    if (size < 16)
        return;

    /* Memory prefetch hints (no observable side effects in fuzzer) */
    _mm_prefetch((const char *) aligned_buf, _MM_HINT_T0);  /* Temporal */
    _mm_prefetch((const char *) aligned_buf, _MM_HINT_T1);  /* L2 cache */
    _mm_prefetch((const char *) aligned_buf, _MM_HINT_T2);  /* L3 cache */
    _mm_prefetch((const char *) aligned_buf, _MM_HINT_NTA); /* Non-temporal */

    /* Memory fences (ordering primitives) */
    _mm_sfence(); /* Store fence */
    _mm_lfence(); /* Load fence */
    _mm_mfence(); /* Full memory fence */

    /* CPU pause hint (spin-wait optimization) */
    _mm_pause();

    /* Cache line flush (use aligned buffer to avoid crashes) */
    _mm_clflush(aligned_buf);

    /* MXCSR control/status register */
    unsigned int mxcsr_orig = _mm_getcsr();
    SINK_I32(mxcsr_orig);

    /* Test setting different rounding modes (restore original after) */
    _mm_setcsr((mxcsr_orig & ~_MM_ROUND_MASK) | _MM_ROUND_DOWN);
    SINK_I32(_mm_getcsr());
    _mm_setcsr((mxcsr_orig & ~_MM_ROUND_MASK) | _MM_ROUND_UP);
    SINK_I32(_mm_getcsr());
    _mm_setcsr((mxcsr_orig & ~_MM_ROUND_MASK) | _MM_ROUND_TOWARD_ZERO);
    SINK_I32(_mm_getcsr());

    /* Restore original MXCSR */
    _mm_setcsr(mxcsr_orig);
}

/* libFuzzer Entry Point */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    /* Run all test strategies */
    test_shuffle_indices(data, size);
    test_alignment(data, size);
    test_integer_overflow(data, size);
    test_common_instructions(data, size);
    test_float_edge_cases(data, size);
    test_extract_insert(data, size);
    test_crc32(data, size);
    test_abs_sign(data, size);

    /* Scalar and NaN handling */
    test_scalar_float(data, size);
    test_nan_operand_order(data, size);
    test_ordered_unordered_cmp(data, size);
    test_dot_product(data, size);
    test_double_precision(data, size);
    test_conversion_edge_cases(data, size);

    /* SSE3, SSE4.2, AES, CLMUL extensions */
    test_sse3_ops(data, size);
    test_sse42_string(data, size);
    test_aes_ops(data, size);
    test_clmul_ops(data, size);

    /* Fuzz-controlled parameters */
    test_fuzz_controlled(data, size);

    /* Memory and advanced operations */
    test_streaming_stores(data, size);
    test_rounding_ops(data, size);
    test_masked_moves(data, size);
    test_advanced_integer(data, size);

    /* Invariant checks */
    test_scalar_lane_preservation(data, size);

    /* System intrinsics */
    test_system_intrinsics(data, size);

    return 0;
}
