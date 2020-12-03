#ifndef SSE2NEONTEST_H
#define SSE2NEONTEST_H
#include "common.h"
#define ENUM(c) it_##c,
#define STR(c) #c,
#define CASE(c)                   \
    case it_##c:                  \
        ret = test_##c(*this, i); \
        break;
#define INTRIN_FOREACH(TYPE)          \
    /* SSE */                         \
    TYPE(mm_add_ps)                   \
    TYPE(mm_add_ss)                   \
    TYPE(mm_and_ps)                   \
    TYPE(mm_andnot_ps)                \
    TYPE(mm_avg_pu16)                 \
    TYPE(mm_avg_pu8)                  \
    TYPE(mm_cmpeq_ps)                 \
    TYPE(mm_cmpeq_ss)                 \
    TYPE(mm_cmpge_ps)                 \
    TYPE(mm_cmpge_ss)                 \
    TYPE(mm_cmpgt_ps)                 \
    TYPE(mm_cmpgt_ss)                 \
    TYPE(mm_cmple_ps)                 \
    TYPE(mm_cmple_ss)                 \
    TYPE(mm_cmplt_ps)                 \
    TYPE(mm_cmplt_ss)                 \
    TYPE(mm_cmpneq_ps)                \
    TYPE(mm_cmpneq_ss)                \
    TYPE(mm_cmpnge_ps)                \
    TYPE(mm_cmpnge_ss)                \
    TYPE(mm_cmpngt_ps)                \
    TYPE(mm_cmpngt_ss)                \
    TYPE(mm_cmpnle_ps)                \
    TYPE(mm_cmpnle_ss)                \
    TYPE(mm_cmpnlt_ps)                \
    TYPE(mm_cmpnlt_ss)                \
    TYPE(mm_cmpord_ps)                \
    TYPE(mm_cmpord_ss)                \
    TYPE(mm_cmpunord_ps)              \
    TYPE(mm_cmpunord_ss)              \
    TYPE(mm_comieq_ss)                \
    TYPE(mm_comige_ss)                \
    TYPE(mm_comigt_ss)                \
    TYPE(mm_comile_ss)                \
    TYPE(mm_comilt_ss)                \
    TYPE(mm_comineq_ss)               \
    TYPE(mm_cvt_pi2ps)                \
    TYPE(mm_cvt_ps2pi)                \
    TYPE(mm_cvt_si2ss)                \
    TYPE(mm_cvt_ss2si)                \
    TYPE(mm_cvtpi16_ps)               \
    TYPE(mm_cvtpi32_ps)               \
    TYPE(mm_cvtpi32x2_ps)             \
    TYPE(mm_cvtpi8_ps)                \
    TYPE(mm_cvtpu16_ps)               \
    TYPE(mm_cvtpu8_ps)                \
    TYPE(mm_cvtss_f32)                \
    TYPE(mm_div_ps)                   \
    TYPE(mm_div_ss)                   \
    TYPE(mm_free)                     \
    TYPE(mm_load_ps)                  \
    TYPE(mm_load_ps1)                 \
    TYPE(mm_load_ss)                  \
    TYPE(mm_load1_ps)                 \
    TYPE(mm_loadh_pi)                 \
    TYPE(mm_loadl_pi)                 \
    TYPE(mm_loadr_ps)                 \
    TYPE(mm_loadu_ps)                 \
    TYPE(mm_loadu_si16)               \
    TYPE(mm_loadu_si64)               \
    TYPE(mm_malloc)                   \
    TYPE(mm_max_pi16)                 \
    TYPE(mm_max_ps)                   \
    TYPE(mm_max_pu8)                  \
    TYPE(mm_max_ss)                   \
    TYPE(mm_min_pi16)                 \
    TYPE(mm_min_ps)                   \
    TYPE(mm_min_pu8)                  \
    TYPE(mm_min_ss)                   \
    TYPE(mm_move_ss)                  \
    TYPE(mm_movehl_ps)                \
    TYPE(mm_movelh_ps)                \
    TYPE(mm_movemask_ps)              \
    TYPE(mm_mul_ps)                   \
    TYPE(mm_mul_ss)                   \
    TYPE(mm_mulhi_pu16)               \
    TYPE(mm_or_ps)                    \
    TYPE(mm_prefetch)                 \
    TYPE(mm_rcp_ps)                   \
    TYPE(mm_rcp_ss)                   \
    TYPE(mm_rsqrt_ps)                 \
    TYPE(mm_rsqrt_ss)                 \
    TYPE(mm_sad_pu8)                  \
    TYPE(mm_set_ps)                   \
    TYPE(mm_set_ps1)                  \
    TYPE(mm_set_ss)                   \
    TYPE(mm_set1_ps)                  \
    TYPE(mm_setr_ps)                  \
    TYPE(mm_setzero_ps)               \
    TYPE(mm_sfence)                   \
    TYPE(mm_shuffle_ps)               \
    TYPE(mm_sqrt_ps)                  \
    TYPE(mm_sqrt_ss)                  \
    TYPE(mm_store_ps)                 \
    TYPE(mm_store_ss)                 \
    TYPE(mm_storeh_pi)                \
    TYPE(mm_storel_pi)                \
    TYPE(mm_storeu_ps)                \
    TYPE(mm_stream_ps)                \
    TYPE(mm_sub_ps)                   \
    TYPE(mm_sub_ss)                   \
    TYPE(mm_undefined_ps)             \
    TYPE(mm_unpackhi_ps)              \
    TYPE(mm_unpacklo_ps)              \
    TYPE(mm_xor_ps)                   \
    /* SSE2 */                        \
    TYPE(mm_add_epi16)                \
    TYPE(mm_add_epi32)                \
    TYPE(mm_add_epi64)                \
    TYPE(mm_add_epi8)                 \
    TYPE(mm_add_pd)                   \
    TYPE(mm_add_si64)                 \
    TYPE(mm_adds_epi16)               \
    TYPE(mm_adds_epi8)                \
    TYPE(mm_adds_epu16)               \
    TYPE(mm_adds_epu8)                \
    TYPE(mm_and_pd)                   \
    TYPE(mm_and_si128)                \
    TYPE(mm_andnot_pd)                \
    TYPE(mm_andnot_si128)             \
    TYPE(mm_avg_epu16)                \
    TYPE(mm_avg_epu8)                 \
    TYPE(mm_castpd_si128)             \
    TYPE(mm_castps_pd)                \
    TYPE(mm_castps_si128)             \
    TYPE(mm_castsi128_ps)             \
    TYPE(mm_clflush)                  \
    TYPE(mm_cmpeq_epi16)              \
    TYPE(mm_cmpeq_epi32)              \
    TYPE(mm_cmpeq_epi8)               \
    TYPE(mm_cmpgt_epi16)              \
    TYPE(mm_cmpgt_epi32)              \
    TYPE(mm_cmpgt_epi8)               \
    TYPE(mm_cmplt_epi16)              \
    TYPE(mm_cmplt_epi32)              \
    TYPE(mm_cmplt_epi8)               \
    TYPE(mm_cvtepi32_ps)              \
    TYPE(mm_cvtpd_ps)                 \
    TYPE(mm_cvtps_epi32)              \
    TYPE(mm_cvtps_pd)                 \
    TYPE(mm_cvtsd_f64)                \
    TYPE(mm_cvtsi128_si32)            \
    TYPE(mm_cvtsi128_si64)            \
    TYPE(mm_cvtsi32_si128)            \
    TYPE(mm_cvtsi64_si128)            \
    TYPE(mm_cvttps_epi32)             \
    TYPE(mm_extract_epi16)            \
    TYPE(mm_insert_epi16)             \
    TYPE(mm_load_pd)                  \
    TYPE(mm_load_sd)                  \
    TYPE(mm_load_si128)               \
    TYPE(mm_load1_pd)                 \
    TYPE(mm_loadh_pd)                 \
    TYPE(mm_loadl_epi64)              \
    TYPE(mm_loadl_pd)                 \
    TYPE(mm_loadr_pd)                 \
    TYPE(mm_loadu_pd)                 \
    TYPE(mm_loadu_si128)              \
    TYPE(mm_loadu_si32)               \
    TYPE(mm_madd_epi16)               \
    TYPE(mm_max_epi16)                \
    TYPE(mm_max_epu8)                 \
    TYPE(mm_min_epi16)                \
    TYPE(mm_min_epu32)                \
    TYPE(mm_min_epu8)                 \
    TYPE(mm_move_epi64)               \
    TYPE(mm_movemask_epi8)            \
    TYPE(mm_movepi64_pi64)            \
    TYPE(mm_movpi64_epi64)            \
    TYPE(mm_mul_epu32)                \
    TYPE(mm_mul_su32)                 \
    TYPE(mm_mulhi_epi16)              \
    TYPE(mm_mullo_epi16)              \
    TYPE(mm_or_si128)                 \
    TYPE(mm_packs_epi16)              \
    TYPE(mm_packs_epi32)              \
    TYPE(mm_packus_epi16)             \
    TYPE(mm_sad_epu8)                 \
    TYPE(mm_set_epi16)                \
    TYPE(mm_set_epi32)                \
    TYPE(mm_set_epi64)                \
    TYPE(mm_set_epi64x)               \
    TYPE(mm_set_epi8)                 \
    TYPE(mm_set_pd)                   \
    TYPE(mm_set1_epi16)               \
    TYPE(mm_set1_epi32)               \
    TYPE(mm_set1_epi64)               \
    TYPE(mm_set1_epi64x)              \
    TYPE(mm_set1_epi8)                \
    TYPE(mm_setr_epi16)               \
    TYPE(mm_setr_epi32)               \
    TYPE(mm_setr_epi64)               \
    TYPE(mm_setr_epi8)                \
    TYPE(mm_setzero_si128)            \
    TYPE(mm_sll_epi16)                \
    TYPE(mm_sll_epi32)                \
    TYPE(mm_sll_epi64)                \
    TYPE(mm_slli_epi16)               \
    TYPE(mm_slli_epi32)               \
    TYPE(mm_slli_epi64)               \
    TYPE(mm_slli_si128)               \
    TYPE(mm_sra_epi16)                \
    TYPE(mm_sra_epi32)                \
    TYPE(mm_srai_epi16)               \
    TYPE(mm_srai_epi32)               \
    TYPE(mm_srl_epi16)                \
    TYPE(mm_srl_epi32)                \
    TYPE(mm_srl_epi64)                \
    TYPE(mm_srli_epi16)               \
    TYPE(mm_srli_epi32)               \
    TYPE(mm_srli_epi64)               \
    TYPE(mm_srli_si128)               \
    TYPE(mm_store_pd)                 \
    TYPE(mm_store_si128)              \
    TYPE(mm_storel_epi64)             \
    TYPE(mm_storeu_pd)                \
    TYPE(mm_storeu_si128)             \
    TYPE(mm_stream_si128)             \
    TYPE(mm_sub_epi16)                \
    TYPE(mm_sub_epi32)                \
    TYPE(mm_sub_epi64)                \
    TYPE(mm_sub_epi8)                 \
    TYPE(mm_sub_si64)                 \
    TYPE(mm_subs_epi16)               \
    TYPE(mm_subs_epi8)                \
    TYPE(mm_subs_epu16)               \
    TYPE(mm_subs_epu8)                \
    TYPE(mm_unpackhi_epi16)           \
    TYPE(mm_unpackhi_epi32)           \
    TYPE(mm_unpackhi_epi64)           \
    TYPE(mm_unpackhi_epi8)            \
    TYPE(mm_unpacklo_epi16)           \
    TYPE(mm_unpacklo_epi32)           \
    TYPE(mm_unpacklo_epi64)           \
    TYPE(mm_unpacklo_epi8)            \
    TYPE(mm_xor_pd)                   \
    TYPE(mm_xor_si128)                \
    /* SSE3 */                        \
    TYPE(mm_addsub_ps)                \
    TYPE(mm_hadd_ps)                  \
    TYPE(mm_hsub_ps)                  \
    TYPE(mm_movehdup_ps)              \
    TYPE(mm_moveldup_ps)              \
    /* SSSE3 */                       \
    TYPE(mm_abs_epi16)                \
    TYPE(mm_abs_epi32)                \
    TYPE(mm_abs_epi8)                 \
    TYPE(mm_abs_pi16)                 \
    TYPE(mm_abs_pi32)                 \
    TYPE(mm_abs_pi8)                  \
    TYPE(mm_hadd_epi16)               \
    TYPE(mm_hadd_epi32)               \
    TYPE(mm_hadd_pi16)                \
    TYPE(mm_hadd_pi32)                \
    TYPE(mm_hadds_epi16)              \
    TYPE(mm_hsub_epi16)               \
    TYPE(mm_hsub_epi32)               \
    TYPE(mm_hsubs_epi16)              \
    TYPE(mm_maddubs_epi16)            \
    TYPE(mm_mulhrs_epi16)             \
    TYPE(mm_shuffle_epi8)             \
    TYPE(mm_sign_epi16)               \
    TYPE(mm_sign_epi32)               \
    TYPE(mm_sign_epi8)                \
    TYPE(mm_sign_pi16)                \
    TYPE(mm_sign_pi32)                \
    TYPE(mm_sign_pi8)                 \
    /* SSE4.1 */                      \
    TYPE(mm_blend_epi16)              \
    TYPE(mm_blendv_epi8)              \
    TYPE(mm_blendv_ps)                \
    TYPE(mm_ceil_ps)                  \
    TYPE(mm_cmpeq_epi64)              \
    TYPE(mm_cvtepi16_epi32)           \
    TYPE(mm_cvtepi16_epi64)           \
    TYPE(mm_cvtepi32_epi64)           \
    TYPE(mm_cvtepi8_epi16)            \
    TYPE(mm_cvtepi8_epi32)            \
    TYPE(mm_cvtepi8_epi64)            \
    TYPE(mm_cvtepu16_epi32)           \
    TYPE(mm_cvtepu16_epi64)           \
    TYPE(mm_cvtepu32_epi64)           \
    TYPE(mm_cvtepu8_epi16)            \
    TYPE(mm_cvtepu8_epi32)            \
    TYPE(mm_cvtepu8_epi64)            \
    TYPE(mm_dp_ps)                    \
    TYPE(mm_extract_epi32)            \
    TYPE(mm_extract_epi64)            \
    TYPE(mm_extract_epi8)             \
    TYPE(mm_extract_ps)               \
    TYPE(mm_floor_ps)                 \
    TYPE(mm_insert_epi32)             \
    TYPE(mm_insert_epi64)             \
    TYPE(mm_insert_epi8)              \
    TYPE(mm_max_epi32)                \
    TYPE(mm_max_epi8)                 \
    TYPE(mm_max_epu32)                \
    TYPE(mm_min_epi32)                \
    TYPE(mm_minpos_epu16)             \
    TYPE(mm_mul_epi32)                \
    TYPE(mm_mullo_epi32)              \
    TYPE(mm_packus_epi32)             \
    TYPE(mm_round_ps)                 \
    TYPE(mm_stream_load_si128)        \
    TYPE(mm_test_all_ones)            \
    TYPE(mm_test_all_zeros)           \
    TYPE(mm_testc_si128)              \
    TYPE(mm_testz_si128)              \
    /* SSE4.2 */                      \
    TYPE(mm_cmpgt_epi64)              \
    TYPE(mm_crc32_u16)                \
    TYPE(mm_crc32_u32)                \
    TYPE(mm_crc32_u64)                \
    TYPE(mm_crc32_u8)                 \
    /* AES */                         \
    TYPE(mm_aesenc_si128)             \
    TYPE(mm_aesenclast_si128)         \
    TYPE(mm_aeskeygenassist_si128)    \
    /* FMA */                         \
    TYPE(mm_fmadd_ps)                 \
    /* Others */                      \
    TYPE(mm_clmulepi64_si128)         \
    TYPE(mm_popcnt_u32)               \
    TYPE(mm_popcnt_u64)               \
    TYPE(mm_shuffle_epi32)            \
    TYPE(mm_shuffle_epi32_default)    \
    TYPE(mm_shuffle_epi32_splat)      \
    TYPE(mm_shufflehi_epi16)          \
    TYPE(mm_shufflehi_epi16_function) \
    TYPE(mm_shufflelo_epi16)          \
    TYPE(mm_shufflelo_epi16_function) \
    TYPE(last) /* This indicates the end of macros */

namespace SSE2NEON
{
// The way unit tests are implemented is that 10,000 random floating point and
// integer vec4 numbers are generated as sample data.
//
// A short C implementation of every intrinsic is implemented and compared to
// the actual expected results from the corresponding SSE intrinsic against all
// of the 10,000 randomized input vectors. When running on ARM, then the results
// are compared to the NEON approximate version.
extern const char *instructionString[];
enum InstructionTest { INTRIN_FOREACH(ENUM) };

class SSE2NEONTest
{
public:
    static SSE2NEONTest *create(void);  // create the test.

    // Run test of this instruction;
    // Passed: TEST_SUCCESS (1)
    // Failed: TEST_FAIL (0)
    // Unimplemented: TEST_UNIMPL (-1)
    virtual result_t runTest(InstructionTest test) = 0;
    virtual void release(void) = 0;
};

}  // namespace SSE2NEON

#endif
