#ifndef SSE2NEONTEST_H
#define SSE2NEONTEST_H

namespace SSE2NEON
{
// The way unit tests are implemented is that 10,000 random floating point and
// integer vec4 numbers are generated as sample data.
//
// A short C implementation of every intrinsic is implemented and compared to
// the actual expected results from the corresponding SSE intrinsic against all
// of the 10,000 randomized input vectors. When running on ARM, then the results
// are compared to the NEON approximate version.
enum InstructionTest {
    // SSE
    IT_MM_ADD_PS,
    IT_MM_ADD_SS,  // Unimplemented
    IT_MM_ANDNOT_PS,
    IT_MM_AND_PS,
    IT_MM_CMPEQ_PS,
    IT_MM_CMPEQ_SS,
    IT_MM_CMPGE_PS,
    IT_MM_CMPGE_SS,
    IT_MM_CMPGT_PS,
    IT_MM_CMPGT_SS,
    IT_MM_CMPLE_PS,
    IT_MM_CMPLE_SS,
    IT_MM_CMPLT_PS,
    IT_MM_CMPLT_SS,
    IT_MM_CMPNEQ_PS,
    IT_MM_CMPNEQ_SS,
    IT_MM_CMPNGE_PS,
    IT_MM_CMPNGE_SS,
    IT_MM_CMPNGT_PS,
    IT_MM_CMPNGT_SS,
    IT_MM_CMPNLE_PS,
    IT_MM_CMPNLE_SS,
    IT_MM_CMPNLT_PS,
    IT_MM_CMPNLT_SS,
    IT_MM_CMPORD_PS,
    IT_MM_CMPORD_SS,
    IT_MM_CMPUNORD_PS,
    IT_MM_CMPUNORD_SS,
    IT_MM_COMIEQ_SS,
    IT_MM_COMIGE_SS,
    IT_MM_COMIGT_SS,
    IT_MM_COMILE_SS,
    IT_MM_COMILT_SS,
    IT_MM_COMINEQ_SS,
    IT_MM_CVTSS_F32,  // Unimplemented
    IT_MM_DIV_PS,     // Unimplemented
    IT_MM_DIV_SS,     // Unimplemented
    IT_MM_LOAD1_PS,
    IT_MM_LOADH_PI,
    IT_MM_LOADL_PI,
    IT_MM_LOAD_PS,   // Unimplemented
    IT_MM_LOAD_SS,   // Unimplemented
    IT_MM_LOADU_PS,  // Unimplemented
    IT_MM_MALLOC,
    IT_MM_MAX_PS,
    IT_MM_MAX_SS,  // Unimplemented
    IT_MM_MIN_PS,
    IT_MM_MIN_SS,  // Unimplemented
    IT_MM_MOVEMASK_PS,
    IT_MM_MOVE_SS,
    IT_MM_MUL_PS,
    IT_MM_OR_PS,
    IT_MM_RCP_PS,
    IT_MM_RSQRT_PS,  // Unimplemented
    IT_MM_SET1_PS,
    IT_MM_SET_PS1,
    IT_MM_SET_PS,
    IT_MM_SETR_PS,  // Unimplemented
    IT_MM_SET_SS,
    IT_MM_SETZERO_PS,
    IT_MM_SFENCE,  // Unimplemented
    IT_MM_SHUFFLE_PS,
    IT_MM_SQRT_PS,  // Unimplemented
    IT_MM_SQRT_SS,  // Unimplemented
    IT_MM_STOREH_PI,
    IT_MM_STOREL_PI,
    IT_MM_STORE_PS,
    IT_MM_STORE_SS,   // Unimplemented
    IT_MM_STOREU_PS,  // Unimplemented
    IT_MM_SUB_PS,
    IT_MM_UNPACKHI_PS,  // Unimplemented
    IT_MM_UNPACKLO_PS,  // Unimplemented
    IT_MM_XOR_PS,       // Unimplemented

    // SSE2
    IT_MM_ADD_EPI16,  // Unimplemented
    IT_MM_ADD_EPI32,
    IT_MM_ADD_EPI8,
    IT_MM_ADDS_EPI16,
    IT_MM_ADDS_EPU8,
    IT_MM_ANDNOT_SI128,
    IT_MM_AND_SI128,
    IT_MM_AVG_EPU16,
    IT_MM_AVG_EPU8,
    IT_MM_CASTPS_SI128,  // Unimplemented
    IT_MM_CASTSI128_PS,  // Unimplemented
    IT_MM_CLFLUSH,       // Unimplemented
    IT_MM_CMPEQ_EPI16,
    IT_MM_CMPEQ_EPI8,
    IT_MM_CMPGT_EPI16,
    IT_MM_CMPGT_EPI32,
    IT_MM_CMPGT_EPI8,
    IT_MM_CMPLT_EPI16,
    IT_MM_CMPLT_EPI32,
    IT_MM_CMPLT_EPI8,
    IT_MM_CVTEPI32_PS,
    IT_MM_CVTPS_EPI32,
    IT_MM_CVTSI128_SI32,  // Unimplemented
    IT_MM_CVTSI32_SI128,  // Unimplemented
    IT_MM_CVTTPS_EPI32,
    IT_MM_LOAD_SD,
    IT_MM_LOAD_PD,
    IT_MM_LOADU_PD,
    IT_MM_LOAD_SI128,  // Unimplemented
    IT_MM_LOADU_SI128,
    IT_MM_MADD_EPI16,
    IT_MM_MAX_EPI16,
    IT_MM_MAX_EPU8,
    IT_MM_MIN_EPI16,
    IT_MM_MIN_EPU8,
    IT_MM_MOVEMASK_EPI8,
    IT_MM_MUL_EPU32,
    IT_MM_MULHI_EPI16,
    IT_MM_MULLO_EPI16,
    IT_MM_OR_SI128,
    IT_MM_PACKS_EPI16,   // Unimplemented
    IT_MM_PACKS_EPI32,   // Unimplemented
    IT_MM_PACKUS_EPI16,  // Unimplemented
    IT_MM_SET1_EPI16,
    IT_MM_SET1_EPI32,
    IT_MM_SET1_EPI8,
    IT_MM_SET_EPI16,
    IT_MM_SET_EPI32,
    IT_MM_SET_EPI8,
    IT_MM_SETR_EPI32,
    IT_MM_SETZERO_SI128,
    IT_MM_SLL_EPI16,
    IT_MM_SLL_EPI32,
    IT_MM_SLL_EPI64,
    IT_MM_SLLI_EPI16,
    IT_MM_SRA_EPI16,
    IT_MM_SRA_EPI32,
    IT_MM_SRL_EPI16,
    IT_MM_SRL_EPI32,
    IT_MM_SRL_EPI64,
    IT_MM_SRLI_EPI16,
    IT_MM_STOREL_EPI64,  // Unimplemented
    IT_MM_STORE_SI128,   // Unimplemented
    IT_MM_STOREU_SI128,
    IT_MM_STREAM_SI128,  // Unimplemented
    IT_MM_SUB_EPI32,
    IT_MM_SUB_EPI64,
    IT_MM_SUB_EPI8,
    IT_MM_SUBS_EPU16,
    IT_MM_SUBS_EPU8,
    IT_MM_UNPACKHI_EPI16,  // Unimplemented
    IT_MM_UNPACKHI_EPI32,  // Unimplemented
    IT_MM_UNPACKHI_EPI8,   // Unimplemented
    IT_MM_UNPACKLO_EPI16,  // Unimplemented
    IT_MM_UNPACKLO_EPI32,  // Unimplemented
    IT_MM_UNPACKLO_EPI8,   // Unimplemented
    IT_MM_XOR_SI128,       // Unimplemented

    // SSE3
    IT_MM_HADD_PS,  // Unimplemented
    IT_MM_MOVEHDUP_PS,
    IT_MM_MOVELDUP_PS,

    // SSSE3
    IT_MM_HADD_EPI16,
    IT_MM_MADDUBS_EPI16,
    IT_MM_SHUFFLE_EPI8,

    // SSE4.1
    IT_MM_BLENDV_PS,  // Unimplemented
    IT_MM_CEIL_PS,
    IT_MM_CMPEQ_EPI64,
    IT_MM_FLOOR_PS,
    IT_MM_MAX_EPI32,  // Unimplemented
    IT_MM_MIN_EPI32,  // Unimplemented
    IT_MM_MINPOS_EPU16,
    IT_MM_MULLO_EPI32,  // Unimplemented
    IT_MM_ROUND_PS,
    IT_MM_TEST_ALL_ZEROS,
    IT_MM_TESTZ_SI128,

    // SSE4.2
    IT_MM_CRC32_U16,
    IT_MM_CRC32_U32,
    IT_MM_CRC32_U64,
    IT_MM_CRC32_U8,

    // AES
    IT_MM_AESENC_SI128,
    IT_MM_AESKEYGENASSIST_SI128,

    // Others
    IT_MM_CLMULEPI64_SI128,
    IT_MM_POPCNT_U32,
    IT_MM_POPCNT_U64,
    IT_MM_SHUFFLE_EPI32_DEFAULT,     // Unimplemented
    IT_MM_SHUFFLE_EPI32_FUNCTION,    // Unimplemented
    IT_MM_SHUFFLE_EPI32_SINGLE,      // Unimplemented
    IT_MM_SHUFFLE_EPI32_SPLAT,       // Unimplemented
    IT_MM_SHUFFLEHI_EPI16_FUNCTION,  // Unimplemented

    IT_LAST
};

class SSE2NEONTest
{
public:
    static SSE2NEONTest *create(void);  // create the test.
    static const char *getInstructionTestString(InstructionTest test);

    // Run test of this instruction; return true if it passed, false if it
    // failed
    virtual bool runTest(InstructionTest test) = 0;
    virtual void release(void) = 0;
};

}  // namespace SSE2NEON

#endif
