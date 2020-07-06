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
    IT_MM_SETZERO_SI128,  // Unit test implemented and verified as fully working
    IT_MM_SETZERO_PS,     // Unit test implemented and verified as fully working
    IT_MM_SET1_PS,        // Unit test implemented and verified as fully working
    IT_MM_SET_PS1,        // Unit test implemented and verified as fully working
    IT_MM_SET_PS,         // Unit test implemented and verified as fully working
    IT_MM_SET_SS,         // Unit test implemented and verified as fully working
    IT_MM_SET_EPI8,       // Unit test implemented and verified as fully working
    IT_MM_SET1_EPI32,     // Unit test implemented and verified as fully working
    IT_MM_SET_EPI32,      // Unit test implemented and verified as fully working
    IT_MM_STORE_PS,       // Unit test implemented and verified as fully working
    IT_MM_STOREL_PI,      // Unit test implemented and verified as fully working
    IT_MM_STOREH_PI,      // Unit test implemented and verified as fully working
    IT_MM_SHUFFLE_PS,     // Unit test implemented and verified as fully working
    IT_MM_LOAD1_PS,       // Unit test implemented and verified as fully working
    IT_MM_LOADL_PI,       // Unit test implemented and verified as fully working
    IT_MM_LOADH_PI,       // Unit test implemented and verified as fully working
    IT_MM_ANDNOT_PS,      // Unit test implemented and verified as fully working
    IT_MM_ANDNOT_SI128,   // Unit test implemented and verified as fully working
    IT_MM_AND_SI128,      // Unit test implemented and verified as fully working
    IT_MM_AND_PS,         // Unit test implemented and verified as fully working
    IT_MM_OR_PS,          // Unit test implemented and verified as fully working
    IT_MM_OR_SI128,       // Unit test implemented and verified as fully working
    IT_MM_MOVEMASK_PS,    // Unit test implemented and verified as fully working
    IT_MM_MOVEMASK_EPI8,  // Unit test implemented and verified as fully working
    IT_MM_SUB_PS,         // Unit test implemented and verified as fully working
    IT_MM_SUB_EPI64,      // Unit test implemented
    IT_MM_SUB_EPI32,      // Unit test implemented and verified as fully working
    IT_MM_ADD_PS,         // Unit test implemented and verified as fully working
    IT_MM_ADD_EPI32,      // Unit test implemented and verified as fully working
    IT_MM_MULLO_EPI16,    // Unit test implemented and verified as fully working
    IT_MM_MUL_PS,         // Unit test implemented and verified as fully working
    IT_MM_RCP_PS,         // Unit test implemented and verified as fully working
    IT_MM_MAX_PS,         // Unit test implemented and verified as fully working
    IT_MM_MIN_PS,         // Unit test implemented and verified as fully working
    IT_MM_MIN_EPI16,      // Unit test implemented and verified as fully working
    IT_MM_MULHI_EPI16,    // Unit test implemented and verified as fully working
    IT_MM_CMPLT_PS,       // Unit test implemented and verified as fully working
    IT_MM_CMPGT_PS,       // Unit test implemented and verified as fully working
    IT_MM_CMPGE_PS,       // Unit test implemented and verified as fully working
    IT_MM_CMPLE_PS,       // Unit test implemented and verified as fully working
    IT_MM_CMPEQ_PS,       // Unit test implemented and verified as fully working
    IT_MM_CMPLT_EPI16,    // Unit test implemented and verified as fully working
    IT_MM_CMPLT_EPI32,    // Unit test implemented and verified as fully working
    IT_MM_CMPGT_EPI32,    // Unit test implemented and verified as fully working
    IT_MM_CVTTPS_EPI32,   // Unit test implemented and verified as fully working
    IT_MM_CVTEPI32_PS,    // Unit test implemented and verified as fully working
    IT_MM_CVTPS_EPI32,    // Unit test implemented and verified as fully working
    IT_MM_CVTSS_F32,      // Unit test *not yet implemented*
    IT_MM_SETR_PS,        // Unit test *not yet implemented*
    IT_MM_STOREU_PS,      // Unit test *not yet implemented*
    IT_MM_STORE_SI128,    // Unit test *not yet implemented*
    IT_MM_STORE_SS,       // Unit test *not yet implemented*
    IT_MM_STOREL_EPI64,   // Unit test *not yet implemented*
    IT_MM_LOAD_PS,        // Unit test *not yet implemented*
    IT_MM_LOADU_PS,       // Unit test *not yet implemented*
    IT_MM_LOAD_SD,        // Unit test implemented and verified as fully working
    IT_MM_LOAD_SS,        // Unit test *not yet implemented*
    IT_MM_CMPNEQ_PS,      // Unit test *not yet implemented*
    IT_MM_XOR_PS,         // Unit test *not yet implemented*
    IT_MM_XOR_SI128,      // Unit test *not yet implemented*
    IT_MM_SHUFFLE_EPI8,   // Unit test implemented and verified as fully working
    IT_MM_SHUFFLE_EPI32_DEFAULT,     // Unit test *not yet implemented*
    IT_MM_SHUFFLE_EPI32_FUNCTION,    // Unit test *not yet implemented*
    IT_MM_SHUFFLE_EPI32_SPLAT,       // Unit test *not yet implemented*
    IT_MM_SHUFFLE_EPI32_SINGLE,      // Unit test *not yet implemented*
    IT_MM_SHUFFLEHI_EPI16_FUNCTION,  // Unit test *not yet implemented*
    IT_MM_ADD_SS,                    // Unit test *not yet implemented*
    IT_MM_ADD_EPI16,                 // Unit test *not yet implemented*
    IT_MM_MADD_EPI16,     // Unit test implemented and verified as fully working
    IT_MM_MADDUBS_EPI16,  // Unit test implemented
    IT_MM_MULLO_EPI32,    // Unit test *not yet implemented*
    IT_MM_MUL_EPU32,      // Unit test implemented
    IT_MM_DIV_PS,         // Unit test *not yet implemented*
    IT_MM_DIV_SS,         // Unit test *not yet implemented*
    IT_MM_SQRT_PS,        // Unit test *not yet implemented*
    IT_MM_SQRT_SS,        // Unit test *not yet implemented*
    IT_MM_RSQRT_PS,       // Unit test implemented and verified as fully working
    IT_MM_MAX_SS,         // Unit test *not yet implemented*
    IT_MM_MIN_SS,         // Unit test *not yet implemented*
    IT_MM_MAX_EPI32,      // Unit test *not yet implemented*
    IT_MM_MIN_EPI32,      // Unit test *not yet implemented*
    IT_MM_MINPOS_EPU16,   // Unit test implemented and verified as fully working
    IT_MM_HADD_PS,        // Unit test *not yet implemented*
    IT_MM_HADD_EPI16,     // Unit test implemented
    IT_MM_CMPORD_PS,      // Unit test *not yet implemented*
    IT_MM_COMILT_SS,      // Unit test implemented
    IT_MM_COMIGT_SS,      // Unit test implemented
    IT_MM_COMILE_SS,      // Unit test implemented
    IT_MM_COMIGE_SS,      // Unit test implemented
    IT_MM_COMIEQ_SS,      // Unit test implemented
    IT_MM_COMINEQ_SS,     // Unit test implemented
    IT_MM_CVTSI128_SI32,  // Unit test *not yet implemented*
    IT_MM_CVTSI32_SI128,  // Unit test *not yet implemented*
    IT_MM_CASTPS_SI128,   // Unit test *not yet implemented*
    IT_MM_CASTSI128_PS,   // Unit test *not yet implemented*
    IT_MM_LOAD_SI128,     // Unit test *not yet implemented*
    IT_MM_PACKS_EPI16,    // Unit test *not yet implemented*
    IT_MM_PACKUS_EPI16,   // Unit test *not yet implemented*
    IT_MM_PACKS_EPI32,    // Unit test *not yet implemented*
    IT_MM_UNPACKLO_EPI8,  // Unit test *not yet implemented*
    IT_MM_UNPACKLO_EPI16,  // Unit test *not yet implemented*
    IT_MM_UNPACKLO_EPI32,  // Unit test *not yet implemented*
    IT_MM_UNPACKLO_PS,     // Unit test *not yet implemented*
    IT_MM_UNPACKHI_PS,     // Unit test *not yet implemented*
    IT_MM_UNPACKHI_EPI8,   // Unit test *not yet implemented*
    IT_MM_UNPACKHI_EPI16,  // Unit test *not yet implemented*
    IT_MM_UNPACKHI_EPI32,  // Unit test *not yet implemented*
    IT_MM_SFENCE,          // Unit test *not yet implemented*
    IT_MM_STREAM_SI128,    // Unit test *not yet implemented*
    IT_MM_CLFLUSH,         // Unit test *not yet implemented*

    IT_MM_SET1_EPI16,
    IT_MM_SET_EPI16,
    IT_MM_SLLI_EPI16,  // Unit test implemented
    IT_MM_SLL_EPI16,   // Unit test implemented
    IT_MM_SLL_EPI32,   // Unit test implemented
    IT_MM_SLL_EPI64,   // Unit test implemented
    IT_MM_SRL_EPI16,   // Unit test implemented
    IT_MM_SRL_EPI32,   // Unit test implemented
    IT_MM_SRL_EPI64,   // Unit test implemented
    IT_MM_SRA_EPI16,   // Unit test implemented and verified as fully working
    IT_MM_SRA_EPI32,   // Unit test implemented and verified as fully working
    IT_MM_SRLI_EPI16,
    IT_MM_CMPEQ_EPI16,
    IT_MM_CMPEQ_EPI64,

    IT_MM_SET1_EPI8,
    IT_MM_ADDS_EPU8,
    IT_MM_SUBS_EPU8,
    IT_MM_MAX_EPU8,
    IT_MM_CMPEQ_EPI8,
    IT_MM_ADDS_EPI16,
    IT_MM_MAX_EPI16,
    IT_MM_SUBS_EPU16,
    IT_MM_CMPGT_EPI16,
    IT_MM_LOADU_SI128,
    IT_MM_STOREU_SI128,
    IT_MM_ADD_EPI8,
    IT_MM_CMPGT_EPI8,
    IT_MM_CMPLT_EPI8,
    IT_MM_SUB_EPI8,
    IT_MM_SETR_EPI32,
    IT_MM_MIN_EPU8,
    IT_MM_TEST_ALL_ZEROS,
    IT_MM_AVG_EPU8,
    IT_MM_AVG_EPU16,
    IT_MM_POPCNT_U32,
    IT_MM_POPCNT_U64,
#if !defined(__arm__) && __ARM_ARCH != 7
    IT_MM_AESENC_SI128,
#endif
    IT_MM_CLMULEPI64_SI128,
    IT_MM_MALLOC,
    IT_MM_CRC32_U8,   // Unit test implemented
    IT_MM_CRC32_U16,  // Unit test implemented
    IT_MM_CRC32_U32,  // Unit test implemented
    IT_MM_CRC32_U64,  // Unit test implemented
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
