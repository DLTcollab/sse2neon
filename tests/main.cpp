/* ARM64EC requires sse2neon.h to be included FIRST. See common.h for details.
 */
#if defined(_M_ARM64EC)
#include "sse2neon.h"
#endif

#include <stdint.h>
#include <stdio.h>
#include "impl.h"

int main(int /*argc*/, const char ** /*argv*/)
{
    SSE2NEON::SSE2NEONTest *test = SSE2NEON::SSE2NEONTest::create();
    uint32_t passCount = 0;
    uint32_t failedCount = 0;
    uint32_t ignoreCount = 0;
    for (uint32_t i = 0; i < SSE2NEON::it_last; i++) {
        SSE2NEON::InstructionTest it = SSE2NEON::InstructionTest(i);
        SSE2NEON::result_t ret = test->runTest(it);
        // If the test fails, we will run it again so we can step into the
        // debugger and figure out why!
        if (ret == SSE2NEON::TEST_FAIL) {
            printf("Test %-35s [" COLOR_RED "FAIL" COLOR_RESET "]\n",
                   SSE2NEON::instructionString[it]);
            failedCount++;
        } else if (ret == SSE2NEON::TEST_UNIMPL) {
            printf("Test %-35s [SKIP]\n", SSE2NEON::instructionString[it]);
            ignoreCount++;
        } else {
            printf("Test %-35s [ " COLOR_GREEN "OK" COLOR_RESET " ]\n",
                   SSE2NEON::instructionString[it]);
            passCount++;
        }
    }
    test->release();
    printf(
        "SSE2NEONTest Complete!\n"
        "Passed:  %d\n"
        "Failed:  %d\n"
        "Ignored: %d\n"
        "Coverage rate: %.2f%%\n",
        passCount, failedCount, ignoreCount,
        static_cast<float>(passCount) /
            static_cast<float>(passCount + failedCount + ignoreCount) * 100);

    return failedCount ? -1 : 0;
}
