#include <stdint.h>
#include <stdio.h>
#include "impl.h"

int main(int /*argc*/, const char ** /*argv*/)
{
    SSE2NEON::SSE2NEONTest *test = SSE2NEON::SSE2NEONTest::create();
    uint32_t passCount = 0;
    uint32_t failedCount = 0;
    for (uint32_t i = 0; i < SSE2NEON::IT_LAST; i++) {
        SSE2NEON::InstructionTest it = SSE2NEON::InstructionTest(i);
        printf("Running Test %s\n",
               SSE2NEON::SSE2NEONTest::getInstructionTestString(it));
        bool ok = test->runTest(it);
        // If the test fails, we will run it again so we can step into the
        // debugger and figure out why!
        if (!ok) {
            printf("**FAILURE** SSE2NEONTest %s\n",
                   SSE2NEON::SSE2NEONTest::getInstructionTestString(it));
        }
        if (ok) {
            passCount++;
        } else {
            failedCount++;
        }
    }
    test->release();
    printf("SSE2NEONTest Complete: Passed %d tests : Failed %d\n", passCount,
           failedCount);

    return failedCount ? -1 : 0;
}
