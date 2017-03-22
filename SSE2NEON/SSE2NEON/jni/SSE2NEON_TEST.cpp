/**********************************
 Java Native Interface library
**********************************/
#include <jni.h>
#include <android/log.h>
#include "../../../SSE2NEONTEST.h"
#include <stdint.h>

/** This is the C++ implementation of the Java native method.
@param env Pointer to JVM environment
@param thiz Reference to Java this object
*/
extern "C"
JNIEXPORT void JNICALL
Java_sse2neon_test_SSE2NEON_sse2neonNative( JNIEnv* env, jobject thiz )
{
	// Enter code here
    __android_log_print(ANDROID_LOG_INFO, "SSE2NEON", "%s\n", "SSE2NEON");


    SSE2NEON::SSE2NEONTest *test = SSE2NEON::SSE2NEONTest::create();
    uint32_t passCount = 0;
    uint32_t failedCount = 0;
    for (uint32_t i = 0; i < SSE2NEON::IT_LAST; i++)
    {
        SSE2NEON::InstructionTest it = SSE2NEON::InstructionTest(i);
        __android_log_print(ANDROID_LOG_INFO, "SSE2NEON", "Running Test %s\n", SSE2NEON::SSE2NEONTest::getInstructionTestString(it));
        bool ok = test->runTest(it);
        // If the test fails, we will run it again so we can step into the debugger and figure out why!
        if (!ok)
        {
            __android_log_print(ANDROID_LOG_INFO, "SSE2NEON", "**FAILURE** SSE2NEONTest %s", SSE2NEON::SSE2NEONTest::getInstructionTestString(it));
            //            test->runTest(it); // Uncomment this to step through the code to find the failure case
        }
        if (ok)
        {
            passCount++;
        }
        else
        {
            failedCount++;
        }
    }
    test->release();
    __android_log_print(ANDROID_LOG_INFO, "SSE2NEON", "SSE2NEONTest Complete: Passed %d tests : Failed %d\n", passCount, failedCount);
}
