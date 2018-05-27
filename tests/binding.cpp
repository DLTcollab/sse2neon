#include "binding.h"

#ifdef WIN32
#include <emmintrin.h>
#include <xmmintrin.h>

#include <crtdbg.h>
#include <malloc.h>

#else

#include <stdio.h>
#include <stdlib.h>

#endif

namespace SSE2NEON
{
#ifdef WIN32
void *platformAlignedAlloc(size_t size)
{
    return _aligned_malloc(size, 16);
}

void platformAlignedFree(void *ptr)
{
    _aligned_free(ptr);
}

#else

void *platformAlignedAlloc(size_t size)
{
    // return ::memalign(16, size);
    void *address;
    int ret = posix_memalign(&address, 16, size);
    if (ret != 0) {
        fprintf(stderr, "Error at File %s line number %d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    return address;
}

void platformAlignedFree(void *ptr)
{
    free(ptr);
}

#endif

}  // namespace SSE2NEON
