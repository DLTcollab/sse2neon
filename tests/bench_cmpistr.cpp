/**
 * Benchmark for the _SIDD_CMP_EQUAL_ANY string-comparison intrinsics.
 *
 * Measures three dimensions:
 *   1. Throughput: independent calls (pipeline utilization)
 *   2. Latency: dependent chain (true instruction latency)
 *   3. In-context: strcspn-like delimiter scan (realistic usage)
 *
 * EQUAL_ANY is the relation SVE2 MATCH computes directly, so this is the
 * benchmark that answers whether SSE2NEON_ENABLE_SVE is worth defining.
 * Build the same source twice and compare:
 *
 *   make bench-cmpistr                                     # NEON path
 *   make bench-cmpistr SVE=1 FEATURE=sve2                  # SVE2 path
 *   make bench-cmpistr CROSS_COMPILE=aarch64-linux-gnu-    # AArch64 + QEMU
 *
 * The set operand is null-terminated below 16 bytes, because that is what a
 * delimiter set looks like in practice and it is also the case the SVE2 path
 * has to pad: MATCH scans the whole 128-bit segment, so lanes at and beyond
 * la are filled with a[0].
 */

#include <benchmark/benchmark.h>

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
#include "sse2neon.h"
#else
#include <nmmintrin.h>
#endif

#include <cstdint>
#include <cstring>

/* Simple xorshift32 PRNG for reproducible random data. */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

const int BYTE_MODE = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY;
const int WORD_MODE = _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY;

const int N_DATA = 1024;

/* The needle: a five-character delimiter set, null-terminated so the implicit
 * length is 5 rather than 16. */
__m128i set_byte;
__m128i set_word;

/* Haystacks, by where the first set member falls: nowhere, lane 15, lane 0.
 * The three differ in how much of the segment the aggregation has to walk. */
__m128i data_nomatch[N_DATA];
__m128i data_late[N_DATA];
__m128i data_early[N_DATA];
__m128i data_rand[N_DATA];

void init_data()
{
    volatile char dyn_zero = 0;  // Prevent constant-folding at -O3

    const char delims[16] = {' ', '\t', '\n', ',', ';', 0, 0, 0,
                             0,   0,    0,    0,   0,   0, 0, 0};
    set_byte = _mm_loadu_si128(reinterpret_cast<const __m128i *>(delims));

    const int16_t wdelims[8] = {' ', '\t', '\n', ',', ';', 0, 0, 0};
    set_word = _mm_loadu_si128(reinterpret_cast<const __m128i *>(wdelims));

    uint32_t rng = 42;
    for (int i = 0; i < N_DATA; i++) {
        uint32_t r[4];
        for (int j = 0; j < 4; j++)
            r[j] = xorshift32(&rng);
        data_rand[i] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r));

        /* 'x' is in neither set, so these are built from it and then seeded
         * with one delimiter. dyn_zero keeps them off the constant path. */
        char none[16], late[16], early[16];
        for (int j = 0; j < 16; j++)
            none[j] = late[j] = early[j] = static_cast<char>(dyn_zero | 'x');
        late[15] = ',';
        early[0] = ',';

        data_nomatch[i] =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(none));
        data_late[i] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(late));
        data_early[i] =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(early));
    }
}

static void BM_Throughput_Index_NoMatch(benchmark::State &state)
{
    unsigned int acc[4] = {0};
    int i = 0;
    for (auto _ : state) {
        acc[i & 3] += static_cast<unsigned int>(
            _mm_cmpistri(set_byte, data_nomatch[i & (N_DATA - 1)], BYTE_MODE));
        i++;
    }
    benchmark::DoNotOptimize(acc[0] + acc[1] + acc[2] + acc[3]);
}
BENCHMARK(BM_Throughput_Index_NoMatch);

static void BM_Throughput_Index_MatchLate(benchmark::State &state)
{
    unsigned int acc[4] = {0};
    int i = 0;
    for (auto _ : state) {
        acc[i & 3] += static_cast<unsigned int>(
            _mm_cmpistri(set_byte, data_late[i & (N_DATA - 1)], BYTE_MODE));
        i++;
    }
    benchmark::DoNotOptimize(acc[0] + acc[1] + acc[2] + acc[3]);
}
BENCHMARK(BM_Throughput_Index_MatchLate);

static void BM_Throughput_Index_MatchEarly(benchmark::State &state)
{
    unsigned int acc[4] = {0};
    int i = 0;
    for (auto _ : state) {
        acc[i & 3] += static_cast<unsigned int>(
            _mm_cmpistri(set_byte, data_early[i & (N_DATA - 1)], BYTE_MODE));
        i++;
    }
    benchmark::DoNotOptimize(acc[0] + acc[1] + acc[2] + acc[3]);
}
BENCHMARK(BM_Throughput_Index_MatchEarly);

static void BM_Throughput_Index_Random(benchmark::State &state)
{
    unsigned int acc[4] = {0};
    int i = 0;
    for (auto _ : state) {
        acc[i & 3] += static_cast<unsigned int>(
            _mm_cmpistri(set_byte, data_rand[i & (N_DATA - 1)], BYTE_MODE));
        i++;
    }
    benchmark::DoNotOptimize(acc[0] + acc[1] + acc[2] + acc[3]);
}
BENCHMARK(BM_Throughput_Index_Random);

static void BM_Throughput_Mask_Random(benchmark::State &state)
{
    __m128i acc = _mm_setzero_si128();
    int i = 0;
    for (auto _ : state) {
        acc = _mm_xor_si128(
            acc,
            _mm_cmpistrm(set_byte, data_rand[i & (N_DATA - 1)], BYTE_MODE));
        i++;
    }
    benchmark::DoNotOptimize(acc);
}
BENCHMARK(BM_Throughput_Mask_Random);

/* The word form takes the other MATCH element size and half the lanes. */
static void BM_Throughput_Index_Word_Random(benchmark::State &state)
{
    unsigned int acc[4] = {0};
    int i = 0;
    for (auto _ : state) {
        acc[i & 3] += static_cast<unsigned int>(
            _mm_cmpistri(set_word, data_rand[i & (N_DATA - 1)], WORD_MODE));
        i++;
    }
    benchmark::DoNotOptimize(acc[0] + acc[1] + acc[2] + acc[3]);
}
BENCHMARK(BM_Throughput_Index_Word_Random);

/* Each call consumes the previous result, so these measure latency rather
 * than how many the pipeline keeps in flight. */
static void BM_Latency_Index(benchmark::State &state)
{
    __m128i vec = data_rand[0];
    for (auto _ : state) {
        int index = _mm_cmpistri(set_byte, vec, BYTE_MODE);
        /* Folded back in so the next call cannot start before this one
         * retires, which is the whole point of the chain. */
        vec = _mm_xor_si128(vec, _mm_set1_epi8(static_cast<char>(index)));
    }
    benchmark::DoNotOptimize(vec);
}
BENCHMARK(BM_Latency_Index);

static void BM_Latency_Mask(benchmark::State &state)
{
    __m128i vec = data_rand[0];
    for (auto _ : state) {
        vec = _mm_cmpistrm(set_byte, vec, BYTE_MODE);
    }
    benchmark::DoNotOptimize(vec);
}
BENCHMARK(BM_Latency_Mask);

char text[4096];

/* strcspn: walk until a character from the set turns up. This is the loop
 * EQUAL_ANY exists for, and the one a tokenizer actually runs. */
static int strcspn_sse42(const char *buf, int len, __m128i set)
{
    for (int i = 0; i <= len - 16; i += 16) {
        __m128i chunk =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(buf + i));
        int index = _mm_cmpistri(set, chunk, BYTE_MODE);
        if (index != 16)
            return i + index;
    }
    return len;
}

static void BM_Strcspn_FoundAt2048(benchmark::State &state)
{
    memset(text, 'x', sizeof(text));
    text[2048] = ',';
    for (auto _ : state) {
        benchmark::DoNotOptimize(strcspn_sse42(text, sizeof(text), set_byte));
    }
}
BENCHMARK(BM_Strcspn_FoundAt2048);

static void BM_Strcspn_NotFound(benchmark::State &state)
{
    memset(text, 'x', sizeof(text));
    for (auto _ : state) {
        benchmark::DoNotOptimize(strcspn_sse42(text, sizeof(text), set_byte));
    }
}
BENCHMARK(BM_Strcspn_NotFound);

static void BM_Strcspn_FoundAt0(benchmark::State &state)
{
    memset(text, 'x', sizeof(text));
    text[0] = ',';
    for (auto _ : state) {
        benchmark::DoNotOptimize(strcspn_sse42(text, sizeof(text), set_byte));
    }
}
BENCHMARK(BM_Strcspn_FoundAt0);

int main(int argc, char **argv)
{
    init_data();
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
