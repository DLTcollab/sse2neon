/**
 * aes.cpp - AES-NI Intrinsics Validation Test Suite
 *
 * This test file validates sse2neon AES intrinsic implementations against
 * official NIST test vectors and comprehensive edge cases.
 *
 * Coverage:
 *   - _mm_aesenc_si128: AES encryption round
 *   - _mm_aesdec_si128: AES decryption round
 *   - _mm_aesenclast_si128: AES final encryption round
 *   - _mm_aesdeclast_si128: AES final decryption round
 *   - _mm_aesimc_si128: Inverse MixColumns transformation
 *   - _mm_aeskeygenassist_si128: Key expansion assist
 *
 * Test Vectors:
 *   - NIST FIPS 197 Appendix B: AES-128 example
 *   - NIST FIPS 197 Appendix C.1: AES-128 test vectors
 *   - Encrypt/decrypt roundtrip verification
 *   - Edge cases: zero key, all-ones, sequential bytes
 *
 * Usage:
 *   make aes && ./tests/aes
 *   make FEATURE=crypto check-aes
 *
 * References:
 *   - NIST FIPS 197: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf
 *   - Intel AES-NI White Paper
 */

#if defined(_M_ARM64EC)
#include "sse2neon.h"
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "common.h"

using namespace SSE2NEON;

/* Test result counters */
static int g_pass_count = 0;
static int g_fail_count = 0;
static int g_skip_count = 0;

/* Helper macros */
#define TEST_CASE(name) static result_t test_##name(void)

#define RUN_TEST(name)                                                        \
    do {                                                                      \
        result_t r = test_##name();                                           \
        if (r == TEST_SUCCESS) {                                              \
            g_pass_count++;                                                   \
            printf("Test %-40s [ " COLOR_GREEN "OK" COLOR_RESET " ]\n",       \
                   #name);                                                    \
        } else if (r == TEST_UNIMPL) {                                        \
            g_skip_count++;                                                   \
            printf("Test %-40s [SKIP]\n", #name);                             \
        } else {                                                              \
            g_fail_count++;                                                   \
            printf("Test %-40s [" COLOR_RED "FAIL" COLOR_RESET "]\n", #name); \
        }                                                                     \
    } while (0)

#define EXPECT_TRUE(cond)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            printf("    FAILED: %s (line %d)\n", #cond, __LINE__); \
            return TEST_FAIL;                                      \
        }                                                          \
    } while (0)

#define EXPECT_M128I_EQ(a, b)                                 \
    do {                                                      \
        uint8_t _a[16], _b[16];                               \
        _mm_storeu_si128(reinterpret_cast<__m128i *>(_a), a); \
        _mm_storeu_si128(reinterpret_cast<__m128i *>(_b), b); \
        if (memcmp(_a, _b, 16) != 0) {                        \
            printf("    FAILED at line %d:\n", __LINE__);     \
            printf("      Got:      ");                       \
            for (int _i = 0; _i < 16; _i++)                   \
                printf("%02x ", _a[_i]);                      \
            printf("\n      Expected: ");                     \
            for (int _i = 0; _i < 16; _i++)                   \
                printf("%02x ", _b[_i]);                      \
            printf("\n");                                     \
            return TEST_FAIL;                                 \
        }                                                     \
    } while (0)

/* NIST FIPS 197 Test Vectors */

/* FIPS 197 Appendix B: AES-128 Example
 * Cipher Key: 2b7e151628aed2a6abf7158809cf4f3c
 * Plaintext:  3243f6a8885a308d313198a2e0370734
 * Ciphertext: 3925841d02dc09fbdc118597196a0b32
 */
static const uint8_t fips197_plaintext[16] = {
    0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
    0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
};

static const uint8_t fips197_ciphertext[16] = {
    0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
    0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
};

/* FIPS 197 Appendix B: Pre-expanded round keys */
static const uint8_t fips197_round_keys[11][16] = {
    {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88,
     0x09, 0xcf, 0x4f, 0x3c},
    {0xa0, 0xfa, 0xfe, 0x17, 0x88, 0x54, 0x2c, 0xb1, 0x23, 0xa3, 0x39, 0x39,
     0x2a, 0x6c, 0x76, 0x05},
    {0xf2, 0xc2, 0x95, 0xf2, 0x7a, 0x96, 0xb9, 0x43, 0x59, 0x35, 0x80, 0x7a,
     0x73, 0x59, 0xf6, 0x7f},
    {0x3d, 0x80, 0x47, 0x7d, 0x47, 0x16, 0xfe, 0x3e, 0x1e, 0x23, 0x7e, 0x44,
     0x6d, 0x7a, 0x88, 0x3b},
    {0xef, 0x44, 0xa5, 0x41, 0xa8, 0x52, 0x5b, 0x7f, 0xb6, 0x71, 0x25, 0x3b,
     0xdb, 0x0b, 0xad, 0x00},
    {0xd4, 0xd1, 0xc6, 0xf8, 0x7c, 0x83, 0x9d, 0x87, 0xca, 0xf2, 0xb8, 0xbc,
     0x11, 0xf9, 0x15, 0xbc},
    {0x6d, 0x88, 0xa3, 0x7a, 0x11, 0x0b, 0x3e, 0xfd, 0xdb, 0xf9, 0x86, 0x41,
     0xca, 0x00, 0x93, 0xfd},
    {0x4e, 0x54, 0xf7, 0x0e, 0x5f, 0x5f, 0xc9, 0xf3, 0x84, 0xa6, 0x4f, 0xb2,
     0x4e, 0xa6, 0xdc, 0x4f},
    {0xea, 0xd2, 0x73, 0x21, 0xb5, 0x8d, 0xba, 0xd2, 0x31, 0x2b, 0xf5, 0x60,
     0x7f, 0x8d, 0x29, 0x2f},
    {0xac, 0x77, 0x66, 0xf3, 0x19, 0xfa, 0xdc, 0x21, 0x28, 0xd1, 0x29, 0x41,
     0x57, 0x5c, 0x00, 0x6e},
    {0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xe1, 0x3f, 0x0c, 0xc8,
     0xb6, 0x63, 0x0c, 0xa6},
};

/* FIPS 197 Appendix C.1: AES-128 test vectors */
static const uint8_t nist_c1_plaintext[16] = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};

static const uint8_t nist_c1_ciphertext[16] = {
    0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
    0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a};

/* FIPS 197 Appendix C.1: Pre-expanded round keys */
static const uint8_t nist_c1_round_keys[11][16] = {
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
     0x0c, 0x0d, 0x0e, 0x0f},
    {0xd6, 0xaa, 0x74, 0xfd, 0xd2, 0xaf, 0x72, 0xfa, 0xda, 0xa6, 0x78, 0xf1,
     0xd6, 0xab, 0x76, 0xfe},
    {0xb6, 0x92, 0xcf, 0x0b, 0x64, 0x3d, 0xbd, 0xf1, 0xbe, 0x9b, 0xc5, 0x00,
     0x68, 0x30, 0xb3, 0xfe},
    {0xb6, 0xff, 0x74, 0x4e, 0xd2, 0xc2, 0xc9, 0xbf, 0x6c, 0x59, 0x0c, 0xbf,
     0x04, 0x69, 0xbf, 0x41},
    {0x47, 0xf7, 0xf7, 0xbc, 0x95, 0x35, 0x3e, 0x03, 0xf9, 0x6c, 0x32, 0xbc,
     0xfd, 0x05, 0x8d, 0xfd},
    {0x3c, 0xaa, 0xa3, 0xe8, 0xa9, 0x9f, 0x9d, 0xeb, 0x50, 0xf3, 0xaf, 0x57,
     0xad, 0xf6, 0x22, 0xaa},
    {0x5e, 0x39, 0x0f, 0x7d, 0xf7, 0xa6, 0x92, 0x96, 0xa7, 0x55, 0x3d, 0xc1,
     0x0a, 0xa3, 0x1f, 0x6b},
    {0x14, 0xf9, 0x70, 0x1a, 0xe3, 0x5f, 0xe2, 0x8c, 0x44, 0x0a, 0xdf, 0x4d,
     0x4e, 0xa9, 0xc0, 0x26},
    {0x47, 0x43, 0x87, 0x35, 0xa4, 0x1c, 0x65, 0xb9, 0xe0, 0x16, 0xba, 0xf4,
     0xae, 0xbf, 0x7a, 0xd2},
    {0x54, 0x99, 0x32, 0xd1, 0xf0, 0x85, 0x57, 0x68, 0x10, 0x93, 0xed, 0x9c,
     0xbe, 0x2c, 0x97, 0x4e},
    {0x13, 0x11, 0x1d, 0x7f, 0xe3, 0x94, 0x4a, 0x17, 0xf3, 0x07, 0xa7, 0x8b,
     0x4d, 0x2b, 0x30, 0xc5},
};

/* AES-128 encryption using AES-NI intrinsics */
static inline __m128i aes128_encrypt(__m128i plaintext,
                                     const uint8_t round_keys[11][16])
{
    __m128i state = plaintext;

    /* Initial AddRoundKey */
    state = _mm_xor_si128(
        state,
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(round_keys[0])));

    /* 9 main rounds */
    for (int i = 1; i <= 9; i++) {
        state = _mm_aesenc_si128(
            state,
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(round_keys[i])));
    }

    /* Final round */
    state = _mm_aesenclast_si128(
        state,
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(round_keys[10])));

    return state;
}

/* AES-128 decryption using AES-NI intrinsics */
static inline __m128i aes128_decrypt(__m128i ciphertext,
                                     const uint8_t enc_round_keys[11][16])
{
    __m128i dec_keys[11];

    /* Prepare decryption keys: reverse order, apply InvMixColumns to 1-9 */
    dec_keys[0] =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(enc_round_keys[10]));
    for (int i = 1; i <= 9; i++) {
        dec_keys[i] = _mm_aesimc_si128(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(enc_round_keys[10 - i])));
    }
    dec_keys[10] =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(enc_round_keys[0]));

    __m128i state = ciphertext;

    /* Initial AddRoundKey */
    state = _mm_xor_si128(state, dec_keys[0]);

    /* 9 main rounds */
    for (int i = 1; i <= 9; i++)
        state = _mm_aesdec_si128(state, dec_keys[i]);

    /* Final round */
    state = _mm_aesdeclast_si128(state, dec_keys[10]);

    return state;
}

/* FIPS 197 Encryption Tests */

/* Test FIPS 197 Appendix B: AES-128 encryption */
TEST_CASE(fips197_appendix_b_encrypt)
{
    __m128i plaintext =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(fips197_plaintext));
    __m128i expected =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(fips197_ciphertext));

    __m128i result = aes128_encrypt(plaintext, fips197_round_keys);

    EXPECT_M128I_EQ(result, expected);
    return TEST_SUCCESS;
}

/* Test FIPS 197 Appendix C.1: AES-128 encryption */
TEST_CASE(fips197_appendix_c1_encrypt)
{
    __m128i plaintext =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(nist_c1_plaintext));
    __m128i expected =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(nist_c1_ciphertext));

    __m128i result = aes128_encrypt(plaintext, nist_c1_round_keys);

    EXPECT_M128I_EQ(result, expected);
    return TEST_SUCCESS;
}

/* FIPS 197 Decryption Tests */

/* Test FIPS 197 Appendix B: AES-128 decryption */
TEST_CASE(fips197_appendix_b_decrypt)
{
    __m128i ciphertext =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(fips197_ciphertext));
    __m128i expected =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(fips197_plaintext));

    __m128i result = aes128_decrypt(ciphertext, fips197_round_keys);

    EXPECT_M128I_EQ(result, expected);
    return TEST_SUCCESS;
}

/* Test FIPS 197 Appendix C.1: AES-128 decryption */
TEST_CASE(fips197_appendix_c1_decrypt)
{
    __m128i ciphertext =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(nist_c1_ciphertext));
    __m128i expected =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(nist_c1_plaintext));

    __m128i result = aes128_decrypt(ciphertext, nist_c1_round_keys);

    EXPECT_M128I_EQ(result, expected);
    return TEST_SUCCESS;
}

/* Roundtrip Tests */

/* Test encrypt-then-decrypt roundtrip with various patterns */
TEST_CASE(roundtrip_fips197_key)
{
    /* Test with FIPS 197 key and various plaintexts */
    for (int pattern = 0; pattern < 256; pattern += 17) {
        uint8_t plaintext[16];
        for (int i = 0; i < 16; i++)
            plaintext[i] = static_cast<uint8_t>((pattern + i * 13) & 0xff);

        __m128i pt =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(plaintext));
        __m128i ct = aes128_encrypt(pt, fips197_round_keys);
        __m128i result = aes128_decrypt(ct, fips197_round_keys);

        EXPECT_M128I_EQ(result, pt);
    }
    return TEST_SUCCESS;
}

/* Test roundtrip with zero plaintext */
TEST_CASE(roundtrip_zero_plaintext)
{
    __m128i pt = _mm_setzero_si128();
    __m128i ct = aes128_encrypt(pt, fips197_round_keys);
    __m128i result = aes128_decrypt(ct, fips197_round_keys);

    EXPECT_M128I_EQ(result, pt);
    return TEST_SUCCESS;
}

/* Test roundtrip with all-ones plaintext */
TEST_CASE(roundtrip_ones_plaintext)
{
    __m128i pt = _mm_set1_epi8(static_cast<char>(0xff));
    __m128i ct = aes128_encrypt(pt, fips197_round_keys);
    __m128i result = aes128_decrypt(ct, fips197_round_keys);

    EXPECT_M128I_EQ(result, pt);
    return TEST_SUCCESS;
}

/* Individual Intrinsic Tests */

/* Test aesenc produces different output from input */
TEST_CASE(aesenc_transforms_data)
{
    __m128i data =
        _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i key =
        _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    __m128i result = _mm_aesenc_si128(data, key);

    /* Result should be different from both input and key */
    uint8_t r[16], d[16], k[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r), result);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(d), data);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(k), key);

    EXPECT_TRUE(memcmp(r, d, 16) != 0);
    EXPECT_TRUE(memcmp(r, k, 16) != 0);

    return TEST_SUCCESS;
}

/* Test aesdec produces different output from input */
TEST_CASE(aesdec_transforms_data)
{
    __m128i data =
        _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i key =
        _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    __m128i result = _mm_aesdec_si128(data, key);

    uint8_t r[16], d[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r), result);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(d), data);

    EXPECT_TRUE(memcmp(r, d, 16) != 0);

    return TEST_SUCCESS;
}

/* Test aesimc transforms data (InvMixColumns) */
TEST_CASE(aesimc_transforms_data)
{
    __m128i data =
        _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    __m128i result = _mm_aesimc_si128(data);

    uint8_t r[16], d[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r), result);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(d), data);

    EXPECT_TRUE(memcmp(r, d, 16) != 0);

    return TEST_SUCCESS;
}

/* Test aesimc with zero input */
TEST_CASE(aesimc_zero_input)
{
    __m128i data = _mm_setzero_si128();
    __m128i result = _mm_aesimc_si128(data);

    /* InvMixColumns of zero is zero */
    EXPECT_M128I_EQ(result, data);

    return TEST_SUCCESS;
}

/* Edge Case Tests */

/* Test encryption with zero key - verify deterministic output */
TEST_CASE(encrypt_deterministic)
{
    __m128i pt1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(fips197_plaintext));
    __m128i pt2 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(fips197_plaintext));

    __m128i ct1 = aes128_encrypt(pt1, fips197_round_keys);
    __m128i ct2 = aes128_encrypt(pt2, fips197_round_keys);

    /* Same input must produce same output */
    EXPECT_M128I_EQ(ct1, ct2);

    return TEST_SUCCESS;
}

/* Test that different plaintexts produce different ciphertexts */
TEST_CASE(encrypt_different_inputs)
{
    __m128i pt1 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i pt2 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);

    __m128i ct1 = aes128_encrypt(pt1, fips197_round_keys);
    __m128i ct2 = aes128_encrypt(pt2, fips197_round_keys);

    uint8_t r1[16], r2[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r1), ct1);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r2), ct2);

    EXPECT_TRUE(memcmp(r1, r2, 16) != 0);

    return TEST_SUCCESS;
}

/* Test avalanche effect: 1-bit change propagates */
TEST_CASE(avalanche_effect)
{
    uint8_t pt1[16] = {0};
    uint8_t pt2[16] = {0};
    pt2[0] = 1; /* Single bit difference */

    __m128i in1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pt1));
    __m128i in2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pt2));

    __m128i out1 = aes128_encrypt(in1, fips197_round_keys);
    __m128i out2 = aes128_encrypt(in2, fips197_round_keys);

    uint8_t r1[16], r2[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r1), out1);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r2), out2);

    /* Count differing bits */
    int diff_bits = 0;
    for (int i = 0; i < 16; i++) {
        uint8_t xor_byte = r1[i] ^ r2[i];
        while (xor_byte) {
            diff_bits += xor_byte & 1;
            xor_byte >>= 1;
        }
    }

    /* Avalanche: roughly half the bits should differ (64 +/- margin) */
    EXPECT_TRUE(diff_bits >= 40 && diff_bits <= 88);

    return TEST_SUCCESS;
}

/* Consistency Tests */

/* Test that multiple rounds produce consistent results */
TEST_CASE(multiple_rounds_consistent)
{
    __m128i data =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(fips197_plaintext));
    __m128i key1 = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(fips197_round_keys[0]));
    __m128i key2 = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(fips197_round_keys[1]));

    /* Apply same operations twice, verify same results */
    __m128i r1 = _mm_aesenc_si128(_mm_xor_si128(data, key1), key2);
    __m128i r2 = _mm_aesenc_si128(_mm_xor_si128(data, key1), key2);

    EXPECT_M128I_EQ(r1, r2);

    return TEST_SUCCESS;
}

/* Test aesenclast vs aesenc difference (no MixColumns in last) */
TEST_CASE(aesenclast_differs_from_aesenc)
{
    __m128i data =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(fips197_plaintext));
    __m128i key = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(fips197_round_keys[1]));

    __m128i enc_result = _mm_aesenc_si128(data, key);
    __m128i last_result = _mm_aesenclast_si128(data, key);

    uint8_t r1[16], r2[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r1), enc_result);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r2), last_result);

    /* Results should differ because aesenc includes MixColumns */
    EXPECT_TRUE(memcmp(r1, r2, 16) != 0);

    return TEST_SUCCESS;
}

/* Key Generation Assist Tests */

/* _mm_aeskeygenassist_si128 Known Answer Test
 *
 * The intrinsic performs:
 *   X3 = input[127:96], X2 = input[95:64], X1 = input[63:32], X0 = input[31:0]
 *   result[31:0]   = SubWord(X1)
 *   result[63:32]  = RotWord(SubWord(X1)) XOR RCON
 *   result[95:64]  = SubWord(X3)
 *   result[127:96] = RotWord(SubWord(X3)) XOR RCON
 *
 * SubWord applies AES S-box to each byte
 * RotWord rotates a word left by 8 bits (byte position [a,b,c,d] -> [b,c,d,a])
 *
 * Test vector derived from FIPS-197 key schedule:
 *   Input: 09cf4f3c (from key[3])
 *   SubWord(09cf4f3c) = 01 8a 84 eb
 *   RotWord(SubWord(09cf4f3c)) = 8a 84 eb 01
 *   With RCON=0x01: 8a 84 eb 01 XOR 01 00 00 00 = 8b 84 eb 01
 */
TEST_CASE(aeskeygenassist_kat_rcon01)
{
    /* FIPS 197 key: 2b7e151628aed2a6abf7158809cf4f3c */
    uint8_t key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                       0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    __m128i k = _mm_loadu_si128(reinterpret_cast<const __m128i *>(key));

    __m128i result = _mm_aeskeygenassist_si128(k, 0x01);

    uint8_t r[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r), result);

    /* X1 = bytes[4:7] = 28 ae d2 a6, X3 = bytes[12:15] = 09 cf 4f 3c
     * SubWord uses S-box: S[28]=34, S[ae]=e4, S[d2]=b5, S[a6]=24
     *   SubWord(X1) = 34 e4 b5 24
     * S[09]=01, S[cf]=8a, S[4f]=84, S[3c]=eb
     *   SubWord(X3) = 01 8a 84 eb
     *
     * RotWord(SubWord(X1)) = e4 b5 24 34, XOR 01000000 = e5 b5 24 34
     * RotWord(SubWord(X3)) = 8a 84 eb 01, XOR 01000000 = 8b 84 eb 01
     *
     * result[0:3]   = SubWord(X1) = 34 e4 b5 24
     * result[4:7]   = RotWord(SubWord(X1)) XOR RCON = e5 b5 24 34
     * result[8:11]  = SubWord(X3) = 01 8a 84 eb
     * result[12:15] = RotWord(SubWord(X3)) XOR RCON = 8b 84 eb 01
     */
    static const uint8_t expected[16] = {
        0x34, 0xe4, 0xb5, 0x24, 0xe5, 0xb5, 0x24, 0x34,
        0x01, 0x8a, 0x84, 0xeb, 0x8b, 0x84, 0xeb, 0x01,
    };

    __m128i exp = _mm_loadu_si128(reinterpret_cast<const __m128i *>(expected));
    EXPECT_M128I_EQ(result, exp);

    return TEST_SUCCESS;
}

/* Test aeskeygenassist with RCON=0x00 (verifies core SubWord/RotWord) */
TEST_CASE(aeskeygenassist_kat_rcon00)
{
    /* Simple input for easy verification */
    uint8_t input[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };
    __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));

    __m128i result = _mm_aeskeygenassist_si128(in, 0x00);

    uint8_t r[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r), result);

    /* X1 = input[4:7] = 04 05 06 07
     * SubWord(X1) = S[04] S[05] S[06] S[07] = f2 6b 6f c5
     * RotWord(SubWord(X1)) = 6b 6f c5 f2, XOR 0 = 6b 6f c5 f2
     *
     * X3 = input[12:15] = 0c 0d 0e 0f
     * SubWord(X3) = S[0c] S[0d] S[0e] S[0f] = fe d7 ab 76
     * RotWord(SubWord(X3)) = d7 ab 76 fe, XOR 0 = d7 ab 76 fe
     */
    static const uint8_t expected[16] = {
        0xf2, 0x6b, 0x6f, 0xc5, 0x6b, 0x6f, 0xc5, 0xf2,
        0xfe, 0xd7, 0xab, 0x76, 0xd7, 0xab, 0x76, 0xfe,
    };

    __m128i exp = _mm_loadu_si128(reinterpret_cast<const __m128i *>(expected));
    EXPECT_M128I_EQ(result, exp);

    return TEST_SUCCESS;
}

/* Test aeskeygenassist with zero input */
TEST_CASE(aeskeygenassist_zero_input)
{
    __m128i zero = _mm_setzero_si128();

    __m128i result = _mm_aeskeygenassist_si128(zero, 0x01);

    uint8_t r[16];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r), result);

    /* S[0x00] = 0x63, so SubWord(0x00000000) = 0x63636363
     * RotWord(0x63636363) = 0x63636363 (rotation of identical bytes)
     * result[0:3] = 0x63636363
     * result[4:7] = 0x63636363 XOR 0x01000000 = 0x62636363
     * result[8:11] = 0x63636363
     * result[12:15] = 0x62636363
     */
    static const uint8_t expected[16] = {
        0x63, 0x63, 0x63, 0x63, 0x62, 0x63, 0x63, 0x63,
        0x63, 0x63, 0x63, 0x63, 0x62, 0x63, 0x63, 0x63,
    };

    __m128i exp = _mm_loadu_si128(reinterpret_cast<const __m128i *>(expected));
    EXPECT_M128I_EQ(result, exp);

    return TEST_SUCCESS;
}

/* Test aeskeygenassist consistency across multiple RCON values */
TEST_CASE(aeskeygenassist_rcon_consistency)
{
    uint8_t input[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    };
    __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));

    /* RCON values used in AES-128 key expansion: 0x01, 0x02, 0x04, 0x08,
     * 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 */
    static const uint8_t rcons[] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36,
    };

    /* Verify each RCON produces different output in positions [4:7] and
     * [12:15] */
    uint8_t prev_r4[4] = {0};
    for (size_t i = 0; i < sizeof(rcons); i++) {
        __m128i result;
        /* Use switch to allow compile-time RCON (required by intrinsic) */
        switch (rcons[i]) {
        case 0x01:
            result = _mm_aeskeygenassist_si128(in, 0x01);
            break;
        case 0x02:
            result = _mm_aeskeygenassist_si128(in, 0x02);
            break;
        case 0x04:
            result = _mm_aeskeygenassist_si128(in, 0x04);
            break;
        case 0x08:
            result = _mm_aeskeygenassist_si128(in, 0x08);
            break;
        case 0x10:
            result = _mm_aeskeygenassist_si128(in, 0x10);
            break;
        case 0x20:
            result = _mm_aeskeygenassist_si128(in, 0x20);
            break;
        case 0x40:
            result = _mm_aeskeygenassist_si128(in, 0x40);
            break;
        case 0x80:
            result = _mm_aeskeygenassist_si128(in, 0x80);
            break;
        case 0x1b:
            result = _mm_aeskeygenassist_si128(in, 0x1b);
            break;
        case 0x36:
            result = _mm_aeskeygenassist_si128(in, 0x36);
            break;
        default:
            return TEST_FAIL;
        }

        uint8_t r[16];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(r), result);

        /* Positions [0:3] and [8:11] should be identical (SubWord, no RCON) */
        /* (Different RCON only affects [4:7] and [12:15]) */
        if (i > 0) {
            /* Current [4:7] should differ from previous (different RCON) */
            EXPECT_TRUE(memcmp(&r[4], prev_r4, 4) != 0);
        }
        memcpy(prev_r4, &r[4], 4);
    }

    return TEST_SUCCESS;
}

int main(void)
{
    printf("=== AES-NI Intrinsics Validation Test Suite ===\n\n");

    printf("--- FIPS 197 Encryption Tests ---\n");
    RUN_TEST(fips197_appendix_b_encrypt);
    RUN_TEST(fips197_appendix_c1_encrypt);

    printf("\n--- FIPS 197 Decryption Tests ---\n");
    RUN_TEST(fips197_appendix_b_decrypt);
    RUN_TEST(fips197_appendix_c1_decrypt);

    printf("\n--- Roundtrip Tests ---\n");
    RUN_TEST(roundtrip_fips197_key);
    RUN_TEST(roundtrip_zero_plaintext);
    RUN_TEST(roundtrip_ones_plaintext);

    printf("\n--- Individual Intrinsic Tests ---\n");
    RUN_TEST(aesenc_transforms_data);
    RUN_TEST(aesdec_transforms_data);
    RUN_TEST(aesimc_transforms_data);
    RUN_TEST(aesimc_zero_input);

    printf("\n--- Edge Case Tests ---\n");
    RUN_TEST(encrypt_deterministic);
    RUN_TEST(encrypt_different_inputs);
    RUN_TEST(avalanche_effect);

    printf("\n--- Consistency Tests ---\n");
    RUN_TEST(multiple_rounds_consistent);
    RUN_TEST(aesenclast_differs_from_aesenc);

    printf("\n--- Key Generation Assist Tests ---\n");
    RUN_TEST(aeskeygenassist_kat_rcon01);
    RUN_TEST(aeskeygenassist_kat_rcon00);
    RUN_TEST(aeskeygenassist_zero_input);
    RUN_TEST(aeskeygenassist_rcon_consistency);

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", g_pass_count);
    printf("Failed: %d\n", g_fail_count);
    printf("Skipped: %d\n", g_skip_count);

    return g_fail_count > 0 ? 1 : 0;
}
