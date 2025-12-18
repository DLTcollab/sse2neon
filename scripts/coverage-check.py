#!/usr/bin/env python3
"""
Coverage verification for sse2neon intrinsics.

Compares implemented intrinsics in sse2neon.h against the Intel Intrinsics
Guide reference list to identify gaps and calculate coverage by instruction set.

Reference: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

Usage:
    python3 scripts/coverage-check.py [--verbose] [--badge]

CI Badge Integration:
    python3 scripts/coverage-check.py --badge  # outputs: 99.6
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Intel Intrinsics Guide reference list, organized by instruction set.
# This list covers: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AES, PCLMULQDQ
# Note: MMX is excluded as it's obsolete; sse2neon focuses on SSE family
#
# Categories:
#   - Intrinsics that sse2neon SHOULD implement (core functionality)
#   - Excluded: AVX/AVX2/AVX-512 (not in sse2neon scope)
#   - Excluded: Some deprecated/obscure MMX variants
#
# Note: This list is curated from Intel documentation, not exhaustive for
# all possible aliases. Some intrinsics have multiple names (e.g., _mm_cvtsi32_si128
# vs _mm_cvtsi32_si64) - we list the canonical forms.

INTEL_INTRINSICS = {
    # =========================================================================
    # SSE (Streaming SIMD Extensions)
    # =========================================================================
    "SSE": [
        # Arithmetic
        "_mm_add_ps",
        "_mm_add_ss",
        "_mm_sub_ps",
        "_mm_sub_ss",
        "_mm_mul_ps",
        "_mm_mul_ss",
        "_mm_div_ps",
        "_mm_div_ss",
        "_mm_sqrt_ps",
        "_mm_sqrt_ss",
        "_mm_rcp_ps",
        "_mm_rcp_ss",
        "_mm_rsqrt_ps",
        "_mm_rsqrt_ss",
        "_mm_min_ps",
        "_mm_min_ss",
        "_mm_max_ps",
        "_mm_max_ss",
        # Logical
        "_mm_and_ps",
        "_mm_andnot_ps",
        "_mm_or_ps",
        "_mm_xor_ps",
        # Comparison
        "_mm_cmpeq_ps",
        "_mm_cmpeq_ss",
        "_mm_cmplt_ps",
        "_mm_cmplt_ss",
        "_mm_cmple_ps",
        "_mm_cmple_ss",
        "_mm_cmpgt_ps",
        "_mm_cmpgt_ss",
        "_mm_cmpge_ps",
        "_mm_cmpge_ss",
        "_mm_cmpneq_ps",
        "_mm_cmpneq_ss",
        "_mm_cmpnlt_ps",
        "_mm_cmpnlt_ss",
        "_mm_cmpnle_ps",
        "_mm_cmpnle_ss",
        "_mm_cmpngt_ps",
        "_mm_cmpngt_ss",
        "_mm_cmpnge_ps",
        "_mm_cmpnge_ss",
        "_mm_cmpord_ps",
        "_mm_cmpord_ss",
        "_mm_cmpunord_ps",
        "_mm_cmpunord_ss",
        "_mm_comieq_ss",
        "_mm_comilt_ss",
        "_mm_comile_ss",
        "_mm_comigt_ss",
        "_mm_comige_ss",
        "_mm_comineq_ss",
        "_mm_ucomieq_ss",
        "_mm_ucomilt_ss",
        "_mm_ucomile_ss",
        "_mm_ucomigt_ss",
        "_mm_ucomige_ss",
        "_mm_ucomineq_ss",
        # Conversion
        "_mm_cvt_pi2ps",
        "_mm_cvt_ps2pi",
        "_mm_cvt_si2ss",
        "_mm_cvt_ss2si",
        "_mm_cvtpi8_ps",
        "_mm_cvtpi16_ps",
        "_mm_cvtpi32_ps",
        "_mm_cvtpi32x2_ps",
        "_mm_cvtps_pi8",
        "_mm_cvtps_pi16",
        "_mm_cvtps_pi32",
        "_mm_cvtpu8_ps",
        "_mm_cvtpu16_ps",
        "_mm_cvtsi32_ss",
        "_mm_cvtsi64_ss",
        "_mm_cvtss_f32",
        "_mm_cvtss_si32",
        "_mm_cvtss_si64",
        "_mm_cvtt_ps2pi",
        "_mm_cvtt_ss2si",
        "_mm_cvttps_pi32",
        "_mm_cvttss_si32",
        "_mm_cvttss_si64",
        # Load
        "_mm_load_ps",
        "_mm_load_ps1",
        "_mm_load_ss",
        "_mm_load1_ps",
        "_mm_loadh_pi",
        "_mm_loadl_pi",
        "_mm_loadr_ps",
        "_mm_loadu_ps",
        # Store
        "_mm_store_ps",
        "_mm_store_ps1",
        "_mm_store_ss",
        "_mm_store1_ps",
        "_mm_storeh_pi",
        "_mm_storel_pi",
        "_mm_storer_ps",
        "_mm_storeu_ps",
        "_mm_stream_ps",
        # Set
        "_mm_set_ps",
        "_mm_set_ps1",
        "_mm_set_ss",
        "_mm_set1_ps",
        "_mm_setr_ps",
        "_mm_setzero_ps",
        # Miscellaneous
        "_mm_move_ss",
        "_mm_movehl_ps",
        "_mm_movelh_ps",
        "_mm_movemask_ps",
        "_mm_shuffle_ps",
        "_mm_unpackhi_ps",
        "_mm_unpacklo_ps",
        "_mm_prefetch",
        "_mm_sfence",
        "_mm_getcsr",
        "_mm_setcsr",
        "_mm_get_flush_zero_mode",
        "_mm_set_flush_zero_mode",
        "_mm_get_rounding_mode",
        "_mm_set_rounding_mode",
        # Memory allocation
        "_mm_malloc",
        "_mm_free",
        # SSE integer extensions (often grouped with SSE)
        "_mm_avg_pu8",
        "_mm_avg_pu16",
        "_mm_extract_pi16",
        "_mm_insert_pi16",
        "_mm_max_pi16",
        "_mm_max_pu8",
        "_mm_min_pi16",
        "_mm_min_pu8",
        "_mm_movemask_pi8",
        "_mm_mulhi_pu16",
        "_mm_sad_pu8",
        "_mm_shuffle_pi16",
        "_mm_maskmove_si64",
        "_m_maskmovq",
        "_mm_stream_pi",
        # Undefined value
        "_mm_undefined_ps",
    ],
    # =========================================================================
    # SSE2 (Streaming SIMD Extensions 2)
    # =========================================================================
    "SSE2": [
        # Double-precision arithmetic
        "_mm_add_pd",
        "_mm_add_sd",
        "_mm_sub_pd",
        "_mm_sub_sd",
        "_mm_mul_pd",
        "_mm_mul_sd",
        "_mm_div_pd",
        "_mm_div_sd",
        "_mm_sqrt_pd",
        "_mm_sqrt_sd",
        "_mm_min_pd",
        "_mm_min_sd",
        "_mm_max_pd",
        "_mm_max_sd",
        # Double-precision logical
        "_mm_and_pd",
        "_mm_andnot_pd",
        "_mm_or_pd",
        "_mm_xor_pd",
        # Double-precision comparison
        "_mm_cmpeq_pd",
        "_mm_cmpeq_sd",
        "_mm_cmplt_pd",
        "_mm_cmplt_sd",
        "_mm_cmple_pd",
        "_mm_cmple_sd",
        "_mm_cmpgt_pd",
        "_mm_cmpgt_sd",
        "_mm_cmpge_pd",
        "_mm_cmpge_sd",
        "_mm_cmpneq_pd",
        "_mm_cmpneq_sd",
        "_mm_cmpnlt_pd",
        "_mm_cmpnlt_sd",
        "_mm_cmpnle_pd",
        "_mm_cmpnle_sd",
        "_mm_cmpngt_pd",
        "_mm_cmpngt_sd",
        "_mm_cmpnge_pd",
        "_mm_cmpnge_sd",
        "_mm_cmpord_pd",
        "_mm_cmpord_sd",
        "_mm_cmpunord_pd",
        "_mm_cmpunord_sd",
        "_mm_comieq_sd",
        "_mm_comilt_sd",
        "_mm_comile_sd",
        "_mm_comigt_sd",
        "_mm_comige_sd",
        "_mm_comineq_sd",
        "_mm_ucomieq_sd",
        "_mm_ucomilt_sd",
        "_mm_ucomile_sd",
        "_mm_ucomigt_sd",
        "_mm_ucomige_sd",
        "_mm_ucomineq_sd",
        # Integer arithmetic
        "_mm_add_epi8",
        "_mm_add_epi16",
        "_mm_add_epi32",
        "_mm_add_epi64",
        "_mm_adds_epi8",
        "_mm_adds_epi16",
        "_mm_adds_epu8",
        "_mm_adds_epu16",
        "_mm_sub_epi8",
        "_mm_sub_epi16",
        "_mm_sub_epi32",
        "_mm_sub_epi64",
        "_mm_subs_epi8",
        "_mm_subs_epi16",
        "_mm_subs_epu8",
        "_mm_subs_epu16",
        "_mm_mul_epu32",
        "_mm_mul_su32",
        "_mm_mulhi_epi16",
        "_mm_mulhi_epu16",
        "_mm_mullo_epi16",
        "_mm_madd_epi16",
        "_mm_avg_epu8",
        "_mm_avg_epu16",
        "_mm_sad_epu8",
        "_mm_min_epi16",
        "_mm_min_epu8",
        "_mm_max_epi16",
        "_mm_max_epu8",
        # Integer logical
        "_mm_and_si128",
        "_mm_andnot_si128",
        "_mm_or_si128",
        "_mm_xor_si128",
        # Integer comparison
        "_mm_cmpeq_epi8",
        "_mm_cmpeq_epi16",
        "_mm_cmpeq_epi32",
        "_mm_cmpgt_epi8",
        "_mm_cmpgt_epi16",
        "_mm_cmpgt_epi32",
        "_mm_cmplt_epi8",
        "_mm_cmplt_epi16",
        "_mm_cmplt_epi32",
        # Integer shift
        "_mm_sll_epi16",
        "_mm_sll_epi32",
        "_mm_sll_epi64",
        "_mm_slli_epi16",
        "_mm_slli_epi32",
        "_mm_slli_epi64",
        "_mm_slli_si128",
        "_mm_srl_epi16",
        "_mm_srl_epi32",
        "_mm_srl_epi64",
        "_mm_srli_epi16",
        "_mm_srli_epi32",
        "_mm_srli_epi64",
        "_mm_srli_si128",
        "_mm_sra_epi16",
        "_mm_sra_epi32",
        "_mm_srai_epi16",
        "_mm_srai_epi32",
        # Conversion
        "_mm_cvtepi32_pd",
        "_mm_cvtepi32_ps",
        "_mm_cvtpd_epi32",
        "_mm_cvtpd_pi32",
        "_mm_cvtpd_ps",
        "_mm_cvtpi32_pd",
        "_mm_cvtps_epi32",
        "_mm_cvtps_pd",
        "_mm_cvtsd_f64",
        "_mm_cvtsd_si32",
        "_mm_cvtsd_si64",
        "_mm_cvtsd_ss",
        "_mm_cvtsi32_sd",
        "_mm_cvtsi32_si128",
        "_mm_cvtsi64_sd",
        "_mm_cvtsi64_si128",
        "_mm_cvtsi64x_sd",
        "_mm_cvtsi64x_si128",
        "_mm_cvtsi128_si32",
        "_mm_cvtsi128_si64",
        "_mm_cvtsi128_si64x",
        "_mm_cvtss_sd",
        "_mm_cvttpd_epi32",
        "_mm_cvttpd_pi32",
        "_mm_cvttps_epi32",
        "_mm_cvttsd_si32",
        "_mm_cvttsd_si64",
        "_mm_cvttsd_si64x",
        # Load
        "_mm_load_pd",
        "_mm_load_pd1",
        "_mm_load_sd",
        "_mm_load1_pd",
        "_mm_loadh_pd",
        "_mm_loadl_pd",
        "_mm_loadr_pd",
        "_mm_loadu_pd",
        "_mm_load_si128",
        "_mm_loadu_si128",
        "_mm_loadu_si16",
        "_mm_loadu_si32",
        "_mm_loadu_si64",
        "_mm_loadl_epi64",
        # Store
        "_mm_store_pd",
        "_mm_store_pd1",
        "_mm_store_sd",
        "_mm_store1_pd",
        "_mm_storeh_pd",
        "_mm_storel_pd",
        "_mm_storer_pd",
        "_mm_storeu_pd",
        "_mm_store_si128",
        "_mm_storeu_si128",
        "_mm_storeu_si16",
        "_mm_storeu_si32",
        "_mm_storeu_si64",
        "_mm_storel_epi64",
        "_mm_stream_pd",
        "_mm_stream_si128",
        "_mm_stream_si32",
        "_mm_stream_si64",
        "_mm_maskmoveu_si128",
        # Set
        "_mm_set_pd",
        "_mm_set_pd1",
        "_mm_set_sd",
        "_mm_set1_pd",
        "_mm_setr_pd",
        "_mm_setzero_pd",
        "_mm_set_epi8",
        "_mm_set_epi16",
        "_mm_set_epi32",
        "_mm_set_epi64",
        "_mm_set_epi64x",
        "_mm_set1_epi8",
        "_mm_set1_epi16",
        "_mm_set1_epi32",
        "_mm_set1_epi64",
        "_mm_set1_epi64x",
        "_mm_setr_epi8",
        "_mm_setr_epi16",
        "_mm_setr_epi32",
        "_mm_setr_epi64",
        "_mm_setzero_si128",
        # Pack/Unpack
        "_mm_packs_epi16",
        "_mm_packs_epi32",
        "_mm_packus_epi16",
        "_mm_unpackhi_epi8",
        "_mm_unpackhi_epi16",
        "_mm_unpackhi_epi32",
        "_mm_unpackhi_epi64",
        "_mm_unpacklo_epi8",
        "_mm_unpacklo_epi16",
        "_mm_unpacklo_epi32",
        "_mm_unpacklo_epi64",
        "_mm_unpackhi_pd",
        "_mm_unpacklo_pd",
        # Miscellaneous
        "_mm_move_sd",
        "_mm_movemask_pd",
        "_mm_movemask_epi8",
        "_mm_shuffle_pd",
        "_mm_shuffle_epi32",
        "_mm_shufflehi_epi16",
        "_mm_shufflelo_epi16",
        "_mm_extract_epi16",
        "_mm_insert_epi16",
        "_mm_clflush",
        "_mm_lfence",
        "_mm_mfence",
        "_mm_pause",
        # Cast (reinterpret)
        "_mm_castpd_ps",
        "_mm_castpd_si128",
        "_mm_castps_pd",
        "_mm_castps_si128",
        "_mm_castsi128_pd",
        "_mm_castsi128_ps",
        # Undefined values
        "_mm_undefined_pd",
        "_mm_undefined_si128",
        # 64-bit operations
        "_mm_add_si64",
        "_mm_movepi64_pi64",
        "_mm_movpi64_epi64",
    ],
    # =========================================================================
    # SSE3 (Streaming SIMD Extensions 3)
    # =========================================================================
    "SSE3": [
        # Horizontal operations
        "_mm_hadd_ps",
        "_mm_hadd_pd",
        "_mm_hsub_ps",
        "_mm_hsub_pd",
        # Add/subtract
        "_mm_addsub_ps",
        "_mm_addsub_pd",
        # Duplicate
        "_mm_movehdup_ps",
        "_mm_moveldup_ps",
        "_mm_movedup_pd",
        # Load
        "_mm_lddqu_si128",
        # Monitor/Wait (not typically emulated)
        "_mm_monitor",
        "_mm_mwait",
    ],
    # =========================================================================
    # SSSE3 (Supplemental SSE3)
    # =========================================================================
    "SSSE3": [
        # Absolute value
        "_mm_abs_epi8",
        "_mm_abs_epi16",
        "_mm_abs_epi32",
        "_mm_abs_pi8",
        "_mm_abs_pi16",
        "_mm_abs_pi32",
        # Horizontal add/subtract
        "_mm_hadd_epi16",
        "_mm_hadd_epi32",
        "_mm_hadds_epi16",
        "_mm_hsub_epi16",
        "_mm_hsub_epi32",
        "_mm_hsubs_epi16",
        "_mm_hadd_pi16",
        "_mm_hadd_pi32",
        "_mm_hadds_pi16",
        "_mm_hsub_pi16",
        "_mm_hsub_pi32",
        "_mm_hsubs_pi16",
        # Multiply and add
        "_mm_maddubs_epi16",
        "_mm_maddubs_pi16",
        # Packed multiply high with round and scale
        "_mm_mulhrs_epi16",
        "_mm_mulhrs_pi16",
        # Shuffle bytes
        "_mm_shuffle_epi8",
        "_mm_shuffle_pi8",
        # Sign
        "_mm_sign_epi8",
        "_mm_sign_epi16",
        "_mm_sign_epi32",
        "_mm_sign_pi8",
        "_mm_sign_pi16",
        "_mm_sign_pi32",
        # Align right
        "_mm_alignr_epi8",
        "_mm_alignr_pi8",
    ],
    # =========================================================================
    # SSE4.1
    # =========================================================================
    "SSE4.1": [
        # Blend
        "_mm_blend_pd",
        "_mm_blend_ps",
        "_mm_blend_epi16",
        "_mm_blendv_pd",
        "_mm_blendv_ps",
        "_mm_blendv_epi8",
        # Dot product
        "_mm_dp_ps",
        "_mm_dp_pd",
        # Extract/Insert
        "_mm_extract_epi8",
        "_mm_extract_epi32",
        "_mm_extract_epi64",
        "_mm_extract_ps",
        "_mm_insert_epi8",
        "_mm_insert_epi32",
        "_mm_insert_epi64",
        "_mm_insert_ps",
        # Floor/Ceiling
        "_mm_ceil_pd",
        "_mm_ceil_ps",
        "_mm_ceil_sd",
        "_mm_ceil_ss",
        "_mm_floor_pd",
        "_mm_floor_ps",
        "_mm_floor_sd",
        "_mm_floor_ss",
        "_mm_round_pd",
        "_mm_round_ps",
        "_mm_round_sd",
        "_mm_round_ss",
        # Min/Max
        "_mm_max_epi8",
        "_mm_max_epi32",
        "_mm_max_epu16",
        "_mm_max_epu32",
        "_mm_min_epi8",
        "_mm_min_epi32",
        "_mm_min_epu16",
        "_mm_min_epu32",
        # Multiply
        "_mm_mul_epi32",
        "_mm_mullo_epi32",
        # Pack
        "_mm_packus_epi32",
        # Compare
        "_mm_cmpeq_epi64",
        # Convert with sign/zero extension
        "_mm_cvtepi8_epi16",
        "_mm_cvtepi8_epi32",
        "_mm_cvtepi8_epi64",
        "_mm_cvtepi16_epi32",
        "_mm_cvtepi16_epi64",
        "_mm_cvtepi32_epi64",
        "_mm_cvtepu8_epi16",
        "_mm_cvtepu8_epi32",
        "_mm_cvtepu8_epi64",
        "_mm_cvtepu16_epi32",
        "_mm_cvtepu16_epi64",
        "_mm_cvtepu32_epi64",
        # Test
        "_mm_test_all_ones",
        "_mm_test_all_zeros",
        "_mm_test_mix_ones_zeros",
        "_mm_testc_si128",
        "_mm_testnzc_si128",
        "_mm_testz_si128",
        # Minimum position
        "_mm_minpos_epu16",
        # Multiple packed sum of absolute difference
        "_mm_mpsadbw_epu8",
        # Stream load
        "_mm_stream_load_si128",
    ],
    # =========================================================================
    # SSE4.2
    # =========================================================================
    "SSE4.2": [
        # String comparison (PCMPISTRI, PCMPISTRM, PCMPESTRI, PCMPESTRM)
        "_mm_cmpistrm",
        "_mm_cmpistri",
        "_mm_cmpistra",
        "_mm_cmpistrc",
        "_mm_cmpistro",
        "_mm_cmpistrs",
        "_mm_cmpistrz",
        "_mm_cmpestrm",
        "_mm_cmpestri",
        "_mm_cmpestra",
        "_mm_cmpestrc",
        "_mm_cmpestro",
        "_mm_cmpestrs",
        "_mm_cmpestrz",
        # Compare greater than
        "_mm_cmpgt_epi64",
        # CRC32
        "_mm_crc32_u8",
        "_mm_crc32_u16",
        "_mm_crc32_u32",
        "_mm_crc32_u64",
        # POPCNT (often grouped with SSE4.2)
        "_mm_popcnt_u32",
        "_mm_popcnt_u64",
    ],
    # =========================================================================
    # AES-NI (Advanced Encryption Standard New Instructions)
    # =========================================================================
    "AES": [
        "_mm_aesdec_si128",
        "_mm_aesdeclast_si128",
        "_mm_aesenc_si128",
        "_mm_aesenclast_si128",
        "_mm_aesimc_si128",
        "_mm_aeskeygenassist_si128",
    ],
    # =========================================================================
    # PCLMULQDQ (Carry-Less Multiplication)
    # =========================================================================
    "PCLMULQDQ": [
        "_mm_clmulepi64_si128",
    ],
}

# Aliases and alternative names that should be counted as equivalent
# Intel uses both lowercase function names and uppercase macro names
INTRINSIC_ALIASES = {
    "_mm_set_pd1": "_mm_set1_pd",
    "_mm_set_ps1": "_mm_set1_ps",
    "_mm_load_pd1": "_mm_load1_pd",
    "_mm_load_ps1": "_mm_load1_ps",
    "_mm_store_pd1": "_mm_store1_pd",
    "_mm_store_ps1": "_mm_store1_ps",
    "_mm_cvtsi64x_sd": "_mm_cvtsi64_sd",
    "_mm_cvtsi64x_si128": "_mm_cvtsi64_si128",
    "_mm_cvtsi128_si64x": "_mm_cvtsi128_si64",
    "_mm_cvttsd_si64x": "_mm_cvttsd_si64",
    "_m_empty": "_mm_empty",
    # Uppercase macro names -> lowercase equivalents
    "_mm_get_rounding_mode": "_mm_get_rounding_mode",  # implemented as _MM_GET_ROUNDING_MODE
    "_mm_set_rounding_mode": "_mm_set_rounding_mode",  # implemented as _MM_SET_ROUNDING_MODE
}

# Intrinsics intentionally not implemented (with reason)
NOT_IMPLEMENTED_REASONS = {
    # x86 power management
    "_mm_monitor": "x86 power management, no ARM equivalent",
    "_mm_mwait": "x86 power management, no ARM equivalent",
    # Non-temporal hints
    "_mm_stream_load_si128": "Non-temporal load hint, maps to regular load on ARM",
}


def extract_sse2neon_intrinsics(header_path: str) -> set:
    """
    Parse sse2neon.h and extract all implemented intrinsic names.

    Returns a set of intrinsic function names.
    """
    intrinsics = set()

    # Pattern matches FORCE_INLINE followed by return type and function name
    # Examples:
    #   FORCE_INLINE __m128 _mm_add_ps(__m128 a, __m128 b)
    #   FORCE_INLINE int _mm_comieq_ss(__m128 a, __m128 b)
    #   FORCE_INLINE void *_mm_malloc(size_t size, size_t align)
    #   FORCE_INLINE unsigned int _MM_GET_ROUNDING_MODE(void)
    pattern = re.compile(
        r"FORCE_INLINE\s+"  # FORCE_INLINE keyword
        r".+[\s\*]"  # return type (greedy match, ending in space or *)
        r"(_[mM](?:[mM][0-9]*)?_[a-zA-Z0-9_]+)"  # intrinsic name: _mm_*, _MM_*, _m_* etc
        r"\s*\("  # opening paren
    )

    # Match macro definitions with arguments: #define _mm_xxx(...)
    macro_pattern = re.compile(
        r"^#define\s+(_m(?:m[0-9]*)?_[a-z0-9_]+)\s*[\(\\]", re.MULTILINE
    )

    # Match simple alias definitions: #define _mm_xxx _mm_yyy
    # Allow trailing whitespace and comments
    alias_pattern = re.compile(
        r"^#define\s+(_m(?:m[0-9]*)?_[a-z0-9_]+)\s+(_m(?:m[0-9]*)?_[a-z0-9_]+)"
        r"(?:\s*(?://.*|/\*.*\*/)?)?\s*$",
        re.MULTILINE,
    )

    # Match uppercase macro accessors: #define _MM_GET_xxx _sse2neon_mm_get_xxx
    uppercase_pattern = re.compile(
        r"^#define\s+(_MM_[A-Z0-9_]+)\s+_sse2neon_mm_[a-z0-9_]+", re.MULTILINE
    )

    try:
        with open(header_path, "r") as f:
            content = f.read()
    except IOError as e:
        print(f"Error: Could not read {header_path}: {e}", file=sys.stderr)
        return set()

    # Find all FORCE_INLINE functions
    for match in pattern.finditer(content):
        name = match.group(1)
        # Skip internal helpers and shuffle specializations
        if name.startswith("_sse2neon_"):
            continue
        if re.match(r"_mm_shuffle_ps_\d+|_mm_shuffle_epi_\d+", name):
            continue
        # Convert uppercase _MM_* to lowercase for comparison
        if name.startswith("_MM_"):
            intrinsics.add(name.lower())
        else:
            intrinsics.add(name)

    # Find macro definitions with arguments
    for match in macro_pattern.finditer(content):
        name = match.group(1)
        if name.startswith("_sse2neon_"):
            continue
        # Add all _mm_* and _m_* macro intrinsics
        if name.startswith("_mm_") or name.startswith("_m_"):
            intrinsics.add(name)

    # Find simple alias definitions
    for match in alias_pattern.finditer(content):
        alias_name = match.group(1)
        if alias_name.startswith("_mm_") or alias_name.startswith("_m_"):
            intrinsics.add(alias_name)

    # Find uppercase macro accessors (_MM_GET_FLUSH_ZERO_MODE -> _mm_get_flush_zero_mode)
    for match in uppercase_pattern.finditer(content):
        macro_name = match.group(1)
        # Convert to lowercase intrinsic name
        intrinsic_name = macro_name.lower()
        intrinsics.add(intrinsic_name)

    return intrinsics


def normalize_intrinsic(name: str) -> str:
    """Normalize intrinsic name using alias mapping."""
    return INTRINSIC_ALIASES.get(name, name)


def analyze_coverage(implemented: set):
    """
    Compare implemented intrinsics against Intel reference.

    Returns coverage statistics by instruction set.
    """
    # Normalize implemented set
    implemented_normalized = {normalize_intrinsic(i) for i in implemented}

    results = {}
    all_missing = []

    for isa, reference in INTEL_INTRINSICS.items():
        # Normalize reference
        reference_normalized = {normalize_intrinsic(i) for i in reference}

        # Find missing and implemented
        missing = reference_normalized - implemented_normalized
        found = reference_normalized & implemented_normalized

        # Calculate coverage
        total = len(reference_normalized)
        implemented_count = len(found)
        coverage = (implemented_count / total * 100) if total > 0 else 0.0

        results[isa] = {
            "total": total,
            "implemented": implemented_count,
            "missing": len(missing),
            "coverage": coverage,
            "missing_list": sorted(missing),
        }

        all_missing.extend([(isa, m) for m in missing])

    # Find extra intrinsics (implemented but not in reference)
    all_reference = set()
    for ref in INTEL_INTRINSICS.values():
        all_reference.update(normalize_intrinsic(i) for i in ref)

    extra = implemented_normalized - all_reference
    # Filter out internal helpers and well-known non-Intel intrinsics
    extra = {
        e
        for e in extra
        if not e.startswith("_sse2neon_")
        and not e.startswith("_MM_")
        and not re.match(r"_mm_shuffle_ps_\d+|_mm_shuffle_epi_\d+", e)
    }

    return results, all_missing, extra


def print_report(results: dict, missing: list, extra: set, verbose: bool = False):
    """Print coverage report to stdout."""
    print("=" * 70)
    print("sse2neon Intrinsic Coverage Report")
    print("=" * 70)
    print()

    # Summary table
    print(f"{'Instruction Set':<15} {'Implemented':>12} {'Total':>8} {'Coverage':>10}")
    print("-" * 50)

    total_impl = 0
    total_ref = 0

    for isa in [
        "SSE",
        "SSE2",
        "SSE3",
        "SSSE3",
        "SSE4.1",
        "SSE4.2",
        "AES",
        "PCLMULQDQ",
    ]:
        if isa not in results:
            continue
        r = results[isa]
        total_impl += r["implemented"]
        total_ref += r["total"]
        status = "" if r["coverage"] == 100.0 else f" (-{r['missing']})"
        print(
            f"{isa:<15} {r['implemented']:>12} {r['total']:>8} {r['coverage']:>9.1f}%{status}"
        )

    print("-" * 50)
    overall = (total_impl / total_ref * 100) if total_ref > 0 else 0.0
    print(f"{'TOTAL':<15} {total_impl:>12} {total_ref:>8} {overall:>9.1f}%")
    print()

    # Missing intrinsics
    if missing:
        print("Missing Intrinsics by Instruction Set:")
        print("-" * 50)

        by_isa = {}
        for isa, name in missing:
            by_isa.setdefault(isa, []).append(name)

        for isa in [
            "SSE",
            "SSE2",
            "SSE3",
            "SSSE3",
            "SSE4.1",
            "SSE4.2",
            "AES",
            "PCLMULQDQ",
        ]:
            if isa in by_isa:
                names = by_isa[isa]
                print(f"\n{isa} ({len(names)} missing):")
                for n in sorted(names):
                    note = ""
                    if n in NOT_IMPLEMENTED_REASONS:
                        note = f"  # {NOT_IMPLEMENTED_REASONS[n]}"
                    print(f"  {n}{note}")
        print()

    # Extra intrinsics (not in Intel reference but implemented)
    if extra and verbose:
        print("Extra Intrinsics (not in Intel reference, may be aliases/extensions):")
        print("-" * 50)
        for name in sorted(extra):
            print(f"  {name}")
        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Verify sse2neon intrinsic coverage against Intel reference"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show extra intrinsics not in Intel reference",
    )
    parser.add_argument(
        "--header", default=None, help="Path to sse2neon.h (default: auto-detect)"
    )
    parser.add_argument(
        "--badge",
        action="store_true",
        help="Output only coverage percentage (for CI badges)",
    )
    args = parser.parse_args()

    # Find sse2neon.h
    if args.header:
        header_path = args.header
    else:
        # Try common locations
        script_dir = Path(__file__).parent
        candidates = [
            script_dir.parent / "sse2neon.h",
            Path("sse2neon.h"),
        ]
        header_path = None
        for c in candidates:
            if c.exists():
                header_path = str(c)
                break

        if not header_path:
            print("Error: Could not find sse2neon.h", file=sys.stderr)
            print("Use --header to specify path", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(header_path):
        print(f"Error: File not found: {header_path}", file=sys.stderr)
        sys.exit(1)

    # Extract implemented intrinsics
    implemented = extract_sse2neon_intrinsics(header_path)

    # Analyze coverage
    results, missing, extra = analyze_coverage(implemented)

    # Badge output (just the percentage)
    if args.badge:
        total_impl = sum(r["implemented"] for r in results.values())
        total_ref = sum(r["total"] for r in results.values())
        overall = (total_impl / total_ref * 100) if total_ref > 0 else 0.0
        print(f"{overall:.1f}")
        return 0

    # Output
    print_report(results, missing, extra, args.verbose)

    # Exit with error if coverage < 100% for unexpected gaps
    total_missing = sum(r["missing"] for r in results.values())
    if total_missing > 0:
        # Only return error for truly missing intrinsics (not intentionally skipped)
        actionable_missing = [
            (isa, name) for isa, name in missing if name not in NOT_IMPLEMENTED_REASONS
        ]
        if actionable_missing:
            # Print actionable items for CI visibility
            print("\nUnexpected gaps requiring attention:", file=sys.stderr)
            for isa, name in actionable_missing:
                print(f"  {isa}: {name}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
