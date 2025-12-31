# SSE2NEON Performance Tier Analysis (AArch64)

This document classifies SSE intrinsics by their NEON implementation complexity
on AArch64 (ARMv8-A 64-bit). Analysis uses `unifdef` to preprocess conditional
compilation paths, providing accurate instruction counts for the AArch64 target.

> Note: ARMv7 (32-bit ARM) implementations may differ significantly due to
> missing instructions like `vrndnq_f32` (directed rounding) and require fallback
> paths. Use `--no-unifdef` flag to analyze the raw file including all code paths.

To regenerate this file:
```bash
python3 scripts/gen-perf-report.py > perf-tier.md
```

For detailed analysis:
```bash
python3 scripts/analyze-tiers.py              # AArch64 analysis (default)
python3 scripts/analyze-tiers.py --no-unifdef # Raw file (all architectures)
python3 scripts/analyze-tiers.py --json       # JSON output
python3 scripts/analyze-tiers.py --tier 4     # Filter by tier
python3 scripts/analyze-tiers.py --weighted   # Instruction-weighted counts
```

---

## SSE2NEON Performance Classification (AArch64)

Based on static analysis of NEON instruction counts per SSE intrinsic.
Cycle estimates based on ARM Cortex-A72 (ARMv8-A) Software Optimization Guide.

### Summary

| Metric | Value |
|--------|-------|
| Total SSE Intrinsics | 468 |
| Direct Mappings (T1) | 367 (78.4%) |
| Moderate Emulation (T2-T3) | 87 (18.6%) |
| Complex Emulation (T4) | 14 (3.0%) |
| Avg NEON Ops/Intrinsic | 1.89 |

### Performance Tiers

| Tier | NEON Ops | Estimated Cycles | Description |
|------|----------|------------------|-------------|
| T1 | 1-2 | 1-3 | Direct NEON mapping, near-native performance |
| T2 | 3-5 | 4-8 | Few NEON operations, slight overhead |
| T3 | 6-10 | 8-15 | Moderate emulation, noticeable overhead |
| T4 | 10+ or special | 15-50+ | Complex/algorithmic, significant overhead |

> **Note**: T4 classification uses multi-factor analysis beyond raw NEON counts:
> - Table lookups (e.g., `_mm_shuffle_epi8`) are T4 due to algorithmic complexity
> - Pure scalar fallbacks (e.g., `_mm_crc32_u16` without HW CRC) are T4
> - Loop-based implementations are T4 regardless of instruction count
> - Mixed scalar+NEON with high effective cost may be promoted to T4

### Performance-Critical Intrinsics (T4)

These intrinsics have significant emulation overhead. Consider alternative
algorithms when porting performance-critical code.

| Intrinsic | NEON Ops | Notes |
|-----------|----------|-------|
| `_mm_mpsadbw_epu8` | 22 | SAD computation, very expensive |
| `_mm_sqrt_ps` | 15 | Newton-Raphson refinement |
| `_mm_cvttpd_pi32` | 14 |  |
| `_mm_rsqrt_ps` | 13 | Newton-Raphson refinement |
| `_mm_aesdec_si128` | 12 | Use HW crypto when available |
| `_mm_dp_ps` | 9 |  |
| `_mm_minpos_epu16` | 9 | Horizontal minimum search |
| `_mm_aesenc_si128` | 9 | Use HW crypto when available |
| `_mm_aesimc_si128` | 9 | Use HW crypto when available |
| `_mm_dp_pd` | 4 |  |
| `_mm_shuffle_epi8` | 3 |  |
| `_mm_shuffle_pi8` | 3 |  |
| `_mm_aesenclast_si128` | 3 | Use HW crypto when available |
| `_mm_aesdeclast_si128` | 3 | Use HW crypto when available |

### Efficient Intrinsics (T1 - Single NEON Instruction)

These intrinsics map directly to single NEON instructions:

- Arithmetic: `_mm_abs_epi16`, `_mm_abs_epi32`, `_mm_abs_epi8`, `_mm_abs_pi16`, `_mm_abs_pi32`, `_mm_abs_pi8`, `_mm_add_epi16`, `_mm_add_epi32`, ... (+29 more)
- Comparison: `_mm_cmpeq_epi16`, `_mm_cmpeq_epi32`, `_mm_cmpeq_epi64`, `_mm_cmpeq_epi8`, `_mm_cmpeq_pd`, `_mm_cmpeq_ps`, `_mm_cmpgt_epi16`, `_mm_cmpgt_epi32`, ... (+9 more)
- Logical: `_mm_and_pd`, `_mm_and_ps`, `_mm_and_si128`, `_mm_andnot_pd`, `_mm_andnot_ps`, `_mm_andnot_si128`, `_mm_floor_pd`, `_mm_floor_ps`, ... (+18 more)
- Load/Store: `_mm_load1_pd`, `_mm_load1_ps`, `_mm_load_pd`, `_mm_load_ps`, `_mm_load_si128`, `_mm_loadu_ps`, `_mm_loadu_si128`, `_mm_set1_epi16`, ... (+24 more)
- Conversion: `_mm_cvt_ps2pi`, `_mm_cvt_si2ss`, `_mm_cvt_ss2si`, `_mm_cvtepi32_ps`, `_mm_cvtps_epi32`, `_mm_cvtps_pi16`, `_mm_cvtsd_f64`, `_mm_cvtsd_si32`, ... (+13 more)
- Math: `_mm_ceil_pd`, `_mm_ceil_ps`, `_mm_floor_pd`, `_mm_floor_ps`, `_mm_max_epi16`, `_mm_max_epi32`, `_mm_max_epi8`, `_mm_max_epu16`, ... (+13 more)

### Complete Tier Classification

<details>
<summary>Click to expand full list</summary>

#### Tier 1 (367 intrinsics)

`_mm_abs_epi16`, `_mm_abs_epi32`, `_mm_abs_epi8`, `_mm_abs_pi16`
`_mm_abs_pi32`, `_mm_abs_pi8`, `_mm_add_epi16`, `_mm_add_epi32`
`_mm_add_epi64`, `_mm_add_epi8`, `_mm_add_pd`, `_mm_add_ps`, `_mm_add_sd`
`_mm_add_si64`, `_mm_adds_epi16`, `_mm_adds_epi8`, `_mm_adds_epu16`
`_mm_adds_epu8`, `_mm_addsub_pd`, `_mm_addsub_ps`, `_mm_and_pd`, `_mm_and_ps`
`_mm_and_si128`, `_mm_andnot_pd`, `_mm_andnot_ps`, `_mm_andnot_si128`
`_mm_avg_epu16`, `_mm_avg_epu8`, `_mm_avg_pu16`, `_mm_avg_pu8`
`_mm_blendv_epi8`, `_mm_blendv_pd`, `_mm_blendv_ps`, `_mm_castpd_ps`
`_mm_castpd_si128`, `_mm_castps_pd`, `_mm_castps_si128`, `_mm_castsi128_pd`
`_mm_castsi128_ps`, `_mm_ceil_pd`, `_mm_ceil_ps`, `_mm_ceil_sd`
`_mm_ceil_ss`, `_mm_clflush`, `_mm_clmulepi64_si128`, `_mm_cmpeq_epi16`
`_mm_cmpeq_epi32`, `_mm_cmpeq_epi64`, `_mm_cmpeq_epi8`, `_mm_cmpeq_pd`
`_mm_cmpeq_ps`, `_mm_cmpeq_sd`, `_mm_cmpeq_ss`, `_mm_cmpestra`
`_mm_cmpestrc`, `_mm_cmpestri`, `_mm_cmpestrm`, `_mm_cmpestro`
`_mm_cmpestrs`, `_mm_cmpestrz`, `_mm_cmpge_pd`, `_mm_cmpge_ps`
`_mm_cmpge_sd`, `_mm_cmpge_ss`, `_mm_cmpgt_epi16`, `_mm_cmpgt_epi32`
`_mm_cmpgt_epi64`, `_mm_cmpgt_epi8`, `_mm_cmpgt_pd`, `_mm_cmpgt_ps`
`_mm_cmpgt_sd`, `_mm_cmpgt_ss`, `_mm_cmpistra`, `_mm_cmpistrc`
`_mm_cmpistri`, `_mm_cmpistrm`, `_mm_cmpistro`, `_mm_cmpistrs`
`_mm_cmpistrz`, `_mm_cmple_pd`, `_mm_cmple_ps`, `_mm_cmple_sd`
`_mm_cmple_ss`, `_mm_cmplt_epi16`, `_mm_cmplt_epi32`, `_mm_cmplt_epi8`
`_mm_cmplt_pd`, `_mm_cmplt_ps`, `_mm_cmplt_sd`, `_mm_cmplt_ss`
`_mm_cmpneq_pd`, `_mm_cmpneq_ps`, `_mm_cmpneq_sd`, `_mm_cmpneq_ss`
`_mm_cmpnge_ps`, `_mm_cmpnge_sd`, `_mm_cmpnge_ss`, `_mm_cmpngt_ps`
`_mm_cmpngt_sd`, `_mm_cmpngt_ss`, `_mm_cmpnle_ps`, `_mm_cmpnle_sd`
`_mm_cmpnle_ss`, `_mm_cmpnlt_ps`, `_mm_cmpnlt_sd`, `_mm_cmpnlt_ss`
`_mm_cmpord_sd`, `_mm_cmpord_ss`, `_mm_cmpunord_sd`, `_mm_cmpunord_ss`
`_mm_comieq_sd`, `_mm_comieq_ss`, `_mm_comige_sd`, `_mm_comige_ss`
`_mm_comigt_sd`, `_mm_comigt_ss`, `_mm_comile_sd`, `_mm_comile_ss`
`_mm_comilt_sd`, `_mm_comilt_ss`, `_mm_comineq_sd`, `_mm_comineq_ss`
`_mm_cvt_pi2ps`, `_mm_cvt_ps2pi`, `_mm_cvt_si2ss`, `_mm_cvt_ss2si`
`_mm_cvtepi16_epi32`, `_mm_cvtepi16_epi64`, `_mm_cvtepi32_epi64`
`_mm_cvtepi32_pd`, `_mm_cvtepi32_ps`, `_mm_cvtepi8_epi16`
`_mm_cvtepi8_epi32`, `_mm_cvtepu16_epi32`, `_mm_cvtepu16_epi64`
`_mm_cvtepu32_epi64`, `_mm_cvtepu8_epi16`, `_mm_cvtepu8_epi32`
`_mm_cvtpd_epi32`, `_mm_cvtpd_ps`, `_mm_cvtpi16_ps`, `_mm_cvtpi32_pd`
`_mm_cvtpi32_ps`, `_mm_cvtpi32x2_ps`, `_mm_cvtps_epi32`, `_mm_cvtps_pd`
`_mm_cvtps_pi16`, `_mm_cvtps_pi8`, `_mm_cvtpu16_ps`, `_mm_cvtsd_f64`
`_mm_cvtsd_si32`, `_mm_cvtsd_si64`, `_mm_cvtsi128_si32`, `_mm_cvtsi128_si64`
`_mm_cvtsi32_sd`, `_mm_cvtsi32_si128`, `_mm_cvtsi64_sd`, `_mm_cvtsi64_si128`
`_mm_cvtsi64_ss`, `_mm_cvtss_f32`, `_mm_cvtss_sd`, `_mm_cvtss_si64`
`_mm_cvtt_ps2pi`, `_mm_cvtt_ss2si`, `_mm_cvttpd_epi32`, `_mm_cvttps_epi32`
`_mm_cvttsd_si32`, `_mm_cvttsd_si64`, `_mm_cvttss_si64`, `_mm_div_pd`
`_mm_div_ps`, `_mm_div_ss`, `_mm_empty`, `_mm_floor_pd`, `_mm_floor_ps`
`_mm_floor_sd`, `_mm_floor_ss`, `_mm_free`, `_mm_lfence`, `_mm_load1_pd`
`_mm_load1_ps`, `_mm_load_pd`, `_mm_load_ps`, `_mm_load_sd`, `_mm_load_si128`
`_mm_load_ss`, `_mm_loadh_pd`, `_mm_loadh_pi`, `_mm_loadl_epi64`
`_mm_loadl_pd`, `_mm_loadl_pi`, `_mm_loadr_pd`, `_mm_loadu_pd`
`_mm_loadu_ps`, `_mm_loadu_si128`, `_mm_loadu_si16`, `_mm_loadu_si32`
`_mm_loadu_si64`, `_mm_max_epi16`, `_mm_max_epi32`, `_mm_max_epi8`
`_mm_max_epu16`, `_mm_max_epu32`, `_mm_max_epu8`, `_mm_max_pi16`
`_mm_max_pu8`, `_mm_max_sd`, `_mm_max_ss`, `_mm_mfence`, `_mm_min_epi16`
`_mm_min_epi32`, `_mm_min_epi8`, `_mm_min_epu16`, `_mm_min_epu32`
`_mm_min_epu8`, `_mm_min_pi16`, `_mm_min_pu8`, `_mm_min_sd`, `_mm_min_ss`
`_mm_monitor`, `_mm_move_epi64`, `_mm_move_sd`, `_mm_move_ss`
`_mm_movedup_pd`, `_mm_movehdup_ps`, `_mm_movehl_ps`, `_mm_moveldup_ps`
`_mm_movelh_ps`, `_mm_movepi64_pi64`, `_mm_movpi64_epi64`, `_mm_mul_epi32`
`_mm_mul_epu32`, `_mm_mul_pd`, `_mm_mul_ps`, `_mm_mul_sd`, `_mm_mul_ss`
`_mm_mul_su32`, `_mm_mulhi_epu16`, `_mm_mulhi_pu16`, `_mm_mulhrs_epi16`
`_mm_mulhrs_pi16`, `_mm_mullo_epi16`, `_mm_mullo_epi32`, `_mm_mwait`
`_mm_or_pd`, `_mm_or_ps`, `_mm_or_si128`, `_mm_packs_epi16`
`_mm_packs_epi32`, `_mm_packus_epi16`, `_mm_packus_epi32`, `_mm_pause`
`_mm_popcnt_u32`, `_mm_popcnt_u64`, `_mm_prefetch`, `_mm_rcp_ss`
`_mm_round_pd`, `_mm_round_ps`, `_mm_round_sd`, `_mm_round_ss`
`_mm_rsqrt_ss`, `_mm_sad_epu8`, `_mm_set1_epi16`, `_mm_set1_epi32`
`_mm_set1_epi64`, `_mm_set1_epi64x`, `_mm_set1_epi8`, `_mm_set1_pd`
`_mm_set1_ps`, `_mm_set_epi64`, `_mm_set_epi64x`, `_mm_set_ps1`, `_mm_set_sd`
`_mm_set_ss`, `_mm_setcsr`, `_mm_setr_epi64`, `_mm_setr_pd`, `_mm_setzero_pd`
`_mm_setzero_ps`, `_mm_setzero_si128`, `_mm_sfence`, `_mm_shuffle_epi_0101`
`_mm_shuffle_epi_0122`, `_mm_shuffle_epi_0321`, `_mm_shuffle_epi_1001`
`_mm_shuffle_epi_1010`, `_mm_shuffle_epi_1032`, `_mm_shuffle_epi_2103`
`_mm_shuffle_epi_2211`, `_mm_shuffle_epi_2301`, `_mm_shuffle_epi_3332`
`_mm_shuffle_ps_0011`, `_mm_shuffle_ps_0022`, `_mm_shuffle_ps_0101`
`_mm_shuffle_ps_0321`, `_mm_shuffle_ps_1001`, `_mm_shuffle_ps_1010`
`_mm_shuffle_ps_1032`, `_mm_shuffle_ps_1133`, `_mm_shuffle_ps_2103`
`_mm_shuffle_ps_2200`, `_mm_shuffle_ps_2301`, `_mm_shuffle_ps_3202`
`_mm_shuffle_ps_3210`, `_mm_slli_epi16`, `_mm_slli_epi32`, `_mm_slli_epi64`
`_mm_sqrt_pd`, `_mm_sqrt_sd`, `_mm_sqrt_ss`, `_mm_store_pd`, `_mm_store_pd1`
`_mm_store_ps`, `_mm_store_sd`, `_mm_store_si128`, `_mm_store_ss`
`_mm_storeh_pd`, `_mm_storeh_pi`, `_mm_storel_epi64`, `_mm_storel_pd`
`_mm_storel_pi`, `_mm_storer_pd`, `_mm_storeu_pd`, `_mm_storeu_ps`
`_mm_storeu_si128`, `_mm_storeu_si16`, `_mm_storeu_si32`, `_mm_storeu_si64`
`_mm_stream_load_si128`, `_mm_stream_pd`, `_mm_stream_pi`, `_mm_stream_ps`
`_mm_stream_si128`, `_mm_stream_si32`, `_mm_stream_si64`, `_mm_sub_epi16`
`_mm_sub_epi32`, `_mm_sub_epi64`, `_mm_sub_epi8`, `_mm_sub_pd`, `_mm_sub_ps`
`_mm_sub_sd`, `_mm_sub_si64`, `_mm_sub_ss`, `_mm_subs_epi16`, `_mm_subs_epi8`
`_mm_subs_epu16`, `_mm_subs_epu8`, `_mm_test_all_ones`, `_mm_undefined_pd`
`_mm_undefined_ps`, `_mm_undefined_si128`, `_mm_unpackhi_epi16`
`_mm_unpackhi_epi32`, `_mm_unpackhi_epi64`, `_mm_unpackhi_epi8`
`_mm_unpackhi_pd`, `_mm_unpackhi_ps`, `_mm_unpacklo_epi16`
`_mm_unpacklo_epi32`, `_mm_unpacklo_epi64`, `_mm_unpacklo_epi8`
`_mm_unpacklo_pd`, `_mm_unpacklo_ps`, `_mm_xor_pd`, `_mm_xor_ps`
`_mm_xor_si128`

#### Tier 2 (73 intrinsics)

`_mm_add_ss`, `_mm_cmpnge_pd`, `_mm_cmpngt_pd`, `_mm_cmpnle_pd`
`_mm_cmpnlt_pd`, `_mm_cmpord_pd`, `_mm_cmpord_ps`, `_mm_cmpunord_pd`
`_mm_cmpunord_ps`, `_mm_cvtepi8_epi64`, `_mm_cvtepu8_epi64`, `_mm_cvtpi8_ps`
`_mm_cvtpu8_ps`, `_mm_cvtsd_ss`, `_mm_div_sd`, `_mm_hadd_epi16`
`_mm_hadd_epi32`, `_mm_hadd_pd`, `_mm_hadd_pi16`, `_mm_hadd_pi32`
`_mm_hadd_ps`, `_mm_hadds_epi16`, `_mm_hadds_pi16`, `_mm_hsub_epi16`
`_mm_hsub_epi32`, `_mm_hsub_pd`, `_mm_hsub_pi16`, `_mm_hsub_pi32`
`_mm_hsub_ps`, `_mm_hsubs_epi16`, `_mm_hsubs_pi16`, `_mm_loadr_ps`
`_mm_madd_epi16`, `_mm_maskmove_si64`, `_mm_maskmoveu_si128`, `_mm_max_pd`
`_mm_max_ps`, `_mm_min_pd`, `_mm_min_ps`, `_mm_movemask_pd`
`_mm_mulhi_epi16`, `_mm_set_epi16`, `_mm_set_epi32`, `_mm_set_epi8`
`_mm_set_pd`, `_mm_set_ps`, `_mm_setr_epi16`, `_mm_setr_epi32`
`_mm_setr_epi8`, `_mm_setr_ps`, `_mm_shuffle_ps_2001`, `_mm_shuffle_ps_2010`
`_mm_shuffle_ps_2032`, `_mm_sign_epi16`, `_mm_sign_epi32`, `_mm_sign_epi8`
`_mm_sign_pi16`, `_mm_sign_pi32`, `_mm_sign_pi8`, `_mm_sll_epi16`
`_mm_sll_epi32`, `_mm_sll_epi64`, `_mm_sra_epi16`, `_mm_sra_epi32`
`_mm_srai_epi16`, `_mm_srl_epi16`, `_mm_srl_epi32`, `_mm_srl_epi64`
`_mm_store_ps1`, `_mm_storer_ps`, `_mm_test_all_zeros`, `_mm_testc_si128`
`_mm_testz_si128`

#### Tier 3 (14 intrinsics)

`_mm_aeskeygenassist_si128`, `_mm_crc32_u16`, `_mm_crc32_u32`
`_mm_crc32_u64`, `_mm_crc32_u8`, `_mm_cvtpd_pi32`, `_mm_maddubs_epi16`
`_mm_maddubs_pi16`, `_mm_movemask_epi8`, `_mm_movemask_pi8`
`_mm_movemask_ps`, `_mm_rcp_ps`, `_mm_sad_pu8`, `_mm_test_mix_ones_zeros`

#### Tier 4 (14 intrinsics)

`_mm_aesdec_si128`, `_mm_aesdeclast_si128`, `_mm_aesenc_si128`
`_mm_aesenclast_si128`, `_mm_aesimc_si128`, `_mm_cvttpd_pi32`, `_mm_dp_pd`
`_mm_dp_ps`, `_mm_minpos_epu16`, `_mm_mpsadbw_epu8`, `_mm_rsqrt_ps`
`_mm_shuffle_epi8`, `_mm_shuffle_pi8`, `_mm_sqrt_ps`

</details>
