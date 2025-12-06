#!/usr/bin/env python3
"""
SSE2NEON Intrinsic Tier Analysis Tool

Analyzes sse2neon.h to classify intrinsics by their NEON instruction complexity
and estimate relative performance tiers.

This tool uses unifdef to preprocess the header for AArch64-specific analysis,
removing ARMv7 fallback paths that would skew instruction counts.

Performance Tiers (AArch64/ARMv8-A Reference):
  T1 (1:1-2):  Direct NEON mapping, 1-2 instructions, ~1-2 cycles
  T2 (1:3-5):  Few NEON ops, 3-5 instructions, ~3-8 cycles
  T3 (1:6-10): Moderate emulation, 6-10 instructions, ~10-20 cycles
  T4 (1:10+):  Complex emulation, 10+ instructions or algorithmic, ~20+ cycles

Usage:
  python3 scripts/analyze-tiers.py [--json] [--markdown] [--verbose]
  python3 scripts/analyze-tiers.py --no-unifdef  # Analyze raw file without preprocessing

Multi-Level Analysis:
  Level 1 (default): Fast regex-based pattern matching - used for initial classification
  Level 2 (--clang-ast): Clang AST analysis for T2/T3/T4 - more precise NEON counting
  Level 3 (--asm): Assembly generation analysis - most accurate, requires ARM64 cross-compiler

Examples:
  python3 scripts/analyze-tiers.py                    # Level 1 only (fast)
  python3 scripts/analyze-tiers.py --clang-ast        # Level 1 + Level 2 for T2/T3/T4
  python3 scripts/analyze-tiers.py --clang-ast --asm  # All levels (most accurate)
  python3 scripts/analyze-tiers.py --clang-ast -j 4   # Parallel AST analysis (4 workers)

Performance Options:
  -j, --jobs N     Parallel workers for AST/assembly analysis (default: 1)
  --clear-cache    Clear the persistent cache (~/.cache/sse2neon-tiers/)
  --weighted       Use weighted instruction counts for tier classification

Validation Options:
  --validate       Validate results against reference data and report accuracy
                   Shows tier classification accuracy and NEON count error rates

The persistent cache stores analysis results based on function body hash.
Cache invalidation is automatic when function bodies change.

Requirements for Level 3 (--asm):
  - ARM64 cross-compiler: aarch64-linux-gnu-gcc or native ARM64 gcc/clang
  - Or clang with AArch64 target support
"""

import re
import sys
import json
import os
import shutil
import subprocess
import tempfile
import hashlib
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional Clang AST support for precise analysis
CLANG_AVAILABLE = False
try:
    import clang.cindex
    from clang.cindex import Index, CursorKind
    CLANG_AVAILABLE = True
except ImportError:
    pass

# unifdef flags for AArch64 analysis
# These define the target platform and enable AArch64-specific code paths
UNIFDEF_AARCH64_FLAGS = [
    '-DSSE2NEON_ARCH_AARCH64=1',
    '-D__aarch64__=1',
    '-D__ARM_FEATURE_DIRECTED_ROUNDING=1',
    '-D__ARM_FEATURE_FMA=1',
    '-D__ARM_FEATURE_CRYPTO=1',
    '-D__ARM_FEATURE_CRC32=1',
    # Undefine ARMv7-specific or optional features
    '-U__ARM_FEATURE_FRINT',  # ARMv8.5+ optional feature
]

# Reference data for validation: manually verified NEON instruction counts
# Format: intrinsic_name -> (expected_neon_count, expected_tier, source)
# Sources: 'asm' = verified against ARM64 assembly, 'manual' = code review
REFERENCE_DATA = {
    # T1: Direct mappings (1-2 NEON ops)
    '_mm_add_ps': (1, 1, 'asm'),           # vaddq_f32
    '_mm_sub_ps': (1, 1, 'asm'),           # vsubq_f32
    '_mm_mul_ps': (1, 1, 'asm'),           # vmulq_f32
    '_mm_and_ps': (1, 1, 'asm'),           # vandq_u32
    '_mm_or_ps': (1, 1, 'asm'),            # vorrq_u32
    '_mm_xor_ps': (1, 1, 'asm'),           # veorq_u32
    '_mm_cmpeq_ps': (1, 1, 'asm'),         # vceqq_f32
    '_mm_cmplt_ps': (1, 1, 'asm'),         # vcltq_f32
    '_mm_max_ps': (1, 1, 'asm'),           # vmaxq_f32
    '_mm_min_ps': (1, 1, 'asm'),           # vminq_f32
    '_mm_add_epi32': (1, 1, 'asm'),        # vaddq_s32
    '_mm_sub_epi32': (1, 1, 'asm'),        # vsubq_s32
    '_mm_setzero_ps': (1, 1, 'asm'),       # vdupq_n_f32(0)
    '_mm_set1_ps': (1, 1, 'asm'),          # vdupq_n_f32
    '_mm_load_ps': (1, 1, 'asm'),          # vld1q_f32
    '_mm_store_ps': (1, 1, 'asm'),         # vst1q_f32
    '_mm_cvtepi32_ps': (1, 1, 'asm'),      # vcvtq_f32_s32

    # T2: Few operations (3-5 NEON ops)
    '_mm_hadd_ps': (4, 2, 'asm'),          # vpaddq + vzip
    '_mm_shuffle_ps': (3, 2, 'manual'),    # Multiple vext/vtrn
    '_mm_unpacklo_ps': (1, 1, 'asm'),      # vzip1q_f32 (single instruction)
    '_mm_unpackhi_ps': (1, 1, 'asm'),      # vzip2q_f32 (single instruction)
    '_mm_blendv_ps': (3, 2, 'asm'),        # vbslq with compare
    '_mm_round_ps': (3, 2, 'asm'),         # vrndnq + masks

    # T3: Moderate emulation (6-10 NEON ops)
    '_mm_cvtps_epi32': (5, 2, 'manual'),   # Round + convert
    '_mm_packus_epi16': (4, 2, 'asm'),     # vqmovun x2
    '_mm_packs_epi32': (4, 2, 'asm'),      # vqmovn x2

    # T4: Complex emulation (10+ NEON ops or algorithmic)
    '_mm_rsqrt_ps': (13, 4, 'asm'),        # Newton-Raphson iteration
    '_mm_sqrt_ps': (13, 4, 'asm'),         # Newton-Raphson + vsqrtq
    '_mm_dp_ps': (11, 4, 'manual'),        # Dot product emulation
    '_mm_mpsadbw_epu8': (22, 4, 'asm'),    # SAD computation
    '_mm_minpos_epu16': (9, 4, 'asm'),     # Horizontal min search
    '_mm_shuffle_epi8': (3, 4, 'asm'),     # Table lookup (vtbl)
}


@dataclass
class ValidationResult:
    """Result of validating analysis against reference data."""
    intrinsic: str
    expected_neon: int
    actual_neon: int
    expected_tier: int
    actual_tier: int
    source: str
    neon_match: bool
    tier_match: bool
    neon_error: float  # Percentage error in NEON count

    @property
    def is_valid(self) -> bool:
        return self.tier_match

    @property
    def is_exact(self) -> bool:
        return self.neon_match and self.tier_match


@dataclass
class ConfidenceMetrics:
    """Confidence metrics for tier classification."""
    # Base confidence from analysis method
    method_confidence: float  # 0.0-1.0

    # Factors that increase confidence
    has_assembly_verification: bool = False
    has_ast_analysis: bool = False
    matches_reference: bool = False

    # Factors that decrease confidence
    has_complex_conditionals: bool = False
    has_platform_variants: bool = False
    near_tier_boundary: bool = False

    @property
    def overall_confidence(self) -> float:
        """Compute overall confidence score (0.0-1.0)."""
        score = self.method_confidence

        # Boost for verification methods
        if self.has_assembly_verification:
            score = min(1.0, score + 0.2)
        if self.has_ast_analysis:
            score = min(1.0, score + 0.1)
        if self.matches_reference:
            score = min(1.0, score + 0.15)

        # Penalties for uncertainty factors
        if self.has_complex_conditionals:
            score = max(0.0, score - 0.15)
        if self.has_platform_variants:
            score = max(0.0, score - 0.1)
        if self.near_tier_boundary:
            score = max(0.0, score - 0.1)

        return round(score, 2)

    @property
    def confidence_level(self) -> str:
        """Return human-readable confidence level."""
        c = self.overall_confidence
        if c >= 0.9:
            return 'high'
        elif c >= 0.7:
            return 'medium'
        elif c >= 0.5:
            return 'low'
        else:
            return 'very-low'


def validate_against_reference(results: List['IntrinsicAnalysis']) -> List[ValidationResult]:
    """
    Validate analysis results against reference data.

    Returns list of ValidationResult for each intrinsic with reference data.
    """
    validations = []

    for analysis in results:
        if analysis.name in REFERENCE_DATA:
            expected_neon, expected_tier, source = REFERENCE_DATA[analysis.name]
            actual_neon = analysis.precise_neon_count or analysis.neon_count

            neon_match = actual_neon == expected_neon
            tier_match = analysis.tier == expected_tier

            # Calculate percentage error
            if expected_neon > 0:
                neon_error = abs(actual_neon - expected_neon) / expected_neon * 100
            else:
                neon_error = 0.0 if actual_neon == 0 else 100.0

            validations.append(ValidationResult(
                intrinsic=analysis.name,
                expected_neon=expected_neon,
                actual_neon=actual_neon,
                expected_tier=expected_tier,
                actual_tier=analysis.tier,
                source=source,
                neon_match=neon_match,
                tier_match=tier_match,
                neon_error=neon_error
            ))

    return validations


def compute_confidence(analysis: 'IntrinsicAnalysis',
                       has_asm: bool = False) -> ConfidenceMetrics:
    """
    Compute confidence metrics for an intrinsic's tier classification.

    Args:
        analysis: The intrinsic analysis result
        has_asm: Whether assembly verification was performed

    Returns:
        ConfidenceMetrics with confidence scores
    """
    # Base confidence depends on analysis method used
    if has_asm:
        base = 0.9  # Assembly is most reliable
    elif analysis.precise_neon_count is not None:
        base = 0.8  # Clang AST is reliable
    else:
        base = 0.6  # Regex-based is less reliable

    metrics = ConfidenceMetrics(method_confidence=base)
    metrics.has_assembly_verification = has_asm
    metrics.has_ast_analysis = analysis.precise_neon_count is not None
    metrics.matches_reference = analysis.name in REFERENCE_DATA

    # Check for complexity factors
    if analysis.has_aarch64_path and analysis.has_armv7_path:
        metrics.has_platform_variants = True

    if analysis.complexity:
        cx = analysis.complexity
        if cx.has_conditional_neon:
            metrics.has_complex_conditionals = True

    # Check if near tier boundary
    n = analysis.precise_neon_count or analysis.neon_count
    # Tier boundaries: 2, 5, 10
    boundaries = [2, 5, 10]
    for b in boundaries:
        if abs(n - b) <= 1:
            metrics.near_tier_boundary = True
            break

    return metrics


def print_validation_report(validations: List[ValidationResult]) -> None:
    """Print validation report to stderr."""
    if not validations:
        print("No reference data available for validation", file=sys.stderr)
        return

    exact_matches = sum(1 for v in validations if v.is_exact)
    tier_matches = sum(1 for v in validations if v.tier_match)
    total = len(validations)

    print(f"\n## Validation Against Reference Data", file=sys.stderr)
    print(f"\nValidated {total} intrinsics with reference data:", file=sys.stderr)
    print(f"  - Exact NEON count match: {exact_matches}/{total} ({100*exact_matches/total:.1f}%)",
          file=sys.stderr)
    print(f"  - Correct tier classification: {tier_matches}/{total} ({100*tier_matches/total:.1f}%)",
          file=sys.stderr)

    # Show mismatches (sorted by name for consistent output)
    mismatches = [v for v in validations if not v.is_exact]
    if mismatches:
        print(f"\n### Discrepancies ({len(mismatches)}):", file=sys.stderr)
        for v in sorted(mismatches, key=lambda x: x.intrinsic):
            status = '✓' if v.tier_match else '✗'
            print(f"  {status} {v.intrinsic}: expected={v.expected_neon}(T{v.expected_tier}), "
                  f"actual={v.actual_neon}(T{v.actual_tier}), error={v.neon_error:.0f}% [{v.source}]",
                  file=sys.stderr)

    # Calculate average error
    avg_error = sum(v.neon_error for v in validations) / total
    print(f"\nAverage NEON count error: {avg_error:.1f}%", file=sys.stderr)

# NEON intrinsic patterns that map to actual instructions.
# Each pattern corresponds to one or more ARM NEON instructions.
# Patterns are grouped by category with typical cycle counts noted.
NEON_INTRINSIC_PATTERNS = [
    # Load/Store (1 cycle typically)
    # Include _lane variants for lane-specific load/store
    r'vld[1234]q?(_lane)?_[a-z0-9_]+',
    r'vst[1234]q?(_lane)?_[a-z0-9_]+',
    r'vget[q]?_lane_[a-z0-9_]+',
    r'vget_(low|high)_[a-z0-9_]+',
    r'vcombine_[a-z0-9_]+',
    r'vdup[q]?_n_[a-z0-9_]+',
    r'vdupq?_lane[q]?_[a-z0-9_]+',
    r'vmovq?_n_[a-z0-9_]+',
    r'vcreate_[a-z0-9_]+',
    r'vset_lane_[a-z0-9_]+',
    r'vsetq_lane_[a-z0-9_]+',

    # Arithmetic (1 cycle)
    # Separate mul to add _lane variants
    r'v(add|sub|neg|abs|div)[q]?_[a-z0-9_]+',
    r'vmul[q]?(_lane[q]?)?_[a-z0-9_]+',
    r'vmla[l]?[q]?(_lane[q]?)?_[a-z0-9_]+',
    r'vmls[l]?[q]?(_lane[q]?)?_[a-z0-9_]+',

    # FMA with lane variants (1-2 cycles)
    r'vfma[q]?_[a-z0-9_]+',
    r'vfms[q]?_[a-z0-9_]+',
    r'vfma_lane[q]?_[a-z0-9_]+',
    r'vfms_lane[q]?_[a-z0-9_]+',

    # Min/Max (1 cycle)
    r'v(min|max)[q]?_[a-z0-9_]+',
    r'vp(min|max)[q]?_[a-z0-9_]+',

    # Comparison (1 cycle)
    r'vc(eq|gt|ge|lt|le)[q]?_[a-z0-9_]+',

    # Bitwise test (1 cycle)
    r'vtst[q]?_[a-z0-9_]+',

    # Logical (1 cycle)
    r'v(and|orr|eor|bic|orn|mvn|bsl)[q]?_[a-z0-9_]+',

    # Shift (1 cycle)
    r'vshl[q]?_[a-z0-9_]+',
    r'vshr[q]?_n_[a-z0-9_]+',
    r'vrshr[q]?_n_[a-z0-9_]+',
    r'vsra[q]?_n_[a-z0-9_]+',
    r'vrsra[q]?_n_[a-z0-9_]+',
    r'vshl_n_[a-z0-9_]+',
    r'vshlq_n_[a-z0-9_]+',
    r'vqshl[q]?_[a-z0-9_]+',

    # Rounding/Conversion (1-2 cycles)
    r'vcvt[q]?_[a-z0-9_]+',
    r'vrnd[nmap]?[q]?_[a-z0-9_]+',

    # Reciprocal estimates (3-5 cycles with refinement)
    r'vrecpe[q]?_[a-z0-9_]+',
    r'vrecps[q]?_[a-z0-9_]+',
    r'vrsqrte[q]?_[a-z0-9_]+',
    r'vrsqrts[q]?_[a-z0-9_]+',
    r'vsqrt[q]?_[a-z0-9_]+',

    # Pairwise (1-2 cycles)
    r'vpadd[lq]?_[a-z0-9_]+',
    r'vpadal[q]?_[a-z0-9_]+',

    # Table lookup (2-4 cycles)
    r'vqtbl[1234]q?_[a-z0-9_]+',
    r'vqtbx[1234]q?_[a-z0-9_]+',
    r'vtbl[1234]_[a-z0-9_]+',
    r'vtbx[1234]_[a-z0-9_]+',

    # Zip/Unzip/Transpose (1-2 cycles)
    r'vzip[12]?q?_[a-z0-9_]+',
    r'vuzp[12]?q?_[a-z0-9_]+',
    r'vtrn[12]?q?_[a-z0-9_]+',

    # Extract (1 cycle)
    r'vext[q]?_[a-z0-9_]+',

    # Reverse (1 cycle)
    r'vrev(16|32|64)[q]?_[a-z0-9_]+',

    # Narrow/Widen (1 cycle)
    r'vmovl_[a-z0-9_]+',
    r'vmovn_[a-z0-9_]+',
    r'vqmovn_[a-z0-9_]+',
    r'vqmovun_[a-z0-9_]+',

    # Saturating arithmetic (1 cycle)
    r'vq(add|sub|abs|neg)[q]?_[a-z0-9_]+',

    # Saturating doubling multiply high (1-2 cycles)
    # Include _lane variants and vqrdmulh
    r'v(qrdmulh|qdmulh)[q]?(_lane[q]?)?_[a-z0-9_]+',
    r'vqdmlal(_lane)?_[a-z0-9_]+',
    r'vqdmlsl(_lane)?_[a-z0-9_]+',

    # Count (1 cycle)
    r'v(cnt|clz|cls)[q]?_[a-z0-9_]+',

    # CRC (crypto extension, 1-3 cycles)
    r'__crc32[a-z]*',
    r'vcrc32[a-z]*',

    # AES (crypto extension, 3-4 cycles)
    r'vaes[ed]q_[a-z0-9_]+',
    r'vaes[im]?cq_[a-z0-9_]+',

    # SHA (crypto extension)
    r'vsha[0-9a-z]+_[a-z0-9_]+',

    # Polynomial multiply
    r'vmull_p[0-9]+',
    r'vmull_high_p[0-9]+',

    # Dot product (ARMv8.2+)
    r'vdot[q]?_[a-z0-9_]+',

    # Move
    r'vmov_[a-z0-9_]+',

    # Across lanes reduction (2-3 cycles)
    r'v(add|max|min)v[q]?_[a-z0-9_]+',
]

# Compile patterns for efficiency
NEON_PATTERNS = [re.compile(p) for p in NEON_INTRINSIC_PATTERNS]

# vreinterpret doesn't generate instructions (type cast only)
REINTERPRET_PATTERN = re.compile(r'vreinterpret[q]?_[a-z0-9_]+_[a-z0-9_]+')

# Instruction cost mapping based on ARM Cortex-A72 Software Optimization Guide
# Maps NEON intrinsic patterns to actual machine instruction counts
# Cost 0: No instruction generated (type casts, compiler hints)
# Cost 1: Single NEON instruction (most common)
# Cost 2-4: Multi-instruction sequences (div, reciprocal refinement)
NEON_INSTRUCTION_COSTS = {
    # Zero-cost operations (no actual instruction)
    r'vcreate_': 0,           # Just a type cast
    r'vget_(low|high)_': 0,   # Register aliasing (may be free)
    r'vcombine_': 0,          # Register pairing (often free)

    # Single instruction (cost 1) - default for most operations
    # Explicitly listed for documentation, most patterns default to 1

    # Multi-instruction operations
    r'vrecpe': 2,             # Reciprocal estimate + step
    r'vrecps': 1,             # Single reciprocal step
    r'vrsqrte': 2,            # Reciprocal sqrt estimate
    r'vrsqrts': 1,            # Reciprocal sqrt step
    r'vsqrt': 1,              # Single sqrt (ARMv8.2+, otherwise 3-4)

    # Table lookups (2-4 cycles depending on table size)
    r'vqtbl[1]': 1,           # Single register table lookup
    r'vqtbl[234]': 2,         # Multi-register table lookup
    r'vtbl[1]': 1,
    r'vtbl[234]': 2,

    # Widening operations (may be 2 instructions for full result)
    r'vmull': 1,              # Widening multiply
    r'vmlal': 1,              # Widening multiply-accumulate
    r'vmlsl': 1,              # Widening multiply-subtract
    r'vaddl': 1,              # Widening add
    r'vsubl': 1,              # Widening subtract

    # Narrowing operations (may need additional saturation)
    r'vqmovn': 1,             # Saturating narrow
    r'vqmovun': 1,            # Saturating unsigned narrow

    # Pairwise across vector (2-3 cycles, sequential dependency)
    r'vaddv': 2,              # Add across vector (reduction)
    r'vmaxv': 2,              # Max across vector
    r'vminv': 2,              # Min across vector
    r'vsaddlv': 2,            # Signed add long across
    r'vuaddlv': 2,            # Unsigned add long across

    # Crypto operations (if available)
    r'vaeseq': 2,             # AES encrypt round
    r'vaesdq': 2,             # AES decrypt round
    r'vaesmcq': 1,            # AES mix columns
    r'vaesimcq': 1,           # AES inverse mix columns
    r'vsha1': 2,              # SHA1 operations
    r'vsha256': 2,            # SHA256 operations
    r'vmull_p64': 2,          # Polynomial multiply (PMULL)

    # CRC operations
    r'__crc32': 1,            # CRC32 (single instruction if HW support)
}

# Compile cost patterns
NEON_COST_PATTERNS = [(re.compile(p), cost) for p, cost in NEON_INSTRUCTION_COSTS.items()]


def get_intrinsic_cost(intrinsic_name: str) -> int:
    """Get the instruction cost for a NEON intrinsic.

    Returns the number of machine instructions this intrinsic typically generates.
    Based on ARM Cortex-A72 Software Optimization Guide.
    """
    for pattern, cost in NEON_COST_PATTERNS:
        if pattern.search(intrinsic_name):
            return cost
    return 1  # Default: single instruction


class ResultCache:
    """Persistent disk cache for analysis results.

    Caches intrinsic analysis results based on function body hash.
    Cache invalidation happens automatically when:
    - Function body changes (different hash)
    - Cache version changes (format updates)
    - Cache file is older than TTL (default: 7 days)

    Cache location: ~/.cache/sse2neon-tiers/
    """
    CACHE_VERSION = 1
    DEFAULT_TTL_DAYS = 7

    def __init__(self, cache_dir: str = None, ttl_days: int = None):
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'sse2neon-tiers'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = (ttl_days or self.DEFAULT_TTL_DAYS) * 86400
        self._hits = 0
        self._misses = 0

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{key}.pkl"

    def _is_valid(self, cache_path: Path) -> bool:
        """Check if cache file is valid (exists and not expired)."""
        if not cache_path.exists():
            return False
        age = time.time() - cache_path.stat().st_mtime
        return age < self.ttl_seconds

    def get(self, func_name: str, body_hash: str) -> Optional[dict]:
        """Get cached result for a function.

        Args:
            func_name: Function name (for organizing cache)
            body_hash: Hash of function body (for invalidation)

        Returns:
            Cached result dict or None if not found/invalid
        """
        cache_path = self._get_cache_path(f"{func_name}_{body_hash[:16]}")
        if not self._is_valid(cache_path):
            self._misses += 1
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            if data.get('version') != self.CACHE_VERSION:
                self._misses += 1
                return None
            if data.get('body_hash') != body_hash:
                self._misses += 1
                return None
            self._hits += 1
            return data.get('result')
        except Exception:
            self._misses += 1
            return None

    def set(self, func_name: str, body_hash: str, result: dict) -> None:
        """Store result in cache.

        Args:
            func_name: Function name
            body_hash: Hash of function body
            result: Analysis result to cache
        """
        cache_path = self._get_cache_path(f"{func_name}_{body_hash[:16]}")
        try:
            data = {
                'version': self.CACHE_VERSION,
                'body_hash': body_hash,
                'result': result,
                'timestamp': time.time()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass  # Cache failures are non-fatal

    def get_stats(self) -> Tuple[int, int]:
        """Return (hits, misses) statistics."""
        return self._hits, self._misses

    def clear(self) -> int:
        """Clear all cached results. Returns count of files removed."""
        count = 0
        for cache_file in self.cache_dir.glob('*.pkl'):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        return count


def compute_body_hash(body: str) -> str:
    """Compute hash of function body for cache key."""
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass
class ComplexityAnalysis:
    """Detailed complexity analysis for an intrinsic."""
    # Loop analysis
    loop_count: int = 0           # Number of loops detected
    loop_types: List[str] = field(default_factory=list)  # 'for', 'while', 'do-while'
    nested_depth: int = 0         # Maximum nesting depth
    has_data_dependent_loop: bool = False  # Loop bound depends on input

    # Memory access patterns
    sequential_access: int = 0    # Contiguous memory access (efficient)
    scattered_access: int = 0     # Non-contiguous access (cache-unfriendly)
    lane_extractions: int = 0     # vgetq_lane operations (scalar extraction)

    # Algorithmic indicators
    has_recursion_hint: bool = False  # Recursive patterns (rare in intrinsics)
    has_conditional_neon: bool = False  # NEON ops inside conditionals
    has_table_lookup: bool = False  # Table-based computation
    has_reduction: bool = False   # Horizontal reduction operations

    # Scalar operation breakdown
    scalar_arithmetic: int = 0    # +, -, *, /, %
    scalar_bitwise: int = 0       # &, |, ^, <<, >>
    scalar_comparison: int = 0    # <, >, ==, !=
    scalar_array_access: int = 0  # arr[i] patterns

    @property
    def total_scalar_ops(self) -> int:
        return self.scalar_arithmetic + self.scalar_bitwise + self.scalar_comparison

    @property
    def complexity_score(self) -> int:
        """Compute overall complexity score (0-100)."""
        score = 0
        # Loops add significant complexity
        score += self.loop_count * 15
        score += self.nested_depth * 10
        if self.has_data_dependent_loop:
            score += 20
        # Scattered access is expensive
        score += self.scattered_access * 5
        score += self.lane_extractions * 3
        # Algorithmic patterns
        if self.has_conditional_neon:
            score += 10
        if self.has_table_lookup:
            score += 15
        if self.has_reduction:
            score += 5
        # Scalar operations
        score += self.total_scalar_ops * 2
        score += self.scalar_array_access * 3
        return min(100, score)


@dataclass
class IntrinsicAnalysis:
    """Analysis result for a single intrinsic."""
    name: str
    line_start: int
    line_end: int
    neon_count: int           # Raw count of NEON intrinsic calls
    reinterpret_count: int
    scalar_count: int         # Number of scalar operations detected
    has_aarch64_path: bool
    has_armv7_path: bool
    has_loop: bool
    body: str = ""
    tier: int = 0
    tier_label: str = ""
    neon_intrinsics: List[str] = field(default_factory=list)
    precise_neon_count: Optional[int] = None  # Updated by Clang AST analysis
    weighted_neon_count: Optional[int] = None  # Instruction-weighted count
    complexity: Optional[ComplexityAnalysis] = None  # Detailed complexity analysis


def extract_intrinsic_body(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """Extract the complete function body from starting line."""
    brace_count = 0
    started = False
    body_lines = []
    end_idx = start_idx

    for i in range(start_idx, len(lines)):
        line = lines[i]
        body_lines.append(line)

        for char in line:
            if char == '{':
                brace_count += 1
                started = True
            elif char == '}':
                brace_count -= 1

        if started and brace_count == 0:
            end_idx = i
            break

    return '\n'.join(body_lines), end_idx


def count_neon_intrinsics(body: str, weighted: bool = False) -> Tuple[int, List[str]]:
    """Count actual NEON intrinsics in function body with context-aware matching.

    Improvements over simple regex:
    1. Removes comments and string literals to avoid false positives
    2. Validates matches are actual function calls (followed by '(')
    3. Uses word boundaries to prevent partial matches
    4. Deduplicates by position - prefers longest match when patterns overlap
    5. Optionally returns weighted instruction count based on ARM documentation

    Args:
        body: Function body to analyze
        weighted: If True, return instruction-weighted count instead of call count

    Returns:
        Tuple of (count, list of matched intrinsic names)
        Count is either raw call count or weighted instruction count
    """
    # Remove comments first
    body = re.sub(r'//.*$', '', body, flags=re.MULTILINE)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)

    # Remove string literals to avoid matching intrinsic names in strings
    body = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', body)

    # Track matches by position to handle overlapping patterns
    # Key: match start position, Value: (intrinsic_name, match_length)
    matches_by_pos: Dict[int, Tuple[str, int]] = {}

    # Find all potential NEON intrinsic calls with word boundary and call validation
    # Pattern: word boundary + NEON name + optional whitespace + opening paren
    for pattern in NEON_PATTERNS:
        # Wrap pattern with word boundary and function call check
        call_pattern = re.compile(r'\b(' + pattern.pattern + r')\s*\(')
        for match in call_pattern.finditer(body):
            pos = match.start(1)
            intrinsic_name = match.group(1)
            name_len = len(intrinsic_name)
            # Keep longest match at each position (handles overlapping patterns)
            if pos not in matches_by_pos or name_len > matches_by_pos[pos][1]:
                matches_by_pos[pos] = (intrinsic_name, name_len)

    found = [name for name, _ in matches_by_pos.values()]

    if weighted:
        # Calculate weighted instruction count
        total_cost = sum(get_intrinsic_cost(name) for name in found)
        return total_cost, found
    else:
        return len(found), found


def count_reinterprets(body: str) -> int:
    """Count vreinterpret casts (don't generate instructions).

    Uses same context-aware approach as count_neon_intrinsics.
    """
    # Remove comments and strings
    body = re.sub(r'//.*$', '', body, flags=re.MULTILINE)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
    body = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', body)

    # Match actual function calls with word boundary
    call_pattern = re.compile(r'\b(' + REINTERPRET_PATTERN.pattern + r')\s*\(')
    return len(call_pattern.findall(body))


class ClangASTAnalyzer:
    """
    Clang AST analyzer for precise NEON intrinsic counting.

    Uses libclang to parse C code and traverse the AST to find
    actual NEON intrinsic calls, providing more accurate counts
    than regex-based approaches.

    Optimizations:
    - Caches header file content to avoid re-reading
    - Hash-based caching for identical function bodies
    - Tracks cache statistics for debugging
    """

    def __init__(self, header_file: str):
        if not CLANG_AVAILABLE:
            raise RuntimeError("libclang not available. Install with: pip install clang")
        self.header_file = header_file
        self.index = Index.create()

        # Cache the header file content
        with open(header_file, 'r') as f:
            self._header_content = f.read()

        # Hash-based cache: body_hash -> neon_count
        self._result_cache: Dict[str, Optional[int]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def extract_function_content(self, content: str, func_name: str) -> Optional[str]:
        """Extract specific function content from header file."""
        pattern = rf'FORCE_INLINE\s+\S+\s+({re.escape(func_name)})\s*\([^)]*\)\s*\{{'
        matches = list(re.finditer(pattern, content))

        if not matches:
            return None

        match = matches[0]
        start = match.start()

        # Find the complete function body by counting braces
        brace_count = 0
        for i, char in enumerate(content[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return content[start:i+1]

        return None

    def traverse_ast(self, cursor, level=0):
        """Traverse AST and yield all nodes."""
        yield cursor
        for child in cursor.get_children():
            yield from self.traverse_ast(child, level + 1)

    def is_neon_intrinsic(self, cursor) -> bool:
        """Check if AST node represents a NEON intrinsic call."""
        if cursor.kind == CursorKind.CALL_EXPR:
            func_name = cursor.spelling
            if not func_name:
                func_name = str(cursor.displayname)

            for pattern in NEON_PATTERNS:
                if pattern.match(func_name):
                    return True
        return False

    def get_cache_stats(self) -> Tuple[int, int]:
        """Return (cache_hits, cache_misses) statistics."""
        return self._cache_hits, self._cache_misses

    def count_neon_intrinsics_ast(self, func_name: str) -> Optional[int]:
        """
        Count NEON intrinsics using Clang AST traversal.

        Returns the count or None if analysis fails.
        Uses hash-based caching for identical function bodies.
        """
        func_content = self.extract_function_content(self._header_content, func_name)
        if not func_content:
            return None

        # Hash-based cache lookup
        import hashlib
        body_hash = hashlib.md5(func_content.encode()).hexdigest()
        if body_hash in self._result_cache:
            self._cache_hits += 1
            return self._result_cache[body_hash]

        self._cache_misses += 1

        # Create a wrapper to make it compilable
        wrapper = '''
#include <arm_neon.h>
#define FORCE_INLINE static inline __attribute__((always_inline))
#define SSE2NEON_ARCH_AARCH64 1

''' + func_content

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as tmp:
                tmp.write(wrapper)
                tmp.flush()
                temp_path = tmp.name

            # Parse with Clang
            tu = self.index.parse(
                temp_path,
                args=[
                    '-I.',
                    '-target', 'aarch64-linux-gnu',
                    '-D__aarch64__',
                    '-DSSE2NEON_ARCH_AARCH64=1',
                ]
            )

            neon_count = 0
            for node in self.traverse_ast(tu.cursor):
                if self.is_neon_intrinsic(node):
                    neon_count += 1

            # Cache successful result
            self._result_cache[body_hash] = neon_count
            return neon_count

        except Exception:
            # Cache failure as None
            self._result_cache[body_hash] = None
            return None

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


class AssemblyAnalyzer:
    """
    Level 3 analyzer: Generate and analyze actual assembly output.

    Most accurate but slowest method - compiles intrinsic and counts
    actual ARM64 SIMD instructions in the generated assembly.
    Reserved for T4 intrinsics where precision is critical.
    """

    # ARM64 SIMD instruction patterns (NEON/ASIMD)
    ARM64_SIMD_PATTERNS = [
        # Vector arithmetic
        r'\b(fadd|fsub|fmul|fdiv|fneg|fabs|fmax|fmin|fmaxnm|fminnm)[vsd]?\b',
        r'\b(add|sub|mul|neg|abs)[vp]?\d*[bhsdq]?\b',
        # Vector compare
        r'\b(fcm|cm)(eq|ge|gt|le|lt|hi|hs|lo|ls)[vsd]?\b',
        # Vector logical
        r'\b(and|orr|eor|bic|orn|mvn|bsl|bit|bif)[v]?\b',
        # Vector shift
        r'\b(shl|shr|sshr|ushr|ssra|usra|srshr|urshr)[v]?\b',
        r'\b(sqshl|uqshl|sqrshl|uqrshl)\b',
        # Vector load/store
        r'\bld[1234][r]?\b',
        r'\bst[1234]\b',
        # Vector move/dup
        r'\b(dup|ins|mov|movi|mvni|fmov)[vsd]?\b',
        # Vector convert
        r'\b(fcvt|scvtf|ucvtf|fcvtz|fcvtn|fcvtl)[snuzm]*\b',
        # Vector permute
        r'\b(zip[12]|uzp[12]|trn[12]|ext|rev|tbl|tbx)\b',
        # Vector reduce
        r'\b(addv|saddlv|uaddlv|smaxv|umaxv|sminv|uminv|fmaxv|fminv)\b',
        # FMA operations
        r'\b(fmla|fmls|fmadd|fmsub|fnmadd|fnmsub)[vsd]?\b',
        # Saturating arithmetic
        r'\b(sqadd|uqadd|sqsub|uqsub|sqneg|sqabs)\b',
        # Widening/narrowing
        r'\b(saddl|uaddl|ssubl|usubl|saddw|uaddw|ssubw|usubw)\b',
        r'\b(addhn|subhn|raddhn|rsubhn|sqxtn|uqxtn|sqxtun)\b',
        # Multiply accumulate
        r'\b(mla|mls|smlal|umlal|smlsl|umlsl|smull|umull)\b',
        r'\b(sqdmulh|sqrdmulh|sqdmlal|sqdmlsl|sqdmull)\b',
        # Reciprocal
        r'\b(frecpe|frecps|frsqrte|frsqrts|fsqrt)\b',
        # Pairwise
        r'\b(addp|faddp|smaxp|umaxp|sminp|uminp|fmaxp|fminp)\b',
        # Crypto (if enabled)
        r'\b(aese|aesd|aesmc|aesimc|sha1|sha256|pmull)\b',
        # CRC (if enabled)
        r'\b(crc32[cbhwx]?)\b',
    ]

    def __init__(self, header_file: str, compiler: str = None):
        self.header_file = header_file
        self._header_content = None
        self._result_cache: Dict[str, int] = {}

        # Find suitable cross-compiler
        self.compiler = compiler or self._find_compiler()
        if not self.compiler:
            raise RuntimeError("No ARM64 cross-compiler found. "
                               "Install aarch64-linux-gnu-gcc or use native ARM64.")

        # Compile patterns
        self._simd_patterns = [re.compile(p, re.IGNORECASE) for p in self.ARM64_SIMD_PATTERNS]

    def _find_compiler(self) -> Optional[str]:
        """Find available ARM64 compiler."""
        import shutil
        candidates = [
            'aarch64-linux-gnu-gcc',
            'aarch64-linux-gnu-clang',
            'clang',  # May support --target=aarch64
            'gcc',    # On native ARM64
        ]
        for cc in candidates:
            if shutil.which(cc):
                # Verify it can target ARM64
                try:
                    result = subprocess.run(
                        [cc, '--version'],
                        capture_output=True, text=True, timeout=5
                    )
                    if 'aarch64' in cc or 'aarch64' in result.stdout.lower():
                        return cc
                    # For clang, check if it supports aarch64 target
                    if 'clang' in cc:
                        return cc
                except Exception:
                    continue
        return None

    def _get_header_content(self) -> str:
        """Lazy load header content."""
        if self._header_content is None:
            with open(self.header_file, 'r') as f:
                self._header_content = f.read()
        return self._header_content

    def count_simd_instructions(self, func_name: str) -> Optional[int]:
        """
        Compile intrinsic and count SIMD instructions in generated assembly.

        Returns instruction count or None if compilation fails.
        """
        import hashlib

        # Extract function content
        content = self._get_header_content()
        pattern = rf'FORCE_INLINE\s+\S+\s+({re.escape(func_name)})\s*\([^)]*\)\s*\{{'
        match = re.search(pattern, content)
        if not match:
            return None

        start = match.start()
        brace_count = 0
        end = start
        for i, char in enumerate(content[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        func_content = content[start:end]

        # Check cache
        body_hash = hashlib.md5(func_content.encode()).hexdigest()
        if body_hash in self._result_cache:
            return self._result_cache[body_hash]

        # Create test wrapper that calls the intrinsic
        # Extract return type and parameters
        sig_match = re.search(r'FORCE_INLINE\s+(\S+)\s+' + re.escape(func_name) + r'\s*\(([^)]*)\)', func_content)
        if not sig_match:
            return None

        ret_type = sig_match.group(1)
        params = sig_match.group(2).strip()

        wrapper = f'''
#include "sse2neon.h"

{ret_type} test_wrapper({params if params else 'void'}) {{
'''
        # Generate call with dummy arguments based on parameter types
        if params and params != 'void':
            param_list = []
            for p in params.split(','):
                p = p.strip()
                # Extract parameter name (last word)
                parts = p.split()
                if parts:
                    param_list.append(parts[-1].replace('*', '').replace('&', ''))
            wrapper += f'    return {func_name}({", ".join(param_list)});\n'
        else:
            wrapper += f'    return {func_name}();\n'

        wrapper += '}\n'

        temp_c = None
        temp_s = None
        try:
            # Write source
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(wrapper)
                temp_c = f.name

            temp_s = temp_c.replace('.c', '.s')

            # Compile to assembly
            cmd = [self.compiler]
            if 'clang' in self.compiler and 'aarch64' not in self.compiler:
                cmd.extend(['--target=aarch64-linux-gnu'])
            cmd.extend([
                '-S', '-O2',
                '-I', os.path.dirname(self.header_file) or '.',
                '-D__aarch64__',
                '-DSSE2NEON_PRECISE_MINMAX=0',
                '-o', temp_s,
                temp_c
            ])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self._result_cache[body_hash] = None
                return None

            # Parse assembly and count SIMD instructions
            with open(temp_s, 'r') as f:
                asm_content = f.read()

            simd_count = 0
            for line in asm_content.split('\n'):
                line = line.strip()
                # Skip directives and labels
                if not line or line.startswith('.') or line.endswith(':'):
                    continue
                # Check for SIMD instructions
                for pattern in self._simd_patterns:
                    if pattern.search(line):
                        simd_count += 1
                        break

            self._result_cache[body_hash] = simd_count
            return simd_count

        except Exception:
            self._result_cache[body_hash] = None
            return None

        finally:
            for f in [temp_c, temp_s]:
                if f and os.path.exists(f):
                    os.unlink(f)


def refine_with_clang_ast(results: List['IntrinsicAnalysis'],
                          header_file: str,
                          tiers_to_refine: List[int] = None,
                          parallel_jobs: int = 1) -> List['IntrinsicAnalysis']:
    """
    Apply Clang AST analysis to specified tier intrinsics for more precise NEON count.

    Args:
        results: List of intrinsic analyses
        header_file: Path to sse2neon.h
        tiers_to_refine: List of tiers to refine (default: [2, 3, 4])
        parallel_jobs: Number of parallel workers (1 = sequential)

    Returns:
        Updated list of analyses with precise_neon_count populated
    """
    if not CLANG_AVAILABLE:
        print("Warning: libclang not available, skipping AST refinement", file=sys.stderr)
        return results

    if tiers_to_refine is None:
        tiers_to_refine = [2, 3, 4]

    # Filter intrinsics that need refinement
    to_refine = [a for a in results if a.tier in tiers_to_refine]
    if not to_refine:
        return results

    mode_str = f"parallel ({parallel_jobs} workers)" if parallel_jobs > 1 else "sequential"
    print(f"Applying Clang AST analysis to {len(to_refine)} T{'/T'.join(map(str, tiers_to_refine))} "
          f"intrinsics ({mode_str})...", file=sys.stderr)

    try:
        analyzer = ClangASTAnalyzer(header_file)
    except Exception as e:
        print(f"Warning: Failed to initialize Clang AST analyzer: {e}", file=sys.stderr)
        return results

    def analyze_one(analysis: 'IntrinsicAnalysis') -> Tuple[str, Optional[int]]:
        """Analyze a single intrinsic (thread-safe via analyzer's internal caching)."""
        precise_count = analyzer.count_neon_intrinsics_ast(analysis.name)
        return (analysis.name, precise_count)

    # Collect results (parallel or sequential)
    ast_results: Dict[str, Optional[int]] = {}

    if parallel_jobs > 1:
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {executor.submit(analyze_one, a): a for a in to_refine}
            for future in as_completed(futures):
                name, count = future.result()
                ast_results[name] = count
    else:
        for analysis in to_refine:
            name, count = analyze_one(analysis)
            ast_results[name] = count

    # Apply results
    refined_count = 0
    skipped_count = 0
    for analysis in results:
        if analysis.name in ast_results:
            precise_count = ast_results[analysis.name]
            if precise_count is not None:
                # Sanity check: if AST returns 0 but regex found NEON ops,
                # the AST likely failed to parse properly - skip this result
                if precise_count == 0 and analysis.neon_count > 0:
                    skipped_count += 1
                    continue

                analysis.precise_neon_count = precise_count
                old_tier = analysis.tier
                analysis.tier, analysis.tier_label = classify_tier(analysis)
                if old_tier != analysis.tier:
                    refined_count += 1

    # Report statistics
    cache_hits, cache_misses = analyzer.get_cache_stats()
    msg_parts = []
    if refined_count > 0:
        msg_parts.append(f"refined {refined_count}")
    if skipped_count > 0:
        msg_parts.append(f"skipped {skipped_count} (AST parse issues)")
    if cache_hits > 0:
        msg_parts.append(f"cache hits: {cache_hits}")

    if msg_parts:
        print(f"  {', '.join(msg_parts)}", file=sys.stderr)

    return results


def refine_with_assembly(results: List['IntrinsicAnalysis'],
                          header_file: str,
                          tiers_to_refine: List[int] = None,
                          parallel_jobs: int = 1) -> List['IntrinsicAnalysis']:
    """
    Level 3: Apply assembly generation analysis to specified tier intrinsics.

    Most accurate method - compiles intrinsics and counts actual SIMD instructions
    in generated ARM64 assembly. Reserved for T4 intrinsics where precision is critical.

    Args:
        results: List of intrinsic analyses
        header_file: Path to sse2neon.h
        tiers_to_refine: List of tiers to refine (default: [4] for T4 only)
        parallel_jobs: Number of parallel workers (1 = sequential)

    Returns:
        Updated list with assembly_simd_count populated for analyzed intrinsics
    """
    if tiers_to_refine is None:
        tiers_to_refine = [4]  # Only T4 by default (most expensive analysis)

    # Filter intrinsics that need refinement
    to_refine = [a for a in results if a.tier in tiers_to_refine]
    if not to_refine:
        return results

    mode_str = f"parallel ({parallel_jobs} workers)" if parallel_jobs > 1 else "sequential"
    print(f"Applying assembly analysis to {len(to_refine)} T{'/T'.join(map(str, tiers_to_refine))} "
          f"intrinsics ({mode_str})...", file=sys.stderr)

    try:
        analyzer = AssemblyAnalyzer(header_file)
        print(f"  Using compiler: {analyzer.compiler}", file=sys.stderr)
    except RuntimeError as e:
        print(f"Warning: {e}", file=sys.stderr)
        return results

    def analyze_one(analysis: 'IntrinsicAnalysis') -> Tuple[str, Optional[int], int]:
        """Analyze a single intrinsic. Returns (name, simd_count, regex_count)."""
        simd_count = analyzer.count_simd_instructions(analysis.name)
        return (analysis.name, simd_count, analysis.neon_count)

    # Collect results (parallel or sequential)
    asm_results: List[Tuple[str, Optional[int], int]] = []

    if parallel_jobs > 1:
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = [executor.submit(analyze_one, a) for a in to_refine]
            for future in as_completed(futures):
                asm_results.append(future.result())
    else:
        for analysis in to_refine:
            asm_results.append(analyze_one(analysis))

    # Process results
    analyzed_count = 0
    failed_count = 0

    for name, simd_count, regex_count in asm_results:
        if simd_count is not None:
            analyzed_count += 1
            # Log significant differences
            if abs(simd_count - regex_count) > 3:
                print(f"    {name}: regex={regex_count}, asm={simd_count}",
                      file=sys.stderr)
        else:
            failed_count += 1

    msg = f"  Analyzed {analyzed_count} intrinsic(s)"
    if failed_count > 0:
        msg += f", {failed_count} failed to compile"
    print(msg, file=sys.stderr)

    return results


def analyze_complexity(body: str) -> ComplexityAnalysis:
    """
    Perform detailed complexity analysis on function body.

    Analyzes:
    - Loop structures (for, while, do-while) and nesting
    - Memory access patterns (sequential vs scattered)
    - Scalar operations (arithmetic, bitwise, comparisons)
    - Algorithmic indicators (tables, reductions, conditionals)

    Returns:
        ComplexityAnalysis with detailed breakdown
    """
    result = ComplexityAnalysis()

    # Remove comments for accurate analysis
    clean_body = re.sub(r'//.*$', '', body, flags=re.MULTILINE)
    clean_body = re.sub(r'/\*.*?\*/', '', clean_body, flags=re.DOTALL)

    # --- Loop Analysis ---
    # Detect for loops
    for_loops = re.findall(r'\bfor\s*\([^)]*\)', clean_body)
    result.loop_count += len(for_loops)
    result.loop_types.extend(['for'] * len(for_loops))

    # Detect while loops (not do-while)
    while_loops = re.findall(r'(?<!\bdo\s)\bwhile\s*\([^)]*\)', clean_body)
    result.loop_count += len(while_loops)
    result.loop_types.extend(['while'] * len(while_loops))

    # Detect do-while loops
    do_while_loops = re.findall(r'\bdo\s*\{', clean_body)
    result.loop_count += len(do_while_loops)
    result.loop_types.extend(['do-while'] * len(do_while_loops))

    # Calculate nesting depth by counting brace levels around loop keywords
    if result.loop_count > 0:
        max_depth = 0
        depth = 0
        in_loop_context = False
        for i, char in enumerate(clean_body):
            if char == '{':
                depth += 1
                if in_loop_context:
                    max_depth = max(max_depth, depth)
            elif char == '}':
                depth -= 1
            # Check if we're entering a loop
            if clean_body[i:i+3] == 'for' or clean_body[i:i+5] == 'while':
                in_loop_context = True
            elif char == ';' and depth == 1:
                in_loop_context = False
        result.nested_depth = max_depth

    # Data-dependent loops (bounds depend on input, not constants)
    # Look for: for(... i < n ...) where n is not a literal
    data_dep_pattern = re.compile(r'\bfor\s*\([^;]*;\s*\w+\s*[<>=]+\s*([a-zA-Z_]\w*)')
    for match in data_dep_pattern.finditer(clean_body):
        bound_var = match.group(1)
        # Check if bound is a non-constant (not a number)
        if not re.match(r'^\d+$', bound_var):
            result.has_data_dependent_loop = True
            break

    # --- Memory Access Patterns ---
    # Sequential access: vld1q, vst1q (contiguous loads/stores)
    sequential_patterns = [
        r'\bvld1q?_[a-z0-9_]+\s*\(',    # Vector load
        r'\bvst1q?_[a-z0-9_]+\s*\(',    # Vector store
        r'\bvld[234]q?_[a-z0-9_]+\s*\(',  # Multi-vector load
    ]
    for pattern in sequential_patterns:
        result.sequential_access += len(re.findall(pattern, clean_body))

    # Scattered access: lane operations, indexed access
    scattered_patterns = [
        r'\bvgetq?_lane_[a-z0-9_]+\s*\(',  # Lane extraction
        r'\bvsetq?_lane_[a-z0-9_]+\s*\(',  # Lane insertion
        r'\bvld1q?_lane_[a-z0-9_]+\s*\(',  # Lane load
    ]
    for pattern in scattered_patterns:
        matches = len(re.findall(pattern, clean_body))
        result.scattered_access += matches
        if 'vget' in pattern:
            result.lane_extractions += matches

    # --- Algorithmic Indicators ---
    # Table lookup operations
    if re.search(r'\bvtbl[1234]q?_[a-z0-9_]+\s*\(', clean_body):
        result.has_table_lookup = True
    if re.search(r'\bvqtbl[1234]q?_[a-z0-9_]+\s*\(', clean_body):
        result.has_table_lookup = True

    # Reduction operations (horizontal ops)
    reduction_patterns = [
        r'\bvaddvq?_[a-z0-9_]+\s*\(',   # Add across vector
        r'\bvmaxvq?_[a-z0-9_]+\s*\(',   # Max across vector
        r'\bvminvq?_[a-z0-9_]+\s*\(',   # Min across vector
        r'\bvpaddq?_[a-z0-9_]+\s*\(',   # Pairwise add
    ]
    for pattern in reduction_patterns:
        if re.search(pattern, clean_body):
            result.has_reduction = True
            break

    # NEON operations inside conditionals (indicates branching complexity)
    # Look for: if (...) { ... NEON ... }
    if_blocks = re.findall(r'\bif\s*\([^)]*\)\s*\{[^}]*\}', clean_body, re.DOTALL)
    for block in if_blocks:
        for neon_pattern in ['vadd', 'vsub', 'vmul', 'vld', 'vst', 'vget', 'vset']:
            if neon_pattern in block:
                result.has_conditional_neon = True
                break

    # --- Scalar Operation Breakdown ---
    # Skip common false positives
    skip_vars = {'i', 'j', 'k', 'n', 'idx', 'index', 'iter', 'cnt', 'len', 'size'}

    # Scalar arithmetic operations with actual computation
    # Pattern: type var = expr (where expr involves computation)
    scalar_decl = re.compile(
        r'\b(u?int(8|16|32|64)_t|float|double)\s+(\w+)\s*=\s*([^;]+);'
    )
    for match in scalar_decl.finditer(clean_body):
        var_name = match.group(3)
        init_value = match.group(4).strip()

        # Skip loop counters and simple literals
        if var_name in skip_vars:
            continue
        if re.match(r'^(0x[0-9a-fA-F]+|0b[01]+|[+-]?\d+\.?\d*[fF]?)$', init_value):
            continue
        if 'sizeof' in init_value:
            continue

        # Categorize by operation type
        if re.search(r'[+\-*/%]', init_value) and not re.search(r'[&|^]', init_value):
            result.scalar_arithmetic += 1
        elif re.search(r'[&|^]|<<|>>', init_value):
            result.scalar_bitwise += 1

    # Comparison operations in conditionals
    comparisons = re.findall(r'\bif\s*\([^)]*([<>=!]=?)[^)]*\)', clean_body)
    result.scalar_comparison = len(comparisons)

    # Array element access patterns: arr[i], ptr[idx]
    array_access = re.findall(r'\w+\s*\[\s*\w+\s*\]', clean_body)
    # Filter out common patterns like IMM[x] (immediate lookup tables)
    for access in array_access:
        if not re.match(r'^[A-Z_]+\[', access):  # Skip macro-style lookups
            result.scalar_array_access += 1

    return result


def count_scalar_ops(body: str) -> int:
    """
    Count scalar operations in function body (legacy wrapper).

    For detailed analysis, use analyze_complexity() instead.
    Returns the total count of scalar operations detected.
    """
    complexity = analyze_complexity(body)
    return complexity.total_scalar_ops + complexity.scalar_array_access


def classify_tier(analysis: IntrinsicAnalysis, use_weighted: bool = False) -> Tuple[int, str]:
    """Classify intrinsic into performance tier.

    Uses multi-factor classification considering:
    - NEON operation count (weighted or raw)
    - Loop presence and complexity
    - Scalar operation patterns
    - Memory access patterns (via complexity analysis)
    - Algorithmic indicators (table lookups, reductions)

    Args:
        analysis: The intrinsic analysis result
        use_weighted: If True, use instruction-weighted counts instead of call counts
    """
    # Select count source based on priority:
    # 1. precise_neon_count (from Clang AST) - most accurate call count
    # 2. weighted_neon_count (if use_weighted) - instruction-weighted count
    # 3. neon_count (regex-based) - fallback
    if use_weighted and analysis.weighted_neon_count is not None:
        n = analysis.weighted_neon_count
    elif analysis.precise_neon_count is not None:
        n = analysis.precise_neon_count
    else:
        n = analysis.neon_count
    s = analysis.scalar_count
    cx = analysis.complexity  # Detailed complexity analysis

    # --- T4: Algorithmic Complexity ---
    # Loop-based implementations (always expensive)
    if analysis.has_loop:
        label_parts = ["T4:algorithmic"]
        if cx and cx.loop_count > 1:
            label_parts.append(f"{cx.loop_count}loops")
        if cx and cx.nested_depth > 1:
            label_parts.append(f"depth{cx.nested_depth}")
        if cx and cx.has_data_dependent_loop:
            label_parts.append("data-dep")
        return 4, "+".join(label_parts)

    # Table lookup implementations (algorithmically complex)
    if cx and cx.has_table_lookup:
        return 4, "T4:table-lookup"

    # Pure scalar fallbacks (no NEON at all)
    if s > 0 and n == 0:
        return 4, "T4:scalar"

    # High complexity score from detailed analysis
    if cx and cx.complexity_score >= 50:
        return 4, f"T4:complex(score={cx.complexity_score})"

    # --- Mixed Scalar+NEON ---
    if s > 0 and n > 0:
        # Factor in memory access patterns
        scattered_penalty = 0
        if cx:
            scattered_penalty = cx.scattered_access * 2
            if cx.has_conditional_neon:
                scattered_penalty += 5

        # Effective complexity: NEON ops + weighted scalar ops + memory penalties
        effective_ops = n + (s * 2) + scattered_penalty

        if effective_ops <= 4:
            return 2, f"T2:mixed({n}N+{s}S)"
        elif effective_ops <= 10:
            return 3, f"T3:mixed({n}N+{s}S)"
        else:
            return 4, f"T4:mixed({n}N+{s}S)"

    # --- Pure NEON Implementations ---
    # Apply scattered access penalty for inefficient memory patterns
    access_penalty = 0
    if cx and cx.scattered_access > 2:
        access_penalty = cx.scattered_access - 2  # First 2 are "free"

    adjusted_n = n + access_penalty

    # Reduction operations add latency
    if cx and cx.has_reduction:
        adjusted_n += 2

    if adjusted_n <= 2:
        return 1, "T1:direct"

    if adjusted_n <= 5:
        return 2, "T2:few_ops"

    if adjusted_n <= 10:
        return 3, "T3:moderate"

    return 4, "T4:complex"


def analyze_intrinsic(name: str, body: str, line_start: int, line_end: int) -> IntrinsicAnalysis:
    """Analyze a single intrinsic implementation."""
    neon_count, neon_list = count_neon_intrinsics(body)
    reinterpret_count = count_reinterprets(body)

    # Perform detailed complexity analysis
    complexity = analyze_complexity(body)
    scalar_count = complexity.total_scalar_ops + complexity.scalar_array_access

    # Compute weighted instruction count (more accurate than raw call count)
    weighted_count = sum(get_intrinsic_cost(intr) for intr in neon_list)

    # Use complexity analysis for loop detection (more accurate)
    has_loop = complexity.loop_count > 0

    analysis = IntrinsicAnalysis(
        name=name,
        line_start=line_start,
        line_end=line_end,
        neon_count=neon_count,
        reinterpret_count=reinterpret_count,
        scalar_count=scalar_count,
        has_aarch64_path='SSE2NEON_ARCH_AARCH64' in body or '__aarch64__' in body,
        has_armv7_path='#else' in body and ('SSE2NEON_ARCH_AARCH64' in body or '__aarch64__' in body),
        has_loop=has_loop,
        body=body,
        neon_intrinsics=neon_list,
        weighted_neon_count=weighted_count,
        complexity=complexity,
    )

    analysis.tier, analysis.tier_label = classify_tier(analysis)
    return analysis


def preprocess_with_unifdef(filepath: str) -> Optional[str]:
    """
    Preprocess sse2neon.h with unifdef to get AArch64-specific code paths.

    Returns the preprocessed content as a string, or None if unifdef is not available.
    Note: unifdef may warn about complex macros but still produce usable output.
    """
    unifdef_path = shutil.which('unifdef')
    if not unifdef_path:
        return None

    try:
        # Add -b to process complex expressions as blank (keep both branches)
        # This handles cases like: #if ((__ARM_ARCH == 8) && defined(...))
        cmd = [unifdef_path, '-b'] + UNIFDEF_AARCH64_FLAGS + [filepath]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # unifdef returns 0 if output is unmodified, 1 if modified, 2 on error
        # Even with return code 2, output may be usable (just truncated/partial)
        if result.stdout and len(result.stdout) > 1000:  # Sanity check
            return result.stdout
        else:
            print(f"Warning: unifdef produced insufficient output", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Warning: unifdef error: {e}", file=sys.stderr)
        return None


def parse_sse2neon(filepath: str, use_unifdef: bool = True) -> List[IntrinsicAnalysis]:
    """
    Parse sse2neon.h and analyze all intrinsics.

    Args:
        filepath: Path to sse2neon.h
        use_unifdef: If True, preprocess with unifdef for AArch64-specific analysis
    """
    content = None

    if use_unifdef:
        content = preprocess_with_unifdef(filepath)
        if content:
            print("Using unifdef for AArch64-specific analysis", file=sys.stderr)
        else:
            print("Warning: unifdef not available, analyzing raw file", file=sys.stderr)

    if content is None:
        with open(filepath, 'r') as f:
            content = f.read()

    lines = content.split('\n')
    results = []
    seen_intrinsics = set()  # Track already-processed intrinsics

    # Match FORCE_INLINE function definitions (implementations only, not forward declarations)
    # Forward declarations end with ';' while implementations have '{'
    pattern = re.compile(
        r'^FORCE_INLINE\s+\S+\s+(_mm\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        name = match.group(1)

        # Skip duplicates (some intrinsics may have multiple #ifdef paths)
        if name in seen_intrinsics:
            continue
        seen_intrinsics.add(name)

        # Find line number (note: line numbers are from preprocessed content)
        line_start = content[:match.start()].count('\n')
        body, line_end = extract_intrinsic_body(lines, line_start)

        analysis = analyze_intrinsic(name, body, line_start + 1, line_end + 1)
        results.append(analysis)

    return results


def generate_summary(results: List[IntrinsicAnalysis]) -> Dict:
    """Generate summary statistics."""
    tier_counts = defaultdict(int)
    tier_intrinsics = defaultdict(list)

    for r in results:
        tier_counts[r.tier] += 1
        tier_intrinsics[r.tier].append(r.name)

    total = len(results)
    total_neon = sum(r.neon_count for r in results)
    total_reinterpret = sum(r.reinterpret_count for r in results)
    total_scalar = sum(r.scalar_count for r in results)

    # Count mixed implementations (scalar + NEON)
    mixed_count = sum(1 for r in results if r.scalar_count > 0 and r.neon_count > 0)
    pure_scalar_count = sum(1 for r in results if r.scalar_count > 0 and r.neon_count == 0)

    return {
        'total_intrinsics': total,
        'total_neon_calls': total_neon,
        'total_scalar_ops': total_scalar,
        'total_reinterprets': total_reinterpret,
        'avg_neon_per_intrinsic': round(total_neon / total, 2) if total else 0,
        'avg_scalar_per_intrinsic': round(total_scalar / total, 2) if total else 0,
        'mixed_implementations': mixed_count,
        'pure_scalar_implementations': pure_scalar_count,
        'tier_distribution': {
            f'T{i}': {
                'count': tier_counts[i],
                'percentage': round(100 * tier_counts[i] / total, 1) if total else 0,
            }
            for i in range(1, 5)
        },
        'tier_1_examples': tier_intrinsics[1][:10],
        'tier_4_examples': tier_intrinsics[4][:10],
    }


def print_markdown_report(results: List[IntrinsicAnalysis], summary: Dict):
    """Print analysis as markdown report."""
    print("# SSE2NEON Intrinsic Tier Analysis\n")
    print("## Summary\n")
    print(f"- **Total Intrinsics**: {summary['total_intrinsics']}")
    print(f"- **Total NEON Calls**: {summary['total_neon_calls']}")
    print(f"- **Total Scalar Ops**: {summary['total_scalar_ops']}")
    print(f"- **Total vreinterpret Casts**: {summary['total_reinterprets']}")
    print(f"- **Avg NEON/Intrinsic**: {summary['avg_neon_per_intrinsic']}")
    print(f"- **Avg Scalar/Intrinsic**: {summary['avg_scalar_per_intrinsic']}")
    print(f"- **Mixed (NEON+Scalar)**: {summary['mixed_implementations']}")
    print(f"- **Pure Scalar**: {summary['pure_scalar_implementations']}\n")

    print("## Tier Distribution\n")
    print("| Tier | Description | Count | % |")
    print("|------|-------------|-------|---|")
    tier_desc = {
        'T1': 'Direct mapping (1-2 ops)',
        'T2': 'Few operations (3-5 ops)',
        'T3': 'Moderate emulation (6-10 ops)',
        'T4': 'Complex/algorithmic (10+ ops)',
    }
    for tier, data in summary['tier_distribution'].items():
        print(f"| {tier} | {tier_desc[tier]} | {data['count']} | {data['percentage']}% |")

    print("\n## Tier Classification Details\n")

    # Group by tier
    by_tier = defaultdict(list)
    for r in results:
        by_tier[r.tier].append(r)

    for tier in range(1, 5):
        intrinsics = by_tier[tier]
        if not intrinsics:
            continue

        print(f"### Tier {tier} ({len(intrinsics)} intrinsics)\n")

        # Show examples
        examples = sorted(intrinsics, key=lambda x: (x.neon_count, x.scalar_count))[:20]
        print("| Intrinsic | NEON | Scalar | Line | Notes |")
        print("|-----------|------|--------|------|-------|")
        for r in examples:
            notes = []
            if r.has_aarch64_path:
                notes.append("AArch64")
            if r.has_armv7_path:
                notes.append("ARMv7")
            if r.has_loop:
                notes.append("loop")
            if r.scalar_count > 0 and r.neon_count > 0:
                notes.append("mixed")
            print(f"| `{r.name}` | {r.neon_count} | {r.scalar_count} | {r.line_start} | {', '.join(notes)} |")
        print()


def print_json_report(results: List[IntrinsicAnalysis], summary: Dict):
    """Print analysis as JSON."""
    output = {
        'summary': summary,
        'intrinsics': [
            {
                'name': r.name,
                'tier': r.tier,
                'tier_label': r.tier_label,
                'neon_count': r.neon_count,
                'precise_neon_count': r.precise_neon_count,
                'scalar_count': r.scalar_count,
                'reinterpret_count': r.reinterpret_count,
                'line_start': r.line_start,
                'line_end': r.line_end,
                'has_aarch64_path': r.has_aarch64_path,
                'has_armv7_path': r.has_armv7_path,
                'has_loop': r.has_loop,
            }
            for r in results
        ]
    }
    print(json.dumps(output, indent=2))


def print_verbose_report(results: List[IntrinsicAnalysis], show_confidence: bool = False):
    """Print detailed verbose report.

    Args:
        results: List of intrinsic analyses
        show_confidence: If True, compute and display confidence metrics
    """
    for r in sorted(results, key=lambda x: (-x.tier, -x.neon_count, -x.scalar_count)):
        print(f"\n{'='*60}")
        print(f"Intrinsic: {r.name}")
        print(f"Tier: T{r.tier} ({r.tier_label})")
        print(f"Lines: {r.line_start}-{r.line_end}")
        neon_info = f"NEON ops: {r.neon_count}"
        if r.precise_neon_count is not None:
            neon_info += f" (precise: {r.precise_neon_count})"
        if r.weighted_neon_count is not None and r.weighted_neon_count != r.neon_count:
            neon_info += f" (weighted: {r.weighted_neon_count})"
        print(f"{neon_info}, Scalar ops: {r.scalar_count}, vreinterpret: {r.reinterpret_count}")
        if r.neon_intrinsics:
            unique_intrinsics = list(set(r.neon_intrinsics))[:10]
            print(f"NEON intrinsics: {', '.join(unique_intrinsics)}")

        # Show confidence metrics if requested
        if show_confidence:
            confidence = compute_confidence(r, has_asm=False)
            conf_symbol = {'high': '✓', 'medium': '~', 'low': '?', 'very-low': '✗'}
            print(f"Confidence: {confidence.overall_confidence:.0%} ({confidence.confidence_level}) "
                  f"{conf_symbol.get(confidence.confidence_level, '?')}")
            factors = []
            if confidence.has_ast_analysis:
                factors.append("+AST")
            if confidence.matches_reference:
                factors.append("+ref")
            if confidence.has_complex_conditionals:
                factors.append("-cond")
            if confidence.has_platform_variants:
                factors.append("-plat")
            if confidence.near_tier_boundary:
                factors.append("-boundary")
            if factors:
                print(f"  Factors: {' '.join(factors)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Analyze SSE2NEON intrinsic tiers (AArch64-focused)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
By default, this tool uses unifdef to preprocess sse2neon.h for AArch64,
removing ARMv7 fallback paths that would inflate instruction counts.

Examples:
  python3 scripts/analyze-tiers.py              # AArch64 analysis (default)
  python3 scripts/analyze-tiers.py --no-unifdef # Raw file analysis
  python3 scripts/analyze-tiers.py --tier 4     # Show T4 (complex) intrinsics
  python3 scripts/analyze-tiers.py --clang-ast  # Use Clang AST for T2+ refinement
""")
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--markdown', action='store_true', help='Output as Markdown')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--tier', type=int, help='Filter by tier (1-4)')
    parser.add_argument('--no-unifdef', action='store_true',
                        help='Analyze raw file without unifdef preprocessing')
    parser.add_argument('--clang-ast', action='store_true',
                        help='Use Clang AST for precise T2+ analysis (requires libclang)')
    parser.add_argument('--ast-tiers', type=str, default='2,3,4',
                        help='Comma-separated tiers to refine with AST (default: 2,3,4)')
    parser.add_argument('--asm', '--assembly', action='store_true', dest='assembly',
                        help='Use assembly generation for T4 analysis (most accurate, slowest)')
    parser.add_argument('--asm-tiers', type=str, default='4',
                        help='Comma-separated tiers to analyze with assembly (default: 4)')
    parser.add_argument('--asm-compiler', type=str, default=None,
                        help='Compiler for assembly generation (default: auto-detect)')
    parser.add_argument('--weighted', action='store_true',
                        help='Use weighted instruction counts for tier classification')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel workers for AST/assembly analysis (default: 1)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear the analysis result cache and exit')
    parser.add_argument('--validate', action='store_true',
                        help='Validate results against reference data and report accuracy')
    parser.add_argument('file', nargs='?', default='sse2neon.h', help='Path to sse2neon.h')
    args = parser.parse_args()

    # Handle --clear-cache
    if args.clear_cache:
        cache = ResultCache()
        count = cache.clear()
        print(f"Cleared {count} cached results from {cache.cache_dir}", file=sys.stderr)
        sys.exit(0)

    # Find sse2neon.h
    filepath = Path(args.file)
    if not filepath.exists():
        # Try relative to script
        script_dir = Path(__file__).parent.parent
        filepath = script_dir / 'sse2neon.h'

    if not filepath.exists():
        print(f"Error: Cannot find sse2neon.h at {filepath}", file=sys.stderr)
        sys.exit(1)

    use_unifdef = not args.no_unifdef
    results = parse_sse2neon(str(filepath), use_unifdef=use_unifdef)

    # Apply Clang AST refinement if requested
    if args.clang_ast:
        if not CLANG_AVAILABLE:
            print("Error: --clang-ast requires libclang. Install with: pip install clang",
                  file=sys.stderr)
            sys.exit(1)
        tiers_to_refine = [int(t.strip()) for t in args.ast_tiers.split(',')]
        results = refine_with_clang_ast(results, str(filepath), tiers_to_refine,
                                        parallel_jobs=args.jobs)

    # Level 3: Assembly analysis (most accurate, slowest)
    if args.assembly:
        asm_tiers = [int(t.strip()) for t in args.asm_tiers.split(',')]
        results = refine_with_assembly(results, str(filepath), asm_tiers,
                                       parallel_jobs=args.jobs)

    # Reclassify with weighted instruction counts if requested
    if args.weighted:
        print("Using weighted instruction counts for classification...", file=sys.stderr)
        for analysis in results:
            analysis.tier, analysis.tier_label = classify_tier(analysis, use_weighted=True)

    # Validation against reference data
    if args.validate:
        validations = validate_against_reference(results)
        print_validation_report(validations)

    if args.tier:
        results = [r for r in results if r.tier == args.tier]

    summary = generate_summary(results)

    if args.json:
        print_json_report(results, summary)
    elif args.verbose:
        print_verbose_report(results, show_confidence=args.validate)
    else:
        print_markdown_report(results, summary)


if __name__ == '__main__':
    main()
