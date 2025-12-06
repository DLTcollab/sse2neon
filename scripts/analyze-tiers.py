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
"""

import re
import sys
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path

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


@dataclass
class IntrinsicAnalysis:
    """Analysis result for a single intrinsic."""
    name: str
    line_start: int
    line_end: int
    neon_count: int
    reinterpret_count: int
    has_aarch64_path: bool
    has_armv7_path: bool
    has_loop: bool
    has_scalar_fallback: bool
    body: str = ""
    tier: int = 0
    tier_label: str = ""
    neon_intrinsics: List[str] = field(default_factory=list)


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


def count_neon_intrinsics(body: str) -> Tuple[int, List[str]]:
    """Count actual NEON intrinsics in function body."""
    found = []

    # Remove comments
    body = re.sub(r'//.*$', '', body, flags=re.MULTILINE)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)

    for pattern in NEON_PATTERNS:
        matches = pattern.findall(body)
        found.extend(matches)

    return len(found), found


def count_reinterprets(body: str) -> int:
    """Count vreinterpret casts (don't generate instructions)."""
    return len(REINTERPRET_PATTERN.findall(body))


def classify_tier(analysis: IntrinsicAnalysis) -> Tuple[int, str]:
    """Classify intrinsic into performance tier."""
    n = analysis.neon_count

    # Algorithmic/loop-based implementations
    if analysis.has_loop:
        return 4, "T4:algorithmic"

    # Scalar fallbacks are typically slow
    if analysis.has_scalar_fallback and n == 0:
        return 4, "T4:scalar"

    # Direct mappings
    if n <= 2:
        return 1, "T1:direct"

    # Few operations
    if n <= 5:
        return 2, "T2:few_ops"

    # Moderate emulation
    if n <= 10:
        return 3, "T3:moderate"

    # Complex emulation
    return 4, "T4:complex"


def analyze_intrinsic(name: str, body: str, line_start: int, line_end: int) -> IntrinsicAnalysis:
    """Analyze a single intrinsic implementation."""
    neon_count, neon_list = count_neon_intrinsics(body)
    reinterpret_count = count_reinterprets(body)

    # Detect loops: look for 'for (' or 'while (' to avoid false matches
    has_loop = bool(re.search(r'\b(for|while)\s*\(', body))

    # Heuristic to detect scalar fallback implementations:
    # Look for scalar integer type declarations with assignments that suggest
    # element-by-element processing (e.g., "uint8_t val = ...").
    # This may have false positives for loop counters, but effectively identifies
    # non-vectorized paths in most cases.
    has_scalar_fallback = bool(re.search(
        r'\b(u?int(8|16|32|64)_t)\s+\w+\s*=\s*[^;]+;', body
    )) and neon_count == 0

    analysis = IntrinsicAnalysis(
        name=name,
        line_start=line_start,
        line_end=line_end,
        neon_count=neon_count,
        reinterpret_count=reinterpret_count,
        has_aarch64_path='SSE2NEON_ARCH_AARCH64' in body or '__aarch64__' in body,
        has_armv7_path='#else' in body and ('SSE2NEON_ARCH_AARCH64' in body or '__aarch64__' in body),
        has_loop=has_loop,
        has_scalar_fallback=has_scalar_fallback,
        body=body,
        neon_intrinsics=neon_list,
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

    return {
        'total_intrinsics': total,
        'total_neon_calls': total_neon,
        'total_reinterprets': total_reinterpret,
        'avg_neon_per_intrinsic': round(total_neon / total, 2) if total else 0,
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
    print(f"- **Total vreinterpret Casts**: {summary['total_reinterprets']}")
    print(f"- **Avg NEON/Intrinsic**: {summary['avg_neon_per_intrinsic']}\n")

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
        examples = sorted(intrinsics, key=lambda x: x.neon_count)[:20]
        print("| Intrinsic | NEON Ops | Line | Notes |")
        print("|-----------|----------|------|-------|")
        for r in examples:
            notes = []
            if r.has_aarch64_path:
                notes.append("AArch64")
            if r.has_armv7_path:
                notes.append("ARMv7")
            if r.has_loop:
                notes.append("loop")
            print(f"| `{r.name}` | {r.neon_count} | {r.line_start} | {', '.join(notes)} |")
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


def print_verbose_report(results: List[IntrinsicAnalysis]):
    """Print detailed verbose report."""
    for r in sorted(results, key=lambda x: (-x.tier, -x.neon_count)):
        print(f"\n{'='*60}")
        print(f"Intrinsic: {r.name}")
        print(f"Tier: T{r.tier} ({r.tier_label})")
        print(f"Lines: {r.line_start}-{r.line_end}")
        print(f"NEON ops: {r.neon_count}, vreinterpret: {r.reinterpret_count}")
        if r.neon_intrinsics:
            print(f"NEON intrinsics: {', '.join(set(r.neon_intrinsics)[:10])}")


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
""")
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--markdown', action='store_true', help='Output as Markdown')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--tier', type=int, help='Filter by tier (1-4)')
    parser.add_argument('--no-unifdef', action='store_true',
                        help='Analyze raw file without unifdef preprocessing')
    parser.add_argument('file', nargs='?', default='sse2neon.h', help='Path to sse2neon.h')
    args = parser.parse_args()

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

    if args.tier:
        results = [r for r in results if r.tier == args.tier]

    summary = generate_summary(results)

    if args.json:
        print_json_report(results, summary)
    elif args.verbose:
        print_verbose_report(results)
    else:
        print_markdown_report(results, summary)


if __name__ == '__main__':
    main()
