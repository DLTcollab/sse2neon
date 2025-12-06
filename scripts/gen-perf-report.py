#!/usr/bin/env python3
"""
Generate performance documentation for SSE2NEON.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HEADER = """# SSE2NEON Performance Tier Analysis (AArch64)

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
"""


def main():
    parser = argparse.ArgumentParser(description='Generate performance documentation')
    parser.add_argument('--clang-ast', action='store_true',
                        help='Use Clang AST for precise T2+ analysis')
    parser.add_argument('--asm', '--assembly', action='store_true', dest='assembly',
                        help='Use assembly generation for T4 analysis (most accurate)')
    parser.add_argument('--weighted', action='store_true',
                        help='Use weighted instruction counts (ARM Cortex-A72 costs)')
    args = parser.parse_args()

    # Run analyze_tiers.py and get JSON output
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / 'analyze-tiers.py'), '--json']
    if args.clang_ast:
        cmd.append('--clang-ast')
    if args.assembly:
        cmd.append('--asm')
    if args.weighted:
        cmd.append('--weighted')
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result.stdout)
    # Use all intrinsics - analyze_tiers.py already filters to _mm* functions
    intrinsics = data['intrinsics']

    # Print header with regeneration instructions
    print(HEADER)
    print('## SSE2NEON Performance Classification (AArch64)')
    print()
    print('Based on static analysis of NEON instruction counts per SSE intrinsic.')
    print('Cycle estimates based on ARM Cortex-A72 (ARMv8-A) Software Optimization Guide.')
    print()

    # Summary table
    print('### Summary')
    print()
    print('| Metric | Value |')
    print('|--------|-------|')
    print(f'| Total SSE Intrinsics | {len(intrinsics)} |')

    t1_count = sum(1 for i in intrinsics if i['tier'] == 1)
    t23_count = sum(1 for i in intrinsics if i['tier'] in [2, 3])
    t4_count = sum(1 for i in intrinsics if i['tier'] == 4)

    print(f'| Direct Mappings (T1) | {t1_count} ({100*t1_count/len(intrinsics):.1f}%) |')
    print(f'| Moderate Emulation (T2-T3) | {t23_count} ({100*t23_count/len(intrinsics):.1f}%) |')
    print(f'| Complex Emulation (T4) | {t4_count} ({100*t4_count/len(intrinsics):.1f}%) |')

    avg_neon = sum(i['neon_count'] for i in intrinsics) / len(intrinsics)
    print(f'| Avg NEON Ops/Intrinsic | {avg_neon:.2f} |')
    print()

    # Tier explanation
    print('### Performance Tiers')
    print()
    print('| Tier | NEON Ops | Estimated Cycles | Description |')
    print('|------|----------|------------------|-------------|')
    print('| T1 | 1-2 | 1-3 | Direct NEON mapping, near-native performance |')
    print('| T2 | 3-5 | 4-8 | Few NEON operations, slight overhead |')
    print('| T3 | 6-10 | 8-15 | Moderate emulation, noticeable overhead |')
    print('| T4 | 10+ or special | 15-50+ | Complex/algorithmic, significant overhead |')
    print()
    print('> **Note**: T4 classification uses multi-factor analysis beyond raw NEON counts:')
    print('> - Table lookups (e.g., `_mm_shuffle_epi8`) are T4 due to algorithmic complexity')
    print('> - Pure scalar fallbacks (e.g., `_mm_crc32_u16` without HW CRC) are T4')
    print('> - Loop-based implementations are T4 regardless of instruction count')
    print('> - Mixed scalar+NEON with high effective cost may be promoted to T4')
    print()

    # Hot intrinsics to watch
    print('### Performance-Critical Intrinsics (T4)')
    print()
    print('These intrinsics have significant emulation overhead. Consider alternative')
    print('algorithms when porting performance-critical code.')
    print()
    print('| Intrinsic | NEON Ops | Notes |')
    print('|-----------|----------|-------|')

    t4 = sorted([i for i in intrinsics if i['tier'] == 4], key=lambda x: -x['neon_count'])
    for item in t4[:15]:
        name = item['name']
        neon = item['neon_count']
        notes = []
        if 'aes' in name.lower():
            notes.append('Use HW crypto when available')
        elif 'mpsadbw' in name:
            notes.append('SAD computation, very expensive')
        elif 'round' in name:
            notes.append('Rounding modes emulation')
        elif 'rsqrt' in name or 'sqrt' in name:
            notes.append('Newton-Raphson refinement')
        elif 'crc' in name:
            notes.append('Use HW CRC when available')
        elif 'clmul' in name:
            notes.append('Carryless multiply')
        elif 'madd' in name:
            notes.append('Multiply-add with widening')
        elif 'minpos' in name:
            notes.append('Horizontal minimum search')
        print(f'| `{name}` | {neon} | {" ".join(notes)} |')

    print()
    print('### Efficient Intrinsics (T1 - Single NEON Instruction)')
    print()
    print('These intrinsics map directly to single NEON instructions:')
    print()

    t1_one = sorted([i for i in intrinsics if i['tier'] == 1 and i['neon_count'] == 1],
                    key=lambda x: x['name'])

    # Group by category
    categories = {
        'Arithmetic': ['add', 'sub', 'mul', 'div', 'neg', 'abs'],
        'Comparison': ['cmpeq', 'cmpgt', 'cmplt', 'test'],
        'Logical': ['and', 'or', 'xor', 'not'],
        'Load/Store': ['load', 'store', 'set', 'get'],
        'Conversion': ['cvt', 'cast'],
        'Math': ['sqrt', 'ceil', 'floor', 'round', 'min', 'max'],
    }

    for cat, keywords in categories.items():
        items = [i['name'] for i in t1_one if any(k in i['name'].lower() for k in keywords)]
        if items:
            display = ', '.join(f'`{n}`' for n in items[:8])
            if len(items) > 8:
                display += f', ... (+{len(items)-8} more)'
            print(f'- {cat}: {display}')

    # Full tier breakdown
    print()
    print('### Complete Tier Classification')
    print()
    print('<details>')
    print('<summary>Click to expand full list</summary>')
    print()

    for tier in range(1, 5):
        tier_items = sorted([i for i in intrinsics if i['tier'] == tier], key=lambda x: x['name'])
        print(f'#### Tier {tier} ({len(tier_items)} intrinsics)')
        print()
        if tier_items:
            # Show as compact list
            names = [f'`{i["name"]}`' for i in tier_items]
            # Wrap at ~80 chars
            line = ''
            for name in names:
                if len(line) + len(name) > 75:
                    print(line)
                    line = name
                else:
                    line = line + ', ' + name if line else name
            if line:
                print(line)
        print()

    print('</details>')


if __name__ == '__main__':
    main()
