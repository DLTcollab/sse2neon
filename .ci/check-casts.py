#!/usr/bin/env python3
"""
Validate type cast usage in sse2neon.h for C/C++ compatibility, optimization,
and strict aliasing compliance.

This script enforces PR #671's cast conventions:
1. Use _sse2neon_*_cast macros instead of raw C++ casts
2. Detect old C-style casts that should use the wrapper macros
3. Identify potential strict aliasing violations
4. Ensure cast patterns are compiler-optimization friendly

Strict Aliasing Rule (C99 6.5/7, C++11 [basic.lval]):
  Accessing an object through a pointer of incompatible type is undefined
  behavior. The compiler assumes pointers of different types don't alias,
  enabling aggressive optimizations. Violations can cause:
  - Incorrect code generation at -O2/-O3
  - Values not being reloaded from memory
  - Unpredictable behavior across compiler versions

Safe Type Punning Patterns:
  1. Use union (C99 guaranteed, C++ implementation-defined but widely supported)
  2. Use memcpy (always safe, compiler optimizes to single instruction)
  3. Use char* (char can alias anything per standard)
  4. Use __attribute__((may_alias)) on GCC/Clang
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict

# Strict aliasing compatible type groups
# Types within the same group can safely alias each other
COMPATIBLE_TYPE_GROUPS = {
    # Signed/unsigned pairs of same size are compatible
    'int8': {'int8_t', 'uint8_t', 'char', 'signed char', 'unsigned char'},
    'int16': {'int16_t', 'uint16_t', 'short', 'unsigned short'},
    'int32': {'int32_t', 'uint32_t', 'int', 'unsigned int', 'unsigned'},
    'int64': {'int64_t', 'uint64_t', 'long', 'unsigned long', 'long long', 'unsigned long long'},
    # Floating point (distinct from integers - aliasing violation!)
    'float32': {'float', 'float32_t'},
    'float64': {'double', 'float64_t'},
    # char* can alias anything (special case in standard)
    'char_ptr': {'char *', 'unsigned char *', 'signed char *', 'uint8_t *', 'int8_t *'},
    # void* is a universal pointer type
    'void_ptr': {'void *'},
}

# Known strict aliasing violations: source_type -> incompatible target types
ALIASING_VIOLATIONS = {
    # Integer pointer to float pointer - VIOLATION
    ('int32_t *', 'float *'),
    ('uint32_t *', 'float *'),
    ('int *', 'float *'),
    ('unsigned *', 'float *'),
    ('int32_t *', 'float32_t *'),
    ('uint32_t *', 'float32_t *'),
    # Float pointer to integer pointer - VIOLATION
    ('float *', 'int32_t *'),
    ('float *', 'uint32_t *'),
    ('float *', 'int *'),
    ('float *', 'unsigned *'),
    ('float32_t *', 'int32_t *'),
    ('float32_t *', 'uint32_t *'),
    # 64-bit integer to double - VIOLATION
    ('int64_t *', 'double *'),
    ('uint64_t *', 'double *'),
    ('double *', 'int64_t *'),
    ('double *', 'uint64_t *'),
    # SIMD vector types to scalar pointers - potential issues
    ('__m128 *', 'float *'),
    ('__m128i *', 'int *'),
    ('__m128i *', 'int32_t *'),
    ('__m128d *', 'double *'),
}

# Cast macro guidelines for contributors
CAST_GUIDELINES = """
Cast Macro Guidelines for sse2neon Contributors
================================================

sse2neon uses wrapper macros for type casts to maintain compatibility with
both C and C++ compilers while ensuring strict aliasing compliance.

Why Use These Macros?
---------------------
1. C/C++ Compatibility: Raw C++ casts break C compilation
2. Compiler Optimization: C++ casts provide type information for better codegen
3. Strict Aliasing: Proper casts help compiler's alias analysis
4. Code Safety: Explicit cast types prevent accidental unsafe conversions

Strict Aliasing Rule
--------------------
The compiler assumes pointers of different types don't point to the same
memory (with exceptions for char* and signed/unsigned variants). This enables
optimizations like:
  - Keeping values in registers instead of reloading from memory
  - Reordering memory operations
  - Vectorization and loop optimizations

VIOLATION EXAMPLE (undefined behavior at -O2):
  int32_t i = 0x3f800000;
  float f = *(float *)&i;  // BAD: int32_t* -> float* violates aliasing

SAFE ALTERNATIVES:
  1. Union (preferred for sse2neon):
     union { int32_t i; float f; } u;
     u.i = 0x3f800000;
     float f = u.f;

  2. memcpy (always safe, compiler optimizes away):
     int32_t i = 0x3f800000;
     float f;
     memcpy(&f, &i, sizeof(f));

  3. Use project's recast functions:
     float f = sse2neon_recast_u32_f32(0x3f800000);

Cast Macros
-----------
  1. _sse2neon_static_cast(type, expr)
     - Use for: Numeric conversions (int<->float value conversion)
     - Safe: Compiler generates proper conversion code
     - Example: _sse2neon_static_cast(int32_t, float_value)

  2. _sse2neon_reinterpret_cast(type, expr)
     - Use for: Pointer type conversions, NEON vector reinterpretation
     - WARNING: May violate strict aliasing if misused!
     - Safe uses:
       * void* <-> typed pointer
       * char*/uint8_t* <-> any pointer (char can alias anything)
       * Same-size signed/unsigned pointer conversion
       * NEON vreinterpret_* intrinsics (compiler-blessed)
     - Example: _sse2neon_reinterpret_cast(uint8_t *, void_ptr)

  3. _sse2neon_const_cast(type, expr)
     - Use for: Removing const qualifier
     - Example: _sse2neon_const_cast(int *, const_int_ptr)

FORBIDDEN:
  - Raw C++ casts: static_cast<>, reinterpret_cast<>, const_cast<>
  - Direct int* <-> float* casts (strict aliasing violation)
  - Old C-style pointer casts between incompatible types

Quick Reference - Safe Type Punning:
  +--------------------------------+----------------------------------+
  | Pattern                        | Safety                           |
  +--------------------------------+----------------------------------+
  | (float *)&int_var              | UNSAFE - aliasing violation      |
  | union { int i; float f; }      | SAFE - use this                  |
  | memcpy(&f, &i, sizeof(f))      | SAFE - compiler optimizes        |
  | vreinterpret_f32_s32(v)        | SAFE - NEON blessed              |
  | (uint8_t *)ptr                 | SAFE - char* can alias anything  |
  | (int32_t *)(void *)ptr         | Technically OK via void*         |
  +--------------------------------+----------------------------------+

For more details, see: https://github.com/DLTcollab/sse2neon/pull/671
"""

# Types commonly used in sse2neon
KNOWN_TYPES = {
    'int', 'unsigned', 'long', 'short', 'char',
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'float', 'double',
    'float16_t', 'float32_t', 'float64_t',
    'int8x8_t', 'int8x16_t', 'int16x4_t', 'int16x8_t',
    'int32x2_t', 'int32x4_t', 'int64x1_t', 'int64x2_t',
    'uint8x8_t', 'uint8x16_t', 'uint16x4_t', 'uint16x8_t',
    'uint32x2_t', 'uint32x4_t', 'uint64x1_t', 'uint64x2_t',
    'float32x2_t', 'float32x4_t', 'float64x1_t', 'float64x2_t',
    'poly8x8_t', 'poly8x16_t', 'poly16x4_t', 'poly16x8_t',
    'poly64x1_t', 'poly64x2_t', 'poly128_t',
    '__m128', '__m128i', '__m128d', '__m64',
    'SIMDVec',
}


@dataclass
class CastIssue:
    """Represents a potential cast issue found in the code."""
    line_num: int
    line: str
    issue_type: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    fix_hint: str = ''
    optimization_note: str = ''


@dataclass
class CastStats:
    """Statistics about cast usage in the file."""
    static_cast_count: int = 0
    reinterpret_cast_count: int = 0
    const_cast_count: int = 0
    c_style_pointer_cast_count: int = 0
    aliasing_safe_count: int = 0
    aliasing_unsafe_count: int = 0
    static_cast_uses: List[Tuple[int, str]] = field(default_factory=list)
    reinterpret_cast_uses: List[Tuple[int, str]] = field(default_factory=list)
    const_cast_uses: List[Tuple[int, str]] = field(default_factory=list)


def normalize_type(t: str) -> str:
    """Normalize type string for comparison."""
    t = t.strip()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\s*\*\s*', ' *', t)
    return t.strip()


def get_base_type(ptr_type: str) -> str:
    """Extract base type from pointer type."""
    return ptr_type.replace('*', '').replace('const', '').strip()


def is_char_pointer(t: str) -> bool:
    """Check if type is a char pointer (can alias anything)."""
    base = get_base_type(t).lower()
    return base in {'char', 'unsigned char', 'signed char', 'uint8_t', 'int8_t'}


def is_void_pointer(t: str) -> bool:
    """Check if type is void pointer."""
    return 'void' in t and '*' in t


def types_can_alias(type1: str, type2: str) -> bool:
    """
    Check if two pointer types can legally alias each other.
    Returns True if aliasing is safe, False if it's a violation.
    """
    t1 = normalize_type(type1)
    t2 = normalize_type(type2)

    # Same type always OK
    if t1 == t2:
        return True

    # char* can alias anything
    if is_char_pointer(t1) or is_char_pointer(t2):
        return True

    # void* can alias anything
    if is_void_pointer(t1) or is_void_pointer(t2):
        return True

    # Check if types are in the same compatibility group
    base1 = get_base_type(t1)
    base2 = get_base_type(t2)

    for group_types in COMPATIBLE_TYPE_GROUPS.values():
        if base1 in group_types and base2 in group_types:
            return True

    # Check known violations
    if (t1, t2) in ALIASING_VIOLATIONS or (t2, t1) in ALIASING_VIOLATIONS:
        return False

    # Integer <-> float is always a violation
    int_types = {'int', 'unsigned', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t',
                 'short', 'long', 'int16_t', 'uint16_t'}
    float_types = {'float', 'double', 'float32_t', 'float64_t'}

    if (base1 in int_types and base2 in float_types) or \
       (base1 in float_types and base2 in int_types):
        return False

    # Default: assume compatible (may need refinement)
    return True


def find_raw_cpp_casts(content: str) -> List[CastIssue]:
    """Find raw C++ casts not using _sse2neon_* macros."""
    issues = []
    lines = content.split('\n')
    macro_def_pattern = re.compile(r'#define\s+_sse2neon_(static|reinterpret|const)_cast')

    cast_patterns = [
        (r'\bstatic_cast\s*<\s*([^>]+)\s*>\s*\(([^)]+)\)',
         'raw_static_cast', '_sse2neon_static_cast'),
        (r'\breinterpret_cast\s*<\s*([^>]+)\s*>\s*\(([^)]+)\)',
         'raw_reinterpret_cast', '_sse2neon_reinterpret_cast'),
        (r'\bconst_cast\s*<\s*([^>]+)\s*>\s*\(([^)]+)\)',
         'raw_const_cast', '_sse2neon_const_cast'),
    ]

    for i, line in enumerate(lines, 1):
        if macro_def_pattern.search(line):
            continue
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        for pattern, issue_type, replacement in cast_patterns:
            match = re.search(pattern, line)
            if match:
                target_type = match.group(1).strip()
                expr = match.group(2).strip()
                issues.append(CastIssue(
                    line_num=i,
                    line=stripped,
                    issue_type=issue_type,
                    description='Raw C++ cast breaks C compilation',
                    severity='error',
                    fix_hint=f'Replace with: {replacement}({target_type}, {expr})',
                    optimization_note='Use wrapper macro for C/C++ compatibility'
                ))
            elif re.search(pattern.split(r'\s*\(')[0], line):
                cast_name = issue_type.replace('raw_', '')
                issues.append(CastIssue(
                    line_num=i,
                    line=stripped,
                    issue_type=issue_type,
                    description=f'Raw {cast_name} breaks C compilation',
                    severity='error',
                    fix_hint=f'Replace with: {replacement}(type, expr)'
                ))

    return issues


def find_c_style_casts(content: str) -> Tuple[List[CastIssue], List[Tuple[int, str, str]]]:
    """Find old C-style casts that should use _sse2neon_* macros."""
    issues = []
    found_casts = []
    lines = content.split('\n')

    pointer_cast_pattern = re.compile(
        r'\(\s*'
        r'((?:const\s+)?'
        r'(?:unsigned\s+|signed\s+)?'
        r'(?:' + '|'.join(re.escape(t) for t in KNOWN_TYPES) + r')'
        r'\s*\*+)\s*\)'
        r'\s*'
        r'([a-zA-Z_&][a-zA-Z0-9_.\[\]>-]*)'
    )

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue
        if stripped.startswith('#'):
            continue
        if '_sse2neon_' in line:
            continue

        for match in pointer_cast_pattern.finditer(line):
            cast_type = match.group(1).strip()
            expr = match.group(2).strip()
            found_casts.append((i, cast_type, expr))

            issues.append(CastIssue(
                line_num=i,
                line=stripped if len(stripped) <= 80 else stripped[:77] + '...',
                issue_type='c_style_pointer_cast',
                description=f'C-style pointer cast ({cast_type})',
                severity='warning',
                fix_hint=f'Use: _sse2neon_reinterpret_cast({cast_type}, {expr})',
                optimization_note='Explicit cast helps strict aliasing analysis'
            ))

    return issues, found_casts


def find_strict_aliasing_violations(content: str) -> List[CastIssue]:
    """
    Detect potential strict aliasing violations.

    Look for patterns like:
    - (float *)&int_var
    - (int32_t *)float_ptr
    - *(float *)&int_var (type punning through pointer)
    """
    issues = []
    lines = content.split('\n')

    # Pattern: *(type *)&var - dereference of cast address
    type_pun_pattern = re.compile(
        r'\*\s*\(\s*'
        r'((?:const\s+)?'
        r'(?:unsigned\s+|signed\s+)?'
        r'(?:int|float|double|int32_t|uint32_t|int64_t|uint64_t|float32_t|float64_t)'
        r'\s*\*)\s*\)'
        r'\s*&\s*'
        r'([a-zA-Z_][a-zA-Z0-9_]*)'
    )

    # Pattern: (type *)&var - address cast (potential aliasing setup)
    addr_cast_pattern = re.compile(
        r'\(\s*'
        r'((?:const\s+)?'
        r'(?:unsigned\s+|signed\s+)?'
        r'(?:int|float|double|int32_t|uint32_t|int64_t|uint64_t|float32_t|float64_t)'
        r'\s*\*)\s*\)'
        r'\s*&\s*'
        r'([a-zA-Z_][a-zA-Z0-9_]*)'
    )

    # Pattern for detecting source type from variable declarations
    var_types: Dict[str, str] = {}

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue
        if stripped.startswith('#'):
            continue

        # Skip lines using safe patterns
        if 'memcpy' in line or 'union' in line.lower():
            continue
        if 'vreinterpret' in line:  # NEON reinterpret is safe
            continue
        if '_sse2neon_' in line:
            continue

        # Track variable declarations (simplified)
        decl_match = re.match(
            r'(int|float|double|int32_t|uint32_t|int64_t|uint64_t|float32_t|float64_t)\s+'
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[=;]',
            stripped
        )
        if decl_match:
            var_types[decl_match.group(2)] = decl_match.group(1)

        # Check for type punning through pointer dereference
        for match in type_pun_pattern.finditer(line):
            cast_to = match.group(1).strip()
            var_name = match.group(2).strip()

            # Try to infer source type
            src_type = var_types.get(var_name, 'unknown')

            # Check if this is a known violation pattern
            cast_to_base = get_base_type(cast_to)
            if src_type != 'unknown':
                if not types_can_alias(src_type + ' *', cast_to):
                    issues.append(CastIssue(
                        line_num=i,
                        line=stripped if len(stripped) <= 70 else stripped[:67] + '...',
                        issue_type='strict_aliasing_violation',
                        description=f'Type punning {src_type}* → {cast_to_base}* violates strict aliasing',
                        severity='warning',
                        fix_hint=f'Use union or memcpy for type punning, or sse2neon_recast_*',
                        optimization_note='Undefined behavior at -O2/-O3; compiler may misoptimize'
                    ))

        # Check for address casts that set up aliasing violations
        for match in addr_cast_pattern.finditer(line):
            cast_to = match.group(1).strip()
            var_name = match.group(2).strip()
            src_type = var_types.get(var_name, 'unknown')

            if src_type != 'unknown':
                cast_to_base = get_base_type(cast_to)

                # int* <-> float* is the classic violation
                int_types = {'int', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t'}
                float_types = {'float', 'double', 'float32_t', 'float64_t'}

                if (src_type in int_types and cast_to_base in float_types) or \
                   (src_type in float_types and cast_to_base in int_types):
                    issues.append(CastIssue(
                        line_num=i,
                        line=stripped if len(stripped) <= 70 else stripped[:67] + '...',
                        issue_type='potential_aliasing_violation',
                        description=f'Cast ({cast_to})&{var_name} may violate strict aliasing',
                        severity='info',
                        fix_hint='If value is dereferenced, use union/memcpy instead',
                        optimization_note='Safe if only used with memcpy; UB if dereferenced directly'
                    ))

    return issues


def check_reinterpret_cast_aliasing(stats: 'CastStats', content: str) -> List[CastIssue]:
    """Check _sse2neon_reinterpret_cast uses for potential aliasing issues."""
    issues = []
    lines = content.split('\n')

    # Pattern to extract full reinterpret_cast usage
    reinterpret_pattern = re.compile(
        r'_sse2neon_reinterpret_cast\s*\(\s*([^,]+)\s*,\s*([^)]+)\)'
    )

    for i, line in enumerate(lines, 1):
        if '#define' in line:
            continue

        for match in reinterpret_pattern.finditer(line):
            target_type = match.group(1).strip()
            source_expr = match.group(2).strip()

            # Check for obviously safe patterns
            if is_char_pointer(target_type) or is_void_pointer(target_type):
                continue

            # Check for NEON vector types (usually safe)
            if any(t in target_type for t in ['x8_t', 'x4_t', 'x2_t', 'x1_t', 'x16_t']):
                continue

            # Check for &var pattern (address-of)
            if source_expr.startswith('&'):
                var_name = source_expr[1:].strip()
                # This is a pointer-from-address cast - higher risk
                if '*' in target_type:
                    target_base = get_base_type(target_type)
                    int_types = {'int', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t',
                                 'short', 'int16_t', 'uint16_t'}
                    float_types = {'float', 'double', 'float32_t', 'float64_t'}

                    # Detect int <-> float pointer aliasing
                    if target_base in float_types:
                        issues.append(CastIssue(
                            line_num=i,
                            line=line.strip()[:70] + '...' if len(line.strip()) > 70 else line.strip(),
                            issue_type='reinterpret_aliasing_risk',
                            description=f'reinterpret_cast to {target_type} from address',
                            severity='info',
                            fix_hint='Ensure source type is compatible or use union/memcpy',
                            optimization_note='May violate strict aliasing if source is integer type'
                        ))

    return issues


def analyze_cast_usage(content: str) -> CastStats:
    """Analyze the usage patterns of _sse2neon_*_cast macros."""
    stats = CastStats()
    lines = content.split('\n')

    static_pattern = re.compile(r'_sse2neon_static_cast\s*\(\s*([^,]+),')
    reinterpret_pattern = re.compile(r'_sse2neon_reinterpret_cast\s*\(\s*([^,]+),')
    const_pattern = re.compile(r'_sse2neon_const_cast\s*\(\s*([^,]+),')

    for i, line in enumerate(lines, 1):
        if '#define' in line and '_sse2neon_' in line:
            continue

        for match in static_pattern.finditer(line):
            stats.static_cast_count += 1
            stats.static_cast_uses.append((i, match.group(1).strip()))

        for match in reinterpret_pattern.finditer(line):
            stats.reinterpret_cast_count += 1
            stats.reinterpret_cast_uses.append((i, match.group(1).strip()))

        for match in const_pattern.finditer(line):
            stats.const_cast_count += 1
            stats.const_cast_uses.append((i, match.group(1).strip()))

    return stats


def check_cast_appropriateness(stats: CastStats) -> List[CastIssue]:
    """Check if casts are used appropriately for their purpose."""
    issues = []

    for line_num, target_type in stats.static_cast_uses:
        if '*' in target_type:
            issues.append(CastIssue(
                line_num=line_num,
                line=f'_sse2neon_static_cast({target_type}, ...)',
                issue_type='static_cast_pointer',
                description=f'static_cast to pointer type "{target_type}"',
                severity='warning',
                fix_hint='Use _sse2neon_reinterpret_cast for pointer conversions',
                optimization_note='reinterpret_cast is semantically correct for pointers'
            ))

    numeric_pattern = re.compile(r'^(int|uint|float|double|char|short|long)\d*(_t)?$')
    for line_num, target_type in stats.reinterpret_cast_uses:
        clean_type = target_type.replace('const', '').replace('volatile', '').strip()
        if numeric_pattern.match(clean_type) and '*' not in target_type:
            issues.append(CastIssue(
                line_num=line_num,
                line=f'_sse2neon_reinterpret_cast({target_type}, ...)',
                issue_type='reinterpret_cast_numeric',
                description=f'reinterpret_cast to numeric type "{target_type}"',
                severity='info',
                fix_hint='Consider _sse2neon_static_cast for numeric conversions',
                optimization_note='static_cast generates proper conversion code'
            ))

    return issues


def analyze_macro_consistency(content: str) -> List[CastIssue]:
    """Check that the C and C++ macro definitions are consistent."""
    issues = []
    lines = content.split('\n')

    cpp_defs = {}
    c_defs = {}
    in_cpp = False
    in_c = False

    for i, line in enumerate(lines, 1):
        if '#ifdef __cplusplus' in line or '#if defined(__cplusplus)' in line:
            in_cpp = True
            in_c = False
        elif '#else' in line and in_cpp:
            in_cpp = False
            in_c = True
        elif '#endif' in line:
            in_cpp = False
            in_c = False

        macro_match = re.match(
            r'#define\s+(_sse2neon_\w+_cast)\s*\(([^)]+)\)\s+(.+)',
            line.strip()
        )
        if macro_match:
            macro_name = macro_match.group(1)
            if in_cpp:
                cpp_defs[macro_name] = i
            elif in_c:
                c_defs[macro_name] = i

    for macro_name in set(cpp_defs.keys()) | set(c_defs.keys()):
        if macro_name not in cpp_defs:
            issues.append(CastIssue(
                line_num=c_defs.get(macro_name, 0),
                line=macro_name,
                issue_type='missing_cpp_def',
                description=f'{macro_name} missing C++ definition',
                severity='error',
                fix_hint='Add C++ definition in #ifdef __cplusplus block'
            ))
        elif macro_name not in c_defs:
            issues.append(CastIssue(
                line_num=cpp_defs.get(macro_name, 0),
                line=macro_name,
                issue_type='missing_c_def',
                description=f'{macro_name} missing C definition',
                severity='error',
                fix_hint='Add C definition in #else block'
            ))

    return issues


def validate_sse2neon_header(filepath: str, check_c_style: bool = True,
                              check_aliasing: bool = True
                              ) -> Tuple[bool, List[CastIssue], CastStats]:
    """Main validation function for sse2neon.h cast usage."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return False, [CastIssue(0, '', 'file_not_found',
                                 f'File not found: {filepath}', 'error')], CastStats()

    all_issues = []

    # Check for raw C++ casts (errors)
    all_issues.extend(find_raw_cpp_casts(content))

    # Check for old C-style casts (warnings)
    if check_c_style:
        c_style_issues, _ = find_c_style_casts(content)
        all_issues.extend(c_style_issues)

    # Check for strict aliasing violations
    if check_aliasing:
        all_issues.extend(find_strict_aliasing_violations(content))

    # Analyze cast usage
    stats = analyze_cast_usage(content)

    # Check cast appropriateness
    all_issues.extend(check_cast_appropriateness(stats))

    # Check reinterpret_cast for aliasing risks
    if check_aliasing:
        all_issues.extend(check_reinterpret_cast_aliasing(stats, content))

    # Check macro consistency
    all_issues.extend(analyze_macro_consistency(content))

    is_valid = not any(issue.severity == 'error' for issue in all_issues)

    return is_valid, all_issues, stats


def print_report(is_valid: bool, issues: List[CastIssue], stats: CastStats,
                 verbose: bool = False, show_guidelines: bool = False):
    """Print a formatted validation report."""
    print()
    print("Checking casts for C/C++ compatibility and strict aliasing (UB at -O2/-O3)")
    print()

    total_macros = (stats.static_cast_count +
                    stats.reinterpret_cast_count +
                    stats.const_cast_count)

    print("Cast Macro Usage:")
    print(f"  _sse2neon_static_cast:      {stats.static_cast_count:4d}  (numeric conversions)")
    print(f"  _sse2neon_reinterpret_cast: {stats.reinterpret_cast_count:4d}  (pointer/type punning)")
    print(f"  _sse2neon_const_cast:       {stats.const_cast_count:4d}  (const removal)")
    print(f"  {'─' * 42}")
    print(f"  Total cast macros:          {total_macros:4d}")
    print()

    errors = [i for i in issues if i.severity == 'error']
    warnings = [i for i in issues if i.severity == 'warning']
    infos = [i for i in issues if i.severity == 'info']

    # Categorize by type
    aliasing_issues = [i for i in issues if 'aliasing' in i.issue_type]
    cast_issues = [i for i in issues if 'aliasing' not in i.issue_type]

    if errors:
        print("─" * 78)
        print(f"ERRORS ({len(errors)}) - Must fix before merge")
        print("─" * 78)
        for issue in errors:
            print(f"\n  Line {issue.line_num}: {issue.description}")
            if issue.line:
                code = issue.line if len(issue.line) <= 65 else issue.line[:62] + '...'
                print(f"    Code: {code}")
            if issue.fix_hint:
                print(f"    Fix:  {issue.fix_hint}")
        print()

    if warnings:
        print("─" * 78)
        print(f"WARNINGS ({len(warnings)}) - Should fix for optimization")
        print("─" * 78)

        # Separate aliasing warnings from cast warnings
        aliasing_warns = [w for w in warnings if 'aliasing' in w.issue_type]
        cast_warns = [w for w in warnings if 'aliasing' not in w.issue_type]

        if aliasing_warns:
            print(f"\n  Strict Aliasing Issues ({len(aliasing_warns)}):")
            if verbose:
                for issue in aliasing_warns[:10]:
                    print(f"    Line {issue.line_num}: {issue.description}")
                    if issue.fix_hint:
                        print(f"      Fix: {issue.fix_hint}")
                if len(aliasing_warns) > 10:
                    print(f"    ... and {len(aliasing_warns) - 10} more")
            else:
                print(f"    Use -v to see details. These may cause UB at -O2/-O3.")

        if cast_warns:
            print(f"\n  C-style Pointer Casts ({len(cast_warns)}):")
            if verbose:
                for issue in cast_warns[:15]:
                    print(f"    Line {issue.line_num}: {issue.description}")
                    if issue.fix_hint:
                        print(f"      Fix: {issue.fix_hint}")
                if len(cast_warns) > 15:
                    print(f"    ... and {len(cast_warns) - 15} more")
            else:
                print(f"    Use -v to see all locations.")
        print()

    if infos and verbose:
        print("─" * 78)
        print(f"INFO ({len(infos)}) - Optimization suggestions")
        print("─" * 78)
        for issue in infos[:10]:
            print(f"\n  Line {issue.line_num}: {issue.description}")
            if issue.fix_hint:
                print(f"    Hint: {issue.fix_hint}")
            if issue.optimization_note:
                print(f"    Why:  {issue.optimization_note}")
        if len(infos) > 10:
            print(f"\n  ... and {len(infos) - 10} more")
        print()

    print("─" * 78)
    print(f"Summary: {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} info")
    print("─" * 78)

    if verbose or aliasing_issues:
        print()
        print("Strict Aliasing & Optimization:")
        print("  - The compiler assumes different pointer types don't alias")
        print("  - int* <-> float* casts can cause misoptimization at -O2/-O3")
        print("  - Use union, memcpy, or sse2neon_recast_* for type punning")
        print("  - char*/uint8_t* can safely alias any type")
        print()

    print()
    if is_valid:
        if warnings:
            print("  [PASS] No errors (warnings should be reviewed for optimization)")
        else:
            print("  [PASS] All cast and aliasing checks passed")
    else:
        print("  [FAIL] Validation failed")
        print()
        print("  Fix errors above. For guidelines: python3 .ci/check-casts.py --help-casts")

    if show_guidelines or (not is_valid and errors):
        print()
        print(CAST_GUIDELINES)


def main():
    parser = argparse.ArgumentParser(
        description='Validate casts and strict aliasing in sse2neon.h',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 .ci/check-casts.py                # Basic validation
  python3 .ci/check-casts.py -v             # Verbose with all warnings
  python3 .ci/check-casts.py --strict       # Treat warnings as errors
  python3 .ci/check-casts.py --help-casts   # Show guidelines

Strict Aliasing:
  The compiler assumes pointers of different types (e.g., int* vs float*)
  don't point to the same memory. Violations cause undefined behavior and
  can result in incorrect code at -O2/-O3. This check identifies:
  - Direct type punning: *(float *)&int_var
  - Suspicious pointer casts between incompatible types
  - Missing use of union/memcpy for safe type punning

For more details, see: https://github.com/DLTcollab/sse2neon/pull/671
"""
    )
    parser.add_argument('filepath', nargs='?', default='sse2neon.h',
                        help='Path to sse2neon.h (default: sse2neon.h)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show all warnings and suggestions')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Minimal output: only errors and result')
    parser.add_argument('--strict', action='store_true',
                        help='Treat warnings as errors')
    parser.add_argument('--no-c-style-check', action='store_true',
                        help='Skip C-style cast checks')
    parser.add_argument('--no-aliasing-check', action='store_true',
                        help='Skip strict aliasing checks')
    parser.add_argument('--help-casts', action='store_true',
                        help='Show cast and aliasing guidelines')

    args = parser.parse_args()

    if args.help_casts:
        print(CAST_GUIDELINES)
        sys.exit(0)

    is_valid, issues, stats = validate_sse2neon_header(
        args.filepath,
        check_c_style=not args.no_c_style_check,
        check_aliasing=not args.no_aliasing_check
    )

    if args.strict:
        for issue in issues:
            if issue.severity == 'warning':
                issue.severity = 'error'
        is_valid = not any(issue.severity == 'error' for issue in issues)

    if args.quiet:
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        if errors:
            for issue in errors:
                print(f"ERROR line {issue.line_num}: {issue.description}")
        if warnings:
            aliasing = len([w for w in warnings if 'aliasing' in w.issue_type])
            casts = len(warnings) - aliasing
            if aliasing:
                print(f"WARN: {aliasing} potential aliasing issue(s)")
            if casts:
                print(f"WARN: {casts} C-style cast(s)")
        print(f"Result: {'PASS' if is_valid else 'FAIL'}")
    else:
        print_report(is_valid, issues, stats, args.verbose)

    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
