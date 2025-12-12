#!/usr/bin/env python3
"""
Validate preprocessor macro usage in sse2neon.h for consistency and correctness.

This script enforces macro hygiene by detecting:
1. Typos in architecture check macros (e.g., missing underscores)
2. Inconsistent usage of consolidated macros vs raw checks
3. Potential macro definition issues

WHY THIS MATTERS:
Raw `defined(__aarch64__)` checks miss MSVC's `_M_ARM64` and some compilers'
`__arm64__`, causing those builds to silently fall back to slower ARMv7 code
paths. Using the canonical `SSE2NEON_ARCH_AARCH64` macro ensures all 64-bit
ARM builds get the optimized AArch64 code paths.

The canonical macros defined in sse2neon.h are:
- SSE2NEON_ARCH_AARCH64: Use instead of raw __aarch64__ checks
- SSE2NEON_COMPILER_GCC_COMPAT: Use instead of raw __GNUC__ || __clang__
- SSE2NEON_COMPILER_CLANG: Use instead of raw __clang__
- SSE2NEON_COMPILER_MSVC: Use instead of raw _MSC_VER

LEGITIMATE EXCEPTIONS (raw macro usage allowed):
1. The macro definition itself (lines 219-223)
2. Apple-specific checks: __APPLE__ && __aarch64__ (line 416)
3. Compiler version workarounds: __GNUC__ == 10 && ... && __aarch64__ (line 873)
4. Builtin detection: __aarch64__ || __has_builtin(...) (line 1152)
5. Feature-gated inline asm: __aarch64__ && __ARM_FEATURE_CRC32 (lines 9449-9513)
   These use GCC-style inline assembly that won't compile on MSVC anyway.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class MacroIssue:
    """Represents a macro usage issue."""

    line_num: int
    line_content: str
    issue_type: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None


@dataclass
class MacroStats:
    """Statistics about macro usage."""

    canonical_count: int = 0
    raw_count: int = 0
    legitimate_raw: int = 0
    issues: List[MacroIssue] = field(default_factory=list)


# Known typo patterns that indicate bugs
TYPO_PATTERNS = [
    # Missing leading underscore on architecture macros
    (
        r"defined\s*\(\s*aarch64__\s*\)",
        "defined(__aarch64__)",
        "Missing leading underscore in aarch64 check",
    ),
    (
        r"defined\s*\(\s*__aarch64\s*\)",
        "defined(__aarch64__)",
        "Missing trailing underscores in aarch64 check",
    ),
    (
        r"defined\s*\(\s*aarch64\s*\)",
        "defined(__aarch64__)",
        "Missing underscores in aarch64 check",
    ),
    (
        r"defined\s*\(\s*arm64__\s*\)",
        "defined(__arm64__)",
        "Missing leading underscore in arm64 check",
    ),
    (
        r"defined\s*\(\s*__arm64\s*\)",
        "defined(__arm64__)",
        "Missing trailing underscores in arm64 check",
    ),
    # Missing underscores on compiler macros
    (
        r"defined\s*\(\s*GNUC__\s*\)",
        "defined(__GNUC__)",
        "Missing leading underscore in GNUC check",
    ),
    (
        r"defined\s*\(\s*__GNUC\s*\)",
        "defined(__GNUC__)",
        "Missing trailing underscores in GNUC check",
    ),
    (
        r"defined\s*\(\s*clang__\s*\)",
        "defined(__clang__)",
        "Missing leading underscore in clang check",
    ),
    (
        r"defined\s*\(\s*__clang\s*\)",
        "defined(__clang__)",
        "Missing trailing underscores in clang check",
    ),
    # Common typos
    (
        r"defined\s*\(\s*_M_ARM64_\s*\)",
        "defined(_M_ARM64)",
        "Extra trailing underscore in _M_ARM64",
    ),
    (
        r"defined\s*\(\s*__M_ARM64\s*\)",
        "defined(_M_ARM64)",
        "Double underscore prefix in _M_ARM64",
    ),
]

# Patterns for raw macro usage that should use canonical macros
RAW_MACRO_PATTERNS = {
    "aarch64": {
        "pattern": r"defined\s*\(\s*__aarch64__\s*\)",
        "negated_pattern": r"!\s*defined\s*\(\s*__aarch64__\s*\)",
        "canonical": "SSE2NEON_ARCH_AARCH64",
        "exception_patterns": [
            # The definition line itself
            r"#define\s+SSE2NEON_ARCH_AARCH64",
            # Feature-gated: architecture AND specific feature required
            # These use GCC inline asm that won't compile on MSVC anyway
            r"defined\s*\(\s*__aarch64__\s*\)\s*&&\s*defined\s*\(\s*__ARM_FEATURE_",
            # Platform-specific with additional constraint
            r"defined\s*\(\s*__APPLE__\s*\)\s*&&\s*.*defined\s*\(\s*__aarch64__\s*\)",
            # Compiler version workarounds
            r"__GNUC__\s*==\s*\d+.*defined\s*\(\s*__aarch64__\s*\)",
            # Part of multi-platform definition (the canonical macro definition)
            r"defined\s*\(\s*__aarch64__\s*\)\s*\|\|\s*defined\s*\(\s*__arm64__\s*\)",
            r"defined\s*\(\s*__aarch64__\s*\)\s*\|\|\s*__has_builtin",
        ],
    },
    "gnuc": {
        "pattern": r"defined\s*\(\s*__GNUC__\s*\)\s*\|\|\s*defined\s*\(\s*__clang__\s*\)",
        "canonical": "SSE2NEON_COMPILER_GCC_COMPAT",
        "exception_patterns": [
            r"#define\s+SSE2NEON_COMPILER_GCC_COMPAT",
        ],
    },
    "msvc": {
        "pattern": r"defined\s*\(\s*_MSC_VER\s*\)",
        "negated_pattern": r"!\s*defined\s*\(\s*_MSC_VER\s*\)",
        # Also match #ifdef _MSC_VER and bare (_MSC_VER)
        "ifdef_pattern": r"#ifdef\s+_MSC_VER",
        "bare_pattern": r"\(\s*_MSC_VER\s*\)",
        "canonical": "SSE2NEON_COMPILER_MSVC",
        "exception_patterns": [
            # The compiler detection block where SSE2NEON_COMPILER_MSVC is defined
            r"#if\s+defined\s*\(\s*_MSC_VER\s*\)\s*$",
            r"#elif\s+defined\s*\(\s*_MSC_VER\s*\)\s*$",
        ],
    },
}

# Architecture macro definitions to verify exist
REQUIRED_MACRO_DEFINITIONS = [
    ("SSE2NEON_ARCH_AARCH64", r"#define\s+SSE2NEON_ARCH_AARCH64\s+[01]"),
    ("SSE2NEON_COMPILER_GCC_COMPAT", r"#define\s+SSE2NEON_COMPILER_GCC_COMPAT\s+[01]"),
    ("SSE2NEON_COMPILER_CLANG", r"#define\s+SSE2NEON_COMPILER_CLANG\s+[01]"),
    ("SSE2NEON_COMPILER_MSVC", r"#define\s+SSE2NEON_COMPILER_MSVC\s+[01]"),
]


def check_typos(lines: List[str]) -> List[MacroIssue]:
    """Check for typos in macro definitions."""
    issues = []

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue

        for pattern, correct, description in TYPO_PATTERNS:
            if re.search(pattern, line):
                issues.append(
                    MacroIssue(
                        line_num=line_num,
                        line_content=line.rstrip(),
                        issue_type="typo",
                        description=description,
                        severity="error",
                        suggestion=f"Use {correct}",
                    )
                )

    return issues


def is_exception_line(line: str, prev_line: str, exception_patterns: List[str]) -> bool:
    """Check if a line matches any exception pattern."""
    # Check current line and context (for multi-line conditions)
    context = prev_line + "\n" + line if prev_line else line
    for pattern in exception_patterns:
        if re.search(pattern, context):
            return True
    return False


def check_raw_macro_usage(lines: List[str]) -> Tuple[MacroStats, List[MacroIssue]]:
    """Check for raw macro usage that should use canonical macros.

    Raw __aarch64__ checks are treated as ERRORS because they cause
    _M_ARM64/__arm64__ builds to silently use slower fallback paths.

    Raw _MSC_VER checks should use SSE2NEON_COMPILER_MSVC for consistency.
    """
    stats = MacroStats()
    issues = []

    for line_num, line in enumerate(lines, 1):
        # Skip comments and documentation
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue

        # Check for canonical macro usage (good)
        if "SSE2NEON_ARCH_AARCH64" in line:
            stats.canonical_count += 1
            continue
        if "SSE2NEON_COMPILER_GCC_COMPAT" in line:
            stats.canonical_count += 1
            continue
        if "SSE2NEON_COMPILER_MSVC" in line:
            stats.canonical_count += 1
            continue
        if "SSE2NEON_COMPILER_CLANG" in line:
            stats.canonical_count += 1
            continue

        prev_line = lines[line_num - 2] if line_num > 1 else ""

        # Check for raw aarch64 usage (both positive and negated)
        aarch64_info = RAW_MACRO_PATTERNS["aarch64"]
        has_raw_check = re.search(aarch64_info["pattern"], line)
        has_negated_check = re.search(aarch64_info.get("negated_pattern", r"$^"), line)

        if has_raw_check or has_negated_check:
            if is_exception_line(line, prev_line, aarch64_info["exception_patterns"]):
                stats.legitimate_raw += 1
            else:
                stats.raw_count += 1
                # Determine the correct replacement
                if has_negated_check:
                    suggestion = "Replace with #if !SSE2NEON_ARCH_AARCH64"
                else:
                    suggestion = "Replace with #if SSE2NEON_ARCH_AARCH64"

                issues.append(
                    MacroIssue(
                        line_num=line_num,
                        line_content=line.rstrip(),
                        issue_type="inconsistent_arch_check",
                        description=(
                            f"Raw __aarch64__ check misses _M_ARM64/__arm64__ builds. "
                            f'Use {aarch64_info["canonical"]} for correct 64-bit ARM detection.'
                        ),
                        severity="error",
                        suggestion=suggestion,
                    )
                )
            continue

        # Check for raw _MSC_VER usage
        msvc_info = RAW_MACRO_PATTERNS["msvc"]
        has_msvc_check = re.search(msvc_info["pattern"], line)
        has_msvc_negated = re.search(msvc_info.get("negated_pattern", r"$^"), line)
        has_ifdef_msvc = re.search(msvc_info.get("ifdef_pattern", r"$^"), line)
        has_bare_msvc = re.search(msvc_info.get("bare_pattern", r"$^"), line)

        if has_msvc_check or has_msvc_negated or has_ifdef_msvc or has_bare_msvc:
            if is_exception_line(line, prev_line, msvc_info["exception_patterns"]):
                stats.legitimate_raw += 1
            else:
                stats.raw_count += 1
                # Determine the correct replacement based on pattern
                if has_msvc_negated:
                    suggestion = "Replace with #if !SSE2NEON_COMPILER_MSVC"
                elif has_ifdef_msvc:
                    suggestion = "Replace with #if SSE2NEON_COMPILER_MSVC"
                else:
                    suggestion = "Replace with SSE2NEON_COMPILER_MSVC"

                issues.append(
                    MacroIssue(
                        line_num=line_num,
                        line_content=line.rstrip(),
                        issue_type="inconsistent_compiler_check",
                        description=(
                            f'Raw _MSC_VER check should use {msvc_info["canonical"]} '
                            f"for consistent compiler detection."
                        ),
                        severity="error",
                        suggestion=suggestion,
                    )
                )

    return stats, issues


def check_macro_definitions(content: str) -> List[MacroIssue]:
    """Verify required macro definitions exist."""
    issues = []

    for macro_name, pattern in REQUIRED_MACRO_DEFINITIONS:
        if not re.search(pattern, content):
            issues.append(
                MacroIssue(
                    line_num=0,
                    line_content="",
                    issue_type="missing_definition",
                    description=f"Required macro {macro_name} not found or has incorrect format",
                    severity="error",
                    suggestion=f"Ensure {macro_name} is defined as 0 or 1",
                )
            )

    return issues


def check_ifdef_patterns(lines: List[str]) -> List[MacroIssue]:
    """Check for problematic #ifdef/#ifndef patterns."""
    issues = []

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Check for #ifdef SSE2NEON_ARCH_AARCH64 (should be #if)
        if re.match(r"#ifdef\s+SSE2NEON_ARCH_AARCH64", stripped):
            issues.append(
                MacroIssue(
                    line_num=line_num,
                    line_content=line.rstrip(),
                    issue_type="ifdef_misuse",
                    description="SSE2NEON_ARCH_AARCH64 is always defined; use #if instead of #ifdef",
                    severity="error",
                    suggestion="Use #if SSE2NEON_ARCH_AARCH64 instead",
                )
            )

        # Similarly for compiler macros
        for macro in [
            "SSE2NEON_COMPILER_GCC_COMPAT",
            "SSE2NEON_COMPILER_CLANG",
            "SSE2NEON_COMPILER_MSVC",
        ]:
            if re.match(rf"#ifdef\s+{macro}", stripped):
                issues.append(
                    MacroIssue(
                        line_num=line_num,
                        line_content=line.rstrip(),
                        issue_type="ifdef_misuse",
                        description=f"{macro} is always defined; use #if instead of #ifdef",
                        severity="error",
                        suggestion=f"Use #if {macro} instead",
                    )
                )

    return issues


def check_clang_cl_msvc_patterns(lines: List[str]) -> List[MacroIssue]:
    """Warn about MSVC-only conditionals that also catch Clang-CL."""
    issues = []

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if not stripped.startswith(("#if", "#elif")):
            continue
        if "SSE2NEON_COMPILER_MSVC" not in line:
            continue

        # Check current line and continuation lines for CLANG guard
        full_condition = line
        check_idx = idx
        while full_condition.rstrip().endswith("\\") and check_idx + 1 < len(lines):
            check_idx += 1
            full_condition += "\n" + lines[check_idx]
        if "SSE2NEON_COMPILER_CLANG" in full_condition:
            continue

        # Legitimate standalone usages
        if re.match(r"#elif\s+SSE2NEON_COMPILER_MSVC\b", stripped):
            continue
        if "SSE2NEON_INCLUDE_WINDOWS_H" in line:
            continue

        lookahead = "\n".join(lines[idx + 1 : idx + 6])
        if re.search(r"#include\s+<intrin\.h>", lookahead):
            continue
        if re.search(r"#include\s+<windows\.h>", lookahead):
            continue
        if re.search(r"#include\s+<processthreadsapi\.h>", lookahead):
            continue
        # SSE2NEON_ALLOC_DEFINED intentionally includes Clang-CL (uses MSVC runtime)
        if re.search(r"SSE2NEON_ALLOC_DEFINED", lookahead):
            continue

        issues.append(
            MacroIssue(
                line_num=idx + 1,
                line_content=line.rstrip(),
                issue_type="clang_cl_guard",
                description=(
                    "Standalone SSE2NEON_COMPILER_MSVC check may include Clang-CL "
                    "in MSVC-only code paths that use MSVC intrinsics."
                ),
                severity="warning",
                suggestion="Guard with && !SSE2NEON_COMPILER_CLANG or refactor the condition",
            )
        )

    return issues


def analyze_file(
    filepath: Path, verbose: bool = False, quiet: bool = False
) -> Tuple[bool, MacroStats]:
    """Analyze a file for macro issues."""
    content = filepath.read_text()
    lines = content.splitlines()

    all_issues = []

    # Run all checks
    all_issues.extend(check_typos(lines))
    all_issues.extend(check_macro_definitions(content))
    all_issues.extend(check_ifdef_patterns(lines))
    all_issues.extend(check_clang_cl_msvc_patterns(lines))

    stats, raw_issues = check_raw_macro_usage(lines)
    all_issues.extend(raw_issues)
    stats.issues = all_issues

    # Print results
    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]

    if errors and not quiet:
        print(f"\n{'='*70}")
        print("ERRORS (must fix):")
        print(f"{'='*70}")
        for issue in errors:
            print(f"\n  Line {issue.line_num}: [{issue.issue_type}]")
            print(f"    {issue.description}")
            if issue.line_content:
                # Truncate long lines
                content_display = issue.line_content
                if len(content_display) > 70:
                    content_display = content_display[:67] + "..."
                print(f"    Code: {content_display}")
            if issue.suggestion:
                print(f"    Fix:  {issue.suggestion}")

    if warnings and verbose and not quiet:
        print(f"\n{'='*70}")
        print("WARNINGS:")
        print(f"{'='*70}")
        for issue in warnings:
            print(f"\n  Line {issue.line_num}: [{issue.issue_type}]")
            print(f"    {issue.description}")
            if issue.line_content:
                content_display = issue.line_content
                if len(content_display) > 70:
                    content_display = content_display[:67] + "..."
                print(f"    Code: {content_display}")
            if issue.suggestion:
                print(f"    Fix:  {issue.suggestion}")

    # Print summary
    print(f"\n{'='*70}")
    print("MACRO USAGE SUMMARY:")
    print(f"{'='*70}")
    print(f"  Canonical macro usage (SSE2NEON_ARCH_*): {stats.canonical_count}")
    print(f"  Legitimate raw usage (exceptions):      {stats.legitimate_raw}")
    print(f"  Inconsistent raw usage (errors):        {stats.raw_count}")
    print(f"  Total errors:   {len(errors)}")
    print(f"  Total warnings: {len(warnings)}")

    if errors:
        print(f"\n{'='*70}")
        print("RATIONALE:")
        print(f"{'='*70}")
        print("  Raw `defined(__aarch64__)` checks miss MSVC _M_ARM64 and some")
        print("  compilers' __arm64__, causing those builds to silently use")
        print("  slower ARMv7 fallback code. Use SSE2NEON_ARCH_AARCH64 instead.")

    # Return success if no errors
    return len(errors) == 0, stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate preprocessor macro usage in sse2neon.h",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="sse2neon.h",
        help="File to analyze (default: sse2neon.h)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show warnings in addition to errors",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show summary, not individual issues",
    )

    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Analyzing macro usage in: {filepath}")

    success, stats = analyze_file(filepath, verbose=args.verbose, quiet=args.quiet)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
