#!/usr/bin/env python3
"""
Golden Reference Data Generator for sse2neon Differential Testing

This script builds and runs the differential test harness on x86 to generate
golden reference data that can be used to verify NEON implementations on ARM.

Usage:
    python3 scripts/gen-golden.py [output_dir]

Requirements:
    - Must run on x86/x86_64 platform
    - GCC or Clang compiler with SSE support
    - Output directory defaults to 'golden/'

The generated golden data files contain the expected outputs from native x86
SSE intrinsics. These are compared against sse2neon outputs on ARM platforms.

Known x86/ARM differences are documented in IMPROVE.md and handled with
appropriate tolerances in the verification phase.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys


def check_platform():
    """Verify we're running on x86."""
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "i386", "i686", "x86"):
        print(f"ERROR: This script must run on x86. Detected: {machine}")
        print("Golden data generation requires native SSE execution.")
        sys.exit(1)
    return machine


def find_compiler():
    """Find a suitable C++ compiler."""
    for compiler in ["g++", "clang++"]:
        if shutil.which(compiler):
            return compiler
    print("ERROR: No C++ compiler found (tried g++, clang++)")
    sys.exit(1)


def build_differential(project_root, compiler):
    """Build the differential test harness for x86."""
    build_dir = os.path.join(project_root, "build_differential")
    os.makedirs(build_dir, exist_ok=True)

    # Source files
    sources = ["tests/differential.cpp", "tests/common.cpp", "tests/binding.cpp"]

    # Compiler flags for x86 SSE support
    cflags = [
        "-O2",
        "-Wall",
        "-std=gnu++14",
        "-maes",
        "-mpclmul",
        "-mssse3",
        "-msse4.2",
        "-I.",
    ]

    output = os.path.join(build_dir, "differential")

    cmd = [compiler] + cflags + sources + ["-o", output, "-lm"]

    print(f"Building differential test harness...")
    print(f"  Compiler: {compiler}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Build failed with exit code {e.returncode}")
        sys.exit(1)

    return output


def generate_golden(executable, output_dir):
    """Run the differential harness to generate golden data."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [executable, "--generate", output_dir]

    print(f"\nGenerating golden reference data...")
    print(f"  Output directory: {output_dir}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Golden generation failed with exit code {e.returncode}")
        sys.exit(1)

    # Count generated files
    golden_files = [f for f in os.listdir(output_dir) if f.endswith(".golden")]
    print(f"\nGenerated {len(golden_files)} golden data files")

    return len(golden_files)


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden reference data for sse2neon differential testing"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="golden",
        help="Output directory for golden data (default: golden/)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing golden data before generating",
    )
    args = parser.parse_args()

    # Find project root (parent of scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("sse2neon Golden Data Generator")
    print("=" * 40)

    # Platform check
    machine = check_platform()
    print(f"Platform: {machine} (OK)")

    # Find compiler
    compiler = find_compiler()
    print(f"Compiler: {compiler}")

    # Output directory
    output_dir = os.path.join(project_root, args.output_dir)
    if args.clean and os.path.exists(output_dir):
        print(f"\nCleaning existing golden data in {output_dir}...")
        shutil.rmtree(output_dir)

    # Build
    executable = build_differential(project_root, compiler)
    print(f"Built: {executable}")

    # Generate
    count = generate_golden(executable, output_dir)

    print("\n" + "=" * 40)
    print("Golden data generation complete!")
    print(f"\nTo verify on ARM:")
    print(f"  1. Copy '{args.output_dir}/' to ARM target")
    print(f"  2. Run: make check-differential GOLDEN_DIR={args.output_dir}")
    print("\nSee IMPROVE.md for known x86/ARM semantic differences.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
