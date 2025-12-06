#!/usr/bin/env bash

# Check if perf-tier.md is in sync with sse2neon.h
# Returns 0 if in sync, 1 if out of sync

set -e

# Generate fresh report (includes header)
python3 scripts/gen-perf-report.py > perf-tier-generated.md

# Direct comparison - gen-perf-report.py now outputs the complete file
if diff -q perf-tier.md perf-tier-generated.md > /dev/null 2>&1; then
    echo "perf-tier.md is up to date"
    rm -f perf-tier-generated.md
    exit 0
else
    echo "perf-tier.md is out of sync with sse2neon.h"
    echo ""
    echo "To regenerate:"
    echo "  python3 scripts/gen-perf-report.py > perf-tier.md"
    echo ""
    echo "For detailed analysis:"
    echo "  python3 scripts/analyze-tiers.py --markdown"
    rm -f perf-tier-generated.md
    exit 1
fi
