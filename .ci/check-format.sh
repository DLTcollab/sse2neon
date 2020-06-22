#!/usr/bin/env bash

GCC_REL=9.2-2019.12
SOURCES=$(find $(git rev-parse --show-toplevel) | egrep "\.(cpp|h)\$" | egrep -v "gcc-arm-${GCC_REL}-x86_64-aarch64-none-linux-gnu|gcc-arm-${GCC_REL}-x86_64-arm-none-linux-gnueabihf")

set -x

exit $(clang-format --output-replacements-xml ${SOURCES} | egrep -c "</replacement>")
