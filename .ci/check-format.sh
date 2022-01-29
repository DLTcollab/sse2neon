#!/usr/bin/env bash

GCC_REL=10.3-2021.07
SOURCES=$(find $(git rev-parse --show-toplevel) | egrep "\.(cpp|h)\$" | egrep -v "gcc-arm-${GCC_REL}-x86_64-aarch64-none-linux-gnu|gcc-arm-${GCC_REL}-x86_64-arm-none-linux-gnueabihf")

set -x

for file in ${SOURCES};
do
    clang-format-12 ${file} > expected-format
    diff -u -p --label="${file}" --label="expected coding style" ${file} expected-format
done
exit $(clang-format-12 --output-replacements-xml ${SOURCES} | egrep -c "</replacement>")
