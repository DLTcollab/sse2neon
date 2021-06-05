#!/usr/bin/env bash

MACHINE_TYPE=`uname -m`
if [ ${MACHINE_TYPE} != 'x86_64' ]; then
    exit
fi

OS_TYPE=`uname -s`
if [ ${OS_TYPE} != 'Linux' ]; then
    exit
fi

# Clang/LLVM is natively a cross-compiler.
# TODO: Do cross-compilation using Clang
# https://clang.llvm.org/docs/CrossCompilation.html
if [ $(printenv CXX | grep clang) ]; then
    exit
fi

GCC_REL=10.2-2020.11

set -x

make clean
export PATH=gcc-arm-${GCC_REL}-x86_64-aarch64-none-linux-gnu/bin:$PATH
make CROSS_COMPILE=aarch64-none-linux-gnu- check || exit 1 # ARMv8-A

make clean
export PATH=gcc-arm-${GCC_REL}-x86_64-arm-none-linux-gnueabihf/bin:$PATH
make CROSS_COMPILE=arm-none-linux-gnueabihf- check || exit 1 # ARMv7-A
