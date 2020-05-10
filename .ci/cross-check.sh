#!/usr/bin/env bash

MACHINE_TYPE=`uname -m`
if [ ${MACHINE_TYPE} != 'x86_64' ]; then
    exit
fi

OS_TYPE=`uname -s`
if [ ${OS_TYPE} != 'Linux' ]; then
    exit
fi

GCC_REL=gcc-linaro-7.5.0-2019.12

set -x

make clean
export PATH=${GCC_REL}-x86_64_aarch64-linux-gnu/bin:$PATH
make CROSS_COMPILE=aarch64-linux-gnu- check || exit 1 # ARMv8-A

make clean
export PATH=${GCC_REL}-x86_64_arm-linux-gnueabihf/bin:$PATH
make CROSS_COMPILE=arm-linux-gnueabihf- check || exit 1 # ARMv7-A
