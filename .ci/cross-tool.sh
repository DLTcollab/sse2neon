#!/usr/bin/env bash

LINARO=https://releases.linaro.org/components/toolchain/binaries/latest-7
GCC_REL=gcc-linaro-7.5.0-2019.12

MACHINE_TYPE=`uname -m`
if [ ${MACHINE_TYPE} != 'x86_64' ]; then
    exit
fi

OS_TYPE=`uname -s`
if [ ${OS_TYPE} != 'Linux' ]; then
    exit
fi

set -x

curl -L \
    ${LINARO}/arm-linux-gnueabihf/${GCC_REL}-x86_64_arm-linux-gnueabihf.tar.xz \
    | tar -Jx || exit 1

curl -L \
    ${LINARO}/aarch64-linux-gnu/${GCC_REL}-x86_64_aarch64-linux-gnu.tar.xz \
    | tar -Jx || exit 1
