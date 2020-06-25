#!/usr/bin/env bash

ARM_MIRROR=https://github.com/DLTcollab/toolchain-arm/raw/master
GCC_REL=9.2-2019.12

MACHINE_TYPE=`uname -m`
if [ ${MACHINE_TYPE} != 'x86_64' ]; then
    exit
fi

OS_TYPE=`uname -s`
if [ ${OS_TYPE} != 'Linux' ]; then
    exit
fi

set -x

sudo apt-get install -y curl xz-utils qemu-user

curl -L \
    ${ARM_MIRROR}/gcc-arm-${GCC_REL}-x86_64-arm-none-linux-gnueabihf.tar.xz \
    | tar -Jx || exit 1

curl -L \
    ${ARM_MIRROR}/gcc-arm-${GCC_REL}-x86_64-aarch64-none-linux-gnu.tar.xz \
    | tar -Jx || exit 1
