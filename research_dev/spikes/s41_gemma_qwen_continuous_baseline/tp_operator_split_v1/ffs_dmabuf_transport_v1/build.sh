#!/bin/sh
set -eu

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ndk_root=${ANDROID_NDK_ROOT:-/home/myid/zs89458/android/android-ndk-r27c}
android_cc=${ANDROID_CC:-$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang}
host_cxx=${HOST_CXX:-g++}
host_libusb=${HOST_LIBUSB:-}

if [ -z "$host_libusb" ]; then
    if pkg-config --exists libusb-1.0 2>/dev/null; then
        host_libusb=$(pkg-config --libs libusb-1.0)
    elif [ -e /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 ]; then
        host_libusb=-l:libusb-1.0.so.0
    else
        host_libusb=-lusb-1.0
    fi
fi

"$android_cc" -std=c11 -O3 -Wall -Wextra -Werror -pthread \
    "$script_root/ffs_dmabuf_phone.c" \
    -o "$script_root/ffs_dmabuf_phone.android"

"$host_cxx" -std=c++17 -O3 -Wall -Wextra -Werror \
    "$script_root/ffs_dmabuf_host.cpp" \
    "$script_root/../../../../../examples/layersplit/ffn-split-usb-client.cpp" \
    $host_libusb \
    -o "$script_root/ffs_dmabuf_host"
