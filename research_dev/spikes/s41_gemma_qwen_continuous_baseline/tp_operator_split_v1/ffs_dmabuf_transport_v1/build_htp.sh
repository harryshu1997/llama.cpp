#!/bin/sh
set -eu

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_root/../../../../.." && pwd)
ndk_root=${ANDROID_NDK_ROOT:-/home/myid/zs89458/android/android-ndk-r27c}
android_cxx=${ANDROID_CXX:-$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++}
host_cxx=${HOST_CXX:-g++}
runtime_root=${S41_HEXAGON_RUNTIME_ROOT:-$repo_root/build-s11-e0-cp15-android/bin}
hexagon_lib=${S41_HEXAGON_LIB:-$runtime_root/libggml-hexagon.so}
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

"$android_cxx" -std=c++20 -O3 -Wall -Wextra -Werror \
    -I"$repo_root/ggml/include" \
    "$script_root/ffs_dmabuf_htp_worker.cpp" \
    -L"$runtime_root" -lggml -lggml-base "$hexagon_lib" \
    -o "$script_root/ffs_dmabuf_htp_worker.android"

"$host_cxx" -std=c++17 -O3 -Wall -Wextra -Werror \
    "$script_root/ffs_dmabuf_htp_host.cpp" $host_libusb \
    -o "$script_root/ffs_dmabuf_htp_host"
