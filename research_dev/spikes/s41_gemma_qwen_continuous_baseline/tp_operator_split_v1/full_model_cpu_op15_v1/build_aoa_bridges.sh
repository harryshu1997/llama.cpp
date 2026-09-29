#!/bin/sh
set -eu

output_dir=${1:-}
if [ -z "$output_dir" ]; then
    echo "usage: $0 OUTPUT_DIR" >&2
    exit 2
fi

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
host_cxx=${CXX:-c++}
android_cc=${ANDROID_CC:-}
if [ -z "$android_cc" ]; then
    if [ -z "${ANDROID_NDK_ROOT:-}" ]; then
        echo "set ANDROID_CC or ANDROID_NDK_ROOT" >&2
        exit 2
    fi
    android_cc=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang
fi
if [ ! -x "$android_cc" ]; then
    echo "Android compiler is not executable: $android_cc" >&2
    exit 2
fi

mkdir -p "$output_dir"
"$host_cxx" -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
    "$script_root/aoa_host_bridge.cpp" \
    -Wl,-l:libusb-1.0.so.0 -o "$output_dir/aoa_host_bridge"
"$android_cc" -std=c11 -O2 -Wall -Wextra -Werror -pthread \
    "$script_root/aoa_phone_bridge.c" -o "$output_dir/aoa_phone_bridge.android"
sha256sum "$output_dir/aoa_host_bridge" \
    "$output_dir/aoa_phone_bridge.android"
