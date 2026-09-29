#!/bin/sh
set -eu

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_root/../../../../.." && pwd)
ndk_root=${ANDROID_NDK_ROOT:-/home/myid/zs89458/android/android-ndk-r27c}
android_cxx=${ANDROID_CXX:-$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++}
host_cxx=${HOST_CXX:-g++}
runtime_root=${S41_FFN_RUNTIME_ROOT:-$repo_root/build-ffn-overlap-android/bin}
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

"$android_cxx" -std=c++20 -O3 -DNDEBUG -DANDROID -fPIE \
    -static-libstdc++ -Wall -Wextra -Wno-unused-function \
    -I"$repo_root/ggml/src" -I"$repo_root/ggml/include" \
    "$repo_root/examples/layersplit/ffn-split-worker.cpp" \
    -L"$runtime_root" -Wl,--no-undefined -Wl,--gc-sections \
    -lggml -lggml-cpu -lggml-opencl -lggml-hexagon -lggml-base \
    -latomic -ldl -lm -pthread \
    -o "$script_root/llama-ffn-split-worker.android"

"$host_cxx" -std=c++17 -O3 -Wall -Wextra -Werror \
    -I"$repo_root/examples/layersplit" \
    "$script_root/ffn_dmabuf_bridge.cpp" \
    "$repo_root/examples/layersplit/ffn-split-usb-client.cpp" \
    $host_libusb \
    -o "$script_root/ffn_dmabuf_bridge"
