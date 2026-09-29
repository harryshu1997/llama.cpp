#!/bin/sh
set -eu

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_root/../../../../.." && pwd)
ndk_root=${ANDROID_NDK_ROOT:-/home/myid/zs89458/android/android-ndk-r27c}
android_cxx=${ANDROID_CXX:-$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++}
runtime_root=${S41_FFN_RUNTIME_ROOT:-$repo_root/build-ffn-overlap-android/bin}

"$android_cxx" -std=c++20 -O3 -DNDEBUG -DANDROID -fPIE \
    -static-libstdc++ -Wall -Wextra -Werror \
    -I"$script_root/.." -I"$repo_root/ggml/src" -I"$repo_root/ggml/include" \
    "$script_root/dual_backend_ffn.cpp" \
    -L"$runtime_root" -Wl,--no-undefined -Wl,--gc-sections \
    -lggml -lggml-cpu -lggml-opencl -lggml-hexagon -lggml-base \
    -latomic -ldl -lm -pthread \
    -o "$script_root/dual_backend_ffn.android"
