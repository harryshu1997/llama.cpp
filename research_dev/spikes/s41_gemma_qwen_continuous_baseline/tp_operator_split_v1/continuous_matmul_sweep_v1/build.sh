#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../../.." && pwd)
android_runtime=${S41_ANDROID_RUNTIME:-$repo_root/build-q6k-android/bin}
ndk_root=${S41_ANDROID_NDK:-/home/myid/zs89458/android/android-ndk-r27c}
android_cxx=$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++

mkdir -p "$script_dir/bin"

g++ -std=c++17 -O3 -DNDEBUG \
    -I"$repo_root/ggml/include" \
    -I"$repo_root/ggml/src" \
    "$script_dir/backend_matmul_bench.cpp" \
    -L"$repo_root/build-cpu/bin" \
    -Wl,-rpath,"$repo_root/build-cpu/bin" \
    -lggml -lggml-cpu -lggml-base -pthread -ldl -lm \
    -o "$script_dir/bin/backend_matmul_bench.host"

"$android_cxx" -std=c++17 -O3 -DNDEBUG -DANDROID \
    -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_HEXAGON \
    -I"$repo_root/ggml/include" \
    -I"$repo_root/ggml/src" \
    "$script_dir/backend_matmul_bench.cpp" \
    -L"$android_runtime" \
    -Wl,-rpath,/data/local/tmp/continuous-matmul-v1 \
    -static-libstdc++ -Wl,--no-undefined -Wl,--gc-sections \
    -lggml -lggml-cpu -lggml-hexagon -lggml-base \
    -pthread -latomic -ldl -lm \
    -o "$script_dir/bin/backend_matmul_bench.android"

