#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
android_runtime=${S42_ANDROID_RUNTIME:-$repo_root/build-q6k-android/bin}
ffn_runtime=${S42_FFN_ANDROID_RUNTIME:-$repo_root/build-ffn-overlap-android/bin}
ndk_root=${S42_ANDROID_NDK:-/home/myid/zs89458/android/android-ndk-r27c}
android_cxx=$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++

mkdir -p "$script_dir/bin"

g++ -std=c++17 -O3 -DNDEBUG \
    -I"$repo_root/ggml/include" \
    -I"$repo_root/ggml/src" \
    "$script_dir/ffn_shape_bench.cpp" \
    -L"$repo_root/build-cpu/bin" \
    -Wl,-rpath,"$repo_root/build-cpu/bin" \
    -lggml -lggml-cpu -lggml-base -pthread -ldl -lm \
    -o "$script_dir/bin/ffn_shape_bench.host"

"$android_cxx" -std=c++17 -O2 -DNDEBUG -DANDROID \
    -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_HEXAGON \
    -I"$repo_root/ggml/include" \
    -I"$repo_root/ggml/src" \
    "$script_dir/multi_session_residency_probe.cpp" \
    -L"$android_runtime" \
    -Wl,-rpath,/data/local/tmp/s42-multi-session-v1 \
    -static-libstdc++ -Wl,--no-undefined -Wl,--gc-sections \
    -lggml -lggml-cpu -lggml-hexagon -lggml-base \
    -pthread -latomic -ldl -lm \
    -o "$script_dir/bin/multi_session_residency_probe.android"

"$android_cxx" -std=c++17 -O3 -DNDEBUG -DANDROID \
    -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_HEXAGON \
    -I"$repo_root/ggml/include" \
    -I"$repo_root/ggml/src" \
    "$script_dir/ffn_shape_bench.cpp" \
    -L"$android_runtime" \
    -Wl,-rpath,/data/local/tmp/s42-multi-session-v1 \
    -static-libstdc++ -Wl,--no-undefined -Wl,--gc-sections \
    -lggml -lggml-cpu -lggml-hexagon -lggml-base \
    -pthread -latomic -ldl -lm \
    -o "$script_dir/bin/ffn_shape_bench.android"

"$android_cxx" -std=c++17 -O3 -DNDEBUG -DANDROID \
    -I"$repo_root/examples/layersplit" \
    "$script_dir/resident_ffn_router.cpp" \
    -static-libstdc++ -pthread \
    -o "$script_dir/bin/resident_ffn_router.android"

"$android_cxx" -std=c++20 -O3 -DNDEBUG -DANDROID \
    -DFFN_SPLIT_WORKER_NO_MAIN \
    -I"$repo_root/examples/layersplit" \
    -I"$repo_root/ggml/include" -I"$repo_root/ggml/src" \
    "$repo_root/examples/layersplit/ffn-split-worker.cpp" \
    "$script_dir/resident_ffn_workers.cpp" \
    -L"$ffn_runtime" -Wl,--no-undefined -Wl,--gc-sections \
    -lggml -lggml-cpu -lggml-opencl -lggml-hexagon -lggml-base \
    -static-libstdc++ -pthread -latomic -ldl -lm \
    -o "$script_dir/bin/resident_ffn_workers.android"
