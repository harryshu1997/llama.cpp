#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${GEMMA_HTP_BUILD:-$task_root/third_party/llama.cpp/build-android-htp}
output=${GEMMA_HTP_OUTPUT:-$task_root/build/gemma4_htp_service}
image=${TOOLCHAIN_IMAGE:-snapdragon-toolchain-hostgcc:v0.3}
relative_build=$(realpath --relative-to="$task_root" "$build_dir")
relative_output=$(realpath --relative-to="$task_root" -m "$output")
if [[ "$relative_build" == ../* || "$relative_output" == ../* || -e "$output" ]]; then
    echo "Use existing in-repo Android libraries and a new in-repo output filename." >&2
    exit 2
fi
test -f "$build_dir/bin/libggml-hexagon.so"
mkdir -p -- "$(dirname -- "$output")"
docker run --rm --volume "$task_root:/workspace" --workdir /workspace \
    --platform linux/amd64 -u "$(id -u):$(id -g)" "$image" \
    /opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ \
    -O3 -DNDEBUG -std=c++17 -fPIE -pie -static-libstdc++ \
    -I/workspace/third_party/llama.cpp/ggml/include \
    -I/workspace/third_party/llama.cpp/ggml/src -I/workspace/third_party/llama.cpp/src \
    /workspace/src/gemma4_htp_service.cpp -L"/workspace/$relative_build/bin" \
    -Wl,-rpath,'$ORIGIN' -lggml -lggml-base -lggml-hexagon -lggml-cpu -llog -ldl -lm \
    -o "/workspace/$relative_output"
