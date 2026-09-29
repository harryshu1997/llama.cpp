#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
llama_build=${LLAMA_BUILD:-$project_root/third_party/llama.cpp/build-android-htp}
toolchain_image=${TOOLCHAIN_IMAGE:-snapdragon-toolchain-hostgcc:v0.3}
output=${OUTPUT:-$project_root/build/op12_htp_benchmark}

test -f "$llama_build/bin/libggml.so"
mkdir -p "$(dirname "$output")"

docker run --rm \
  --volume "$project_root:/workspace" \
  --workdir /workspace \
  --platform linux/amd64 \
  -u "$(id -u):$(id -g)" \
  "$toolchain_image" \
  /opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ \
  -O3 -DNDEBUG -std=c++17 -fPIE -pie -static-libstdc++ \
  -I/workspace/third_party/llama.cpp/ggml/include \
  /workspace/src/op12_htp_benchmark.cpp \
  -L/workspace/third_party/llama.cpp/build-android-htp/bin \
  -Wl,-rpath,'$ORIGIN' \
  -lggml -lggml-base -lggml-hexagon -lggml-cpu -llog -ldl -lm \
  -o /workspace/build/op12_htp_benchmark

file "$output"
