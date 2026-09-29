#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
llama_build=${LLAMA_BUILD:-$project_root/third_party/llama.cpp/build-android-htp}
toolchain_image=${TOOLCHAIN_IMAGE:-snapdragon-toolchain-hostgcc:v0.3}
output=${OUTPUT:-$project_root/build/op12_htp_service}
output_relative=$(realpath --relative-to="$project_root" -m "$output")
build_relative=$(realpath --relative-to="$project_root" "$llama_build")
if [[ "$output_relative" == ../* || "$build_relative" == ../* ]]; then
  echo "OUTPUT and LLAMA_BUILD must be inside this repository" >&2
  exit 1
fi

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
  -I/workspace/third_party/llama.cpp/ggml/src \
  -I/workspace/third_party/llama.cpp/src \
  /workspace/src/op12_htp_service.cpp \
  -L"/workspace/$build_relative/bin" \
  -Wl,-rpath,'$ORIGIN' \
  -lggml -lggml-base -lggml-hexagon -lggml-cpu -llog -ldl -lm \
  -o "/workspace/$output_relative"

file "$output"
