#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
native_dir="$PWD/third_party/llama.cpp-native-8b4b3558f145"
if [[ -n "$(git -C "$native_dir" status --porcelain)" ]]; then
    echo "Native comparison requires a pristine llama.cpp checkout." >&2
    exit 1
fi
g++ -std=c++17 -O3 -Wall -Wextra -fPIC -shared -pthread \
    -I "$native_dir/ggml/include" src/native_expert_compare.cpp \
    -L "$native_dir/build/bin" -Wl,-rpath,"$native_dir/build/bin" \
    -lggml -lggml-base -lcudart -o build/libnative_expert_compare.so
