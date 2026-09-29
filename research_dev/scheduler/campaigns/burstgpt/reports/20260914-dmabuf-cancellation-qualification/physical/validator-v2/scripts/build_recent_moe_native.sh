#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
llama_dir=${LLAMA_DIR:-$task_root/third_party/llama.cpp-native-8b4b3558f145}
cmake_bin=${CMAKE_BIN:-cmake}
cuda_compiler=${CUDA_COMPILER:-/mnt/storage/s21_deps/cuda-13.2.1/bin/nvcc}
build_jobs=${BUILD_JOBS:-8}
commit=8b4b3558f1459c13e4aa38d5c94d306a00dc6acd

if [[ ! -x "$cuda_compiler" ]]; then
    echo "Set CUDA_COMPILER to an installed CUDA nvcc; no system packages are changed." >&2
    exit 2
fi
if [[ ! -e "$llama_dir" ]]; then
    mkdir -p -- "$(dirname -- "$llama_dir")"
    git init "$llama_dir"
    git -C "$llama_dir" remote add origin https://github.com/ggml-org/llama.cpp.git
    git -C "$llama_dir" fetch --depth 1 origin "$commit"
    git -C "$llama_dir" checkout --detach FETCH_HEAD
fi
if [[ $(git -C "$llama_dir" rev-parse HEAD) != "$commit" ]] ||
   [[ -n $(git -C "$llama_dir" status --porcelain) ]]; then
    echo "Use a fresh directory or the pristine pinned checkout; existing work is preserved." >&2
    exit 2
fi

cuda_root=$(dirname -- "$(dirname -- "$cuda_compiler")")
"$cmake_bin" -S "$llama_dir" -B "$llama_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_COMPILER="$cuda_compiler" -DCUDAToolkit_ROOT="$cuda_root" \
    -DCMAKE_BUILD_RPATH="$cuda_root/lib64;$cuda_root/lib" \
    -DCMAKE_CUDA_ARCHITECTURES=89
"$cmake_bin" --build "$llama_dir/build" --target llama-server llama-bench --parallel "$build_jobs"
"$llama_dir/build/bin/llama-server" --version
