#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${GEMMA_LLAMA_SOURCE:-$task_root/third_party/llama.cpp}
build_dir=${GEMMA_LLAMA_BUILD:-$source_dir/build-gemma-phone}
cmake_bin=${CMAKE_BIN:-cmake}
compiler=${CUDA_COMPILER:-/mnt/storage/s21_deps/cuda-13.2.1/bin/nvcc}
python_bin=${PYTHON_BIN:-python}
if [[ -e "$build_dir" || ! -x "$compiler" ]]; then
    echo "Use a fresh hybrid build directory and an existing CUDA compiler; no files are downloaded." >&2
    exit 2
fi
if [[ $(git -C "$source_dir" rev-parse HEAD) != 8b4b3558f1459c13e4aa38d5c94d306a00dc6acd ]]; then
    echo "Use the pinned experimental source checkout." >&2
    exit 2
fi
cuda_root=$(dirname -- "$(dirname -- "$compiler")")
"$cmake_bin" -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_COMPILER="$compiler" -DCUDAToolkit_ROOT="$cuda_root" \
    -DCMAKE_BUILD_RPATH="$cuda_root/lib64;$cuda_root/lib" -DCMAKE_CUDA_ARCHITECTURES=89
"$cmake_bin" --build "$build_dir" --target llama-server --parallel "${BUILD_JOBS:-8}"
cd -- "$task_root"
"${CXX:-c++}" -std=c++17 -O3 -fPIC -shared -I"$source_dir/src" \
    src/gemma_usb_exchange.cpp -o "$build_dir/bin/libgemma_usb_exchange.so"
"${CXX:-c++}" -std=c++17 -O3 -fPIC -shared -I"$source_dir/src" \
    src/gemma_direct_exchange.cpp -o "$build_dir/bin/libgemma_direct_exchange.so"
"$python_bin" -m src.gemma_phone_benchmark record-build --source "$source_dir" --build "$build_dir"
