#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
CMAKE=/mnt/storage/s21_deps/cmake-4.2.3-linux-x86_64/bin/cmake
"$CMAKE" -S "$TASK_ROOT/native-source" -B "$TASK_ROOT/cuda-build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/mnt/storage/s21_deps/cuda-13.2.1/bin/nvcc \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_MTMD=OFF \
    -DS41_SERVER_FFN_SPLIT=ON
env LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib \
    "$CMAKE" --build "$TASK_ROOT/cuda-build" --target llama-server llama-ffn-remote-resident-probe -j 4
