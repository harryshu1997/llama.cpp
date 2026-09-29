#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M0-20260920-LuVndR
cmake=/mnt/storage/s21_deps/cmake-4.2.3-linux-x86_64/bin/cmake
export LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -n 9
test -f "$deploy/native-source/cmake/build-info.cmake"
test -f "$deploy/native-source/common/build-info.cpp.in"
mkdir -p "$deploy/software"
"$cmake" -S "$deploy/native-source" -B "$deploy/cuda-build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/mnt/storage/s21_deps/cuda-13.2.1/bin/nvcc \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DS41_SERVER_FFN_SPLIT=ON \
    > "$deploy/software/configure-restore.log" 2>&1
"$cmake" --build "$deploy/cuda-build" \
    --target llama-server llama-ffn-remote-resident-probe llama-ffn-split-worker llama-ffn-split-usb-close \
    -j8 > "$deploy/software/build-restore.log" 2>&1
cp "$deploy/cuda-build/CMakeCache.txt" "$deploy/software/CMakeCache-restore.txt"
sha256sum "$deploy/cuda-build/bin/llama-server" \
    "$deploy/cuda-build/bin/llama-ffn-remote-resident-probe" \
    "$deploy/cuda-build/bin/"*.so* > "$deploy/software/RUNTIME_RESTORE_SHA256.txt"
