#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != /* || -e $1 ]]; then
    echo "usage: $0 <absolute-new-output-directory>" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(cd -- "$here/../../../.." && pwd)
cuda_root=${S42_CUDA_ROOT:-/mnt/storage/s21_deps/cuda-13.2.1}
host_cxx=${HOST_CXX:-g++}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
output=$1
bridge_source=$repo_root/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/ffs_dmabuf_transport_v1/ffn_dmabuf_bridge.cpp

for path in "$bridge_source" "$here/cuda_weight_prefetch_probe.cu" "$nvcc"; do
    if [[ ! -e $path ]]; then
        echo "missing build dependency: $path" >&2
        exit 1
    fi
done

if pkg-config --exists libusb-1.0 2>/dev/null; then
    read -r -a libusb_flags <<<"$(pkg-config --libs libusb-1.0)"
elif [[ -e /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 ]]; then
    libusb_flags=(-l:libusb-1.0.so.0)
else
    echo "libusb runtime is unavailable" >&2
    exit 1
fi

mkdir "$output"
"$host_cxx" -std=c++17 -O3 -Wall -Wextra -Werror \
    -I"$repo_root/examples/layersplit" \
    "$bridge_source" "${libusb_flags[@]}" \
    -o "$output/ffn_dmabuf_bridge-prefetch-v1"
"$nvcc" -std=c++17 -O3 -Xcompiler=-Wall,-Wextra,-Werror \
    "$here/cuda_weight_prefetch_probe.cu" \
    -o "$output/cuda_weight_prefetch_probe-v1"
sha256sum \
    "$bridge_source" \
    "$here/cuda_weight_prefetch_probe.cu" \
    "$output/ffn_dmabuf_bridge-prefetch-v1" \
    "$output/cuda_weight_prefetch_probe-v1" \
    >"$output/SHA256SUMS.txt"
cat "$output/SHA256SUMS.txt"
