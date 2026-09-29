#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${GEMMA_USB_BIND_OUTPUT:-$task_root/build/gemma_usb_bind}
relative_output=$(realpath --relative-to="$task_root" -m "$output")
if [[ "$relative_output" == ../* || -e "$output" ]]; then
    echo "Choose a fresh in-repo output; existing helpers are preserved." >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$output")"
docker run --rm --volume "$task_root:/workspace" --workdir /workspace \
    --platform linux/amd64 -u "$(id -u):$(id -g)" \
    "${TOOLCHAIN_IMAGE:-snapdragon-toolchain-hostgcc:v0.3}" \
    /opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ \
    -O2 -std=c++17 -fPIE -pie -static-libstdc++ src/gemma_usb_bind.cpp \
    -o "/workspace/$relative_output"
