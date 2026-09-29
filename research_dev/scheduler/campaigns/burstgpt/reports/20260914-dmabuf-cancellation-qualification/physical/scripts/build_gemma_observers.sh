#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${GEMMA_OBSERVER_BUILD:-$task_root/build/gemma-observers-retest}
relative_output=$(realpath --relative-to="$task_root" -m "$output")
if [[ -e "$output" || "$relative_output" == ../* || "$relative_output" == . ]]; then
    echo "Choose a fresh in-repo observer build directory." >&2
    exit 2
fi
mkdir -p "$output"
"${CXX:-c++}" -O2 -std=c++17 "$task_root/src/gemma_process_snapshot.cpp" -o "$output/gemma_process_snapshot_host"
docker run --rm --pull=never --network=none --volume "$task_root:/workspace" --workdir /workspace \
    --platform linux/amd64 -u "$(id -u):$(id -g)" \
    "${TOOLCHAIN_IMAGE:-snapdragon-toolchain-hostgcc:v0.3}" \
    /opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ \
    -O2 -std=c++17 -fPIE -pie -static-libstdc++ src/gemma_process_snapshot.cpp \
    -o "/workspace/$relative_output/gemma_process_snapshot_android"
