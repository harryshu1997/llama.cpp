#!/usr/bin/env bash
# Build only. Never opens USB, runs ADB, deploys, or changes firmware.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${DIRECT_BUILD_OUTPUT:-$task_root/build/direct_usb_v4}
relative=$(realpath --relative-to="$task_root" -m "$output")
if [[ "$relative" == ../* || "$relative" == . || -e "$output" ]]; then
    echo 'Choose a fresh in-repo DIRECT_BUILD_OUTPUT directory.' >&2
    exit 2
fi
mkdir -p "$output"
cxx=${CXX:-c++}
"$cxx" -std=c++17 -O3 -Wall -Wextra -Werror -I"$task_root/third_party/llama.cpp/src" \
    "$task_root/src/direct_usb_benchmark.cpp" -o "$output/direct_usb_benchmark"
"$cxx" -std=c++17 -O3 -Wall -Wextra -Werror -fPIC -shared -I"$task_root/third_party/llama.cpp/src" \
    "$task_root/src/gemma_direct_exchange.cpp" -o "$output/libgemma_direct_exchange.so"
"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror -I"$task_root/third_party/llama.cpp/src" \
    "$task_root/src/test_direct_usb.cpp" -o "$output/test_direct_usb"
install -m 644 "$task_root/scripts/op15_gemma_functionfs_session.sh" "$output/session.sh"
install -m 644 "$task_root/scripts/direct_usb_lifecycle.sh" "$output/direct_usb_lifecycle.sh"
if [[ "${DIRECT_BUILD_ANDROID:-0}" != 1 ]]; then
    echo "Host build complete: $output (no tests or devices run). Set DIRECT_BUILD_ANDROID=1 for Android compilation."
    exit 0
fi
image=${TOOLCHAIN_IMAGE:-snapdragon-toolchain-hostgcc:v0.3}
docker image inspect "$image" >/dev/null
docker run --rm --network none --volume "$task_root:/project" --workdir /project \
    --env "DIRECT_GRAPH_CACHE_ENTRIES=${DIRECT_GRAPH_CACHE_ENTRIES:-0}" --env "DIRECT_GRAPH_CACHE_MIB=${DIRECT_GRAPH_CACHE_MIB:-256}" \
    -u "$(id -u):$(id -g)" "$image" bash /project/scripts/build_direct_usb_android.sh "/project/$relative"
