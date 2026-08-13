#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != /* || -e $1 ]]; then
    echo "usage: $0 <absolute-new-output-directory>" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(cd -- "$here/../../.." && pwd)
host_cxx=${HOST_CXX:-g++}
output=$1

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
    "$here/phone_arbiter_bridge.cpp" "${libusb_flags[@]}" \
    -o "$output/phone_arbiter_bridge-v1"
"$host_cxx" -std=c++17 -O3 -Wall -Wextra -Werror -pthread \
    -I"$repo_root/examples/layersplit" \
    "$here/phone_arbiter_probe.cpp" \
    -o "$output/phone_arbiter_probe-v1"
sha256sum \
    "$here/phone_arbiter_bridge.cpp" \
    "$here/phone_arbiter_probe.cpp" \
    "$output/phone_arbiter_bridge-v1" \
    "$output/phone_arbiter_probe-v1" \
    >"$output/SHA256SUMS.txt"
cat "$output/SHA256SUMS.txt"
