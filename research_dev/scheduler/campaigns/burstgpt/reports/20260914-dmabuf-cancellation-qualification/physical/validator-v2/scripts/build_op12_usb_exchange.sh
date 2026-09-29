#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$project_root/build"
g++ -std=c++17 -O3 -Wall -Wextra -fPIC -shared \
    -I"$project_root/third_party/llama.cpp/src" \
    "$project_root/src/op12_usb_exchange.cpp" \
    -o "$project_root/build/libop12_usb_exchange.so"
