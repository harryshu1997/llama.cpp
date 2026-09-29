#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
CACHE_ROOT=${CACHE_ROOT:-/mnt/storage/moe-resident-routing-4060ti-op15-cache}
NDK_VERSION=${NDK_VERSION:-r27d}
OP15_HIDDEN=${OP15_HIDDEN:-2048}
OP15_INTERMEDIATE=${OP15_INTERMEDIATE:-512}
OUTPUT_NAME=${OUTPUT_NAME:-op15_expert_service}
NDK_ZIP="$CACHE_ROOT/downloads/android-ndk-${NDK_VERSION}-linux.zip"
NDK_ROOT="$CACHE_ROOT/build/android-ndk-${NDK_VERSION}"
NDK_URL="https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip"

mkdir -p "$CACHE_ROOT/downloads" "$CACHE_ROOT/build" "$PROJECT_ROOT/build"
if [[ ! -f "$NDK_ZIP" ]]; then
  curl -fL --retry 3 -o "$NDK_ZIP.part" "$NDK_URL"
  mv "$NDK_ZIP.part" "$NDK_ZIP"
fi
if [[ ! -x "$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++" ]]; then
  unzip -q "$NDK_ZIP" -d "$CACHE_ROOT/build"
fi

CXX="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"
"$CXX" -O3 -DNDEBUG -std=c++17 -fPIE -pie -static-libstdc++ \
  -DOP15_HIDDEN="$OP15_HIDDEN" -DOP15_INTERMEDIATE="$OP15_INTERMEDIATE" \
  "$PROJECT_ROOT/src/op15_opencl_service.cpp" \
  -ldl -o "$PROJECT_ROOT/build/$OUTPUT_NAME"
file "$PROJECT_ROOT/build/$OUTPUT_NAME"
