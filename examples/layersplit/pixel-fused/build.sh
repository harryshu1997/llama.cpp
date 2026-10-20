#!/bin/bash
# WS15 Pixel FFN worker with fused attention-projection calls.
#   build.sh host    OUT_DIR   x86 build against the repo's build-host ggml (tests, fused_numerics, host e2e)
#   build.sh android OUT_DIR   the qualified Pixel recipe (snapdragon-toolchain-hostgcc:v0.3, NDK r28b, android28)
# The android recipe is the one that rebuilds the qualified worker (pixel-cpugpu-v1, sha 5d824455...) byte for byte
# from its source; ANDROID_GGML (default: the M3 pixel10pro-vulkan tree) provides the ggml headers and the libraries
# it links against (the phone loads its own pinned libggml*.so next to the binary).
set -euo pipefail
MODE=${1:?host|android}; OUT=$(mkdir -p "$2" && cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd); LAYERSPLIT=$(dirname "$HERE"); REPO=$(cd "$LAYERSPLIT/../.." && pwd)
case "$MODE" in
  host)
    g++ -std=c++17 -O2 -DNDEBUG -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -Wall -Wextra \
        -Wno-unused-function -I "$REPO/ggml/include" -I "$REPO/ggml/src" -I "$LAYERSPLIT" -I "$HERE" \
        "$HERE/ffn-split-worker.cpp" -L "$REPO/build-host/bin" -Wl,-rpath,"$REPO/build-host/bin" \
        -lggml -lggml-cpu -lggml-base -lpthread -o "$OUT/llama-ffn-split-worker" ;;
  android)
    GGML=${ANDROID_GGML:-$REPO/research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/software/pixel10pro-vulkan}
    [ -f "$GGML/source/ggml/include/ggml.h" ] && [ -f "$GGML/build/bin/libggml-cpu.so" ] || {
        echo "ANDROID_GGML must name the Android ggml tree (source/ggml/include, build/bin): $GGML" >&2; exit 2; }
    STAGE=$(mktemp -d)
    cp "$HERE"/ffn-split-worker.cpp "$HERE"/pixel_packed_cpu.h "$HERE"/pixel_f16_layout.h \
       "$LAYERSPLIT"/ffn-split-protocol.h "$LAYERSPLIT"/ffn-split-dmabuf.h "$LAYERSPLIT"/ffn-split-functionfs.h \
       "$LAYERSPLIT"/ffn-split-worker-entry.h "$STAGE"/
    docker run --rm --user "$(id -u):$(id -g)" -v "$GGML":/work:ro -v "$STAGE":/diag snapdragon-toolchain-hostgcc:v0.3 \
        /opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++ \
        -std=c++17 -march=armv8.2-a+fp16+fp16fml -O3 -DNDEBUG -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU \
        -DGGML_USE_VULKAN -Wall -Wextra -Werror -Wno-unused-function -static-libstdc++ -I/work/source/ggml/include \
        -I/work/source/ggml/src -I/diag /diag/ffn-split-worker.cpp -L/work/build/bin '-Wl,-rpath,$ORIGIN' \
        -lggml -lggml-cpu -lggml-vulkan -lggml-base -latomic -lm -o /diag/llama-ffn-split-worker
    cp "$STAGE/llama-ffn-split-worker" "$OUT/" && rm -rf "$STAGE" ;;
  *) echo "mode must be host or android" >&2; exit 2 ;;
esac
sha256sum "$OUT/llama-ffn-split-worker"
