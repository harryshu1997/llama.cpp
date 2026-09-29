#!/usr/bin/env bash
set -euo pipefail
src=/project/third_party/llama.cpp
libs=/project/candidate-bundle
sdk=/opt/hexagon/6.4.0.2
cxx=/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
includes=(-I"$src/ggml/include" -I"$src/ggml/src" -I"$src/src"
    -I"$src/ggml/src/ggml-hexagon/htp" -I"$src/build-android-htp/ggml/src/ggml-hexagon"
    -I"$sdk/incs" -I"$sdk/incs/stddef" -I"$sdk/utils/examples"
    -I"$sdk/ipc/fastrpc/rpcmem/inc" -I"$sdk/ipc/fastrpc/rtld/ship/android_aarch64")
"$cxx" -std=c++17 -O3 -DNDEBUG -fPIE -pie -static-libstdc++ "${includes[@]}" \
    /project/src/direct_phone_service.cpp -L"$libs" -Wl,-rpath,'$ORIGIN' \
    -lggml -lggml-base -lggml-hexagon -lggml-cpu -llog -ldl -lm \
    -o "$libs/direct_phone_service"
install -m 644 /project/scripts/op15_gemma_functionfs_session.sh "$libs/session.sh"
install -m 644 /project/scripts/direct_usb_lifecycle.sh "$libs/direct_usb_lifecycle.sh"
