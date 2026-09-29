#!/usr/bin/env bash
# Runs inside the existing local toolchain image; uses only this project's sources.
set -euo pipefail
[ "$#" = 1 ]
output=$1
case "$output" in /project/build/*) ;; *) exit 2;; esac
cache_entries=${DIRECT_GRAPH_CACHE_ENTRIES:-0}
cache_mib=${DIRECT_GRAPH_CACHE_MIB:-256}
if [[ ! "$cache_entries" =~ ^(0|[1-9][0-9]{0,2})$ || ! "$cache_mib" =~ ^[1-9][0-9]{0,3}$ ]] ||
        (( cache_entries > 256 || cache_mib < 2 || cache_mib > 1024 )); then
    echo 'Invalid direct graph cache limits (entries 0..256, MiB 2..1024).' >&2
    exit 2
fi
src=/project/third_party/llama.cpp
libs=$src/build-android-htp/bin
generated=$src/build-android-htp/ggml/src/ggml-hexagon
sdk=/opt/hexagon/6.4.0.2
ndk=/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin
cc=$ndk/aarch64-linux-android34-clang
cxx=$ndk/aarch64-linux-android34-clang++
includes=(-I"$src/ggml/include" -I"$src/ggml/src" -I"$src/src" -I"$src/ggml/src/ggml-hexagon/htp" -I"$generated"
    -I"$sdk/incs" -I"$sdk/incs/stddef" -I"$sdk/utils/examples" -I"$sdk/ipc/fastrpc/rpcmem/inc" -I"$sdk/ipc/fastrpc/rtld/ship/android_aarch64")
"$cc" -O3 -fPIC "${includes[@]}" -c "$generated/htp_iface_stub.c" -o "$output/htp_iface_stub.o"
"$cxx" -std=c++17 -O3 -DNDEBUG -fPIC -shared -static-libstdc++ \
    -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_SCHED_MAX_COPIES=4 \
    "${includes[@]}" "$src/ggml/src/ggml-hexagon/ggml-hexagon.cpp" "$src/ggml/src/ggml-hexagon/htp-drv.cpp" \
    "$output/htp_iface_stub.o" -L"$libs" -lggml-base -ldl -llog -lm \
    -Wl,-soname,libggml-hexagon.so -Wl,-rpath,'$ORIGIN' -o "$output/libggml-hexagon.so"
"$cxx" -std=c++17 -O3 -DNDEBUG -fPIE -pie -static-libstdc++ "${includes[@]}" \
    -DGEMMA_DIRECT_GRAPH_CACHE_ENTRIES="$cache_entries" -DGEMMA_DIRECT_GRAPH_CACHE_MIB="$cache_mib" \
    /project/src/direct_phone_service.cpp -L"$output" -L"$libs" \
    -Wl,-rpath,'$ORIGIN' -lggml -lggml-base -lggml-hexagon -lggml-cpu -llog -ldl -lm \
    -o "$output/direct_phone_service"
"$cxx" -std=c++17 -O3 -DNDEBUG -fPIE -pie -static-libstdc++ \
    /project/src/gemma_usb_bind.cpp -o "$output/gemma_usb_bind"
for library in libggml.so libggml-base.so libggml-cpu.so; do
    install -m 644 "$libs/$library" "$output/$library"
done
install -m 644 "$generated/libggml-htp-v81.so" "$output/libggml-htp-v81.so"
echo "Android service and own shared-DMA backend built. Nothing deployed or executed on a phone."
echo "Graph cache: entries=$cache_entries (0=layer-count policy), retained MiB per slot=$cache_mib."
