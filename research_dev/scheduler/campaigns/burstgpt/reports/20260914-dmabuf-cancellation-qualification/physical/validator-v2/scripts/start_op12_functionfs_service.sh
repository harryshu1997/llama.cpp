#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
adb_serial=${ADB_SERIAL:-5ae7a43d}
service_binary=${SERVICE_BINARY:-$project_root/build/op12_htp_service}
llama_build=${LLAMA_BUILD:-$project_root/third_party/llama.cpp/build-android-htp}
remote_dir=${REMOTE_DIR:-/data/local/tmp/moe_htp_service}
remote_bank=${REMOTE_BANK:-/data/local/tmp/qwen35_six_layer_phone_bank.fp16}
ffs_dir=${FFS_DIR:-/dev/usb-ffs/moe}
dma_heap=${DMA_HEAP:-/dev/dma_heap/system}
remote_log=$remote_dir/functionfs.log
remote_pid=$remote_dir/functionfs.pid
adb_command=(adb -s "$adb_serial")

"$project_root/scripts/check_op12_functionfs.sh"
PHONE_PORT=${PHONE_PORT:-27183} "$project_root/scripts/stop_op12_htp_service.sh"
"${adb_command[@]}" shell "test -f '$remote_bank'"
"${adb_command[@]}" shell "mkdir -p '$remote_dir'"
"${adb_command[@]}" push \
  "$service_binary" \
  "$llama_build/bin/libggml-base.so" \
  "$llama_build/bin/libggml-cpu.so" \
  "$llama_build/bin/libggml-hexagon.so" \
  "$llama_build/bin/libggml.so" \
  "$llama_build/ggml/src/ggml-hexagon/libggml-htp-v75.so" \
  "$remote_dir/" >/dev/null
"${adb_command[@]}" shell "chmod 755 '$remote_dir'/op12_htp_service '$remote_dir'/*.so"

fallback_argument=
if [[ ${ALLOW_COPY_FALLBACK:-0} == 1 ]]; then
  fallback_argument=--allow-copy-fallback
fi
old_pid=$("${adb_command[@]}" shell "test -f '$remote_pid' && cat '$remote_pid'" | tr -d '\r' || true)
if [[ "$old_pid" =~ ^[0-9]+$ ]]; then
  old_command=$("${adb_command[@]}" shell \
    "if [ -r /proc/$old_pid/cmdline ]; then tr '\0' ' ' < /proc/$old_pid/cmdline; fi" \
    | tr -d '\r' || true)
  if [[ "$old_command" == *op12_htp_service* ]]; then
    "${adb_command[@]}" shell kill "$old_pid" || true
  fi
fi
"${adb_command[@]}" shell "
  rm -f '$remote_log' '$remote_pid'
  cd '$remote_dir'
  nohup env LD_LIBRARY_PATH='$remote_dir' ADSP_LIBRARY_PATH='$remote_dir' \
    GGML_HEXAGON_DEVICES=HTP0:0 \
    ./op12_htp_service --weights '$remote_bank' --experts 16 --layers 6 \
    --transport functionfs --ffs-dir '$ffs_dir' --dma-heap '$dma_heap' \
    $fallback_argument >'$remote_log' 2>&1 </dev/null &
  echo \$! >'$remote_pid'
"

for _ in $(seq 1 180); do
  service_log=$("${adb_command[@]}" shell "cat '$remote_log' 2>/dev/null" | tr -d '\r' || true)
  if grep -q 'READY transport=functionfs' <<<"$service_log"; then
    printf '%s\n' "$service_log"
    exit 0
  fi
  if grep -q 'FATAL=' <<<"$service_log"; then
    printf '%s\n' "$service_log" >&2
    exit 1
  fi
  sleep 1
done

"${adb_command[@]}" shell "cat '$remote_log'" >&2 || true
exit 1
