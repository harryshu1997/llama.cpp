#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
adb_serial=${ADB_SERIAL:-5ae7a43d}
phone_port=${PHONE_PORT:-27184}
service_binary=${SERVICE_BINARY:-$project_root/build/op12_htp_service}
llama_build=${LLAMA_BUILD:-$project_root/third_party/llama.cpp/build-android-htp}
remote_dir=${REMOTE_DIR:-/data/local/tmp/moe_htp_service}
remote_bank=${REMOTE_BANK:-/data/local/tmp/qwen35_six_layer_phone_bank.fp16}
remote_log=$remote_dir/service.log
remote_pid=$remote_dir/service.pid
completion_poll=${HTP_COMPLETION_POLL:-0}
cpu_mask=${PHONE_CPU_MASK:-}
usb_listen_address=${PHONE_USB_LISTEN_ADDRESS:-}
usb_arguments=
if [[ -n "$usb_listen_address" ]]; then
  if [[ ! "$usb_listen_address" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "PHONE_USB_LISTEN_ADDRESS must be an IPv4 literal" >&2
    exit 1
  fi
  # The service additionally verifies that this is exactly rndis0's assigned address.
  usb_arguments="--usb-listen-address $usb_listen_address"
fi
if [[ "$completion_poll" != 0 && "$completion_poll" != 1 ]]; then
  echo "HTP_COMPLETION_POLL must be 0 or 1" >&2
  exit 1
fi
cpu_prefix=
if [[ -n "$cpu_mask" ]]; then
  if [[ ! "$cpu_mask" =~ ^[[:xdigit:]]+$ || "$cpu_mask" =~ ^0+$ ]]; then
    echo "PHONE_CPU_MASK must be a nonzero hexadecimal affinity mask" >&2
    exit 1
  fi
  cpu_prefix="taskset $cpu_mask"
fi
adb_command=(adb -s "$adb_serial")

"${adb_command[@]}" get-state >/dev/null
phone_model=$("${adb_command[@]}" shell getprop ro.product.model | tr -d '\r')
if [[ "$phone_model" != CPH2583 ]]; then
  echo "Expected OnePlus 12 CPH2583, found: $phone_model" >&2
  exit 1
fi
"${adb_command[@]}" shell "test -f '$remote_bank'"
"${adb_command[@]}" shell "mkdir -p '$remote_dir'"

old_pid=$("${adb_command[@]}" shell "test -f '$remote_pid' && cat '$remote_pid'" | tr -d '\r' || true)
if [[ "$old_pid" =~ ^[0-9]+$ ]]; then
  old_command=$("${adb_command[@]}" shell \
    "if [ -r /proc/$old_pid/cmdline ]; then tr '\0' ' ' < /proc/$old_pid/cmdline; fi" \
    | tr -d '\r' || true)
  if [[ "$old_command" == *op12_htp_service* ]]; then
    "${adb_command[@]}" shell kill "$old_pid" || true
  fi
fi

"${adb_command[@]}" push \
  "$service_binary" \
  "$llama_build/bin/libggml-base.so" \
  "$llama_build/bin/libggml-cpu.so" \
  "$llama_build/bin/libggml-hexagon.so" \
  "$llama_build/bin/libggml.so" \
  "$llama_build/ggml/src/ggml-hexagon/libggml-htp-v75.so" \
  "$remote_dir/" >/dev/null
"${adb_command[@]}" shell "chmod 755 '$remote_dir'/op12_htp_service '$remote_dir'/*.so"
"${adb_command[@]}" shell "
  rm -f '$remote_log' '$remote_pid'
  cd '$remote_dir'
  nohup env LD_LIBRARY_PATH='$remote_dir' ADSP_LIBRARY_PATH='$remote_dir' \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL='$completion_poll' \
    $cpu_prefix ./op12_htp_service --weights '$remote_bank' --experts 16 --layers 6 \
    --port '$phone_port' --warmups 2 $usb_arguments >'$remote_log' 2>&1 </dev/null &
  echo \$! >'$remote_pid'
"
adb -s "$adb_serial" forward "tcp:$phone_port" "tcp:$phone_port" >/dev/null
printf 'PHONE_SETTINGS completion_poll=%s cpu_mask=%s\n' "$completion_poll" "${cpu_mask:-inherited}"

for _ in $(seq 1 180); do
  service_log=$("${adb_command[@]}" shell "cat '$remote_log' 2>/dev/null" | tr -d '\r' || true)
  if grep -q 'READY' <<<"$service_log"; then
    printf '%s\n' "$service_log"
    exit 0
  fi
  current_pid=$("${adb_command[@]}" shell "test -f '$remote_pid' && cat '$remote_pid'" | tr -d '\r' || true)
  process_alive=false
  if [[ "$current_pid" =~ ^[0-9]+$ ]]; then
    process_command=$("${adb_command[@]}" shell \
      "if [ -r /proc/$current_pid/cmdline ]; then tr '\0' ' ' < /proc/$current_pid/cmdline; fi" \
      | tr -d '\r' || true)
    [[ "$process_command" == *op12_htp_service* ]] && process_alive=true
  fi
  if grep -qE 'FATAL=|dspqueue_read failed|dspqueue_write failed' <<<"$service_log" ||
      [[ "$process_alive" != true ]]; then
    printf '%s\n' "$service_log" >&2
    exit 1
  fi
  sleep 1
done

"${adb_command[@]}" shell "cat '$remote_log'" >&2 || true
exit 1
