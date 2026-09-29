#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/home/myid/zs89458/Documents/moe-resident-routing-a6000}
adb_serial=${ADB_SERIAL:-5ae7a43d}
phone_port=${PHONE_PORT:-27183}
service_binary=${SERVICE_BINARY:-$project_root/build/op15_expert_service}
phone_bank=${PHONE_BANK:-/mnt/data_s4t/zs89458-moe-routing/llamacpp_op12/qwen35_six_layer_phone_bank.fp16}
expected_bank_sha256=ad4f92d9d72c7ce85d39bb27e4e93ad0302ec7ab0b9dcdff27b5a1776576e263
remote_binary=/data/local/tmp/qwen35_op12_expert_service
remote_bank=/data/local/tmp/qwen35_six_layer_phone_bank.fp16
remote_log=/data/local/tmp/qwen35_op12_expert_service.log
remote_pid=/data/local/tmp/qwen35_op12_expert_service.pid
adb_command=(adb -s "$adb_serial")

"${adb_command[@]}" get-state
phone_model=$("${adb_command[@]}" shell getprop ro.product.model | tr -d '\r')
if [[ "$phone_model" != CPH2583 ]]; then
  echo "Expected OnePlus 12 CPH2583, found: $phone_model" >&2
  exit 1
fi

actual_bank_sha256=$(sha256sum "$phone_bank" | awk '{print $1}')
if [[ "$actual_bank_sha256" != "$expected_bank_sha256" ]]; then
  echo "Phone bank SHA-256 mismatch: $actual_bank_sha256" >&2
  exit 1
fi

"${adb_command[@]}" push "$service_binary" "$remote_binary"
"${adb_command[@]}" shell chmod 755 "$remote_binary"

remote_bank_sha256=$("${adb_command[@]}" shell "sha256sum '$remote_bank' 2>/dev/null" | awk '{print $1}' | tr -d '\r' || true)
if [[ "$remote_bank_sha256" != "$expected_bank_sha256" ]]; then
  "${adb_command[@]}" push "$phone_bank" "$remote_bank"
fi

old_pid=$("${adb_command[@]}" shell "test -f '$remote_pid' && cat '$remote_pid'" | tr -d '\r' || true)
if [[ "$old_pid" =~ ^[0-9]+$ ]]; then
  old_command=$("${adb_command[@]}" shell "tr '\0' ' ' < /proc/$old_pid/cmdline 2>/dev/null" | tr -d '\r' || true)
  if [[ "$old_command" == *qwen35_op12_expert_service* ]]; then
    "${adb_command[@]}" shell kill "$old_pid" || true
  fi
fi

# Stop the earlier copy of this same six-layer OP12 service if it owns the
# benchmark port under the repository's legacy binary name.
legacy_pid=$("${adb_command[@]}" shell pidof op15_expert_service | tr -d '\r' || true)
if [[ "$legacy_pid" =~ ^[0-9]+$ ]]; then
  legacy_command=$("${adb_command[@]}" shell "tr '\0' ' ' < /proc/$legacy_pid/cmdline 2>/dev/null" | tr -d '\r' || true)
  if [[ "$legacy_command" == *"--experts 16 --layers 6 --port $phone_port"* ]]; then
    "${adb_command[@]}" shell kill "$legacy_pid" || true
  fi
fi

"${adb_command[@]}" shell "rm -f '$remote_log' '$remote_pid'; nohup '$remote_binary' --weights '$remote_bank' --experts 16 --layers 6 --port '$phone_port' --warmups 3 >'$remote_log' 2>&1 </dev/null & echo \$! >'$remote_pid'"
adb -s "$adb_serial" forward "tcp:$phone_port" "tcp:$phone_port"

for _ in $(seq 1 120); do
  service_log=$("${adb_command[@]}" shell "cat '$remote_log' 2>/dev/null" | tr -d '\r' || true)
  if grep -q 'READY' <<<"$service_log"; then
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
