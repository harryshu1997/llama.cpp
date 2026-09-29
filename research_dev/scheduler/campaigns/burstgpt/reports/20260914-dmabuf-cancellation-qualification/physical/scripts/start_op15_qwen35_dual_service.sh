#!/usr/bin/env bash
set -euo pipefail

ADB_SERIAL=${ADB_SERIAL:-3C15AU002CL00000}
ADB_PORT=${ADB_PORT:-5037}
PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
PHONE_BANK_LOW=${PHONE_BANK_LOW:-$PROJECT_ROOT/data/qwen35_all40_phone_banks.low.fp16}
PHONE_BANK_HIGH=${PHONE_BANK_HIGH:-$PROJECT_ROOT/data/qwen35_all40_phone_banks.high.fp16}
PHONE_PORT_LOW=${PHONE_PORT_LOW:-27183}
PHONE_PORT_HIGH=${PHONE_PORT_HIGH:-27184}
PHONE_EXPERTS=${PHONE_EXPERTS:-16}
PHONE_LAYERS_PER_SHARD=${PHONE_LAYERS_PER_SHARD:-20}
PHONE_ALLOW_SYNTHETIC=${PHONE_ALLOW_SYNTHETIC:-0}
ADB=(adb -P "$ADB_PORT" -s "$ADB_SERIAL")

"${ADB[@]}" get-state
"${ADB[@]}" push "$PROJECT_ROOT/build/op15_expert_service" /data/local/tmp/op15_expert_service
"${ADB[@]}" shell chmod 755 /data/local/tmp/op15_expert_service

LOW_ARGUMENT=synthetic
HIGH_ARGUMENT=synthetic
if [[ -f "$PHONE_BANK_LOW" && -f "$PHONE_BANK_HIGH" ]]; then
  expected_bytes=$((PHONE_LAYERS_PER_SHARD * PHONE_EXPERTS * 6 * 1024 * 1024))
  low_bytes=$(stat -c %s "$PHONE_BANK_LOW")
  high_bytes=$(stat -c %s "$PHONE_BANK_HIGH")
  if [[ "$low_bytes" -ne "$expected_bytes" || "$high_bytes" -ne "$expected_bytes" ]]; then
    printf 'bank halves must each contain %d bytes; found %d and %d\n' \
      "$expected_bytes" "$low_bytes" "$high_bytes" >&2
    exit 1
  fi
  "${ADB[@]}" push "$PHONE_BANK_LOW" /data/local/tmp/qwen35_phone_banks_low.fp16
  "${ADB[@]}" push "$PHONE_BANK_HIGH" /data/local/tmp/qwen35_phone_banks_high.fp16
  LOW_ARGUMENT=/data/local/tmp/qwen35_phone_banks_low.fp16
  HIGH_ARGUMENT=/data/local/tmp/qwen35_phone_banks_high.fp16
elif [[ "$PHONE_ALLOW_SYNTHETIC" != 1 ]]; then
  printf 'missing real Qwen3.5 phone bank shard: %s or %s\n' \
    "$PHONE_BANK_LOW" "$PHONE_BANK_HIGH" >&2
  exit 1
fi

for pid_file in \
  /data/local/tmp/op15_expert_service.pid \
  /data/local/tmp/op15_expert_service_low.pid \
  /data/local/tmp/op15_expert_service_high.pid; do
  old_pid=$("${ADB[@]}" shell "test -f $pid_file && cat $pid_file" | tr -d '\r' || true)
  if [[ "$old_pid" =~ ^[0-9]+$ ]]; then
    old_command=$("${ADB[@]}" shell "tr '\0' ' ' < /proc/$old_pid/cmdline 2>/dev/null" | tr -d '\r' || true)
    if [[ "$old_command" == *op15_expert_service* ]]; then
      "${ADB[@]}" shell kill "$old_pid" || true
    fi
  fi
done

"${ADB[@]}" shell "rm -f /data/local/tmp/op15_expert_service_low.log /data/local/tmp/op15_expert_service_low.pid; nohup /data/local/tmp/op15_expert_service --weights $LOW_ARGUMENT --experts $PHONE_EXPERTS --layers $PHONE_LAYERS_PER_SHARD --port $PHONE_PORT_LOW --warmups 3 >/data/local/tmp/op15_expert_service_low.log 2>&1 </dev/null & echo \$! >/data/local/tmp/op15_expert_service_low.pid"
"${ADB[@]}" shell "rm -f /data/local/tmp/op15_expert_service_high.log /data/local/tmp/op15_expert_service_high.pid; nohup /data/local/tmp/op15_expert_service --weights $HIGH_ARGUMENT --experts $PHONE_EXPERTS --layers $PHONE_LAYERS_PER_SHARD --port $PHONE_PORT_HIGH --warmups 3 >/data/local/tmp/op15_expert_service_high.log 2>&1 </dev/null & echo \$! >/data/local/tmp/op15_expert_service_high.pid"

"${ADB[@]}" forward "tcp:$PHONE_PORT_LOW" "tcp:$PHONE_PORT_LOW"
"${ADB[@]}" forward "tcp:$PHONE_PORT_HIGH" "tcp:$PHONE_PORT_HIGH"

for _ in $(seq 1 120); do
  low_log=$("${ADB[@]}" shell cat /data/local/tmp/op15_expert_service_low.log 2>/dev/null | tr -d '\r' || true)
  high_log=$("${ADB[@]}" shell cat /data/local/tmp/op15_expert_service_high.log 2>/dev/null | tr -d '\r' || true)
  if grep -q 'FATAL=' <<<"$low_log" || grep -q 'FATAL=' <<<"$high_log"; then
    printf '%s\n%s\n' "$low_log" "$high_log" >&2
    exit 1
  fi
  if grep -q 'READY' <<<"$low_log" && grep -q 'READY' <<<"$high_log"; then
    printf '%s\n%s\n' "$low_log" "$high_log"
    exit 0
  fi
  sleep 1
done

"${ADB[@]}" shell cat /data/local/tmp/op15_expert_service_low.log >&2 || true
"${ADB[@]}" shell cat /data/local/tmp/op15_expert_service_high.log >&2 || true
exit 1
