#!/usr/bin/env bash
set -euo pipefail

ADB_SERIAL=${ADB_SERIAL:-3C15AU002CL00000}
ADB_PORT=${ADB_PORT:-5037}
PHONE_PORT=${PHONE_PORT:-27183}
PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
PHONE_BANK=${PHONE_BANK:-$PROJECT_ROOT/data/qwen35_layer20_phone_bank.fp16}
PHONE_EXPERTS=${PHONE_EXPERTS:-16}
PHONE_LAYERS=${PHONE_LAYERS:-1}
ADB=(adb -P "$ADB_PORT" -s "$ADB_SERIAL")

"${ADB[@]}" get-state
"${ADB[@]}" push "$PROJECT_ROOT/build/op15_expert_service" /data/local/tmp/op15_expert_service
"${ADB[@]}" shell chmod 755 /data/local/tmp/op15_expert_service

WEIGHT_ARGUMENT=synthetic
if [[ -f "$PHONE_BANK" ]]; then
  "${ADB[@]}" push "$PHONE_BANK" /data/local/tmp/op15_phone_banks.fp16
  WEIGHT_ARGUMENT=/data/local/tmp/op15_phone_banks.fp16
fi

OLD_PID=$("${ADB[@]}" shell 'test -f /data/local/tmp/op15_expert_service.pid && cat /data/local/tmp/op15_expert_service.pid' | tr -d '\r' || true)
if [[ "$OLD_PID" =~ ^[0-9]+$ ]]; then
  OLD_COMMAND=$("${ADB[@]}" shell "tr '\\0' ' ' < /proc/$OLD_PID/cmdline 2>/dev/null" | tr -d '\r' || true)
  if [[ "$OLD_COMMAND" == *op15_expert_service* ]]; then
    "${ADB[@]}" shell kill "$OLD_PID" || true
  fi
fi

"${ADB[@]}" shell "rm -f /data/local/tmp/op15_expert_service.log /data/local/tmp/op15_expert_service.pid; nohup /data/local/tmp/op15_expert_service --weights $WEIGHT_ARGUMENT --experts $PHONE_EXPERTS --layers $PHONE_LAYERS --port $PHONE_PORT --warmups 3 >/data/local/tmp/op15_expert_service.log 2>&1 </dev/null & echo \$! >/data/local/tmp/op15_expert_service.pid"
adb -P "$ADB_PORT" forward "tcp:$PHONE_PORT" "tcp:$PHONE_PORT"

for _ in $(seq 1 60); do
  LOG=$("${ADB[@]}" shell cat /data/local/tmp/op15_expert_service.log 2>/dev/null | tr -d '\r' || true)
  if grep -q 'READY' <<<"$LOG"; then
    printf '%s\n' "$LOG"
    exit 0
  fi
  if grep -q 'FATAL=' <<<"$LOG"; then
    printf '%s\n' "$LOG" >&2
    exit 1
  fi
  sleep 1
done
"${ADB[@]}" shell cat /data/local/tmp/op15_expert_service.log >&2 || true
exit 1
