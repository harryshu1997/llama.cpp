#!/usr/bin/env bash
set -euo pipefail

ADB_SERIAL=${ADB_SERIAL:-3C15AU002CL00000}
ADB_PORT=${ADB_PORT:-5037}
PHONE_PORT=${PHONE_PORT:-27184}
PROJECT_ROOT=${PROJECT_ROOT:-/home/zhihao/moe-resident-routing-4060ti-op15}
PHONE_BANK=${PHONE_BANK:-$PROJECT_ROOT/data/olmoe_layer8_phone_bank.fp16}
PHONE_EXPERTS=${PHONE_EXPERTS:-16}
SERVICE_BINARY=${SERVICE_BINARY:-$PROJECT_ROOT/build/op15_expert_service_olmoe}
REMOTE_BINARY=/data/local/tmp/op15_expert_service_olmoe
REMOTE_BANK=/data/local/tmp/olmoe_layer8_phone_bank.fp16
PID_FILE=/data/local/tmp/op15_expert_service_olmoe.pid
LOG_FILE=/data/local/tmp/op15_expert_service_olmoe.log
ADB=(adb -P "$ADB_PORT" -s "$ADB_SERIAL")

if [[ ! -x "$SERVICE_BINARY" ]]; then
  echo "missing OLMoE service binary: $SERVICE_BINARY" >&2
  exit 1
fi
if [[ ! -f "$PHONE_BANK" ]]; then
  echo "missing OLMoE phone bank: $PHONE_BANK" >&2
  exit 1
fi

"${ADB[@]}" get-state
"${ADB[@]}" push "$SERVICE_BINARY" "$REMOTE_BINARY"
"${ADB[@]}" shell chmod 755 "$REMOTE_BINARY"
"${ADB[@]}" push "$PHONE_BANK" "$REMOTE_BANK"

OLD_PID=$("${ADB[@]}" shell "test -f $PID_FILE && cat $PID_FILE" | tr -d '\r' || true)
if [[ "$OLD_PID" =~ ^[0-9]+$ ]]; then
  OLD_COMMAND=$("${ADB[@]}" shell "tr '\0' ' ' < /proc/$OLD_PID/cmdline 2>/dev/null" | tr -d '\r' || true)
  if [[ "$OLD_COMMAND" == *op15_expert_service_olmoe* ]]; then
    "${ADB[@]}" shell kill "$OLD_PID" || true
  fi
fi

"${ADB[@]}" shell "rm -f $LOG_FILE $PID_FILE; nohup $REMOTE_BINARY --weights $REMOTE_BANK --experts $PHONE_EXPERTS --port $PHONE_PORT --warmups 3 >$LOG_FILE 2>&1 </dev/null & echo \$! >$PID_FILE"
adb -P "$ADB_PORT" forward "tcp:$PHONE_PORT" "tcp:$PHONE_PORT"

for _ in $(seq 1 90); do
  LOG=$("${ADB[@]}" shell cat "$LOG_FILE" 2>/dev/null | tr -d '\r' || true)
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
"${ADB[@]}" shell cat "$LOG_FILE" >&2 || true
exit 1
