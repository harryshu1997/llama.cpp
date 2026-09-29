#!/usr/bin/env bash
set -euo pipefail

ADB_SERIAL=${ADB_SERIAL:-3C15AU002CL00000}
ADB_PORT=${ADB_PORT:-5037}
PHONE_PORT=${PHONE_PORT:-27186}
PID_FILE=/data/local/tmp/op15_expert_service_granite4.pid
ADB=(adb -P "$ADB_PORT" -s "$ADB_SERIAL")

PID=$("${ADB[@]}" shell "test -f $PID_FILE && cat $PID_FILE" | tr -d '\r' || true)
if [[ "$PID" =~ ^[0-9]+$ ]]; then
  "${ADB[@]}" shell kill "$PID" || true
fi
"${ADB[@]}" shell rm -f "$PID_FILE"
adb -P "$ADB_PORT" forward --remove "tcp:$PHONE_PORT" 2>/dev/null || true
