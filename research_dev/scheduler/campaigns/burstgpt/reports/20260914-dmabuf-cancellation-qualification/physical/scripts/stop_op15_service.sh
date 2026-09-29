#!/usr/bin/env bash
set -euo pipefail

ADB_SERIAL=${ADB_SERIAL:-3C15AU002CL00000}
ADB_PORT=${ADB_PORT:-5037}
PHONE_PORT=${PHONE_PORT:-27183}
ADB=(adb -P "$ADB_PORT" -s "$ADB_SERIAL")
PID=$("${ADB[@]}" shell 'test -f /data/local/tmp/op15_expert_service.pid && cat /data/local/tmp/op15_expert_service.pid' | tr -d '\r' || true)
if [[ "$PID" =~ ^[0-9]+$ ]]; then
  COMMAND=$("${ADB[@]}" shell "tr '\\0' ' ' < /proc/$PID/cmdline 2>/dev/null" | tr -d '\r' || true)
  if [[ "$COMMAND" == *op15_expert_service* ]]; then
    "${ADB[@]}" shell kill "$PID"
  fi
fi
adb -P "$ADB_PORT" forward --remove "tcp:$PHONE_PORT" 2>/dev/null || true

