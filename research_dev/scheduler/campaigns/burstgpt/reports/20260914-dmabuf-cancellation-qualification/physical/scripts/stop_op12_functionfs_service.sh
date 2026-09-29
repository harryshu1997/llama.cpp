#!/usr/bin/env bash
set -euo pipefail

adb_serial=${ADB_SERIAL:-5ae7a43d}
remote_dir=${REMOTE_DIR:-/data/local/tmp/moe_htp_service}
remote_pid=$remote_dir/functionfs.pid
adb_command=(adb -s "$adb_serial")

pid=$("${adb_command[@]}" shell "test -f '$remote_pid' && cat '$remote_pid'" | tr -d '\r' || true)
if [[ "$pid" =~ ^[0-9]+$ ]]; then
  command=$("${adb_command[@]}" shell \
    "if [ -r /proc/$pid/cmdline ]; then tr '\0' ' ' < /proc/$pid/cmdline; fi" \
    | tr -d '\r' || true)
  if [[ "$command" == *op12_htp_service* ]]; then
    "${adb_command[@]}" shell kill "$pid" || true
  fi
fi
