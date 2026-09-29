#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <auto|control|op15> <repeat-index> <output-root> <adb-port>" >&2
    exit 2
fi

arm=$1
repeat_index=$2
output=$3
adb_port=$4
if [[ $arm != auto && $arm != control && $arm != op15 ]]; then
    echo "invalid arm: $arm" >&2
    exit 2
fi
if [[ ! $repeat_index =~ ^[0-9]+$ ]]; then
    echo "invalid repeat index: $repeat_index" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
launcher=$here/small_model_overlay_v1/run_fp16_small_overlay_arm.sh
if [[ ! -x $launcher ]]; then
    echo "missing unified physical launcher: $launcher" >&2
    exit 1
fi

large_policy=runtime-auto
small_policy=runtime-scheduler
if [[ $arm == control ]]; then
    large_policy=cpu-overflow
    small_policy=static-cpu
fi

exec "$launcher" \
    "$output" \
    "$large_policy" \
    "$small_policy" \
    "$adb_port" \
    qualification
