#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <output-root> <adb-port>" >&2
    exit 2
fi

output=$1
adb_port=$2
if [[ $output != /* || -e $output ]]; then
    echo "output root must be an unused absolute path: $output" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
s42_root=$(cd -- "$here/../.." && pwd)
arm_runner=$s42_root/hybrid_overflow_v1/run_gpu_overflow_arm.sh
analyzer=$here/analyze_shape_balance_abba.py

mkdir -p "$output"

S42_SPLIT_POLICY_VARIANT=qualified \
    "$arm_runner" op15 1 "$output/fixed-r1" "$adb_port"
S42_SPLIT_POLICY_VARIANT=shape-balanced \
    "$arm_runner" op15 1 "$output/tuned-r1" "$adb_port"
S42_SPLIT_POLICY_VARIANT=shape-balanced \
    "$arm_runner" op15 2 "$output/tuned-r2" "$adb_port"
S42_SPLIT_POLICY_VARIANT=qualified \
    "$arm_runner" op15 2 "$output/fixed-r2" "$adb_port"

python3 "$analyzer" \
    --fixed-r1 "$output/fixed-r1" \
    --tuned-r1 "$output/tuned-r1" \
    --tuned-r2 "$output/tuned-r2" \
    --fixed-r2 "$output/fixed-r2" \
    --output "$output/SHAPE_BALANCE_ABBA.json"
