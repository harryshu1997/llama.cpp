#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 BENCH BACKEND THREADS IO" >&2
    exit 2
fi

bench=$1
backend=$2
threads=$3
io=$4
warmup=${S41_WARMUP:-2}
iterations=${S41_ITERATIONS:-7}
read -r -a m_values <<< "${S41_M_VALUES:-1 8 32 128 512}"

for m in "${m_values[@]}"; do
    "$bench" "$backend" q4_0 3840 4096 "$m" 1 \
        "$warmup" "$iterations" "$threads" "$io"
    "$bench" "$backend" q4_0 3840 2048 "$m" 2 \
        "$warmup" "$iterations" "$threads" "$io"
    "$bench" "$backend" q4_0 3840 2048 "$m" 4 \
        "$warmup" "$iterations" "$threads" "$io"
    "$bench" "$backend" q4_0 4096 3840 "$m" 1 \
        "$warmup" "$iterations" "$threads" "$io"
    "$bench" "$backend" q4_0 2048 3840 "$m" 1 \
        "$warmup" "$iterations" "$threads" "$io"
done
