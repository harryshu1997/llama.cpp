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
read -r -a m_values <<< "${S41_M_VALUES:-1 2 4 8}"
read -r -a row_values <<< "${S41_LM_ROWS:-32768}"

for m in "${m_values[@]}"; do
    for rows in "${row_values[@]}"; do
        "$bench" "$backend" q6_K 3840 "$rows" "$m" 1 \
            "$warmup" "$iterations" "$threads" "$io"
    done
done
