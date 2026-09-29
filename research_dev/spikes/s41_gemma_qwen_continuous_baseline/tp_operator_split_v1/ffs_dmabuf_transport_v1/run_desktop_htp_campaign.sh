#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <output-root>" >&2
    exit 2
fi

output_root=$1
script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
runner=${S41_HTP_CASE_RUNNER:-$script_root/run_desktop_htp_case.sh}

run_workload() {
    workload=$1
    elements=$2
    for repetition in 1 2 3; do
        if [ $((repetition % 2)) -eq 1 ]; then
            "$runner" malloc "$elements" 50 300 \
                "$workload.malloc.r$repetition" "$output_root"
            "$runner" devmem "$elements" 50 300 \
                "$workload.devmem.r$repetition" "$output_root"
        else
            "$runner" devmem "$elements" 50 300 \
                "$workload.devmem.r$repetition" "$output_root"
            "$runner" malloc "$elements" 50 300 \
                "$workload.malloc.r$repetition" "$output_root"
        fi
    done
}

mkdir -p "$output_root"
run_workload attention_state 320
run_workload hidden_m1 2560
run_workload hidden_m8 20480
