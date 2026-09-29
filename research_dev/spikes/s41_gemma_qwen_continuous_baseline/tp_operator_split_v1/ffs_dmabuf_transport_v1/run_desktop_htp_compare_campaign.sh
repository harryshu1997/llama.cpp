#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <output-root>" >&2
    exit 2
fi

output_root=$1
script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
runner=${S41_HTP_CASE_RUNNER:-$script_root/run_desktop_htp_case.sh}

run_variant() {
    variant=$1
    workload=$2
    elements=$3
    warmup=$4
    iterations=$5
    repetition=$6
    case_name=$workload.$variant.r$repetition

    case "$variant" in
        staged_malloc)
            S41_HTP_DEVICE_MODE=htp-staged-sqr \
                "$runner" malloc "$elements" "$warmup" "$iterations" \
                "$case_name" "$output_root"
            ;;
        copy_malloc)
            S41_HTP_DEVICE_MODE=htp-copy-sqr \
                "$runner" malloc "$elements" "$warmup" "$iterations" \
                "$case_name" "$output_root"
            ;;
        dmabuf_malloc)
            S41_HTP_DEVICE_MODE=htp-sqr \
                "$runner" malloc "$elements" "$warmup" "$iterations" \
                "$case_name" "$output_root"
            ;;
        dmabuf_devmem)
            S41_HTP_DEVICE_MODE=htp-sqr \
                "$runner" devmem "$elements" "$warmup" "$iterations" \
                "$case_name" "$output_root"
            ;;
        *)
            echo "invalid variant" >&2
            exit 2
            ;;
    esac
}

run_workload() {
    workload=$1
    elements=$2
    warmup=$3
    iterations=$4

    variants=(staged_malloc copy_malloc dmabuf_malloc dmabuf_devmem)
    for repetition in 1 2 3; do
        offset=$((repetition - 1))
        for position in 0 1 2 3; do
            index=$(((position + offset) % 4))
            run_variant "${variants[$index]}" "$workload" "$elements" \
                "$warmup" "$iterations" "$repetition"
        done
    done
}

mkdir -p "$output_root"
run_workload attention_state 320 50 300
run_workload hidden_m1 2560 50 300
run_workload hidden_m8 20480 50 300
run_workload one_mib 262144 20 100
