#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <output-root>" >&2
    exit 2
fi

output_root=$1
script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
runner=${S41_CASE_RUNNER:-$script_root/run_desktop_case.sh}

run_variant() {
    variant=$1
    workload=$2
    request_bytes=$3
    warmup=$4
    iterations=$5
    repetition=$6
    case_name=$workload.$variant.r$repetition

    case "$variant" in
        copy_malloc)
            "$runner" copy sync malloc "$request_bytes" 64 "$warmup" \
                "$iterations" 1 "$case_name" "$output_root"
            ;;
        dmabuf_malloc)
            "$runner" dmabuf sync malloc "$request_bytes" 64 "$warmup" \
                "$iterations" 1 "$case_name" "$output_root"
            ;;
        dmabuf_devmem)
            "$runner" dmabuf sync devmem "$request_bytes" 64 "$warmup" \
                "$iterations" 1 "$case_name" "$output_root"
            ;;
        dmabuf_devmem_q4)
            "$runner" dmabuf async devmem "$request_bytes" 64 "$warmup" \
                "$iterations" 4 "$case_name" "$output_root"
            ;;
        dmabuf_devmem_q3)
            "$runner" dmabuf async devmem "$request_bytes" 64 "$warmup" \
                "$iterations" 3 "$case_name" "$output_root"
            ;;
        *)
            echo "invalid variant" >&2
            exit 2
            ;;
    esac
}

run_workload() {
    workload=$1
    request_bytes=$2
    warmup=$3
    iterations=$4
    shift 4
    variants=("$@")
    variant_count=${#variants[@]}

    for repetition in 1 2 3; do
        offset=$((repetition - 1))
        for ((position = 0; position < variant_count; position++)); do
            index=$(((position + offset) % variant_count))
            run_variant "${variants[$index]}" "$workload" "$request_bytes" \
                "$warmup" "$iterations" "$repetition"
        done
    done
}

mkdir -p "$output_root"
run_workload upload_1m 1048576 20 200 \
    copy_malloc dmabuf_malloc dmabuf_devmem dmabuf_devmem_q4
run_workload upload_4m 4194304 10 100 \
    copy_malloc dmabuf_malloc dmabuf_devmem dmabuf_devmem_q3
run_workload upload_15m 15728640 3 30 \
    copy_malloc dmabuf_malloc dmabuf_devmem
