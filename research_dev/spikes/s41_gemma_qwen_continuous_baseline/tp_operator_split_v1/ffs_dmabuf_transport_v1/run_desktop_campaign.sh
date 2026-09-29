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
    response_bytes=$4
    warmup=$5
    iterations=$6
    repetition=$7
    case_name=${workload}.${variant}.r${repetition}
    case "$variant" in
        copy_malloc)
            "$runner" copy sync malloc "$request_bytes" "$response_bytes" \
                "$warmup" "$iterations" 1 "$case_name" "$output_root"
            ;;
        dmabuf_malloc)
            "$runner" dmabuf sync malloc "$request_bytes" "$response_bytes" \
                "$warmup" "$iterations" 1 "$case_name" "$output_root"
            ;;
        dmabuf_devmem)
            "$runner" dmabuf sync devmem "$request_bytes" "$response_bytes" \
                "$warmup" "$iterations" 1 "$case_name" "$output_root"
            ;;
        *) echo "unknown variant: $variant" >&2; exit 2 ;;
    esac
}

run_workload() {
    workload=$1
    request_bytes=$2
    response_bytes=$3
    warmup=$4
    iterations=$5

    run_variant copy_malloc "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 1
    run_variant dmabuf_malloc "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 1
    run_variant dmabuf_devmem "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 1

    run_variant dmabuf_malloc "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 2
    run_variant dmabuf_devmem "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 2
    run_variant copy_malloc "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 2

    run_variant dmabuf_devmem "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 3
    run_variant copy_malloc "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 3
    run_variant dmabuf_malloc "$workload" "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" 3
}

mkdir -p "$output_root"
run_workload attention 1308 1384 50 300
run_workload hidden_m1 10268 10344 50 300
run_workload swiglu 69660 34920 50 300
run_workload hidden_m8 81948 82024 50 300
run_workload upload_1m 1048604 104 20 100
run_workload download_1m 64 1048680 20 100

for repetition in 1 2 3; do
    "$runner" dmabuf async devmem 10268 10344 50 300 4 \
        "hidden_m1.dmabuf_devmem_q4.r$repetition" "$output_root"
    "$runner" dmabuf async devmem 81948 82024 50 300 4 \
        "hidden_m8.dmabuf_devmem_q4.r$repetition" "$output_root"
done
