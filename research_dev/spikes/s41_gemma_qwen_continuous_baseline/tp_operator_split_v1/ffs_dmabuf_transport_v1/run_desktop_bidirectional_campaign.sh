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
    repetition=$2
    case "$variant" in
        h2d_1m)
            request_bytes=1048576
            response_bytes=64
            host_mode=async
            warmup=20
            iterations=200
            depth=4
            suffix=dmabuf_devmem_q4
            ;;
        d2h_1m)
            request_bytes=64
            response_bytes=1048576
            host_mode=async
            warmup=20
            iterations=200
            depth=4
            suffix=dmabuf_devmem_q4
            ;;
        duplex_1m)
            request_bytes=1048576
            response_bytes=1048576
            host_mode=async
            warmup=20
            iterations=200
            depth=4
            suffix=dmabuf_devmem_q4
            ;;
        h2d_4m)
            request_bytes=4194304
            response_bytes=64
            host_mode=async
            warmup=10
            iterations=100
            depth=3
            suffix=dmabuf_devmem_q3
            ;;
        d2h_4m)
            request_bytes=64
            response_bytes=4194304
            host_mode=async
            warmup=10
            iterations=100
            depth=3
            suffix=dmabuf_devmem_q3
            ;;
        h2d_15m)
            request_bytes=15728640
            response_bytes=64
            host_mode=sync
            warmup=3
            iterations=30
            depth=1
            suffix=dmabuf_devmem
            ;;
        d2h_15m)
            request_bytes=64
            response_bytes=15728640
            host_mode=sync
            warmup=3
            iterations=30
            depth=1
            suffix=dmabuf_devmem
            ;;
        *)
            echo "invalid variant" >&2
            exit 2
            ;;
    esac
    "$runner" dmabuf "$host_mode" devmem "$request_bytes" "$response_bytes" \
        "$warmup" "$iterations" "$depth" \
        "$variant.$suffix.r$repetition" "$output_root"
}

run_group() {
    repetition=$1
    shift
    variants=("$@")
    variant_count=${#variants[@]}
    offset=$((repetition - 1))
    for ((position = 0; position < variant_count; position++)); do
        index=$(((position + offset) % variant_count))
        run_variant "${variants[$index]}" "$repetition"
    done
}

mkdir -p "$output_root"
for repetition in 1 2 3; do
    run_group "$repetition" d2h_1m duplex_1m h2d_1m
    run_group "$repetition" d2h_4m h2d_4m
    run_group "$repetition" d2h_15m h2d_15m
done
