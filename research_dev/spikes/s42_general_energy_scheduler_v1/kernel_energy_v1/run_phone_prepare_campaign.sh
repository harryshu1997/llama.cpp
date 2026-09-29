#!/usr/bin/env bash

set -euo pipefail

if (( $# != 2 )); then
    echo "usage: $0 TOOL_ROOT RESULT_ROOT" >&2
    exit 2
fi

tool_root=$1
result_root=$2
runtime=/data/local/tmp/s42-kernel-energy-v1/runtime

run_case() {
    local case_id=$1
    local command=$2

    if [[ -f "$result_root/$case_id.json" ]]; then
        echo "SKIP $case_id"
        return
    fi
    if [[ -d "$result_root/$case_id.capture" ]]; then
        mv "$result_root/$case_id.capture" \
            "$result_root/$case_id.capture.invalid-interrupted"
    fi
    python3 "$tool_root/run_phone_case.py" \
        --case-id "$case_id" \
        --mode adb \
        --output "$result_root/$case_id.capture" \
        --timeout-s 300 \
        --adb-command "$command"
    python3 "$tool_root/analyze_phone_case.py" \
        --capture "$result_root/$case_id.capture" \
        --output "$result_root/$case_id.json"
    sleep 3
}

prepare_case() {
    local case_id=$1
    local benchmark=$2
    local gpu_mode=$3
    local batch=$4
    local iterations=$5
    local repetition=$6

    command="cd $runtime && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
    command+="GGML_HEXAGON_MBUF=4192 GGML_HEXAGON_NHVX=4 "
    command+="S41_DISABLE_GRAPH_CACHE=1 ./dual_backend_ffn.android "
    command+="--k 3840 --n-ff 15360 --htp-columns 9152 "
    command+="--gpu-columns 512 --batch $batch --type q4_0 "
    command+="--gpu-mode $gpu_mode --benchmark $benchmark "
    command+="--warmup 2 --iterations $iterations"
    run_case "${case_id}-r${repetition}" "$command"
}

for repetition in 1 2 3; do
    prepare_case phone-htp-q4-repack-9664 \
        repack-htp-control native 1 300 "$repetition"
    prepare_case phone-adreno-q4-upload-512 \
        repack-gpu-shard native 1 400 "$repetition"
    prepare_case phone-adreno-q4-to-f16-reconstruct-512 \
        reconstruct-gpu-shard f16-xmem 16 600 "$repetition"
    prepare_case phone-adreno-f16-upload-512 \
        repack-gpu-shard f16-xmem 16 500 "$repetition"
    prepare_case phone-adreno-f16-xmem-prepare-512 \
        prepare-gpu-xmem f16-xmem 16 60 "$repetition"
done
