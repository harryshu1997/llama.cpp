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

kernel_case() {
    local backend=$1
    local batch=$2
    local repetition=$3
    local iterations
    local benchmark
    local gpu_mode
    local gpu_columns
    local htp_columns

    if [[ "$backend" == htp ]]; then
        benchmark=htp-control
        gpu_mode=native
        gpu_columns=512
        htp_columns=9152
        case "$batch" in
            1) iterations=8000 ;;
            8) iterations=3200 ;;
            32) iterations=3000 ;;
            128) iterations=2600 ;;
            *) exit 2 ;;
        esac
        case_id="htp-gemma-q4-ffn-m${batch}-r${repetition}"
    else
        benchmark=gpu-shard
        gpu_columns=512
        htp_columns=9152
        if (( batch == 1 )); then
            gpu_mode=native
            iterations=7000
        else
            gpu_mode=f16-xmem
            iterations=2400
        fi
        case_id="adreno-gemma-${gpu_mode}-ffn512-m${batch}-r${repetition}"
    fi

    command="cd $runtime && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
    command+="GGML_HEXAGON_MBUF=4192 GGML_HEXAGON_NHVX=4 "
    command+="S41_DISABLE_GRAPH_CACHE=1 ./dual_backend_ffn.android "
    command+="--k 3840 --n-ff 15360 --htp-columns $htp_columns "
    command+="--gpu-columns $gpu_columns --batch $batch --type q4_0 "
    command+="--gpu-mode $gpu_mode --benchmark $benchmark "
    command+="--warmup 20 --iterations $iterations"
    run_case "$case_id" "$command"
}

idle_case() {
    local repetition=$1
    run_case "phone-idle-r${repetition}" "sleep 10"
}

idle_case 2
for batch in 128 32 16 1; do
    kernel_case adreno "$batch" 2
done
for batch in 128 32 8 1; do
    kernel_case htp "$batch" 2
done
for batch in 1 8 32 128; do
    kernel_case htp "$batch" 3
done
for batch in 1 16 32 128; do
    kernel_case adreno "$batch" 3
done
idle_case 3
