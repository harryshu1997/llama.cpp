#!/usr/bin/env bash

set -euo pipefail

if (( $# != 2 )); then
    echo "usage: $0 TOOL_ROOT RESULT_ROOT" >&2
    exit 2
fi

tool_root=$1
result_root=$2
server=${S42_LLAMA_SERVER:-/home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server}
qwen=${S42_QWEN_MODEL:-/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf}
gemma=${S42_GEMMA_MODEL:-/home/zhihao/models/gemma-4-12B-it-Q8_0-7b56.gguf}
cuda_lib=${S42_CUDA_LIB:-/mnt/storage/s21_deps/cuda-13.2.1/lib}

run_case() {
    local case_id=$1
    local source_model=$2
    local target_model=$3
    local port=$4
    local output="$result_root/$case_id.json"

    if [[ -f "$output" ]]; then
        echo "SKIP $case_id"
        return
    fi
    python3 "$tool_root/measure_desktop.py" \
        --case-id "$case_id" \
        --output "$output" \
        --window stdout-realtime \
        -- \
        python3 "$tool_root/model_switch_probe.py" \
        --server "$server" \
        --source-model "$source_model" \
        --target-model "$target_model" \
        --cuda-lib-dir "$cuda_lib" \
        --port "$port"
    sleep 5
}

for repetition in 1 2 3; do
    run_case "model-switch-gemma-to-qwen-r${repetition}" \
        "$gemma" "$qwen" $((18900 + repetition))
    run_case "model-switch-qwen-to-gemma-r${repetition}" \
        "$qwen" "$gemma" $((18910 + repetition))
done
