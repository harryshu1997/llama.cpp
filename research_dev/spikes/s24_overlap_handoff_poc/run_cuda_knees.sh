#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
output_dir=${1:-"$s24_dir/results/cp3_cuda_knees"}
model=${S24_MODEL:-/mnt/storage/s21_models/gemma-4-12b-it-Q8_0.gguf}
binary=${S24_BINARY:-"$repo_root/build-s21-cuda/bin/llama-layersplit"}
cuda_lib=${S24_CUDA_LIB:-/mnt/storage/s21_deps/cuda-13.2.1/lib64}

if [[ -e "$output_dir" ]]; then
    echo "error: output already exists: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"

declare -a worker_pids=()
cleaned=0

cleanup() {
    if [[ "$cleaned" -eq 1 ]]; then
        return
    fi
    cleaned=1
    for worker_pid in "${worker_pids[@]:-}"; do
        if kill -0 "$worker_pid" 2>/dev/null; then
            kill "$worker_pid" 2>/dev/null || true
        fi
    done
    for worker_pid in "${worker_pids[@]:-}"; do
        wait "$worker_pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

wait_for_worker() {
    local worker_name=$1
    local worker_pid=$2
    local log_file=$3
    for _attempt in $(seq 1 3000); do
        if grep -q '\[stagenet\] listening' "$log_file"; then
            return
        fi
        if ! kill -0 "$worker_pid" 2>/dev/null; then
            echo "error: $worker_name exited during load" >&2
            tail -80 "$log_file" >&2 || true
            return 1
        fi
        sleep 0.1
    done
    echo "error: $worker_name load timed out" >&2
    return 1
}

launch_worker() {
    local worker_name=$1
    local layer_start=$2
    local layer_end=$3
    local mode=$4
    local port=$5
    local max_streams=$6
    local log_file="$output_dir/$worker_name.log"

    env \
        LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        LLAMA_LAYER_START="$layer_start" \
        LLAMA_LAYER_END="$layer_end" \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        "$binary" \
        -m "$model" \
        --mode "$mode" \
        --port "$port" \
        --driver-batch "$max_streams" \
        --driver-context 512 \
        --driver-max-prefill 64 \
        --devices CUDA0 \
        -ngl 99 \
        > "$log_file" 2>&1 &
    local worker_pid=$!
    worker_pids+=("$worker_pid")
    wait_for_worker "$worker_name" "$worker_pid" "$log_file"
}

launch_worker cuda-prefix 0 8 stagenet 24180 4
launch_worker cuda-mid 8 16 stagenet 24181 4
launch_worker cuda-tail 16 48 tailv3 24182 8

python3 "$s24_dir/batch_knee.py" \
    --worker cuda-prefix \
    --endpoint 127.0.0.1:24180 \
    --candidates 1,2,4 \
    --output "$output_dir/cuda-prefix-knee.json" \
    > "$output_dir/cuda-prefix-knee.stdout"
python3 "$s24_dir/batch_knee.py" \
    --worker cuda-mid \
    --endpoint 127.0.0.1:24181 \
    --candidates 1,2,4 \
    --output "$output_dir/cuda-mid-knee.json" \
    > "$output_dir/cuda-mid-knee.stdout"
python3 "$s24_dir/batch_knee.py" \
    --worker cuda-tail \
    --endpoint 127.0.0.1:24182 \
    --candidates 1,2,4,8 \
    --output "$output_dir/cuda-tail-knee.json" \
    > "$output_dir/cuda-tail-knee.stdout"

PYTHONPATH="$repo_root/research_dev/spikes/s22_slo_overlap_pipeline" \
python3 - <<'PY'
from stage_v3_client import StageV3Client

for port in (24180, 24181, 24182):
    client = StageV3Client.connect("127.0.0.1", port, 30.0)
    try:
        client.hello()
        client.stop()
    finally:
        client.close()
PY

worker_failure=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then
        worker_failure=1
    fi
done
cleaned=1

sha256sum \
    "$model" \
    "$binary" \
    "$s24_dir/batch_knee.py" \
    > "$output_dir/artifact-sha256.txt"

if [[ "$worker_failure" -ne 0 ]]; then
    echo "error: at least one worker exited unsuccessfully" >&2
    exit 2
fi

printf '%s\n' "$output_dir"
