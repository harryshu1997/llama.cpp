#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
output_dir=${1:-"$s24_dir/results/cp1_cuda_capacity"}
model=${S24_MODEL:-/mnt/storage/s21_models/gemma-4-12b-it-Q8_0.gguf}
binary=${S24_BINARY:-"$repo_root/build-s21-cuda/bin/llama-layersplit"}
cuda_lib=${S24_CUDA_LIB:-/mnt/storage/s21_deps/cuda-13.2.1/lib64}
timeout_s=${S24_LOAD_TIMEOUT_S:-300}

if [[ -e "$output_dir" ]]; then
    echo "error: output already exists: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"

declare -a worker_names=()
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

sample_gpu() {
    local label=$1
    nvidia-smi \
        --query-gpu=timestamp,index,name,uuid,memory.total,memory.used,memory.free,power.draw \
        --format=csv,noheader > "$output_dir/gpu-$label.csv"
    nvidia-smi \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader > "$output_dir/gpu-processes-$label.csv" || true
}

sample_process() {
    local worker_name=$1
    local worker_pid=$2
    {
        printf 'worker\tpid\tvmrss_kib\tvmhwm_kib\tpss_kib\n'
        vmrss=$(awk '/^VmRSS:/ {print $2}' "/proc/$worker_pid/status")
        vmhwm=$(awk '/^VmHWM:/ {print $2}' "/proc/$worker_pid/status")
        pss=$(awk '/^Pss:/ {print $2}' "/proc/$worker_pid/smaps_rollup")
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$worker_name" "$worker_pid" "$vmrss" "$vmhwm" "$pss"
    } > "$output_dir/process-$worker_name.tsv"
    tr '\0' ' ' < "/proc/$worker_pid/cmdline" \
        > "$output_dir/process-$worker_name.cmdline"
    printf '\n' >> "$output_dir/process-$worker_name.cmdline"
}

wait_for_worker() {
    local worker_name=$1
    local worker_pid=$2
    local log_file=$3
    local waited=0
    while ! grep -q '\[stagenet\] listening' "$log_file"; do
        if ! kill -0 "$worker_pid" 2>/dev/null; then
            echo "error: $worker_name exited during load" >&2
            tail -80 "$log_file" >&2 || true
            return 1
        fi
        if (( waited >= timeout_s * 10 )); then
            echo "error: $worker_name load timed out" >&2
            return 1
        fi
        sleep 0.1
        ((waited += 1))
    done
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
    worker_names+=("$worker_name")
    worker_pids+=("$worker_pid")
    printf '%s\n' "$worker_pid" > "$output_dir/$worker_name.pid"
    wait_for_worker "$worker_name" "$worker_pid" "$log_file"
    sample_process "$worker_name" "$worker_pid"
    sample_gpu "after-$worker_name"
}

sample_gpu before

launch_worker cuda-prefix 0 8 stagenet 24180 4
launch_worker cuda-mid 8 16 stagenet 24181 4
launch_worker cuda-tail 16 48 tailv3 24182 8

for worker_index in "${!worker_pids[@]}"; do
    sample_process \
        "${worker_names[$worker_index]}" \
        "${worker_pids[$worker_index]}"
done
sample_gpu resident
nvidia-smi -q -d MEMORY > "$output_dir/nvidia-memory-resident.txt"
LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ldd "$binary" > "$output_dir/binary-ldd.txt"
{
    printf 'schema=s24-cuda-runtime-sha256-v1\n'
    printf 'link_directory=%s\n' "$cuda_lib"
    for library in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
        [[ -e "$cuda_lib/$library" ]] || {
            echo "error: linked CUDA runtime is missing: $cuda_lib/$library" >&2
            exit 2
        }
        resolved=$(readlink -f "$cuda_lib/$library")
        printf '%s  %s  %s\n' \
            "$(sha256sum "$resolved" | awk '{print $1}')" \
            "$(stat -Lc '%s' "$resolved")" \
            "$resolved"
    done
} > "$output_dir/cuda-runtime-sha256.txt"

PYTHONPATH="$repo_root/research_dev/spikes/s22_slo_overlap_pipeline" \
python3 - "$output_dir/worker-hellos.json" <<'PY'
import json
import sys
from dataclasses import asdict

from stage_v3_client import StageV3Client

workers = {
    "cuda-prefix": ("127.0.0.1", 24180),
    "cuda-mid": ("127.0.0.1", 24181),
    "cuda-tail": ("127.0.0.1", 24182),
}
result = {}
for name, endpoint in workers.items():
    client = StageV3Client.connect(*endpoint, 30.0)
    try:
        result[name] = {
            "hello": asdict(client.hello()),
            "status": asdict(client.status()),
        }
        client.stop()
    finally:
        client.close()
with open(sys.argv[1], "w", encoding="ascii") as stream:
    json.dump(result, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY

worker_failure=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then
        worker_failure=1
    fi
done
cleaned=1
sample_gpu after

sha256sum \
    "$model" \
    "$binary" \
    "$repo_root/examples/layersplit/layersplit.cpp" \
    > "$output_dir/artifact-sha256.txt"

if [[ "$worker_failure" -ne 0 ]]; then
    echo "error: at least one worker exited unsuccessfully" >&2
    exit 2
fi

printf '%s\n' "$output_dir"
