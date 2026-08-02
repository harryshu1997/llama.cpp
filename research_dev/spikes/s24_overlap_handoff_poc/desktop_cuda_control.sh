#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
action=${1:-}
session_dir=${2:-}
model=${S24_MODEL:-/home/zhihao/models/gemma-4-12B-it-Q8_0-7b56.gguf}
binary=${S24_BINARY:-"$repo_root/build-s21-cuda/bin/llama-layersplit"}
cuda_lib=${S24_CUDA_LIB:-/mnt/storage/s21_deps/cuda-13.2.1/lib64}
context=${S24_CUDA_CONTEXT:-600}
max_prefill=${S24_CUDA_MAX_PREFILL:-64}
n_gen=${S24_CUDA_N_GEN:-64}
prefix_port=${S24_CUDA_PREFIX_PORT:-24180}
mid_port=${S24_CUDA_MID_PORT:-24181}
tail_port=${S24_CUDA_TAIL_PORT:-24182}
prefix_streams=${S24_CUDA_PREFIX_STREAMS:-4}
mid_streams=${S24_CUDA_MID_STREAMS:-4}
tail_streams=${S24_CUDA_TAIL_STREAMS:-8}
prefix_layer_start=${S24_CUDA_PREFIX_LAYER_START:-0}
prefix_layer_end=${S24_CUDA_PREFIX_LAYER_END:-8}
mid_layer_start=${S24_CUDA_MID_LAYER_START:-8}
mid_layer_end=${S24_CUDA_MID_LAYER_END:-16}
tail_layer_start=${S24_CUDA_TAIL_LAYER_START:-16}
tail_layer_end=${S24_CUDA_TAIL_LAYER_END:-48}
model_sha=${S24_MODEL_SHA256:-7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848}
binary_sha=${S24_BINARY_SHA256:-56759235bafd45b071c0cbcb3107e1d9cc93b1c66df98364a5faa2505c518db6}
source_sha=${S24_SOURCE_SHA256:-104d3485432acf2d70596d62748ffe1c0c19f41366e502425be0af8c7cdb07d8}

fail() {
    echo "error: $*" >&2
    exit 2
}

[[ "$action" =~ ^(start|status|stop|collect)$ ]] || \
    fail "usage: $0 {start|status|stop|collect} SESSION_DIR"
[[ -n "$session_dir" ]] || fail "SESSION_DIR is required"
for value in "$context" "$max_prefill" "$n_gen" \
    "$prefix_port" "$mid_port" "$tail_port" "$prefix_streams" \
    "$mid_streams" "$tail_streams" "$prefix_layer_start" \
    "$prefix_layer_end" "$mid_layer_start" "$mid_layer_end" \
    "$tail_layer_start" "$tail_layer_end"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fail "numeric configuration is invalid"
done
(( context >= max_prefill + n_gen )) || fail "context must cover max-prefill plus n-gen"
(( prefix_layer_start < prefix_layer_end && mid_layer_start < mid_layer_end && \
   tail_layer_start < tail_layer_end )) || fail "CUDA layer range is invalid"

pid_file() {
    printf '%s/%s.pid' "$session_dir" "$1"
}

worker_log() {
    printf '%s/%s.log' "$session_dir" "$1"
}

worker_alive() {
    local name=$1
    local path
    path=$(pid_file "$name")
    [[ -f "$path" ]] || return 1
    local pid
    pid=$(<"$path")
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

launch_worker() {
    local name=$1
    local layer_start=$2
    local layer_end=$3
    local mode=$4
    local port=$5
    local streams=$6
    local log
    log=$(worker_log "$name")
    printf '%q ' env \
        "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "LLAMA_LAYER_START=$layer_start" \
        "LLAMA_LAYER_END=$layer_end" \
        "LAYERSPLIT_MODEL_SHA256=$model_sha" \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        "$binary" -m "$model" --mode "$mode" --port "$port" \
        --driver-batch "$streams" --driver-context "$context" \
        --driver-max-prefill "$max_prefill" -n "$n_gen" \
        --devices CUDA0 -ngl 99 \
        > "$session_dir/$name.command.txt"
    printf '\n' >> "$session_dir/$name.command.txt"
    nohup env \
        LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        LLAMA_LAYER_START="$layer_start" \
        LLAMA_LAYER_END="$layer_end" \
        LAYERSPLIT_MODEL_SHA256="$model_sha" \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        "$binary" \
        -m "$model" \
        --mode "$mode" \
        --port "$port" \
        --driver-batch "$streams" \
        --driver-context "$context" \
        --driver-max-prefill "$max_prefill" \
        -n "$n_gen" \
        --devices CUDA0 \
        -ngl 99 \
        > "$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$(pid_file "$name")"
}

wait_ready() {
    local name=$1
    local log
    log=$(worker_log "$name")
    for _attempt in $(seq 1 3600); do
        if grep -q '\[stagenet\] listening' "$log"; then
            return
        fi
        if ! worker_alive "$name"; then
            tail -80 "$log" >&2 || true
            fail "$name exited during load"
        fi
        sleep 0.1
    done
    fail "$name load timed out"
}

safe_force_stop() {
    local name=$1
    local path
    path=$(pid_file "$name")
    [[ -f "$path" ]] || return
    local pid
    pid=$(<"$path")
    [[ "$pid" =~ ^[0-9]+$ ]] || return
    if kill -0 "$pid" 2>/dev/null; then
        local command_line
        command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
        if [[ "$command_line" == *llama-layersplit* ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    fi
}

case "$action" in
    start)
        [[ ! -e "$session_dir" ]] || fail "session output already exists: $session_dir"
        [[ -f "$model" && -x "$binary" && -d "$cuda_lib" ]] || \
            fail "desktop CUDA artifacts are missing"
        [[ $(sha256sum "$model" | awk '{print $1}') == "$model_sha" ]] || \
            fail "desktop Q8 model hash mismatch"
        [[ $(sha256sum "$binary" | awk '{print $1}') == "$binary_sha" ]] || \
            fail "desktop V3 binary hash mismatch"
        [[ $(sha256sum "$repo_root/examples/layersplit/layersplit.cpp" | awk '{print $1}') == "$source_sha" ]] || \
            fail "desktop V3 source hash mismatch"
        mkdir -p "$session_dir"
        starting=1
        cleanup_start() {
            local status=$?
            if [[ "$starting" -eq 1 ]]; then
                safe_force_stop cuda-prefix
                safe_force_stop cuda-mid
                safe_force_stop cuda-tail
            fi
            exit "$status"
        }
        trap cleanup_start EXIT INT TERM
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free \
            --format=csv,noheader > "$session_dir/gpu-before.csv"
        launch_worker cuda-prefix "$prefix_layer_start" "$prefix_layer_end" \
            stagenet "$prefix_port" "$prefix_streams"
        wait_ready cuda-prefix
        launch_worker cuda-mid "$mid_layer_start" "$mid_layer_end" \
            stagenet "$mid_port" "$mid_streams"
        wait_ready cuda-mid
        launch_worker cuda-tail "$tail_layer_start" "$tail_layer_end" \
            tailv3 "$tail_port" "$tail_streams"
        wait_ready cuda-tail
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free \
            --format=csv,noheader > "$session_dir/gpu-resident.csv"
        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader > "$session_dir/gpu-processes-resident.csv"
        sha256sum \
            "$model" \
            "$binary" \
            "$repo_root/examples/layersplit/layersplit.cpp" \
            > "$session_dir/artifact-sha256.txt"
        {
            printf 'schema=s24-desktop-cuda-session-v1\n'
            printf 'context=%s\n' "$context"
            printf 'prefix_endpoint=127.0.0.1:%s\n' "$prefix_port"
            printf 'mid_endpoint=127.0.0.1:%s\n' "$mid_port"
            printf 'tail_endpoint=127.0.0.1:%s\n' "$tail_port"
            printf 'prefix_streams=%s\n' "$prefix_streams"
            printf 'mid_streams=%s\n' "$mid_streams"
            printf 'tail_streams=%s\n' "$tail_streams"
            printf 'prefix_layer_range=%s:%s\n' "$prefix_layer_start" "$prefix_layer_end"
            printf 'mid_layer_range=%s:%s\n' "$mid_layer_start" "$mid_layer_end"
            printf 'tail_layer_range=%s:%s\n' "$tail_layer_start" "$tail_layer_end"
            printf 'model_sha256=%s\n' "$model_sha"
        } > "$session_dir/session.env"
        starting=0
        trap - EXIT INT TERM
        printf '%s\n' "$session_dir"
        ;;
    status)
        for name in cuda-prefix cuda-mid cuda-tail; do
            state=dead
            worker_alive "$name" && state=live
            printf '%s %s\n' "$name" "$state"
        done
        ;;
    stop)
        safe_force_stop cuda-prefix
        safe_force_stop cuda-mid
        safe_force_stop cuda-tail
        printf 'FORCED_STOP_NO_PLACEMENT_ACCEPTANCE\n' > "$session_dir/forced-stop.txt"
        ;;
    collect)
        for name in cuda-prefix cuda-mid cuda-tail; do
            worker_alive "$name" && fail "$name is still live; send StageNet STOP first"
        done
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free \
            --format=csv,noheader > "$session_dir/gpu-after.csv"
        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader > "$session_dir/gpu-processes-after.csv" || true
        (
            cd "$session_dir"
            find . -type f ! -name SHA256SUMS.txt -print0 |
                LC_ALL=C sort -z |
                xargs -0 sha256sum > SHA256SUMS.txt
        )
        printf '%s\n' "$session_dir"
        ;;
esac
