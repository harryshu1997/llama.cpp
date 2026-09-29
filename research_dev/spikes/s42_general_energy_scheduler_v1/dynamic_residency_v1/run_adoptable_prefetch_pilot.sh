#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <repeat-index> <absolute-new-output-root> <adb-port>" >&2
    exit 2
fi

repeat_index=$1
output=$2
adb_port=$3
if [[ ! $repeat_index =~ ^[0-9]+$ || $repeat_index -eq 0 \
        || $output != /* || -e $output || -e ${output}.phone-capture \
        || -e ${output}.adoption ]]; then
    echo "invalid adoptable pilot arguments" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
legacy_root=${S42_LEGACY_ROOT:-/home/zhihao/s41-dynamic-ffn-v1}
burst_dir=${S42_BURST_DIR:-$legacy_root/campaign/server_trace_v2}
trace=$legacy_root/campaign/input/REQUESTS_SEMANTIC_SOURCE.jsonl
qwen_arm=${S42_QWEN_ARM:?set S42_QWEN_ARM}
gemma_server=${S42_ADOPTABLE_GEMMA_SERVER:?set S42_ADOPTABLE_GEMMA_SERVER}
gemma_model=${S42_GEMMA_MODEL:-/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf}
bridge=${S42_ADOPTABLE_BRIDGE:?set S42_ADOPTABLE_BRIDGE}
server_bin=$(dirname -- "$gemma_server")
lib_dir=${S42_CUDA_LIB_DIR:-/mnt/storage/s21_deps/cuda-13.2.1/lib}
gemma_port=${S42_GEMMA_PORT:-18590}
stage_offset=15838752
stage_bytes=2013265920
chunk_bytes=${S42_ADOPTABLE_CHUNK_BYTES:-4194304}
chunks_per_window=${S42_ADOPTABLE_CHUNKS_PER_WINDOW:-9}
gpu_reserve_bytes=536870912
adoption=${output}.adoption

for path in "$qwen_arm" "$gemma_server" "$gemma_model" "$bridge" \
        "$here/run_gemma_adoption_probe.py" \
        "$here/monitor_process_memory.py" \
        "$here/capture_cgroup_memory.py"; do
    if [[ ! -e $path ]]; then
        echo "missing adoptable pilot dependency: $path" >&2
        exit 1
    fi
done
mkdir "$adoption"
runtime_dir=$(mktemp -d /tmp/s42-adoptable-prefetch.XXXXXX)
socket=$runtime_dir/fence.sock
arm_file=$runtime_dir/armed
qwen_ready_file=$runtime_dir/qwen.ready
qwen_release_file=$runtime_dir/qwen.release
memory_stop_file=$runtime_dir/memory.stop
gemma_pid=
qwen_pid=
memory_pid=
cleanup() {
    if [[ -n $memory_pid ]] && kill -0 "$memory_pid" 2>/dev/null; then
        touch -- "$memory_stop_file"
        wait "$memory_pid" 2>/dev/null || true
    fi
    if [[ -n $qwen_pid ]] && kill -0 "$qwen_pid" 2>/dev/null; then
        kill -INT "$qwen_pid" 2>/dev/null || true
        wait "$qwen_pid" 2>/dev/null || true
    fi
    if [[ -n $gemma_pid ]] && kill -0 "$gemma_pid" 2>/dev/null; then
        kill -INT "$gemma_pid" 2>/dev/null || true
        wait "$gemma_pid" 2>/dev/null || true
    fi
    rm -f -- "$socket" "$arm_file" "$qwen_ready_file" \
        "$qwen_release_file" "$memory_stop_file"
    rmdir -- "$runtime_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

S42_QWEN_BRIDGE="$bridge" \
S42_QWEN_GPU_LAYERS=15 \
S42_PREFETCH_ARM_FILE="$arm_file" \
S42_FFN_PREFETCH_FENCE_SOCKET="$socket" \
S42_FFN_PREFETCH_FENCE_GROUP_FIRST_LAYER=0 \
S42_FFN_PREFETCH_FENCE_GROUP_LAST_LAYER=11 \
S42_FFN_PREFETCH_FENCE_TOTAL_BYTES="$stage_bytes" \
S42_QWEN_SERVER_READY_FILE="$qwen_ready_file" \
S42_QWEN_SERVER_RELEASE_FILE="$qwen_release_file" \
S42_PHONE_RUN_TAG=dynamic-adoptable-v1 \
S42_QWEN_ONLY_PHONE=1 \
S42_PHONE_WORKERS=/data/local/tmp/s41-opoffload-dmabuf-v1/resident_qwen_workers-v1 \
S42_PHONE_SESSION=/data/local/tmp/s41-opoffload-dmabuf-v1/resident_qwen_session-v1.sh \
"$qwen_arm" op15 "$repeat_index" "$output" "$adb_port" \
    >"$adoption/qwen-arm.stdout" 2>"$adoption/qwen-arm.stderr" &
qwen_pid=$!

qwen_ready=0
for _ in $(seq 1 1800); do
    if [[ -e $qwen_ready_file ]]; then
        qwen_ready=1
        break
    fi
    if ! kill -0 "$qwen_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $qwen_ready -ne 1 ]]; then
    echo "Qwen server did not reach the staging barrier" >&2
    exit 1
fi
qwen_port=$((18460 + repeat_index))
mapfile -t qwen_server_pids < <(
    pgrep -f -- "^[^ ]*/llama-server .*--port $qwen_port( |$)" || true
)
if [[ ${#qwen_server_pids[@]} -ne 1 ]]; then
    echo "Qwen server process identity is not unique" >&2
    exit 1
fi
qwen_server_pid=${qwen_server_pids[0]}

env \
    "LD_LIBRARY_PATH=$lib_dir:$server_bin" \
    "S42_FENCED_TENSOR_SOCKET=$socket" \
    "S42_FENCED_TENSOR_ARM_FILE=$arm_file" \
    S42_FENCED_TENSOR_NAME=token_embd.weight \
    "S42_FENCED_TENSOR_EXPECTED_OFFSET=$stage_offset" \
    "S42_FENCED_TENSOR_EXPECTED_BYTES=$stage_bytes" \
    "S42_FENCED_TENSOR_CHUNK_BYTES=$chunk_bytes" \
    "S42_FENCED_TENSOR_CHUNKS_PER_WINDOW=$chunks_per_window" \
    "S42_FENCED_TENSOR_GPU_RESERVE_BYTES=$gpu_reserve_bytes" \
    "$gemma_server" \
    --model "$gemma_model" --alias gemma-adopted-f16 --fit off \
    --ctx-size 32768 --parallel 8 --batch-size 4096 --ubatch-size 512 \
    --flash-attn on --cont-batching --kv-unified --no-cache-idle-slots \
    --cache-type-k f16 --cache-type-v f16 --split-mode none \
    --n-gpu-layers 1 --main-gpu 0 --device CUDA0 \
    --host 127.0.0.1 --port "$gemma_port" --metrics --slots \
    --no-webui --log-colors off --log-timestamps --log-verbosity 1 \
    >"$adoption/gemma.stdout" 2>"$adoption/gemma.stderr" &
gemma_pid=$!

python3 "$here/monitor_process_memory.py" \
    --process "qwen:$qwen_server_pid" --process "gemma:$gemma_pid" \
    --stop-file "$memory_stop_file" \
    --output "$adoption/PROCESS_MEMORY_V1.json" &
memory_pid=$!

ready=0
for _ in $(seq 1 900); do
    if grep -q '^S42_FENCED_TENSOR_READY ' \
            "$adoption/gemma.stderr" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$gemma_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $ready -ne 1 ]]; then
    echo "Gemma fenced tensor did not become source-READY" >&2
    exit 1
fi

touch -- "$qwen_release_file"
wait "$qwen_pid"
qwen_pid=

server_ready=0
for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$gemma_port/health" \
            >/dev/null 2>&1; then
        server_ready=1
        break
    fi
    if ! kill -0 "$gemma_pid" 2>/dev/null; then
        break
    fi
    sleep 0.2
done
if [[ $server_ready -ne 1 ]]; then
    echo "adopted Gemma server did not publish READY" >&2
    exit 1
fi

python3 "$here/run_gemma_adoption_probe.py" \
    --burst-dir "$burst_dir" --requests "$trace" --request-index 50 \
    --port "$gemma_port" --server-pid "$gemma_pid" \
    --raw-output "$adoption/gemma-request-050.raw" \
    --output "$adoption/GEMMA_EXECUTION_V1.json"

touch -- "$memory_stop_file"
wait "$memory_pid"
memory_pid=
python3 "$here/capture_cgroup_memory.py" \
    --output "$adoption/CGROUP_MEMORY_V1.json"

kill -INT "$gemma_pid"
wait "$gemma_pid"
gemma_pid=
sha256sum \
    "$gemma_server" "$bridge" "$qwen_arm" \
    "$output/RESULT.json" \
    "${output}.phone-capture/PHONE_ENERGY_V3.json" \
    "${output}.phone-capture/bridge.stderr" \
    "$adoption/gemma.stderr" \
    "$adoption/PROCESS_MEMORY_V1.json" \
    "$adoption/CGROUP_MEMORY_V1.json" \
    "$adoption/GEMMA_EXECUTION_V1.json" \
    >"$adoption/SHA256SUMS.txt"
trap - EXIT INT TERM
cleanup
cat "$adoption/GEMMA_EXECUTION_V1.json"
