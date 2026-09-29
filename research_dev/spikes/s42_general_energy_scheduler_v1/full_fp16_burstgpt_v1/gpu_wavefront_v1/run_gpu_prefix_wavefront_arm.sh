#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 <adaptive|control|mechanics|qualified|staged> <repeat-index> <output-root> <adb-port> <profile-template>" >&2
    exit 2
fi

mode=$1
repeat_index=$2
output=$3
adb_port=$4
profile_template=$5
if [[ $mode != adaptive && $mode != control \
        && $mode != mechanics && $mode != qualified && $mode != staged ]]; then
    echo "invalid GPU-prefix wavefront mode: $mode" >&2
    exit 2
fi
if [[ ! $repeat_index =~ ^[0-9]+$ || $repeat_index -eq 0 \
        || $output != /* || -e $output \
        || -e ${output}.phone-capture || -e ${output}.wavefront \
        || $profile_template != /* || ! -f $profile_template ]]; then
    echo "invalid GPU-prefix arm arguments" >&2
    exit 2
fi
if [[ ${S42_ENABLE_GPU_PREFIX_PHYSICAL:-} != YES ]]; then
    echo "physical GPU-prefix execution is disabled; set S42_ENABLE_GPU_PREFIX_PHYSICAL=YES" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=${S42_UNIFIED_REPO_ROOT:-$(cd -- "$here/../../../../.." && pwd)}
qwen_arm=${S42_QWEN_ARM:-$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/run_qwen_full_energy_arm.sh}
qwen_bridge=${S42_WAVEFRONT_BRIDGE:?set S42_WAVEFRONT_BRIDGE}
layersplit=${S42_LAYERSPLIT:?set S42_LAYERSPLIT}
qwen_server=${S42_QWEN_SERVER:?set S42_QWEN_SERVER}
stage_manifest=${S42_GPU_PREFIX_STAGE_MANIFEST:?set S42_GPU_PREFIX_STAGE_MANIFEST}
gemma_model=${S42_GEMMA_MODEL:-/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf}
gemma_model_sha256=${S42_GEMMA_MODEL_SHA256:-ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf}
burst_dir=${S42_BURST_DIR:-$repo_root/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1}
trace=${S42_BURSTGPT_REQUESTS:-$burst_dir/REQUESTS_SEMANTIC_SOURCE.jsonl}
wavefront_cuda_lib_dir=${S42_WAVEFRONT_CUDA_LIB_DIR:-/mnt/storage/s21_deps/cuda-13.2.1/lib}
cuda_mps=${S42_WAVEFRONT_CUDA_MPS:-YES}
mps_control=${S42_WAVEFRONT_MPS_CONTROL:-/usr/bin/nvidia-cuda-mps-control}
qwen_mps_active_thread_pct=${S42_WAVEFRONT_QWEN_MPS_ACTIVE_THREAD_PCT:-80}
stage_mps_active_thread_pct=${S42_WAVEFRONT_STAGE_MPS_ACTIVE_THREAD_PCT:-100}
qwen_mps_client_priority=${S42_WAVEFRONT_QWEN_MPS_CLIENT_PRIORITY:-1}
stage_mps_client_priority=${S42_WAVEFRONT_STAGE_MPS_CLIENT_PRIORITY:-0}
gemma_indices=${S42_WAVEFRONT_GEMMA_INDICES:-50}
qwen_indices=${S42_WAVEFRONT_QWEN_INDICES:-52,53,31}
gemma_threads=${S42_WAVEFRONT_GEMMA_THREADS:-8}
gemma_cpus=${S42_WAVEFRONT_GEMMA_CPUS:-0,2,4,6,8,10,12,14}
gemma_protected_cpus=${S42_WAVEFRONT_GEMMA_PROTECTED_CPUS:-16-23}
gemma_ubatch=${S42_WAVEFRONT_GEMMA_UBATCH:-8}
gemma_prefill_mode=${S42_WAVEFRONT_GEMMA_PREFILL_MODE:-async_stream}
qwen_gpu_layers=${S42_WAVEFRONT_QWEN_GPU_LAYERS:-15}
qwen_log_verbosity=${S42_WAVEFRONT_QWEN_LOG_VERBOSITY:-4}
qwen_cpus=${S42_WAVEFRONT_QWEN_CPUS:-0,2,4,6,8,10,12,14}
phone_bridge_cpus=${S42_WAVEFRONT_PHONE_BRIDGE_CPUS:-15}
stage_worker_cpus=${S42_WAVEFRONT_STAGE_WORKER_CPUS:-1}
gate_cpus=${S42_WAVEFRONT_GATE_CPUS:-3}
qwen_tail_fence_layer=${S42_WAVEFRONT_QWEN_TAIL_FENCE_LAYER:-0}
qwen_tail_fence_join_layer=${S42_WAVEFRONT_QWEN_TAIL_FENCE_JOIN_LAYER:--1}
prefix_layers=${S42_WAVEFRONT_GPU_PREFIX_LAYERS:-1}
maximum_backfills=${S42_WAVEFRONT_MAX_BACKFILLS:-auto}
prepare_max_replays=${S42_WAVEFRONT_PREPARE_MAX_REPLAYS:-4}
prepare_required_consecutive=${S42_WAVEFRONT_PREPARE_REQUIRED_CONSECUTIVE:-2}
phone_run_tag=${S42_WAVEFRONT_RUN_TAG:-gpu-prefix-${mode}-r${repeat_index}}
phone_filler_port=${S42_WAVEFRONT_PHONE_FILLER_PORT:-}
phone_router=${S42_WAVEFRONT_PHONE_ROUTER:-/data/local/tmp/s41-opoffload-dmabuf-v1/resident_ffn_router-terminal-v1}
phone_residency_layout=${S42_PHONE_RESIDENCY_LAYOUT:-gemma23-qwen12-full-v1}
phone_defer_filler_until_protected_done=0
if [[ $mode == staged ]]; then
    phone_defer_filler_until_protected_done=1
fi
case $phone_residency_layout in
    gemma23-qwen12-full-v1)
        default_phone_ffn_layers=0-22
        ;;
    gemma46-qwen6-full-v1)
        default_phone_ffn_layers=0-45
        ;;
    *)
        echo "invalid GPU-prefix phone residency layout" >&2
        exit 2
        ;;
esac
phone_ffn_layers=${S42_WAVEFRONT_PHONE_FFN_LAYERS:-$default_phone_ffn_layers}
phone_ffn_columns=${S42_WAVEFRONT_PHONE_FFN_COLUMNS:-6144}
phone_ffn_prefill_columns=${S42_WAVEFRONT_PHONE_FFN_PREFILL_COLUMNS:-6144}
phone_ffn_timeout_ms=${S42_WAVEFRONT_PHONE_FFN_TIMEOUT_MS:-120000}
phone_ffn_vmem=${S42_PHONE_FFN_VMEM:-3328}
phone_minimum_available_kib=${S42_PHONE_MIN_AVAILABLE_KIB:-2097152}
capture=${output}.wavefront

profile_id=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["profile_id"])' \
    "$profile_template")
profile_chunk_rows=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["candidate"]["prefill_chunk_rows"])' \
    "$profile_template")
if [[ $gemma_indices != 50 || ! -f $trace \
        || ! $profile_chunk_rows =~ ^[0-9]+$ \
        || $profile_chunk_rows -eq 0 ]]; then
    echo "invalid GPU-prefix trace or chunk geometry" >&2
    exit 2
fi
gemma_input_tokens=$(python3 -c \
    'import json,sys; target=int(sys.argv[2]); row=next(value for value in (json.loads(line) for line in open(sys.argv[1], encoding="utf-8")) if value.get("request_index") == target); tokens=row.get("prompt_tokens"); assert isinstance(tokens, list) and len(tokens) == row.get("input_tokens") and all(type(token) is int and token >= 0 for token in tokens); print(len(tokens))' \
    "$trace" "$gemma_indices")
required_prefix_backfills=$((
    (gemma_input_tokens + profile_chunk_rows - 1) / profile_chunk_rows
))
if [[ $maximum_backfills == auto ]]; then
    maximum_backfills=$required_prefix_backfills
fi
if [[ ! ( $profile_id == fp16-burstgpt-gemma-prefix-mechanics-v1 \
            && $profile_chunk_rows -eq 2 ) \
        && ! ( $profile_id == fp16-burstgpt-gemma-prefix-m1-mechanics-v1 \
            && $profile_chunk_rows -eq 1 ) \
        && ! ( $profile_id == fp16-burstgpt-gemma-prefix-m8-macro-mechanics-v1 \
            && $profile_chunk_rows -eq 8 ) \
        && ! ( $profile_id == fp16-burstgpt-gemma-prefix-m8-staged-mechanics-v1 \
            && $profile_chunk_rows -eq 8 ) ]]; then
    echo "GPU-prefix profile template identity mismatch" >&2
    exit 2
fi
if [[ ! $gemma_model_sha256 =~ ^[0-9a-f]{64}$ \
        || $gemma_indices != 50 \
        || ! $gemma_threads =~ ^[0-9]+$ || $gemma_threads -eq 0 \
        || ! $gemma_cpus =~ ^[0-9,-]+$ \
        || ! $gemma_protected_cpus =~ ^[0-9,-]+$ \
        || ! $gemma_ubatch =~ ^[0-9]+$ || $gemma_ubatch -eq 0 \
        || $gemma_ubatch -gt 16 \
        || ! ( $gemma_prefill_mode == async_stream \
            || $gemma_prefill_mode == sync_chunked ) \
        || ! $profile_chunk_rows =~ ^[0-9]+$ \
        || $profile_chunk_rows -eq 0 \
        || $profile_chunk_rows -gt $gemma_ubatch \
        || ! -x $qwen_server \
        || ! $qwen_gpu_layers =~ ^[0-9]+$ || $qwen_gpu_layers -gt 41 \
        || ! $qwen_log_verbosity =~ ^[1-5]$ \
        || ! $qwen_cpus =~ ^[0-9,-]+$ \
        || ! $phone_bridge_cpus =~ ^[0-9,-]+$ \
        || ! $stage_worker_cpus =~ ^[0-9,-]+$ \
        || ! $gate_cpus =~ ^[0-9,-]+$ \
        || ! $qwen_tail_fence_layer =~ ^[0-9]+$ \
        || $qwen_tail_fence_layer -gt 11 \
        || ! $qwen_tail_fence_join_layer =~ ^-?[0-9]+$ \
        || $qwen_tail_fence_join_layer -lt -1 \
        || $qwen_tail_fence_join_layer -ge 40 \
        || ( $qwen_tail_fence_join_layer -ge 0 \
            && $qwen_tail_fence_join_layer -le $qwen_tail_fence_layer ) \
        || ( $qwen_tail_fence_join_layer -ge 0 \
            && $qwen_tail_fence_join_layer -ne $((40 - qwen_gpu_layers)) ) \
        || ! $prefix_layers =~ ^[0-9]+$ || $prefix_layers -ne 1 \
        || ! $maximum_backfills =~ ^[0-9]+$ \
        || $maximum_backfills -eq 0 || $maximum_backfills -gt 512 \
        || $maximum_backfills -ne $required_prefix_backfills \
        || ! $prepare_max_replays =~ ^[0-9]+$ \
        || $prepare_max_replays -eq 0 || $prepare_max_replays -gt 16 \
        || ! $prepare_required_consecutive =~ ^[0-9]+$ \
        || $prepare_required_consecutive -eq 0 \
        || $prepare_required_consecutive -gt $prepare_max_replays \
        || ! $phone_filler_port =~ ^[0-9]+$ \
        || $phone_filler_port -le 0 || $phone_filler_port -gt 65535 \
        || $phone_router != /data/local/tmp/s41-opoffload-dmabuf-v1/resident_ffn_router-terminal-v1 \
        || $phone_ffn_layers != "$default_phone_ffn_layers" \
        || ! $phone_ffn_columns =~ ^[0-9]+$ \
        || $phone_ffn_columns -ne 6144 \
        || ! $phone_ffn_prefill_columns =~ ^[0-9]+$ \
        || $phone_ffn_prefill_columns -ne 6144 \
        || ! $phone_ffn_timeout_ms =~ ^[0-9]+$ \
        || $phone_ffn_timeout_ms -eq 0 \
        || $phone_ffn_timeout_ms -gt 600000 \
        || ! $phone_ffn_vmem =~ ^[0-9]+$ \
        || $phone_ffn_vmem -lt 3200 || $phone_ffn_vmem -gt 3328 \
        || ! $phone_minimum_available_kib =~ ^[0-9]+$ \
        || $phone_minimum_available_kib -lt 2097152 \
        || $cuda_mps != YES \
        || ! $qwen_mps_active_thread_pct =~ ^[0-9]+$ \
        || $qwen_mps_active_thread_pct -lt 1 \
        || $qwen_mps_active_thread_pct -gt 100 \
        || ! $stage_mps_active_thread_pct =~ ^[0-9]+$ \
        || $stage_mps_active_thread_pct -lt 1 \
        || $stage_mps_active_thread_pct -gt 100 \
        || ! $qwen_mps_client_priority =~ ^[01]$ \
        || ! $stage_mps_client_priority =~ ^[01]$ ]]; then
    echo "invalid GPU-prefix runtime bounds" >&2
    exit 2
fi
if ! command -v taskset >/dev/null \
        || ! taskset --cpu-list "$gemma_cpus" true >/dev/null 2>&1 \
        || ! taskset --cpu-list "$gemma_protected_cpus" true >/dev/null 2>&1 \
        || ! taskset --cpu-list "$qwen_cpus" true >/dev/null 2>&1 \
        || ! taskset --cpu-list "$phone_bridge_cpus" true >/dev/null 2>&1 \
        || ! taskset --cpu-list "$stage_worker_cpus" true >/dev/null 2>&1 \
        || ! taskset --cpu-list "$gate_cpus" true >/dev/null 2>&1; then
    echo "invalid or unavailable Gemma CPU affinity" >&2
    exit 2
fi

for path in \
        "$qwen_arm" "$qwen_bridge" "$layersplit" "$stage_manifest" \
        "$mps_control" \
        "$gemma_model" "$trace" "$wavefront_cuda_lib_dir" \
        "$here/gpu_stage_wavefront_gate.py" \
        "$here/run_gemma_wavefront_driver.py" \
        "$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/capture_process_affinity.py" \
        "$here/materialize_stage_wavefront_profile.py" \
        "$here/analyze_gpu_prefix_wavefront_run.py"; do
    if [[ ! -e $path ]]; then
        echo "missing GPU-prefix dependency: $path" >&2
        exit 1
    fi
done
if pgrep -f '^.*/llama-layersplit .*--mode (stagenet|pipedriver)' >/dev/null; then
    echo "another LayerSplit pipeline is active" >&2
    pgrep -af 'llama-layersplit' >&2 || true
    exit 1
fi
if pgrep -f '^/usr/bin/nvidia-cuda-mps-(control|server)( |$)' >/dev/null; then
    echo "another CUDA MPS daemon is active" >&2
    pgrep -af '^/usr/bin/nvidia-cuda-mps-(control|server)( |$)' >&2 || true
    exit 1
fi

mkdir "$capture"
runtime_dir=$(mktemp -d /tmp/s42-gpu-prefix.XXXXXX)
mps_pipe=$runtime_dir/cuda-mps-pipe
mps_log=$capture/cuda-mps
mps_receipt=$capture/CUDA_MPS_RECEIPT.json
qwen_affinity_receipt=$capture/QWEN_AFFINITY_RECEIPT.json
wavefront_affinity_receipt=$capture/WAVEFRONT_AFFINITY_RECEIPT.json
mkdir "$mps_pipe" "$mps_log"
fence_socket=$runtime_dir/fence.sock
arm_file=$runtime_dir/paid.arm
qwen_ready_file=$runtime_dir/qwen.ready
qwen_release_file=$runtime_dir/qwen.release
qwen_complete_file=$runtime_dir/qwen.complete
paid_tail_file=$runtime_dir/paid.tail
qwen_paid_ready_file=$runtime_dir/qwen-paid.ready
qwen_paid_release_file=$runtime_dir/qwen-paid.release
candidate_ready_file=$runtime_dir/gemma-candidate.ready
candidate_prepare_file=$runtime_dir/gemma-candidate.prepare
candidate_prepared_file=$runtime_dir/gemma-candidate.prepared
driver_ready_file=$runtime_dir/gemma.ready
runtime_profile=$capture/RUNTIME_PROFILE.json
gpu_snapshot=$capture/GPU_SNAPSHOT.csv
gate_result=$capture/GATE_RESULT.json
driver_result=$capture/GEMMA_RESULT.json
worker_port=$((19260 + repeat_index))
gate_port=$((19360 + repeat_index))

qwen_pid=
worker_pid=
stage_worker_pid=
gate_pid=
driver_pid=
mps_started=0
stop_mps() {
    if [[ $mps_started -eq 1 ]]; then
        printf 'quit\n' | env \
            CUDA_MPS_PIPE_DIRECTORY="$mps_pipe" \
            CUDA_MPS_LOG_DIRECTORY="$mps_log" \
            "$mps_control" >/dev/null 2>&1 || true
        mps_started=0
    fi
}
cleanup() {
    for process_id in "$driver_pid" "$gate_pid" "$worker_pid" "$qwen_pid"; do
        if [[ -n $process_id ]] && kill -0 "$process_id" 2>/dev/null; then
            kill -TERM "$process_id" 2>/dev/null || true
            wait "$process_id" 2>/dev/null || true
        fi
    done
    stop_mps
    rm -f -- \
        "$fence_socket" "$arm_file" "$qwen_ready_file" \
        "$qwen_release_file" "$qwen_complete_file" "$paid_tail_file" \
        "$qwen_paid_ready_file" "$qwen_paid_release_file" \
        "$candidate_ready_file" "$candidate_prepare_file" \
        "$candidate_prepared_file" "$driver_ready_file"
    rmdir -- "$mps_pipe" 2>/dev/null || true
    rmdir -- "$runtime_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

mps_started=1
if ! env \
        CUDA_VISIBLE_DEVICES=0 \
        CUDA_MPS_PIPE_DIRECTORY="$mps_pipe" \
        CUDA_MPS_LOG_DIRECTORY="$mps_log" \
        "$mps_control" -d; then
    echo "CUDA MPS control daemon did not start" >&2
    exit 1
fi
export CUDA_MPS_PIPE_DIRECTORY="$mps_pipe"
export CUDA_MPS_LOG_DIRECTORY="$mps_log"
unset CUDA_VISIBLE_DEVICES

tail_fence_join_env=(-u S41_SERVER_FFN_TAIL_FENCE_JOIN_LAYER)
if [[ $qwen_tail_fence_join_layer -ge 0 ]]; then
    tail_fence_join_env=(
        S41_SERVER_FFN_TAIL_FENCE_JOIN_LAYER="$qwen_tail_fence_join_layer"
    )
fi
env "${tail_fence_join_env[@]}" \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$qwen_mps_active_thread_pct" \
CUDA_MPS_CLIENT_PRIORITY="$qwen_mps_client_priority" \
S42_QWEN_BRIDGE="$qwen_bridge" \
S42_QWEN_SERVER="$qwen_server" \
S42_QWEN_FILLER_PORT="$phone_filler_port" \
S42_PHONE_ROUTER="$phone_router" \
S42_PHONE_RESIDENCY_LAYOUT="$phone_residency_layout" \
S42_PHONE_MAX_SESSIONS=0 \
S42_PHONE_FFN_VMEM="$phone_ffn_vmem" \
S42_PHONE_MIN_AVAILABLE_KIB="$phone_minimum_available_kib" \
S42_QWEN_GPU_LAYERS="$qwen_gpu_layers" \
S42_QWEN_LOG_VERBOSITY="$qwen_log_verbosity" \
S42_QWEN_CPUS="$qwen_cpus" \
S42_PHONE_BRIDGE_CPUS="$phone_bridge_cpus" \
S42_QWEN_AFFINITY_RECEIPT="$qwen_affinity_receipt" \
S42_QWEN_INDICES="$qwen_indices" \
S42_BURST_DIR="$burst_dir" \
S42_BURSTGPT_REQUESTS="$trace" \
S42_PREFETCH_ARM_FILE="$arm_file" \
S41_SERVER_FFN_TAIL_FENCE_SOCKET="$fence_socket" \
S41_SERVER_FFN_TAIL_FENCE_LAYER="$qwen_tail_fence_layer" \
S42_QWEN_SERVER_READY_FILE="$qwen_ready_file" \
S42_QWEN_SERVER_RELEASE_FILE="$qwen_release_file" \
S42_QWEN_COMPLETE_FILE="$qwen_complete_file" \
S42_QWEN_PAID_READY_FILE="$qwen_paid_ready_file" \
S42_QWEN_PAID_RELEASE_FILE="$qwen_paid_release_file" \
S42_PHONE_PROTECTED_DONE_FILE="$qwen_complete_file" \
S42_PHONE_DEFER_FILLER_UNTIL_PROTECTED_DONE="$phone_defer_filler_until_protected_done" \
S42_PAID_TAIL_FILE="$paid_tail_file" \
S42_PAID_TAIL_TIMEOUT_S=7200 \
S42_PHONE_RUN_TAG="$phone_run_tag" \
bash "$qwen_arm" op15 "$repeat_index" "$output" "$adb_port" \
    >"$capture/qwen-arm.stdout" 2>"$capture/qwen-arm.stderr" &
qwen_pid=$!

qwen_ready=0
for _ in $(seq 1 3600); do
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
    echo "Qwen server did not reach the GPU-prefix setup barrier" >&2
    exit 1
fi

taskset --cpu-list "$stage_worker_cpus" \
env "LD_LIBRARY_PATH=$wavefront_cuda_lib_dir:$(dirname -- "$layersplit")" \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$stage_mps_active_thread_pct" \
    CUDA_MPS_CLIENT_PRIORITY="$stage_mps_client_priority" \
    LLAMA_LAYER_START=0 \
    LLAMA_LAYER_END="$prefix_layers" \
    LAYERSPLIT_MODEL_SHA256="$gemma_model_sha256" \
    LAYERSPLIT_MEMORY_CERT=1 \
    LAYERSPLIT_PLACEMENT_CERT=1 \
    "$layersplit" \
    -m "$gemma_model" --mode stagenet --port "$worker_port" \
    --devices CUDA0 -ngl 99 -t 1 -tb 1 -n 2048 \
    --driver-requests 1 --driver-batch 1 \
    --driver-context 4096 --driver-max-prefill 512 --driver-ubatch 512 \
    >"$capture/stage-worker.stdout" \
    2>"$capture/stage-worker.stderr" &
worker_pid=$!
stage_worker_pid=$worker_pid

worker_ready=0
for _ in $(seq 1 3600); do
    if grep -q '^\[stagenet\] listening ' \
            "$capture/stage-worker.stderr" 2>/dev/null; then
        worker_ready=1
        break
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $worker_ready -ne 1 ]]; then
    echo "resident Gemma CUDA prefix did not become ready" >&2
    exit 1
fi

nvidia-smi --query-gpu=uuid,memory.total,memory.free \
    --format=csv,noheader,nounits >"$gpu_snapshot"
python3 "$here/materialize_stage_wavefront_profile.py" \
    --template "$profile_template" \
    --gpu-snapshot "$gpu_snapshot" \
    --worker-log "$capture/stage-worker.stderr" \
    --stage-manifest "$stage_manifest" \
    --worker-host 127.0.0.1 --worker-port "$worker_port" \
    --output "$runtime_profile"

gate_args=(
    python3 "$here/gpu_stage_wavefront_gate.py"
    --mode "$mode"
    --profile "$runtime_profile"
    --listen-host 127.0.0.1 --listen-port "$gate_port"
    --worker-host 127.0.0.1 --worker-port "$worker_port"
    --fence-socket "$fence_socket"
    --arm-file "$arm_file"
    --qwen-complete-file "$qwen_complete_file"
    --paid-tail-file "$paid_tail_file"
    --candidate-ready-file "$candidate_ready_file"
    --prepare-file "$candidate_prepare_file"
    --prepared-file "$candidate_prepared_file"
    --prepare-max-replays "$prepare_max_replays"
    --prepare-required-consecutive "$prepare_required_consecutive"
    --output "$gate_result"
    --pipeline-id "gemma-prefix-burstgpt-${gemma_indices//,/-}"
    --max-backfills "$maximum_backfills"
    --timeout-s 7200
)
taskset --cpu-list "$gate_cpus" "${gate_args[@]}" \
    >"$capture/gate.stdout" 2>"$capture/gate.stderr" &
gate_pid=$!

gate_ready=0
for _ in $(seq 1 600); do
    if grep -q '"event": "ready"' "$capture/gate.stdout" 2>/dev/null; then
        gate_ready=1
        break
    fi
    if ! kill -0 "$gate_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $gate_ready -ne 1 ]]; then
    echo "unified GPU-prefix wavefront gate did not become ready" >&2
    exit 1
fi

driver_args=(
    python3 "$here/run_gemma_wavefront_driver.py"
    --layersplit "$layersplit" --model "$gemma_model"
    --requests "$trace" --indices "$gemma_indices"
    --arm-file "$arm_file" --ready-file "$driver_ready_file"
    --route-kind gpu-prefix --gpu-prefix-layers "$prefix_layers"
    --gate-host 127.0.0.1 --gate-port "$gate_port"
    --threads "$gemma_threads"
    --cpu-list "$gemma_cpus"
    --protected-cpu-list "$gemma_protected_cpus"
    --ubatch "$gemma_ubatch"
    --prefill-mode "$gemma_prefill_mode"
    --library-path "$wavefront_cuda_lib_dir"
    --stderr "$capture/gemma-driver.stderr"
    --output "$driver_result"
    --ffn-host 127.0.0.1 --ffn-port "$phone_filler_port"
    --ffn-layers "$phone_ffn_layers"
    --ffn-columns "$phone_ffn_columns"
    --ffn-prefill-columns "$phone_ffn_prefill_columns"
    --ffn-decode-columns "$phone_ffn_columns"
    --ffn-timeout-ms "$phone_ffn_timeout_ms"
    --ffn-f16-io
)
if [[ $gemma_protected_cpus != "$gemma_cpus" ]]; then
    driver_args+=(--protected-done-file "$qwen_complete_file")
fi
env "LD_LIBRARY_PATH=$(dirname -- "$layersplit"):$wavefront_cuda_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "${driver_args[@]}" \
    >"$capture/gemma-controller.stdout" \
    2>"$capture/gemma-controller.stderr" &
driver_pid=$!

driver_ready=0
for _ in $(seq 1 3600); do
    if [[ -e $driver_ready_file ]]; then
        driver_ready=1
        break
    fi
    if ! kill -0 "$driver_pid" 2>/dev/null \
            || ! kill -0 "$gate_pid" 2>/dev/null \
            || ! kill -0 "$worker_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $driver_ready -ne 1 ]]; then
    echo "persistent Gemma CPU tail did not complete warmup" >&2
    exit 1
fi
python3 \
    "$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/capture_process_affinity.py" \
    --entry "gpu-stage:$worker_pid:$stage_worker_cpus" \
    --entry "wavefront-gate:$gate_pid:$gate_cpus" \
    --output "$wavefront_affinity_receipt"

touch -- "$qwen_release_file"
qwen_paid_ready=0
for _ in $(seq 1 3600); do
    if [[ -s $qwen_paid_ready_file ]]; then
        qwen_paid_ready=1
        break
    fi
    if ! kill -0 "$qwen_pid" 2>/dev/null \
            || ! kill -0 "$driver_pid" 2>/dev/null \
            || ! kill -0 "$gate_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $qwen_paid_ready -ne 1 ]]; then
    echo "Qwen did not reach the paid preparation barrier" >&2
    exit 1
fi
python3 -c \
    'import os,sys,time; from pathlib import Path; path=sys.argv[1]; temporary=f"{path}.tmp-{os.getpid()}"; Path(temporary).write_text(f"{time.monotonic_ns()}\n", encoding="ascii"); os.replace(temporary, path)' \
    "$arm_file"
candidate_ready=0
for _ in $(seq 1 3600); do
    if [[ -s $candidate_ready_file ]]; then
        candidate_ready=1
        break
    fi
    if ! kill -0 "$qwen_pid" 2>/dev/null \
            || ! kill -0 "$driver_pid" 2>/dev/null \
            || ! kill -0 "$gate_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $candidate_ready -ne 1 ]]; then
    echo "Gemma candidate did not become ready before paid Qwen" >&2
    exit 1
fi
python3 -c \
    'import os,sys,time; from pathlib import Path; path=sys.argv[1]; temporary=f"{path}.tmp-{os.getpid()}"; Path(temporary).write_text(f"{time.monotonic_ns()}\n", encoding="ascii"); os.replace(temporary, path)' \
    "$candidate_prepare_file"
candidate_prepared=0
for _ in $(seq 1 3600); do
    if [[ -s $candidate_prepared_file ]]; then
        candidate_prepared=1
        break
    fi
    if ! kill -0 "$qwen_pid" 2>/dev/null \
            || ! kill -0 "$driver_pid" 2>/dev/null \
            || ! kill -0 "$gate_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $candidate_prepared -ne 1 ]]; then
    echo "Gemma candidate did not complete pre-boundary preparation" >&2
    exit 1
fi
python3 -c \
    'import os,sys,time; from pathlib import Path; path=sys.argv[1]; temporary=f"{path}.tmp-{os.getpid()}"; Path(temporary).write_text(f"{time.monotonic_ns()}\n", encoding="ascii"); os.replace(temporary, path)' \
    "$qwen_paid_release_file"
set +e
wait "$qwen_pid"
qwen_rc=$?
qwen_pid=
wait "$driver_pid"
driver_rc=$?
driver_pid=
wait "$gate_pid"
gate_rc=$?
gate_pid=
wait "$worker_pid"
worker_rc=$?
worker_pid=
set -e
stop_mps
if [[ $qwen_rc -ne 0 || $driver_rc -ne 0 || $gate_rc -ne 0 \
        || $worker_rc -ne 0 \
        || ! -f $output/RESULT.json \
        || ! -f ${output}.phone-capture/PHONE_ENERGY_V3.json \
        || ! -f $gate_result || ! -f $driver_result ]]; then
    echo "GPU-prefix arm failed: qwen=$qwen_rc driver=$driver_rc gate=$gate_rc worker=$worker_rc" >&2
    exit 1
fi

python3 - "$mps_log/server.log" "$mps_log/control.log" \
        "$gpu_snapshot" "$stage_worker_pid" \
        "$qwen_mps_active_thread_pct" "$stage_mps_active_thread_pct" \
        "$qwen_mps_client_priority" "$stage_mps_client_priority" \
        "$mps_receipt" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

server_path = Path(sys.argv[1])
control_path = Path(sys.argv[2])
gpu_path = Path(sys.argv[3])
stage_worker_pid = int(sys.argv[4])
qwen_active_thread_percentage = int(sys.argv[5])
stage_active_thread_percentage = int(sys.argv[6])
qwen_client_priority = int(sys.argv[7])
stage_client_priority = int(sys.argv[8])
output_path = Path(sys.argv[9])
server = server_path.read_text(encoding="ascii")
control = control_path.read_text(encoding="ascii")
gpu_uuid = gpu_path.read_text(encoding="ascii").split(",", 1)[0].strip()
connected_pids = sorted({
    int(value)
    for value in re.findall(r"Client \{PID: ([0-9]+), Context ID: [0-9]+\} connected", server)
})
priority_by_pid = {}
for pid_text, priority_text in re.findall(
    r"Priority level of client \{([0-9]+)\} is: ([01]) ", server
):
    pid = int(pid_text)
    priority = int(priority_text)
    if pid in priority_by_pid and priority_by_pid[pid] != priority:
        raise SystemExit("CUDA MPS client priority changed")
    priority_by_pid[pid] = priority
qwen_client_pids = [pid for pid in connected_pids if pid != stage_worker_pid]
device_uuids = sorted(set(re.findall(r"\(uuid (GPU-[^)]+)\) is associated", server)))
version = re.search(r"CUDA MPS Control binary version: ([0-9]+)", control)
if (
    stage_worker_pid not in connected_pids
    or len(connected_pids) != 2
    or len(qwen_client_pids) != 1
    or priority_by_pid.get(stage_worker_pid) != stage_client_priority
    or priority_by_pid.get(qwen_client_pids[0]) != qwen_client_priority
    or gpu_uuid not in device_uuids
    or version is None
    or "Invalid CUDA_VISIBLE_DEVICES" in server
    or "Server has been notified to shutdown" not in server
    or "Exiting" not in server
    or "Exit with status 0" not in control
):
    raise SystemExit("CUDA MPS lifecycle receipt is invalid")
value = {
    "client_priorities": {
        str(pid): priority_by_pid[pid] for pid in connected_pids
    },
    "connected_pids": connected_pids,
    "control_log_sha256": hashlib.sha256(control.encode("ascii")).hexdigest(),
    "control_version": int(version.group(1)),
    "device_uuids": device_uuids,
    "gpu_uuid": gpu_uuid,
    "qwen_active_thread_percentage": qwen_active_thread_percentage,
    "qwen_client_pids": qwen_client_pids,
    "qwen_client_priority": qwen_client_priority,
    "schema": "s42-cuda-mps-receipt-v2",
    "server_log_sha256": hashlib.sha256(server.encode("ascii")).hexdigest(),
    "stage_worker_pid": stage_worker_pid,
    "stage_active_thread_percentage": stage_active_thread_percentage,
    "stage_client_priority": stage_client_priority,
    "status": "PASS",
}
output_path.write_text(
    json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY

python3 "$here/analyze_gpu_prefix_wavefront_run.py" \
    --mode "$mode" \
    --qwen-result "$output/RESULT.json" \
    --phone-energy "${output}.phone-capture/PHONE_ENERGY_V3.json" \
    --gate-result "$gate_result" \
    --driver-result "$driver_result" \
    --profile "$runtime_profile" \
    --stage-manifest "$stage_manifest" \
    --bridge-log "${output}.phone-capture/bridge.stderr" \
    --router-log "${output}.phone-capture/router.log" \
    --phone-session-log "${output}.phone-capture/session.log" \
    --phone-workers-log "${output}.phone-capture/resident-workers.log" \
    --driver-log "$capture/gemma-driver.stderr" \
    --worker-log "$capture/stage-worker.stderr" \
    --mps-receipt "$mps_receipt" \
    --qwen-affinity-receipt "$qwen_affinity_receipt" \
    --wavefront-affinity-receipt "$wavefront_affinity_receipt" \
    --server-log "${output}.phone-capture/server.stderr" \
    --expected-qwen-server "$qwen_server" \
    --expected-qwen-gpu-layers "$qwen_gpu_layers" \
    --expected-qwen-log-verbosity "$qwen_log_verbosity" \
    --expected-qwen-cpus "$qwen_cpus" \
    --expected-phone-bridge-cpus "$phone_bridge_cpus" \
    --expected-stage-worker-cpus "$stage_worker_cpus" \
    --expected-wavefront-gate-cpus "$gate_cpus" \
    --expected-qwen-tail-fence-layer "$qwen_tail_fence_layer" \
    --expected-qwen-tail-fence-join-layer "$qwen_tail_fence_join_layer" \
    --expected-prefix-layers "$prefix_layers" \
    --expected-gemma-index 50 \
    --expected-gemma-cpus "$gemma_cpus" \
    --expected-gemma-protected-cpus "$gemma_protected_cpus" \
    --expected-gemma-threads "$gemma_threads" \
    --expected-gemma-ubatch "$gemma_ubatch" \
    --expected-gemma-prefill-mode "$gemma_prefill_mode" \
    --expected-maximum-backfills "$maximum_backfills" \
    --expected-prepare-max-replays "$prepare_max_replays" \
    --expected-prepare-required-consecutive "$prepare_required_consecutive" \
    --expected-phone-ffn-timeout-ms "$phone_ffn_timeout_ms" \
    --expected-phone-residency-layout "$phone_residency_layout" \
    --expected-phone-vmem-mib "$phone_ffn_vmem" \
    --expected-phone-min-available-kib "$phone_minimum_available_kib" \
    --expected-qwen-mps-active-thread-pct "$qwen_mps_active_thread_pct" \
    --expected-stage-mps-active-thread-pct "$stage_mps_active_thread_pct" \
    --expected-qwen-mps-client-priority "$qwen_mps_client_priority" \
    --expected-stage-mps-client-priority "$stage_mps_client_priority" \
    --output "$capture/GPU_PREFIX_RUN_V1.json"
sha256sum \
    "$qwen_bridge" "$qwen_server" "$layersplit" "$stage_manifest" \
    "$output/RESULT.json" \
    "${output}.phone-capture/PHONE_ENERGY_V3.json" \
    "${output}.phone-capture/bridge.stderr" \
    "${output}.phone-capture/router.log" \
    "${output}.phone-capture/session.log" \
    "${output}.phone-capture/resident-workers.log" \
    "$runtime_profile" "$gate_result" "$driver_result" \
    "$mps_receipt" \
    "$qwen_affinity_receipt" "$wavefront_affinity_receipt" \
    "$capture/gemma-driver.stderr" \
    "$capture/stage-worker.stderr" \
    "$capture/GPU_PREFIX_RUN_V1.json" \
    >"$capture/SHA256SUMS.txt"
trap - EXIT INT TERM
cleanup
cat "$capture/GPU_PREFIX_RUN_V1.json"
