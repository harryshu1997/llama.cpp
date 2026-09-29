#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 <control|mechanics|qualified> <repeat-index> <output-root> <adb-port> <profile-template>" >&2
    exit 2
fi

mode=$1
repeat_index=$2
output=$3
adb_port=$4
profile_template=$5
if [[ $mode != control && $mode != mechanics && $mode != qualified ]]; then
    echo "invalid wavefront mode: $mode" >&2
    exit 2
fi
if [[ ! $repeat_index =~ ^[0-9]+$ || $repeat_index -eq 0 \
        || $output != /* || -e $output \
        || -e ${output}.phone-capture || -e ${output}.wavefront \
        || $profile_template != /* || ! -f $profile_template ]]; then
    echo "invalid wavefront arm arguments" >&2
    exit 2
fi
if [[ ${S42_ENABLE_GPU_WAVEFRONT_PHYSICAL:-} != YES ]]; then
    echo "physical wavefront execution is disabled; set S42_ENABLE_GPU_WAVEFRONT_PHYSICAL=YES" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=${S42_UNIFIED_REPO_ROOT:-$(cd -- "$here/../../../../.." && pwd)}
legacy_root=${S42_LEGACY_ROOT:-/home/zhihao/s41-dynamic-ffn-v1}
qwen_arm=${S42_QWEN_ARM:-$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/run_qwen_full_energy_arm.sh}
qwen_bridge=${S42_WAVEFRONT_BRIDGE:?set S42_WAVEFRONT_BRIDGE}
lm_worker=${S42_LM_HEAD_WORKER:?set S42_LM_HEAD_WORKER}
layersplit=${S42_LAYERSPLIT:?set S42_LAYERSPLIT}
gemma_model=${S42_GEMMA_MODEL:-/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf}
trace=${S42_BURSTGPT_REQUESTS:-$legacy_root/campaign/input/REQUESTS_SEMANTIC_SOURCE.jsonl}
wavefront_cuda_lib_dir=${S42_WAVEFRONT_CUDA_LIB_DIR:-/home/zhihao/s43-wan-pilot/cuda12-lib}
lm_head_weight_sha256=${S42_LM_HEAD_WEIGHT_SHA256:?set S42_LM_HEAD_WEIGHT_SHA256}
gemma_indices=${S42_WAVEFRONT_GEMMA_INDICES:-50}
qwen_indices=${S42_WAVEFRONT_QWEN_INDICES:-52,53,31}
gemma_threads=${S42_WAVEFRONT_GEMMA_THREADS:-8}
gemma_cpus=${S42_WAVEFRONT_GEMMA_CPUS:-0,2,4,6,8,10,12,14}
gemma_ubatch=${S42_WAVEFRONT_GEMMA_UBATCH:-512}
qwen_gpu_layers=${S42_WAVEFRONT_QWEN_GPU_LAYERS:-15}
maximum_backfills=${S42_WAVEFRONT_MAX_BACKFILLS:-1}
lm_head_rows=${S42_WAVEFRONT_LM_HEAD_ROWS:-262112}
lm_head_top_k=${S42_WAVEFRONT_LM_HEAD_TOP_K:-32}
phone_run_tag=${S42_WAVEFRONT_RUN_TAG:-gpu-wavefront-${mode}-r${repeat_index}}
phone_filler_port=${S42_WAVEFRONT_PHONE_FILLER_PORT:-}
phone_router=${S42_WAVEFRONT_PHONE_ROUTER:-}
phone_ffn_layers=${S42_WAVEFRONT_PHONE_FFN_LAYERS:-0-22}
phone_ffn_columns=${S42_WAVEFRONT_PHONE_FFN_COLUMNS:-6144}
phone_ffn_prefill_columns=${S42_WAVEFRONT_PHONE_FFN_PREFILL_COLUMNS:-0}
phone_ffn_timeout_ms=${S42_WAVEFRONT_PHONE_FFN_TIMEOUT_MS:-120000}
phone_ffn_vmem=${S42_PHONE_FFN_VMEM:-3328}
phone_minimum_available_kib=${S42_PHONE_MIN_AVAILABLE_KIB:-2097152}
capture=${output}.wavefront

if [[ $mode != control ]]; then
    profile_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile_id"])' "$profile_template")
    if [[ $profile_id == fp16-wavefront-control-only-v1 ]]; then
        echo "treatment requires a calibrated profile, not the control-only template" >&2
        exit 2
    fi
fi
if [[ ! $gemma_threads =~ ^[0-9]+$ || $gemma_threads -eq 0 \
        || ! $gemma_cpus =~ ^[0-9,-]+$ \
        || ! $gemma_ubatch =~ ^[0-9]+$ || $gemma_ubatch -eq 0 \
        || $gemma_ubatch -gt 512 \
        || ! $qwen_gpu_layers =~ ^[0-9]+$ || $qwen_gpu_layers -gt 41 \
        || ! $maximum_backfills =~ ^[0-9]+$ \
        || ! $lm_head_rows =~ ^[0-9]+$ || $lm_head_rows -eq 0 \
        || ! $lm_head_top_k =~ ^[0-9]+$ || $lm_head_top_k -eq 0 \
        || $lm_head_top_k -gt $lm_head_rows \
        || ! $phone_ffn_vmem =~ ^[0-9]+$ \
        || $phone_ffn_vmem -lt 3200 || $phone_ffn_vmem -gt 3328 \
        || ! $phone_minimum_available_kib =~ ^[0-9]+$ \
        || $phone_minimum_available_kib -lt 2097152 ]]; then
    echo "invalid wavefront runtime bounds" >&2
    exit 2
fi
if ! command -v taskset >/dev/null \
        || ! taskset --cpu-list "$gemma_cpus" true >/dev/null 2>&1; then
    echo "invalid or unavailable Gemma CPU affinity" >&2
    exit 2
fi
if [[ $mode == mechanics && $maximum_backfills -eq 0 ]]; then
    echo "mechanics mode requires a bounded positive backfill count" >&2
    exit 2
fi
phone_max_sessions=1
if [[ -n $phone_filler_port ]]; then
    phone_max_sessions=0
    if [[ ! $phone_filler_port =~ ^[0-9]+$ \
            || $phone_filler_port -le 0 || $phone_filler_port -gt 65535 \
            || -z $phone_router || $phone_router != /* \
            || ! $phone_ffn_columns =~ ^[0-9]+$ \
            || $phone_ffn_columns -ne 6144 \
            || ! $phone_ffn_prefill_columns =~ ^[0-9]+$ \
            || $phone_ffn_prefill_columns -ne 0 \
                && $phone_ffn_prefill_columns -ne $phone_ffn_columns \
            || ! $phone_ffn_timeout_ms =~ ^[0-9]+$ \
            || $phone_ffn_timeout_ms -eq 0 \
            || $phone_ffn_timeout_ms -gt 600000 \
            || $phone_ffn_layers != 0-22 ]]; then
        echo "invalid wavefront phone arbiter route" >&2
        exit 2
    fi
fi

for path in \
        "$qwen_arm" "$qwen_bridge" "$lm_worker" "$layersplit" \
        "$gemma_model" "$trace" "$wavefront_cuda_lib_dir" \
        "$here/gpu_wavefront_gate.py" \
        "$here/run_gemma_wavefront_driver.py" \
        "$here/materialize_wavefront_profile.py" \
        "$here/analyze_wavefront_run.py"; do
    if [[ ! -e $path ]]; then
        echo "missing wavefront dependency: $path" >&2
        exit 1
    fi
done
read -r template_rows template_top_k < <(
    python3 -c '
import json
import sys

worker = json.load(open(sys.argv[1], encoding="ascii"))["worker"]
print(worker["rows"], worker["top_k"])
' "$profile_template"
)
if [[ $lm_head_rows -ne $template_rows \
        || $lm_head_top_k -ne $template_top_k ]]; then
    echo "LM-head runtime geometry differs from the profile template" >&2
    exit 2
fi
if pgrep -f '^.*/llama-lm-head-split-worker' >/dev/null \
        || pgrep -f '^.*/llama-layersplit .*--mode overlapdriver' >/dev/null; then
    echo "another wavefront worker is active" >&2
    pgrep -af 'llama-lm-head-split-worker|llama-layersplit' >&2 || true
    exit 1
fi

mkdir "$capture"
runtime_dir=$(mktemp -d /tmp/s42-gpu-wavefront.XXXXXX)
fence_socket=$runtime_dir/fence.sock
arm_file=$runtime_dir/paid.arm
qwen_ready_file=$runtime_dir/qwen.ready
qwen_release_file=$runtime_dir/qwen.release
qwen_complete_file=$runtime_dir/qwen.complete
paid_tail_file=$runtime_dir/paid.tail
driver_ready_file=$runtime_dir/gemma.ready
runtime_profile=$capture/RUNTIME_PROFILE.json
gpu_snapshot=$capture/GPU_SNAPSHOT.csv
gate_result=$capture/GATE_RESULT.json
driver_result=$capture/GEMMA_RESULT.json
worker_port=$((19060 + repeat_index))
gate_port=$((19160 + repeat_index))

qwen_pid=
worker_pid=
gate_pid=
driver_pid=
cleanup() {
    for process_id in "$driver_pid" "$gate_pid" "$worker_pid" "$qwen_pid"; do
        if [[ -n $process_id ]] && kill -0 "$process_id" 2>/dev/null; then
            kill -TERM "$process_id" 2>/dev/null || true
            wait "$process_id" 2>/dev/null || true
        fi
    done
    rm -f -- \
        "$fence_socket" "$arm_file" "$qwen_ready_file" \
        "$qwen_release_file" "$qwen_complete_file" "$paid_tail_file" \
        "$driver_ready_file"
    rmdir -- "$runtime_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

S42_QWEN_BRIDGE="$qwen_bridge" \
S42_QWEN_FILLER_PORT="$phone_filler_port" \
S42_PHONE_ROUTER="$phone_router" \
S42_PHONE_MAX_SESSIONS="$phone_max_sessions" \
S42_PHONE_FFN_VMEM="$phone_ffn_vmem" \
S42_PHONE_MIN_AVAILABLE_KIB="$phone_minimum_available_kib" \
S42_QWEN_GPU_LAYERS="$qwen_gpu_layers" \
S42_QWEN_INDICES="$qwen_indices" \
S42_PREFETCH_ARM_FILE="$arm_file" \
S42_FFN_PREFETCH_FENCE_SOCKET="$fence_socket" \
S42_FFN_PREFETCH_FENCE_GROUP_FIRST_LAYER=0 \
S42_FFN_PREFETCH_FENCE_GROUP_LAST_LAYER=11 \
S42_QWEN_SERVER_READY_FILE="$qwen_ready_file" \
S42_QWEN_SERVER_RELEASE_FILE="$qwen_release_file" \
S42_QWEN_COMPLETE_FILE="$qwen_complete_file" \
S42_PHONE_PROTECTED_DONE_FILE="$qwen_complete_file" \
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
    echo "Qwen server did not reach the wavefront setup barrier" >&2
    exit 1
fi

env "LD_LIBRARY_PATH=$wavefront_cuda_lib_dir:$(dirname -- "$lm_worker")" \
    CUDA_VISIBLE_DEVICES=0 \
    "$lm_worker" \
    -m "$gemma_model" --backend CUDA0 \
    --bind 127.0.0.1 --port "$worker_port" \
    --rows "$lm_head_rows" --top-k "$lm_head_top_k" --f16-io \
    >"$capture/lm-worker.stdout" 2>"$capture/lm-worker.stderr" &
worker_pid=$!

worker_ready=0
for _ in $(seq 1 3600); do
    if grep -q '^\[lm-head-worker\] ready ' \
            "$capture/lm-worker.stderr" 2>/dev/null; then
        worker_ready=1
        break
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $worker_ready -ne 1 ]]; then
    echo "resident CUDA LM-head worker did not become ready" >&2
    exit 1
fi

nvidia-smi --query-gpu=uuid,memory.total,memory.free \
    --format=csv,noheader,nounits >"$gpu_snapshot"
python3 "$here/materialize_wavefront_profile.py" \
    --template "$profile_template" \
    --gpu-snapshot "$gpu_snapshot" \
    --worker-log "$capture/lm-worker.stderr" \
    --lm-head-weight-sha256 "$lm_head_weight_sha256" \
    --output "$runtime_profile"

gate_args=(
    python3 "$here/gpu_wavefront_gate.py"
    --mode "$mode"
    --profile "$runtime_profile"
    --listen-host 127.0.0.1 --listen-port "$gate_port"
    --worker-host 127.0.0.1 --worker-port "$worker_port"
    --fence-socket "$fence_socket"
    --arm-file "$arm_file"
    --qwen-complete-file "$qwen_complete_file"
    --paid-tail-file "$paid_tail_file"
    --output "$gate_result"
    --pipeline-id "gemma-burstgpt-${gemma_indices//,/-}"
    --timeout-s 7200
)
if [[ $mode == mechanics ]]; then
    gate_args+=(--max-backfills "$maximum_backfills")
fi
"${gate_args[@]}" \
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
    echo "unified GPU wavefront gate did not become ready" >&2
    exit 1
fi

driver_args=(
    python3 "$here/run_gemma_wavefront_driver.py"
    --layersplit "$layersplit" --model "$gemma_model" \
    --requests "$trace" --indices "$gemma_indices" \
    --arm-file "$arm_file" --ready-file "$driver_ready_file" \
    --gate-host 127.0.0.1 --gate-port "$gate_port" \
    --rows "$lm_head_rows" --top-k "$lm_head_top_k" \
    --threads "$gemma_threads" \
    --cpu-list "$gemma_cpus" \
    --ubatch "$gemma_ubatch" \
    --library-path "$wavefront_cuda_lib_dir" \
    --stderr "$capture/gemma-driver.stderr" \
    --output "$driver_result"
)
if [[ -n $phone_filler_port ]]; then
    driver_args+=(
        --ffn-host 127.0.0.1 --ffn-port "$phone_filler_port"
        --ffn-layers "$phone_ffn_layers"
        --ffn-columns "$phone_ffn_columns"
        --ffn-prefill-columns "$phone_ffn_prefill_columns"
        --ffn-decode-columns "$phone_ffn_columns"
        --ffn-timeout-ms "$phone_ffn_timeout_ms"
        --ffn-f16-io
    )
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
    echo "persistent CPU Gemma producer did not complete warmup" >&2
    exit 1
fi

touch -- "$qwen_release_file"
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
set -e

kill -TERM "$worker_pid" 2>/dev/null || true
wait "$worker_pid" 2>/dev/null || true
worker_pid=
if [[ $qwen_rc -ne 0 || $driver_rc -ne 0 || $gate_rc -ne 0 \
        || ! -f $output/RESULT.json \
        || ! -f ${output}.phone-capture/PHONE_ENERGY_V3.json \
        || ! -f $gate_result || ! -f $driver_result ]]; then
    echo "wavefront arm failed: qwen=$qwen_rc driver=$driver_rc gate=$gate_rc" >&2
    exit 1
fi

python3 "$here/analyze_wavefront_run.py" \
    --mode "$mode" \
    --qwen-result "$output/RESULT.json" \
    --phone-energy "${output}.phone-capture/PHONE_ENERGY_V3.json" \
    --gate-result "$gate_result" \
    --driver-result "$driver_result" \
    --profile "$runtime_profile" \
    --bridge-log "${output}.phone-capture/bridge.stderr" \
    --router-log "${output}.phone-capture/router.log" \
    --phone-session-log "${output}.phone-capture/session.log" \
    --phone-workers-log "${output}.phone-capture/resident-workers.log" \
    --driver-log "$capture/gemma-driver.stderr" \
    --server-log "${output}.phone-capture/server.stderr" \
    --expected-qwen-gpu-layers "$qwen_gpu_layers" \
    --expected-gemma-cpus "$gemma_cpus" \
    --expected-gemma-threads "$gemma_threads" \
    --expected-gemma-ubatch "$gemma_ubatch" \
    --expected-phone-ffn-prefill-columns "$phone_ffn_prefill_columns" \
    --expected-phone-ffn-timeout-ms "$phone_ffn_timeout_ms" \
    --expected-phone-vmem-mib "$phone_ffn_vmem" \
    --expected-phone-min-available-kib "$phone_minimum_available_kib" \
    --output "$capture/WAVEFRONT_RUN_V1.json"
sha256sum \
    "$qwen_bridge" "$lm_worker" "$layersplit" \
    "$output/RESULT.json" \
    "${output}.phone-capture/PHONE_ENERGY_V3.json" \
    "${output}.phone-capture/bridge.stderr" \
    "${output}.phone-capture/router.log" \
    "${output}.phone-capture/session.log" \
    "${output}.phone-capture/resident-workers.log" \
    "$runtime_profile" "$gate_result" "$driver_result" \
    "$capture/gemma-driver.stderr" \
    "$capture/WAVEFRONT_RUN_V1.json" \
    >"$capture/SHA256SUMS.txt"
trap - EXIT INT TERM
cleanup
cat "$capture/WAVEFRONT_RUN_V1.json"
