#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <control|op15> <repeat-index> <output-root> <adb-port>" >&2
    exit 2
fi

arm=$1
repeat_index=$2
output=$3
adb_port=$4

if [[ $arm != control && $arm != op15 ]]; then
    echo "invalid arm: $arm" >&2
    exit 2
fi
if [[ $output != /* || -e $output || -e ${output}.phone-capture ]]; then
    echo "invalid output root: $output" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(cd -- "$here/../../../.." && pwd)
legacy_root=${S42_LEGACY_ROOT:-/home/zhihao/s41-dynamic-ffn-v1}
runner=${S42_GPU_OVERFLOW_RUNNER:-$repo_root/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/mixed_scheduler_v1/run_gpu_cold_trace.py}
burst_dir=${S42_BURST_DIR:-$legacy_root/campaign/server_trace_v2}
trace=$legacy_root/campaign/input/REQUESTS_SEMANTIC_SOURCE.jsonl
server=$legacy_root/build-server-ffn-cuda/bin/llama-server
model=/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf
lib_dir=/mnt/storage/s21_deps/cuda-13.2.1/lib
bridge=${S42_GPU_OVERFLOW_BRIDGE:-$legacy_root/ffn_dmabuf_bridge-split-reset-qualified}
serial=3C15AU002CL00000
capture=${output}.phone-capture
plan=$capture/EXECUTION_PLAN.json

phone_base=/data/local/tmp/s41-opoffload-dmabuf-v1
phone_root=$phone_base/gpu-overflow-${arm}-r${repeat_index}
phone_logger=$phone_base/phone_power_logger.sh
phone_policy=$phone_base/phone_power_allow.rules
phone_samples=$phone_root/samples.tsv
phone_active=$phone_root/active
phone_armed=$phone_root/power.armed
phone_done=$phone_root/power.done
phone_model=$phone_base/gemma-4-12B-Q40-dequant-f16.gguf
phone_session=$phone_base/phone_ffn_session_staged.sh
phone_worker=${S42_GPU_OVERFLOW_PHONE_WORKER:-$phone_base/llama-ffn-split-worker-staged}
restore_usb=$phone_base/restore_phone_usb.sh

mkdir -p "$capture"
adb_cmd=(adb -P "$adb_port" -s "$serial")
"${adb_cmd[@]}" get-state >/dev/null

scheduler_mode=control
execution_mode=cuda-cpu
split_policy_variant=${S42_SPLIT_POLICY_VARIANT:-qualified}
if [[ $split_policy_variant != qualified && $split_policy_variant != shape-balanced ]]; then
    echo "invalid split policy variant: $split_policy_variant" >&2
    exit 2
fi
if [[ $arm == op15 ]]; then
    scheduler_mode=shadow
    execution_mode=cuda-cpu-op15
fi

planner=(
    python3 "$here/plan_gpu_overflow.py"
    --mode "$scheduler_mode"
    --trace "$trace"
    --server "$server"
    --model "$model"
    --lib-dir "$lib_dir"
    --adb-port "$adb_port"
    --phone-serial "$serial"
    --split-policy-variant "$split_policy_variant"
    --output "$plan"
)
if [[ $arm == op15 ]]; then
    planner+=(
        --bridge "$bridge"
        --phone-model "$phone_model"
        --phone-session "$phone_session"
        --phone-worker "$phone_worker"
        --restore-usb "$restore_usb"
    )
fi
"${planner[@]}" >"$capture/scheduler-plan.log"

plan_cli=(python3 "$repo_root/research_dev/scheduler/plan_cli.py")
"${plan_cli[@]}" validate --plan "$plan" --mode "$execution_mode" >/dev/null
plan_get() {
    "${plan_cli[@]}" get --plan "$plan" --field "$1"
}

server=$(plan_get artifact.cold_server.path)
model=$(plan_get artifact.cold_model.path)
trace=$(plan_get runtime.trace_path)
lib_dir=$(plan_get runtime.lib_dir)
n_gpu_layers=$(plan_get placement.runtime_gpu_layers)
context=$(plan_get runtime.context)
parallel=$(plan_get runtime.parallel)
batch_size=$(plan_get runtime.batch_size)
ubatch_size=$(plan_get runtime.ubatch_size)
port=$(plan_get runtime.port)

"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
"${adb_cmd[@]}" shell \
    "su -c 'rm -rf $phone_root; mkdir $phone_root'"

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-before.json"

nohup "${adb_cmd[@]}" shell \
    "su -c 'sh $phone_logger $phone_samples $phone_active $phone_armed $phone_done 180'" \
    >"$capture/power-adb.log" 2>&1 &
power_adb_pid=$!

armed=0
for _ in $(seq 1 300); do
    if "${adb_cmd[@]}" shell "su -c 'test -f $phone_armed'" \
            >/dev/null 2>&1; then
        armed=1
        break
    fi
    sleep 0.1
done
if [[ $armed -ne 1 ]]; then
    echo "phone power logger did not arm" >&2
    exit 1
fi

session_adb_pid=
if [[ $arm == control ]]; then
    "${adb_cmd[@]}" shell "su -c 'touch $phone_active'"
else
    bridge=$(plan_get artifact.bridge.path)
    phone_session=$(plan_get artifact.phone_session.path)
    phone_worker=$(plan_get artifact.phone_worker.path)
    phone_model=$(plan_get artifact.phone_model.path)
    restore_usb=$(plan_get artifact.restore_usb.path)
    phone_layers=$(plan_get offload.layer_spec)
    phone_columns=$(plan_get offload.max_columns)
    phone_backend=$(plan_get offload.compute_backend)
    phone_timeout=$(plan_get offload.session_timeout_s)
    phone_max_requests=$(plan_get offload.max_requests)
    phone_max_tokens=$(plan_get offload.max_tokens)
    phone_column_quantum=$(plan_get offload.column_quantum)
    phone_alternate_columns=$(plan_get offload.alternate_columns)
    phone_staged_dmabuf=$(plan_get offload.staged_dmabuf)
    nohup "${adb_cmd[@]}" shell \
        "su -c 'S41_FFN_F16_IO=1 S41_FFN_MAX_TOKENS=$phone_max_tokens S41_FFN_COLUMN_QUANTUM=$phone_column_quantum S41_FFN_ALTERNATE_COLUMNS=$phone_alternate_columns S41_FFN_STAGED_DMABUF=$phone_staged_dmabuf sh $phone_session $phone_worker $phone_model $phone_layers $phone_columns $phone_backend $phone_root $restore_usb $phone_timeout $phone_max_requests'" \
        >"$capture/session-adb.log" 2>&1 &
    session_adb_pid=$!
    aoa_ready=0
    for _ in $(seq 1 300); do
        if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
            aoa_ready=1
            break
        fi
        sleep 1
    done
    if [[ $aoa_ready -ne 1 ]]; then
        echo "phone AOA session did not become ready" >&2
        exit 1
    fi
fi

command=(
    python3 "$runner"
    --mode "$execution_mode"
    --repeat-index "$repeat_index"
    --scheduler-plan "$plan"
    --requests "$trace"
    --server "$server"
    --model "$model"
    --model-profile fp16-proxy
    --lib-dir "$lib_dir"
    --port "$port"
    --ctx-size "$context"
    --parallel "$parallel"
    --batch-size "$batch_size"
    --ubatch-size "$ubatch_size"
    --n-gpu-layers "$n_gpu_layers"
    --cache-type-k f16
    --cache-type-v f16
    --arrival-mode backlog
    --dispatch-order source
    --require-server-energy
    --output "$output"
    --execute
    --confirm RUN_GEMMA_GPU_TRACE
)
if [[ $arm == op15 ]]; then
    command+=(--bridge "$bridge")
fi

set +e
S42_UNIFIED_REPO_ROOT="$repo_root" S41_BURST_DIR="$burst_dir" \
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
systemd-run --user --scope -p MemorySwapMax=0 "${command[@]}" \
    >"$capture/runner.log" 2>&1
run_rc=$?
set -e

if [[ $arm == control ]]; then
    "${adb_cmd[@]}" shell "su -c 'rm -f $phone_active'"
fi
if [[ -n $session_adb_pid ]]; then
    wait "$session_adb_pid" || true
fi

device_ready=0
for _ in $(seq 1 180); do
    if "${adb_cmd[@]}" get-state >/dev/null 2>&1; then
        device_ready=1
        break
    fi
    sleep 1
done
if [[ $device_ready -ne 1 ]]; then
    echo "phone did not restore ADB" >&2
    exit 1
fi

wait "$power_adb_pid"
"${adb_cmd[@]}" shell "su -c 'test -f $phone_done'"
python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-after.json"
"${adb_cmd[@]}" pull "$phone_samples" "$capture/phone-samples.tsv" \
    >"$capture/pull.log"

if [[ $run_rc -ne 0 || ! -f $output/RESULT.json ]]; then
    echo "GPU overflow runner failed: rc=$run_rc" >&2
    exit "$run_rc"
fi

python3 "$burst_dir/analyze_phone_energy.py" \
    --result "$output/RESULT.json" \
    --samples "$capture/phone-samples.tsv" \
    --clock-before "$capture/clock-before.json" \
    --clock-after "$capture/clock-after.json" \
    --output "$capture/PHONE_ENERGY_V3.json"

sha256sum \
    "$plan" \
    "$output/RESULT.json" \
    "$capture/PHONE_ENERGY_V3.json" \
    "$capture/phone-samples.tsv" \
    >"$capture/SHA256SUMS.txt"
cat "$capture/SHA256SUMS.txt"
