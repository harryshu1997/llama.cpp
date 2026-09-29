#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <cpu|op15> <repeat-index> <output-root> <adb-port>" >&2
    exit 2
fi

mode=$1
repeat_index=$2
output=$3
adb_port=$4

if [[ $mode != cpu && $mode != op15 ]]; then
    echo "invalid mode: $mode" >&2
    exit 2
fi
if [[ $output != /* || -e $output || -e ${output}.phone-capture ]]; then
    echo "invalid output root: $output" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
serial=3C15AU002CL00000
capture=${output}.phone-capture
scheduler_plan=${S42_EXECUTION_PLAN:-}
hot_server=/home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server
hot_model=/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf
cuda_lib_dir=/mnt/storage/s21_deps/cuda-13.2.1/lib
cold_server=/home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin/llama-server
cold_model=/home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf
cold_lib_dir=/home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin
bridge=/home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-reset-qualified
trace=/home/zhihao/s41-dynamic-ffn-v1/campaign/input/REQUESTS_SEMANTIC_SOURCE.jsonl
policy_id=i3-hidden-wait
split_policy=1:9664,3:8192,8:4096,128:8192,512:11136
split_io=f16
max_columns=11136
hot_context=24576
hot_parallel=4
cold_context=32768
cold_parallel=8
cold_batch_size=4096
cold_ubatch_size=512
cold_threads=-1
cold_repack=default
request_workers=74

if [[ -n $scheduler_plan ]]; then
    if [[ $scheduler_plan != /* || ! -f $scheduler_plan ]]; then
        echo "invalid scheduler plan: $scheduler_plan" >&2
        exit 2
    fi
    scheduler_root=${S42_UNIFIED_REPO_ROOT:-$(cd -- "$here/../../../../.." && pwd)}
    plan_cli=(python3 "$scheduler_root/research_dev/scheduler/plan_cli.py")
    "${plan_cli[@]}" validate --plan "$scheduler_plan" --mode "$mode" >/dev/null
    plan_get() {
        "${plan_cli[@]}" get --plan "$scheduler_plan" --field "$1"
    }
    hot_server=$(plan_get artifact.hot_server.path)
    hot_model=$(plan_get artifact.hot_model.path)
    cold_server=$(plan_get artifact.cold_server.path)
    cold_model=$(plan_get artifact.cold_model.path)
    cuda_lib_dir=$(plan_get runtime.cuda_lib_dir)
    cold_lib_dir=$(plan_get runtime.cold_lib_dir)
    trace=$(plan_get runtime.trace_path)
    policy_id=$(plan_get runtime.split_policy_id)
    split_policy=$(plan_get runtime.split_table)
    split_io=$(plan_get runtime.split_io)
    max_columns=$(plan_get runtime.split_max_columns)
    hot_context=$(plan_get runtime.hot_context)
    hot_parallel=$(plan_get runtime.hot_parallel)
    cold_context=$(plan_get runtime.cold_context)
    cold_parallel=$(plan_get runtime.cold_parallel)
    cold_batch_size=$(plan_get runtime.cold_batch_size)
    cold_ubatch_size=$(plan_get runtime.cold_ubatch_size)
    cold_threads=$(plan_get runtime.cold_threads)
    cold_repack=$(plan_get runtime.cold_repack)
    request_workers=$(plan_get runtime.request_workers)
    if [[ $mode == op15 ]]; then
        bridge=$(plan_get artifact.bridge.path)
        session=$(plan_get artifact.phone_session.path)
        worker=$(plan_get artifact.phone_worker.path)
        phone_model=$(plan_get artifact.phone_model.path)
        restore=$(plan_get artifact.restore_usb.path)
        phone_layers=$(plan_get offload.layer_spec)
        phone_columns=$(plan_get offload.max_columns)
        phone_backend=$(plan_get offload.compute_backend)
        phone_timeout=$(plan_get offload.session_timeout_s)
        phone_max_requests=$(plan_get offload.max_requests)
        phone_max_tokens=$(plan_get offload.max_tokens)
        phone_column_quantum=$(plan_get offload.column_quantum)
        phone_alternate_columns=$(plan_get offload.alternate_columns)
        if [[ $(plan_get offload.io_type) != f16 ]]; then
            echo "unsupported phone I/O type in scheduler plan" >&2
            exit 2
        fi
    fi
fi

phone_root=/data/local/tmp/s41-opoffload-dmabuf-v1/i2-energy-${mode}-r${repeat_index}
phone_logger=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_logger.sh
phone_policy=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_allow.rules
phone_samples=$phone_root/samples.tsv
phone_active=$phone_root/active
phone_armed=$phone_root/power.armed
phone_done=$phone_root/power.done

mkdir -p "$capture"

adb_cmd=(adb -P "$adb_port" -s "$serial")
"${adb_cmd[@]}" get-state >/dev/null
"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
"${adb_cmd[@]}" shell \
    "su -c 'rm -rf $phone_root; mkdir $phone_root'"

python3 "$here/capture_phone_clock.py" \
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
if [[ $mode == cpu ]]; then
    "${adb_cmd[@]}" shell "su -c 'touch $phone_active'"
else
    session=${session:-/data/local/tmp/s41-opoffload-dmabuf-v1/phone_ffn_session-flex.sh}
    worker=${worker:-/data/local/tmp/s41-opoffload-dmabuf-v1/llama-ffn-split-worker-flex-v2}
    phone_model=${phone_model:-/data/local/tmp/ls-s32/gemma-4-12B-it-Q4_0.gguf}
    restore=${restore:-/data/local/tmp/s41-opoffload-dmabuf-v1/restore_phone_usb.sh}
    phone_layers=${phone_layers:-0-47}
    phone_columns=${phone_columns:-11136}
    phone_backend=${phone_backend:-HTP0}
    phone_timeout=${phone_timeout:-1800}
    phone_max_requests=${phone_max_requests:-120000}
    phone_max_tokens=${phone_max_tokens:-512}
    phone_column_quantum=${phone_column_quantum:-2048}
    phone_alternate_columns=${phone_alternate_columns:-9664}
    nohup "${adb_cmd[@]}" shell \
        "su -c 'S41_FFN_F16_IO=1 S41_FFN_MAX_TOKENS=$phone_max_tokens S41_FFN_COLUMN_QUANTUM=$phone_column_quantum S41_FFN_ALTERNATE_COLUMNS=$phone_alternate_columns sh $session $worker $phone_model $phone_layers $phone_columns $phone_backend $phone_root $restore $phone_timeout $phone_max_requests'" \
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
    python3 "$here/run_server_trace.py"
    --mode "$mode"
    --repeat-index "$repeat_index"
    --policy-id "$policy_id"
    --split-policy "$split_policy"
    --split-io "$split_io"
    --max-columns "$max_columns"
    --requests "$trace"
    --hot-server "$hot_server"
    --hot-model "$hot_model"
    --cuda-lib-dir "$cuda_lib_dir"
    --hot-ctx-size "$hot_context"
    --hot-parallel "$hot_parallel"
    --cold-server "$cold_server"
    --cold-model "$cold_model"
    --cold-lib-dir "$cold_lib_dir"
    --cold-ctx-size "$cold_context"
    --cold-parallel "$cold_parallel"
    --cold-batch-size "$cold_batch_size"
    --cold-ubatch-size "$cold_ubatch_size"
    --cold-threads "$cold_threads"
    --cold-repack "$cold_repack"
    --request-workers "$request_workers"
    --require-server-energy
    --output "$output"
    --execute
    --confirm RUN_BURSTGPT_LLAMA_SERVER_TRACE
)
if [[ $mode == op15 ]]; then
    command+=(
        --bridge "$bridge"
    )
fi
if [[ -n $scheduler_plan ]]; then
    command+=(--scheduler-plan "$scheduler_plan")
fi

set +e
systemd-run --user --scope -p MemorySwapMax=0 "${command[@]}" \
    >"$capture/runner.log" 2>&1
run_rc=$?
set -e

if [[ $mode == cpu ]]; then
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
python3 "$here/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-after.json"
"${adb_cmd[@]}" pull "$phone_samples" "$capture/phone-samples.tsv" \
    >"$capture/pull.log"

if [[ $run_rc -ne 0 || ! -f $output/RESULT.json ]]; then
    echo "trace runner failed: rc=$run_rc" >&2
    exit "$run_rc"
fi

python3 "$here/analyze_phone_energy.py" \
    --result "$output/RESULT.json" \
    --samples "$capture/phone-samples.tsv" \
    --clock-before "$capture/clock-before.json" \
    --clock-after "$capture/clock-after.json" \
    --output "$capture/PHONE_ENERGY_V3.json"

sha256sum \
    "$output/RESULT.json" \
    "$capture/PHONE_ENERGY_V3.json" \
    "$capture/phone-samples.tsv" \
    >"$capture/SHA256SUMS.txt"
cat "$capture/SHA256SUMS.txt"
