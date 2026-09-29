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
legacy_root=${S42_LEGACY_ROOT:-/home/zhihao/s41-dynamic-ffn-v1}
burst_dir=${S42_BURST_DIR:-$legacy_root/campaign/server_trace_v2}
trace=${S42_BURSTGPT_REQUESTS:-$legacy_root/campaign/input/REQUESTS_SEMANTIC_SOURCE.jsonl}
server=${S42_QWEN_SERVER:-$legacy_root/build-server-ffn-cuda/bin/llama-server}
server_bin=$(dirname -- "$server")
model=/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf
lib_dir=/mnt/storage/s21_deps/cuda-13.2.1/lib
bridge=${S42_QWEN_BRIDGE:-$legacy_root/ffn_dmabuf_bridge-swiglu-v1}
serial=3C15AU002CL00000
capture=${output}.phone-capture
bridge_port=$((17660 + repeat_index))
bridge_filler_port=${S42_QWEN_FILLER_PORT:-}
bridge_idle_lower_us=${S42_PHONE_IDLE_LOWER_US:-330000}
bridge_filler_upper_us=${S42_PHONE_FILLER_UPPER_US:-50000}
bridge_guard_us=${S42_PHONE_GUARD_US:-20000}
server_port=$((18460 + repeat_index))

phone_base=/data/local/tmp/s41-opoffload-dmabuf-v1
phone_run_tag=${S42_PHONE_RUN_TAG:-qwen-full-energy}
indices=${S42_QWEN_INDICES:-52,53,31}
dispatch=${S42_QWEN_DISPATCH:-sequential}
phone_residency_layout=${S42_PHONE_RESIDENCY_LAYOUT:-gemma23-qwen12-full-v1}
qwen_policy=${S42_QWEN_POLICY:-1:17408,512:0}
qwen_gpu_layers=${S42_QWEN_GPU_LAYERS:-18}
qwen_log_verbosity=${S42_QWEN_LOG_VERBOSITY:-1}
qwen_cpus=${S42_QWEN_CPUS:-}
phone_bridge_cpus=${S42_PHONE_BRIDGE_CPUS:-}
affinity_receipt=${S42_QWEN_AFFINITY_RECEIPT:-}
prefetch_arm_file=${S42_PREFETCH_ARM_FILE:-}
paid_ready_file=${S42_QWEN_PAID_READY_FILE:-}
paid_release_file=${S42_QWEN_PAID_RELEASE_FILE:-}
server_ready_file=${S42_QWEN_SERVER_READY_FILE:-}
server_release_file=${S42_QWEN_SERVER_RELEASE_FILE:-}
qwen_complete_file=${S42_QWEN_COMPLETE_FILE:-}
paid_tail_file=${S42_PAID_TAIL_FILE:-}
paid_tail_timeout_s=${S42_PAID_TAIL_TIMEOUT_S:-3600}
if [[ ! -x $server ]]; then
    echo "invalid Qwen server: $server" >&2
    exit 2
fi
if [[ ! $phone_run_tag =~ ^[a-z0-9-]+$ ]]; then
    echo "invalid phone run tag: $phone_run_tag" >&2
    exit 2
fi
if [[ $dispatch != sequential && $dispatch != concurrent ]]; then
    echo "invalid Qwen dispatch: $dispatch" >&2
    exit 2
fi
if [[ ! $qwen_gpu_layers =~ ^[0-9]+$ ]] \
        || (( qwen_gpu_layers > 41 )); then
    echo "invalid Qwen GPU layer count: $qwen_gpu_layers" >&2
    exit 2
fi
if [[ ! $qwen_log_verbosity =~ ^[1-5]$ ]]; then
    echo "invalid Qwen log verbosity: $qwen_log_verbosity" >&2
    exit 2
fi
if [[ -n $qwen_cpus || -n $phone_bridge_cpus || -n $affinity_receipt ]]; then
    if [[ ! $qwen_cpus =~ ^[0-9,-]+$ \
            || ! $phone_bridge_cpus =~ ^[0-9,-]+$ \
            || $affinity_receipt != /* || -e $affinity_receipt \
            || ! -f $here/capture_process_affinity.py ]]; then
        echo "invalid Qwen affinity configuration" >&2
        exit 2
    fi
    if ! command -v taskset >/dev/null \
            || ! taskset --cpu-list "$qwen_cpus" true >/dev/null 2>&1 \
            || ! taskset --cpu-list "$phone_bridge_cpus" true \
                >/dev/null 2>&1; then
        echo "invalid Qwen affinity configuration" >&2
        exit 2
    fi
fi
if [[ -n $prefetch_arm_file \
        && ( $prefetch_arm_file != /* || -e $prefetch_arm_file ) ]]; then
    echo "invalid prefetch arm file: $prefetch_arm_file" >&2
    exit 2
fi
if [[ -n $paid_ready_file || -n $paid_release_file ]]; then
    if [[ -z $paid_ready_file || -z $paid_release_file \
            || $paid_ready_file != /* || $paid_release_file != /* \
            || -e $paid_ready_file || -e $paid_release_file \
            || ! -d $(dirname -- "$paid_ready_file") \
            || ! -d $(dirname -- "$paid_release_file") ]]; then
        echo "invalid Qwen paid barrier" >&2
        exit 2
    fi
fi
if [[ -n $server_ready_file || -n $server_release_file ]]; then
    if [[ -z $server_ready_file || -z $server_release_file \
            || $server_ready_file != /* || $server_release_file != /* \
            || -e $server_ready_file || -e $server_release_file ]]; then
        echo "invalid Qwen server barrier" >&2
        exit 2
    fi
fi
if [[ -n $qwen_complete_file || -n $paid_tail_file ]]; then
    if [[ -z $qwen_complete_file || -z $paid_tail_file \
            || $qwen_complete_file != /* || $paid_tail_file != /* \
            || -e $qwen_complete_file || -e $paid_tail_file \
            || ! -d $(dirname -- "$qwen_complete_file") \
            || ! -d $(dirname -- "$paid_tail_file") ]]; then
        echo "invalid paid-tail receipt files" >&2
        exit 2
    fi
fi
phone_root=$phone_base/${phone_run_tag}-${arm}-r${repeat_index}
phone_logger=$phone_base/phone_power_logger.sh
phone_policy=$phone_base/phone_power_allow.rules
phone_samples=$phone_root/samples.tsv
phone_active=$phone_root/active
phone_armed=$phone_root/power.armed
phone_done=$phone_root/power.done
phone_router=${S42_PHONE_ROUTER:-$phone_base/resident_ffn_router-triple-v1}
phone_max_sessions=${S42_PHONE_MAX_SESSIONS:-1}
phone_ffn_vmem=${S42_PHONE_FFN_VMEM:-3328}
phone_minimum_available_kib=${S42_PHONE_MIN_AVAILABLE_KIB:-2097152}
phone_gemma=$phone_base/gemma-4-12B-Q40-dequant-f16.gguf
phone_qwen=$phone_base/Qwen3-14B-Q4KM-dequant-f16.gguf
restore_usb=$phone_base/restore_phone_usb.sh
qwen_only_phone=${S42_QWEN_ONLY_PHONE:-0}
if [[ $qwen_only_phone != 0 && $qwen_only_phone != 1 ]]; then
    echo "invalid Qwen-only phone mode: $qwen_only_phone" >&2
    exit 2
fi
case $phone_residency_layout in
    gemma23-qwen12-full-v1)
        default_phone_workers=$phone_base/resident_ffn_workers-triple-v3
        default_phone_session=$phone_base/resident_ffn_session-triple-v1.sh
        qwen_phone_layer_mask=4095
        qwen_phone_last_layer=11
        ;;
    gemma46-qwen6-full-v1)
        default_phone_workers=$phone_base/resident_ffn_workers-rebalance-v1
        default_phone_session=$phone_base/resident_ffn_session-rebalance-v1.sh
        qwen_phone_layer_mask=63
        qwen_phone_last_layer=5
        ;;
    *)
        echo "invalid phone residency layout: $phone_residency_layout" >&2
        exit 2
        ;;
esac
phone_workers=${S42_PHONE_WORKERS:-$default_phone_workers}
phone_session=${S42_PHONE_SESSION:-$default_phone_session}
if [[ $qwen_only_phone == 1 \
        && $phone_residency_layout != gemma23-qwen12-full-v1 ]]; then
    echo "Qwen-only phone mode does not support the rebalanced layout" >&2
    exit 2
fi
dual_phone_arbiter=0
if [[ ! $phone_max_sessions =~ ^[0-9]+$ ]]; then
    echo "invalid phone max sessions: $phone_max_sessions" >&2
    exit 2
fi
if [[ ! $phone_ffn_vmem =~ ^[0-9]+$ \
        || $phone_ffn_vmem -lt 3200 || $phone_ffn_vmem -gt 3328 \
        || ! $phone_minimum_available_kib =~ ^[0-9]+$ \
        || $phone_minimum_available_kib -lt 2097152 ]]; then
    echo "invalid phone resident memory geometry" >&2
    exit 2
fi
if [[ -n $bridge_filler_port ]]; then
    dual_phone_arbiter=1
    if [[ $arm != op15 || $qwen_only_phone != 0 \
            || ! $bridge_filler_port =~ ^[0-9]+$ \
            || ! $bridge_idle_lower_us =~ ^[0-9]+$ \
            || ! $bridge_filler_upper_us =~ ^[0-9]+$ \
            || ! $bridge_guard_us =~ ^[0-9]+$ ]]; then
        echo "invalid dual phone arbiter configuration" >&2
        exit 2
    fi
    if [[ $bridge_filler_port -le 0 || $bridge_filler_port -gt 65535 \
            || $bridge_filler_port -eq $bridge_port \
            || $bridge_idle_lower_us -eq 0 \
            || $bridge_filler_upper_us -eq 0 \
            || $((bridge_filler_upper_us + bridge_guard_us)) \
                    -gt $bridge_idle_lower_us \
            || $phone_max_sessions != 0 ]]; then
        echo "invalid dual phone arbiter configuration" >&2
        exit 2
    fi
fi

mkdir -p "$capture"
adb_cmd=(adb -P "$adb_port" -s "$serial")
"${adb_cmd[@]}" get-state >/dev/null
if pgrep -f '^.*/llama-server .*--port 18' >/dev/null; then
    echo "another llama-server is running" >&2
    exit 1
fi
"${adb_cmd[@]}" shell \
    "su -c 'test ! -e $phone_root && mkdir $phone_root'"
"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-before.json"

nohup "${adb_cmd[@]}" shell \
    "su -c 'sh $phone_logger $phone_samples $phone_active $phone_armed $phone_done 900'" \
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

"${adb_cmd[@]}" shell \
    "su -c 'S42_QWEN_ONLY=$qwen_only_phone S42_PHONE_RESIDENCY_LAYOUT=$phone_residency_layout S42_FFN_VMEM=$phone_ffn_vmem S42_MIN_AVAILABLE_KIB=$phone_minimum_available_kib nohup sh $phone_session $phone_workers $phone_router $phone_gemma $phone_qwen $phone_root $restore_usb 900 $phone_max_sessions > $phone_root/session-launch.log 2>&1 < /dev/null &'"

aoa_ready=0
for _ in $(seq 1 300); do
    if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
        aoa_ready=1
        break
    fi
    sleep 1
done
if [[ $aoa_ready -ne 1 ]]; then
    echo "resident phone session did not bind FunctionFS" >&2
    exit 1
fi

server_pid=
bridge_pid=
cleanup() {
    if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ -n $bridge_pid ]] && kill -0 "$bridge_pid" 2>/dev/null; then
        if [[ $dual_phone_arbiter -eq 1 ]]; then
            kill -TERM "$bridge_pid" 2>/dev/null || true
        else
            python3 "$here/close_resident_bridge.py" \
                --port "$bridge_port" --layer-mask "$qwen_phone_layer_mask" \
                --n-embd 5120 --columns 17408 >/dev/null 2>&1 || true
        fi
        wait "$bridge_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

bridge_prefix=()
if [[ -n $phone_bridge_cpus ]]; then
    bridge_prefix=(taskset --cpu-list "$phone_bridge_cpus")
fi
if [[ $dual_phone_arbiter -eq 1 ]]; then
    "${bridge_prefix[@]}" "$bridge" \
        127.0.0.1 "$bridge_port" "$bridge_filler_port" \
        malloc-split 0 "$qwen_phone_last_layer" "$bridge_idle_lower_us" \
        "$bridge_filler_upper_us" "$bridge_guard_us" \
        >"$capture/bridge.stdout" 2>"$capture/bridge.stderr" &
else
    "${bridge_prefix[@]}" "$bridge" \
        127.0.0.1 "$bridge_port" malloc-split \
        >"$capture/bridge.stdout" 2>"$capture/bridge.stderr" &
fi
bridge_pid=$!
bridge_ready=0
for _ in $(seq 1 300); do
    if grep -Eq 'ready (bind|protected)=' \
            "$capture/bridge.stderr" 2>/dev/null; then
        bridge_ready=1
        break
    fi
    if ! kill -0 "$bridge_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $bridge_ready -ne 1 ]]; then
    echo "DMA-BUF bridge did not become ready" >&2
    exit 1
fi

server_environment=(env "LD_LIBRARY_PATH=$lib_dir:$server_bin")
if [[ $arm == op15 ]]; then
    server_environment+=(
        S41_SERVER_FFN_HOST=127.0.0.1
        "S41_SERVER_FFN_PORT=$bridge_port"
        S41_SERVER_FFN_N_EMBD=5120
        "S41_SERVER_FFN_LAYER_MASK=$qwen_phone_layer_mask"
        S41_SERVER_FFN_COLUMNS=17408
        S41_SERVER_FFN_F16_IO=1
        S41_SERVER_FFN_ACTIVATION=swiglu
        "S41_SERVER_FFN_POLICY=$qwen_policy"
        S41_SERVER_FFN_TIMEOUT_MS=120000
    )
fi
server_prefix=()
if [[ -n $qwen_cpus ]]; then
    server_prefix=(taskset --cpu-list "$qwen_cpus")
fi
"${server_prefix[@]}" "${server_environment[@]}" "$server" \
    --model "$model" --alias qwen-full-phone --fit off \
    --ctx-size 24576 --parallel 4 --batch-size 2048 --ubatch-size 512 \
    --flash-attn on --cont-batching --kv-unified --no-cache-idle-slots \
    --cache-type-k f16 --cache-type-v f16 --split-mode none \
    --n-gpu-layers "$qwen_gpu_layers" --main-gpu 0 --device CUDA0 \
    --host 127.0.0.1 --port "$server_port" --metrics --slots \
    --no-webui --log-colors off --log-timestamps \
    --log-verbosity "$qwen_log_verbosity" \
    >"$capture/server.stdout" 2>"$capture/server.stderr" &
server_pid=$!

server_ready=0
for _ in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:$server_port/health" >/dev/null 2>&1; then
        server_ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        break
    fi
    sleep 0.5
done
if [[ $server_ready -ne 1 ]]; then
    echo "Qwen server did not become ready" >&2
    exit 1
fi
if [[ -n $affinity_receipt ]]; then
    python3 "$here/capture_process_affinity.py" \
        --entry "qwen-server:$server_pid:$qwen_cpus" \
        --entry "phone-bridge:$bridge_pid:$phone_bridge_cpus" \
        --output "$affinity_receipt"
fi

if [[ -n $server_ready_file ]]; then
    touch -- "$server_ready_file"
    released=0
    for _ in $(seq 1 1800); do
        if [[ -e $server_release_file ]]; then
            released=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null \
                || ! kill -0 "$bridge_pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [[ $released -ne 1 ]]; then
        echo "Qwen server barrier was not released" >&2
        exit 1
    fi
fi

runner_args=(
    python3 "$here/run_qwen_full_energy.py"
    --burst-dir "$burst_dir" --requests "$trace" --indices "$indices" \
    --dispatch "$dispatch" \
    --port "$server_port" --server-pid "$server_pid" \
    --arm "$arm" --repeat-index "$repeat_index" --output "$output" \
)
if [[ -n $prefetch_arm_file ]]; then
    runner_args+=(--prefetch-arm-file "$prefetch_arm_file")
fi
if [[ -n $paid_ready_file ]]; then
    runner_args+=(
        --paid-ready-file "$paid_ready_file"
        --paid-release-file "$paid_release_file"
    )
fi
if [[ -n $qwen_complete_file ]]; then
    runner_args+=(
        --qwen-complete-file "$qwen_complete_file"
        --paid-tail-file "$paid_tail_file"
        --paid-tail-timeout-s "$paid_tail_timeout_s"
    )
fi
"${runner_args[@]}" >"$capture/runner.log" 2>&1

kill -INT "$server_pid"
wait "$server_pid"
server_pid=
if [[ $arm == control ]]; then
    python3 "$here/close_resident_bridge.py" \
        --port "$bridge_port" --layer-mask "$qwen_phone_layer_mask" \
        --n-embd 5120 --columns 17408
fi
wait "$bridge_pid"
bridge_pid=

device_ready=0
for _ in $(seq 1 240); do
    if "${adb_cmd[@]}" get-state >/dev/null 2>&1; then
        device_ready=1
        break
    fi
    sleep 0.5
done
if [[ $device_ready -ne 1 ]]; then
    echo "phone did not restore ADB" >&2
    exit 1
fi

done_ready=0
for _ in $(seq 1 120); do
    if "${adb_cmd[@]}" shell "su -c 'test -f $phone_done'" \
            >/dev/null 2>&1; then
        done_ready=1
        break
    fi
    sleep 0.25
done
if [[ $done_ready -ne 1 ]]; then
    echo "phone power logger did not finish" >&2
    exit 1
fi
wait "$power_adb_pid" || true

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-after.json"
"${adb_cmd[@]}" pull "$phone_samples" "$capture/phone-samples.tsv" \
    >"$capture/pull.log"
for name in session.log resident-workers.log router.log session-launch.log; do
    "${adb_cmd[@]}" pull "$phone_root/$name" "$capture/$name" \
        >>"$capture/pull.log" 2>&1 || true
done

python3 "$burst_dir/analyze_phone_energy.py" \
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
trap - EXIT INT TERM
cat "$capture/SHA256SUMS.txt"
