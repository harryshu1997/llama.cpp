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
if [[ $output != /* || -e $output || -e ${output}.runtime-gates ]]; then
    echo "invalid output root: $output" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(cd -- "$here/../../../.." && pwd)
s42_root=$(cd -- "$here/.." && pwd)
legacy_runner=${S42_LEGACY_RUNNER:-/home/zhihao/s41-dynamic-ffn-v1/campaign/server_trace_v2/run_i2_energy_arm.sh}
serial=3C15AU002CL00000
gate_root=${output}.runtime-gates
stop_file=$gate_root/host-monitor.stop
host_samples=$gate_root/host-gates.jsonl
phone_samples=$gate_root/phone-gates.tsv
phone_root=/data/local/tmp/s41-opoffload-dmabuf-v1/i2-energy-${mode}-r${repeat_index}
phone_active=$phone_root/active
phone_logger=/data/local/tmp/s41-opoffload-dmabuf-v1/s42_runtime_gate_logger_v1.sh
phone_raw=/data/local/tmp/s41-opoffload-dmabuf-v1/s42-gate-${mode}-r${repeat_index}.tsv
phone_armed=/data/local/tmp/s41-opoffload-dmabuf-v1/s42-gate-${mode}-r${repeat_index}.armed
phone_done=/data/local/tmp/s41-opoffload-dmabuf-v1/s42-gate-${mode}-r${repeat_index}.done
plan=$gate_root/EXECUTION_PLAN.json

mkdir -p "$gate_root"
adb_cmd=(adb -P "$adb_port" -s "$serial")
"${adb_cmd[@]}" get-state >/dev/null
scheduler_mode=control
if [[ $mode == op15 ]]; then
    scheduler_mode=enforce
fi
python3 "$here/plan_unified_burstgpt.py" \
    --trace /home/zhihao/s41-dynamic-ffn-v1/campaign/input/REQUESTS_SEMANTIC_SOURCE.jsonl \
    --compiled "$s42_root/runtime_routes_v1/COMPILED_4060TI_OP15_I3_ROUTES_V1.json" \
    --epoch "$s42_root/runtime_routes_v1/I3_CERTIFIED_EPOCH_4060TI_OP15_V1.json" \
    --contracts "$s42_root/runtime_routes_v1/I3_RUNTIME_GATE_CONTRACTS_4060TI_OP15_V1.json" \
    --mode "$scheduler_mode" \
    --adb-port "$adb_port" \
    --phone-serial "$serial" \
    --hot-server /home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server \
    --cold-server /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin/llama-server \
    --hot-model /home/zhihao/models/Qwen3-14B-Q4_K_M.gguf \
    --cold-model /home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf \
    --cuda-lib-dir /mnt/storage/s21_deps/cuda-13.2.1/lib \
    --cold-lib-dir /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin \
    --bridge /home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-reset-qualified \
    --phone-session /data/local/tmp/s41-opoffload-dmabuf-v1/phone_ffn_session-flex.sh \
    --phone-worker /data/local/tmp/s41-opoffload-dmabuf-v1/llama-ffn-split-worker-flex-v2 \
    --phone-model /data/local/tmp/ls-s32/gemma-4-12B-it-Q4_0.gguf \
    --restore-usb /data/local/tmp/s41-opoffload-dmabuf-v1/restore_phone_usb.sh \
    --output "$plan" \
    >"$gate_root/scheduler-plan.log"
"${adb_cmd[@]}" push "$here/phone_runtime_gate_logger.sh" "$phone_logger" \
    >"$gate_root/phone-logger-push.log"
"${adb_cmd[@]}" shell "su -c 'chmod 0755 $phone_logger; rm -f $phone_raw $phone_armed $phone_done'"
"${adb_cmd[@]}" shell "su -c 'for z in /sys/class/thermal/thermal_zone*; do printf \"%s\\t\" \"\$z\"; cat \"\$z/type\"; done'" \
    >"$gate_root/phone-zone-types.tsv"
"${adb_cmd[@]}" shell sha256sum \
    /data/local/tmp/s41-opoffload-dmabuf-v1/llama-ffn-split-worker-flex-v2 \
    >"$gate_root/worker-sha256.txt"
sha256sum \
    /home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-reset-qualified \
    "$legacy_runner" \
    >"$gate_root/host-artifact-sha256.txt"

nohup "${adb_cmd[@]}" shell \
    "su -c 'sh $phone_logger $phone_active $phone_raw $phone_armed $phone_done 300'" \
    </dev/null >"$gate_root/phone-gate-adb.log" 2>&1 &
phone_gate_pid=$!

armed=0
for _ in $(seq 1 300); do
    if "${adb_cmd[@]}" shell "su -c 'test -f $phone_armed'" >/dev/null 2>&1; then
        armed=1
        break
    fi
    sleep 0.1
done
if [[ $armed -ne 1 ]]; then
    echo "phone runtime gate logger did not arm" >&2
    exit 1
fi

python3 "$here/host_runtime_gate_monitor.py" \
    --mode "$mode" \
    --result-root "$output" \
    --stop-file "$stop_file" \
    --output "$host_samples" \
    >"$gate_root/host-monitor.log" 2>&1 &
host_monitor_pid=$!

cleanup() {
    touch "$stop_file"
    wait "$host_monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set +e
S42_EXECUTION_PLAN="$plan" S42_UNIFIED_REPO_ROOT="$repo_root" \
"$legacy_runner" "$mode" "$repeat_index" "$output" "$adb_port" \
    >"$gate_root/legacy-runner.log" 2>&1
run_rc=$?
set -e

touch "$stop_file"
wait "$host_monitor_pid"
trap - EXIT INT TERM

device_ready=0
for _ in $(seq 1 180); do
    if "${adb_cmd[@]}" get-state >/dev/null 2>&1; then
        device_ready=1
        break
    fi
    sleep 1
done
if [[ $device_ready -ne 1 ]]; then
    echo "phone did not return to ADB for runtime-gate collection" >&2
    exit 1
fi

wait "$phone_gate_pid" || true
"${adb_cmd[@]}" shell "su -c 'test -f $phone_done'"
"${adb_cmd[@]}" pull "$phone_raw" "$phone_samples" \
    >"$gate_root/phone-gate-pull.log"

if [[ $run_rc -ne 0 || ! -f $output/RESULT.json ]]; then
    echo "legacy physical arm failed: rc=$run_rc" >&2
    exit "$run_rc"
fi

python3 "$here/analyze_stage6_arm.py" \
    --mode "$mode" \
    --repeat-index "$repeat_index" \
    --result "$output/RESULT.json" \
    --host-telemetry "$host_samples" \
    --phone-telemetry "$phone_samples" \
    --scheduler-plan "$plan" \
    --output "$gate_root/RUNTIME_GATE_RECEIPT.json"

sha256sum \
    "$plan" \
    "$output/RESULT.json" \
    "${output}.phone-capture/PHONE_ENERGY_V3.json" \
    "$gate_root/RUNTIME_GATE_RECEIPT.json" \
    "$host_samples" \
    "$phone_samples" \
    >"$gate_root/SHA256SUMS.txt"
cat "$gate_root/SHA256SUMS.txt"
