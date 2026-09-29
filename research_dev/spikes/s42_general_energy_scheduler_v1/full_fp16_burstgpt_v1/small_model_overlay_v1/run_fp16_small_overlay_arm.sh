#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 <output-root> <cpu-overflow|op15-assistance|runtime-auto> <static-cpu|runtime-scheduler> <adb-port> [qualification|phase-calibration|natural-validation|idle-split-calibration|direct-energy-calibration]" >&2
    exit 2
fi

output=$1
large_model_policy=$2
small_model_policy=$3
adb_port=$4
overlay_variant=${5:-qualification}
min_phone_battery_level=${S42_MIN_PHONE_BATTERY_LEVEL:-20}
preflight_only=${S42_PREFLIGHT_ONLY:-0}
unified_request_indices=${S42_UNIFIED_REQUEST_INDICES:-}
if [[ $output != /* || -e $output ]]; then
    echo "output must be a new absolute path: $output" >&2
    exit 2
fi
if [[ $preflight_only != 0 && $preflight_only != 1 ]]; then
    echo "invalid S42_PREFLIGHT_ONLY: $preflight_only" >&2
    exit 2
fi
if [[ -n $unified_request_indices \
        && ! $unified_request_indices =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid S42_UNIFIED_REQUEST_INDICES" >&2
    exit 2
fi
if [[ ! $min_phone_battery_level =~ ^[0-9]+$ ]] \
        || (( min_phone_battery_level < 1 \
            || min_phone_battery_level > 100 )); then
    echo "invalid S42_MIN_PHONE_BATTERY_LEVEL" >&2
    exit 2
fi
if [[ $large_model_policy != cpu-overflow \
        && $large_model_policy != op15-assistance \
        && $large_model_policy != runtime-auto ]]; then
    echo "invalid large-model policy: $large_model_policy" >&2
    exit 2
fi
if [[ $small_model_policy != static-cpu \
        && $small_model_policy != runtime-scheduler ]]; then
    echo "invalid small-model policy: $small_model_policy" >&2
    exit 2
fi
if [[ $overlay_variant != qualification \
        && $overlay_variant != phase-calibration \
        && $overlay_variant != natural-validation \
        && $overlay_variant != idle-split-calibration \
        && $overlay_variant != direct-energy-calibration ]]; then
    echo "invalid overlay variant: $overlay_variant" >&2
    exit 2
fi
large_arm=control
[[ $large_model_policy == op15-assistance ]] && large_arm=op15
[[ $large_model_policy == runtime-auto ]] && large_arm=auto

here=$(cd -- "$(dirname -- "$0")" && pwd)
fp16_root=$(cd -- "$here/.." && pwd)
repo_root=${S42_UNIFIED_REPO_ROOT:-$(cd -- "$here/../../../../.." && pwd)}
legacy_root=${S42_LEGACY_ROOT:-/home/zhihao/s41-dynamic-ffn-v1}
burst_dir=$repo_root/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1
trace=$burst_dir/REQUESTS_SEMANTIC_SOURCE.jsonl
runner=$repo_root/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/mixed_scheduler_v1/run_hierarchical_trace.py
campaign_root=$repo_root/research_dev/scheduler/campaigns/burstgpt
unified_runner=$campaign_root/runner.py
unified_preflight=$campaign_root/preflight.py
large_catalog_builder=$campaign_root/catalog.py
qwen_manifest=$campaign_root/data/QWEN_MANIFEST.json
gemma_manifest=$campaign_root/data/GEMMA_MANIFEST.json
planner=$fp16_root/plan_full_fp16_burstgpt.py
bootstrap_builder=$fp16_root/build_fp16_model_device_bootstrap.py
request_profile_fitter=$campaign_root/admission_profile.py
snapshot_capture=$fp16_root/capture_runtime_placement_snapshot.py
controller=$here/run_fp16_small_overlay.py
qualifier=$here/qualify_fp16_small_overlay.py
automated_catalog_builder=$campaign_root/overlay_catalog.py
six_model_runner=$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/mixed_model_trace_v1/run_six_model_trace.py
direct_energy_runner=$here/measure_ffn_direct_energy.py
direct_energy_analyzer=$here/analyze_ffn_direct_energy.py
close_helper=$repo_root/research_dev/scheduler/adapters/close_resident_bridge.py
phone_session_source=$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/resident_ffn_session.sh
qwen_result=$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/results/QWEN_FULL_FFN_M1_M4_ENERGY_SCREEN_ABBA_V2.json
gemma_result=$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/hybrid_overflow_v1/results/physical_pair_r1/PAIR.json
placement_result=$fp16_root/results/FULL_FP16_BURSTGPT_ABBA_V2.json
residency_plan=$here/OP15_COMBINED_RESIDENCY_PLAN_V1.json
overlay_manifest=$here/TRACE_MANIFEST.json
overlay_trace=$here/REQUESTS_LLAMA1B_10.jsonl
if [[ $overlay_variant == phase-calibration ]]; then
    overlay_manifest=$here/TRACE_PHASE_CALIBRATION_MANIFEST.json
    overlay_trace=$here/REQUESTS_LLAMA1B_PHASE_CALIBRATION_40.jsonl
elif [[ $overlay_variant == natural-validation ]]; then
    overlay_manifest=$here/TRACE_NATURAL_VALIDATION_MANIFEST.json
    overlay_trace=$here/REQUESTS_LLAMA1B_NATURAL_VALIDATION_40.jsonl
elif [[ $overlay_variant == idle-split-calibration ]]; then
    overlay_manifest=$here/TRACE_IDLE_SPLIT_CALIBRATION_MANIFEST.json
    overlay_trace=$here/REQUESTS_LLAMA1B_IDLE_SPLIT_CALIBRATION_40.jsonl
fi
default_runtime_profile=$here/SCHEDULER_PROFILE_COMPOSITE_USB_V2.json
if [[ $large_model_policy == runtime-auto \
        || $overlay_variant == qualification ]]; then
    default_runtime_profile=$here/SCHEDULER_PROFILE_RUNTIME_AUTO_V1.json
fi
runtime_profile=${S42_RUNTIME_PROFILE:-$default_runtime_profile}
marginal_system_profile=${S42_MARGINAL_SYSTEM_PROFILE:-$here/MARGINAL_SYSTEM_PROFILE_V1.json}
automated_control_overhead_profile=${S42_AUTOMATED_CONTROL_OVERHEAD_PROFILE:-$here/AUTOMATED_CONTROL_OVERHEAD_PROFILE_V1.json}
ffn_manifest_builder=$here/derive_llama_ffn_manifest.py
ffn_policy_compiler=$here/compile_llama_ffn_split.py
ffn_python=${S42_FFN_PYTHON:-python3}
ffn_physical_calibration=${S42_FFN_PHYSICAL_CALIBRATION:-}
ffn_compiled_policy_input=${S42_FFN_COMPILED_POLICY:-}
kernel_campaign=$repo_root/research_dev/scheduler/profiles/MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json
qualified_runtime_dir=$here/results/4060ti_op15_20260813/natural_runtime_v3_matched_v1
qualified_route_profile=$qualified_runtime_dir/RUNTIME_PROFILE.json
qualified_route_audit=$qualified_runtime_dir/RUNTIME_PROFILE_AUDIT.json
forced_static_route=${S42_FORCED_STATIC_ROUTE:-desktop-cpu}
required_runtime_route=${S42_REQUIRED_RUNTIME_ROUTE:-}
llama_ffn_resident_layers=${S42_LLAMA_FFN_RESIDENT_LAYERS:-1}
direct_energy_abba_cycles=${S42_DIRECT_ENERGY_ABBA_CYCLES:-2}
direct_energy_request_indices=${S42_DIRECT_ENERGY_REQUEST_INDICES:-1,4,5,8,9}
direct_energy_idle_duration_s=${S42_DIRECT_ENERGY_IDLE_DURATION_S:-5}
ffn_prewarm_timeout_s=${S42_FFN_PREWARM_TIMEOUT_S:-120}
if [[ -n $ffn_physical_calibration \
        && ( $ffn_physical_calibration != /* \
            || ! -f $ffn_physical_calibration ) ]]; then
    echo "invalid S42_FFN_PHYSICAL_CALIBRATION" >&2
    exit 2
fi
if [[ -n $ffn_compiled_policy_input \
        && ( $ffn_compiled_policy_input != /* \
            || ! -f $ffn_compiled_policy_input ) ]]; then
    echo "invalid S42_FFN_COMPILED_POLICY" >&2
    exit 2
fi
if [[ -n $automated_control_overhead_profile \
        && ( $automated_control_overhead_profile != /* \
            || ! -f $automated_control_overhead_profile ) ]]; then
    echo "invalid S42_AUTOMATED_CONTROL_OVERHEAD_PROFILE" >&2
    exit 2
fi
if [[ -n $required_runtime_route \
        && $required_runtime_route != desktop-cpu \
        && $required_runtime_route != desktop-cuda \
        && $required_runtime_route != phone-adreno \
        && $required_runtime_route != cpu-phone-ffn-split ]]; then
    echo "invalid S42_REQUIRED_RUNTIME_ROUTE: $required_runtime_route" >&2
    exit 2
fi
if [[ -n $required_runtime_route \
        && $small_model_policy != runtime-scheduler ]]; then
    echo "S42_REQUIRED_RUNTIME_ROUTE requires runtime-scheduler" >&2
    exit 2
fi
if [[ $forced_static_route != desktop-cpu \
        && $forced_static_route != cpu-phone-ffn-split ]]; then
    echo "invalid S42_FORCED_STATIC_ROUTE: $forced_static_route" >&2
    exit 2
fi
if [[ ! $llama_ffn_resident_layers =~ ^[0-9]+$ ]] \
        || (( llama_ffn_resident_layers < 1 \
            || llama_ffn_resident_layers > 16 )); then
    echo "invalid S42_LLAMA_FFN_RESIDENT_LAYERS: $llama_ffn_resident_layers" >&2
    exit 2
fi
if [[ ! $direct_energy_abba_cycles =~ ^[0-9]+$ ]] \
        || (( direct_energy_abba_cycles < 2 )); then
    echo "invalid S42_DIRECT_ENERGY_ABBA_CYCLES" >&2
    exit 2
fi
if [[ ! $direct_energy_request_indices =~ ^[0-9]+(,[0-9]+)+$ ]]; then
    echo "invalid S42_DIRECT_ENERGY_REQUEST_INDICES" >&2
    exit 2
fi
if [[ ! $direct_energy_idle_duration_s =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "invalid S42_DIRECT_ENERGY_IDLE_DURATION_S" >&2
    exit 2
fi
if [[ ! $ffn_prewarm_timeout_s =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "invalid S42_FFN_PREWARM_TIMEOUT_S" >&2
    exit 2
fi
llama_ffn_layer_mask=0-$((llama_ffn_resident_layers - 1))

server=$legacy_root/build-server-ffn-cuda/bin/llama-server
server_bin=$(dirname -- "$server")
cpu_server=${S42_CPU_SERVER:-$repo_root/build-s41-server-ffn/bin/llama-server}
qwen_model=/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf
gemma_model=/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf
llama_model=/home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf
cuda_lib_dir=/mnt/storage/s21_deps/cuda-13.2.1/lib
bridge=${S42_FFN_BRIDGE:-$legacy_root/ffn_dmabuf_bridge-swiglu-v1}
serial=3C15AU002CL00000

capture=$output/capture
base_output=$output/base
combined_output=$output/combined
plan=$capture/EXECUTION_PLAN.json
runtime_snapshot=$capture/RUNTIME_PLACEMENT_SNAPSHOT.json
fp16_request_profile=$capture/FP16_REQUEST_ADMISSION_PROFILE.json
controller_ready=$capture/controller.ready.json
resident_release=$capture/resident-release.json

phone_base=/data/local/tmp/s41-opoffload-dmabuf-v1
phone_run_parent=${output%/*}
phone_run_tag=${phone_run_parent##*/}-${output##*/}
if [[ ! $phone_run_tag =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid phone run tag: $phone_run_tag" >&2
    exit 2
fi
phone_root=$phone_base/$phone_run_tag
phone_logger=$phone_base/phone_power_logger.sh
phone_policy=$phone_base/phone_power_allow.rules
phone_samples=$phone_root/samples.tsv
phone_active=$phone_root/active
phone_power_active=$phone_root/power.active
phone_armed=$phone_root/power.armed
phone_done=$phone_root/power.done
phone_workers=$phone_base/resident_ffn_workers-quad-v1
phone_workers_host=$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/bin/resident_ffn_workers.android
phone_router_host=${S42_PHONE_ROUTER_HOST:-$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/bin/resident_ffn_router-terminal-v2.android}
phone_router=${S42_PHONE_ROUTER:-$phone_base/resident_ffn_router-terminal-v2}
phone_session=$phone_root/resident_ffn_session.sh
phone_session_pid_file=$phone_root/resident-session.pid
phone_session_stage=/data/local/tmp/s42-resident-session-$phone_run_tag.sh
phone_workers_stage=/data/local/tmp/s42-resident-workers-$phone_run_tag
phone_router_stage=/data/local/tmp/s42-resident-router-$phone_run_tag
phone_gemma=$phone_base/gemma-4-12B-Q40-dequant-f16.gguf
phone_qwen=$phone_base/Qwen3-14B-Q4KM-dequant-f16.gguf
phone_llama=/data/local/tmp/unifer/llamacpp/Llama-3.2-1B-Instruct-Q4_0.gguf
phone_task_server=/data/local/tmp/llama-ubatch-op15/bin/llama-server
phone_busybox=/data/adb/magisk/busybox
restore_usb=$phone_base/restore_phone_usb-ncm-v3.sh
combined_ready=$phone_root/combined.ready
bind_arm=$phone_root/bind.arm

hot_port=18571
cold_port=18572
gpu_cold_port=18573
qwen_control_port=18571
qwen_phone_port=18572
gemma_control_port=18573
gemma_phone_port=18574
bridge_port=18671
phone_task_port=18382
phone_diagnostic_port=18383
phone_ffn_port=18384
phone_reserve_bytes=805306368
phone_reserve_kib=$((phone_reserve_bytes / 1024))
power_adb_pid=
controller_pid=
base_scope_pid=
controller_unit=$(systemd-escape "s42-fp16-controller-$phone_run_tag")
base_unit=$(systemd-escape "s42-fp16-base-$phone_run_tag")

adb_cmd=(adb -P "$adb_port" -s "$serial")

capture_normal_usb_receipt() {
    local receipt=$1
    PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
        python3 - "$serial" "$adb_port" "$receipt" <<'PY'
import json
from pathlib import Path
import sys

from research_dev.scheduler.adapters import verify_android_usb_restored

serial, adb_port, output = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
receipt = verify_android_usb_restored(
    serial=serial,
    adb_port=adb_port,
    minimum_speed_mbps=5000,
    timeout_s=60,
)
output.write_text(
    json.dumps(
        receipt.to_json(),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="ascii",
)
PY
}

close_resident_bridge_session() {
    local label=$1
    local layer_mask=$2
    local n_embd=$3
    local columns=$4
    local activation=$5
    local bridge_pid=
    local bridge_log=$capture/cleanup-$label-bridge

    timeout 60 "$bridge" 127.0.0.1 "$bridge_port" malloc-split \
        > "$bridge_log.stdout" 2> "$bridge_log.stderr" &
    bridge_pid=$!
    for _ in $(seq 1 120); do
        grep -q '\[ffn-dmabuf-bridge\] ready' "$bridge_log.stderr" \
            2>/dev/null && break
        kill -0 "$bridge_pid" 2>/dev/null || break
        sleep 0.25
    done
    python3 "$close_helper" \
        --port "$bridge_port" \
        --layer-mask "$layer_mask" \
        --n-embd "$n_embd" \
        --columns "$columns" \
        --activation "$activation" \
        --terminate-session \
        > "$capture/cleanup-$label-helper.log" 2>&1 || true
    wait "$bridge_pid" >/dev/null 2>&1 || true
}

recover_resident_usb() {
    lsusb -d 18d1:2d00 >/dev/null 2>&1 || return 0
    if pgrep -f '^.*/ffn_dmabuf_bridge' >/dev/null; then
        echo "resident USB recovery skipped: bridge is still active" \
            > "$capture/cleanup-resident-usb.log"
        return 0
    fi
    close_resident_bridge_session qwen 4095 5120 17408 swiglu
    if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
        close_resident_bridge_session gemma 8388607 3840 6144 geglu
    fi
    if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
        close_resident_bridge_session qwen-final 4095 5120 17408 swiglu
    fi
    for _ in $(seq 1 120); do
        "${adb_cmd[@]}" get-state >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "resident USB recovery did not restore ADB" \
        > "$capture/cleanup-resident-usb.log"
}

interrupt_controller() {
    [[ -n $controller_pid ]] || return 0
    systemctl --user kill --kill-whom=all --signal=SIGINT \
        "$controller_unit.scope" >/dev/null 2>&1 \
        || kill -INT "$controller_pid" >/dev/null 2>&1 \
        || return 0
    for _ in $(seq 1 80); do
        [[ -e /proc/$controller_pid/stat ]] || return 0
        [[ $(awk '{print $3}' "/proc/$controller_pid/stat") == Z ]] \
            && return 0
        sleep 0.25
    done
    systemctl --user stop "$controller_unit.scope" \
        >/dev/null 2>&1 || true
    kill -TERM "$controller_pid" >/dev/null 2>&1 || true
}

interrupt_base_scope() {
    [[ -n $base_scope_pid ]] || return 0
    systemctl --user kill --kill-whom=all --signal=SIGINT \
        "$base_unit.scope" >/dev/null 2>&1 \
        || kill -INT "$base_scope_pid" >/dev/null 2>&1 \
        || true
    for _ in $(seq 1 80); do
        kill -0 "$base_scope_pid" 2>/dev/null || return 0
        sleep 0.25
    done
    systemctl --user stop "$base_unit.scope" >/dev/null 2>&1 || true
    kill -TERM "$base_scope_pid" >/dev/null 2>&1 || true
}

wait_for_phone_power_done() {
    for _ in $(seq 1 300); do
        if "${adb_cmd[@]}" shell "su -c 'test -f $phone_done'" \
                >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    echo "phone power logger did not finish" >&2
    return 1
}

cleanup() {
    if [[ -n $base_scope_pid ]]; then
        interrupt_base_scope
        wait "$base_scope_pid" >/dev/null 2>&1 || true
        base_scope_pid=
    fi
    if [[ -n $controller_pid ]]; then
        interrupt_controller
        wait "$controller_pid" >/dev/null 2>&1 || true
        controller_pid=
    fi
    if [[ -d $capture ]]; then
        recover_resident_usb || true
    fi
    if "${adb_cmd[@]}" get-state >/dev/null 2>&1; then
        "${adb_cmd[@]}" shell \
            "su -c 'if [ -f $phone_session_pid_file ]; then phone_session_pid=\$(cat $phone_session_pid_file); kill \$phone_session_pid 2>/dev/null || true; fi; rm -f $phone_active $phone_power_active'" \
            >/dev/null 2>&1 || true
    fi
    if [[ -n $power_adb_pid ]]; then
        wait "$power_adb_pid" >/dev/null 2>&1 || true
    fi
}

collect_phone_capture() {
    local device_ready=0
    for _ in $(seq 1 360); do
        if "${adb_cmd[@]}" get-state >/dev/null 2>&1; then
            device_ready=1
            break
        fi
        sleep 1
    done
    if [[ $device_ready -ne 1 ]]; then
        echo "phone did not restore ADB" >&2
        return 1
    fi
    "${adb_cmd[@]}" shell \
        "su -c 'rm -f $phone_active $phone_power_active'" \
        >/dev/null 2>&1 || true
    if [[ -n $power_adb_pid ]]; then
        wait "$power_adb_pid" || true
        power_adb_pid=
    fi
    wait_for_phone_power_done
    python3 "$burst_dir/capture_phone_clock.py" \
        --adb-port "$adb_port" --serial "$serial" \
        --output "$capture/clock-after.json"
    "${adb_cmd[@]}" pull "$phone_samples" "$capture/phone-samples.tsv" \
        > "$capture/pull.log"
    for name in session.log resident-workers.log router.log \
            task-server.log session-launch.log diagnostic-http.log; do
        "${adb_cmd[@]}" pull "$phone_root/$name" "$capture/$name" \
            >> "$capture/pull.log" 2>&1 || true
    done
}
trap cleanup EXIT

for path in \
        "$trace" "$runner" "$planner" "$bootstrap_builder" \
        "$unified_runner" "$unified_preflight" "$large_catalog_builder" \
        "$qwen_manifest" "$gemma_manifest" \
        "$request_profile_fitter" \
        "$snapshot_capture" \
        "$controller" "$qualifier" "$automated_catalog_builder" \
        "$six_model_runner" \
        "$direct_energy_runner" \
        "$direct_energy_analyzer" "$close_helper" "$phone_session_source" \
        "$qwen_result" "$gemma_result" \
        "$placement_result" "$residency_plan" "$overlay_manifest" \
        "$overlay_trace" "$runtime_profile" "$server" "$cpu_server" \
        "$marginal_system_profile" \
        "$ffn_manifest_builder" "$ffn_policy_compiler" \
        "$kernel_campaign" "$qualified_route_profile" \
        "$qualified_route_audit" "$phone_workers_host" \
        "$phone_router_host" \
        "$qwen_model" "$gemma_model" "$llama_model" \
        "$cuda_lib_dir" "$bridge"; do
    if [[ ! -e $path ]]; then
        echo "missing physical dependency: $path" >&2
        exit 1
    fi
done
profile_has_split=$(python3 - "$runtime_profile" <<'PY'
import json
import sys

profile = json.load(open(sys.argv[1], encoding="ascii"))
routes = profile.get("routes")
if type(routes) is not list or any(type(route) is not dict for route in routes):
    raise SystemExit("runtime profile routes are invalid")
print(int(any(
    route.get("route_id") == "cpu-phone-ffn-split" for route in routes
)))
PY
)
ffn_route_required=0
if [[ $overlay_variant == direct-energy-calibration \
        || $forced_static_route == cpu-phone-ffn-split \
        || $required_runtime_route == cpu-phone-ffn-split \
        || $profile_has_split -eq 1 ]]; then
    ffn_route_required=1
fi
if (( ffn_route_required )); then
    if [[ $ffn_python == */* && ! -x $ffn_python ]]; then
        echo "invalid S42_FFN_PYTHON: $ffn_python" >&2
        exit 2
    fi
    if ! "$ffn_python" -c 'import numpy' >/dev/null 2>&1; then
        echo "S42_FFN_PYTHON cannot import NumPy: $ffn_python" >&2
        exit 2
    fi
    phone_llama_ffn_model=$phone_llama
else
    phone_llama_ffn_model=
fi
if pgrep -f '^.*/llama-server .*--port 18' >/dev/null \
        || pgrep -f '^.*/ffn_dmabuf_bridge' >/dev/null; then
    echo "another inference or bridge process is active" >&2
    pgrep -af 'llama-server|ffn_dmabuf_bridge' >&2 || true
    exit 1
fi

mkdir -p "$capture"
"${adb_cmd[@]}" get-state >/dev/null
capture_normal_usb_receipt "$capture/PHONE_USB_BEFORE.json"
phone_battery_level=$("${adb_cmd[@]}" shell dumpsys battery 2>/dev/null \
    | awk '/^[[:space:]]*level:/ {gsub("\r", "", $2); print $2; exit}')
if [[ ! $phone_battery_level =~ ^[0-9]+$ ]] \
        || (( phone_battery_level < min_phone_battery_level )); then
    echo "phone battery below safe run floor: ${phone_battery_level:-unknown}% < ${min_phone_battery_level}%" >&2
    exit 1
fi
"${adb_cmd[@]}" shell \
    "su -c 'test ! -e $phone_root && mkdir $phone_root'"
"${adb_cmd[@]}" push "$phone_session_source" "$phone_session_stage" \
    > "$capture/push-phone-session.log"
"${adb_cmd[@]}" push "$phone_workers_host" "$phone_workers_stage" \
    > "$capture/push-phone-workers.log"
"${adb_cmd[@]}" push "$phone_router_host" "$phone_router_stage" \
    > "$capture/push-phone-router.log"
"${adb_cmd[@]}" shell \
    "su -c 'mv $phone_session_stage $phone_session && chmod 0755 $phone_session && mv $phone_workers_stage $phone_workers && chmod 0755 $phone_workers && mv $phone_router_stage $phone_router && chmod 0755 $phone_router'"
sha256sum "$phone_session_source" > "$capture/phone-session.sha256"
sha256sum "$phone_workers_host" > "$capture/phone-workers.sha256"
sha256sum "$phone_router_host" > "$capture/phone-router.sha256"
"${adb_cmd[@]}" shell "su -c 'sha256sum $phone_router'" \
    >> "$capture/phone-router.sha256"
if [[ $(awk 'NR == 1 {print $1}' "$capture/phone-router.sha256") \
        != $(awk 'NR == 2 {print $1}' "$capture/phone-router.sha256") ]]; then
    echo "phone router deployment identity mismatch" >&2
    exit 1
fi
"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
"${adb_cmd[@]}" shell "su -c 'test -x $phone_busybox'"
"${adb_cmd[@]}" shell "su -c 'am kill-all; sleep 3'"

effective_large_model_policy=$large_model_policy
planned_arm=$large_arm
if [[ $overlay_variant != qualification ]]; then
placement_ready=0
for attempt in $(seq 1 12); do
    attempt_snapshot=$capture/RUNTIME_PLACEMENT_SNAPSHOT.$attempt.json
    attempt_plan=$capture/EXECUTION_PLAN.$attempt.json
    planner_command=(
        python3 "$planner"
        --qwen-result "$qwen_result"
        --gemma-result "$gemma_result"
        --placement-result "$placement_result"
        --residency-plan "$residency_plan"
        --runtime-snapshot "$attempt_snapshot"
        --overlay-manifest "$overlay_manifest"
        --output "$attempt_plan"
    )
    if [[ $large_model_policy == runtime-auto ]]; then
        planner_command=(
            python3 "$bootstrap_builder"
            --qwen-result "$qwen_result"
            --gemma-result "$gemma_result"
            --residency-plan "$residency_plan"
            --runtime-snapshot "$attempt_snapshot"
            --additional-host-bytes 770928288
            --additional-phone-bytes 770928288
            --output "$attempt_plan"
        )
    fi
    if python3 "$snapshot_capture" \
            --adb-port "$adb_port" \
            --serial "$serial" \
            --residency-plan "$residency_plan" \
            --output "$attempt_snapshot" \
            >"$capture/runtime-snapshot.$attempt.log" 2>&1 \
            && "${planner_command[@]}" \
            >"$capture/scheduler-plan.$attempt.log" 2>&1; then
        mv "$attempt_snapshot" "$runtime_snapshot"
        mv "$attempt_plan" "$plan"
        placement_ready=1
        break
    fi
    sleep 5
done
if [[ $placement_ready -ne 1 ]]; then
    echo "scheduler found no feasible combined placement" >&2
    exit 1
fi

effective_large_model_policy=$large_model_policy
planned_arm=$large_arm
if [[ $large_model_policy == runtime-auto ]]; then
    planned_arm=$(python3 - "$plan" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="ascii"))
arm = plan.get("execution_arm")
if arm not in {"control", "op15"}:
    raise SystemExit("invalid runtime placement arm")
print(arm)
PY
)
    if [[ $planned_arm == control ]]; then
        effective_large_model_policy=cpu-overflow
    else
        effective_large_model_policy=op15-assistance
    fi
fi

if [[ $large_model_policy == runtime-auto ]]; then
    if [[ $planned_arm == op15 ]]; then
        request_profile_train=${S42_FP16_REQUEST_PROFILE_TRAIN:-/home/zhihao/s42-full-fp16-burstgpt-v1/op15-r1/RESULT.json}
        request_profile_holdout=${S42_FP16_REQUEST_PROFILE_HOLDOUT:-/home/zhihao/s42-full-fp16-burstgpt-v1/op15-r2/RESULT.json}
    else
        request_profile_train=${S42_FP16_REQUEST_PROFILE_TRAIN:-/home/zhihao/s42-full-fp16-burstgpt-v1/control-r1b/RESULT.json}
        request_profile_holdout=${S42_FP16_REQUEST_PROFILE_HOLDOUT:-/home/zhihao/s42-full-fp16-burstgpt-v1/control-r2/RESULT.json}
    fi
    for path in "$request_profile_train" "$request_profile_holdout"; do
        if [[ $path != /* || ! -f $path ]]; then
            echo "invalid F16 request-profile result: $path" >&2
            exit 2
        fi
    done
    python3 "$request_profile_fitter" \
        --train-result "$request_profile_train" \
        --holdout-result "$request_profile_holdout" \
        --arm "$planned_arm" \
        --output "$fp16_request_profile" \
        > "$capture/request-profile-fit.log"
fi
fi

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-before.json"
ip -o link show | awk -F': ' '{print $2}' | sort > "$capture/net-before.txt"

"${adb_cmd[@]}" shell "su -c 'rm -f $phone_power_active'"
nohup "${adb_cmd[@]}" shell \
    "su -c 'sh $phone_logger $phone_samples $phone_power_active $phone_armed $phone_done 7200'" \
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
"${adb_cmd[@]}" shell "su -c 'touch $phone_power_active'"

"${adb_cmd[@]}" shell \
    "su -c 'nohup env S42_USB_NCM=1 S42_USB_NCM_IPV4=1 S42_MIN_AVAILABLE_KIB=$phone_reserve_kib S42_LLAMA_FFN_MODEL=$phone_llama_ffn_model S42_LLAMA_FFN_PORT=$phone_ffn_port S42_LLAMA_FFN_LAYERS=$llama_ffn_layer_mask S42_TASK_SERVER_BINARY=$phone_task_server S42_TASK_SERVER_MODEL=$phone_llama S42_TASK_SERVER_ALIAS=llama-3.2-1b-instruct-q4_0 S42_TASK_SERVER_PORT=$phone_task_port S42_TASK_SERVER_CTX_SIZE=2048 S42_DIAGNOSTIC_PORT=$phone_diagnostic_port S42_BUSYBOX=$phone_busybox S42_COMBINED_MIN_AVAILABLE_KIB=$phone_reserve_kib S42_COMBINED_READY_FILE=$combined_ready S42_BIND_ARM_FILE=$bind_arm sh $phone_session $phone_workers $phone_router $phone_gemma $phone_qwen $phone_root $restore_usb 7200 3 > $phone_root/session-launch.log 2>&1 < /dev/null & echo \$! > $phone_session_pid_file'"

combined_warm=0
for _ in $(seq 1 1200); do
    if "${adb_cmd[@]}" shell "su -c 'test -f $combined_ready'" \
            >/dev/null 2>&1; then
        combined_warm=1
        break
    fi
    sleep 0.25
done
if [[ $combined_warm -ne 1 ]]; then
    echo "combined phone residency did not become warm" >&2
    "${adb_cmd[@]}" shell \
        "su -c 'tail -n 160 $phone_root/session-launch.log; tail -n 160 $phone_root/session.log; tail -n 160 $phone_root/task-server.log'" \
        >&2 || true
    exit 1
fi
"${adb_cmd[@]}" shell "su -c 'cat /proc/meminfo'" \
    > "$capture/phone-meminfo-combined.txt"
"${adb_cmd[@]}" shell \
    "su -c 'sha256sum $phone_llama; cat $combined_ready; pidof llama-server'" \
    > "$capture/phone-combined-receipt.txt"
phone_model_sha=$(awk 'NR == 1 {print $1}' "$capture/phone-combined-receipt.txt")
expected_phone_sha=4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad
if [[ $phone_model_sha != "$expected_phone_sha" ]]; then
    echo "phone Llama model identity mismatch" >&2
    exit 1
fi
phone_total_bytes=$(awk '/^MemTotal:/ {print $2 * 1024}' \
    "$capture/phone-meminfo-combined.txt")
phone_available_bytes=$(awk '/^MemAvailable:/ {print $2 * 1024}' \
    "$capture/phone-meminfo-combined.txt")
if (( phone_available_bytes < phone_reserve_bytes )); then
    echo "combined phone residency is below the 768 MiB reserve" >&2
    exit 1
fi
ffn_manifest=
ffn_compiled_policy=
if (( ffn_route_required )); then
    ffn_manifest=$capture/LLAMA_FFN_MANIFEST.json
    ffn_compiled_policy=$capture/LLAMA_FFN_COMPILED_POLICY.json
    "$ffn_python" "$ffn_manifest_builder" \
        --model "$llama_model" \
        --expected-sha256 "$expected_phone_sha" \
        --expected-size 770928288 \
        --gguf-python-path "$repo_root/gguf-py" \
        --resident-layers "$llama_ffn_resident_layers" \
        --output "$ffn_manifest" \
        > "$capture/ffn-manifest.log"
    ffn_compile_command=(
        python3 "$ffn_policy_compiler"
        --manifest "$ffn_manifest"
        --kernel-campaign "$kernel_campaign"
        --phone-capacity-bytes "$phone_total_bytes"
        --phone-available-bytes "$phone_available_bytes"
        --phone-reserve-bytes "$phone_reserve_bytes"
        --output "$ffn_compiled_policy"
    )
    if [[ -n $ffn_physical_calibration ]]; then
        ffn_compile_command+=(
            --physical-calibration "$ffn_physical_calibration"
        )
    fi
    if [[ -n $ffn_compiled_policy_input ]]; then
        python3 - "$ffn_compiled_policy_input" "$ffn_manifest" \
                "$ffn_physical_calibration" "$phone_total_bytes" \
                "$phone_available_bytes" "$phone_reserve_bytes" <<'PY'
import hashlib
import json
import sys

policy_path, manifest_path, calibration_path = sys.argv[1:4]
capacity, available, reserve = map(int, sys.argv[4:])
canonical = lambda value: (json.dumps(
    value,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
) + "\n").encode("ascii")
policy = json.load(open(policy_path, encoding="ascii"))
manifest = json.load(open(manifest_path, encoding="ascii"))
unsigned = {key: value for key, value in policy.items()
            if key != "record_sha256"}
if not (
    policy.get("schema") == "s42-llama-ffn-vq-compiled-policy-v1"
    and policy.get("record_sha256")
        == hashlib.sha256(canonical(unsigned)).hexdigest()
    and policy.get("evidence", {}).get("manifest_record_sha256")
        == manifest.get("record_sha256")
    and policy.get("qualification", {}).get("physical_shape_calibrated")
        is True
    and policy.get("qualification", {}).get("route_admission")
        == "SHADOW_ONLY_UNTIL_HELDOUT_PHYSICAL_PROFILE"
    and available >= reserve
    and policy.get("memory_accounting", {}).get("phone_capacity_bytes")
        == capacity
):
    raise SystemExit("precompiled FFN policy failed live binding")
if not calibration_path:
    raise SystemExit("precompiled FFN policy requires physical calibration")
calibration = json.load(open(calibration_path, encoding="ascii"))
if policy.get("evidence", {}).get("physical_calibration_record_sha256") \
        != calibration.get("record_sha256"):
    raise SystemExit("precompiled FFN policy calibration mismatch")
PY
        cp -- "$ffn_compiled_policy_input" "$ffn_compiled_policy"
        printf '%s\n' "reused $ffn_compiled_policy_input" \
            > "$capture/ffn-compile.log"
    else
        "${ffn_compile_command[@]}" > "$capture/ffn-compile.log"
    fi
fi
"${adb_cmd[@]}" shell "su -c 'touch $bind_arm'"

aoa_ready=0
for _ in $(seq 1 360); do
    if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
        aoa_ready=1
        break
    fi
    sleep 1
done
if [[ $aoa_ready -ne 1 ]]; then
    echo "composite phone session did not bind FunctionFS" >&2
    exit 1
fi

phone_interface=
phone_host=
for _ in $(seq 1 240); do
    ip -o link show | awk -F': ' '{print $2}' | sort \
        > "$capture/net-after.txt"
    while IFS= read -r candidate; do
        [[ -n $candidate ]] || continue
        nmcli device connect "$candidate" >/dev/null 2>&1 || true
        if python3 -c \
                'import http.client,sys; c=http.client.HTTPConnection(sys.argv[1],int(sys.argv[2]),timeout=1); c.request("GET","/health"); r=c.getresponse(); ok=r.status==200 and b"ok" in r.read(); c.close(); raise SystemExit(0 if ok else 1)' \
                "192.168.42.1" "$phone_task_port" 2>/dev/null; then
            if (( ffn_route_required )) && ! python3 -c \
                    'import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),1); s.close()' \
                    "192.168.42.1" "$phone_ffn_port" 2>/dev/null; then
                continue
            fi
            phone_interface=$candidate
            phone_host=192.168.42.1
            break
        fi
    done < <(comm -13 "$capture/net-before.txt" "$capture/net-after.txt")
    [[ -n $phone_interface ]] && break
    sleep 0.5
done
if [[ -z $phone_interface ]]; then
    echo "resident 1B endpoint is not reachable over IPv4 NCM" >&2
    ip -6 addr show >&2
    exit 1
fi
ip addr show dev "$phone_interface" > "$capture/ncm-interface.txt"

automated_catalog=
if [[ $overlay_variant != direct-energy-calibration ]]; then
    automated_catalog=$capture/AUTOMATED_RUNTIME_CATALOG.json
    host_memory_bytes=$(awk '/^MemTotal:/ {print $2 * 1024}' /proc/meminfo)
    gpu_memory_mib=$(nvidia-smi --query-gpu=memory.total \
        --format=csv,noheader,nounits | awk 'NR == 1 {print int($1)}')
    gpu_memory_bytes=$((gpu_memory_mib * 1024 * 1024))
    control_overhead_args=()
    if [[ -n $automated_control_overhead_profile ]]; then
        control_overhead_args=(
            --control-overhead-profile "$automated_control_overhead_profile"
        )
    fi
    split_catalog_args=()
    catalog_route_profile=$qualified_route_profile
    catalog_route_audit=$qualified_route_audit
    catalog_large_policy=$large_model_policy
    catalog_identity_args=()
    if [[ $overlay_variant == qualification ]]; then
        catalog_route_profile=$here/SCHEDULER_PROFILE_RUNTIME_AUTO_V1.json
        catalog_route_audit=$here/SCHEDULER_PROFILE_RUNTIME_AUTO_V1_AUDIT.json
        catalog_large_policy=all
        catalog_identity_args=(
            --model-identity-audit "$qualified_route_audit"
        )
    fi
    if (( ffn_route_required )); then
        split_catalog_args=(
            --split-endpoint "http://127.0.0.1:18486"
            --ffn-manifest "$ffn_manifest"
            --ffn-policy "$ffn_compiled_policy"
        )
    fi
    python3 "$automated_catalog_builder" \
        --route-profile "$catalog_route_profile" \
        --route-profile-audit "$catalog_route_audit" \
        "${catalog_identity_args[@]}" \
        --kernel-profile "$kernel_campaign" \
        --requests "$overlay_trace" \
        --model-sha256 "$expected_phone_sha" \
        --model-bytes 770928288 \
        --cpu-endpoint "http://127.0.0.1:18484" \
        --gpu-endpoint "http://127.0.0.1:18485" \
        --phone-endpoint "http://$phone_host:$phone_task_port" \
        --host-memory-bytes "$host_memory_bytes" \
        --gpu-memory-bytes "$gpu_memory_bytes" \
        --phone-memory-bytes "$phone_total_bytes" \
        --marginal-system-profile "$marginal_system_profile" \
        --large-model-policy "$catalog_large_policy" \
        "${control_overhead_args[@]}" \
        "${split_catalog_args[@]}" \
        --output "$automated_catalog" \
        > "$capture/automated-catalog.log"
fi

if [[ $overlay_variant == direct-energy-calibration ]]; then
    if [[ -z $ffn_physical_calibration ]]; then
        echo "direct energy calibration requires physical shape calibration" >&2
        exit 1
    fi
    direct_output=$output/direct-energy-measurement
    python3 "$direct_energy_runner" \
        --requests "$overlay_trace" \
        --model "$llama_model" \
        --server-cpu "$cpu_server" \
        --cpu-lib-dir "$cuda_lib_dir" \
        --ffn-manifest "$ffn_manifest" \
        --ffn-policy "$ffn_compiled_policy" \
        --phone-host "$phone_host" \
        --phone-ffn-port "$phone_ffn_port" \
        --request-indices "$direct_energy_request_indices" \
        --abba-cycles "$direct_energy_abba_cycles" \
        --idle-duration-s "$direct_energy_idle_duration_s" \
        --split-prewarm-timeout-s "$ffn_prewarm_timeout_s" \
        --output "$direct_output" \
        --execute \
        --confirm RUN_S42_LLAMA_FFN_DIRECT_ENERGY

    recover_resident_usb
    collect_phone_capture
    python3 "$direct_energy_analyzer" \
        --result "$direct_output/RESULT.json" \
        --resource-samples "$direct_output/resource-samples.jsonl" \
        --phone-samples "$capture/phone-samples.tsv" \
        --clock-before "$capture/clock-before.json" \
        --clock-after "$capture/clock-after.json" \
        --ffn-manifest "$ffn_manifest" \
        --ffn-policy "$ffn_compiled_policy" \
        --split-log "$direct_output/llama1-cpu-phone-ffn.stderr" \
        --output "$capture/LLAMA_FFN_DIRECT_ENERGY.json" \
        --analysis-output "$capture/LLAMA_FFN_DIRECT_ENERGY_ANALYSIS.json"
    sha256sum \
        "$direct_output/RESULT.json" \
        "$direct_output/resource-samples.jsonl" \
        "$capture/phone-samples.tsv" \
        "$ffn_manifest" \
        "$ffn_compiled_policy" \
        "$capture/LLAMA_FFN_DIRECT_ENERGY.json" \
        "$capture/LLAMA_FFN_DIRECT_ENERGY_ANALYSIS.json" \
        > "$capture/SHA256SUMS.txt"
    cat "$capture/SHA256SUMS.txt"
    trap - EXIT
    exit 0
fi

if [[ $overlay_variant == qualification ]]; then
    unified_catalog=$capture/UNIFIED_RUNTIME_CATALOG.json
    python3 "$large_catalog_builder" \
        --qwen-control-endpoint "http://127.0.0.1:$qwen_control_port" \
        --qwen-phone-endpoint "http://127.0.0.1:$qwen_phone_port" \
        --gemma-control-endpoint "http://127.0.0.1:$gemma_control_port" \
        --gemma-phone-endpoint "http://127.0.0.1:$gemma_phone_port" \
        --bridge-host 127.0.0.1 \
        --bridge-port "$bridge_port" \
        --bridge-allocator malloc-split \
        --overlay-catalog "$automated_catalog" \
        --output "$unified_catalog" \
        > "$capture/unified-catalog.log"

    python3 "$unified_preflight" \
        --large-requests "$trace" \
        --overlay-requests "$overlay_trace" \
        --trace-manifest "$overlay_manifest" \
        --capability-catalog "$unified_catalog" \
        --qwen-manifest "$qwen_manifest" \
        --gemma-manifest "$gemma_manifest" \
        --qwen-model "$qwen_model" \
        --gemma-model "$gemma_model" \
        --llama-model "$llama_model" \
        --server "$server" \
        --resident-server "$cpu_server" \
        --cuda-lib-dir "$cuda_lib_dir" \
        --resident-lib-dir "$cuda_lib_dir" \
        --bridge "$bridge" \
        --phone-router "$phone_router_host" \
        --phone-router-receipt "$capture/phone-router.sha256" \
        --close-helper "$close_helper" \
        --phone-diagnostic-endpoint \
            "http://$phone_host:$phone_diagnostic_port" \
        --phone-battery-ppm "$((phone_battery_level * 10000))" \
        --phone-usb-serial "$serial" \
        --phone-normal-usb-receipt "$capture/PHONE_USB_BEFORE.json" \
        --adb-port "$adb_port" \
        --minimum-usb-speed-mbps 5000 \
        --output "$capture/PHYSICAL_PREFLIGHT.json" \
        > "$capture/physical-preflight.log"

    if [[ $preflight_only == 1 ]]; then
        cleanup
        trap - EXIT
        capture_normal_usb_receipt "$capture/PHONE_USB_AFTER.json"
        sha256sum \
            "$unified_catalog" \
            "$capture/PHYSICAL_PREFLIGHT.json" \
            "$capture/PHONE_USB_BEFORE.json" \
            "$capture/PHONE_USB_AFTER.json" \
            "$capture/phone-router.sha256" \
            > "$capture/SHA256SUMS.txt"
        exit 0
    fi

    selection_mode=energy-aware
    if [[ $large_model_policy == cpu-overflow \
            && $small_model_policy == static-cpu ]]; then
        selection_mode=desktop-baseline
    fi
    unified_command=(
        python3 "$unified_runner"
        --large-requests "$trace"
        --overlay-requests "$overlay_trace"
        --trace-manifest "$overlay_manifest"
        --capability-catalog "$unified_catalog"
        --qwen-manifest "$qwen_manifest"
        --gemma-manifest "$gemma_manifest"
        --qwen-model "$qwen_model"
        --gemma-model "$gemma_model"
        --llama-model "$llama_model"
        --server "$server"
        --resident-server "$cpu_server"
        --cuda-lib-dir "$cuda_lib_dir"
        --resident-lib-dir "$cuda_lib_dir"
        --bridge "$bridge"
        --close-helper "$close_helper"
        --phone-diagnostic-endpoint \
            "http://$phone_host:$phone_diagnostic_port"
        --phone-battery-ppm "$((phone_battery_level * 10000))"
        --phone-usb-serial "$serial"
        --adb-port "$adb_port"
        --minimum-usb-speed-mbps 5000
        --selection-mode "$selection_mode"
        --output "$combined_output"
        --execute
        --confirm RUN_UNIFIED_FP16_LLAMA_OVERLAY
    )
    if [[ -n $unified_request_indices ]]; then
        unified_command+=(--request-indices "$unified_request_indices")
    fi
    S42_UNIFIED_REPO_ROOT="$repo_root" \
    PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
    systemd-run --user --scope -p MemorySwapMax=0 \
        "${unified_command[@]}" \
        > "$capture/unified-runner.log" 2>&1

    recover_resident_usb
    capture_normal_usb_receipt "$capture/PHONE_USB_AFTER.json"
    collect_phone_capture
    python3 "$burst_dir/analyze_phone_energy.py" \
        --result "$combined_output/RESULT.json" \
        --samples "$capture/phone-samples.tsv" \
        --clock-before "$capture/clock-before.json" \
        --clock-after "$capture/clock-after.json" \
        --output "$capture/PHONE_ENERGY.json"
    sha256sum \
        "$unified_catalog" \
        "$capture/PHYSICAL_PREFLIGHT.json" \
        "$combined_output/RESULT.json" \
        "$combined_output/SCHEDULER_DECISION_LOG.json" \
        "$capture/PHONE_ENERGY.json" \
        "$capture/PHONE_USB_BEFORE.json" \
        "$capture/PHONE_USB_AFTER.json" \
        "$capture/phone-router.sha256" \
        "$capture/phone-samples.tsv" \
        > "$capture/SHA256SUMS.txt"
    cat "$capture/SHA256SUMS.txt"
    trap - EXIT
    exit 0
fi

controller_command=(
    python3 "$controller"
    --requests "$overlay_trace"
    --manifest "$overlay_manifest"
    --base-output "$base_output"
    --output "$combined_output"
    --ready-file "$controller_ready"
    --resident-release-file "$resident_release"
    --server-cuda "$server"
    --server-cpu "$cpu_server"
    --cuda-lib-dir "$cuda_lib_dir"
    --cpu-lib-dir "$cuda_lib_dir"
    --llama1-model "$llama_model"
    --phone-host "$phone_host"
    --phone-port "$phone_task_port"
    --phone-ffn-port "$phone_ffn_port"
    --phone-diagnostic-port "$phone_diagnostic_port"
    --phone-memory-total-bytes "$phone_total_bytes"
    --phone-memory-available-bytes "$phone_available_bytes"
    --phone-memory-reserve-bytes "$phone_reserve_bytes"
    --phone-model-sha256 "$phone_model_sha"
    --ffn-prewarm-timeout-s "$ffn_prewarm_timeout_s"
    --forced-static-route "$forced_static_route"
    --runtime-profile "$runtime_profile"
    --phone-battery-ppm "$((phone_battery_level * 10000))"
    --marginal-system-profile "$marginal_system_profile"
    --large-model-policy "$effective_large_model_policy"
    --requested-large-model-policy "$large_model_policy"
    --small-model-policy "$small_model_policy"
    --execute
    --confirm RUN_FULL_FP16_LLAMA1B_OVERLAY
)
if [[ -n $automated_catalog ]]; then
    controller_command+=(--automated-catalog "$automated_catalog")
fi
if (( ffn_route_required )); then
    controller_command+=(
        --ffn-manifest "$ffn_manifest"
        --ffn-compiled-policy "$ffn_compiled_policy"
    )
fi
S42_UNIFIED_REPO_ROOT="$repo_root" \
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
systemd-run --user --scope --unit="$controller_unit" \
    -p MemorySwapMax=0 "${controller_command[@]}" \
    > "$capture/controller.log" 2>&1 &
controller_pid=$!
controller_ready_seen=0
for _ in $(seq 1 1200); do
    if [[ -f $controller_ready ]]; then
        controller_ready_seen=1
        break
    fi
    if ! kill -0 "$controller_pid" 2>/dev/null; then
        break
    fi
    sleep 0.25
done
if [[ $controller_ready_seen -ne 1 ]]; then
    echo "small-model controller did not become ready" >&2
    tail -n 160 "$capture/controller.log" >&2 || true
    exit 1
fi

base_command=(
    python3 "$runner"
    --mode fp16-switch
    --requests "$trace"
    --hot-server "$server"
    --hot-model "$qwen_model"
    --cuda-lib-dir "$cuda_lib_dir"
    --cold-server "$server"
    --cold-model "$gemma_model"
    --cold-lib-dir "$server_bin"
    --bridge "$bridge"
    --bridge-allocator malloc-split
    --hot-port "$hot_port"
    --cold-port "$cold_port"
    --gpu-cold-port "$gpu_cold_port"
    --bridge-port "$bridge_port"
    --hot-ctx-size 24576
    --hot-parallel 4
    --cold-ctx-size 32768
    --cold-parallel 8
    --cold-batch-size 4096
    --cold-ubatch-size 512
    --fp16-resident-plan "$plan"
    --fp16-arm "$large_arm"
    --resident-close-helper "$close_helper"
    --resident-release-file "$resident_release"
    --output "$base_output"
    --execute
    --confirm RUN_HIERARCHICAL_BURSTGPT_TRACE
)
if [[ $large_model_policy == runtime-auto ]]; then
    base_command+=(--fp16-request-profile "$fp16_request_profile")
fi
set +e
S42_UNIFIED_REPO_ROOT="$repo_root" \
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
systemd-run --user --scope --unit="$base_unit" \
    -p MemorySwapMax=0 "${base_command[@]}" \
    > "$capture/base-runner.log" 2>&1 &
base_scope_pid=$!
while kill -0 "$base_scope_pid" 2>/dev/null \
        && kill -0 "$controller_pid" 2>/dev/null; do
    sleep 1
done
if ! kill -0 "$controller_pid" 2>/dev/null \
        && kill -0 "$base_scope_pid" 2>/dev/null \
        && [[ ! -f $base_output/RESULT.json ]]; then
    wait "$controller_pid"
    controller_rc=$?
    controller_pid=
    interrupt_base_scope
    wait "$base_scope_pid"
    base_rc=$?
    base_scope_pid=
else
    wait "$base_scope_pid"
    base_rc=$?
    base_scope_pid=
    if [[ $base_rc -ne 0 ]]; then
        interrupt_controller
    fi
    wait "$controller_pid"
    controller_rc=$?
    controller_pid=
fi
set -e
if [[ $base_rc -ne 0 || $controller_rc -ne 0 \
        || ! -f $base_output/RESULT.json \
        || ! -f $combined_output/RESULT.json ]]; then
    echo "combined F16 run failed: base=$base_rc controller=$controller_rc" >&2
    tail -n 160 "$capture/base-runner.log" >&2 || true
    tail -n 160 "$capture/controller.log" >&2 || true
    exit 1
fi

collect_phone_capture

python3 "$burst_dir/analyze_phone_energy.py" \
    --result "$combined_output/RESULT.json" \
    --samples "$capture/phone-samples.tsv" \
    --clock-before "$capture/clock-before.json" \
    --clock-after "$capture/clock-after.json" \
    --output "$capture/PHONE_ENERGY.json"

qualification_command=(
    python3 "$qualifier"
    --result "$combined_output/RESULT.json"
    --cpu-log "$combined_output/llama1-cpu.stderr"
    --phone-log "$capture/task-server.log"
    --cuda-log "$combined_output/llama1-cuda.stderr"
    --resident-release "$resident_release"
    --output "$capture/QUALIFICATION.json"
)
if (( ffn_route_required )); then
    qualification_command+=(
        --split-log "$combined_output/llama1-cpu-phone-ffn.stderr"
    )
fi
if [[ $small_model_policy == runtime-scheduler \
        && $large_model_policy != runtime-auto ]]; then
    qualification_command+=(--require-nonbaseline)
fi
if [[ -n $required_runtime_route ]]; then
    qualification_command+=(--require-route "$required_runtime_route")
fi
set +e
"${qualification_command[@]}"
qualification_rc=$?
set -e
qualification_status=$(python3 - "$capture/QUALIFICATION.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="ascii"))
if value.get("schema") != "s42-fp16-llama1b-physical-qualification-v1":
    raise SystemExit("invalid qualification schema")
print(value.get("status"))
PY
)
if [[ ( $qualification_rc -eq 0 && $qualification_status != PASS ) \
        || ( $qualification_rc -eq 2 && $qualification_status != FAIL ) \
        || ( $qualification_rc -ne 0 && $qualification_rc -ne 2 ) ]]; then
    echo "invalid physical qualification receipt" >&2
    exit 1
fi

hash_inputs=(
    "$plan"
    "$runtime_snapshot"
    "$base_output/RESULT.json"
    "$combined_output/RESULT.json"
    "$combined_output/phone-power-samples.json"
    "$combined_output/resource-samples.jsonl"
    "$six_model_runner"
    "$capture/PHONE_ENERGY.json"
    "$capture/QUALIFICATION.json"
    "$resident_release"
    "$capture/phone-samples.tsv"
)
if [[ $large_model_policy == runtime-auto ]]; then
    hash_inputs+=("$fp16_request_profile")
fi
if (( ffn_route_required )); then
    hash_inputs+=("$ffn_manifest" "$ffn_compiled_policy")
fi
if [[ -n $automated_catalog ]]; then
    hash_inputs+=("$automated_catalog")
fi
sha256sum "${hash_inputs[@]}" > "$capture/SHA256SUMS.txt"
cat "$capture/SHA256SUMS.txt"
trap - EXIT
exit "$qualification_rc"
