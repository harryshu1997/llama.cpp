#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <output-root> <adb-port> <gpu-control-evidence> <helper-evidence>" >&2
    exit 2
fi

output_root=$1
adb_port=$2
gpu_control=$3
helper_evidence=$4

if [[ $output_root != /* || -e $output_root ]]; then
    echo "invalid output root: $output_root" >&2
    exit 2
fi
if [[ ! -f $gpu_control || ! -f $helper_evidence ]]; then
    echo "missing scheduler evidence" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(cd -- "$here/../../../.." && pwd)
burst_dir=$repo_root/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1
runner=$repo_root/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/mixed_scheduler_v1/run_hierarchical_trace.py
trace=$burst_dir/REQUESTS_SEMANTIC_SOURCE.jsonl
serial=3C15AU002CL00000
device_lock=/tmp/s42-burstgpt-overflow-device.lock
phone_logger=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_logger.sh
phone_policy=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_allow.rules
phone_session=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_ffn_session-flex.sh
phone_worker=/data/local/tmp/s41-opoffload-dmabuf-v1/llama-ffn-split-worker-reset-qualified
phone_model=/data/local/tmp/ls-s32/gemma-4-12B-it-Q4_0.gguf
restore_usb=/data/local/tmp/s41-opoffload-dmabuf-v1/restore_phone_usb.sh
bridge=/home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-reset-qualified
split_policy=1:9664,3:8192,8:4096,128:8192,512:11136
adb_cmd=(adb -P "$adb_port" -s "$serial")
active_marker=
power_pid=
session_pid=

if ! mkdir "$device_lock" 2>/dev/null; then
    echo "device lock is held: $device_lock" >&2
    exit 1
fi

cleanup() {
    if [[ -n $active_marker ]]; then
        "${adb_cmd[@]}" shell "su -c 'rm -f $active_marker'" \
            >/dev/null 2>&1 || true
    fi
    if [[ -n $power_pid ]]; then
        wait "$power_pid" 2>/dev/null || true
    fi
    if [[ -n $session_pid ]]; then
        wait "$session_pid" 2>/dev/null || true
    fi
    rmdir "$device_lock" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

require_idle_devices() {
    if pgrep -x llama-server >/dev/null 2>&1 || \
       pgrep -f '/ffn_dmabuf_bridge' >/dev/null 2>&1; then
        echo "foreign inference process owns the devices" >&2
        pgrep -af 'llama-server|ffn_dmabuf_bridge' >&2 || true
        exit 1
    fi
    "${adb_cmd[@]}" get-state >/dev/null
}

run_arm() {
    local arm=$1
    local mode runner_mode output capture plan run_id phone_root
    local samples armed done power_log session_log run_rc
    mode=control
    runner_mode=gpu-switch
    if [[ $arm == shadow ]]; then
        mode=shadow
        runner_mode=op15-switch
    fi
    output=$output_root/$arm
    capture=${output}.capture
    plan=$capture/PLAN.json
    run_id=${output_root##*/}-$arm
    phone_root=/data/local/tmp/s41-opoffload-dmabuf-v1/$run_id
    samples=$phone_root/samples.tsv
    active_marker=$phone_root/power.active
    armed=$phone_root/power.armed
    done=$phone_root/power.done
    power_log=$capture/power-adb.log
    session_log=$capture/session-adb.log

    require_idle_devices
    mkdir -p "$capture"
    planner=(
        python3 "$here/plan_gpu_overflow_burstgpt.py"
        --trace "$trace"
        --gpu-control-result "$gpu_control"
        --mode "$mode"
        --output "$plan"
    )
    if [[ $arm == shadow ]]; then
        planner+=(--helper-result "$helper_evidence")
    fi
    "${planner[@]}" >"$capture/planner.log"

    "${adb_cmd[@]}" shell \
        "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
    "${adb_cmd[@]}" shell \
        "su -c 'test ! -e $phone_root && mkdir -p $phone_root'"
    python3 "$burst_dir/capture_phone_clock.py" \
        --adb-port "$adb_port" --serial "$serial" \
        --output "$capture/clock-before.json"
    sha256sum "$bridge" "$runner" "$plan" \
        >"$capture/host-sha256.txt"
    "${adb_cmd[@]}" shell sha256sum \
        "$phone_worker" "$phone_model" "$phone_session" "$restore_usb" \
        >"$capture/phone-sha256.txt"

    nohup "${adb_cmd[@]}" shell \
        "su -c 'sh $phone_logger $samples $active_marker $armed $done 600'" \
        >"$power_log" 2>&1 &
    power_pid=$!
    local logger_ready=0
    for _ in $(seq 1 300); do
        if "${adb_cmd[@]}" shell "su -c 'test -f $armed'" \
                >/dev/null 2>&1; then
            logger_ready=1
            break
        fi
        sleep 0.1
    done
    if [[ $logger_ready -ne 1 ]]; then
        echo "phone power logger did not arm" >&2
        exit 1
    fi
    "${adb_cmd[@]}" shell "su -c 'touch $active_marker'"

    if [[ $arm == shadow ]]; then
        nohup "${adb_cmd[@]}" shell \
            "su -c 'S41_FFN_F16_IO=1 S41_FFN_MAX_TOKENS=512 S41_FFN_COLUMN_QUANTUM=2048 S41_FFN_ALTERNATE_COLUMNS=9664 sh $phone_session $phone_worker $phone_model 0-47 11136 HTP0 $phone_root $restore_usb 1800 120000'" \
            >"$session_log" 2>&1 &
        session_pid=$!
        local aoa_ready=0
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
        --mode "$runner_mode"
        --requests "$trace"
        --scheduler-plan "$plan"
        --hot-server /home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server
        --hot-model /home/zhihao/models/Qwen3-14B-Q4_K_M.gguf
        --cuda-lib-dir /mnt/storage/s21_deps/cuda-13.2.1/lib
        --cold-server /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin/llama-server
        --cold-model /home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf
        --cold-lib-dir /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin
        --bridge "$bridge"
        --bridge-allocator devmem
        --policy-id i3-hidden-wait
        --split-policy "$split_policy"
        --output "$output"
        --execute
        --confirm RUN_HIERARCHICAL_BURSTGPT_TRACE
    )
    set +e
    systemd-run --user --scope "${command[@]}" \
        >"$capture/runner.log" 2>&1
    run_rc=$?
    set -e

    if [[ -n $session_pid ]]; then
        wait "$session_pid" || true
        session_pid=
    fi
    local adb_ready=0
    for _ in $(seq 1 180); do
        if "${adb_cmd[@]}" get-state >/dev/null 2>&1 && \
           "${adb_cmd[@]}" shell true >/dev/null 2>&1; then
            adb_ready=1
            break
        fi
        sleep 1
    done
    if [[ $adb_ready -ne 1 ]]; then
        echo "phone did not return to ADB" >&2
        exit 1
    fi
    local marker_removed=0
    for _ in $(seq 1 30); do
        if "${adb_cmd[@]}" shell \
                "su -c 'rm -f $active_marker && test ! -e $active_marker'" \
                >/dev/null 2>&1; then
            marker_removed=1
            break
        fi
        sleep 1
    done
    if [[ $marker_removed -ne 1 ]]; then
        echo "phone power marker could not be removed" >&2
        exit 1
    fi
    active_marker=
    wait "$power_pid" || true
    power_pid=
    "${adb_cmd[@]}" shell "su -c 'test -f $done'"
    python3 "$burst_dir/capture_phone_clock.py" \
        --adb-port "$adb_port" --serial "$serial" \
        --output "$capture/clock-after.json"
    "${adb_cmd[@]}" pull "$samples" "$capture/phone-samples.tsv" \
        >"$capture/pull.log"
    "${adb_cmd[@]}" pull "$phone_root/worker.log" "$capture/worker.log" \
        >>"$capture/pull.log" 2>&1 || true
    "${adb_cmd[@]}" pull "$phone_root/session.log" "$capture/session.log" \
        >>"$capture/pull.log" 2>&1 || true

    if [[ $run_rc -ne 0 || ! -f $output/RESULT.json ]]; then
        echo "$arm runner failed: rc=$run_rc" >&2
        exit "$run_rc"
    fi
    python3 "$burst_dir/analyze_phone_energy.py" \
        --result "$output/RESULT.json" \
        --samples "$capture/phone-samples.tsv" \
        --clock-before "$capture/clock-before.json" \
        --clock-after "$capture/clock-after.json" \
        --output "$capture/PHONE_ENERGY_V3.json"
    sha256sum \
        "$plan" "$output/RESULT.json" \
        "$capture/PHONE_ENERGY_V3.json" "$capture/phone-samples.tsv" \
        >"$capture/SHA256SUMS.txt"
}

mkdir "$output_root"
run_arm control
run_arm shadow
trap - EXIT INT TERM
rmdir "$device_lock"
printf '%s\n' "$output_root/control/RESULT.json" \
    "$output_root/shadow/RESULT.json"
