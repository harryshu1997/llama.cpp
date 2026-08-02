#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
expected_repo=${S24_A6000_REPO:-/home/myid/zs89458/Documents/llama.cpp-release}
action=${1:-}
session_dir=${2:-}
op12_serial=${S24_OP12_SERIAL:-5ae7a43d}
op15_serial=${S24_OP15_SERIAL:-3C15AU002CL00000}
remote_dir=${S24_PHONE_DIR:-/data/local/tmp/ls-s24}
op12_model=${S24_OP12_HEAD:-/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf}
op15_model=${S24_OP15_MID:-/data/local/tmp/ls-s24/12b-f16-mid-8-16.gguf}
op12_layer_start=${S24_OP12_LAYER_START:-0}
op12_layer_end=${S24_OP12_LAYER_END:-8}
op15_layer_start=${S24_OP15_LAYER_START:-8}
op15_layer_end=${S24_OP15_LAYER_END:-16}
op12_port=${S24_OP12_PORT:-24280}
op15_port=${S24_OP15_PORT:-24281}
context=${S24_PHONE_CONTEXT:-600}
max_streams=${S24_PHONE_STREAMS:-4}
max_prefill=${S24_PHONE_MAX_PREFILL:-64}
n_gen=${S24_PHONE_N_GEN:-64}
op12_mbuf=${S24_OP12_MBUF:-4192}
op15_mbuf=${S24_OP15_MBUF:-4192}
provision_record=${S24_PROVISION_RECORD:-"$s24_dir/results/a6000_provision/provision.env"}
android_dir=${S24_ANDROID_DIR:-"$repo_root/npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin"}
op12_head_sha=a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8

fail() {
    echo "error: $*" >&2
    exit 2
}

[[ "$repo_root" == "$expected_repo" ]] || \
    fail "this script must run from the authoritative A6000 checkout: $expected_repo"
[[ "$action" =~ ^(start|status|stop|collect)$ ]] || \
    fail "usage: $0 {start|status|stop|collect} SESSION_DIR"
[[ -n "$session_dir" ]] || fail "SESSION_DIR is required"
for value in "$op12_port" "$op15_port" "$context" "$max_streams" \
    "$max_prefill" "$n_gen" "$op12_mbuf" "$op15_mbuf" \
    "$op12_layer_start" "$op12_layer_end" "$op15_layer_start" \
    "$op15_layer_end"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fail "numeric configuration is invalid"
done
(( context >= max_prefill + n_gen )) || fail "context must cover max-prefill plus n-gen"
(( max_streams >= 1 && max_streams <= 64 )) || fail "stream count is out of range"
(( op12_layer_start < op12_layer_end && op15_layer_start < op15_layer_end )) || \
    fail "phone layer range is invalid"

phone_pid_file() {
    printf '%s/%s.phone.pid' "$session_dir" "$1"
}

controller_pid_file() {
    printf '%s/%s.controller.pid' "$session_dir" "$1"
}

worker_log() {
    printf '%s/%s.log' "$session_dir" "$1"
}

controller_alive() {
    local name=$1
    local path
    path=$(controller_pid_file "$name")
    [[ -f "$path" ]] || return 1
    local pid
    pid=$(<"$path")
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    local serial=$op12_serial
    [[ "$name" == OP15 ]] && serial=$op15_serial
    local command_line
    command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    [[ "$command_line" == *adb* && "$command_line" == *"$serial"* ]]
}

remote_pid_alive() {
    local name=$1
    local serial=$2
    local path
    path=$(phone_pid_file "$name")
    [[ -f "$path" ]] || return 1
    local pid
    pid=$(<"$path")
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    local command_line
    command_line=$(adb -s "$serial" shell \
        "tr '\\000' ' ' < /proc/$pid/cmdline" \
        2>/dev/null | tr -d '\r' || true)
    [[ "$command_line" == *"$remote_dir/llama-layersplit"* || \
       "$command_line" == *"./llama-layersplit"* ]]
}

remote_sha256() {
    local serial=$1
    local path=$2
    adb -s "$serial" shell sha256sum "$path" | tr -d '\r' | awk '{print $1}'
}

launch_worker() {
    local name=$1
    local serial=$2
    local model=$3
    local layer_start=$4
    local layer_end=$5
    local port=$6
    local mbuf=$7
    local model_sha256=$8
    local log
    log=$(worker_log "$name")
    local remote_command
    remote_command="cd $remote_dir && echo S24_PHONE_PID=\$\$ >&2 && exec env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=$mbuf LLAMA_LAYER_START=$layer_start LLAMA_LAYER_END=$layer_end LAYERSPLIT_MODEL_SHA256=$model_sha256 LAYERSPLIT_PLACEMENT_CERT=1 ./llama-layersplit -m $model --devices HTP0 -ngl 99 --mode stagenet --port $port -n $n_gen --driver-batch $max_streams --driver-context $context --driver-max-prefill $max_prefill"
    printf '%s\n' "$remote_command" > "$session_dir/$name.command.txt"
    nohup adb -s "$serial" shell "$remote_command" \
        > "$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$(controller_pid_file "$name")"
}

wait_ready() {
    local name=$1
    local serial=$2
    local log
    log=$(worker_log "$name")
    for _attempt in $(seq 1 3600); do
        if grep -q '\[stagenet\] listening' "$log"; then
            local pid
            pid=$(sed -n 's/.*S24_PHONE_PID=\([0-9][0-9]*\).*/\1/p' "$log" | head -n 1)
            [[ "$pid" =~ ^[0-9]+$ ]] || fail "$name did not report its phone PID"
            printf '%s\n' "$pid" > "$(phone_pid_file "$name")"
            remote_pid_alive "$name" "$serial" || fail "$name phone PID is not live"
            return
        fi
        if ! controller_alive "$name"; then
            tail -80 "$log" >&2 || true
            fail "$name ADB controller exited during load"
        fi
        sleep 0.1
    done
    fail "$name worker load timed out"
}

safe_force_stop() {
    local name=$1
    local serial=$2
    local pid_path
    pid_path=$(phone_pid_file "$name")
    if [[ -f "$pid_path" ]]; then
        local pid
        pid=$(<"$pid_path")
        if [[ "$pid" =~ ^[0-9]+$ ]]; then
            local command_line
            command_line=$(adb -s "$serial" shell \
                "tr '\\000' ' ' < /proc/$pid/cmdline" \
                2>/dev/null | tr -d '\r' || true)
            if [[ "$command_line" == *llama-layersplit* ]]; then
                adb -s "$serial" shell kill "$pid" >/dev/null 2>&1 || true
            fi
        fi
    fi
    local controller_path
    controller_path=$(controller_pid_file "$name")
    if [[ -f "$controller_path" ]]; then
        local controller_pid
        controller_pid=$(<"$controller_path")
        if [[ "$controller_pid" =~ ^[0-9]+$ ]] && \
            kill -0 "$controller_pid" 2>/dev/null; then
            kill "$controller_pid" 2>/dev/null || true
            wait "$controller_pid" 2>/dev/null || true
        fi
    fi
}

case "$action" in
    start)
        [[ ${S24_CONFIRM_A6000_CONTROL:-NO} == YES ]] || \
            fail "set S24_CONFIRM_A6000_CONTROL=YES on the A6000 host"
        [[ ! -e "$session_dir" ]] || fail "session output already exists: $session_dir"
        [[ -f "$provision_record" ]] || fail "missing provisioning record: $provision_record"
        [[ -f "$android_dir/llama-layersplit" ]] || fail "missing provisioned Android binary"
        expected_op12_sha=${S24_OP12_MODEL_SHA256:-$op12_head_sha}
        expected_mid_sha=${S24_OP15_MODEL_SHA256:-$(awk -F= '$1 == "mid_sha256" {print $2}' "$provision_record")}
        [[ "$expected_op12_sha" =~ ^[0-9a-f]{64}$ ]] || \
            fail "OP12 model hash is invalid"
        [[ "$expected_mid_sha" =~ ^[0-9a-f]{64}$ ]] || \
            fail "provisioning record has no valid middle-shard hash"
        expected_binary_sha=$(sha256sum "$android_dir/llama-layersplit" | awk '{print $1}')
        mkdir -p "$session_dir"
        starting=1
        cleanup_start() {
            local status=$?
            if [[ "$starting" -eq 1 ]]; then
                safe_force_stop OP12 "$op12_serial"
                safe_force_stop OP15 "$op15_serial"
            fi
            exit "$status"
        }
        trap cleanup_start EXIT INT TERM
        adb devices -l > "$session_dir/adb-devices.txt"
        for serial in "$op12_serial" "$op15_serial"; do
            row=$(awk -v serial="$serial" '$1 == serial {print}' "$session_dir/adb-devices.txt")
            [[ "$row" == *" device "* ]] || fail "$serial is not in ADB device state"
            device_path=$(adb -s "$serial" get-devpath | tr -d '\r')
            [[ "$device_path" == usb:* ]] || fail "$serial is not an ADB USB device"
            printf '%s  %s\n' "$serial" "$device_path" >> "$session_dir/adb-usb-devpaths.txt"
        done
        [[ $(remote_sha256 "$op12_serial" "$op12_model") == "$expected_op12_sha" ]] || \
            fail "OP12 head shard changed after provisioning"
        [[ $(remote_sha256 "$op15_serial" "$op15_model") == "$expected_mid_sha" ]] || \
            fail "OP15 middle shard changed after provisioning"
        [[ $(remote_sha256 "$op12_serial" "$remote_dir/llama-layersplit") == "$expected_binary_sha" ]] || \
            fail "OP12 binary changed after provisioning"
        [[ $(remote_sha256 "$op15_serial" "$remote_dir/llama-layersplit") == "$expected_binary_sha" ]] || \
            fail "OP15 binary changed after provisioning"
        launch_worker OP12 "$op12_serial" "$op12_model" \
            "$op12_layer_start" "$op12_layer_end" "$op12_port" "$op12_mbuf" \
            "$expected_op12_sha"
        launch_worker OP15 "$op15_serial" "$op15_model" \
            "$op15_layer_start" "$op15_layer_end" "$op15_port" "$op15_mbuf" \
            "$expected_mid_sha"
        wait_ready OP12 "$op12_serial"
        wait_ready OP15 "$op15_serial"
        {
            printf 'schema=s24-a6000-phone-session-v1\n'
            printf 'runtime_activation_relay=DESKTOP_DIRECT_WIFI\n'
            printf 'a6000_activation_relay=FORBIDDEN\n'
            printf 'a6000_gpu_use=FORBIDDEN_AND_NOT_INVOKED\n'
            printf 'op12_endpoint=192.168.1.193:%s\n' "$op12_port"
            printf 'op15_endpoint=192.168.1.97:%s\n' "$op15_port"
            printf 'context=%s\n' "$context"
            printf 'max_streams=%s\n' "$max_streams"
            printf 'max_prefill=%s\n' "$max_prefill"
            printf 'op12_mbuf_mib=%s\n' "$op12_mbuf"
            printf 'op15_mbuf_mib=%s\n' "$op15_mbuf"
            printf 'op12_layer_range=%s:%s\n' "$op12_layer_start" "$op12_layer_end"
            printf 'op15_layer_range=%s:%s\n' "$op15_layer_start" "$op15_layer_end"
            printf 'op12_head_sha256=%s\n' "$expected_op12_sha"
            printf 'op15_mid_sha256=%s\n' "$expected_mid_sha"
            printf 'phone_binary_sha256=%s\n' "$expected_binary_sha"
            printf 'provision_record=%s\n' "$provision_record"
        } > "$session_dir/session.env"
        starting=0
        trap - EXIT INT TERM
        printf '%s\n' "$session_dir"
        ;;
    status)
        for pair in "OP12:$op12_serial" "OP15:$op15_serial"; do
            name=${pair%%:*}
            serial=${pair#*:}
            controller=dead
            phone=dead
            controller_alive "$name" && controller=live
            remote_pid_alive "$name" "$serial" && phone=live
            printf '%s controller=%s phone=%s\n' "$name" "$controller" "$phone"
        done
        ;;
    stop)
        safe_force_stop OP12 "$op12_serial"
        safe_force_stop OP15 "$op15_serial"
        printf 'FORCED_STOP_NO_PLACEMENT_ACCEPTANCE\n' > "$session_dir/forced-stop.txt"
        ;;
    collect)
        for pair in "OP12:$op12_serial" "OP15:$op15_serial"; do
            name=${pair%%:*}
            serial=${pair#*:}
            if controller_alive "$name"; then
                fail "$name controller is still live; end StageNet with STOP first"
            fi
            remote_pid_alive "$name" "$serial" && \
                fail "$name phone process is still live"
        done
        (
            cd "$session_dir"
            find . -type f ! -name SHA256SUMS.txt -print0 |
                LC_ALL=C sort -z |
                xargs -0 sha256sum > SHA256SUMS.txt
        )
        printf '%s\n' "$session_dir"
        ;;
esac
