#!/usr/bin/env bash
set -euo pipefail

result_dir=${1:-}
label=${2:-}
cut=${3:-}
phone_backend=${4:-}
phone_model=${5:-}

host_binary=${S41_HOST_BINARY:-/home/zhihao/llama.cpp-release/build-fullmodel-cpu/bin/llama-layersplit}
host_model=${S41_HOST_MODEL:-/home/zhihao/models/gemma-4-12B-it-Q8_0-7b56.gguf}
phone_control=${S41_PHONE_CONTROL:-172.20.173.218:5555}
phone_dir=${S41_PHONE_DIR:-/data/local/tmp/ls-s34-identity}
phone_stage_port=${S41_PHONE_STAGE_PORT:-24381}
host_bridge_port=${S41_HOST_BRIDGE_PORT:-26381}
phone_mbuf=${S41_PHONE_MBUF:-4192}
phone_bridge=${S41_PHONE_BRIDGE:-/data/local/tmp/s41-fullmodel-aoa-v1/aoa_phone_bridge}
host_bridge=${S41_HOST_BRIDGE:-/tmp/s41-fullmodel-aoa-v1/aoa_host_bridge}
aoa_switch=${S41_AOA_SWITCH:-/tmp/s41-fullmodel-aoa-v1/aoa_bench.py}
phone_log_root=${S41_PHONE_LOG_ROOT:-/data/local/tmp/s41-fullmodel-aoa-v1}
prompt=${S41_PROMPT:-Explain why arithmetic intensity matters for offloading a matrix multiplication.}
n_gen=${S41_N_GEN:-32}
requests=${S41_REQUESTS:-5}
warmups=${S41_WARMUPS:-1}
context=${S41_CONTEXT:-256}
max_prefill=${S41_MAX_PREFILL:-128}
threads=${S41_THREADS:-8}

fail() {
    echo "error: $*" >&2
    exit 2
}

[[ -n "$result_dir" ]] || fail "usage: $0 RESULT_DIR LABEL CUT BACKEND PHONE_MODEL"
[[ "$result_dir" == /* ]] || fail "RESULT_DIR must be absolute"
[[ ! -e "$result_dir" ]] || fail "RESULT_DIR already exists"
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid case label"
[[ "$cut" =~ ^[1-9][0-9]*$ ]] || fail "invalid layer cut"
(( cut < 48 )) || fail "layer cut must be below 48"
[[ "$phone_backend" == HTP0 || "$phone_backend" == GPUOpenCL ]] || \
    fail "phone backend must be HTP0 or GPUOpenCL"
[[ "$phone_model" == /* && "$phone_model" != *" "* ]] || \
    fail "phone model must be an absolute path without spaces"
[[ -x "$host_binary" ]] || fail "missing host binary: $host_binary"
[[ -f "$host_model" ]] || fail "missing host model: $host_model"
[[ -x "$host_bridge" ]] || fail "missing host AOA bridge: $host_bridge"
[[ -f "$aoa_switch" ]] || fail "missing AOA switch helper: $aoa_switch"
[[ "$(hostname)" == zhihao-Z690-C-ac ]] || fail "wrong physical host"
for value in "$n_gen" "$requests" "$warmups" "$context" "$max_prefill" \
        "$threads" "$phone_stage_port" "$host_bridge_port"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fail "invalid numeric configuration"
done
(( n_gen > 0 && requests > 0 && context >= max_prefill + n_gen )) || \
    fail "invalid driver shape"

mkdir -p "$result_dir"
phone_stage_log=$phone_log_root/$label-stage.log
phone_bridge_log=$phone_log_root/$label-bridge.log
stage_pid=
phone_bridge_pid=
host_bridge_pid=
accessory_active=0

adb_control() {
    adb -s "$phone_control" "$@"
}

wait_for_control() {
    for _attempt in $(seq 1 100); do
        if [[ "$(adb_control get-state 2>/dev/null || true)" == device ]]; then
            return 0
        fi
        adb connect "$phone_control" >/dev/null 2>&1 || true
        sleep 0.1
    done
    return 1
}

reset_accessory() {
    if lsusb | grep -Eq '18d1:2d0(0|1|4|5)'; then
        python3 "$aoa_switch" reset >/dev/null 2>&1 || true
    fi
    for _attempt in $(seq 1 120); do
        if lsusb | grep -Eq '(22d9:2769|22d9:2772|05c6:908c)'; then
            wait_for_control
            return $?
        fi
        sleep 0.1
    done
    return 1
}

phone_cmdline() {
    local pid=$1
    adb_control shell "tr '\\000' ' ' < /proc/$pid/cmdline" 2>/dev/null |
        tr -d '\r' || true
}

kill_phone_pid() {
    local pid=$1
    local pattern=$2
    [[ "$pid" =~ ^[0-9]+$ ]] || return 0
    local command
    command=$(phone_cmdline "$pid")
    if [[ "$command" == *"$pattern"* ]]; then
        adb_control shell "su -c 'kill $pid'" >/dev/null 2>&1 || true
    fi
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "$host_bridge_pid" =~ ^[0-9]+$ ]] && \
            kill -0 "$host_bridge_pid" 2>/dev/null; then
        kill "$host_bridge_pid" >/dev/null 2>&1 || true
        wait "$host_bridge_pid" 2>/dev/null || true
    fi
    reset_accessory || true
    wait_for_control || true
    kill_phone_pid "$phone_bridge_pid" aoa_phone_bridge
    kill_phone_pid "$stage_pid" llama-layersplit
    exit "$status"
}
trap cleanup EXIT INT TERM

snapshot_phone() {
    local name=$1
    adb_control shell \
        "grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo; grep -E '^(pswpin|pswpout) ' /proc/vmstat; cat /proc/sys/kernel/random/boot_id" \
        > "$result_dir/$name-phone.txt"
}

snapshot() {
    local name=$1
    {
        date --iso-8601=seconds
        hostname
        uname -a
        grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
        awk '$1 == "pswpin" || $1 == "pswpout" {print}' /proc/vmstat
        nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw \
            --format=csv,noheader,nounits
    } > "$result_dir/$name-host.txt"
    snapshot_phone "$name"
}

write_command() {
    local path=$1
    shift
    printf '%q ' "$@" > "$path"
    printf '\n' >> "$path"
}

wait_for_control || fail "OP15 control ADB is offline"
if lsusb | grep -Eq '18d1:2d0(0|1|4|5)'; then
    reset_accessory || fail "failed to restore normal USB mode"
fi
active=$(adb_control shell pidof llama-layersplit 2>/dev/null | tr -d '\r' || true)
[[ -z "$active" ]] || fail "OP15 already has llama-layersplit PID $active"
active=$(adb_control shell pidof aoa_phone_bridge 2>/dev/null | tr -d '\r' || true)
[[ -z "$active" ]] || fail "OP15 already has aoa_phone_bridge PID $active"
adb_control shell test -f "$phone_model" || fail "missing phone model"
adb_control shell test -x "$phone_bridge" || fail "missing phone AOA bridge"
adb_control shell "su -c 'mkdir -p $phone_log_root; rm -f $phone_stage_log $phone_bridge_log'"

snapshot before

stage_command="cd $phone_dir; setsid -d env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=$phone_mbuf LLAMA_LAYER_START=0 LLAMA_LAYER_END=$cut LAYERSPLIT_PLACEMENT_CERT=1 ./llama-layersplit -m $phone_model --devices $phone_backend -ngl 99 --mode stagenet --port $phone_stage_port -n $n_gen --driver-context $context --driver-max-prefill $max_prefill"
printf '%s\n' "$stage_command" > "$result_dir/phone.command.txt"
adb_control shell \
    "su -c '$stage_command >$phone_stage_log 2>&1 </dev/null &'" >/dev/null
for _attempt in $(seq 1 100); do
    stage_pid=$(adb_control shell pidof llama-layersplit 2>/dev/null |
        tr -d '\r' || true)
    [[ "$stage_pid" =~ ^[0-9]+$ ]] && break
    sleep 0.1
done
[[ "$stage_pid" =~ ^[0-9]+$ ]] || fail "phone stage PID was not reported"

for _attempt in $(seq 1 1800); do
    if adb_control shell cat "$phone_stage_log" 2>/dev/null |
            grep -q '\[stagenet\] listening'; then
        break
    fi
    command=$(phone_cmdline "$stage_pid")
    [[ "$command" == *llama-layersplit* ]] || {
        adb_control shell tail -80 "$phone_stage_log" >&2 || true
        fail "phone stage exited during load"
    }
    sleep 0.1
done
adb_control shell cat "$phone_stage_log" 2>/dev/null |
    grep -q '\[stagenet\] listening' || fail "phone stage load timed out"

bridge_command="setsid -d $phone_bridge $phone_stage_port"
printf '%s\n' "$bridge_command" > "$result_dir/phone-bridge.command.txt"
adb_control shell \
    "su -c '$bridge_command >$phone_bridge_log 2>&1 </dev/null &'" >/dev/null
for _attempt in $(seq 1 100); do
    phone_bridge_pid=$(adb_control shell pidof aoa_phone_bridge 2>/dev/null |
        tr -d '\r' || true)
    [[ "$phone_bridge_pid" =~ ^[0-9]+$ ]] && break
    sleep 0.1
done
[[ "$phone_bridge_pid" =~ ^[0-9]+$ ]] || fail "phone bridge PID was not reported"
for _attempt in $(seq 1 100); do
    if adb_control shell cat "$phone_bridge_log" 2>/dev/null |
            grep -q '\[aoa-phone-bridge\] stage connected'; then
        break
    fi
    sleep 0.1
done
adb_control shell cat "$phone_bridge_log" 2>/dev/null |
    grep -q '\[aoa-phone-bridge\] stage connected' || \
    fail "phone AOA bridge did not connect to the stage"

switched=0
for normal_device in 22d9:2772 22d9:2769 05c6:908c; do
    normal_vid=${normal_device%:*}
    normal_pid=${normal_device#*:}
    if python3 "$aoa_switch" --vid "$normal_vid" --pid "$normal_pid" \
            switch > "$result_dir/aoa-switch.txt" 2>&1; then
        switched=1
        break
    fi
done
[[ "$switched" == 1 ]] || fail "failed to enter AOA mode"
for _attempt in $(seq 1 100); do
    if lsusb | grep -Eq '18d1:2d0(0|1|4|5)'; then
        accessory_active=1
        break
    fi
    sleep 0.1
done
[[ "$accessory_active" == 1 ]] || fail "AOA device did not enumerate"

"$host_bridge" "$host_bridge_port" > "$result_dir/host-bridge.log" 2>&1 &
host_bridge_pid=$!
for _attempt in $(seq 1 100); do
    if grep -q '\[aoa-host-bridge\] ready' "$result_dir/host-bridge.log"; then
        break
    fi
    kill -0 "$host_bridge_pid" 2>/dev/null || {
        cat "$result_dir/host-bridge.log" >&2
        fail "host AOA bridge exited during setup"
    }
    sleep 0.1
done
grep -q '\[aoa-host-bridge\] ready' "$result_dir/host-bridge.log" || \
    fail "host AOA bridge did not become ready"

snapshot_phone paid-before
command=(env LLAMA_LAYER_START="$cut" LAYERSPLIT_PLACEMENT_CERT=1
    "$host_binary" -m "$host_model" -ngl 0 -t "$threads" -tb "$threads"
    --chat -p "$prompt" -n "$n_gen"
    --driver-context "$context" --driver-max-prefill "$max_prefill"
    --driver-warmup "$warmups" --driver-requests "$requests"
    --mode pipedriver --host 127.0.0.1 --port "$host_bridge_port")
write_command "$result_dir/host.command.txt" "${command[@]}"
/usr/bin/time -v "${command[@]}" \
    > "$result_dir/host.stdout" 2> "$result_dir/host.stderr"
snapshot_phone paid-after

wait "$host_bridge_pid"
host_bridge_pid=
reset_accessory || fail "failed to leave AOA mode"
accessory_active=0
for _attempt in $(seq 1 100); do
    if [[ -z "$(phone_cmdline "$phone_bridge_pid")" ]]; then
        break
    fi
    sleep 0.1
done
adb_control shell cat "$phone_stage_log" > "$result_dir/phone.log"
adb_control shell cat "$phone_bridge_log" > "$result_dir/phone-bridge.log"
stage_pid=
phone_bridge_pid=

snapshot after
{
    printf 'transport=AOA bulk via transparent TCP bridge\n'
    printf 'usb_device='
    lsusb | grep -E '(22d9:2769|22d9:2772|05c6:908c)' || true
    printf 'usb_tree_begin\n'
    lsusb -t
    printf 'usb_tree_end\n'
    sha256sum "$host_bridge" "$aoa_switch"
    adb_control shell sha256sum "$phone_bridge" | tr -d '\r'
} > "$result_dir/environment.txt"
(
    cd "$result_dir"
    sha256sum -- * > SHA256SUMS.txt
)
trap - EXIT INT TERM
printf '%s\n' "$result_dir"
