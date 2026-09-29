#!/usr/bin/env bash
set -euo pipefail

result_dir=${1:-}
mode=${2:-}
label=${3:-}
cut=${4:-}
phone_backend=${5:-}
phone_model=${6:-}

host_binary=${S41_HOST_BINARY:-/home/zhihao/llama.cpp-release/build-fullmodel-cpu/bin/llama-layersplit}
host_model=${S41_HOST_MODEL:-/home/zhihao/models/gemma-4-12B-it-Q8_0-7b56.gguf}
phone_serial=${S41_PHONE_SERIAL:-3C15AU002CL00000}
phone_dir=${S41_PHONE_DIR:-/data/local/tmp/ls-s34-identity}
phone_port=${S41_PHONE_PORT:-24381}
phone_mbuf=${S41_PHONE_MBUF:-4192}
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

[[ -n "$result_dir" ]] || fail "usage: $0 RESULT_DIR {cpu|phone} [LABEL CUT BACKEND PHONE_MODEL]"
[[ "$result_dir" == /* ]] || fail "RESULT_DIR must be absolute"
[[ ! -e "$result_dir" ]] || fail "RESULT_DIR already exists"
[[ "$mode" == cpu || "$mode" == phone ]] || fail "mode must be cpu or phone"
[[ -x "$host_binary" ]] || fail "missing host binary: $host_binary"
[[ -f "$host_model" ]] || fail "missing host model: $host_model"
[[ "$(hostname)" == zhihao-Z690-C-ac ]] || fail "wrong physical host"
for value in "$n_gen" "$requests" "$warmups" "$context" "$max_prefill" "$threads"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fail "invalid numeric configuration"
done
(( n_gen > 0 && requests > 0 && context >= max_prefill + n_gen )) || \
    fail "invalid driver shape"

if [[ "$mode" == phone ]]; then
    [[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid phone case label"
    [[ "$cut" =~ ^[1-9][0-9]*$ ]] || fail "invalid layer cut"
    (( cut < 48 )) || fail "layer cut must be below 48"
    [[ "$phone_backend" == HTP0 || "$phone_backend" == GPUOpenCL ]] || \
        fail "phone backend must be HTP0 or GPUOpenCL"
    [[ "$phone_model" == /* && "$phone_model" != *" "* ]] || \
        fail "phone model must be an absolute path without spaces"
fi

mkdir -p "$result_dir"

snapshot_phone() {
    local name=$1
    adb -s "$phone_serial" shell \
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
    if [[ "$mode" == phone ]]; then
        snapshot_phone "$name"
    fi
}

write_command() {
    local path=$1
    shift
    printf '%q ' "$@" > "$path"
    printf '\n' >> "$path"
}

common_args=(
    -m "$host_model"
    -ngl 0
    -t "$threads"
    -tb "$threads"
    --chat
    -p "$prompt"
    -n "$n_gen"
    --driver-context "$context"
    --driver-max-prefill "$max_prefill"
    --driver-warmup "$warmups"
    --driver-requests "$requests"
)

snapshot before

if [[ "$mode" == cpu ]]; then
    command=(env LAYERSPLIT_PLACEMENT_CERT=1 "$host_binary" "${common_args[@]}" --mode monodriver)
    write_command "$result_dir/host.command.txt" "${command[@]}"
    /usr/bin/time -v "${command[@]}" \
        > "$result_dir/host.stdout" 2> "$result_dir/host.stderr"
else
    adb -s "$phone_serial" get-state | grep -qx device || fail "OP15 is not online"
    adb -s "$phone_serial" shell test -f "$phone_model" || fail "missing phone model"
    active=$(adb -s "$phone_serial" shell pidof llama-layersplit 2>/dev/null | tr -d '\r' || true)
    [[ -z "$active" ]] || fail "OP15 already has llama-layersplit PID $active"

    adb -s "$phone_serial" forward --remove "tcp:$phone_port" >/dev/null 2>&1 || true
    adb -s "$phone_serial" forward "tcp:$phone_port" "tcp:$phone_port" >/dev/null

    phone_command="cd $phone_dir && echo S41_PHONE_PID=\$\$ >&2 && exec env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=$phone_mbuf LLAMA_LAYER_START=0 LLAMA_LAYER_END=$cut LAYERSPLIT_PLACEMENT_CERT=1 ./llama-layersplit -m $phone_model --devices $phone_backend -ngl 99 --mode stagenet --port $phone_port -n $n_gen --driver-context $context --driver-max-prefill $max_prefill"
    printf '%s\n' "$phone_command" > "$result_dir/phone.command.txt"
    nohup adb -s "$phone_serial" shell "$phone_command" \
        > "$result_dir/phone.log" 2>&1 < /dev/null &
    controller_pid=$!
    phone_pid=

    cleanup() {
        local status=$?
        trap - EXIT INT TERM
        if [[ -n "$phone_pid" ]]; then
            remote_cmd=$(adb -s "$phone_serial" shell \
                "tr '\\000' ' ' < /proc/$phone_pid/cmdline" 2>/dev/null | tr -d '\r' || true)
            if [[ "$remote_cmd" == *llama-layersplit* ]]; then
                adb -s "$phone_serial" shell kill "$phone_pid" >/dev/null 2>&1 || true
            fi
        fi
        if kill -0 "$controller_pid" 2>/dev/null; then
            kill "$controller_pid" >/dev/null 2>&1 || true
            wait "$controller_pid" 2>/dev/null || true
        fi
        adb -s "$phone_serial" forward --remove "tcp:$phone_port" >/dev/null 2>&1 || true
        exit "$status"
    }
    trap cleanup EXIT INT TERM

    for _attempt in $(seq 1 1800); do
        if grep -q '\[stagenet\] listening' "$result_dir/phone.log"; then
            phone_pid=$(sed -n 's/.*S41_PHONE_PID=\([0-9][0-9]*\).*/\1/p' \
                "$result_dir/phone.log" | head -n 1)
            [[ "$phone_pid" =~ ^[0-9]+$ ]] || fail "phone PID was not reported"
            break
        fi
        kill -0 "$controller_pid" 2>/dev/null || {
            tail -80 "$result_dir/phone.log" >&2
            fail "phone worker exited during load"
        }
        sleep 0.1
    done
    [[ -n "$phone_pid" ]] || fail "phone worker load timed out"
    snapshot_phone paid-before

    command=(env LLAMA_LAYER_START="$cut" LAYERSPLIT_PLACEMENT_CERT=1
        "$host_binary" "${common_args[@]}" --mode pipedriver
        --host 127.0.0.1 --port "$phone_port")
    write_command "$result_dir/host.command.txt" "${command[@]}"
    /usr/bin/time -v "${command[@]}" \
        > "$result_dir/host.stdout" 2> "$result_dir/host.stderr"
    snapshot_phone paid-after

    wait "$controller_pid"
    controller_pid=0
    phone_pid=
    adb -s "$phone_serial" forward --remove "tcp:$phone_port" >/dev/null
    trap - EXIT INT TERM
fi

snapshot after
(
    cd "$result_dir"
    sha256sum -- * > SHA256SUMS.txt
)
printf '%s\n' "$result_dir"
