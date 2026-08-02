#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
adb_port=${ADB_PORT:-5038}
op15=${OP15_SERIAL:-3C15AU002CL00000}
op12=${OP12_SERIAL:-5ae7a43d}
op15_physical=${OP15_PHYSICAL_SERIAL:-3C15AU002CL00000}
op12_physical=${OP12_PHYSICAL_SERIAL:-5ae7a43d}
op15_wifi=${OP15_WIFI:-172.20.173.218}
op12_wifi=${OP12_WIFI:-172.20.59.72}
runtime=/data/local/tmp/s39-qwen2-port-v1
model_dir=/data/local/tmp/s39-active-warm/v2/models/qwen2.5-14b-q4_0
model_sha=924a4c39ef9fc6c139875ab6771c2e8172a3b40ffec5c720eca69ad7a0edfae7
op15_shard_sha=4eb56f404eb8e4e64b63ba0870db3a41016486cc137ce7d65d51969902970a16
op12_shard_sha=f741cd302150adb0be9349d9ae257e77f4e7e18be8414e98797b32445ed42737
full_model=${QWEN25_MODEL:-/home/myid/zs89458/Documents/models/Qwen2.5-14B-Instruct-Q4_0.gguf}
op15_shard=${QWEN25_OP15_SHARD:-/home/myid/zs89458/Documents/s39_shards_qwen2/Qwen2.5-14B-Instruct-Q4_0.layers-0-32.gguf}
op12_shard=${QWEN25_OP12_SHARD:-/home/myid/zs89458/Documents/s39_shards_qwen2/Qwen2.5-14B-Instruct-Q4_0.layers-32-48.gguf}
host_binary=${LAYERSPLIT_HOST_BINARY:-$root/build-cuda/bin/llama-layersplit}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out=${1:-$here/results/w4_qwen25_route/run_$stamp}

head_port=39715
tail_port=39732
relay_port=39725
started=0

remote_pid() {
    adb -P "$adb_port" -s "$1" shell "cat '$runtime/$2' 2>/dev/null" |
        tr -d '\r'
}

stop_remote_pid() {
    local serial=$1
    local pid_file=$2
    local pid
    pid=$(remote_pid "$serial" "$pid_file" || true)
    if [[ $pid =~ ^[0-9]+$ ]]; then
        adb -P "$adb_port" -s "$serial" shell "
            if [ -r /proc/$pid/cmdline ] &&
               tr '\\000' ' ' < /proc/$pid/cmdline |
                   grep -qE 'llama-layersplit|llama-stage-direct-relay'; then
                kill '$pid' 2>/dev/null || true
            fi
        " >/dev/null 2>&1 || true
    fi
}

collect_logs() {
    mkdir -p "$out"
    adb -P "$adb_port" -s "$op15" pull \
        "$runtime/w4_qwen25_head.log" "$out/op15_head.log" \
        >/dev/null 2>&1 || true
    adb -P "$adb_port" -s "$op15" pull \
        "$runtime/w4_qwen25_relay.log" "$out/op15_relay.log" \
        >/dev/null 2>&1 || true
    adb -P "$adb_port" -s "$op12" pull \
        "$runtime/w4_qwen25_tail.log" "$out/op12_tail.log" \
        >/dev/null 2>&1 || true
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if (( started )); then
        collect_logs
        stop_remote_pid "$op15" w4_qwen25_relay.pid
        stop_remote_pid "$op15" w4_qwen25_head.pid
        stop_remote_pid "$op12" w4_qwen25_tail.pid
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

require_hash() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "$path" | awk '{print $1}')
    if [[ $actual != "$expected" ]]; then
        printf 'hash mismatch: %s\n' "$path" >&2
        exit 2
    fi
}

require_device() {
    local serial=$1
    if ! adb -P "$adb_port" devices |
        awk 'NR > 1 && $2 == "device" {print $1}' |
        grep -Fxq "$serial"; then
        printf 'device is not ready: %s\n' "$serial" >&2
        exit 2
    fi
}

require_ports_free() {
    local serial=$1
    shift
    local live
    live=$(adb -P "$adb_port" -s "$serial" shell \
        'for p in $(pidof llama-layersplit llama-stage-direct-relay 2>/dev/null); do
             tr "\000" " " < /proc/$p/cmdline
             echo
         done' |
        tr -d '\r')
    local port
    for port in "$@"; do
        if grep -Eq -- "(--port|--listen)[[:space:]]+$port([[:space:]]|$)" \
            <<<"$live"; then
            printf 'device %s already uses route port %s\n' "$serial" "$port" >&2
            exit 2
        fi
    done
}

remote_hash() {
    adb -P "$adb_port" -s "$1" shell "sha256sum '$2'" |
        awk '{print $1}' |
        tr -d '\r'
}

wait_log() {
    local serial=$1
    local log=$2
    local marker=$3
    local pid_file=$4
    local pid
    for _ in $(seq 1 180); do
        if adb -P "$adb_port" -s "$serial" shell \
            "grep -Fq '$marker' '$runtime/$log' 2>/dev/null"; then
            return 0
        fi
        pid=$(remote_pid "$serial" "$pid_file" || true)
        if [[ ! $pid =~ ^[0-9]+$ ]] ||
           ! adb -P "$adb_port" -s "$serial" shell "kill -0 '$pid'" \
               >/dev/null 2>&1; then
            printf 'worker exited before readiness: %s/%s\n' "$serial" "$log" >&2
            adb -P "$adb_port" -s "$serial" shell \
                "tail -80 '$runtime/$log'" >&2 || true
            return 1
        fi
        sleep 1
    done
    printf 'worker readiness timed out: %s/%s\n' "$serial" "$log" >&2
    return 1
}

mkdir -p "$out"
require_device "$op15"
require_device "$op12"
require_ports_free "$op15" "$head_port" "$relay_port"
require_ports_free "$op12" "$tail_port"
require_hash "$full_model" "$model_sha"
require_hash "$op15_shard" "$op15_shard_sha"
require_hash "$op12_shard" "$op12_shard_sha"

op15_device_hash=$(remote_hash "$op15" "$model_dir/weights.gguf")
op12_device_hash=$(remote_hash "$op12" "$model_dir/weights.gguf")
[[ $op15_device_hash == "$op15_shard_sha" ]]
[[ $op12_device_hash == "$op12_shard_sha" ]]

worker_sha=$(sha256sum \
    "$root/npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin/llama-layersplit" |
    awk '{print $1}')
relay_sha=$(sha256sum \
    "$root/npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin/llama-stage-direct-relay" |
    awk '{print $1}')
[[ $(remote_hash "$op15" "$runtime/llama-layersplit") == "$worker_sha" ]]
[[ $(remote_hash "$op12" "$runtime/llama-layersplit") == "$worker_sha" ]]
[[ $(remote_hash "$op15" "$runtime/llama-stage-direct-relay") == "$relay_sha" ]]

op15_boot=$(adb -P "$adb_port" -s "$op15" shell \
    'cat /proc/sys/kernel/random/boot_id' | tr -d '\r')
op12_boot=$(adb -P "$adb_port" -s "$op12" shell \
    'cat /proc/sys/kernel/random/boot_id' | tr -d '\r')
git_commit=$(git -C "$root" rev-parse HEAD)
acquisition=$(date -u +%Y-%m-%dT%H:%M:%SZ)

printf '%s\n' \
    "{\"acquisition_utc\":\"$acquisition\",\"activation_path\":\"OP15_TO_OP12_WIFI_TCP\",\"backend\":\"GPUOpenCL\",\"base_git_commit\":\"$git_commit\",\"driver_batch\":32,\"driver_context\":64,\"model_sha256\":\"$model_sha\",\"op12_adb_target\":\"$op12\",\"op12_boot_id\":\"$op12_boot\",\"op12_layers\":[32,48],\"op12_serial\":\"$op12_physical\",\"op12_shard_sha256\":\"$op12_shard_sha\",\"op12_wifi\":\"$op12_wifi\",\"op15_adb_target\":\"$op15\",\"op15_boot_id\":\"$op15_boot\",\"op15_layers\":[0,32],\"op15_serial\":\"$op15_physical\",\"op15_shard_sha256\":\"$op15_shard_sha\",\"op15_wifi\":\"$op15_wifi\",\"relay_sha256\":\"$relay_sha\",\"schema\":\"s39-qwen25-route-context-v1\",\"weight_path\":\"USB_ADB_BEFORE_SERVICE\",\"worker_sha256\":\"$worker_sha\"}" \
    > "$out/RUN_CONTEXT.json"

env CUDA_VISIBLE_DEVICES=0 LAYERSPLIT_PLACEMENT_CERT=1 \
    "$host_binary" \
    -m "$full_model" \
    --mode monodriver \
    -p France \
    -n 8 \
    --driver-requests 32 \
    --driver-batch 32 \
    --driver-context 64 \
    --driver-max-prefill 1 \
    --devices CUDA0 \
    -ngl 99 \
    > "$out/cuda_b32.log" 2>&1

started=1
adb -P "$adb_port" -s "$op12" shell "
    cd '$runtime'
    rm -f w4_qwen25_tail.log w4_qwen25_tail.pid
    nohup env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
        LLAMA_LAYER_START=32 LLAMA_LAYER_END=48 \
        LAYERSPLIT_MODEL_SHA256='$model_sha' \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        ./llama-layersplit \
        -m '$model_dir/weights.gguf' \
        --mode tailv3 --port '$tail_port' \
        --driver-batch 32 --driver-context 64 --driver-max-prefill 1 \
        --devices GPUOpenCL -ngl 99 \
        >'$runtime/w4_qwen25_tail.log' 2>&1 </dev/null &
    echo \$! >'$runtime/w4_qwen25_tail.pid'
"
wait_log "$op12" w4_qwen25_tail.log \
    "[stagenet] listening on 0.0.0.0:$tail_port" w4_qwen25_tail.pid

adb -P "$adb_port" -s "$op15" shell "
    cd '$runtime'
    rm -f w4_qwen25_head.log w4_qwen25_head.pid
    nohup env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
        LLAMA_LAYER_START=0 LLAMA_LAYER_END=32 \
        LAYERSPLIT_MODEL_SHA256='$model_sha' \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        ./llama-layersplit \
        -m '$model_dir/weights.gguf' \
        --mode stagenet --port '$head_port' \
        --driver-batch 32 --driver-context 64 --driver-max-prefill 1 \
        --devices GPUOpenCL -ngl 99 \
        >'$runtime/w4_qwen25_head.log' 2>&1 </dev/null &
    echo \$! >'$runtime/w4_qwen25_head.pid'
"
wait_log "$op15" w4_qwen25_head.log \
    "[stagenet] listening on 0.0.0.0:$head_port" w4_qwen25_head.pid

adb -P "$adb_port" -s "$op15" shell "
    cd '$runtime'
    rm -f w4_qwen25_relay.log w4_qwen25_relay.pid
    nohup ./llama-stage-direct-relay \
        --listen '$relay_port' \
        --head '127.0.0.1:$head_port' \
        --tail '$op12_wifi:$tail_port' \
        >'$runtime/w4_qwen25_relay.log' 2>&1 </dev/null &
    echo \$! >'$runtime/w4_qwen25_relay.pid'
"
wait_log "$op15" w4_qwen25_relay.log \
    "[direct-relay] listening on 0.0.0.0:$relay_port" w4_qwen25_relay.pid

python3 "$here/direct_order_probe.py" \
    --route-id s39-qwen25-cut32-b32 \
    --relay "$op15_wifi:$relay_port" \
    --order sorted \
    --requests 32 \
    --steps 8 \
    --batch-knee 32 \
    --gather-us 50000 \
    --queue-depth 64 \
    --slo-ms 300000 \
    --timeout 300 \
    --prompt-tokens 49000 \
    --expected-tokens 25,576,8585,3033,702,7228,6649,311 \
    --session-end stop \
    --output "$out/b32.json" \
    > "$out/probe.stdout"

collect_logs
python3 "$here/validate_qwen25_route.py" "$out" \
    --output "$out/route_certificate.json" \
    > "$out/validator.stdout"

(
    cd "$out"
    find . -maxdepth 1 -type f ! -name SHA256SUMS.txt -printf '%P\0' |
        sort -z |
        xargs -0 sha256sum
) > "$out/SHA256SUMS.txt"

started=0
printf 'QWEN25_ROUTE_GATE=%s\n' "$out"
