#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
adb_port=${ADB_PORT:-5038}
op15=${OP15_SERIAL:-3C15AU002CL00000}
op12=${OP12_SERIAL:-5ae7a43d}
op15_wifi=${OP15_WIFI:-172.20.173.218}
op12_wifi=${OP12_WIFI:-172.20.59.72}
runtime=/data/local/tmp/s39-qwen25-q8-v1
full_model=${QWEN25_Q8_MODEL:-/home/myid/zs89458/Documents/models/Qwen2.5-14B-Instruct-Q8_0.gguf}
host_binary=${LAYERSPLIT_HOST_BINARY:-$root/build-cuda/bin/llama-layersplit}
host_relay=${LAYERSPLIT_HOST_RELAY:-$root/build-cuda/bin/llama-stage-direct-relay}
contract=${DELTA_CONTRACT:-$here/W6_DELTA_CONTRACT.json}
base_contract=${HANDOFF_CONTRACT:-$here/W5_HANDOFF_CONTRACT.json}
physical_gate=${PHYSICAL_GATE:-$here/W6_PHYSICAL_GATE.json}
model_sha=23ca481b8226b2492ba8f3eb7af41e0f99d8605c16fb6dec7bc5cf6716b673cf
op15_shard_sha=b9611440eb4901764cef6afb55e08418acb114f1dfdc5b97e2c4acafefcc5375
op12_shard_sha=b66f9f6ace28da341f21f4f0d03ffa05c31cea647d43da4023e373f6021551ac
device_worker_sha=45ff9eda965d9e1688776f779895388b82f8e7d6933fe453019691973997db3f
device_relay_sha=1c809cb50cae6aa86869d61068a05173c719e4542c851e478ee1e033c5456929
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out=${1:-$here/results/w6_phone_cuda_delta/run_$stamp}

phone_head_port=41415
phone_tail_port=41432
phone_relay_port=41425
cuda_head_port=41410
cuda_tail_port=41411
cuda_relay_port=41412

host_pids=()
phone_started=0

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

remote_pid() {
    adb -P "$adb_port" -s "$1" shell \
        "cat '$runtime/$2' 2>/dev/null" | tr -d '\r'
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

collect_phone_logs() {
    mkdir -p "$out"
    adb -P "$adb_port" -s "$op15" pull \
        "$runtime/w6_delta_head.log" "$out/op15_head.log" \
        >/dev/null 2>&1 || true
    adb -P "$adb_port" -s "$op15" pull \
        "$runtime/w6_delta_relay.log" "$out/phone_relay.log" \
        >/dev/null 2>&1 || true
    adb -P "$adb_port" -s "$op12" pull \
        "$runtime/w6_delta_tail.log" "$out/op12_tail.log" \
        >/dev/null 2>&1 || true
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if (( phone_started )); then
        collect_phone_logs
        stop_remote_pid "$op15" w6_delta_relay.pid
        stop_remote_pid "$op15" w6_delta_head.pid
        stop_remote_pid "$op12" w6_delta_tail.pid
    fi
    local pid
    for pid in "${host_pids[@]}"; do
        if [[ $pid =~ ^[0-9]+$ ]]; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
    wait "${host_pids[@]}" >/dev/null 2>&1 || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

require_hash() {
    local path=$1
    local expected=$2
    [[ -f $path ]] || fail "missing artifact: $path"
    local actual
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ $actual == "$expected" ]] || fail "hash mismatch: $path"
}

remote_hash() {
    adb -P "$adb_port" -s "$1" shell "sha256sum '$2'" |
        awk '{print $1}' | tr -d '\r'
}

require_device() {
    local serial=$1
    adb -P "$adb_port" devices |
        awk 'NR > 1 && $2 == "device" {print $1}' |
        grep -Fxq "$serial" ||
        fail "device is not ready: $serial"
}

require_remote_hash() {
    local serial=$1
    local path=$2
    local expected=$3
    local actual
    actual=$(remote_hash "$serial" "$path")
    [[ $actual == "$expected" ]] ||
        fail "remote hash mismatch: $serial:$path"
}

require_phone_ports_free() {
    local serial=$1
    shift
    local live
    live=$(adb -P "$adb_port" -s "$serial" shell \
        'for p in $(pidof llama-layersplit llama-stage-direct-relay 2>/dev/null); do
             tr "\000" " " < /proc/$p/cmdline
             echo
         done' | tr -d '\r')
    local port
    for port in "$@"; do
        if grep -Eq -- "(--port|--listen)[[:space:]]+$port([[:space:]]|$)" \
            <<<"$live"; then
            fail "device $serial already uses route port $port"
        fi
    done
}

require_host_port_free() {
    local port=$1
    if ss -H -ltn "sport = :$port" | grep -q .; then
        fail "host route port is already in use: $port"
    fi
}

wait_phone_log() {
    local serial=$1
    local log=$2
    local marker=$3
    local pid_file=$4
    local pid
    for _ in $(seq 1 600); do
        if adb -P "$adb_port" -s "$serial" shell \
            "grep -Fq '$marker' '$runtime/$log' 2>/dev/null"; then
            return 0
        fi
        pid=$(remote_pid "$serial" "$pid_file" || true)
        if [[ ! $pid =~ ^[0-9]+$ ]] ||
           ! adb -P "$adb_port" -s "$serial" shell "kill -0 '$pid'" \
               >/dev/null 2>&1; then
            adb -P "$adb_port" -s "$serial" shell \
                "tail -100 '$runtime/$log'" >&2 || true
            fail "phone worker exited before readiness: $serial/$log"
        fi
        sleep 1
    done
    fail "phone worker readiness timed out: $serial/$log"
}

wait_host_log() {
    local log=$1
    local marker=$2
    local pid=$3
    for _ in $(seq 1 600); do
        if grep -Fq "$marker" "$log" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            tail -100 "$log" >&2 || true
            fail "host worker exited before readiness: $log"
        fi
        sleep 1
    done
    fail "host worker readiness timed out: $log"
}

wait_host_exit() {
    local pid=$1
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            wait "$pid" || true
            return 0
        fi
        sleep 1
    done
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
    return 1
}

wait_phone_exit() {
    local serial=$1
    local pid_file=$2
    local pid
    pid=$(remote_pid "$serial" "$pid_file" || true)
    if [[ ! $pid =~ ^[0-9]+$ ]]; then
        return 0
    fi
    for _ in $(seq 1 60); do
        if ! adb -P "$adb_port" -s "$serial" shell "kill -0 '$pid'" \
            >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    stop_remote_pid "$serial" "$pid_file"
    return 1
}

[[ ! -e $out ]] || fail "output already exists: $out"
mkdir -p "$out"
require_device "$op15"
require_device "$op12"
require_hash "$full_model" "$model_sha"
[[ -f $contract ]] || fail "missing delta contract: $contract"
[[ -f $base_contract ]] || fail "missing base contract: $base_contract"
[[ -f $physical_gate ]] || fail "missing physical gate: $physical_gate"
[[ -x $host_binary ]] || fail "missing host worker: $host_binary"
[[ -x $host_relay ]] || fail "missing host relay: $host_relay"
require_remote_hash "$op15" "$runtime/llama-layersplit" "$device_worker_sha"
require_remote_hash "$op12" "$runtime/llama-layersplit" "$device_worker_sha"
require_remote_hash \
    "$op15" "$runtime/llama-stage-direct-relay" "$device_relay_sha"
require_remote_hash "$op15" "$runtime/weights.gguf" "$op15_shard_sha"
require_remote_hash "$op12" "$runtime/weights.gguf" "$op12_shard_sha"
require_phone_ports_free "$op15" "$phone_head_port" "$phone_relay_port"
require_phone_ports_free "$op12" "$phone_tail_port"
require_host_port_free "$cuda_head_port"
require_host_port_free "$cuda_tail_port"
require_host_port_free "$cuda_relay_port"

op15_boot=$(adb -P "$adb_port" -s "$op15" shell \
    'cat /proc/sys/kernel/random/boot_id' | tr -d '\r')
op12_boot=$(adb -P "$adb_port" -s "$op12" shell \
    'cat /proc/sys/kernel/random/boot_id' | tr -d '\r')
host_boot=$(cat /proc/sys/kernel/random/boot_id)
host_worker_sha=$(sha256sum "$host_binary" | awk '{print $1}')
host_relay_sha=$(sha256sum "$host_relay" | awk '{print $1}')
contract_sha=$(sha256sum "$contract" | awk '{print $1}')
base_contract_sha=$(sha256sum "$base_contract" | awk '{print $1}')
physical_gate_sha=$(sha256sum "$physical_gate" | awk '{print $1}')
launcher_sha=$(sha256sum "$0" | awk '{print $1}')
probe_sha=$(sha256sum "$here/phone_cuda_delta_probe.py" | awk '{print $1}')
handoff_probe_sha=$(sha256sum \
    "$here/phone_cuda_handoff_probe.py" | awk '{print $1}')
validator_sha=$(sha256sum "$here/validate_phone_cuda_delta.py" | awk '{print $1}')
quality_probe_sha=$(sha256sum "$here/qwen25_quality_probe.py" | awk '{print $1}')
stage_client_sha=$(sha256sum \
    "$here/../s22_slo_overlap_pipeline/stage_v3_client.py" | awk '{print $1}')
endpoint_parser_sha=$(sha256sum \
    "$here/../s22_slo_overlap_pipeline/async_pipeline.py" | awk '{print $1}')
git_commit=$(git -C "$root" rev-parse HEAD)
run_id=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

python3 - "$out/RUN_CONTEXT.json" <<EOF
import json
import pathlib
import time

value = {
    "acquisition_unix_s": int(time.time()),
    "base_contract_sha256": "$base_contract_sha",
    "base_git_commit": "$git_commit",
    "contract_sha256": "$contract_sha",
    "cuda": {
        "boot_id": "$host_boot",
        "device": "CUDA0",
        "head_port": $cuda_head_port,
        "relay_port": $cuda_relay_port,
        "relay_sha256": "$host_relay_sha",
        "tail_port": $cuda_tail_port,
        "worker_sha256": "$host_worker_sha",
    },
    "model_sha256": "$model_sha",
    "op12": {
        "adb_target": "$op12",
        "boot_id": "$op12_boot",
        "layers": [30, 48],
        "shard_sha256": "$op12_shard_sha",
        "wifi": "$op12_wifi",
        "worker_sha256": "$device_worker_sha",
    },
    "op15": {
        "adb_target": "$op15",
        "boot_id": "$op15_boot",
        "layers": [0, 30],
        "relay_sha256": "$device_relay_sha",
        "shard_sha256": "$op15_shard_sha",
        "wifi": "$op15_wifi",
        "worker_sha256": "$device_worker_sha",
    },
    "physical_gate_sha256": "$physical_gate_sha",
    "run_id": "$run_id",
    "schema": "s39-phone-cuda-delta-context-v1",
    "sources": {
        "async_pipeline.py": "$endpoint_parser_sha",
        "phone_cuda_delta_probe.py": "$probe_sha",
        "phone_cuda_handoff_probe.py": "$handoff_probe_sha",
        "qwen25_quality_probe.py": "$quality_probe_sha",
        "run_w6_delta_gate.sh": "$launcher_sha",
        "stage_v3_client.py": "$stage_client_sha",
        "validate_phone_cuda_delta.py": "$validator_sha",
    },
}
raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
pathlib.Path("$out/RUN_CONTEXT.json").write_text(raw, encoding="ascii")
EOF

phone_started=1
adb -P "$adb_port" -s "$op12" shell "
    cd '$runtime'
    rm -f w6_delta_tail.log w6_delta_tail.pid
    nohup env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
        LLAMA_LAYER_START=30 LLAMA_LAYER_END=48 \
        LAYERSPLIT_MODEL_SHA256='$model_sha' \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        ./llama-layersplit \
        -m '$runtime/weights.gguf' \
        --mode tailv3 --port '$phone_tail_port' \
        --driver-batch 8 --driver-context 32 --driver-max-prefill 8 \
        --devices GPUOpenCL -ngl 99 \
        >'$runtime/w6_delta_tail.log' 2>&1 </dev/null &
    echo \$! >'$runtime/w6_delta_tail.pid'
"
wait_phone_log "$op12" w6_delta_tail.log \
    "[stagenet] listening on 0.0.0.0:$phone_tail_port" \
    w6_delta_tail.pid

adb -P "$adb_port" -s "$op15" shell "
    cd '$runtime'
    rm -f w6_delta_head.log w6_delta_head.pid
    nohup env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
        LLAMA_LAYER_START=0 LLAMA_LAYER_END=30 \
        LAYERSPLIT_MODEL_SHA256='$model_sha' \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        ./llama-layersplit \
        -m '$runtime/weights.gguf' \
        --mode stagenet --port '$phone_head_port' \
        --driver-batch 8 --driver-context 32 --driver-max-prefill 8 \
        --devices GPUOpenCL -ngl 99 \
        >'$runtime/w6_delta_head.log' 2>&1 </dev/null &
    echo \$! >'$runtime/w6_delta_head.pid'
"
wait_phone_log "$op15" w6_delta_head.log \
    "[stagenet] listening on 0.0.0.0:$phone_head_port" \
    w6_delta_head.pid

adb -P "$adb_port" -s "$op15" shell "
    cd '$runtime'
    rm -f w6_delta_relay.log w6_delta_relay.pid
    nohup ./llama-stage-direct-relay \
        --listen '$phone_relay_port' \
        --head '127.0.0.1:$phone_head_port' \
        --tail '$op12_wifi:$phone_tail_port' \
        >'$runtime/w6_delta_relay.log' 2>&1 </dev/null &
    echo \$! >'$runtime/w6_delta_relay.pid'
"
wait_phone_log "$op15" w6_delta_relay.log \
    "[direct-relay] listening on 0.0.0.0:$phone_relay_port" \
    w6_delta_relay.pid

env CUDA_VISIBLE_DEVICES=0 \
    LLAMA_LAYER_START=30 LLAMA_LAYER_END=48 \
    LAYERSPLIT_MODEL_SHA256="$model_sha" \
    LAYERSPLIT_PLACEMENT_CERT=1 \
    "$host_binary" \
    -m "$full_model" \
    --mode tailv3 --port "$cuda_tail_port" \
    --driver-batch 8 --driver-context 32 --driver-max-prefill 8 \
    --devices CUDA0 -ngl 99 \
    >"$out/cuda_tail.log" 2>&1 &
cuda_tail_pid=$!
host_pids+=("$cuda_tail_pid")
wait_host_log "$out/cuda_tail.log" \
    "[stagenet] listening on 0.0.0.0:$cuda_tail_port" \
    "$cuda_tail_pid"

env CUDA_VISIBLE_DEVICES=0 \
    LLAMA_LAYER_START=0 LLAMA_LAYER_END=30 \
    LAYERSPLIT_MODEL_SHA256="$model_sha" \
    LAYERSPLIT_PLACEMENT_CERT=1 \
    "$host_binary" \
    -m "$full_model" \
    --mode stagenet --port "$cuda_head_port" \
    --driver-batch 8 --driver-context 32 --driver-max-prefill 8 \
    --devices CUDA0 -ngl 99 \
    >"$out/cuda_head.log" 2>&1 &
cuda_head_pid=$!
host_pids+=("$cuda_head_pid")
wait_host_log "$out/cuda_head.log" \
    "[stagenet] listening on 0.0.0.0:$cuda_head_port" \
    "$cuda_head_pid"

"$host_relay" \
    --listen "$cuda_relay_port" \
    --head "127.0.0.1:$cuda_head_port" \
    --tail "127.0.0.1:$cuda_tail_port" \
    >"$out/cuda_relay.log" 2>&1 &
cuda_relay_pid=$!
host_pids+=("$cuda_relay_pid")
wait_host_log "$out/cuda_relay.log" \
    "[direct-relay] listening on 0.0.0.0:$cuda_relay_port" \
    "$cuda_relay_pid"

set +e
PYTHONDONTWRITEBYTECODE=1 python3 "$here/phone_cuda_delta_probe.py" \
    --contract "$contract" \
    --base-contract "$base_contract" \
    --phone-route "$op15_wifi:$phone_relay_port" \
    --cuda-route "127.0.0.1:$cuda_relay_port" \
    --timeout 300 \
    --journal-dir "$out/ownership_journal" \
    --run-id "$run_id" \
    --output "$out/delta_report.json" \
    >"$out/probe.stdout" 2>"$out/probe.stderr"
probe_rc=$?
set -e

host_exit_ok=1
phone_exit_ok=1
wait_host_exit "$cuda_relay_pid" || host_exit_ok=0
wait_host_exit "$cuda_head_pid" || host_exit_ok=0
wait_host_exit "$cuda_tail_pid" || host_exit_ok=0
wait_phone_exit "$op15" w6_delta_relay.pid || phone_exit_ok=0
wait_phone_exit "$op15" w6_delta_head.pid || phone_exit_ok=0
wait_phone_exit "$op12" w6_delta_tail.pid || phone_exit_ok=0
collect_phone_logs

if [[ -f $out/delta_report.json ]]; then
    set +e
    PYTHONDONTWRITEBYTECODE=1 python3 \
        "$here/validate_phone_cuda_delta.py" \
        "$out/delta_report.json" \
        --journal-dir "$out/ownership_journal" \
        --contract "$contract" \
        --base-contract "$base_contract" \
        --physical-gate "$physical_gate" \
        --run-context "$out/RUN_CONTEXT.json" \
        --op15-log "$out/op15_head.log" \
        --op12-log "$out/op12_tail.log" \
        --cuda-head-log "$out/cuda_head.log" \
        --cuda-tail-log "$out/cuda_tail.log" \
        --output "$out/delta_certificate.json" \
        >"$out/validator.stdout" 2>"$out/validator.stderr"
    validator_rc=$?
    set -e
else
    validator_rc=2
fi

python3 - "$out/EXIT_CODES.json" "$probe_rc" "$validator_rc" \
    "$host_exit_ok" "$phone_exit_ok" <<'EOF'
import json
import pathlib
import sys

value = {
    "host_exit_ok": bool(int(sys.argv[4])),
    "phone_exit_ok": bool(int(sys.argv[5])),
    "probe_rc": int(sys.argv[2]),
    "schema": "s39-phone-cuda-delta-exit-codes-v1",
    "validator_rc": int(sys.argv[3]),
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)
EOF

(
    cd "$out"
    find . -type f ! -name SHA256SUMS.txt -printf '%P\0' |
        sort -z |
        xargs -0 sha256sum
) >"$out/SHA256SUMS.txt"

phone_started=0
if (( probe_rc != 0 || validator_rc != 0 ||
      host_exit_ok == 0 || phone_exit_ok == 0 )); then
    printf 'W6_DELTA_GATE=FAIL probe_rc=%d validator_rc=%d host_exit=%d phone_exit=%d output=%s\n' \
        "$probe_rc" "$validator_rc" "$host_exit_ok" "$phone_exit_ok" "$out" >&2
    exit 3
fi
printf 'W6_DELTA_GATE=PASS output=%s\n' "$out"
