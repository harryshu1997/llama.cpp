#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/myid/zs89458/Documents/llama.cpp-release
MODEL_DIR=/home/myid/zs89458/Documents/models
HOST_BIN="$ROOT/build-cuda/bin/llama-layersplit"
HOST_MODEL="$MODEL_DIR/gemma-4-12B-it-f16.gguf"
PHONE_DIR=/data/local/tmp/ls-s35
PHONE_BIN="$PHONE_DIR/llama-layersplit"
PHONE_MODEL=/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf
OP15=3C15AU002CL00000
CUDA_PORT=26420
TAIL_PORT=26421
OP15_HOST_PORT=26423
OP15_DEVICE_PORT=24423
HOST_MODEL_SHA=bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a
PHONE_MODEL_SHA=a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8
DRIVER_BATCH=2
DRIVER_CONTEXT=3072
DRIVER_MAX_PREFILL=64

usage() {
    echo "usage: $0 start|status|kill SESSION_DIR" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage
action=$1
session=$2

phone_alive() {
    local pid=$1
    adb -s "$OP15" shell "kill -0 $pid" >/dev/null 2>&1
}

case "$action" in
start)
    [[ ! -e "$session" ]] || { echo "session already exists: $session" >&2; exit 2; }
    [[ -x "$HOST_BIN" && -f "$HOST_MODEL" ]] || { echo "host artifacts missing" >&2; exit 2; }
    adb -s "$OP15" get-state | grep -qx device
    adb -s "$OP15" shell "test -x $PHONE_BIN && test -f $PHONE_MODEL"
    mkdir -p "$session"
    sha256sum "$HOST_BIN" "$HOST_MODEL" > "$session/host-sha256.txt"
    host_model_sha=$(awk -v path="$HOST_MODEL" '$2 == path {print $1}' "$session/host-sha256.txt")
    [[ "$host_model_sha" == "$HOST_MODEL_SHA" ]] || { echo "host model hash mismatch" >&2; exit 2; }
    adb -s "$OP15" shell "sha256sum $PHONE_BIN $PHONE_MODEL" \
        | tr -d '\r' > "$session/op15-sha256.txt"
    phone_model_sha=$(awk -v path="$PHONE_MODEL" '$2 == path {print $1}' "$session/op15-sha256.txt")
    [[ "$phone_model_sha" == "$PHONE_MODEL_SHA" ]] || { echo "phone model hash mismatch" >&2; exit 2; }
    adb -s "$OP15" forward "tcp:$OP15_HOST_PORT" "tcp:$OP15_DEVICE_PORT"

    nohup env CUDA_VISIBLE_DEVICES=0 \
        LLAMA_LAYER_START=0 LLAMA_LAYER_END=8 \
        LAYERSPLIT_DYNAMIC_CUT=1 LAYERSPLIT_KV_UNIFIED=1 \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        GGML_DECODE_NO_FA=1 \
        LAYERSPLIT_MODEL_SHA256="$HOST_MODEL_SHA" \
        "$HOST_BIN" -m "$HOST_MODEL" --devices CUDA0 -ngl 99 \
        --mode stagenet --port "$CUDA_PORT" -n 8 \
        --driver-batch "$DRIVER_BATCH" \
        --driver-context "$DRIVER_CONTEXT" \
        --driver-max-prefill "$DRIVER_MAX_PREFILL" \
        > "$session/cuda-head.log" 2>&1 < /dev/null &
    echo $! > "$session/cuda-head.pid"

    nohup env CUDA_VISIBLE_DEVICES=0 \
        LLAMA_LAYER_START=8 LLAMA_LAYER_END=48 \
        LAYERSPLIT_DYNAMIC_CUT=1 LAYERSPLIT_KV_UNIFIED=1 \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        GGML_DECODE_NO_FA=1 \
        LAYERSPLIT_MODEL_SHA256="$HOST_MODEL_SHA" \
        "$HOST_BIN" -m "$HOST_MODEL" --devices CUDA0 -ngl 99 \
        --mode tailv3 --port "$TAIL_PORT" -n 8 \
        --driver-batch "$DRIVER_BATCH" \
        --driver-context "$DRIVER_CONTEXT" \
        --driver-max-prefill "$DRIVER_MAX_PREFILL" \
        > "$session/cuda-tail.log" 2>&1 < /dev/null &
    echo $! > "$session/cuda-tail.pid"

    adb -s "$OP15" shell "cd $PHONE_DIR && (nohup env \
        LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=4192 \
        LLAMA_LAYER_START=0 LLAMA_LAYER_END=8 \
        LAYERSPLIT_DYNAMIC_CUT=1 LAYERSPLIT_KV_UNIFIED=1 \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        GGML_DECODE_NO_FA=1 \
        LAYERSPLIT_MODEL_SHA256=$PHONE_MODEL_SHA \
        $PHONE_BIN -m $PHONE_MODEL --devices HTP0 -ngl 99 \
        --mode stagenet --port $OP15_DEVICE_PORT -n 8 \
        --driver-batch $DRIVER_BATCH \
        --driver-context $DRIVER_CONTEXT \
        --driver-max-prefill $DRIVER_MAX_PREFILL \
        > $PHONE_DIR/s38-op15.log 2>&1 < /dev/null &)" >/dev/null
    op15_pid=""
    for _ in $(seq 1 60); do
        op15_pid=$(adb -s "$OP15" shell "pidof llama-layersplit" | tr -d '\r')
        [[ "$op15_pid" =~ ^[0-9]+$ ]] && break
        sleep 1
    done
    [[ "$op15_pid" =~ ^[0-9]+$ ]] || { echo "OP15 PID discovery failed" >&2; exit 2; }
    printf '%s\n' "$op15_pid" > "$session/op15.pid"

    for _ in $(seq 1 300); do
        cuda_pid=$(cat "$session/cuda-head.pid")
        tail_pid=$(cat "$session/cuda-tail.pid")
        if kill -0 "$cuda_pid" 2>/dev/null \
            && kill -0 "$tail_pid" 2>/dev/null \
            && phone_alive "$op15_pid" \
            && grep -q 'listening on' "$session/cuda-head.log" \
            && grep -q 'listening on' "$session/cuda-tail.log" \
            && adb -s "$OP15" shell "grep -q 'listening on' $PHONE_DIR/s38-op15.log"; then
            "$0" status "$session"
            exit 0
        fi
        sleep 1
    done
    echo "workers did not become ready" >&2
    "$0" status "$session" || true
    exit 2
    ;;
status)
    [[ -d "$session" ]] || { echo "session missing: $session" >&2; exit 2; }
    for name in cuda-head cuda-tail; do
        pid=$(cat "$session/$name.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$name pid=$pid state=live"
        else
            echo "$name pid=$pid state=dead"
        fi
    done
    op15_pid=$(cat "$session/op15.pid")
    if phone_alive "$op15_pid"; then
        echo "op15 pid=$op15_pid state=live"
    else
        echo "op15 pid=$op15_pid state=dead"
    fi
    ;;
kill)
    [[ -d "$session" ]] || { echo "session missing: $session" >&2; exit 2; }
    for name in cuda-head cuda-tail; do
        pid=$(cat "$session/$name.pid" 2>/dev/null || true)
        [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
    done
    op15_pid=$(cat "$session/op15.pid" 2>/dev/null || true)
    [[ -z "$op15_pid" ]] || adb -s "$OP15" shell "kill $op15_pid" >/dev/null 2>&1 || true
    adb -s "$OP15" forward --remove "tcp:$OP15_HOST_PORT" >/dev/null 2>&1 || true
    ;;
*)
    usage
    ;;
esac
