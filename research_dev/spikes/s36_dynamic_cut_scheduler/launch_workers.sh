#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/myid/zs89458/Documents/llama.cpp-release
MODEL_DIR=/home/myid/zs89458/Documents/models
HOST_BIN="$ROOT/build-cuda/bin/llama-layersplit"
HOST_MODEL="$MODEL_DIR/gemma-4-12B-it-f16.gguf"
PHONE_DIR=/data/local/tmp/ls-s35
PHONE_BIN="$PHONE_DIR/llama-layersplit"
PHONE_MODEL=/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf
OP12=5ae7a43d
OP15=3C15AU002CL00000
CUDA_PORT=25420
TAIL_PORT=25421
OP12_HOST_PORT=25422
OP15_HOST_PORT=25423
OP12_DEVICE_PORT=24422
OP15_DEVICE_PORT=24423
HOST_MODEL_SHA=bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a
PHONE_MODEL_SHA=a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8

usage() {
    echo "usage: $0 start|status|kill SESSION_DIR" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage
action=$1
session=$2

phone_alive() {
    local serial=$1
    local pid=$2
    adb -s "$serial" shell "kill -0 $pid" >/dev/null 2>&1
}

case "$action" in
start)
    [[ ! -e "$session" ]] || { echo "session already exists: $session" >&2; exit 2; }
    [[ -x "$HOST_BIN" && -f "$HOST_MODEL" ]] || { echo "host artifacts missing" >&2; exit 2; }
    mkdir -p "$session"
    sha256sum "$HOST_BIN" "$HOST_MODEL" > "$session/host-sha256.txt"
    [[ "$(awk -v path="$HOST_MODEL" '$2 == path {print $1}' "$session/host-sha256.txt")" == "$HOST_MODEL_SHA" ]] || {
        echo "host model hash mismatch" >&2
        exit 2
    }
    for serial in "$OP12" "$OP15"; do
        adb -s "$serial" get-state | grep -qx device
        adb -s "$serial" shell "test -x $PHONE_BIN && test -f $PHONE_MODEL"
        name=op15
        [[ "$serial" == "$OP12" ]] && name=op12
        adb -s "$serial" shell "sha256sum $PHONE_BIN $PHONE_MODEL" \
            | tr -d '\r' > "$session/$name-sha256.txt"
        remote_sha=$(awk -v path="$PHONE_MODEL" '$2 == path {print $1}' "$session/$name-sha256.txt")
        [[ "$remote_sha" == "$PHONE_MODEL_SHA" ]] || {
            echo "$serial phone model hash mismatch" >&2
            exit 2
        }
    done
    adb -s "$OP12" forward "tcp:$OP12_HOST_PORT" "tcp:$OP12_DEVICE_PORT"
    adb -s "$OP15" forward "tcp:$OP15_HOST_PORT" "tcp:$OP15_DEVICE_PORT"

    nohup env CUDA_VISIBLE_DEVICES=0 \
        LLAMA_LAYER_START=0 LLAMA_LAYER_END=4 \
        LAYERSPLIT_DYNAMIC_CUT=1 \
        LAYERSPLIT_KV_UNIFIED=1 \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        LAYERSPLIT_MODEL_SHA256="$HOST_MODEL_SHA" \
        "$HOST_BIN" -m "$HOST_MODEL" --devices CUDA0 -ngl 99 \
        --mode stagenet --port "$CUDA_PORT" -n 4 \
        --driver-batch 64 --driver-context 8 --driver-max-prefill 4 \
        > "$session/cuda.log" 2>&1 &
    echo $! > "$session/cuda.pid"

    nohup env CUDA_VISIBLE_DEVICES=0 \
        LLAMA_LAYER_START=4 LLAMA_LAYER_END=48 \
        LAYERSPLIT_DYNAMIC_CUT=1 \
        LAYERSPLIT_KV_UNIFIED=1 \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        LAYERSPLIT_MODEL_SHA256="$HOST_MODEL_SHA" \
        "$HOST_BIN" -m "$HOST_MODEL" --devices CUDA0 -ngl 99 \
        --mode tailv3 --port "$TAIL_PORT" -n 4 \
        --driver-batch 64 --driver-context 8 --driver-max-prefill 4 \
        > "$session/tail.log" 2>&1 &
    echo $! > "$session/tail.pid"

    adb -s "$OP12" shell "cd $PHONE_DIR && (nohup env \
        LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=4192 \
        LLAMA_LAYER_START=0 LLAMA_LAYER_END=8 \
        LAYERSPLIT_DYNAMIC_CUT=1 LAYERSPLIT_KV_UNIFIED=1 \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        LAYERSPLIT_MODEL_SHA256=$PHONE_MODEL_SHA \
        $PHONE_BIN -m $PHONE_MODEL --devices HTP0 -ngl 99 \
        --mode stagenet --port $OP12_DEVICE_PORT -n 4 \
        --driver-batch 32 --driver-context 8 --driver-max-prefill 4 \
        > $PHONE_DIR/s36-op12.log 2>&1 < /dev/null &)" >/dev/null
    op12_pid=""
    for _ in $(seq 1 30); do
        op12_pid=$(adb -s "$OP12" shell "pidof llama-layersplit" | tr -d '\r')
        [[ -n "$op12_pid" ]] && break
        sleep 1
    done
    [[ "$op12_pid" =~ ^[0-9]+$ ]] || { echo "OP12 PID discovery failed" >&2; exit 2; }
    echo "$op12_pid" > "$session/op12.pid"

    adb -s "$OP15" shell "cd $PHONE_DIR && (nohup env \
        LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=4192 \
        LLAMA_LAYER_START=0 LLAMA_LAYER_END=8 \
        LAYERSPLIT_DYNAMIC_CUT=1 LAYERSPLIT_KV_UNIFIED=1 \
        LAYERSPLIT_PLACEMENT_CERT=1 \
        LAYERSPLIT_MODEL_SHA256=$PHONE_MODEL_SHA \
        $PHONE_BIN -m $PHONE_MODEL --devices HTP0 -ngl 99 \
        --mode stagenet --port $OP15_DEVICE_PORT -n 4 \
        --driver-batch 32 --driver-context 8 --driver-max-prefill 4 \
        > $PHONE_DIR/s36-op15.log 2>&1 < /dev/null &)" >/dev/null
    op15_pid=""
    for _ in $(seq 1 30); do
        op15_pid=$(adb -s "$OP15" shell "pidof llama-layersplit" | tr -d '\r')
        [[ -n "$op15_pid" ]] && break
        sleep 1
    done
    [[ "$op15_pid" =~ ^[0-9]+$ ]] || { echo "OP15 PID discovery failed" >&2; exit 2; }
    echo "$op15_pid" > "$session/op15.pid"

    for _ in $(seq 1 240); do
        cuda_pid=$(cat "$session/cuda.pid")
        tail_pid=$(cat "$session/tail.pid")
        if kill -0 "$cuda_pid" 2>/dev/null \
            && kill -0 "$tail_pid" 2>/dev/null \
            && phone_alive "$OP12" "$op12_pid" \
            && phone_alive "$OP15" "$op15_pid" \
            && grep -q 'listening on' "$session/cuda.log" \
            && grep -q 'listening on' "$session/tail.log" \
            && adb -s "$OP12" shell "grep -q 'listening on' $PHONE_DIR/s36-op12.log" \
            && adb -s "$OP15" shell "grep -q 'listening on' $PHONE_DIR/s36-op15.log"; then
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
    for name in cuda tail; do
        pid=$(cat "$session/$name.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$name pid=$pid state=live"
        else
            echo "$name pid=$pid state=dead"
        fi
    done
    for name_serial in "op12:$OP12" "op15:$OP15"; do
        name=${name_serial%%:*}
        serial=${name_serial#*:}
        pid=$(cat "$session/$name.pid")
        if phone_alive "$serial" "$pid"; then
            echo "$name pid=$pid state=live"
        else
            echo "$name pid=$pid state=dead"
        fi
    done
    ;;
kill)
    [[ -d "$session" ]] || { echo "session missing: $session" >&2; exit 2; }
    for name in cuda tail; do
        pid=$(cat "$session/$name.pid" 2>/dev/null || true)
        [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
    done
    for name_serial in "op12:$OP12" "op15:$OP15"; do
        name=${name_serial%%:*}
        serial=${name_serial#*:}
        pid=$(cat "$session/$name.pid" 2>/dev/null || true)
        [[ -z "$pid" ]] || adb -s "$serial" shell "kill $pid" >/dev/null 2>&1 || true
    done
    adb -s "$OP12" forward --remove "tcp:$OP12_HOST_PORT" >/dev/null 2>&1 || true
    adb -s "$OP15" forward --remove "tcp:$OP15_HOST_PORT" >/dev/null 2>&1 || true
    ;;
*)
    usage
    ;;
esac
