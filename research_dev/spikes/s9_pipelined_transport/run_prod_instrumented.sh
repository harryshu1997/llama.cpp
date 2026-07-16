#!/usr/bin/env bash
# Re-run the CERTIFIED sequential dynamic provisioning (protocol v3, window=1) with
# the measurement-only instrumented host, into a FRESH empty store, to decompose the
# stage wall: stage_host_read/hash/send/ack_wait + stage_remote_* + commit.
# The on-device worker is the unmodified baseline (SHA 0a50ca72...); only the host
# adds timing fields to its JSON. Nothing about the v3 wire changes.
#
# Usage: run_prod_instrumented.sh <device_serial> <htp_backend> <dev_port> <host_port> <out.json>
set -u
DEV="$1"; BACKEND="$2"; DPORT="$3"; HPORT="$4"; OUT="$5"; WINDOW="${6:-1}"
ROOT="/home/myid/zs89458/Documents/llama.cpp-release"
HBIN="$ROOT/build-phone-pim/bin/llama-phone-pim-host"
DDIR="/data/local/tmp/phone_pim"
MODEL="$ROOT/scratchpad/phone_pim/12b-f16-mid-2-3.gguf"
STORE="store_cp1_prod_$$"
ROUTE=17

adb -s "$DEV" forward --remove tcp:$HPORT >/dev/null 2>&1
adb -s "$DEV" shell "rm -rf $DDIR/$STORE" >/dev/null 2>&1
# launch worker (fresh empty store, no --model) in background, capture stderr
( adb -s "$DEV" shell "cd $DDIR && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 \
    ./llama-phone-pim-worker-meas --store-dir $STORE --max-store-mib 2048 --max-model-mib 1024 \
    --min-free-mib 256 --backend $BACKEND --bind 127.0.0.1 --port $DPORT --route-epoch $ROUTE --generation 1" \
    > /tmp/s9prod_worker.log 2>&1 ) &
WP=$!
sleep 4
adb -s "$DEV" forward tcp:$HPORT tcp:$DPORT >/dev/null 2>&1
"$HBIN" -m "$MODEL" --host 127.0.0.1 --port $HPORT --prefix blk.2 --M 16 --repeat 7 \
    --route-epoch $ROUTE --generation 1 --provision if-missing --chunk-mib 4 --stage-window $WINDOW \
    --release --shutdown > "$OUT" 2>/tmp/s9prod_host.err
RC=$?
wait $WP 2>/dev/null
adb -s "$DEV" forward --remove tcp:$HPORT >/dev/null 2>&1
adb -s "$DEV" shell "rm -rf $DDIR/$STORE" >/dev/null 2>&1
echo "host rc=$RC  out=$OUT"
[ -s /tmp/s9prod_host.err ] && echo "host_err: $(tail -3 /tmp/s9prod_host.err)"
echo "--- worker recv_profile ---"; grep 'recv_profile' /tmp/s9prod_worker.log
cp /tmp/s9prod_worker.log "${OUT%.json}_worker.log" 2>/dev/null
echo "--- worker tail ---"; tail -4 /tmp/s9prod_worker.log
