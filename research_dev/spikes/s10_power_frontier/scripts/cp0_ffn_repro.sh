#!/usr/bin/env bash
# CP0 integrity: reproduce the smallest certified resident dense-FFN correctness
# case (blk.2, M=16, repeat=7) on one phone using the CERTIFIED worker
# (0a50ca72...e749), prestaged path. Emits the host JSON verdict record.
# Usage: cp0_ffn_repro.sh <serial> <dport> <hport> <out.json>
set -u
DEV="$1"; DPORT="$2"; HPORT="$3"; OUT="$4"
ROOT="/home/myid/zs89458/Documents/llama.cpp-release"
HBIN="$ROOT/build-phone-pim/bin/llama-phone-pim-host"
MODEL="$ROOT/scratchpad/phone_pim/12b-f16-mid-2-3.gguf"
DDIR="/data/local/tmp/phone_pim"
DMODEL="$DDIR/model_s10.gguf"
ROUTE=1

adb -s "$DEV" forward --remove tcp:$HPORT >/dev/null 2>&1
adb -s "$DEV" shell "pkill -f llama-phone-pim-worker" >/dev/null 2>&1
( adb -s "$DEV" shell "cd $DDIR && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 \
    ./llama-phone-pim-worker --model model_s10.gguf --backend HTP0 --bind 127.0.0.1 --port $DPORT \
    --route-epoch $ROUTE --generation 1" > /tmp/s10_cp0_worker_$DEV.log 2>&1 ) &
WP=$!
sleep 6
adb -s "$DEV" forward tcp:$HPORT tcp:$DPORT >/dev/null 2>&1
"$HBIN" -m "$MODEL" --host 127.0.0.1 --port $HPORT --prefix blk.2 --M 16 --repeat 7 \
    --route-epoch $ROUTE --generation 1 --release --shutdown > "$OUT" 2>/tmp/s10_cp0_host_$DEV.err
RC=$?
wait $WP 2>/dev/null
adb -s "$DEV" forward --remove tcp:$HPORT >/dev/null 2>&1
echo "host rc=$RC out=$OUT"
[ -s /tmp/s10_cp0_host_$DEV.err ] && echo "host_err_tail: $(tail -2 /tmp/s10_cp0_host_$DEV.err)"
