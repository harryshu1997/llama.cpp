#!/usr/bin/env bash
# CP2 performance gate: full-shard dynamic provisioning at window {1,2,4,8}, rotated reps,
# fresh store each run, on one device. Emits one host JSON per run to <out.jsonl>.
# Uses the measurement worker (adds only stderr timing; wire is byte-identical v3).
set -u
DEV="$1"; BACKEND="$2"; BASEPORT="$3"; OUT="$4"; REPS="${5:-5}"
ROOT="/home/myid/zs89458/Documents/llama.cpp-release"
HBIN="$ROOT/build-phone-pim/bin/llama-phone-pim-host"
DDIR="/data/local/tmp/phone_pim"
MODEL="$ROOT/scratchpad/phone_pim/12b-f16-mid-2-3.gguf"
ROUTE=17
: > "$OUT"

one_run() { # window rep
  local W="$1" R="$2"
  local dport=$((BASEPORT + (W*10) + (R%7)))
  local hport=$dport
  local store="store_sweep_${W}_${R}_$$"
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  adb -s "$DEV" shell "rm -rf $DDIR/$store" >/dev/null 2>&1
  ( adb -s "$DEV" shell "cd $DDIR && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 \
      ./llama-phone-pim-worker-meas --store-dir $store --max-store-mib 2048 --max-model-mib 1024 \
      --min-free-mib 256 --backend $BACKEND --bind 127.0.0.1 --port $dport --route-epoch $ROUTE --generation 1" \
      > /tmp/s9sweep_worker.log 2>&1 ) &
  local wp=$!
  sleep 4
  adb -s "$DEV" forward tcp:$hport tcp:$dport >/dev/null 2>&1
  local hj
  hj=$("$HBIN" -m "$MODEL" --host 127.0.0.1 --port $hport --prefix blk.2 --M 16 --repeat 7 \
        --route-epoch $ROUTE --generation 1 --provision if-missing --chunk-mib 4 --stage-window $W \
        --release --shutdown 2>/tmp/s9sweep_host.err)
  local rc=$?
  wait $wp 2>/dev/null
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  adb -s "$DEV" shell "rm -rf $DDIR/$store" >/dev/null 2>&1
  local rp; rp=$(grep -o 'recv_profile {.*}' /tmp/s9sweep_worker.log | sed 's/recv_profile //')
  if [ $rc -eq 0 ] && [ -n "$hj" ]; then
    # splice the worker recv_profile into the host record for a self-contained row
    echo "${hj%\}}, \"worker_recv_profile\": ${rp:-null}, \"rep\": $R }" >> "$OUT"
  else
    echo "{\"error\":\"run\",\"window\":$W,\"rep\":$R,\"rc\":$rc,\"detail\":\"$(tr -d '\n\"' </tmp/s9sweep_host.err | tail -c 200)\"}" >> "$OUT"
  fi
  # brief cooldown to limit thermal drift across the sweep
  sleep 3
}

echo "== sweep $DEV: windows 1/2/4/8, $REPS rotated reps -> $OUT =="
for R in $(seq 1 "$REPS"); do
  for W in 1 2 4 8; do
    one_run "$W" "$R"
  done
done
echo "== sweep done $DEV =="
