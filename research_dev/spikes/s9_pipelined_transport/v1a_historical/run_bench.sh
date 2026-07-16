#!/usr/bin/env bash
# S9-V1A CP1 transport bench harness (measurement-only).
# Drives the standalone llama-phone-pim-bench over the adb-forwarded USB path.
# Emits one JSON object per run to <out>. Nothing here touches the v3 protocol,
# the production worker/host/store, or any on-device durable object.
#
# Usage: run_bench.sh <device_serial> <base_port> <label> <out.jsonl>
set -u
DEV="$1"; BASE_PORT="$2"; LABEL="$3"; OUT="$4"
ROOT="/home/myid/zs89458/Documents/llama.cpp-release"
HBIN="$ROOT/build-phone-pim/bin/llama-phone-pim-bench"
DBIN="./llama-phone-pim-bench"
DDIR="/data/local/tmp/phone_pim"
: > "$OUT"

# one h2d run: role sink on device, host drives; append both JSON lines
run_h2d() { # count window frame_sha final_ack rep
  local count="$1" window="$2" sha="$3" final="$4" rep="$5"
  local dport=$((BASE_PORT + (rep % 7)))
  local hport=$dport
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  local finalflag=""; [ "$final" = "1" ] && finalflag="--final-ack"
  ( adb -s "$DEV" shell "cd $DDIR && $DBIN --role sink --bind 127.0.0.1 --port $dport --payload-bytes 4194304 --count $count --frame-sha $sha $finalflag --timeout-ms 120000" \
      > /tmp/s9bench_sink.json 2>/tmp/s9bench_sink.err ) &
  local sp=$!
  sleep 1.2
  adb -s "$DEV" forward tcp:$hport tcp:$dport >/dev/null 2>&1
  local hj
  hj=$("$HBIN" --role host --host 127.0.0.1 --port $hport --experiment h2d \
        --payload-bytes 4194304 --count $count --window $window --frame-sha $sha $finalflag \
        --label "${LABEL}_h2d_c${count}_w${window}_sha${sha}_f${final}_r${rep}" 2>/tmp/s9bench_host.err)
  wait $sp
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  if [ -n "$hj" ]; then echo "$hj" >> "$OUT"; else echo "{\"error\":\"host\",\"detail\":\"$(tr -d '\n\"' </tmp/s9bench_host.err)\",\"run\":\"${LABEL}_c${count}_w${window}_sha${sha}_f${final}_r${rep}\"}" >> "$OUT"; fi
  local sj; sj=$(cat /tmp/s9bench_sink.json 2>/dev/null)
  [ -n "$sj" ] && echo "$sj" >> "$OUT"
  [ -s /tmp/s9bench_sink.err ] && echo "{\"sink_err\":\"$(tr -d '\n\"' </tmp/s9bench_sink.err)\"}" >> "$OUT"
}

run_d2h() { # count frame_sha rep
  local count="$1" sha="$2" rep="$3"
  local dport=$((BASE_PORT + 100 + (rep % 7)))
  local hport=$dport
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  ( adb -s "$DEV" shell "cd $DDIR && $DBIN --role source --bind 127.0.0.1 --port $dport --payload-bytes 4194304 --frame-sha $sha --timeout-ms 120000" \
      > /tmp/s9bench_src.json 2>/tmp/s9bench_src.err ) &
  local sp=$!
  sleep 1.2
  adb -s "$DEV" forward tcp:$hport tcp:$dport >/dev/null 2>&1
  local hj
  hj=$("$HBIN" --role host --host 127.0.0.1 --port $hport --experiment d2h \
        --payload-bytes 4194304 --count $count --frame-sha $sha \
        --label "${LABEL}_d2h_c${count}_sha${sha}_r${rep}" 2>/tmp/s9bench_host.err)
  wait $sp
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  [ -n "$hj" ] && echo "$hj" >> "$OUT"
  local sj; sj=$(cat /tmp/s9bench_src.json 2>/dev/null)
  [ -n "$sj" ] && echo "$sj" >> "$OUT"
}

REPS="${REPS:-5}"
echo "== $LABEL: h2d window sweep, 64 MiB, frame-sha off, $REPS reps =="
for w in 1 2 4 8; do for r in $(seq 1 $REPS); do run_h2d 16 $w 0 0 $r; done; done
echo "== $LABEL: h2d window sweep, 64 MiB, frame-sha ON, $REPS reps =="
for w in 1 2 4 8; do for r in $(seq 1 $REPS); do run_h2d 16 $w 1 0 $r; done; done
echo "== $LABEL: h2d streaming ceiling (final-ack), 64 MiB, $REPS reps =="
for r in $(seq 1 $REPS); do run_h2d 16 999 0 1 $r; done
echo "== $LABEL: d2h memory-source, 64 MiB, frame-sha off, $REPS reps =="
for r in $(seq 1 $REPS); do run_d2h 16 0 $r; done
echo "== done $LABEL -> $OUT =="
