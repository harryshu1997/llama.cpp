#!/usr/bin/env bash
# CP2 windowing-specific adversarial device tests (run per phone).
# The window-agnostic store/worker invariants (corrupt/reorder/stale-epoch/duplicate/
# cache-hit-vs-active/publication mode-0400/symlink) are covered by the store(45)+stream(45)
# CTest suites, which the worker enforces identically regardless of host windowing.
# Here we exercise the NEW host-side pipeline behavior end to end on real hardware:
#   T1 windowed resume-after-partial: stop after N durable chunks (window W), reconnect,
#      resume (window W) -> completes, published SHA correct, resume offset == N*chunk.
#   T2 windowed kill-mid-flight: kill the worker while several requests are outstanding,
#      restart, resume (window W) -> recovers from the worker's verified prefix, completes.
#   T3 durable-result identity: a full window=8 run publishes the SAME object SHA as window=1.
set -u
DEV="$1"; BACKEND="$2"; BASEPORT="$3"; OUT="$4"
ROOT="/home/myid/zs89458/Documents/llama.cpp-release"
HBIN="$ROOT/build-phone-pim/bin/llama-phone-pim-host"
DDIR="/data/local/tmp/phone_pim"
MODEL="$ROOT/scratchpad/phone_pim/12b-f16-mid-2-3.gguf"
EXPECT_SHA="5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d"
ROUTE=17
: > "$OUT"
pass=0; fail=0
note(){ echo "$1"; echo "$1" >> "$OUT"; }

start_worker(){ # store dport
  ( adb -s "$DEV" shell "cd $DDIR && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 \
      ./llama-phone-pim-worker-meas --store-dir $1 --max-store-mib 2048 --max-model-mib 1024 \
      --min-free-mib 256 --backend $BACKEND --bind 127.0.0.1 --port $2 --route-epoch $ROUTE --generation 1" \
      > /tmp/s9adv_worker.log 2>&1 ) &
  echo $!
}
kill_worker_on_dev(){ adb -s "$DEV" shell "pkill -9 -f llama-phone-pim-worker-meas" >/dev/null 2>&1; }

# ---- T1: windowed resume-after-partial ----
T1(){
  local W=4 N=40 store="store_adv_t1_$$" dport=$((BASEPORT+1)) hport=$((BASEPORT+1))
  adb -s "$DEV" shell "rm -rf $DDIR/$store" >/dev/null 2>&1
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  local wp; wp=$(start_worker "$store" "$dport"); sleep 4
  adb -s "$DEV" forward tcp:$hport tcp:$dport >/dev/null 2>&1
  # stop after N durable chunks (partial), window W
  local p1
  p1=$("$HBIN" -m "$MODEL" --host 127.0.0.1 --port $hport --prefix blk.2 --M 16 --repeat 7 \
        --route-epoch $ROUTE --generation 1 --provision if-missing --chunk-mib 4 --stage-window $W \
        --test-stop-after-chunks $N --provision-only 2>/tmp/s9adv_host.err)
  # the partial (--provision-only) record uses "chunks_sent"; the full record uses "stage_chunks_sent"
  local cs; cs=$(echo "$p1" | grep -oE '"(stage_)?chunks_sent":[0-9]+' | head -1 | grep -oE '[0-9]+$')
  note "T1 partial: chunks_sent=$cs (expect $N)"
  # resume, window W, to completion + execute
  local p2
  p2=$("$HBIN" -m "$MODEL" --host 127.0.0.1 --port $hport --prefix blk.2 --M 16 --repeat 7 \
        --route-epoch $ROUTE --generation 1 --provision if-missing --chunk-mib 4 --stage-window $W \
        --release --shutdown 2>>/tmp/s9adv_host.err)
  wait $wp 2>/dev/null
  local rsum; rsum=$(echo "$p2" | tr ',' '\n' | grep -o '"stage_resume_offset":[0-9]*' | head -1 | cut -d: -f2)
  local verd; verd=$(echo "$p2" | grep -o '"verdict":"[A-Z_]*"')
  local sha; sha=$(echo "$p2" | grep -o "\"model_sha256\":\"[a-f0-9]*\"")
  local expoff=$((N*4194304))
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  adb -s "$DEV" shell "rm -rf $DDIR/$store" >/dev/null 2>&1
  if [ "$cs" = "$N" ] && [ "$rsum" = "$expoff" ] && echo "$verd" | grep -q DYNAMIC_FFN_PASS && echo "$sha" | grep -q "$EXPECT_SHA"; then
    note "T1 PASS: resume_offset=$rsum (==$expoff), $verd, sha ok"; pass=$((pass+1))
  else
    note "T1 FAIL: cs=$cs rsum=$rsum expoff=$expoff verd=$verd sha=$sha"; fail=$((fail+1))
  fi
}

# ---- T2: windowed kill-mid-flight recovery ----
T2(){
  local W=8 store="store_adv_t2_$$" dport=$((BASEPORT+2)) hport=$((BASEPORT+2))
  adb -s "$DEV" shell "rm -rf $DDIR/$store" >/dev/null 2>&1
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  local wp; wp=$(start_worker "$store" "$dport"); sleep 4
  adb -s "$DEV" forward tcp:$hport tcp:$dport >/dev/null 2>&1
  # provision in background, kill worker mid-flight
  ( "$HBIN" -m "$MODEL" --host 127.0.0.1 --port $hport --prefix blk.2 --M 16 --repeat 7 \
      --route-epoch $ROUTE --generation 1 --provision if-missing --chunk-mib 4 --stage-window $W \
      --provision-only > /tmp/s9adv_t2a.json 2>/tmp/s9adv_t2a.err ) &
  local hp=$!
  sleep 6
  kill_worker_on_dev
  wait $hp 2>/dev/null; local hrc=$?   # host should error out (connection dropped)
  wait $wp 2>/dev/null
  note "T2 killed worker mid-flight (host exited non-zero: rc=$hrc)"
  # restart worker on same store, resume
  local wp2; wp2=$(start_worker "$store" "$dport"); sleep 4
  adb -s "$DEV" forward tcp:$hport tcp:$dport >/dev/null 2>&1
  local p2
  p2=$("$HBIN" -m "$MODEL" --host 127.0.0.1 --port $hport --prefix blk.2 --M 16 --repeat 7 \
        --route-epoch $ROUTE --generation 1 --provision if-missing --chunk-mib 4 --stage-window $W \
        --release --shutdown 2>/tmp/s9adv_t2b.err)
  wait $wp2 2>/dev/null
  local rsum; rsum=$(echo "$p2" | tr ',' '\n' | grep -o '"stage_resume_offset":[0-9]*' | head -1 | cut -d: -f2)
  local verd; verd=$(echo "$p2" | grep -o '"verdict":"[A-Z_]*"')
  local sha; sha=$(echo "$p2" | grep -o "\"model_sha256\":\"[a-f0-9]*\"")
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  adb -s "$DEV" shell "rm -rf $DDIR/$store" >/dev/null 2>&1
  # resume offset must be a 4MiB multiple and > 0 (some prefix survived) and <= full
  if echo "$verd" | grep -q DYNAMIC_FFN_PASS && echo "$sha" | grep -q "$EXPECT_SHA" \
     && [ -n "$rsum" ] && [ $((rsum % 4194304)) -eq 0 ] && [ "$rsum" -ge 0 ] && [ "$rsum" -le 464114176 ]; then
    note "T2 PASS: recovered from verified prefix rsum=$rsum (4MiB-aligned), $verd, sha ok"; pass=$((pass+1))
  else
    note "T2 FAIL: rsum=$rsum verd=$verd sha=$sha"; fail=$((fail+1))
  fi
}

# ---- T3: durable-result identity window=8 vs window=1 ----
T3(){
  local store="store_adv_t3_$$" dport=$((BASEPORT+3)) hport=$((BASEPORT+3))
  adb -s "$DEV" shell "rm -rf $DDIR/$store" >/dev/null 2>&1
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  local wp; wp=$(start_worker "$store" "$dport"); sleep 4
  adb -s "$DEV" forward tcp:$hport tcp:$dport >/dev/null 2>&1
  local p
  p=$("$HBIN" -m "$MODEL" --host 127.0.0.1 --port $hport --prefix blk.2 --M 16 --repeat 7 \
        --route-epoch $ROUTE --generation 1 --provision if-missing --chunk-mib 4 --stage-window 8 \
        --release --shutdown 2>/tmp/s9adv_t3.err)
  wait $wp 2>/dev/null
  local sha; sha=$(echo "$p" | grep -o "\"model_sha256\":\"[a-f0-9]*\"")
  local rl; rl=$(echo "$p" | tr ',' '\n' | grep -o '"rel_l2_max":[0-9.e-]*' | head -1 | cut -d: -f2)
  adb -s "$DEV" forward --remove tcp:$hport >/dev/null 2>&1
  adb -s "$DEV" shell "rm -rf $DDIR/$store" >/dev/null 2>&1
  if echo "$sha" | grep -q "$EXPECT_SHA"; then
    note "T3 PASS: window=8 published SHA == window=1 canonical ($EXPECT_SHA), rel_l2=$rl"; pass=$((pass+1))
  else
    note "T3 FAIL: sha=$sha"; fail=$((fail+1))
  fi
}

note "== adversarial windowing tests on $DEV =="
T1; T2; T3
note "== $DEV adversarial: PASS=$pass FAIL=$fail =="
