#!/usr/bin/env bash
# CP2.4: reproducible ASan/UBSan test of the ACTUAL host window>1 pipeline.
# Drives the real ASan-built host and worker over loopback with a synthetic object
# (provision-only, so no backend/model parse is needed) at windows 1,2,4,8 and a
# kill/resume case. Fails (nonzero) on any sanitizer diagnostic, non-zero exit, or a
# non-published outcome. Exercises the FIFO, prefix-digest retention, 64 MiB cap, and
# resume paths under ASan+UBSan.
set -u
ROOT="/home/myid/zs89458/Documents/llama.cpp-release"
HOST="$ROOT/build-phone-pim-asan/bin/llama-phone-pim-host"
WORKER="$ROOT/build-phone-pim-asan/bin/llama-phone-pim-worker"
[ -x "$HOST" ] && [ -x "$WORKER" ] || { echo "FAIL: ASan binaries missing"; exit 2; }
export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:exitcode=99"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
TMP=$(mktemp -d /tmp/s9_asan_win.XXXXXX)
trap 'rm -rf "$TMP"; pkill -f "$WORKER" 2>/dev/null' EXIT
OBJ="$TMP/obj.bin"; head -c 41943040 /dev/urandom > "$OBJ" # 40 MiB -> 10 x 4 MiB chunks
fails=0
sanitizer_hit() { grep -Eq "runtime error:|ERROR: AddressSanitizer|ERROR: LeakSanitizer|SUMMARY: UndefinedBehavior" "$1"; }

run_case() { # name window extra_host_args...  (worker+host share port)
  local name="$1" window="$2"; shift 2
  local port=$((41800 + RANDOM % 200))
  local store="$TMP/store_$name"
  "$WORKER" --store-dir "$store" --max-store-mib 512 --max-model-mib 256 --min-free-mib 32 \
     --backend HTP0 --bind 127.0.0.1 --port "$port" --route-epoch 3 --generation 1 \
     > "$TMP/${name}_worker.out" 2>&1 &
  local wp=$!
  sleep 0.7
  "$HOST" -m "$OBJ" --host 127.0.0.1 --port "$port" --prefix blk.2 --M 16 --repeat 7 \
     --route-epoch 3 --generation 1 --provision if-missing --chunk-mib 4 --stage-window "$window" \
     --provision-only "$@" > "$TMP/${name}_host.out" 2>&1
  local hrc=$?
  kill "$wp" 2>/dev/null; wait "$wp" 2>/dev/null
  local ok=1
  [ $hrc -eq 0 ] || { echo "  $name: host exit $hrc (expected 0)"; ok=0; }
  grep -q '"verdict":"DYNAMIC_PROVISION_PASS"' "$TMP/${name}_host.out" || { echo "  $name: not DYNAMIC_PROVISION_PASS"; ok=0; }
  grep -q '"model_source":"published_store"' "$TMP/${name}_host.out" || { echo "  $name: not published_store"; ok=0; }
  for f in "$TMP/${name}_host.out" "$TMP/${name}_worker.out"; do
    if sanitizer_hit "$f"; then echo "  $name: SANITIZER diagnostic in $(basename $f)"; ok=0; fi
  done
  if [ $ok -eq 1 ]; then echo "  $name (window=$window): PASS"; else echo "  $name (window=$window): FAIL"; fails=$((fails+1)); fi
}

echo "== ASan/UBSan host window pipeline test =="
for w in 1 2 4 8; do run_case "w$w" "$w"; done

# Windowed partial + resume under ASan: stop after 4 durable chunks (window 4), then resume.
PORT=$((42100 + RANDOM % 200)); STORE="$TMP/store_resume"
"$WORKER" --store-dir "$STORE" --max-store-mib 512 --max-model-mib 256 --min-free-mib 32 \
   --backend HTP0 --bind 127.0.0.1 --port "$PORT" --route-epoch 3 --generation 1 > "$TMP/resume_worker.out" 2>&1 &
WP=$!; sleep 0.7
"$HOST" -m "$OBJ" --host 127.0.0.1 --port "$PORT" --prefix blk.2 --M 16 --repeat 7 \
   --route-epoch 3 --generation 1 --provision if-missing --chunk-mib 4 --stage-window 4 \
   --test-stop-after-chunks 4 --provision-only > "$TMP/resume_a.out" 2>&1
"$HOST" -m "$OBJ" --host 127.0.0.1 --port "$PORT" --prefix blk.2 --M 16 --repeat 7 \
   --route-epoch 3 --generation 1 --provision if-missing --chunk-mib 4 --stage-window 4 \
   --provision-only > "$TMP/resume_b.out" 2>&1
rrc=$?
kill "$WP" 2>/dev/null; wait "$WP" 2>/dev/null
ok=1
grep -q '"verdict":"DYNAMIC_PROVISION_PARTIAL"' "$TMP/resume_a.out" || { echo "  resume-a: not PARTIAL"; ok=0; }
[ $rrc -eq 0 ] && grep -q '"verdict":"DYNAMIC_PROVISION_PASS"' "$TMP/resume_b.out" || { echo "  resume-b: not PASS"; ok=0; }
grep -q '"resume_offset":16777216' "$TMP/resume_b.out" || { echo "  resume-b: wrong resume offset (want 16777216)"; ok=0; }
for f in "$TMP"/resume_*.out; do if sanitizer_hit "$f"; then echo "  resume: SANITIZER in $(basename $f)"; ok=0; fi; done
if [ $ok -eq 1 ]; then echo "  resume (window=4): PASS"; else echo "  resume (window=4): FAIL"; fails=$((fails+1)); fi

echo "== ASan window test: $([ $fails -eq 0 ] && echo ALL PASS || echo "$fails FAILED") =="
exit $fails
