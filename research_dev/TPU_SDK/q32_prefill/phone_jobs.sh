#!/bin/bash
# Desktop-side sequential job runner for the Pixel TPU probe (adb port 5037, Pixel only).
# usage: phone_jobs.sh JOBFILE OUTDIR
# JOBFILE lines: TAG MODEL_PATH INPUT_PATH CASES WARMUP TIMED METRIC_CALLS HOLD_S ENERGY(0|1) KEEP(0|1) [ENV=VAL,...]
#   optional last column: comma-separated phone env for run1.sh, e.g. PERF=5 or NCOPIES=10,MINAVAIL=3000
# ENERGY=1 runs a root ODPM sampler (odpm_sampler.sh) on the phone around the (unprivileged) probe;
# the probe itself always runs as the shell user (the TPU dispatch runtime is unusable as root/magisk).
set -u
A="adb -P 5037 -s 5A040DLCH004ES"
PD=/data/local/tmp/s43-q32-tpu
JOBS=$1; OUT=$2; mkdir -p "$OUT"
sh_() { $A shell "$@" < /dev/null | tr -d '\r'; }
boot0=$(sh_ cat /proc/sys/kernel/random/boot_id)
echo "start $(date -Is) boot=$boot0" >> "$OUT/RUN.txt"
while read -r tag model input cases warm timed metrics hold energy keep perf; do
  [ -z "${tag:-}" ] && continue
  case $tag in \#*) continue;; esac
  echo "=== $tag $(date -Is)" >> "$OUT/RUN.txt"
  for f in "$model" "$input"; do
    b=$(basename "$f"); l=$(stat -c %s "$f"); p=$(sh_ "stat -c %s $PD/$b 2>/dev/null")
    if [ "$l" != "$p" ]; then
      t0=$(date +%s.%N); $A push "$f" "$PD/$b" < /dev/null >> "$OUT/RUN.txt" 2>&1; t1=$(date +%s.%N)
      echo "pushed $b bytes=$l s=$(echo "$t1 - $t0" | bc)" >> "$OUT/RUN.txt"
    fi
  done
  mb=$(basename "$model"); ib=$(basename "$input")
  sh_ "sha256sum $PD/$mb $PD/q32_tpu_probe $PD/libLiteRt.so $PD/libLiteRtDispatch_GoogleTensor.so" > "$OUT/$tag.hashes"
  penv=""
  [ -n "${perf:-}" ] && penv=$(echo "$perf" | tr ',' ' ')
  if [ "$energy" = 1 ]; then
    sh_ "rm -f $PD/odpm.$tag.txt $PD/odpm.$tag.stop"
    $A shell "su -c 'sh $PD/odpm_sampler.sh $PD/odpm.$tag.txt $PD/odpm.$tag.stop 3000'" < /dev/null > /dev/null 2>&1 &
    sampler=$!
    sleep 3   # idle baseline before the probe starts
  fi
  $A shell "$penv sh $PD/run1.sh $tag $mb $ib $cases $warm $timed $metrics $hold" < /dev/null > "$OUT/$tag.log" 2>&1 &
  bg=$!
  if [ "$hold" -gt 0 ]; then
    for i in $(seq 1 600); do
      grep -q hold_begin "$OUT/$tag.log" && break
      kill -0 $bg 2>/dev/null || break
      sleep 1
    done
    pid=$(grep -m1 '^pid=' "$OUT/$tag.log" | tr -dc 0-9)
    if grep -q hold_begin "$OUT/$tag.log"; then
      sh_ "su -c 'dmabuf_dump $pid'" > "$OUT/$tag.dmabuf.txt" 2>&1
      sh_ "su -c 'cat /proc/$pid/smaps_rollup; grep -E \"^(MemAvailable|MemFree|Cached|Shmem|Unevictable)\" /proc/meminfo'" > "$OUT/$tag.mem.txt" 2>&1
    fi
  fi
  wait $bg; rc=$?
  echo "exit $tag rc=$rc $(date -Is)" >> "$OUT/RUN.txt"
  if [ "$energy" = 1 ]; then
    sleep 2
    sh_ "touch $PD/odpm.$tag.stop"
    wait $sampler
    $A pull "$PD/odpm.$tag.txt" "$OUT/" < /dev/null > /dev/null 2>&1
    sh_ "rm -f $PD/odpm.$tag.txt $PD/odpm.$tag.stop"
  fi
  for c in $(seq 0 $((cases - 1))); do
    $A pull "$PD/out.$tag.case$c.f32" "$OUT/" < /dev/null > /dev/null 2>&1
  done
  sh_ "rm -f $PD/out.$tag.case*.f32"
  [ "$keep" = 1 ] || sh_ "rm -f $PD/$mb"
done < "$JOBS"
boot1=$(sh_ cat /proc/sys/kernel/random/boot_id)
echo "done $(date -Is) boot=$boot1 same_boot=$([ "$boot0" = "$boot1" ] && echo 1 || echo 0)" >> "$OUT/RUN.txt"
echo ALLDONE >> "$OUT/RUN.txt"
