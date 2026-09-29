#!/bin/bash
# HVX-only (GGML_HEXAGON_NHMX=0) diagnostic, 2 reps
set -u
cd /home/zhihao/s43-dual-ffn-bench-20260923
LOCK=/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
for rep in 1 2; do
  case $rep in 1) ORDER="1.0 0.9 0.8 0.7";; 2) ORDER="0.7 0.8 0.9 1.0";; esac
  flock -w 1800 $LOCK bash -c "
    for f in $ORDER; do
      tag=hvx${rep}_f\${f}
      echo \"=== \$tag \$(date -Is)\" > logs/\$tag.log
      EXTRA_ENV=GGML_HEXAGON_NHMX=0 ./phone_run2.sh \$tag --frac \$f --layers 6 --layer-start 10 --sweeps 50 --warmup 3 --batches 4,8 --no-cpu-ref >> logs/\$tag.log 2>&1
      rc=\$?; echo \"rc=\$rc\" >> logs/\$tag.log; [ \$rc = 9 ] && exit 9
      sleep 15
    done
  " || { echo "diag rep $rep aborted" >> logs/diag.status; exit 1; }
done
echo DIAGDONE >> logs/diag.status
