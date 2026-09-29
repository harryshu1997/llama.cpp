#!/bin/bash
# MoE edge-energy screen: routing histogram + host energy arms (reference vs paging under caps).
set -u
D=/mnt/storage/s43-moe-energy-20260919-v1
M=/home/zhihao/moe-energy-20260919/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=/home/zhihao/moe-energy-20260919/runs
BIN=$D/cuda-build/bin
export LD_LIBRARY_PATH=$BIN:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
export LANG=C.UTF-8 LC_ALL=C.UTF-8
mkdir -p $OUT
LOCK=/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
exec 9>$LOCK
flock -n 9 || { echo "rig lock held" ; exit 3; }
echo "start $(date -Is)" > $OUT/RUN.log

step() { echo "== $1 $(date -Is)" | tee -a $OUT/RUN.log; }

# 1. Routing histogram on 65,536 WikiText-2 test tokens, chunks of 2048.
if [ ! -f $OUT/routing_histogram.json ]; then
  step histogram
  $BIN/llama-moe-routing-histogram -m $M -f $D/data/wiki.test.raw -c 2048 -b 2048 -ub 512 -ngl 99 -ot exps=CPU -t 16 \
      --max-tokens 65536 --out $OUT/routing_histogram.json > $OUT/histogram.log 2>&1 || echo "histogram failed rc=$?" | tee -a $OUT/RUN.log
fi

GATE="python3 $D/gate/moe_energy_gate.py --server $BIN/llama-server --lib-dir $BIN --model $M --prompt-file $D/data/wiki.test.raw --prompt-tokens 1024 --n-predict 128 --repeat 2 --ngl 99 --override-tensor exps=CPU --ctx 4096 --batch 2048 --ubatch 512"

# 2. Reference: everything in RAM, warm page cache, two thread counts.
for T in 16 8; do
  step "reference-t$T"
  $GATE --arm reference-t$T --threads $T --out $OUT/reference-t$T 2>&1 | tee -a $OUT/RUN.log
done

# 3. Paging arms: cold page cache, cgroup caps that cannot hold the expert bank.
for CAP in 10 6; do
  step "paging-${CAP}g"
  $GATE --arm paging-${CAP}g --threads 16 --memory-max $(( CAP * 1024 * 1024 * 1024 )) --drop-cache --out $OUT/paging-${CAP}g 2>&1 | tee -a $OUT/RUN.log
done

step done
