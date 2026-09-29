#!/bin/bash
set -u
D=/mnt/storage/s43-moe-energy-20260919-v1
M=/home/zhihao/moe-energy-20260919/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=/home/zhihao/moe-energy-20260919/runs
BIN=$D/cuda-build/bin
export LD_LIBRARY_PATH=$BIN:/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
export LANG=C.UTF-8 LC_ALL=C.UTF-8
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -n 9 || { echo "rig lock held"; exit 3; }
echo "start $(date -Is)" > $OUT/RUN2.log
GATE="python3 $D/gate/moe_energy_gate.py --server $BIN/llama-server --lib-dir $BIN --model $M --prompt-file $D/data/wiki.test.raw --prompt-tokens 1024 --n-predict 128 --repeat 2 --ngl 99 --override-tensor exps=CPU --ctx 4096 --batch 2048 --ubatch 512 --threads 16"
for CAP in 10 6 4; do
  echo "== paging-${CAP}g-r2 $(date -Is)" | tee -a $OUT/RUN2.log
  $GATE --arm paging-${CAP}g-r2 --memory-max $(( CAP * 1024 * 1024 * 1024 )) --drop-cache --out $OUT/paging-${CAP}g-r2 2>&1 | grep -v "^  File\|^    \|~~~\|\^\^\^" | tee -a $OUT/RUN2.log
done
echo "== done $(date -Is)" | tee -a $OUT/RUN2.log
