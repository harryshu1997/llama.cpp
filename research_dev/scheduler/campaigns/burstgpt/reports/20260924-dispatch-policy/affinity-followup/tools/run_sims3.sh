#!/bin/bash
# Replays of the rebased trees: base2 (current main) and root2 (current main + fix).
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad/evidence-fixes/shared/gguf-py
for run in dp allon dev2base ltbase; do
  for tree in base2 root2; do
    for p in off wc aff; do
      [ -f ../sim2/${run}_${tree}_${p}.json ] && continue
      timeout 3000 python3 simulate_dispatch.py ../$tree $p --run $run --json ../sim2/${run}_${tree}_${p}.json --quiet > ../sim2/${run}_${tree}_${p}.log 2>&1
      echo "$run $tree $p exit $?" >> ../sim2/STATUS
    done
  done
done
echo done >> ../sim2/STATUS
