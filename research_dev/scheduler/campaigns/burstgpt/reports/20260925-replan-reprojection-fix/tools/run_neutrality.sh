#!/bin/bash
# Replays base vs root for every recorded run and policy, in parallel.
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad/evidence-fixes/shared/gguf-py
mkdir -p ../sim/neutral
for run in dp dev2base allon ltbase op15lt1; do
  for tree in base root; do
    for p in off wc aff; do
      for mode in desktop-baseline energy-aware; do
        out=../sim/neutral/${run}_${tree}_${p}_${mode}
        ( timeout 3000 /usr/bin/python3 simulate_dispatch.py ../$tree $p --run $run --selection-mode $mode --json $out.json --quiet > $out.log 2>&1; echo "$run $tree $p $mode exit $?" >> ../sim/neutral/STATUS ) &
      done
    done
  done
done
wait
echo done >> ../sim/neutral/STATUS
