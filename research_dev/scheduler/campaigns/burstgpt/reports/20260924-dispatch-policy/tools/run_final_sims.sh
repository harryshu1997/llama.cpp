#!/bin/bash
cd "$(dirname "$0")"
mkdir -p ../sim/final
for run in dev2base ltbase; do
  timeout 3000 python3 simulate_dispatch.py ../newbase off --run $run --json ../sim/final/${run}_newbase_off.json --quiet > ../sim/final/${run}_newbase_off.log 2>&1
  for p in off wc aff aff:3:300; do
    n=$(echo $p | tr ':' '_')
    timeout 3000 python3 simulate_dispatch.py ../newroot $p --run $run --json ../sim/final/${run}_newroot_$n.json --quiet > ../sim/final/${run}_newroot_$n.log 2>&1
  done
done
echo done > ../sim/final/DONE
