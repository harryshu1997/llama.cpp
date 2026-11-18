#!/bin/bash
# launch_g16r_opencl.sh [REPEATS=1]: rerun of the Pixel-assisted 1x arm only (derived-mpaoa) after the worker's accept-timeout
# fix; the desktop arm g161nph (01:04-01:33 UTC) stands. Arm: g16r<r>new.
set -u
R=/mnt/storage/s43-two-phone-eval-20260925
REPEATS=${1:-1}
cd $R/rig || exit 1
. $R/rig/env_5070ti.sh
export RIG_READY_MAX_TEMP=${RIG_READY_MAX_TEMP:-320}
export RIG_READY_MAX_STATUS=${RIG_READY_MAX_STATUS:-0}
pgrep -x bash -a | grep -q "matrix_queue2.sh" && { echo "a queue is still running:"; pgrep -x bash -a | grep matrix_queue2.sh; exit 2; }
flock -n /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock true || { echo "rig lock held"; exit 2; }
grep -q "opencl" /mnt/storage/s43-ws12-pixel-placement-20261002/DERIVE_AOA_EVIDENCE || { echo "evidence pointer is not the OpenCL bundle"; exit 4; }
rm -f $R/PAUSE
bash gpu_footprint.sh $R/chains/GPU-FOOTPRINT-g16r-before.txt
setsid nohup bash matrix_queue2.sh g16r "$REPEATS" "new=derived-mpaoa:template-eval2-s2-dp-online-5070ti" > $R/chains/QUEUE-g16r.out 2>&1 < /dev/null &
sleep 2; echo "launched queue g16r ($REPEATS repeat(s) of 1 arm); log $R/chains/QUEUE-g16r.log"
