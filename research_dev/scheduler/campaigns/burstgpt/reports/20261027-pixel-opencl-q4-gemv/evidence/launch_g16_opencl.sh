#!/bin/bash
# launch_g16_opencl.sh [REPEATS=1] [SUFFIX]: 1x fleet trace pair with the Pixel OpenCL FFN worker (evidence pointer switched by
# switch_pixel_evidence.sh). Arms: g16<r>nph (desktop) + g16<r>new (derived-mpaoa), the g62 templates.
set -u
R=/mnt/storage/s43-two-phone-eval-20260925
REPEATS=${1:-1}; SUFFIX=${2:-}
cd $R/rig || exit 1
. $R/rig/env_5070ti.sh
export RIG_READY_MAX_TEMP=${RIG_READY_MAX_TEMP:-320}
export RIG_READY_MAX_STATUS=${RIG_READY_MAX_STATUS:-0}
pgrep -x bash -a | grep -q "matrix_queue2.sh" && { echo "a queue is still running:"; pgrep -x bash -a | grep matrix_queue2.sh; exit 2; }
flock -n /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock true || { echo "rig lock held"; exit 2; }
[ -d $R/template-eval2-s2-dp-online-5070ti ] || { echo "missing template"; exit 3; }
grep -q "opencl" /mnt/storage/s43-ws12-pixel-placement-20261002/DERIVE_AOA_EVIDENCE || { echo "evidence pointer is not the OpenCL bundle"; exit 4; }
rm -f $R/PAUSE
bash gpu_footprint.sh $R/chains/GPU-FOOTPRINT-g16$SUFFIX-before.txt
setsid nohup bash matrix_queue2.sh g16$SUFFIX "$REPEATS" "nph=desktop:template-eval2-s2-dp-online-5070ti new=derived-mpaoa:template-eval2-s2-dp-online-5070ti" > $R/chains/QUEUE-g16$SUFFIX.out 2>&1 < /dev/null &
sleep 2; echo "launched queue g16$SUFFIX ($REPEATS repeat(s) of 2 arms); log $R/chains/QUEUE-g16$SUFFIX.log"
