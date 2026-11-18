#!/bin/bash
# Same-night A/B of the Pixel worker: CPU-Pixel arm (g16c1new, pointer = cpu-path bundle) then OpenCL-Pixel arm (g16o1new, r2 bundle).
set -u
R=/mnt/storage/s43-two-phone-eval-20260925; W=/mnt/storage/s43-ws12-pixel-placement-20261002; A=/mnt/storage/s43-pixel-aoa-20261001-v1
LOG=$R/chains/AB-cpu-vs-opencl.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
CPU_EV=$(cat $W/DERIVE_AOA_EVIDENCE.cpu-path); CL_EV=$A/evidence-aoa-opencl-1-23-5070ti-r2/PIXEL_EVIDENCE_AOA.json
run_arm(){ local prefix=$1 ev=$2
  bash $R/rig/switch_pixel_evidence.sh "$ev" ab-$prefix 2>/dev/null | tee -a $LOG
  cd $R/rig; . $R/rig/env_5070ti.sh; export RIG_READY_MAX_TEMP=320 RIG_READY_MAX_STATUS=0
  pgrep -x bash -a | grep -q "matrix_queue2.sh" && { log "queue still running"; return 2; }
  rm -f $R/PAUSE; bash gpu_footprint.sh $R/chains/GPU-FOOTPRINT-$prefix-before.txt
  setsid nohup bash matrix_queue2.sh $prefix 1 "new=derived-mpaoa:template-eval2-s2-dp-online-5070ti" > $R/chains/QUEUE-$prefix.out 2>&1 < /dev/null &
  log "launched queue $prefix (pointer $ev)"
  for i in $(seq 1 240); do sleep 30; grep -q "queue done\|ABORT" $R/chains/QUEUE-$prefix.log 2>/dev/null && break; done
  log "queue $prefix finished: $(grep -E 'arm .* exit=' $R/chains/QUEUE-$prefix.log | tail -1)"
}
run_arm g16c "$CPU_EV"
run_arm g16o "$CL_EV"
log "AB done; pointer now $(cat $W/DERIVE_AOA_EVIDENCE)"
