#!/bin/bash
# prima_series.sh PREFIX N [TEMPLATE]  - N metered prima.cpp arms PREFIX1..PREFIXN via prima_chain.sh with a cool-down
# between them (default 600 s, PRIMA_COOLDOWN_S). Env passes through (PRIMA_PHONES, PRIMA_EXTRA, PRIMA_LAYER_WINDOW).
set -u
R=/mnt/storage/s43-two-phone-eval-20260925
PREFIX=${1:?PREFIX}; N=${2:?N}; TEMPLATE=${3:-template-q8two-ts3-anchor-hfix-v2}
LOG=$R/chains/PRIMA-SERIES-$PREFIX.log; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "series $PREFIX x$N ($TEMPLATE) phones=${PRIMA_PHONES:-default} extra=${PRIMA_EXTRA:-}"
for i in $(seq 1 $N); do
  ATT=$PREFIX$i
  [ -e $R/inputs-prima-$ATT/run-eval ] && { log "skip $ATT: already ran"; continue; }
  log "run $ATT"
  bash $R/rig/prima_chain.sh $ATT $TEMPLATE >> $LOG 2>&1; RC=$?
  log "run $ATT exit=$RC"
  python3 - "$R/inputs-prima-$ATT/run-eval/run/RESULT.json" <<'PY' 2>/dev/null | tee -a $LOG
import json,sys
r=json.load(open(sys.argv[1])); e=r.get("energy_summary",{})
print("  status", r["status"], r["status_reasons"], "counts", r["counts"], "duration_s", round(r["duration_us"]/1e6), "cpu_J", e.get("cpu_package_energy_j"), "gpu_J", e.get("gpu_board_energy_j"), "windows", r.get("prima_receipt",{}).get("layer_windows",[])[-1:])
PY
  [ $i -lt $N ] && { log "cool-down ${PRIMA_COOLDOWN_S:-600} s"; sleep ${PRIMA_COOLDOWN_S:-600}; }
done
log "series $PREFIX done"
