#!/bin/bash
# prima_chain.sh ATT [TEMPLATE]  - wait for an idle rig + lock + both phones >= 50 %, then run the prima.cpp arm under
# meter_phones.sh (controlled charging), as block_g_chain.sh did for the RPC arm. Env passes through to the launcher
# (PRIMA_PHONES, PRIMA_LAYER_WINDOW, PRIMA_EXTRA, ...). Logs: $R/chains/PRIMA-$ATT.log
set -u
R=/mnt/storage/s43-two-phone-eval-20260925; D=/mnt/storage/s42-trace-v2-20260921-prep
LOCK=/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
ATT=${1:?ATT}; TEMPLATE=${2:-template-q8two-ts3-anchor-hfix-v2}
LOG=$R/chains/PRIMA-$ATT.log; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
level(){ timeout 15 adb -P 5037 -s $1 shell dumpsys battery < /dev/null | tr -d '\r' | awk '/^  level:/{print $2}'; }
MIN=${MIN_LEVEL:-50}
log "prima arm $ATT ($TEMPLATE): waiting for an idle rig, a free lock and both phones >= $MIN %"
for i in $(seq 1 1440); do
  busy=$(pgrep -f "queue_retry[.]sh|run_chain_eval.py --status|rpc_llamacpp_baseline run|prima_baseline.py run|weight_stream_microbench run" | wc -l)
  if [ "$busy" = 0 ] && flock -n $LOCK true 2>/dev/null; then
    A=$(level 3C15AU002CL00000); B=$(level 5A040DLCH004ES)
    if [ -n "$A" ] && [ -n "$B" ] && [ "$A" -ge $MIN ] && [ "$B" -ge $MIN ]; then log "rig free, battery op15=$A pixel=$B"; break; fi
    [ $((i % 20)) = 0 ] && log "rig free but battery op15=${A:-?} pixel=${B:-?} < $MIN: waiting"
  fi
  sleep 30
done
[ "${A:-0}" -ge $MIN ] 2>/dev/null && [ "${B:-0}" -ge $MIN ] 2>/dev/null || { log "ABORT: conditions not met within 12 h"; exit 6; }
export EVAL_REPO=$D/source LANG=C.UTF-8
log "arm $ATT/prima start"
$R/rig/meter_phones.sh --charging controlled $R/meter/$ATT-prima $ATT-prima -- $R/rig/launch_prima_baseline.sh $ATT $TEMPLATE >> $LOG 2>&1
RC=$?; log "arm $ATT/prima exit=$RC"
ls $R/inputs-prima-$ATT/run-eval/run/ 2>/dev/null | tr '\n' ' ' | tee -a $LOG; echo | tee -a $LOG
exit $RC
