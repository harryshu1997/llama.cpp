#!/bin/bash
# matrix_queue2.sh PREFIX REPEATS "LABEL=KIND:TEMPLATE ..." [SEED]   (desktop, run with setsid nohup)
# Like matrix_queue.sh, but every entry names its own template; ATT = PREFIX<repeat><LABEL> (one prepare per ATT).
# Repeats x arms in a per-repeat shuffled order (seeded). Before every arm: rig_ready.sh must say READY (abort on
# LATCHED: the OP15 charger needs a replug, tell the user), and both phones must be at >= MIN_LEVEL % battery (waits,
# charging normally, up to 3 h). Each arm runs under meter_phones.sh --charging controlled (both phones metered,
# OP15 mmi charging off, Pixel battery-only; restored by the meter's trap). Stops at the first failed arm.
set -u
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=$1; REPEATS=$2; KINDS=$3; SEED=${4:-20260930}
MIN_LEVEL=${MIN_LEVEL:-50}
ADB="/usr/bin/adb -P 5037"
Q=$R/chains/QUEUE-$PREFIX.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$Q"; }
level(){ $ADB -s "$1" shell dumpsys battery < /dev/null 2>/dev/null | awk '/^  level:/{print $2; exit}'; }
log "queue2 start prefix=$PREFIX repeats=$REPEATS entries=[$KINDS] seed=$SEED"
for r in $(seq 1 "$REPEATS"); do
  ORDER=$(python3 -c "import random,sys; k=sys.argv[1].split(); random.Random(int(sys.argv[2])*1000+int(sys.argv[3])).shuffle(k); print(' '.join(k))" "$KINDS" "$SEED" "$r")
  log "repeat $r order=[$ORDER]"
  for ENTRY in $ORDER; do
    LABEL=${ENTRY%%=*}; SPEC=${ENTRY#*=}; KIND=${SPEC%%:*}; TEMPLATE=${SPEC#*:}; ATT=$PREFIX$r$LABEL
    [ -d "$R/$TEMPLATE" ] || { log "ABORT missing template $R/$TEMPLATE"; exit 6; }
    WAITED=0
    for i in $(seq 1 5760); do
      OUT=$("$R/rig_ready.sh" < /dev/null 2>&1)
      case "$OUT" in
        *LATCHED*) [ $WAITED = 1 ] || log "WAITING charger latched: replug the OP15 (queue resumes after the replug): $OUT"; WAITED=1;;
        *READY*) break;;
      esac
      [ $i -ge 360 ] && [ $WAITED = 0 ] && break
      sleep 30
    done
    [ $WAITED = 1 ] && log "latch cleared: $OUT"
    case "$OUT" in *READY*) ;; *) log "ABORT rig not ready: $OUT"; exit 4;; esac
    for i in $(seq 1 360); do
      A=$(level 3C15AU002CL00000); B=$(level 5A040DLCH004ES)
      [ -n "$A" ] && [ -n "$B" ] && [ "$A" -ge "$MIN_LEVEL" ] && [ "$B" -ge "$MIN_LEVEL" ] && break; sleep 30
    done
    log "arm $ATT/$KIND start (rig: $OUT; battery op15=$A pixel=$B)"
    "$HERE/meter_phones.sh" --charging controlled "$R/meter/$ATT-$KIND" "$ATT-$KIND" -- "$HERE/launch_one_arm.sh" "$ATT" "$TEMPLATE" "$KIND" >> "$Q" 2>&1
    RC=$?
    log "arm $ATT/$KIND exit=$RC"
    if [ $RC -ne 0 ]; then
      if [ "$KIND" = legacy ]; then ARMDIR=$R/inputs-desktop-legacy-$ATT; else ARMDIR=$R/inputs-$KIND-$ATT; fi
      if grep -q "charger latched" "$R/chains/CHAIN-$ATT-$KIND.jsonl" 2>/dev/null && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('status')=='PASS' and all(r.get('completion') for r in d['request_results']) else 1)" "$ARMDIR/run-eval/run/RESULT.json" 2>/dev/null; then
        log "arm $ATT/$KIND completed (RESULT PASS) but the post-run check found the charger latched: data kept, flagged LATCH_AT_POST_CHECK; next arm waits for the replug"
      else
        log "ABORT arm failed"; exit 5
      fi
    fi
  done
done
log "queue done"
