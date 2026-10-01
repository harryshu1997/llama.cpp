#!/bin/bash
# matrix_queue.sh PREFIX TEMPLATE REPEATS "KIND KIND ..." [SEED]   (desktop, run with setsid nohup)
# Repeats x arms in a per-repeat shuffled order (seeded). Before every arm: rig_ready.sh must say READY (abort on
# LATCHED: the OP15 charger needs a replug, tell the user), and both phones must be at >= MIN_LEVEL % battery (waits,
# charging normally, up to 3 h). Each arm runs under meter_phones.sh --charging controlled (both phones metered,
# OP15 mmi charging off, Pixel battery-only; restored by the meter's trap). Stops at the first failed arm.
set -u
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=$1; TEMPLATE=$2; REPEATS=$3; KINDS=$4; SEED=${5:-20260930}
MIN_LEVEL=${MIN_LEVEL:-50}
ADB="/usr/bin/adb -P 5037"
Q=$R/chains/QUEUE-$PREFIX.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$Q"; }
level(){ $ADB -s "$1" shell dumpsys battery < /dev/null 2>/dev/null | awk '/^  level:/{print $2; exit}'; }
log "queue start prefix=$PREFIX template=$TEMPLATE repeats=$REPEATS kinds=[$KINDS] seed=$SEED"
for r in $(seq 1 "$REPEATS"); do
  ATT=$PREFIX$r
  ORDER=$(python3 -c "import random,sys; k=sys.argv[1].split(); random.Random(int(sys.argv[2])*1000+int(sys.argv[3])).shuffle(k); print(' '.join(k))" "$KINDS" "$SEED" "$r")
  log "repeat $r att=$ATT order=[$ORDER]"
  for KIND in $ORDER; do
    for i in $(seq 1 360); do
      OUT=$("$R/rig_ready.sh" < /dev/null 2>&1); case "$OUT" in *LATCHED*) log "ABORT charger latched (replug the OP15): $OUT"; exit 3;; *READY*) break;; esac; sleep 30
    done
    case "$OUT" in *READY*) ;; *) log "ABORT rig not ready after 3 h: $OUT"; exit 4;; esac
    for i in $(seq 1 360); do
      A=$(level 3C15AU002CL00000); B=$(level 5A040DLCH004ES)
      [ -n "$A" ] && [ -n "$B" ] && [ "$A" -ge "$MIN_LEVEL" ] && [ "$B" -ge "$MIN_LEVEL" ] && break; sleep 30
    done
    log "arm $ATT/$KIND start (rig: $OUT; battery op15=$A pixel=$B)"
    "$HERE/meter_phones.sh" --charging controlled "$R/meter/$ATT-$KIND" "$ATT-$KIND" -- "$HERE/launch_one_arm.sh" "$ATT" "$TEMPLATE" "$KIND" >> "$Q" 2>&1
    RC=$?
    log "arm $ATT/$KIND exit=$RC"
    [ $RC -eq 0 ] || { log "ABORT arm failed"; exit 5; }
  done
done
log "queue done"
