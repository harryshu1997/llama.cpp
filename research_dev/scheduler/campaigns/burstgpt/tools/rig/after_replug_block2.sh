#!/bin/bash
# after_replug_block2.sh  (desktop, setsid nohup) - after the OP15 replug: finish block 1 (p0m3 legacy + desktop), then block 2.
# Block 2 (prefix p2m, 3 repeats, arm order shuffled per repeat; start temperature + battery logged per arm):
#   ctl   = frozen full system (paper_config_v1, unchanged)         -> interleaved control for every contrast
#   er    = ctl + event_replanning                                  -> planner-off arm with the planner's event handling
#   pa    = er + active joint planner                               -> contrast pa vs er (planner on/off, same events)
#   dpon  = ctl + online GPU power control                          -> contrast dpon vs ctl (power on/off, same scheduling)
#   dpor  = ctl + oracle GPU power control                          -> contrast dpon vs dpor (online vs oracle, same scheduling)
#   dpond = dispatcher-only (no phones) + online power control      -> strongest no-phone baseline
#   dft   = out-of-the-box llama.cpp (stock router mode, defaults)  -> extra reference baseline (not the headline)
set -u
R=${EVAL_ROOT:-/mnt/storage/s43-two-phone-eval-20260925}
HERE=$(cd "$(dirname "$0")" && pwd)
Q=$R/chains/QUEUE-p0m.log
ADB="/usr/bin/adb -P 5037"
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$Q"; }
level(){ $ADB -s "$1" shell dumpsys battery < /dev/null 2>/dev/null | awk '/^  level:/{print $2; exit}'; }
for KIND in legacy desktop; do
  for i in $(seq 1 360); do
    OUT=$("$R/rig_ready.sh" < /dev/null 2>&1); case "$OUT" in *LATCHED*) log "ABORT charger latched (replug the OP15): $OUT"; exit 3;; *READY*) break;; esac; sleep 30
  done
  case "$OUT" in *READY*) ;; *) log "ABORT rig not ready after 3 h: $OUT"; exit 4;; esac
  for i in $(seq 1 360); do
    A=$(level 3C15AU002CL00000); B=$(level 5A040DLCH004ES)
    [ -n "$A" ] && [ -n "$B" ] && [ "$A" -ge 50 ] && [ "$B" -ge 50 ] && break; sleep 30
  done
  log "arm p0m3/$KIND start (resume after replug; battery op15=$A pixel=$B)"
  "$HERE/meter_phones.sh" --charging controlled "$R/meter/p0m3-$KIND" "p0m3-$KIND" -- "$HERE/launch_one_arm.sh" p0m3 template-eval2-s2 "$KIND" >> "$Q" 2>&1
  RC=$?; log "arm p0m3/$KIND exit=$RC"; [ $RC -eq 0 ] || { log "ABORT arm failed"; exit 5; }
done
log "block 1 done; starting block 2 (QUEUE-p2m)"
exec bash "$HERE/matrix_queue2.sh" p2m 3 "ctl=two-phone:template-eval2-s2 er=two-phone:template-eval2-s2-er pa=two-phone:template-eval2-s2-planner-active dpon=two-phone:template-eval2-s2-dp-online dpor=two-phone:template-eval2-s2-dp-oracle dpond=desktop:template-eval2-s2-dp-online dft=default:template-eval2-s2" 20260930
