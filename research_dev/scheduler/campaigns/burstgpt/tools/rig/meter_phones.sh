#!/bin/bash
# Meter BOTH phones (OP15 + Pixel 10 Pro) around one command on the desktop, e.g. one arm:
#
#   meter_phones.sh [--charging as-is|controlled] [--op15-period S] [--pixel-period S] OUTDIR NAME -- \
#       /mnt/storage/s43-two-phone-eval-20260925/launch_arm.sh ATT template-eval2-s2
#
# then   fleet_energy.py --run ATT=<inputs dir> --meter ATT=OUTDIR/METER.json
#
# What it does (run it OUTSIDE the rig lock: the wrapped launch_arm.sh takes the lock itself, and a nested flock
# on the same file would deadlock; the script refuses to start while the lock is held or run_chain_eval is running):
#  1. checks both phones are online (adb -P 5037), records a state snapshot (dumpsys battery + charge-control nodes)
#  2. --charging controlled (the measured-energy mode):
#       OP15  /sys/class/oplus_chg/battery/mmi_charging_enable = 0  (as run_phone_energy.sh/pe1): USB keeps powering
#             the phone up to the 500 mA SDP cap, the battery supplies the rest and only discharges;
#       Pixel /sys/devices/platform/google,charger/charge_stop_level = capacity - 10 (min 5): the google charger
#             suspends the USB input while SoC > stop level -> battery-only, every joule through the fuel gauge
#             (tested 2026-09-30: input off < 1 s after the write, back < 1 s after the restore).
#     Both original values are read first and restored by an EXIT/INT/TERM trap, then verified (OP15 mmi value,
#     Pixel stop level + USB input current back > 20 mA); the verification lands in OUTDIR/NAME-meter.log.
#     --charging as-is (default) changes nothing; the accountant's formula (USB + net battery) covers charging too.
#  3. pushes phone_power_sampler.sh and starts it as root on both phones (nohup, stop-file terminated, never killed):
#     OP15 every 1 s (usb V/I/limit, battery V/I/counter; the coulomb counter is the battery truth, its current_now
#     reads ~0.5x), Pixel every 0.25 s (usb V/I/limit, battery V/I/avg/counter; IBUS and the fuel-gauge current
#     refresh every ~0.15-0.45 s, a read costs < 1 ms).
#  4. host/phone clock anchors (host epoch + CLOCK_MONOTONIC around `cat /proc/uptime`) at start and stop.
#  5. runs the command, stops the samplers, pulls the logs and writes OUTDIR/METER.json in the POWER.json schema
#     run_chain_eval.py uses (keys op15/pixel: serial, local, anchor_*, end_*), plus period, mode and control values.
# Read-only on the phones except the two charge-control nodes of --charging controlled.
set -u
ADB=${METER_ADB:-"/usr/bin/adb -P 5037"}   # METER_ADB / METER_LOCK / METER_SETTLE_S: test hooks only
SETTLE=${METER_SETTLE_S:-5}
OP15=3C15AU002CL00000
PIXEL=5A040DLCH004ES
PHONE_DIR=/data/local/tmp/ws3-meter
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=as-is; OP15_PERIOD=1; PIXEL_PERIOD=0.25; MAX_SAMPLES=100000
OP15_MMI=/sys/class/oplus_chg/battery/mmi_charging_enable
PIXEL_STOP=/sys/devices/platform/google,charger/charge_stop_level
OP15_NODES="usb/current_now usb/voltage_now usb/input_current_limit battery/current_now battery/voltage_now battery/charge_counter $OP15_MMI"
PIXEL_NODES="usb/current_now usb/voltage_now usb/input_current_limit battery/current_now battery/current_avg battery/voltage_now battery/charge_counter $PIXEL_STOP"

while [ $# -gt 0 ]; do
  case "$1" in
    --charging) MODE=$2; shift 2 ;;
    --op15-period) OP15_PERIOD=$2; shift 2 ;;
    --pixel-period) PIXEL_PERIOD=$2; shift 2 ;;
    --) shift; break ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) if [ -z "${OUT:-}" ]; then OUT=$1; elif [ -z "${NAME:-}" ]; then NAME=$1; else echo "extra argument $1" >&2; exit 2; fi; shift ;;
  esac
done
[ -n "${OUT:-}" ] && [ -n "${NAME:-}" ] && [ $# -gt 0 ] || { sed -n '2,8p' "$0"; exit 2; }
case "$MODE" in as-is|controlled) ;; *) echo "--charging must be as-is or controlled" >&2; exit 2 ;; esac
mkdir -p "$OUT" || exit 2
LOG=$OUT/$NAME-meter.log
[ -e "$OUT/METER.json" ] && { echo "$OUT/METER.json exists" >&2; exit 2; }
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
sh1(){ $ADB -s "$1" shell "$2" < /dev/null 2>/dev/null | tr -d '\r'; }
su1(){ sh1 "$1" "su -c '$2'"; }
now(){ python3 -c 'import time; print("%.6f %.6f" % (time.time(), time.monotonic()))'; }

LOCK=${METER_LOCK:-/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock}
[ -z "$(pgrep -f '^python3 [^ ]*[r]un_chain_eval[.]py' 2>/dev/null)" ] || { log "run_chain_eval is running: not metering"; exit 3; }
[ ! -e "$LOCK" ] || flock -n "$LOCK" true || { log "rig lock is held: not metering"; exit 3; }
for s in $OP15 $PIXEL; do
  [ "$($ADB -s $s get-state < /dev/null 2>/dev/null)" = device ] || { log "phone $s not online"; exit 3; }
done

snapshot(){  # $1 = tag
  for s in $OP15 $PIXEL; do
    { echo "## $1 $s $(date -u +%FT%TZ)"; sh1 $s "dumpsys battery"; } >> "$OUT/$NAME-state.txt"
  done
  echo "## $1 op15 mmi_charging_enable=$(su1 $OP15 "cat $OP15_MMI") pixel charge_stop_level=$(su1 $PIXEL "cat $PIXEL_STOP") pixel usb_ua=$(su1 $PIXEL 'cat /sys/class/power_supply/usb/current_now')" >> "$OUT/$NAME-state.txt"
}

ORIG_MMI=""; ORIG_STOP=""; ORIG_USB=0; STARTED=""; CHANGED=""
cleanup(){  # EXIT/INT/TERM: stop any sampler still running, then restore charging
  local label serial
  for label in $STARTED; do
    [ "$label" = op15 ] && serial=$OP15 || serial=$PIXEL
    su1 $serial "touch $PHONE_DIR/$NAME-$label.stop"
  done
  restore
}
restore(){
  [ "$MODE" = controlled ] && [ -n "$CHANGED" ] || return 0
  if [ -n "$ORIG_MMI" ]; then su1 $OP15 "echo $ORIG_MMI > $OP15_MMI"; fi
  if [ -n "$ORIG_STOP" ]; then su1 $PIXEL "echo $ORIG_STOP > $PIXEL_STOP"; fi
  sleep "$SETTLE"
  local mmi stop usb ok=1
  mmi=$(su1 $OP15 "cat $OP15_MMI"); stop=$(su1 $PIXEL "cat $PIXEL_STOP"); usb=$(su1 $PIXEL 'cat /sys/class/power_supply/usb/current_now')
  [ -z "$ORIG_MMI" ] || [ "$mmi" = "$ORIG_MMI" ] || ok=0
  [ -z "$ORIG_STOP" ] || [ "$stop" = "$ORIG_STOP" ] || ok=0
  if [ "$ORIG_USB" -gt 20000 ] 2>/dev/null; then  # input was on before: it must be on again
    [ -n "$usb" ] && [ "$usb" -gt 20000 ] 2>/dev/null || ok=0
  fi
  log "charging restored: op15 mmi=$mmi (orig $ORIG_MMI) pixel stop=$stop (orig $ORIG_STOP) pixel usb_ua=$usb verified=$ok"
  [ $ok = 1 ] || log "WARNING: charging state NOT verified; check both phones by hand"
  MODE=restored
}
trap 'cleanup' EXIT
trap 'exit 130' INT TERM

snapshot before
if [ "$MODE" = controlled ]; then
  ORIG_MMI=$(su1 $OP15 "cat $OP15_MMI"); ORIG_STOP=$(su1 $PIXEL "cat $PIXEL_STOP")
  ORIG_USB=$(su1 $PIXEL 'cat /sys/class/power_supply/usb/current_now'); ORIG_USB=${ORIG_USB:-0}
  CAP=$(sh1 $PIXEL "cat /sys/class/power_supply/battery/capacity")
  case "$ORIG_MMI$ORIG_STOP$CAP" in ''|*[!0-9]*) log "cannot read charge-control state (mmi=$ORIG_MMI stop=$ORIG_STOP cap=$CAP)"; exit 4 ;; esac
  LEVEL=$((CAP - 10)); [ $LEVEL -lt 5 ] && LEVEL=5
  [ "$CAP" -ge 20 ] || { log "Pixel capacity $CAP % < 20 %: not starting battery-only mode"; exit 4; }
  CHANGED=1
  su1 $OP15 "echo 0 > $OP15_MMI"
  su1 $PIXEL "echo $LEVEL > $PIXEL_STOP"
  sleep "$SETTLE"
  NOW_MMI=$(su1 $OP15 "cat $OP15_MMI"); NOW_STOP=$(su1 $PIXEL "cat $PIXEL_STOP")
  log "charging controlled: op15 mmi $ORIG_MMI -> $NOW_MMI, pixel stop $ORIG_STOP -> $NOW_STOP (capacity $CAP), pixel usb_ua=$(su1 $PIXEL 'cat /sys/class/power_supply/usb/current_now')"
  [ "$NOW_MMI" = 0 ] && [ "$NOW_STOP" = "$LEVEL" ] || { log "charge control did not take effect: not running"; exit 4; }
fi

declare -A START_JSON
start_one(){  # label serial period nodes
  local label=$1 serial=$2 period=$3 nodes=$4 out stop t0 up t1
  out=$PHONE_DIR/$NAME-$label.txt; stop=$PHONE_DIR/$NAME-$label.stop
  sh1 $serial "mkdir -p $PHONE_DIR" > /dev/null
  $ADB -s $serial push "$HERE/phone_power_sampler.sh" $PHONE_DIR/ < /dev/null > /dev/null 2>&1 || { log "push failed on $label"; exit 5; }
  [ -z "$(sh1 $serial "ls $out $stop 2>/dev/null")" ] || { log "sampler files exist on $label: $out"; exit 5; }
  t0=$(now); up=$(sh1 $serial "cat /proc/uptime" | cut -d' ' -f1); t1=$(now)
  su1 $serial "nohup sh $PHONE_DIR/phone_power_sampler.sh $out $stop $period $MAX_SAMPLES $nodes > /dev/null 2>&1 < /dev/null &"
  STARTED="$STARTED $label"
  START_JSON[$label]=$(python3 -c 'import json,sys
e0,m0=map(float,sys.argv[1].split()); e1,m1=map(float,sys.argv[2].split())
print(json.dumps({"serial":sys.argv[3],"out":sys.argv[4],"stop":sys.argv[5],"period_s":float(sys.argv[6]),
  "anchor_host_epoch_s":(e0+e1)/2,"anchor_host_monotonic_s":(m0+m1)/2,"anchor_phone_uptime_s":float(sys.argv[7]),
  "anchor_uncertainty_s":e1-e0}))' "$t0" "$t1" "$serial" "$out" "$stop" "$period" "$up")
}
stop_one(){  # label serial -> prints the finished JSON row
  local label=$1 serial=$2 row out stop i t0 up t1 local_file
  row=${START_JSON[$label]}
  out=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["out"])' "$row")
  stop=${out%.txt}.stop
  su1 $serial "touch $stop"
  for i in $(seq 1 20); do sh1 $serial "tail -n 1 $out" | grep -q "^# stop" && break; sleep 1; done
  t0=$(now); up=$(sh1 $serial "cat /proc/uptime" | cut -d' ' -f1); t1=$(now)
  local_file=$OUT/$NAME-POWER-$label.txt
  $ADB -s $serial pull "$out" "$local_file" < /dev/null > /dev/null 2>&1 || log "pull failed on $label" >&2
  python3 -c 'import json,sys
row=json.loads(sys.argv[1]); e0,m0=map(float,sys.argv[2].split()); e1,m1=map(float,sys.argv[3].split())
import os
text=open(sys.argv[5]).read() if os.path.isfile(sys.argv[5]) else ""
row.update({"local":sys.argv[5],"end_host_epoch_s":(e0+e1)/2,"end_host_monotonic_s":(m0+m1)/2,"end_phone_uptime_s":float(sys.argv[4]),
  "samples":sum(1 for l in text.splitlines() if l and not l.startswith("#")),"stopped":"# stop" in text})
print(json.dumps(row))' "$row" "$t0" "$t1" "$up" "$local_file"
}

start_one op15 $OP15 "$OP15_PERIOD" "$OP15_NODES"
start_one pixel $PIXEL "$PIXEL_PERIOD" "$PIXEL_NODES"
log "samplers started (op15 ${OP15_PERIOD}s, pixel ${PIXEL_PERIOD}s, charging $MODE); running: $*"
"$@" < /dev/null
CODE=$?
log "command exit $CODE"
R_OP15=$(stop_one op15 $OP15); R_PIXEL=$(stop_one pixel $PIXEL); STARTED=""
python3 -c 'import json,sys
meter={"op15":json.loads(sys.argv[1]),"pixel":json.loads(sys.argv[2])}
for row in meter.values(): row["charging_mode"]=sys.argv[3]
json.dump(meter,open(sys.argv[4],"w"),indent=1); print(open(sys.argv[4]).read())' "$R_OP15" "$R_PIXEL" "$MODE" "$OUT/METER.json" >> "$LOG"
restore
snapshot after
trap - EXIT
log "done: $OUT/METER.json (command exit $CODE)"
exit $CODE
