#!/bin/bash
# Measured OP15 energy: disable battery charging (mmi_charging_enable=0) for one standard two-phone arm, re-enable after.
R=/mnt/storage/s43-two-phone-eval-20260925
ATT=${1:-pe1}
A="adb -P 5037 -s 3C15AU002CL00000 shell"
LOG=$R/chains/PHONE-ENERGY-$ATT.log
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
batt(){ $A "su -c 'cat /sys/class/power_supply/battery/charge_counter /sys/class/power_supply/battery/voltage_now /sys/class/power_supply/battery/status /sys/class/power_supply/battery/current_now /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null 2>/dev/null | tr '\n' ' '; }
reenable(){ $A "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null >/dev/null 2>&1; sleep 3; log "charging re-enabled: $(batt)"; }
trap reenable EXIT
log "before: $(batt) level=$($A dumpsys battery < /dev/null 2>/dev/null | grep -m1 level | grep -o '[0-9]*')"
$A "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null >/dev/null 2>&1; sleep 5
log "charging disabled: $(batt)"
"$R/launch_arm.sh" "$ATT" template-eval2
log "chain exit $?; after: $(batt) level=$($A dumpsys battery < /dev/null 2>/dev/null | grep -m1 level | grep -o '[0-9]*')"
