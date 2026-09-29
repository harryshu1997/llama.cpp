#!/bin/bash
# Prints "READY" when the OP15 is at thermal status 0, <= 32.0 C, charger not latched, and the rig lock is free.
A="adb -P 5037 -s 3C15AU002CL00000 shell"
ST=$($A "dumpsys thermalservice" < /dev/null 2>/dev/null | grep -m1 "Thermal Status" | grep -o "[0-9]*")
T=$($A dumpsys battery < /dev/null 2>/dev/null | grep -m1 temperature | grep -o "[0-9]*")
NC=$($A "su -c cat /sys/class/oplus_chg/battery/battery_notify_code" < /dev/null 2>/dev/null | tr -d '\r\n ')
flock -n /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock true && L=free || L=held
echo "thermal=$ST temp=$T notify=$NC lock=$L"
if [ "$NC" = "512" ]; then echo "LATCHED"; exit 3; fi
[ "$ST" = "0" ] && [ -n "$T" ] && [ "$T" -le 320 ] && [ "$L" = "free" ] && echo READY
