#!/bin/bash
A="/usr/bin/adb -P 5037 -s 3C15AU002CL00000"; L=/data/local/tmp/s43-dual-smoke-20260923
echo "=== procs"; $A shell "ps -A -o PID,PPID,ETIME,TIME,NAME | grep -i 'ffn-split\|21468\|21467'" </dev/null
$A shell kill -TERM 21468 </dev/null; sleep 3
if $A shell "ps -A -o PID,NAME | grep -qw 21468" </dev/null; then echo STILL-ALIVE-21468; else echo worker-21468-stopped; fi
echo "=== worker.log key lines"; $A shell "grep -n 'secondary\|dual\|ready\|S43\|error\|failed\|backend\|RESIDENCY' $L/worker.log | head -40" </dev/null
echo "=== lock holders now"; ps -eo pid,etime,cmd | grep -v grep | grep "flock" | head
echo "=== battery"; $A shell dumpsys battery </dev/null | grep -E "level|status|voltage|temperature|USB powered"
echo "=== notify"; $A shell "su -c 'cat /sys/class/oplus_chg/battery/battery_notify_code'" </dev/null
