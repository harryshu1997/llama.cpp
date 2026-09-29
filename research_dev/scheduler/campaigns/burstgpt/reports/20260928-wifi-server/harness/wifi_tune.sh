#!/usr/bin/env bash
# Per-call WiFi latency tuning for the phone FFN workers (measured 2026-09-29, ../WIFI_TUNING.md).
# Run on the machine that has the phones on USB adb (the desktop). Every change is runtime-only and
# reverted by `off` (or a reboot).
#
#   ./wifi_tune.sh on|off|status [--adb-port 5037] [--op15 SERIAL] [--pixel SERIAL]
#
# on   OP15: mark the workers' and echo server's replies DSCP 46 (EF -> WMM voice queue; its best-effort
#            uplink queue averaged 1.2 ms channel access vs 0.17 ms for voice) + disable CPU idle states
#            above WFI.  Pixel: disable CPU idle states above WFI (its WiFi IRQs land on a core that
#            sleeps between calls). Host-side marking is NOT used: it made the Pixel slower.
# off  remove the rules and re-enable every idle state.
# Energy: disabling deep idle raises phone idle power (unmeasured); DSCP marking is free.
set -euo pipefail
port=5037; op15=3C15AU002CL00000; pixel=5A040DLCH004ES; action=""
while [ $# -gt 0 ]; do
    case "$1" in
        --adb-port) port=$2; shift 2 ;; --op15) op15=$2; shift 2 ;; --pixel) pixel=$2; shift 2 ;;
        on|off|status) action=$1; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$action" ] || { echo "usage: $0 on|off|status" >&2; exit 2; }
idle_script='
for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state*; do
  n=${s##*state}
  case $1 in off) [ $n -gt 0 ] && echo 1 > $s/disable ;; on) echo 0 > $s/disable ;; esac
done
echo "idle_states_disabled=$(cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable | grep -c 1)"'
rule='-p tcp -m multiport --sports 7070:7074 -j DSCP --set-dscp 46'
run() { adb -P "$port" -s "$1" shell "su -c 'sh -s'" < /dev/stdin; }
idle() { printf '%s\n' "set -- $2" "$idle_script" | run "$1"; }
dscp_count() { echo "iptables -t mangle -S OUTPUT | grep -c 'DSCP' || true" | run "$op15"; }
case $action in
    on)
        [ "$(dscp_count)" -gt 0 ] || echo "iptables -t mangle -A OUTPUT $rule" | run "$op15"
        echo -n "op15 "; idle "$op15" off; echo -n "pixel "; idle "$pixel" off ;;
    off)
        while [ "$(dscp_count)" -gt 0 ]; do echo "iptables -t mangle -D OUTPUT $rule" | run "$op15"; done
        echo -n "op15 "; idle "$op15" on; echo -n "pixel "; idle "$pixel" on ;;
    status)
        echo -n "op15 "; idle "$op15" status; echo -n "pixel "; idle "$pixel" status ;;
esac
echo "op15 dscp_rules=$(dscp_count)"
