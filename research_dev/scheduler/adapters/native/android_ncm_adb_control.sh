#!/system/bin/sh
set -eu

if [ "$#" -ne 5 ]; then
    echo "usage: $0 <state-directory> <functionfs-gadget> <tcp-port> <serial> <timeout-seconds>" >&2
    exit 2
fi
control_root=$1
gadget=$2
port=$3
serial=$4
timeout_seconds=$5
case "$control_root:$gadget" in
    /data/local/tmp/*:/config/usb_gadget/*) ;;
    *) echo "invalid control paths" >&2; exit 2 ;;
esac
case "$port:$timeout_seconds" in
    *[!0-9:]*|:*|*:) echo "invalid control limits" >&2; exit 2 ;;
esac
test "$port" -gt 0 && test "$port" -le 65535
test "$timeout_seconds" -gt 0 && test "$timeout_seconds" -le 7200
test "$(getprop ro.serialno)" = "$serial"
test "$(getprop ro.adb.secure)" = 1
test "$(getprop init.svc.adbd)" = running
configured_port=$(getprop service.adb.tcp.port)
if [ -z "$configured_port" ]; then
    configured_port=$(getprop persist.adb.tcp.port)
fi
test "$configured_port" = "$port"
test -z "$(cat "$gadget/UDC")"
test ! -e "$control_root/control.ready"
printf '%s\n' "$(cat /proc/sys/kernel/random/boot_id)" > "$control_root/control.boot"
printf '%s\n' armed > "$control_root/control.ready"

attempt=0
while [ -z "$(cat "$gadget/UDC")" ]; do
    test ! -e "$control_root/control.cancel" || exit 0
    attempt=$((attempt + 1))
    test "$attempt" -lt "$timeout_seconds"
    sleep 1
done
test ! -e "$control_root/control.cancel" || exit 0
# FunctionFS disables the Android gadget, whose init action stops adbd.
# Restore only the already configured authenticated TCP service, not USB.
test "$(getprop sys.usb.config)" = none
attempt=0
while [ "$(getprop init.svc.adbd)" = stopping ]; do
    attempt=$((attempt + 1))
    test "$attempt" -lt 50
    sleep 0.1
done
case "$(getprop init.svc.adbd)" in
    stopped) start adbd ;;
    running) ;;
    *) echo "unexpected adbd lifecycle" >&2; exit 1 ;;
esac
printf '%s\n' "$(date +%s) $(getprop init.svc.adbd) $(cat "$gadget/UDC")" \
    > "$control_root/control.applied"
