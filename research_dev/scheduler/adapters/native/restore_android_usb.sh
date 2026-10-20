#!/system/bin/sh
set +e

if [ "$#" -ne 1 ]; then
    exit 2
fi

session_root=$1
g1=$(cat "$session_root/android-gadget.path" 2>/dev/null)
g2=$(cat "$session_root/functionfs-gadget.path" 2>/dev/null)
udc=$(cat "$session_root/phone-udc.name" 2>/dev/null)
ffs_root=$(cat "$session_root/functionfs-root.path" 2>/dev/null)
normal_usb_config=$(cat "$session_root/android-usb-config" 2>/dev/null)

case "$g1:$g2:$ffs_root:$udc:$normal_usb_config" in
    /*:/*:/*:[A-Za-z0-9._-]*:[A-Za-z0-9,._-]*) ;;
    *) exit 2 ;;
esac

# 2026-10-04: a stale acknowledgement must not satisfy this restoration's watchdog (see the end of the script)
rm -f "$session_root/host-ack"

if [ -f "$session_root/worker.pid" ]; then
    worker_pid=$(cat "$session_root/worker.pid" 2>/dev/null)
    case "$worker_pid" in
        *[!0-9]*|'') ;;
        *) kill -TERM "$worker_pid" 2>/dev/null ;;
    esac
fi

attempt=0
while [ -e "$g2/UDC" ] && [ -n "$(cat "$g2/UDC" 2>/dev/null)" ] \
        && [ "$attempt" -lt 100 ]; do
    printf '\n' > "$g2/UDC" 2>/dev/null
    attempt=$((attempt + 1))
    sleep 0.05
done
rm -f "$g2/configs/b.1/f1"
rm -f "$g2/configs/b.1/f2"
umount "$ffs_root" 2>/dev/null
rmdir "$ffs_root" 2>/dev/null
rmdir "$g2/functions/ffs.s41" 2>/dev/null
rmdir "$g2/functions/ncm.usb0" 2>/dev/null

setprop sys.usb.config "$normal_usb_config"
attempt=0
while [ -e "$g1/UDC" ] && [ -z "$(cat "$g1/UDC" 2>/dev/null)" ] \
        && [ "$attempt" -lt 200 ]; do
    attempt=$((attempt + 1))
    sleep 0.05
done
rm -f "$session_root/active"

if [ ! -e "$g1/UDC" ] || [ -z "$(cat "$g1/UDC" 2>/dev/null)" ]; then
    exit 1
fi

# adbd watchdog (2026-10-04, fault run p6h2flt): after the gadget went back to Android the ADB interface was
# enumerated but adbd never answered, a replug did not revive it, and the rig waited for a human. The host
# creates host-ack over adb as soon as `adb get-state` says "device"; without it adbd is restarted, then the
# whole USB configuration is cycled once. A new session (active) ends the watch. Log: restore-watchdog.log.
ack_timeout=${S41_RESTORE_ACK_TIMEOUT_S:-75}
(
    log="$session_root/restore-watchdog.log"
    wait_for_ack() {
        waited=0
        while :; do
            if [ -f "$session_root/host-ack" ]; then
                echo "$(date +%s) host-ack after ${waited}s ($1)" >> "$log"
                return 0
            fi
            if [ -f "$session_root/active" ]; then
                echo "$(date +%s) new session active: watch ends ($1)" >> "$log"
                return 0
            fi
            [ "$waited" -ge "$ack_timeout" ] && return 1
            sleep 1
            waited=$((waited + 1))
        done
    }
    wait_for_ack "after restore" && exit 0
    echo "$(date +%s) no host-ack after ${ack_timeout}s: adbd=$(getprop init.svc.adbd) usb=$(getprop sys.usb.state); restarting adbd" >> "$log"
    setprop ctl.restart adbd
    wait_for_ack "after adbd restart" && exit 0
    echo "$(date +%s) still no host-ack: cycling the USB configuration" >> "$log"
    setprop sys.usb.config none
    sleep 2
    setprop sys.usb.config "$normal_usb_config"
    sleep 3
    setprop ctl.restart adbd
    wait_for_ack "after USB config cycle" && exit 0
    echo "$(date +%s) gave up: adbd=$(getprop init.svc.adbd) usb=$(getprop sys.usb.state)" >> "$log"
) > /dev/null 2>&1 &
exit 0
