#!/system/bin/sh
# Stop the known, temporary diagnostic controller without relying on the
# textual spelling of configfs symlinks (configfs normalizes them to relative).
# Run detached with su -mm so ADB re-enumeration cannot interrupt restoration.
set -eu
session=/data/local/tmp/moe_ffs_20260909
gadget=/config/usb_gadget/g1
link=$gadget/configs/b.1/moe_root_20260909
function=$gadget/functions/ffs.moe_root_20260909
mount_path=/dev/usb-ffs/moe_root_20260909
[ "$(id -u)" = 0 ]
[ "$(getprop ro.serialno)" = 5ae7a43d ]
[ "$(getprop ro.product.model)" = CPH2583 ]
[ "$(getprop sys.usb.config)" = adb ]
[ "$(cat "$gadget/UDC")" = a600000.dwc3 ]
[ "$(readlink -f "$gadget/configs/b.1/f1")" = "$gadget/functions/ffs.adb" ]
[ "$(readlink -f "$link")" = "$function" ]
controller_pid=$(cat "$session/controller.pid")
case "$controller_pid" in ''|*[!0-9]*) exit 2;; esac
[ "$controller_pid" -gt 1 ]
controller_command=$(tr '\0' ' ' < "/proc/$controller_pid/cmdline")
case "$controller_command" in
    *"/system/bin/sh /data/local/tmp/moe_ffs_20260909/controller.sh"*) ;;
    *) printf 'Controller identity differs; refusing to signal it\n' >&2; exit 2;;
esac

# Pause only our verified controller while unlinking only our MoE function.
# Its existing EXIT trap then stops its child service and restores ADB.
trap 'kill -CONT "$controller_pid" 2>/dev/null || true' EXIT
kill -STOP "$controller_pid"
printf '\n' > "$gadget/UDC"
rm "$link"
kill -TERM "$controller_pid"
kill -CONT "$controller_pid"
attempt=0
while [ "$attempt" -lt 20 ]; do
    if [ "$(cat "$gadget/UDC")" = a600000.dwc3 ] &&
       [ ! -L "$link" ] && [ ! -e "$function" ] && [ ! -e "$mount_path" ]; then
        printf 'RESTORATION_VERIFIED=1\n'
        printf 'USB_CONFIG=%s\n' "$(getprop sys.usb.config)"
        printf 'USB_STATE=%s\n' "$(getprop sys.usb.state)"
        printf 'SELINUX=%s\n' "$(getenforce)"
        exit 0
    fi
    attempt=$((attempt + 1))
    sleep 1
done
printf 'RESTORATION_NOT_VERIFIED=1\n' >&2
exit 2
