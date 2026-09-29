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
exit 0
