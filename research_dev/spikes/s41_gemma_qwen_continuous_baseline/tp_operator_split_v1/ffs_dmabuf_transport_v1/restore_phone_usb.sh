#!/system/bin/sh
set +e

session_root=$1
g1=/config/usb_gadget/g1
g2=/config/usb_gadget/g2
udc=a600000.dwc3
ffs_root=/dev/usb-ffs/s41

if [ -f "$session_root/worker.pid" ]; then
    worker_pid=$(cat "$session_root/worker.pid" 2>/dev/null)
    case "$worker_pid" in
        *[!0-9]*|'') ;;
        *) kill -TERM "$worker_pid" 2>/dev/null ;;
    esac
fi

if [ -e "$g2/UDC" ]; then
    printf '\n' > "$g2/UDC" 2>/dev/null
fi
sleep 0.1
ip -6 rule del priority 9999 from fe80::2/128 to fe80::/64 lookup 1033 \
    2>/dev/null
ip -6 route del fe80::/64 dev usb0 table 1033 2>/dev/null
rm -f "$g2/configs/b.1/f1"
rm -f "$g2/configs/b.1/f2"
rmdir "$g2/functions/ffs.s41" 2>/dev/null
rmdir "$g2/functions/ncm.usb0" 2>/dev/null
umount "$ffs_root" 2>/dev/null
rmdir "$ffs_root" 2>/dev/null

if [ -e "$g1/UDC" ] && [ -z "$(cat "$g1/UDC" 2>/dev/null)" ]; then
    printf '%s' "$udc" > "$g1/UDC" 2>/dev/null
fi
rm -f "$session_root/active"
