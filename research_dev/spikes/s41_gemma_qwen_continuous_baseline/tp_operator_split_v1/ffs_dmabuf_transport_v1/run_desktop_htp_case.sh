#!/bin/bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: $0 <malloc|devmem> <elements> <warmup> <iterations> <case-name> <output-root>" >&2
    exit 2
fi

allocator=$1
elements=$2
warmup=$3
iterations=$4
case_name=$5
output_root=$6
device_mode=${S41_HTP_DEVICE_MODE:-htp-sqr}

serial=${S41_SERIAL:-3C15AU002CL00000}
stage_root=${S41_STAGE_ROOT:-/home/zhihao/s41-ffs-dmabuf-fixed-v1}
phone_root=${S41_HTP_PHONE_ROOT:-/data/local/tmp/s41-ffs-dmabuf-htp-v1}
session_root=$phone_root/$case_name
host_result=$output_root/$case_name.json

case "$allocator" in
    malloc|devmem) ;;
    *) echo "invalid allocator" >&2; exit 2 ;;
esac
case "$device_mode" in
    htp-sqr|htp-copy-sqr|htp-staged-sqr) ;;
    *) echo "invalid HTP device mode" >&2; exit 2 ;;
esac
case "$elements:$warmup:$iterations" in
    *[!0-9:]*|*::*|:*) echo "invalid numeric argument" >&2; exit 2 ;;
esac
case "$case_name" in
    *[!A-Za-z0-9_.-]*|'') echo "invalid case name" >&2; exit 2 ;;
esac
if [ -e "$host_result" ]; then
    echo "result already exists: $host_result" >&2
    exit 2
fi

mkdir -p "$output_root"
total=$((warmup + iterations))
bytes=$((elements * 4))

wait_for_adb() {
    timeout 60 adb -s "$serial" wait-for-device </dev/null
}

restore_on_error() {
    status=$?
    trap - EXIT
    if [ "$status" -eq 0 ]; then
        return
    fi
    if wait_for_adb >/dev/null 2>&1; then
        adb -s "$serial" shell \
            "su -c 'sh $phone_root/restore_phone_usb.sh $session_root'" \
            </dev/null >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap restore_on_error EXIT

adb -s "$serial" shell \
    "su -c 'rm -rf $session_root; mkdir -p $session_root; nohup sh $phone_root/phone_gadget_session.sh $phone_root/ffs_dmabuf_htp_launcher.sh $device_mode $bytes $bytes $total $warmup 1 $session_root $phone_root/restore_phone_usb.sh 60 > $session_root/launch.log 2>&1 < /dev/null &'" \
    </dev/null

enumerated=0
for unused in $(seq 1 300); do
    if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
        enumerated=1
        break
    fi
    sleep 0.1
done
if [ "$enumerated" -ne 1 ]; then
    echo "HTP FunctionFS gadget did not enumerate" >&2
    exit 1
fi

"$stage_root/ffs_dmabuf_htp_host" "$allocator" "$elements" \
    "$warmup" "$iterations" "$host_result"

wait_for_adb
adb -s "$serial" exec-out \
    "su -c 'cat $session_root/worker.log'" \
    </dev/null > "$output_root/$case_name.phone.log"
adb -s "$serial" exec-out \
    "su -c 'cat $session_root/session.log'" \
    </dev/null > "$output_root/$case_name.session.log"
adb -s "$serial" exec-out \
    "su -c \"dmesg | grep -E 'arm-smmu.*fault|context fault|Kernel panic|FFS: Failed' | tail -100 || true\"" \
    </dev/null > "$output_root/$case_name.kernel_faults.log"
adb -s "$serial" exec-out \
    "su -c \"cat /config/usb_gadget/g1/UDC; cat /config/usb_gadget/g2/UDC; cat /sys/class/udc/a600000.dwc3/current_speed; ps -A | grep -E 'ffs_dmabuf|phone_gadget' || true\"" \
    </dev/null > "$output_root/$case_name.terminal.log"

grep -Eq "complete( mode=[^ ]+)? paid=$iterations" \
    "$output_root/$case_name.phone.log"
grep -q 'worker_status=0' "$output_root/$case_name.session.log"
test ! -s "$output_root/$case_name.kernel_faults.log"
test "$(sed -n '1p' "$output_root/$case_name.terminal.log" | tr -d '\r')" = a600000.dwc3
test -z "$(sed -n '2p' "$output_root/$case_name.terminal.log" | tr -d '\r')"
test "$(sed -n '3p' "$output_root/$case_name.terminal.log" | tr -d '\r')" = super-speed

trap - EXIT
