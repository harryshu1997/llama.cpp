#!/bin/bash
set -euo pipefail

adb() { /usr/bin/adb -P 5037 "$@"; }

if [ "$#" -ne 10 ]; then
    echo "usage: $0 <copy|dmabuf> <sync|async> <malloc|devmem> <request> <response> <warmup> <iterations> <depth> <case-name> <output-root>" >&2
    exit 2
fi

device_mode=$1
host_mode=$2
allocator=$3
request_bytes=$4
response_bytes=$5
warmup=$6
iterations=$7
depth=$8
case_name=$9
output_root=${10}

serial=${S41_SERIAL:-3C15AU002CL00000}
stage_root=${S41_STAGE_ROOT:-/home/zhihao/s41-ffs-dmabuf-fixed-v1}
phone_root=${S41_PHONE_ROOT:-/data/local/tmp/s41-ffs-dmabuf-v1}
phone_session=${S41_PHONE_SESSION:?canonical qualification session script required}
phone_restore=${S41_PHONE_RESTORE:?canonical restore script required}
android_gadget=${S41_ANDROID_GADGET:-/config/usb_gadget/g1}
functionfs_gadget=${S41_FUNCTIONFS_GADGET:-/config/usb_gadget/g2}
functionfs_root=${S41_FUNCTIONFS_ROOT:-/dev/usb-ffs/s41}
phone_udc=${S41_PHONE_UDC:-a600000.dwc3}
session_root=$phone_root/$case_name
host_result=$output_root/$case_name.json

case "$device_mode:$host_mode:$allocator" in
    copy:sync:malloc|dmabuf:sync:malloc|dmabuf:sync:devmem|dmabuf:async:malloc|dmabuf:async:devmem) ;;
    *) echo "unsupported transport combination" >&2; exit 2 ;;
esac
case "$request_bytes:$response_bytes:$warmup:$iterations:$depth" in
    *[!0-9:]*|*::*|:*) echo "invalid numeric argument" >&2; exit 2 ;;
esac
case "$case_name" in
    *[!A-Za-z0-9_.-]*|'') echo "invalid case name" >&2; exit 2 ;;
esac
if [ "$host_mode" = sync ] && [ "$depth" -ne 1 ]; then
    echo "sync mode requires depth 1" >&2
    exit 2
fi
if [ -e "$host_result" ]; then
    echo "result already exists: $host_result" >&2
    exit 2
fi

mkdir -p "$output_root"
total=$((warmup + iterations))

wait_for_adb() {
    timeout 60 adb -s "$serial" wait-for-device </dev/null
}

restore_on_error() {
    status=$?
    trap - EXIT
    if [ "$status" -eq 0 ]; then
        return
    fi
    echo "transport failed; preserve the session and its managed watchdog, no host reset" >&2
    exit "$status"
}
trap restore_on_error EXIT

adb -s "$serial" shell \
    "su -c 'test ! -e $session_root && mkdir $session_root && nohup env S42_ENERGY_TAIL_SECONDS=1 SCHEDULER_ANDROID_GADGET=$android_gadget SCHEDULER_FUNCTIONFS_GADGET=$functionfs_gadget SCHEDULER_FUNCTIONFS_ROOT=$functionfs_root SCHEDULER_PHONE_UDC=$phone_udc sh $phone_session $phone_root/ffs_dmabuf_phone.android $device_mode $request_bytes $response_bytes $total $warmup $depth $session_root $phone_restore 45 > $session_root/launch.log 2>&1 < /dev/null &'" \
    </dev/null

enumerated=0
for unused in $(seq 1 100); do
    if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
        enumerated=1
        break
    fi
    sleep 0.1
done
if [ "$enumerated" -ne 1 ]; then
    echo "custom FunctionFS gadget did not enumerate" >&2
    exit 1
fi
sleep 0.25

host_log=$output_root/$case_name.host.log
if ! S41_DEVICE_MODE=$device_mode S41_CASE_NAME=$case_name \
    "$stage_root/ffs_dmabuf_host" "$host_mode" "$allocator" \
    "$request_bytes" "$response_bytes" "$warmup" "$iterations" \
    "$depth" "$host_result" > "$host_log" 2>&1; then
    cat "$host_log" >&2
    exit 1
fi
cat "$host_log"

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

grep -q 'complete status=0' "$output_root/$case_name.phone.log"
grep -q 'worker_status=0' "$output_root/$case_name.session.log"
test ! -s "$output_root/$case_name.kernel_faults.log"
test "$(sed -n '1p' "$output_root/$case_name.terminal.log" | tr -d '\r')" = a600000.dwc3
test -z "$(sed -n '2p' "$output_root/$case_name.terminal.log" | tr -d '\r')"
test "$(sed -n '3p' "$output_root/$case_name.terminal.log" | tr -d '\r')" = super-speed

trap - EXIT
