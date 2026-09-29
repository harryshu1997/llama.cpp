#!/bin/sh
set -eu

serial=${S41_PHONE_SERIAL:-3C15AU002CL00000}
phone_binary=${S41_PHONE_BINARY:-/data/local/tmp/s41-aoa-async-v1/aoa_buffered_daemon}
script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

printf 'captured_utc='
date -u +%Y-%m-%dT%H:%M:%SZ
printf 'host='
hostname
printf 'kernel='
uname -srvo
printf 'gpu='
nvidia-smi -i 0 --query-gpu=name,uuid,driver_version \
    --format=csv,noheader
printf 'libusb_package='
dpkg-query -W -f='${Version}\n' libusb-1.0-0
printf 'usb_device='
lsusb | grep -E '(18d1:2d0(0|1|4|5)|22d9:2769|22d9:2772|05c6:908c)'
printf 'usb_tree_begin\n'
lsusb -t
printf 'usb_tree_end\n'
printf 'adb_state='
adb -s "$serial" get-state
printf 'phone_serial=%s\n' "$serial"
printf 'phone_boot_id='
adb -s "$serial" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r'
printf 'phone_product='
adb -s "$serial" shell getprop ro.product.model | tr -d '\r'
printf 'phone_build='
adb -s "$serial" shell getprop ro.build.fingerprint | tr -d '\r'
printf 'phone_usb_config='
adb -s "$serial" shell getprop sys.usb.config | tr -d '\r'
printf 'phone_usb_state='
adb -s "$serial" shell getprop sys.usb.state | tr -d '\r'
printf 'phone_usb_speed='
adb -s "$serial" shell \
    "su -c 'cat /sys/class/udc/a600000.dwc3/current_speed'" | tr -d '\r'
printf 'phone_worker_pid='
adb -s "$serial" shell pidof aoa_buffered_daemon 2>/dev/null | tr -d '\r' || true
printf '\n'
printf 'phone_binary_sha256='
adb -s "$serial" shell sha256sum "$phone_binary" | tr -d '\r'
printf 'host_artifacts_begin\n'
sha256sum "$script_root/aoa_async_host" \
    "$script_root/aoa_buffered_daemon.android" \
    "$script_root/run_case.sh" "$script_root/run_campaign.sh"
printf 'host_artifacts_end\n'
