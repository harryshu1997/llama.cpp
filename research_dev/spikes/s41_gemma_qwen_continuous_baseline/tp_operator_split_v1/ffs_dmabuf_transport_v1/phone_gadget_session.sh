#!/system/bin/sh
set -eu

if [ "$#" -ne 10 ]; then
    echo "usage: $0 <binary> <copy|dmabuf> <request> <response> <total> <warmup> <depth> <session-root> <restore-script> <timeout-seconds>" >&2
    exit 2
fi

binary=$1
mode=$2
request_bytes=$3
response_bytes=$4
total=$5
warmup=$6
depth=$7
session_root=$8
restore_script=$9
timeout_seconds=${10}

g1=/config/usb_gadget/g1
g2=/config/usb_gadget/g2
udc=a600000.dwc3
ffs_root=/dev/usb-ffs/s41
ready_file=$session_root/descriptors.ready
worker_log=$session_root/worker.log

mkdir -p "$session_root"
rm -f "$ready_file"

worker_pid=
watchdog_pid=
cleanup() {
    rm -f "$session_root/active"
    if [ -n "$watchdog_pid" ]; then
        kill "$watchdog_pid" 2>/dev/null || true
    fi
    sh "$restore_script" "$session_root" || true
}
trap cleanup EXIT INT TERM
trap '' HUP

sh "$restore_script" "$session_root" || true
: > "$session_root/active"
(
    sleep "$timeout_seconds"
    if [ -f "$session_root/active" ]; then
        echo "[ffs-session] watchdog restoring USB" >> "$session_root/session.log"
        sh "$restore_script" "$session_root"
    fi
) &
watchdog_pid=$!

mkdir -p "$ffs_root"
mkdir "$g2/functions/ffs.s41"
mount -t functionfs s41 "$ffs_root"

"$binary" "$mode" "$ffs_root" "$request_bytes" "$response_bytes" \
    "$total" "$warmup" "$depth" "$ready_file" > "$worker_log" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$session_root/worker.pid"

ready=0
attempt=0
while [ "$attempt" -lt 200 ]; do
    if [ -f "$ready_file" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.05
done
if [ "$ready" -ne 1 ]; then
    echo "[ffs-session] worker did not publish descriptors" >&2
    cat "$worker_log" >&2 || true
    exit 1
fi

if [ -n "$(cat "$g2/UDC" 2>/dev/null)" ]; then
    printf '\n' > "$g2/UDC"
fi
rm -f "$g2/configs/b.1/f1"
printf '0x18d1' > "$g2/idVendor"
printf '0x2d00' > "$g2/idProduct"
printf '0x0320' > "$g2/bcdUSB"
printf '0x0100' > "$g2/bcdDevice"
printf '500' > "$g2/configs/b.1/MaxPower"
printf '0x80' > "$g2/configs/b.1/bmAttributes"
printf 'S41 FunctionFS' > "$g2/strings/0x409/manufacturer"
printf 'S41 DMA-BUF transport' > "$g2/strings/0x409/product"
printf 'S41FFS0001' > "$g2/strings/0x409/serialnumber"
printf 's41_ffs_dmabuf' > "$g2/configs/b.1/strings/0x409/configuration"
ln -s "$g2/functions/ffs.s41" "$g2/configs/b.1/f1"

printf '\n' > "$g1/UDC"
printf '%s' "$udc" > "$g2/UDC"
echo "[ffs-session] custom gadget bound mode=$mode" >> "$session_root/session.log"

set +e
wait "$worker_pid"
worker_status=$?
set -e
worker_pid=
echo "[ffs-session] worker_status=$worker_status" >> "$session_root/session.log"
if [ "${S42_ENERGY_TAIL_SECONDS:-0}" -gt 0 ]; then
    sleep "$S42_ENERGY_TAIL_SECONDS"
fi
exit "$worker_status"
