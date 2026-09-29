#!/bin/sh
set -eu

if [ "$#" -ne 8 ]; then
    echo "usage: $0 <serial|buffered> <sync|async> <queue-depth> <request-bytes> <response-bytes> <warmup> <iterations> <result-dir>" >&2
    exit 2
fi

phone_mode=$1
host_mode=$2
queue_depth=$3
request_bytes=$4
response_bytes=$5
warmup=$6
iterations=$7
result_dir=$8

case "$phone_mode:$host_mode" in
    serial:sync|buffered:sync|buffered:async) ;;
    *)
        echo "invalid phone/host mode" >&2
        exit 2
        ;;
esac
case "$host_mode:$queue_depth" in
    sync:1|async:*) ;;
    *)
        echo "sync mode requires queue depth 1" >&2
        exit 2
        ;;
esac

serial=${S41_PHONE_SERIAL:-3C15AU002CL00000}
phone_root=${S41_PHONE_ROOT:-/data/local/tmp/s41-aoa-async-v1}
phone_binary=${S41_PHONE_BINARY:-$phone_root/aoa_buffered_daemon}
script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
host_binary=${S41_HOST_BINARY:-$script_root/aoa_async_host}
aoa_switch=${S41_AOA_SWITCH:-$script_root/../aoa_bench.py}
keep_accessory=${S41_KEEP_ACCESSORY:-1}
fresh_accessory=${S41_FRESH_ACCESSORY:-0}
total_requests=$((warmup + iterations))
case_name=${S41_CASE_NAME:-${phone_mode}_${host_mode}_q${queue_depth}_${request_bytes}_${response_bytes}}
phone_log=$phone_root/$case_name.worker.log
mkdir -p "$result_dir"

worker_transport_pid=
worker_device_pid=
wait_for_adb() {
    wait_attempt=0
    while [ "$wait_attempt" -lt 100 ]; do
        if [ "$(adb -s "$serial" get-state 2>/dev/null || true)" = "device" ]; then
            return 0
        fi
        wait_attempt=$((wait_attempt + 1))
        sleep 0.1
    done
    return 1
}

reset_accessory() {
    sleep 0.5
    reset_attempt=0
    while [ "$reset_attempt" -lt 3 ]; do
        if ! lsusb | grep -Eq '18d1:2d0(0|1|4|5)'; then
            wait_for_adb
            return $?
        fi
        python3 "$aoa_switch" reset >/dev/null 2>&1 || true
        settle_attempt=0
        stable_normal=0
        while [ "$settle_attempt" -lt 80 ]; do
            if lsusb | grep -Eq '(22d9:2769|22d9:2772|05c6:908c)'; then
                stable_normal=$((stable_normal + 1))
                if [ "$stable_normal" -ge 20 ]; then
                    wait_for_adb
                    return $?
                fi
            else
                stable_normal=0
            fi
            settle_attempt=$((settle_attempt + 1))
            sleep 0.1
        done
        reset_attempt=$((reset_attempt + 1))
    done
    return 1
}

cleanup() {
    if [ -n "$worker_transport_pid" ]; then
        kill "$worker_transport_pid" >/dev/null 2>&1 || true
    fi
    case "$worker_device_pid" in
        *[!0-9]*|'') ;;
        *) adb -s "$serial" shell "su -c 'kill -9 $worker_device_pid'" \
            >/dev/null 2>&1 || true ;;
    esac
    if [ "$keep_accessory" -ne 1 ]; then
        reset_accessory || true
    fi
}
trap cleanup EXIT INT TERM

if [ "$(adb -s "$serial" get-state 2>/dev/null || true)" != "device" ]; then
    echo "phone is not available through adb" >&2
    exit 3
fi
if [ ! -x "$host_binary" ]; then
    echo "host binary is missing: $host_binary" >&2
    exit 3
fi
if [ "$fresh_accessory" -eq 1 ] &&
        lsusb | grep -Eq '18d1:2d0(0|1|4|5)'; then
    if ! reset_accessory; then
        echo "failed to refresh the AOA enumeration" >&2
        exit 3
    fi
fi

adb -s "$serial" shell "su -c 'mkdir -p $phone_root; : > $phone_log'"
adb -s "$serial" shell \
    "su -c '$phone_binary $phone_mode $request_bytes $response_bytes $total_requests $warmup $queue_depth > $phone_log 2>&1'" &
worker_transport_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 50 ]; do
    if adb -s "$serial" shell cat "$phone_log" 2>/dev/null |
            grep -q '\[aoa-buffer\] configured'; then
        ready=1
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    adb -s "$serial" shell cat "$phone_log" >&2 || true
    echo "phone worker did not configure" >&2
    exit 3
fi
worker_device_pid=$(adb -s "$serial" shell pidof aoa_buffered_daemon \
    2>/dev/null | tr -d '\r')

if ! lsusb | grep -Eq '18d1:2d0(0|1|4|5)'; then
    switched=0
    for normal_device in 22d9:2769 22d9:2772 05c6:908c; do
        normal_vid=${normal_device%:*}
        normal_pid=${normal_device#*:}
        if python3 "$aoa_switch" --vid "$normal_vid" --pid "$normal_pid" \
                switch; then
            switched=1
            break
        fi
    done
    if [ "$switched" -ne 1 ]; then
        echo "phone did not enter AOA mode" >&2
        exit 3
    fi
fi

endpoint_ready=0
attempt=0
while [ "$attempt" -lt 100 ]; do
    if adb -s "$serial" shell cat "$phone_log" 2>/dev/null |
            grep -q '\[aoa-buffer\] endpoint open'; then
        endpoint_ready=1
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done
if [ "$endpoint_ready" -ne 1 ]; then
    adb -s "$serial" shell cat "$phone_log" >&2 || true
    echo "phone worker did not open the AOA endpoint" >&2
    exit 3
fi

S41_PHONE_MODE=$phone_mode S41_CASE_NAME=$case_name \
S41_WORKLOAD=${S41_WORKLOAD:-unknown} S41_REPETITION=${S41_REPETITION:-0} \
"$host_binary" "$host_mode" "$request_bytes" "$response_bytes" \
    "$warmup" "$iterations" "$queue_depth" \
    "$result_dir/$case_name.json"

wait "$worker_transport_pid"
worker_transport_pid=
worker_device_pid=
adb -s "$serial" pull "$phone_log" \
    "$result_dir/$case_name.worker.log" >/dev/null

trap - EXIT INT TERM
if [ "$keep_accessory" -ne 1 ]; then
    if ! reset_accessory; then
        echo "failed to leave AOA mode" >&2
        exit 3
    fi
fi
