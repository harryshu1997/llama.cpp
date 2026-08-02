#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <HTP|OpenCL> <noop|sqr> <repeats> <result-dir>" >&2
    exit 2
fi

backend=$1
op=$2
repeats=$3
result_dir=$4
serial=${S41_PHONE_SERIAL:-3C15AU002CL00000}
adb_port=${ADB_SERVER_PORT:-5037}
elements=${S41_DUMMY_ELEMENTS:-2816}
warmup=${S41_DUMMY_WARMUP:-50}
iterations=${S41_DUMMY_ITERS:-600}
requests=$((warmup + iterations))
case_name=$(printf '%s_%s_%s' "$backend" "$op" "$repeats" |
    tr '[:upper:]' '[:lower:]')

case "$backend" in
    HTP)
        runtime=htp
        backend_env="GGML_HEXAGON_PROFILE=1"
        ;;
    OpenCL)
        runtime=opencl
        backend_env=""
        ;;
    *)
        echo "unsupported backend: $backend" >&2
        exit 2
        ;;
esac

phone_root=/data/local/tmp/s41-backend-dummy-v1/$runtime
phone_log=$phone_root/$case_name.log
desktop_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$result_dir"

cleanup() {
    if [ -n "${worker_transport_pid:-}" ]; then
        kill "$worker_transport_pid" >/dev/null 2>&1 || true
    fi
    case "${worker_device_pid:-}" in
        *[!0-9]*|'') ;;
        *)
            ADB_SERVER_PORT=$adb_port adb -s "$serial" shell \
                "su -c 'kill -9 $worker_device_pid'" \
                >/dev/null 2>&1 || true
            ;;
    esac
    python3 "$desktop_root/aoa_bench.py" reset >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

device_ready=0
attempt=0
while [ "$attempt" -lt 30 ]; do
    if [ "$(ADB_SERVER_PORT=$adb_port adb -s "$serial" get-state \
            2>/dev/null || true)" = "device" ]; then
        device_ready=1
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done
if [ "$device_ready" -ne 1 ]; then
    echo "phone is not available through adb" >&2
    exit 3
fi
ADB_SERVER_PORT=$adb_port adb -s "$serial" shell \
    "su -c 'cd $phone_root && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. $backend_env ./backend_dummy_worker $elements $backend $op $repeats $requests > $case_name.log 2>&1'" &
worker_transport_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 30 ]; do
    if ADB_SERVER_PORT=$adb_port adb -s "$serial" shell cat "$phone_log" \
            2>/dev/null | grep -q '\[dummy-worker\] ready'; then
        ready=1
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    ADB_SERVER_PORT=$adb_port adb -s "$serial" shell cat "$phone_log" >&2 || true
    echo "worker did not become ready" >&2
    exit 3
fi
worker_device_pid=$(ADB_SERVER_PORT=$adb_port adb -s "$serial" shell \
    pidof backend_dummy_worker 2>/dev/null | tr -d '\r')

switched=0
attempt=0
while [ "$attempt" -lt 5 ]; do
    if python3 "$desktop_root/aoa_bench.py" \
            --vid 22d9 --pid 2769 switch; then
        switched=1
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done
if [ "$switched" -ne 1 ]; then
    echo "phone did not enter AOA mode" >&2
    exit 3
fi
sleep 2
python3 "$desktop_root/backend_dummy_bench.py" \
    --elements "$elements" --warmup "$warmup" --iters "$iterations" \
    --backend "$backend" --op "$op" --repeats "$repeats" \
    --output "$result_dir/$case_name.json"

wait "$worker_transport_pid"
worker_transport_pid=
worker_device_pid=
python3 "$desktop_root/aoa_bench.py" reset
trap - EXIT INT TERM
device_ready=0
attempt=0
while [ "$attempt" -lt 30 ]; do
    if [ "$(ADB_SERVER_PORT=$adb_port adb -s "$serial" get-state \
            2>/dev/null || true)" = "device" ]; then
        device_ready=1
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done
if [ "$device_ready" -ne 1 ]; then
    echo "phone did not return to adb after AOA reset" >&2
    exit 3
fi
ADB_SERVER_PORT=$adb_port adb -s "$serial" pull \
    "$phone_log" "$result_dir/$case_name.worker.log" >/dev/null
if [ "$backend" = "OpenCL" ]; then
    ADB_SERVER_PORT=$adb_port adb -s "$serial" pull \
        "$phone_root/cl_profiling.csv" \
        "$result_dir/$case_name.cl_profiling.csv" >/dev/null
    ADB_SERVER_PORT=$adb_port adb -s "$serial" pull \
        "$phone_root/cl_trace.json" \
        "$result_dir/$case_name.cl_trace.json" >/dev/null
fi
