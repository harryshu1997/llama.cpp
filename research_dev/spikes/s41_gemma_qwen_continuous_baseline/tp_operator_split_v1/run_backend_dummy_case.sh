#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <HTP|HTPFast|OpenCL|OpenCLFast|OpenCLRebuilt|OpenCLRebuiltFast|OpenCLFlush|OpenCLFlushFast|OpenCLDoorbellDispatch|OpenCLPersistent> <noop|sqr> <repeats> <result-dir>" >&2
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
        worker_backend=HTP
        backend_env="GGML_HEXAGON_PROFILE=1"
        worker_executable=backend_dummy_worker
        ready_pattern='\[dummy-worker\] ready'
        profile_artifacts=0
        ;;
    HTPFast)
        runtime=htp
        worker_backend=HTP
        backend_env=""
        worker_executable=backend_dummy_worker
        ready_pattern='\[dummy-worker\] ready'
        profile_artifacts=0
        ;;
    OpenCL)
        runtime=opencl
        worker_backend=OpenCL
        backend_env=""
        worker_executable=backend_dummy_worker
        ready_pattern='\[dummy-worker\] ready'
        profile_artifacts=1
        ;;
    OpenCLFast)
        runtime=htp
        worker_backend=OpenCL
        backend_env=""
        worker_executable=backend_dummy_worker
        ready_pattern='\[dummy-worker\] ready'
        profile_artifacts=0
        ;;
    OpenCLRebuilt)
        runtime=opencl_flush_profile
        worker_backend=OpenCL
        backend_env=""
        worker_executable=backend_dummy_worker
        ready_pattern='\[dummy-worker\] ready'
        profile_artifacts=1
        ;;
    OpenCLRebuiltFast)
        runtime=opencl_flush
        worker_backend=OpenCL
        backend_env=""
        worker_executable=backend_dummy_worker
        ready_pattern='\[dummy-worker\] ready'
        profile_artifacts=0
        ;;
    OpenCLFlush)
        runtime=opencl_flush_profile
        worker_backend=OpenCL
        backend_env="GGML_OPENCL_EARLY_FLUSH=1"
        worker_executable=backend_dummy_worker
        ready_pattern='\[dummy-worker\] ready'
        profile_artifacts=1
        ;;
    OpenCLFlushFast)
        runtime=opencl_flush
        worker_backend=OpenCL
        backend_env="GGML_OPENCL_EARLY_FLUSH=1"
        worker_executable=backend_dummy_worker
        ready_pattern='\[dummy-worker\] ready'
        profile_artifacts=0
        ;;
    OpenCLPersistent)
        runtime=opencl_persistent
        worker_backend=OpenCLPersistent
        backend_env=""
        worker_executable=opencl_persistent_dummy_worker
        ready_pattern='\[persistent-worker\] ready'
        profile_artifacts=0
        ;;
    OpenCLDoorbellDispatch)
        runtime=opencl_persistent
        worker_backend=OpenCLDoorbellDispatch
        backend_env=""
        worker_executable=opencl_persistent_dummy_worker
        ready_pattern='\[persistent-worker\] ready'
        profile_artifacts=0
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
    "su -c ': > $phone_log'"
if [ "$profile_artifacts" -eq 1 ]; then
    ADB_SERVER_PORT=$adb_port adb -s "$serial" shell \
        "su -c 'rm -f $phone_root/cl_profiling.csv $phone_root/cl_trace.json'"
fi
ADB_SERVER_PORT=$adb_port adb -s "$serial" shell \
    "su -c 'cd $phone_root && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. $backend_env ./$worker_executable $elements $worker_backend $op $repeats $requests > $case_name.log 2>&1'" &
worker_transport_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 30 ]; do
    if ADB_SERVER_PORT=$adb_port adb -s "$serial" shell cat "$phone_log" \
            2>/dev/null | grep -q "$ready_pattern"; then
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
    pidof "$worker_executable" 2>/dev/null | tr -d '\r')

switched=0
attempt=0
while [ "$attempt" -lt 60 ]; do
    if lsusb | grep -Eq '18d1:2d0(0|1|4|5)'; then
        switched=1
        break
    fi
    for normal_pid in 2769 2772; do
        if python3 "$desktop_root/aoa_bench.py" \
                --vid 22d9 --pid "$normal_pid" switch; then
            switched=1
            break
        fi
    done
    if [ "$switched" -eq 1 ]; then
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
trap - EXIT INT TERM
device_ready=0
attempt=0
while [ "$attempt" -lt 5 ]; do
    python3 "$desktop_root/aoa_bench.py" reset || true
    sleep 3
    if [ "$(ADB_SERVER_PORT=$adb_port adb -s "$serial" get-state \
            2>/dev/null || true)" = "device" ]; then
        device_ready=1
        break
    fi
    attempt=$((attempt + 1))
done
if [ "$device_ready" -ne 1 ]; then
    echo "phone did not return to adb after AOA reset" >&2
    exit 3
fi
ADB_SERVER_PORT=$adb_port adb -s "$serial" pull \
    "$phone_log" "$result_dir/$case_name.worker.log" >/dev/null
if [ "$profile_artifacts" -eq 1 ]; then
    if ADB_SERVER_PORT=$adb_port adb -s "$serial" shell \
            test -f "$phone_root/cl_profiling.csv"; then
        ADB_SERVER_PORT=$adb_port adb -s "$serial" pull \
            "$phone_root/cl_profiling.csv" \
            "$result_dir/$case_name.cl_profiling.csv" >/dev/null
        ADB_SERVER_PORT=$adb_port adb -s "$serial" pull \
            "$phone_root/cl_trace.json" \
            "$result_dir/$case_name.cl_trace.json" >/dev/null
    fi
fi
