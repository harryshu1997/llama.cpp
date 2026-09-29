#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <HTPGraph|OpenCLGraph|OpenCLDispatch|OpenCLPersistent> <rmsnorm|swiglu|attention> <n_kv> <result-dir>" >&2
    exit 2
fi

backend=$1
op=$2
n_kv=$3
result_dir=$4
serial=${S41_PHONE_SERIAL:-3C15AU002CL00000}
adb_port=${ADB_SERVER_PORT:-5037}
warmup=${S41_MODEL_OP_WARMUP:-50}
iterations=${S41_MODEL_OP_ITERS:-300}
variants=${S41_MODEL_OP_VARIANTS:-4}
timeout_ms=${S41_MODEL_OP_TIMEOUT_MS:-60000}
requests=$((warmup + iterations))
desktop_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
reference_dir=${S41_MODEL_OP_REFERENCE_DIR:-$desktop_root/references}
reference=$reference_dir/${op}_kv${n_kv}_v${variants}.json
case_name=$(printf '%s_%s_kv%s' "$backend" "$op" "$n_kv" |
    tr '[:upper:]' '[:lower:]')

case "$backend" in
    HTPGraph)
        phone_root=/data/local/tmp/s41-persistent-model-ops-v1/graph
        worker=model_ops_graph_worker
        worker_args="$op HTP $n_kv $requests"
        ready_pattern='\[model-graph\] ready'
        ;;
    OpenCLGraph)
        phone_root=/data/local/tmp/s41-persistent-model-ops-v1/graph
        worker=model_ops_graph_worker
        worker_args="$op OpenCL $n_kv $requests"
        ready_pattern='\[model-graph\] ready'
        ;;
    OpenCLDispatch)
        phone_root=/data/local/tmp/s41-persistent-model-ops-v1/opencl
        worker=model_ops_opencl_worker
        worker_args="$op OpenCLDispatch $n_kv $requests"
        ready_pattern='\[model-opencl\] ready'
        ;;
    OpenCLPersistent)
        phone_root=/data/local/tmp/s41-persistent-model-ops-v1/opencl
        worker=model_ops_opencl_worker
        worker_args="$op OpenCLPersistent $n_kv $requests"
        ready_pattern='\[model-opencl\] ready'
        ;;
    *)
        echo "unsupported backend: $backend" >&2
        exit 2
        ;;
esac

mkdir -p "$result_dir" "$reference_dir"
if [ ! -f "$reference" ]; then
    python3 "$desktop_root/model_ops_bench.py" \
        --op "$op" --n-kv "$n_kv" --variants "$variants" \
        --reference "$reference" --reference-only
fi

phone_log=$phone_root/$case_name.log
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
    python3 "$desktop_root/../aoa_bench.py" reset >/dev/null 2>&1 || true
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
ADB_SERVER_PORT=$adb_port adb -s "$serial" shell \
    "su -c 'cd $phone_root && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./$worker $worker_args > $case_name.log 2>&1'" &
worker_transport_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 180 ]; do
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
    pidof "$worker" 2>/dev/null | tr -d '\r')

switched=0
attempt=0
while [ "$attempt" -lt 60 ]; do
    if lsusb | grep -Eq '18d1:2d0(0|1|4|5)'; then
        switched=1
        break
    fi
    for normal_device in 22d9:2769 22d9:2772 05c6:908c; do
        normal_vid=${normal_device%:*}
        normal_pid=${normal_device#*:}
        if python3 "$desktop_root/../aoa_bench.py" \
                --vid "$normal_vid" --pid "$normal_pid" switch; then
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

python3 "$desktop_root/model_ops_bench.py" \
    --op "$op" --backend "$backend" --n-kv "$n_kv" \
    --warmup "$warmup" --iters "$iterations" \
    --timeout-ms "$timeout_ms" \
    --reference "$reference" \
    --output "$result_dir/$case_name.json"

wait "$worker_transport_pid"
worker_transport_pid=
worker_device_pid=
trap - EXIT INT TERM

device_ready=0
attempt=0
while [ "$attempt" -lt 5 ]; do
    python3 "$desktop_root/../aoa_bench.py" reset || true
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
