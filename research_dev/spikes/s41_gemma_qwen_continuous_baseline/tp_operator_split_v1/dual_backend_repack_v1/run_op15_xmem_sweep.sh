#!/bin/sh
set -eu

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_root/../../../../.." && pwd)
runtime_root=${S41_FFN_RUNTIME_ROOT:-$repo_root/build-ffn-overlap-android/bin}
htp_skel=${S41_HTP_V81_SKEL:-$repo_root/build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v81.so}
serial=${S41_PHONE_SERIAL:-3C15AU002CL00000}
remote_root=/data/local/tmp/s41-dual-repack-v1
result_root=${1:-$script_root/results/xmem-$(date +%Y%m%dT%H%M%S)}
total_columns=${S41_TOTAL_COLUMNS:-9664}
gpu_columns_list=${S41_GPU_COLUMNS:-"64 128 256 512 1024 1472 2048 3072 4096 8192"}
batch_list=${S41_BATCHES:-"16 32 64 128"}
warmup=${S41_WARMUP:-10}
iterations=${S41_ITERATIONS:-30}

if [ "$serial" != "3C15AU002CL00000" ] && \
   [ "${S41_ALLOW_NON_OP15:-0}" != "1" ]; then
    echo "refusing non-OP15 serial without S41_ALLOW_NON_OP15=1: $serial" >&2
    exit 2
fi
if [ "$(adb -s "$serial" get-state 2>/dev/null || true)" != "device" ]; then
    echo "OP15 is not available through adb: $serial" >&2
    exit 3
fi

mkdir -p "$result_root"
adb -s "$serial" shell "mkdir -p $remote_root"
adb -s "$serial" push "$script_root/dual_backend_ffn.android" \
    "$runtime_root/libggml.so" "$runtime_root/libggml-base.so" \
    "$runtime_root/libggml-cpu.so" "$runtime_root/libggml-opencl.so" \
    "$runtime_root/libggml-hexagon.so" "$htp_skel" \
    "$remote_root/" >/dev/null
adb -s "$serial" shell "chmod 755 $remote_root/dual_backend_ffn.android"

for batch in $batch_list; do
    for gpu_columns in $gpu_columns_list; do
        htp_columns=$((total_columns - gpu_columns))
        if [ "$htp_columns" -le 0 ]; then
            continue
        fi
        case_name=m${batch}_h${htp_columns}_g${gpu_columns}
        log_path=$result_root/$case_name.log
        echo "running $case_name" >&2
        status=0
        adb -s "$serial" shell \
            "su -c 'cd $remote_root && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
             GGML_HEXAGON_MBUF=4192 GGML_HEXAGON_NHVX=4 \
             S41_DISABLE_GRAPH_CACHE=1 ./dual_backend_ffn.android \
             --k 3840 --n-ff 15360 --htp-columns $htp_columns \
             --gpu-columns $gpu_columns --batch $batch --type q4_0 \
             --gpu-mode f16-xmem --warmup $warmup \
             --iterations $iterations'" > "$log_path" 2>&1 || status=$?
        result=$(grep 'DUAL_REPACK_RESULT' "$log_path" || true)
        if [ -z "$result" ]; then
            tail -40 "$log_path" >&2
            if [ "$status" -eq 0 ]; then
                status=1
            fi
            exit "$status"
        fi
        printf '%s adb_status=%d\n' "$result" "$status" |
            tee -a "$result_root/summary.txt"
    done
done

echo "$result_root"
