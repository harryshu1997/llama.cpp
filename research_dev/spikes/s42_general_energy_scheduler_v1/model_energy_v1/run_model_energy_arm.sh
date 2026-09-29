#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <qwen-cuda|gemma-cpu|gemma-op15> <repeat-index> <output-root> <adb-port>" >&2
    exit 2
fi

route=$1
repeat_index=$2
output=$3
adb_port=$4

if [[ $route != qwen-cuda && $route != gemma-cpu && $route != gemma-op15 ]]; then
    echo "invalid route: $route" >&2
    exit 2
fi
if [[ ! $repeat_index =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid repeat index: $repeat_index" >&2
    exit 2
fi
if [[ $output != /* || -e $output || -e ${output}.phone-capture ]]; then
    echo "invalid output root: $output" >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "foreign llama-server process" >&2
    exit 75
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "foreign CUDA process" >&2
    exit 75
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
serial=3C15AU002CL00000
capture=${output}.phone-capture
phone_root=/data/local/tmp/s41-opoffload-dmabuf-v1/model-energy-${route}-r${repeat_index}
phone_logger=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_logger.sh
phone_policy=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_allow.rules
phone_samples=$phone_root/samples.tsv
phone_active=$phone_root/active
phone_armed=$phone_root/power.armed
phone_done=$phone_root/power.done

mkdir -p "$capture"
adb_cmd=(adb -P "$adb_port" -s "$serial")
"${adb_cmd[@]}" get-state >/dev/null
"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
"${adb_cmd[@]}" shell \
    "su -c 'rm -rf $phone_root; mkdir $phone_root'"

python3 "$here/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-before.json"

nohup "${adb_cmd[@]}" shell \
    "su -c 'sh $phone_logger $phone_samples $phone_active $phone_armed $phone_done 300'" \
    >"$capture/power-adb.log" 2>&1 &
power_adb_pid=$!

armed=0
for _ in $(seq 1 300); do
    if "${adb_cmd[@]}" shell "su -c 'test -f $phone_armed'" \
            >/dev/null 2>&1; then
        armed=1
        break
    fi
    sleep 0.1
done
if [[ $armed -ne 1 ]]; then
    echo "phone power logger did not arm" >&2
    exit 1
fi

session_adb_pid=
if [[ $route == gemma-op15 ]]; then
    session=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_ffn_session-flex.sh
    worker=/data/local/tmp/s41-opoffload-dmabuf-v1/llama-ffn-split-worker-flex-v2
    phone_model=/data/local/tmp/ls-s32/gemma-4-12B-it-Q4_0.gguf
    restore=/data/local/tmp/s41-opoffload-dmabuf-v1/restore_phone_usb.sh
    nohup "${adb_cmd[@]}" shell \
        "su -c 'S41_FFN_F16_IO=1 S41_FFN_MAX_TOKENS=512 S41_FFN_COLUMN_QUANTUM=2048 S41_FFN_ALTERNATE_COLUMNS=9664 sh $session $worker $phone_model 0-47 11136 HTP0 $phone_root $restore 3600 120000'" \
        >"$capture/session-adb.log" 2>&1 &
    session_adb_pid=$!
    aoa_ready=0
    for _ in $(seq 1 300); do
        if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
            aoa_ready=1
            break
        fi
        sleep 1
    done
    if [[ $aoa_ready -ne 1 ]]; then
        echo "phone AOA session did not become ready" >&2
        exit 1
    fi
else
    "${adb_cmd[@]}" shell "su -c 'touch $phone_active'"
fi

command=(
    python3 "$here/run_model_energy.py"
    --route "$route"
    --repeat-index "$repeat_index"
    --cases "$here/CASES_V1.jsonl"
    --catalog "$here/MODELS_V1.json"
    --port 19100
    --output "$output"
    --execute
    --confirm RUN_S42_MODEL_ENERGY_V1
)
if [[ $route == qwen-cuda ]]; then
    command+=(
        --server /home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server
        --model /home/zhihao/models/Qwen3-14B-Q4_K_M.gguf
        --lib-dir /mnt/storage/s21_deps/cuda-13.2.1/lib
    )
else
    command+=(
        --server /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin/llama-server
        --model /home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf
        --lib-dir /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin
    )
    if [[ $route == gemma-op15 ]]; then
        command+=(
            --bridge /home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-reset-qualified
        )
    fi
fi

set +e
systemd-run --user --scope -p MemorySwapMax=0 "${command[@]}" \
    >"$capture/runner.log" 2>&1
run_rc=$?
set -e

if [[ $route != gemma-op15 ]]; then
    "${adb_cmd[@]}" shell "su -c 'rm -f $phone_active'"
fi
if [[ -n $session_adb_pid ]]; then
    wait "$session_adb_pid" || true
fi

device_ready=0
for _ in $(seq 1 180); do
    if "${adb_cmd[@]}" get-state >/dev/null 2>&1; then
        device_ready=1
        break
    fi
    sleep 1
done
if [[ $device_ready -ne 1 ]]; then
    echo "phone did not restore ADB" >&2
    exit 1
fi

wait "$power_adb_pid"
"${adb_cmd[@]}" shell "su -c 'test -f $phone_done'"
python3 "$here/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-after.json"
"${adb_cmd[@]}" pull "$phone_samples" "$capture/phone-samples.tsv" \
    >"$capture/pull.log"

if [[ $run_rc -ne 0 || ! -f $output/RESULT.json ]]; then
    echo "model-energy runner failed: rc=$run_rc" >&2
    exit "$run_rc"
fi

python3 "$here/attach_phone_energy.py" \
    --result "$output/RESULT.json" \
    --samples "$capture/phone-samples.tsv" \
    --clock-before "$capture/clock-before.json" \
    --clock-after "$capture/clock-after.json" \
    --output "$output/PHONE_ENERGY.json"

sha256sum \
    "$output/RESULT.json" \
    "$output/PHONE_ENERGY.json" \
    "$capture/phone-samples.tsv" \
    >"$capture/SHA256SUMS.txt"
cat "$capture/SHA256SUMS.txt"
