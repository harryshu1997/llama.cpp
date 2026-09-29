#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <output> <server-baseline|runtime-scheduler> <adb-port>" >&2
    exit 2
fi

output=$1
mode=$2
adb_port=$3

if [[ $output != /* || -e $output || -e ${output}.capture ]]; then
    echo "invalid output: $output" >&2
    exit 2
fi
if [[ $mode != server-baseline && $mode != runtime-scheduler ]]; then
    echo "invalid mode: $mode" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
burst_dir=$(cd -- "$here/../../s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1" && pwd)
capture=${output}.capture
serial=3C15AU002CL00000
run_id=${output##*/}
phone_root=/data/local/tmp/s42-three-model/$run_id
phone_logger=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_logger.sh
phone_policy=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_allow.rules
phone_samples=$phone_root/samples.tsv
phone_active=$phone_root/active
phone_armed=$phone_root/power.armed
phone_done=$phone_root/power.done
power_adb_pid=

adb_cmd=(adb -P "$adb_port" -s "$serial")

cleanup() {
    "${adb_cmd[@]}" shell "su -c 'rm -f $phone_active'" >/dev/null 2>&1 || true
    if [[ -n $power_adb_pid ]]; then
        wait "$power_adb_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir -p "$capture"
"${adb_cmd[@]}" get-state >/dev/null
"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
"${adb_cmd[@]}" shell \
    "su -c 'test ! -e $phone_root && mkdir -p $phone_root'"

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-before.json"

nohup "${adb_cmd[@]}" shell \
    "su -c 'sh $phone_logger $phone_samples $phone_active $phone_armed $phone_done 600'" \
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
"${adb_cmd[@]}" shell "su -c 'touch $phone_active'"

command=(
    python3 "$here/run_three_model_trace.py"
    --requests "$here/REQUESTS_BURSTGPT_LLAMA1B_84.jsonl"
    --manifest "$here/TRACE_MANIFEST.json"
    --server-cuda /home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server
    --server-cpu /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin/llama-server
    --cuda-lib-dir /mnt/storage/s21_deps/cuda-13.2.1/lib
    --cpu-lib-dir /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin
    --qwen14-model /home/zhihao/models/Qwen3-14B-Q4_K_M.gguf
    --gemma12-model /home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf
    --llama1-model /home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf
    --adb-port "$adb_port"
    --phone-serial "$serial"
    --mode "$mode"
    --output "$output"
    --execute
    --confirm RUN_THREE_MODEL_MIXED_TRACE
)

set +e
systemd-run --user --scope -p MemorySwapMax=0 "${command[@]}" \
    >"$capture/runner.log" 2>&1
run_rc=$?
set -e

"${adb_cmd[@]}" shell "su -c 'rm -f $phone_active'"
wait "$power_adb_pid" || true
power_adb_pid=
"${adb_cmd[@]}" shell "su -c 'test -f $phone_done'"

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-after.json"
"${adb_cmd[@]}" pull "$phone_samples" "$capture/phone-samples.tsv" \
    >"$capture/pull.log"

if [[ $run_rc -ne 0 || ! -f $output/RESULT.json ]]; then
    echo "trace runner failed: rc=$run_rc" >&2
    exit "$run_rc"
fi

python3 "$burst_dir/analyze_phone_energy.py" \
    --result "$output/RESULT.json" \
    --samples "$capture/phone-samples.tsv" \
    --clock-before "$capture/clock-before.json" \
    --clock-after "$capture/clock-after.json" \
    --output "$capture/PHONE_ENERGY.json"

sha256sum \
    "$output/RESULT.json" \
    "$capture/PHONE_ENERGY.json" \
    "$capture/phone-samples.tsv" \
    >"$capture/SHA256SUMS.txt"
cat "$capture/SHA256SUMS.txt"
