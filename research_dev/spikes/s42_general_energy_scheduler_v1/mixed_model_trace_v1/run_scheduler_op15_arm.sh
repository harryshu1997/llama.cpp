#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <output-root> <adb-port> <baseline-result> <op15-result>" >&2
    exit 2
fi

output=$1
adb_port=$2
baseline_result=$3
op15_result=$4

if [[ $output != /* || -e $output || -e ${output}.phone-capture ]]; then
    echo "invalid output root: $output" >&2
    exit 2
fi
if [[ ! -f $baseline_result || ! -f $op15_result ]]; then
    echo "missing scheduler evidence" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
burst_dir=$(cd -- "$here/../../s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1" && pwd)
serial=3C15AU002CL00000
capture=${output}.phone-capture
plan=$capture/SCHEDULER_PLAN.json
run_id=${output##*/}
phone_root=/data/local/tmp/s41-opoffload-dmabuf-v1/$run_id
phone_logger=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_logger.sh
phone_policy=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_allow.rules
phone_samples=$phone_root/samples.tsv
phone_power_active=$phone_root/power.active
phone_armed=$phone_root/power.armed
phone_done=$phone_root/power.done
session=/data/local/tmp/s41-opoffload-dmabuf-v1/phone_ffn_session-flex.sh
worker=/data/local/tmp/s41-opoffload-dmabuf-v1/llama-ffn-split-worker-reset-qualified
model=/data/local/tmp/ls-s32/gemma-4-12B-it-Q4_0.gguf
restore=/data/local/tmp/s41-opoffload-dmabuf-v1/restore_phone_usb.sh
bridge=/home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-reset-qualified
expected_kernel=6.12.23-android16-5-o-g227664cbe007-4k
expected_kernel_btf_sha=f3afcf985b24de3eb5a1d5453bf4ffd95963b806ce99a59430c5a8bf7d201d17
expected_worker_sha=e3d269336bd29ffc12ae46479367105fb002f507266f2a0ebeba4333f2d4de1d
expected_model_sha=494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c
expected_session_sha=784d703eb784d8f1a8ff00cb53291aa1b63bf9c0c871a088d9ebb4c419241f72
expected_bridge_sha=1c5bfb27cd263238cf2e364ee5bdb538d7cd78fc82e55dd5a6228d143f1809d7

mkdir -p "$capture"

python3 "$here/plan_six_model_trace.py" \
    --requests "$here/REQUESTS_MIXED_114.jsonl" \
    --manifest "$here/TRACE_MANIFEST.json" \
    --baseline-result "$baseline_result" \
    --op15-result "$op15_result" \
    --output "$plan" \
    >"$capture/planner.log"

adb_cmd=(adb -P "$adb_port" -s "$serial")
"${adb_cmd[@]}" get-state >/dev/null
phone_kernel=$("${adb_cmd[@]}" shell uname -r | tr -d '\r')
if [[ $phone_kernel != "$expected_kernel" ]]; then
    echo "wrong phone kernel: $phone_kernel" >&2
    exit 1
fi
phone_kernel_btf_sha=$("${adb_cmd[@]}" shell su -c \
    sha256sum /sys/kernel/btf/vmlinux | awk '{print $1}')
phone_worker_sha=$("${adb_cmd[@]}" shell sha256sum "$worker" | awk '{print $1}')
phone_model_sha=$("${adb_cmd[@]}" shell sha256sum "$model" | awk '{print $1}')
phone_session_sha=$("${adb_cmd[@]}" shell sha256sum "$session" | awk '{print $1}')
bridge_sha=$(sha256sum "$bridge" | awk '{print $1}')
if [[ $phone_kernel_btf_sha != "$expected_kernel_btf_sha" ||
      $phone_worker_sha != "$expected_worker_sha" ||
      $phone_model_sha != "$expected_model_sha" ||
      $phone_session_sha != "$expected_session_sha" ||
      $bridge_sha != "$expected_bridge_sha" ]]; then
    echo "phone route identity mismatch" >&2
    exit 1
fi
printf '{"bridge_sha256":"%s","kernel_btf_sha256":"%s","kernel_release":"%s","model_sha256":"%s","session_sha256":"%s","worker_sha256":"%s"}\n' \
    "$bridge_sha" "$phone_kernel_btf_sha" "$phone_kernel" \
    "$phone_model_sha" "$phone_session_sha" "$phone_worker_sha" \
    >"$capture/PHONE_ROUTE_IDENTITY.json"
"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
"${adb_cmd[@]}" shell \
    "su -c 'rm -rf $phone_root; mkdir $phone_root'"

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-before.json"

nohup "${adb_cmd[@]}" shell \
    "su -c 'sh $phone_logger $phone_samples $phone_power_active $phone_armed $phone_done 180'" \
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
"${adb_cmd[@]}" shell "su -c 'touch $phone_power_active'"

nohup "${adb_cmd[@]}" shell \
    "su -c 'S41_FFN_F16_IO=1 S41_FFN_MAX_TOKENS=512 S41_FFN_COLUMN_QUANTUM=2048 S41_FFN_ALTERNATE_COLUMNS=9664 sh $session $worker $model 0-47 11136 HTP0 $phone_root $restore 1800 120000'" \
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

command=(
    python3 "$here/run_six_model_trace.py"
    --requests "$here/REQUESTS_MIXED_114.jsonl"
    --manifest "$here/TRACE_MANIFEST.json"
    --scheduler-plan "$plan"
    --server-cuda /home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server
    --server-cpu /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin/llama-server
    --cuda-lib-dir /mnt/storage/s21_deps/cuda-13.2.1/lib
    --cpu-lib-dir /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin
    --gemma-phone-server /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin/llama-server
    --gemma-phone-lib-dir /home/zhihao/s41-dynamic-ffn-v1/build-server-ffn/bin
    --bridge "$bridge"
    --qwen14-model /home/zhihao/models/Qwen3-14B-Q4_K_M.gguf
    --gemma12-model /home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf
    --qwen06-model /home/zhihao/models/Qwen3-0.6B-Q8_0.gguf
    --llama1-model /home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf
    --qwen8-model /home/zhihao/models/Qwen3-8B-Q8_0.gguf
    --gemma-e2b-model /home/zhihao/models/gemma-4-E2B-it-Q8_0.gguf
    --mmproj /home/zhihao/models/mmproj-gemma-4-E2B-it-Q8_0.gguf
    --output "$output"
    --execute
    --confirm RUN_SIX_MODEL_MIXED_TRACE
)

set +e
systemd-run --user --scope "${command[@]}" >"$capture/runner.log" 2>&1
run_rc=$?
set -e

wait "$session_adb_pid" || true

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

"${adb_cmd[@]}" shell "su -c 'rm -f $phone_power_active'"
wait "$power_adb_pid" || true
"${adb_cmd[@]}" shell "su -c 'test -f $phone_done'"
python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-after.json"
"${adb_cmd[@]}" pull "$phone_samples" "$capture/phone-samples.tsv" \
    >"$capture/pull.log"
"${adb_cmd[@]}" pull "$phone_root/worker.log" "$capture/worker.log" \
    >>"$capture/pull.log" 2>&1 || true
"${adb_cmd[@]}" pull "$phone_root/session.log" "$capture/session.log" \
    >>"$capture/pull.log" 2>&1 || true

if [[ $run_rc -ne 0 || ! -f $output/RESULT.json ]]; then
    echo "trace runner failed: rc=$run_rc" >&2
    exit "$run_rc"
fi

python3 "$burst_dir/analyze_phone_energy.py" \
    --result "$output/RESULT.json" \
    --samples "$capture/phone-samples.tsv" \
    --clock-before "$capture/clock-before.json" \
    --clock-after "$capture/clock-after.json" \
    --output "$capture/PHONE_ENERGY_V3.json"

sha256sum \
    "$plan" \
    "$output/RESULT.json" \
    "$capture/PHONE_ENERGY_V3.json" \
    "$capture/PHONE_ROUTE_IDENTITY.json" \
    "$capture/phone-samples.tsv" \
    >"$capture/SHA256SUMS.txt"
cat "$capture/SHA256SUMS.txt"
