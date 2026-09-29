#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <control|dynamic> <repeat-index> <output-root> <adb-port>" >&2
    exit 2
fi

arm=$1
repeat_index=$2
output=$3
adb_port=$4
if [[ $arm != control && $arm != dynamic ]]; then
    echo "invalid arm: $arm" >&2
    exit 2
fi
if [[ ! $repeat_index =~ ^[1-9][0-9]*$ || $output != /* \
        || -e $output || -e ${output}.phone-capture ]]; then
    echo "invalid adoption screen arguments" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
legacy_root=${S42_LEGACY_ROOT:-/home/zhihao/s41-dynamic-ffn-v1}
burst_dir=${S42_BURST_DIR:-$legacy_root/campaign/server_trace_v2}
trace=$legacy_root/campaign/input/REQUESTS_SEMANTIC_SOURCE.jsonl
qwen_server=${S42_QWEN_SERVER:-$legacy_root/build-server-ffn-cuda/bin/llama-server}
qwen_model=${S42_QWEN_MODEL:-/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf}
gemma_server=${S42_ADOPTABLE_GEMMA_SERVER:?set S42_ADOPTABLE_GEMMA_SERVER}
gemma_model=${S42_GEMMA_MODEL:-/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf}
bridge=${S42_ADOPTABLE_BRIDGE:?set S42_ADOPTABLE_BRIDGE}
cuda_lib_dir=${S42_CUDA_LIB_DIR:-/mnt/storage/s21_deps/cuda-13.2.1/lib}
qwen_server_bin=$(dirname -- "$qwen_server")
capture=${output}.phone-capture
serial=3C15AU002CL00000

arm_offset=0
if [[ $arm == dynamic ]]; then
    arm_offset=20
fi
qwen_port=$((18700 + arm_offset + repeat_index))
gemma_port=$((18740 + arm_offset + repeat_index))
bridge_port=$((18780 + arm_offset + repeat_index))

phone_base=/data/local/tmp/s41-opoffload-dmabuf-v1
phone_run_tag=adoption-screen-${arm}-r${repeat_index}
phone_root=$phone_base/$phone_run_tag
phone_logger=$phone_base/phone_power_logger.sh
phone_policy=$phone_base/phone_power_allow.rules
phone_samples=$phone_root/samples.tsv
phone_active=$phone_root/active
phone_armed=$phone_root/power.armed
phone_done=$phone_root/power.done
phone_workers=${S42_PHONE_WORKERS:-$phone_base/resident_qwen_workers-v1}
phone_router=$phone_base/resident_ffn_router-triple-v1
phone_session=${S42_PHONE_SESSION:-$phone_base/resident_qwen_session-v1.sh}
phone_gemma=$phone_base/gemma-4-12B-Q40-dequant-f16.gguf
phone_qwen=$phone_base/Qwen3-14B-Q4KM-dequant-f16.gguf
restore_usb=$phone_base/restore_phone_usb.sh

for path in \
        "$here/run_adoption_energy_screen.py" \
        "$here/capture_cgroup_memory.py" \
        "$burst_dir/capture_phone_clock.py" \
        "$burst_dir/analyze_phone_energy.py" \
        "$trace" "$qwen_server" "$qwen_model" \
        "$gemma_server" "$gemma_model" "$bridge" "$cuda_lib_dir"; do
    if [[ ! -e $path ]]; then
        echo "missing adoption screen dependency: $path" >&2
        exit 1
    fi
done
if pgrep -f '^.*/llama-server .*--port 18' >/dev/null \
        || pgrep -f '^.*/ffn_dmabuf_bridge' >/dev/null \
        || nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
            | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "another inference, bridge, or CUDA process is active" >&2
    pgrep -af 'llama-server|ffn_dmabuf_bridge' >&2 || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader >&2 || true
    exit 75
fi

mkdir -p "$capture"
runtime_dir=$(mktemp -d /tmp/s42-adoption-energy-screen.XXXXXX)
fence_socket=$runtime_dir/fence.sock
arm_file=$runtime_dir/armed
qwen_pid=
bridge_pid=
power_adb_pid=
cleanup() {
    if [[ -n $qwen_pid ]] && kill -0 "$qwen_pid" 2>/dev/null; then
        kill -INT "$qwen_pid" 2>/dev/null || true
        wait "$qwen_pid" 2>/dev/null || true
    fi
    if [[ -n $bridge_pid ]] && kill -0 "$bridge_pid" 2>/dev/null; then
        kill -INT "$bridge_pid" 2>/dev/null || true
        wait "$bridge_pid" 2>/dev/null || true
    fi
    if [[ -n $power_adb_pid ]] && kill -0 "$power_adb_pid" 2>/dev/null; then
        kill "$power_adb_pid" 2>/dev/null || true
        wait "$power_adb_pid" 2>/dev/null || true
    fi
    rm -f -- "$fence_socket" "$arm_file"
    rmdir -- "$runtime_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

adb_cmd=(adb -P "$adb_port" -s "$serial")
"${adb_cmd[@]}" get-state >/dev/null
"${adb_cmd[@]}" shell \
    "su -c 'test ! -e $phone_root && mkdir $phone_root'"
"${adb_cmd[@]}" shell \
    "su -c '/product/bin/magiskpolicy --live --apply $phone_policy'"
"${adb_cmd[@]}" shell "su -c 'am kill-all; sleep 3'"

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-before.json"

nohup "${adb_cmd[@]}" shell \
    "su -c 'sh $phone_logger $phone_samples $phone_active $phone_armed $phone_done 1800'" \
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

"${adb_cmd[@]}" shell \
    "su -c 'S42_QWEN_ONLY=1 nohup sh $phone_session $phone_workers $phone_router $phone_gemma $phone_qwen $phone_root $restore_usb 1800 1 > $phone_root/session-launch.log 2>&1 < /dev/null &'"

aoa_ready=0
for _ in $(seq 1 360); do
    if lsusb -d 18d1:2d00 >/dev/null 2>&1; then
        aoa_ready=1
        break
    fi
    sleep 1
done
if [[ $aoa_ready -ne 1 ]]; then
    echo "resident Qwen phone session did not bind FunctionFS" >&2
    exit 1
fi

S42_FFN_PREFETCH_FENCE_SOCKET="$fence_socket" \
S42_FFN_PREFETCH_FENCE_GROUP_FIRST_LAYER=0 \
S42_FFN_PREFETCH_FENCE_GROUP_LAST_LAYER=11 \
S42_FFN_PREFETCH_FENCE_TOTAL_BYTES=2013265920 \
"$bridge" 127.0.0.1 "$bridge_port" malloc-split \
    >"$capture/bridge.stdout" 2>"$capture/bridge.stderr" &
bridge_pid=$!
bridge_ready=0
for _ in $(seq 1 300); do
    if grep -q 'ready bind=' "$capture/bridge.stderr" 2>/dev/null; then
        bridge_ready=1
        break
    fi
    if ! kill -0 "$bridge_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $bridge_ready -ne 1 ]]; then
    echo "adoptable bridge did not become ready" >&2
    exit 1
fi

env \
    "LD_LIBRARY_PATH=$cuda_lib_dir:$qwen_server_bin" \
    S41_SERVER_FFN_HOST=127.0.0.1 \
    "S41_SERVER_FFN_PORT=$bridge_port" \
    S41_SERVER_FFN_N_EMBD=5120 \
    S41_SERVER_FFN_LAYER_MASK=4095 \
    S41_SERVER_FFN_COLUMNS=17408 \
    S41_SERVER_FFN_F16_IO=1 \
    S41_SERVER_FFN_ACTIVATION=swiglu \
    S41_SERVER_FFN_POLICY=1:17408,512:0 \
    S41_SERVER_FFN_TIMEOUT_MS=120000 \
    "$qwen_server" \
    --model "$qwen_model" --alias qwen-adoption-energy-screen --fit off \
    --ctx-size 24576 --parallel 4 --batch-size 2048 --ubatch-size 512 \
    --flash-attn on --cont-batching --kv-unified --no-cache-idle-slots \
    --cache-type-k f16 --cache-type-v f16 --split-mode none \
    --n-gpu-layers 15 --main-gpu 0 --device CUDA0 \
    --host 127.0.0.1 --port "$qwen_port" --metrics --slots \
    --no-webui --log-colors off --log-timestamps --log-verbosity 1 \
    >"$capture/qwen.stdout" 2>"$capture/qwen.stderr" &
qwen_pid=$!

qwen_ready=0
for _ in $(seq 1 900); do
    if curl -fsS "http://127.0.0.1:$qwen_port/health" \
            >/dev/null 2>&1; then
        qwen_ready=1
        break
    fi
    if ! kill -0 "$qwen_pid" 2>/dev/null; then
        break
    fi
    sleep 0.2
done
if [[ $qwen_ready -ne 1 ]]; then
    echo "Qwen screen server did not become ready" >&2
    exit 1
fi

python3 "$here/run_adoption_energy_screen.py" \
    --arm "$arm" --repeat-index "$repeat_index" \
    --burst-dir "$burst_dir" --requests "$trace" \
    --qwen-model "$qwen_model" --qwen-port "$qwen_port" \
    --qwen-pid "$qwen_pid" \
    --gemma-server "$gemma_server" --gemma-model "$gemma_model" \
    --gemma-port "$gemma_port" --cuda-lib-dir "$cuda_lib_dir" \
    --fence-socket "$fence_socket" --arm-file "$arm_file" \
    --output "$output" >"$capture/runner.log" 2>&1

python3 "$here/capture_cgroup_memory.py" \
    --output "$capture/CGROUP_MEMORY_V1.json"

kill -INT "$qwen_pid"
wait "$qwen_pid"
qwen_pid=
wait "$bridge_pid"
bridge_pid=

device_ready=0
for _ in $(seq 1 360); do
    if "${adb_cmd[@]}" get-state >/dev/null 2>&1; then
        device_ready=1
        break
    fi
    sleep 0.5
done
if [[ $device_ready -ne 1 ]]; then
    echo "phone did not restore ADB" >&2
    exit 1
fi

done_ready=0
for _ in $(seq 1 240); do
    if "${adb_cmd[@]}" shell "su -c 'test -f $phone_done'" \
            >/dev/null 2>&1; then
        done_ready=1
        break
    fi
    sleep 0.25
done
if [[ $done_ready -ne 1 ]]; then
    echo "phone power logger did not finish" >&2
    exit 1
fi
wait "$power_adb_pid" || true
power_adb_pid=

python3 "$burst_dir/capture_phone_clock.py" \
    --adb-port "$adb_port" --serial "$serial" \
    --output "$capture/clock-after.json"
"${adb_cmd[@]}" pull "$phone_samples" "$capture/phone-samples.tsv" \
    >"$capture/pull.log"
for name in session.log resident-workers.log router.log session-launch.log; do
    "${adb_cmd[@]}" pull "$phone_root/$name" "$capture/$name" \
        >>"$capture/pull.log" 2>&1 || true
done

python3 "$burst_dir/analyze_phone_energy.py" \
    --result "$output/PHONE_BOUNDARY.json" \
    --samples "$capture/phone-samples.tsv" \
    --clock-before "$capture/clock-before.json" \
    --clock-after "$capture/clock-after.json" \
    --output "$capture/PHONE_ENERGY_V3.json"

sha256sum \
    "$output/RESULT.json" "$output/PHONE_BOUNDARY.json" \
    "$output/resource-samples.jsonl" "$output/gemma.stderr" \
    "$capture/PHONE_ENERGY_V3.json" \
    "$capture/phone-samples.tsv" "$capture/bridge.stderr" \
    "$capture/qwen.stderr" "$capture/CGROUP_MEMORY_V1.json" \
    >"$capture/SHA256SUMS.txt"
trap - EXIT INT TERM
cleanup
cat "$output/RESULT.json"
