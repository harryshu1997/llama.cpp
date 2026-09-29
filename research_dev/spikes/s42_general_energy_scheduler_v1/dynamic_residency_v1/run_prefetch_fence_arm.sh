#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <observe|prefetch> <repeat-index> <output-root> <adb-port>" >&2
    exit 2
fi

mode=$1
repeat_index=$2
output=$3
adb_port=$4
if [[ $mode != observe && $mode != prefetch ]]; then
    echo "invalid mode: $mode" >&2
    exit 2
fi
if [[ ! $repeat_index =~ ^[0-9]+$ || $repeat_index -eq 0 ]]; then
    echo "invalid repeat index: $repeat_index" >&2
    exit 2
fi
if [[ $output != /* || -e $output || -e ${output}.phone-capture \
        || -e ${output}.prefetch-probe ]]; then
    echo "invalid output root: $output" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=${S42_UNIFIED_REPO_ROOT:-$(cd -- "$here/../../../.." && pwd)}
qwen_arm=${S42_QWEN_ARM:-$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/multi_session_phone_v1/run_qwen_full_energy_arm.sh}
bridge=${S42_PREFETCH_BRIDGE:?set S42_PREFETCH_BRIDGE}
helper=${S42_PREFETCH_HELPER:?set S42_PREFETCH_HELPER}
gemma_model=${S42_GEMMA_MODEL:-/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf}
probe=${output}.prefetch-probe
stage_offset=${S42_PREFETCH_OFFSET:-15838752}
stage_bytes=${S42_PREFETCH_BYTES:-268435456}
chunk_bytes=${S42_PREFETCH_CHUNK_BYTES:-4194304}
gpu_reserve_bytes=${S42_PREFETCH_GPU_RESERVE_BYTES:-536870912}
chunks_per_window=0
if [[ $mode == prefetch ]]; then
    chunks_per_window=${S42_PREFETCH_CHUNKS_PER_WINDOW:-8}
fi
group_first_layer=${S42_PREFETCH_GROUP_FIRST_LAYER:-0}
group_last_layer=${S42_PREFETCH_GROUP_LAST_LAYER:-11}

for path in "$qwen_arm" "$bridge" "$helper" "$gemma_model"; do
    if [[ ! -e $path ]]; then
        echo "missing physical dependency: $path" >&2
        exit 1
    fi
done
mkdir "$probe"
runtime_dir=$(mktemp -d /tmp/s42-prefetch-fence.XXXXXX)
socket=$runtime_dir/fence.sock
arm_file=$runtime_dir/armed
helper_pid=
cleanup() {
    if [[ -n $helper_pid ]] && kill -0 "$helper_pid" 2>/dev/null; then
        kill -TERM "$helper_pid" 2>/dev/null || true
        wait "$helper_pid" 2>/dev/null || true
    fi
    rm -f -- "$socket" "$arm_file"
    rmdir -- "$runtime_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 "$helper" \
    "$mode" "$socket" "$arm_file" "$gemma_model" \
    "$stage_offset" "$stage_bytes" "$chunk_bytes" \
    "$chunks_per_window" "$gpu_reserve_bytes" \
    >"$probe/helper.stdout" 2>"$probe/helper.stderr" &
helper_pid=$!
helper_ready=0
for _ in $(seq 1 600); do
    if grep -q '^PREFETCH_READY ' "$probe/helper.stderr" 2>/dev/null; then
        helper_ready=1
        break
    fi
    if ! kill -0 "$helper_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ $helper_ready -ne 1 ]]; then
    echo "prefetch helper did not become ready" >&2
    exit 1
fi

run_tag=${S42_PHONE_RUN_TAG:-dynamic-prefetch-v1-${mode}}
set +e
S42_QWEN_BRIDGE="$bridge" \
S42_QWEN_GPU_LAYERS=15 \
S42_PREFETCH_ARM_FILE="$arm_file" \
S42_FFN_PREFETCH_FENCE_SOCKET="$socket" \
S42_FFN_PREFETCH_FENCE_GROUP_FIRST_LAYER="$group_first_layer" \
S42_FFN_PREFETCH_FENCE_GROUP_LAST_LAYER="$group_last_layer" \
S42_PHONE_RUN_TAG="$run_tag" \
"$qwen_arm" op15 "$repeat_index" "$output" "$adb_port" \
    >"$probe/arm.stdout" 2>"$probe/arm.stderr"
arm_rc=$?
set -e

if [[ $arm_rc -eq 0 ]]; then
    set +e
    wait "$helper_pid"
    helper_rc=$?
    set -e
    helper_pid=
else
    helper_rc=1
fi
pass_count=$(grep -c '^PREFETCH_RESULT status=PASS ' \
    "$probe/helper.stdout" || true)
if [[ $arm_rc -ne 0 || $helper_rc -ne 0 \
        || ! -f $output/RESULT.json \
        || ! -f ${output}.phone-capture/PHONE_ENERGY_V3.json \
        || $pass_count -ne 1 ]]; then
    echo "prefetch arm failed: arm_rc=$arm_rc helper_rc=$helper_rc" >&2
    exit 1
fi

python3 "$here/analyze_prefetch_fence_run.py" \
    --mode "$mode" \
    --qwen-result "$output/RESULT.json" \
    --phone-energy "${output}.phone-capture/PHONE_ENERGY_V3.json" \
    --server-log "${output}.phone-capture/server.stderr" \
    --bridge-log "${output}.phone-capture/bridge.stderr" \
    --helper-log "$probe/helper.stdout" \
    --output "$probe/PREFETCH_RUN_V1.json"
sha256sum \
    "$bridge" "$helper" "$qwen_arm" \
    "$output/RESULT.json" \
    "${output}.phone-capture/PHONE_ENERGY_V3.json" \
    "${output}.phone-capture/server.stderr" \
    "${output}.phone-capture/bridge.stderr" \
    "$probe/helper.stdout" "$probe/PREFETCH_RUN_V1.json" \
    >"$probe/SHA256SUMS.txt"
trap - EXIT INT TERM
cleanup
cat "$probe/PREFETCH_RUN_V1.json"
