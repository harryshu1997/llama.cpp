#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s31_dir="$repo_root/research_dev/spikes/s31_latency_balanced_cut"
phone_control="$repo_root/research_dev/spikes/s24_overlap_handoff_poc/a6000_phone_control.sh"
desktop_host=${S31_DESKTOP_HOST:-zhihao@172.20.74.85}
desktop_repo=${S31_DESKTOP_REPO:-/home/zhihao/llama.cpp-release}
desktop_model=${S31_DESKTOP_MODEL:-}
phone_model=${S31_PHONE_MODEL:-/data/local/tmp/ls-s32/gemma-4-12B-it-Q8_0.gguf}
phone_dir=${S31_PHONE_DIR:-/data/local/tmp/ls-s34-identity}
model_sha=${S31_MODEL_SHA256:-7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848}
run_dir=${1:-}
cuts=(1 2 3 4 5 6)

fail() {
    echo "error: $*" >&2
    exit 2
}

[[ -n "$run_dir" ]] || fail "usage: $0 RUN_DIR"
[[ -n "$desktop_model" ]] || \
    fail "S31_DESKTOP_MODEL must name the desktop copy of the shared Q8_0 GGUF"
[[ "$phone_dir" == /* ]] || fail "S31_PHONE_DIR must be an absolute device path"
[[ "$model_sha" =~ ^[0-9a-f]{64}$ ]] || fail "S31_MODEL_SHA256 is invalid"
[[ "$run_dir" == /* ]] || fail "RUN_DIR must be absolute"
[[ ! -e "$run_dir" ]] || fail "RUN_DIR already exists"
run_name=$(basename "$run_dir")
[[ "$run_name" =~ ^[A-Za-z0-9._-]+$ ]] || fail "RUN_DIR basename is unsafe"
remote_s31="$desktop_repo/research_dev/spikes/s31_latency_balanced_cut"
remote_run="$remote_s31/results/$run_name"
remote_control="$desktop_repo/research_dev/spikes/s24_overlap_handoff_poc/desktop_cuda_control.sh"
remote_stage_client="$desktop_repo/research_dev/spikes/s22_slo_overlap_pipeline/stage_v3_client.py"
export S24_PHONE_DIR="$phone_dir"

for serial in 5ae7a43d 3C15AU002CL00000; do
    conflicts=$(timeout 5 adb -s "$serial" shell \
        "pidof llama-layersplit llama-hetero-ubatch 2>/dev/null || true" |
        tr -d '\r')
    [[ -z "$conflicts" ]] || fail "$serial has an active inference process: $conflicts"
done

mkdir -p "$run_dir"
current_phone=
desktop_started=0
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$current_phone" ]]; then
        timeout 20 "$phone_control" stop "$current_phone" >/dev/null 2>&1 || true
    fi
    if [[ "$desktop_started" -eq 1 ]]; then
        ssh "$desktop_host" \
            "bash '$remote_control' stop '$remote_run/desktop-session'" \
            >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

ssh "$desktop_host" \
    "mkdir -p '$remote_s31' '$(dirname "$remote_stage_client")' '$(dirname "$remote_control")'"
rsync -a --exclude 'results/' "$s31_dir/" "$desktop_host:$remote_s31/"
rsync -a "$repo_root/research_dev/spikes/s22_slo_overlap_pipeline/stage_v3_client.py" \
    "$desktop_host:$remote_stage_client"
rsync -a "$repo_root/research_dev/spikes/s24_overlap_handoff_poc/desktop_cuda_control.sh" \
    "$desktop_host:$remote_control"
ssh "$desktop_host" "mkdir -p '$remote_run'"
ssh "$desktop_host" \
    "cd '$desktop_repo' && env S24_MODEL='$desktop_model' S24_MODEL_SHA256='$model_sha' S24_CUDA_CONTEXT=16 S24_CUDA_MAX_PREFILL=8 S24_CUDA_N_GEN=8 S24_CUDA_PREFIX_STREAMS=32 S24_CUDA_MID_STREAMS=32 S24_CUDA_TAIL_STREAMS=32 S24_CUDA_PREFIX_LAYER_START=0 S24_CUDA_PREFIX_LAYER_END=6 S24_CUDA_MID_LAYER_START=6 S24_CUDA_MID_LAYER_END=8 S24_CUDA_TAIL_LAYER_START=8 S24_CUDA_TAIL_LAYER_END=48 bash '$remote_control' start '$remote_run/desktop-session'"
desktop_started=1

last_cut=${cuts[${#cuts[@]}-1]}
for cut in "${cuts[@]}"; do
    phone_session="$run_dir/phone-cut-$cut"
    current_phone=$phone_session
    S24_CONFIRM_A6000_CONTROL=YES \
    S24_OP12_HEAD="$phone_model" \
    S24_OP12_MODEL_SHA256="$model_sha" \
    S24_OP15_MID="$phone_model" \
    S24_OP15_MODEL_SHA256="$model_sha" \
    S24_OP12_LAYER_START=0 \
    S24_OP12_LAYER_END=$cut \
    S24_OP15_LAYER_START=$cut \
    S24_OP15_LAYER_END=8 \
    S24_PHONE_CONTEXT=16 \
    S24_PHONE_STREAMS=32 \
    S24_PHONE_MAX_PREFILL=8 \
    S24_PHONE_N_GEN=8 \
    S24_OP12_MBUF=3336 \
    S24_OP15_MBUF=4192 \
        "$phone_control" start "$phone_session"

    scp -q "$phone_session/session.env" \
        "$desktop_host:$remote_run/phone-cut-$cut.env"
    cuda_end=detach
    [[ "$cut" -eq "$last_cut" ]] && cuda_end=stop
    ssh "$desktop_host" \
        "cd '$remote_s31' && PYTHONDONTWRITEBYTECODE=1 python3 measure_cut.py --cuda-prefix 127.0.0.1:24180 --cuda-mid 127.0.0.1:24181 --op12 192.168.1.193:24280 --op15 192.168.1.97:24281 --cuda-tail 127.0.0.1:24182 --phone-cut '$cut' --gather-us 50000 --reps 2 --phone-session-env '$remote_run/phone-cut-$cut.env' --desktop-session-env '$remote_run/desktop-session/session.env' --cuda-session-end '$cuda_end' --output '$remote_run/cut-$cut.json'"

    for _attempt in $(seq 1 300); do
        status=$(timeout 3 "$phone_control" status "$phone_session" || true)
        if [[ "$status" == *"OP12 controller=dead phone=dead"* && \
              "$status" == *"OP15 controller=dead phone=dead"* ]]; then
            break
        fi
        sleep 0.1
    done
    status=$(timeout 3 "$phone_control" status "$phone_session" || true)
    [[ "$status" == *"OP12 controller=dead phone=dead"* && \
       "$status" == *"OP15 controller=dead phone=dead"* ]] || \
        fail "phone workers did not terminate after cut $cut"
    "$phone_control" collect "$phone_session" >/dev/null
    current_phone=
done

ssh "$desktop_host" "bash '$remote_control' collect '$remote_run/desktop-session'"
desktop_started=0
mkdir -p "$run_dir/desktop"
rsync -a "$desktop_host:$remote_run/" "$run_dir/desktop/"

measurements=()
for cut in "${cuts[@]}"; do
    measurements+=("$run_dir/desktop/cut-$cut.json")
done
PYTHONDONTWRITEBYTECODE=1 python3 "$s31_dir/cut_selector.py" \
    "${measurements[@]}" \
    --expected-cuts 1,2,3,4,5,6 \
    --route-slo-us 12000000 \
    --output "$run_dir/selection.json"

(
    cd "$run_dir"
    find . -type f ! -name SHA256SUMS.txt -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum > SHA256SUMS.txt
)
trap - EXIT INT TERM
printf '%s\n' "$run_dir"
