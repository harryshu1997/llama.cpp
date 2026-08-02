#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
output_dir=${1:-"$s24_dir/results/cp5_controls"}
cp4_dir=${S24_CP4_DIR:-"$s24_dir/results/cp4_fixed_diamond"}
cp4_gate="$cp4_dir/validation/cp4-gates.json"
profiles="$cp4_dir/validation/route-profiles.json"
trace=${S24_CP5_TRACE:-"$s24_dir/deterministic-three-class.json"}
cuda_session_dir="$output_dir/cuda-workers"
op12_endpoint=${S24_OP12_ENDPOINT:-192.168.1.193:24280}
op15_endpoint=${S24_OP15_ENDPOINT:-192.168.1.97:24281}
cuda_prefix_endpoint=${S24_CUDA_PREFIX_ENDPOINT:-127.0.0.1:24180}
cuda_mid_endpoint=${S24_CUDA_MID_ENDPOINT:-127.0.0.1:24181}
cuda_tail_endpoint=${S24_CUDA_TAIL_ENDPOINT:-127.0.0.1:24182}
gpu_uuid=${S24_GPU_UUID:-GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08}
control_gather_us=250000

fail() {
    echo "error: $*" >&2
    exit 2
}

if [[ ${S24_CONFIRM_CP5:-NO} != YES ]]; then
    fail "set S24_CONFIRM_CP5=YES only after CP4 passes and fresh phone workers are ready"
fi
[[ ! -e "$output_dir" ]] || fail "output already exists: $output_dir"
for path in "$cp4_gate" "$profiles" "$trace"; do
    [[ -f "$path" ]] || fail "missing required CP5 input: $path"
done
cp4_status=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["status"])' \
    "$cp4_gate")
[[ "$cp4_status" == CP4_PHYSICAL_PASS ]] || fail "CP4 physical gate has not passed"
mkdir -p "$output_dir/runs"

campaign_complete=0
cuda_started=0
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "$campaign_complete" -eq 0 && "$cuda_started" -eq 1 ]]; then
        "$s24_dir/desktop_cuda_control.sh" stop "$cuda_session_dir" || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

op12_knee=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["selection"]["batch_knee"])' \
    "$cp4_dir/phone-knees/op12-prefix-knee.json")
op15_knee=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["selection"]["batch_knee"])' \
    "$cp4_dir/phone-knees/op15-mid-knee.json")
[[ "$op12_knee" =~ ^[1-4]$ ]] || fail "invalid OP12 knee: $op12_knee"
[[ "$op15_knee" =~ ^[1-4]$ ]] || fail "invalid OP15 knee: $op15_knee"

arrival_at_worker_us() {
    python3 -c '
import json,sys
run=json.load(open(sys.argv[1], encoding="ascii"))
event=min(run["batch_events"][sys.argv[2]], key=lambda row: row["dispatch_ns"])
enqueue_ns=event["dispatch_ns"]-event["max_queue_us"]*1000
print(max(0, (enqueue_ns-run["runtime"]["origin_monotonic_ns"])//1000))
' "$1" "$2"
}

r1_op15_arrival_us=$(arrival_at_worker_us \
    "$cp4_dir/runs/r1-b1-a/runtime.json" op15-mid)
r2_op15_arrival_us=$(arrival_at_worker_us \
    "$cp4_dir/runs/r2-b1-a/runtime.json" op15-mid)
op15_target_us=$((r1_op15_arrival_us > r2_op15_arrival_us \
    ? r1_op15_arrival_us : r2_op15_arrival_us))
r1_op15_delay_us=$((op15_target_us - r1_op15_arrival_us))
r2_op15_delay_us=$((op15_target_us - r2_op15_arrival_us))

"$s24_dir/desktop_cuda_control.sh" start "$cuda_session_dir" \
    > "$output_dir/cuda-start.stdout" \
    2> "$output_dir/cuda-start.stderr"
cuda_started=1

mapfile -t gpu_pids < <(
    for name in cuda-prefix cuda-mid cuda-tail; do
        sed -n '1p' "$cuda_session_dir/$name.pid"
    done
)
[[ "${#gpu_pids[@]}" -eq 3 ]] || fail "did not resolve three CUDA worker PIDs"
for pid in "${gpu_pids[@]}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || fail "invalid CUDA worker PID: $pid"
done

printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    case repetition control cuda_session_id phone_session_id session_end \
    > "$output_dir/sessions.tsv"

session_index=0
run_control() {
    local repetition=$1
    local control=$2
    local session_end=$3
    session_index=$((session_index + 1))
    local name="${control,,}-rep-${repetition}"
    local run_dir="$output_dir/runs/$name"
    mkdir -p "$run_dir"
    local command=(
        python3 "$s24_dir/fixed_diamond_runtime.py"
        --cuda-prefix "$cuda_prefix_endpoint"
        --cuda-mid "$cuda_mid_endpoint"
        --op12 "$op12_endpoint"
        --op15 "$op15_endpoint"
        --cuda-tail "$cuda_tail_endpoint"
        --trace "$trace"
        --control "$control"
        --output "$run_dir/runtime.json"
        --limit-per-route 2
        --route-delay-us "R1:$r1_op15_delay_us"
        --route-delay-us "R2:$r2_op15_delay_us"
        --session-end "$session_end"
        --op12-knee "$op12_knee"
        --op15-knee "$op15_knee"
        --op15-gather-us "$control_gather_us"
        --measure-energy
        --gpu-uuid "$gpu_uuid"
    )
    for pid in "${gpu_pids[@]}"; do
        command+=(--expected-gpu-pid "$pid")
    done
    if [[ "$control" == C3 ]]; then
        command+=(--profiles "$profiles" --allow-numeric-uncertified)
    elif [[ "$control" != C0 ]]; then
        command+=(--allow-numeric-uncertified)
    fi
    printf '%q ' "${command[@]}" > "$run_dir/command.txt"
    printf '\n' >> "$run_dir/command.txt"
    "${command[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$repetition" "$control" "$session_index" \
        "$session_index" "$session_end" >> "$output_dir/sessions.tsv"
}

run_control 1 C0 detach
run_control 1 C1 detach
run_control 1 C2 detach
run_control 1 C3 detach

run_control 2 C3 detach
run_control 2 C2 detach
run_control 2 C1 detach
run_control 2 C0 detach

run_control 3 C1 detach
run_control 3 C0 detach
run_control 3 C3 detach
run_control 3 C2 stop
cuda_started=0

for _attempt in $(seq 1 300); do
    cuda_status=$("$s24_dir/desktop_cuda_control.sh" status "$cuda_session_dir")
    if [[ "$cuda_status" != *live* ]]; then
        break
    fi
    sleep 0.1
done
"$s24_dir/desktop_cuda_control.sh" collect "$cuda_session_dir" \
    > "$output_dir/cuda-collect.stdout" \
    2> "$output_dir/cuda-collect.stderr"

{
    printf 'schema=s24-cp5-control-campaign-v1\n'
    printf 'trace=%s\n' "$trace"
    printf 'profiles=%s\n' "$profiles"
    printf 'selection=first_two_requests_per_route\n'
    printf 'equal_work_requests_per_run=6\n'
    printf 'repetitions_per_control=3\n'
    printf 'run_order=C0,C1,C2,C3,C3,C2,C1,C0,C1,C0,C3,C2\n'
    printf 'first_cuda_runtime_session=1\n'
    printf 'first_phone_runtime_session=1\n'
    printf 'op12_knee=%s\n' "$op12_knee"
    printf 'op15_knee=%s\n' "$op15_knee"
    printf 'control_gather_us=%s\n' "$control_gather_us"
    printf 'r1_op15_delay_us=%s\n' "$r1_op15_delay_us"
    printf 'r2_op15_delay_us=%s\n' "$r2_op15_delay_us"
    printf 'gpu_uuid=%s\n' "$gpu_uuid"
    printf 'energy_scope=RTX_4060_TI_GPU_BOARD_ONLY\n'
    printf 'runtime_activation_path=DESKTOP_DIRECT_WIFI\n'
    printf 'a6000_runtime_relay=FORBIDDEN\n'
    printf 'final_session_end=STOP\n'
} > "$output_dir/campaign.env"

(
    cd "$output_dir"
    find . -type f ! -name SHA256SUMS.pre-validation.txt -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum > SHA256SUMS.pre-validation.txt
)

campaign_complete=1
trap - EXIT INT TERM
printf '%s\n' "$output_dir"
