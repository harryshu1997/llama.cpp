#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
output_dir=${1:-"$s24_dir/results/cp4_fixed_diamond"}
trace=${S24_CP4_TRACE:-"$s24_dir/deterministic-three-class.json"}
cuda_session_dir="$output_dir/cuda-workers"
op12_endpoint=${S24_OP12_ENDPOINT:-192.168.1.193:24280}
op15_endpoint=${S24_OP15_ENDPOINT:-192.168.1.97:24281}
cuda_prefix_endpoint=${S24_CUDA_PREFIX_ENDPOINT:-127.0.0.1:24180}
cuda_mid_endpoint=${S24_CUDA_MID_ENDPOINT:-127.0.0.1:24181}
cuda_tail_endpoint=${S24_CUDA_TAIL_ENDPOINT:-127.0.0.1:24182}

fail() {
    echo "error: $*" >&2
    exit 2
}

if [[ ${S24_CONFIRM_CP4:-NO} != YES ]]; then
    fail "set S24_CONFIRM_CP4=YES after both A6000-controlled phone workers are ready"
fi
[[ ! -e "$output_dir" ]] || fail "output already exists: $output_dir"
[[ -f "$trace" ]] || fail "missing deterministic trace: $trace"
mkdir -p "$output_dir/runs" "$output_dir/activations"

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

"$s24_dir/run_phone_knees.sh" "$output_dir/phone-knees" \
    > "$output_dir/phone-knees.stdout" \
    2> "$output_dir/phone-knees.stderr"

read_knee() {
    python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["selection"]["batch_knee"])' \
        "$1"
}

op12_knee=$(read_knee "$output_dir/phone-knees/op12-prefix-knee.json")
op15_knee=$(read_knee "$output_dir/phone-knees/op15-mid-knee.json")
[[ "$op12_knee" =~ ^[1-4]$ ]] || fail "invalid OP12 knee: $op12_knee"
[[ "$op15_knee" =~ ^[1-4]$ ]] || fail "invalid OP15 knee: $op15_knee"
convergence_per_route=$(((op15_knee + 1) / 2))
convergence_gather_us=250000

"$s24_dir/desktop_cuda_control.sh" start "$cuda_session_dir" \
    > "$output_dir/cuda-start.stdout" \
    2> "$output_dir/cuda-start.stderr"
cuda_started=1

printf '%s\t%s\t%s\t%s\t%s\n' \
    case cuda_session_id phone_session_id session_end control \
    > "$output_dir/sessions.tsv"

session_index=0
run_case() {
    local name=$1
    local control=$2
    local session_end=$3
    local numeric_scope=$4
    shift 4
    session_index=$((session_index + 1))
    local run_dir="$output_dir/runs/$name"
    local activation_dir="$output_dir/activations/$name"
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
        --activation-dir "$activation_dir"
        --capture-boundaries
        --session-end "$session_end"
        --op12-knee "$op12_knee"
        --op15-knee "$op15_knee"
    )
    if [[ "$numeric_scope" == phone ]]; then
        command+=(--allow-numeric-uncertified)
    elif [[ "$numeric_scope" != cuda ]]; then
        fail "unknown numeric scope for $name: $numeric_scope"
    fi
    command+=("$@")
    printf '%q ' "${command[@]}" > "$run_dir/command.txt"
    printf '\n' >> "$run_dir/command.txt"
    "${command[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$session_index" "$((session_index + 1))" \
        "$session_end" "$control" >> "$output_dir/sessions.tsv"
}

run_case r0-b1-a C2 detach cuda --route-filter R0 --limit-per-route 1
run_case r0-b1-b C2 detach cuda --route-filter R0 --limit-per-route 1
run_case r1-b1-a C2 detach phone --route-filter R1 --limit-per-route 1
run_case r1-b1-b C2 detach phone --route-filter R1 --limit-per-route 1
run_case r2-b1-a C2 detach phone --route-filter R2 --limit-per-route 1
run_case r2-b1-b C2 detach phone --route-filter R2 --limit-per-route 1

arrival_at_worker_us() {
    python3 -c '
import json,sys
run=json.load(open(sys.argv[1], encoding="ascii"))
worker=sys.argv[2]
events=run["batch_events"][worker]
event=min(events, key=lambda row: row["dispatch_ns"])
enqueue_ns=event["dispatch_ns"]-event["max_queue_us"]*1000
print(max(0, (enqueue_ns-run["runtime"]["origin_monotonic_ns"])//1000))
' "$1" "$2"
}

r0_tail_arrival_us=$(arrival_at_worker_us \
    "$output_dir/runs/r0-b1-a/runtime.json" cuda-tail)
r1_tail_arrival_us=$(arrival_at_worker_us \
    "$output_dir/runs/r1-b1-a/runtime.json" cuda-tail)
r2_tail_arrival_us=$(arrival_at_worker_us \
    "$output_dir/runs/r2-b1-a/runtime.json" cuda-tail)
r1_op15_arrival_us=$(arrival_at_worker_us \
    "$output_dir/runs/r1-b1-a/runtime.json" op15-mid)
r2_op15_arrival_us=$(arrival_at_worker_us \
    "$output_dir/runs/r2-b1-a/runtime.json" op15-mid)

op15_target_us=$((r1_op15_arrival_us > r2_op15_arrival_us \
    ? r1_op15_arrival_us : r2_op15_arrival_us))
r1_op15_delay_us=$((op15_target_us - r1_op15_arrival_us))
r2_op15_delay_us=$((op15_target_us - r2_op15_arrival_us))

tail_target_us=$((r0_tail_arrival_us > r1_tail_arrival_us \
    ? r0_tail_arrival_us : r1_tail_arrival_us))
tail_target_us=$((tail_target_us > r2_tail_arrival_us \
    ? tail_target_us : r2_tail_arrival_us))
r0_tail_delay_us=$((tail_target_us - r0_tail_arrival_us))
r1_tail_delay_us=$((tail_target_us - r1_tail_arrival_us))
r2_tail_delay_us=$((tail_target_us - r2_tail_arrival_us))

run_case r1-cuda-control C0 detach cuda --route-filter R1 --limit-per-route 1
run_case r2-cuda-control C0 detach cuda --route-filter R2 --limit-per-route 1

run_case r2-b4 C2 detach phone --route-filter R2 --limit-per-route 4
run_case r1-r2-shared-op15 C2 detach phone \
    --route-filter R1 --route-filter R2 \
    --limit-per-route "$convergence_per_route" \
    --route-delay-us "R1:$r1_op15_delay_us" \
    --route-delay-us "R2:$r2_op15_delay_us" \
    --op15-gather-us "$convergence_gather_us"
run_case r0-r1-r2-shared-tail C2 stop phone \
    --limit-per-route "$convergence_per_route" \
    --route-delay-us "R0:$r0_tail_delay_us" \
    --route-delay-us "R1:$r1_tail_delay_us" \
    --route-delay-us "R2:$r2_tail_delay_us" \
    --op15-gather-us "$convergence_gather_us" \
    --cuda-tail-gather-us "$convergence_gather_us"
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
    printf 'schema=s24-cp4-campaign-v1\n'
    printf 'trace=%s\n' "$trace"
    printf 'phone_knee_sessions=1\n'
    printf 'first_cuda_runtime_session=1\n'
    printf 'first_phone_runtime_session=2\n'
    printf 'op12_knee=%s\n' "$op12_knee"
    printf 'op15_knee=%s\n' "$op15_knee"
    printf 'convergence_per_route=%s\n' "$convergence_per_route"
    printf 'convergence_gather_us=%s\n' "$convergence_gather_us"
    printf 'r1_op15_delay_us=%s\n' "$r1_op15_delay_us"
    printf 'r2_op15_delay_us=%s\n' "$r2_op15_delay_us"
    printf 'r0_tail_delay_us=%s\n' "$r0_tail_delay_us"
    printf 'r1_tail_delay_us=%s\n' "$r1_tail_delay_us"
    printf 'r2_tail_delay_us=%s\n' "$r2_tail_delay_us"
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
