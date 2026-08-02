#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
output_dir=${1:-"$s24_dir/results/cp6_real_workloads"}
cp4_dir=${S24_CP4_DIR:-"$s24_dir/results/cp4_fixed_diamond"}
cp5_dir=${S24_CP5_DIR:-"$s24_dir/results/cp5_controls"}
cp4_gate="$cp4_dir/validation/cp4-gates.json"
cp5_gate="$cp5_dir/validation/benefit-gates.json"
profiles="$cp4_dir/validation/route-profiles.json"
dense_trace=${S24_DENSE_TRACE:-"$s24_dir/burstgpt-dense-mechanics.json"}
observed_trace=${S24_OBSERVED_TRACE:-"$s24_dir/burstgpt-observed-context-600.json"}
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

if [[ ${S24_CONFIRM_CP6:-NO} != YES ]]; then
    fail "set S24_CONFIRM_CP6=YES only after CP5 is evaluated and fresh phone workers are ready"
fi
[[ ! -e "$output_dir" ]] || fail "output already exists: $output_dir"
for path in \
    "$cp4_gate" "$cp5_gate" "$profiles" "$dense_trace" "$observed_trace"; do
    [[ -f "$path" ]] || fail "missing required CP6 input: $path"
done
cp4_status=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["status"])' \
    "$cp4_gate")
[[ "$cp4_status" == CP4_PHYSICAL_PASS ]] || fail "CP4 physical gate has not passed"
cp5_status=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["status"])' \
    "$cp5_gate")
[[ "$cp5_status" == BENEFIT_GATES_PASS || "$cp5_status" == BENEFIT_GATE_FAIL ]] || \
    fail "CP5 benefit controls have not been evaluated"
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

"$s24_dir/desktop_cuda_control.sh" start "$cuda_session_dir" \
    > "$output_dir/cuda-start.stdout" \
    2> "$output_dir/cuda-start.stderr"
cuda_started=1

printf '%s\t%s\t%s\t%s\t%s\n' \
    case cuda_session_id phone_session_id session_end control \
    > "$output_dir/sessions.tsv"

run_workload() {
    local name=$1
    local trace=$2
    local control=$3
    local session_id=$4
    local session_end=$5
    shift 5
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
        --session-end "$session_end"
        --op12-knee "$op12_knee"
        --op15-knee "$op15_knee"
        --allow-numeric-uncertified
    )
    if [[ "$control" == C3 ]]; then
        command+=(--profiles "$profiles")
    fi
    command+=("$@")
    printf '%q ' "${command[@]}" > "$run_dir/command.txt"
    printf '\n' >> "$run_dir/command.txt"
    "${command[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$session_id" "$session_id" "$session_end" "$control" \
        >> "$output_dir/sessions.tsv"
}

run_workload dense-mechanics "$dense_trace" C3 1 detach
run_workload observed-context-600 "$observed_trace" C2 2 stop \
    --prefill-chunk 64
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
    printf 'schema=s24-cp6-workload-campaign-v1\n'
    printf 'dense_trace=%s\n' "$dense_trace"
    printf 'dense_control=C3_SLO_SELECTED\n'
    printf 'observed_trace=%s\n' "$observed_trace"
    printf 'observed_control=C2_PINNED_ROUTE_HINTS\n'
    printf 'observed_prefill_chunk=64\n'
    printf 'real_arrival_delay_override=NONE\n'
    printf 'op12_knee=%s\n' "$op12_knee"
    printf 'op15_knee=%s\n' "$op15_knee"
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
