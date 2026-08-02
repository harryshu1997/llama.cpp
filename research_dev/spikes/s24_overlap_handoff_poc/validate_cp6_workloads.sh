#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
campaign_dir=${1:-"$s24_dir/results/cp6_real_workloads"}
phone_log_dir=${2:-"$campaign_dir/phone-logs"}
validation_dir="$campaign_dir/validation"
cuda_log_dir="$campaign_dir/cuda-workers"

fail() {
    echo "error: $*" >&2
    exit 2
}

[[ -d "$campaign_dir" ]] || fail "missing CP6 campaign: $campaign_dir"
[[ -f "$campaign_dir/sessions.tsv" ]] || fail "missing CP6 session ledger"
[[ -d "$phone_log_dir" ]] || fail "missing synchronized A6000 phone logs: $phone_log_dir"
[[ ! -e "$validation_dir" ]] || fail "validation output already exists: $validation_dir"
for path in \
    "$cuda_log_dir/cuda-prefix.log" \
    "$cuda_log_dir/cuda-mid.log" \
    "$cuda_log_dir/cuda-tail.log" \
    "$phone_log_dir/OP12.log" \
    "$phone_log_dir/OP15.log"; do
    [[ -f "$path" ]] || fail "missing worker log: $path"
done
mkdir -p "$validation_dir/sessions"

expected_header=$'case\tcuda_session_id\tphone_session_id\tsession_end\tcontrol'
observed_header=$(head -n 1 "$campaign_dir/sessions.tsv")
[[ "$observed_header" == "$expected_header" ]] || fail "session ledger header differs"

case_count=0
while IFS=$'\t' read -r name cuda_session phone_session session_end control; do
    [[ -n "$name" ]] || continue
    [[ "$cuda_session" =~ ^[0-9]+$ ]] || fail "invalid CUDA session for $name"
    [[ "$phone_session" =~ ^[0-9]+$ ]] || fail "invalid phone session for $name"
    runtime="$campaign_dir/runs/$name/runtime.json"
    [[ -f "$runtime" ]] || fail "missing runtime report for $name"
    python3 "$s24_dir/validate_physical_session.py" \
        --runtime "$runtime" \
        --cuda-session-id "$cuda_session" \
        --phone-session-id "$phone_session" \
        --cuda-prefix-log "$cuda_log_dir/cuda-prefix.log" \
        --cuda-mid-log "$cuda_log_dir/cuda-mid.log" \
        --op12-log "$phone_log_dir/OP12.log" \
        --op15-log "$phone_log_dir/OP15.log" \
        --cuda-tail-log "$cuda_log_dir/cuda-tail.log" \
        --output "$validation_dir/sessions/$name.json" \
        > "$validation_dir/sessions/$name.stdout" \
        2> "$validation_dir/sessions/$name.stderr"
    case_count=$((case_count + 1))
done < <(tail -n +2 "$campaign_dir/sessions.tsv")
[[ "$case_count" -eq 2 ]] || fail "expected two CP6 sessions, found $case_count"

python3 "$s24_dir/verify_cp6.py" \
    --campaign "$campaign_dir" \
    --output "$validation_dir/workload-gates.json" \
    > "$validation_dir/workload-gates.stdout" \
    2> "$validation_dir/workload-gates.stderr"

(
    cd "$campaign_dir"
    find . -type f ! -name SHA256SUMS.txt -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum > SHA256SUMS.txt
)

printf '%s\n' "$validation_dir/workload-gates.json"
