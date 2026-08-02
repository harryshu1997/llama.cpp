#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
s24_dir="$repo_root/research_dev/spikes/s24_overlap_handoff_poc"
campaign_dir=${1:-"$s24_dir/results/cp4_fixed_diamond"}
phone_log_dir=${2:-"$campaign_dir/phone-logs"}
validation_dir="$campaign_dir/validation"
cuda_log_dir="$campaign_dir/cuda-workers"

fail() {
    echo "error: $*" >&2
    exit 2
}

[[ -d "$campaign_dir" ]] || fail "missing CP4 campaign: $campaign_dir"
[[ -f "$campaign_dir/sessions.tsv" ]] || fail "missing CP4 session ledger"
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
mkdir -p "$validation_dir/sessions" "$validation_dir/comparisons"

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
[[ "$case_count" -eq 11 ]] || fail "expected 11 CP4 sessions, found $case_count"

compare() {
    local reference=$1
    local candidate=$2
    local label=$3
    python3 "$s24_dir/compare_physical_runs.py" \
        --reference "$campaign_dir/runs/$reference/runtime.json" \
        --candidate "$campaign_dir/runs/$candidate/runtime.json" \
        --label "$label" \
        --output "$validation_dir/comparisons/$label.json" \
        > "$validation_dir/comparisons/$label.stdout" \
        2> "$validation_dir/comparisons/$label.stderr"
}

compare r0-b1-a r0-b1-b r0-repeat
compare r1-b1-a r1-b1-b r1-repeat
compare r2-b1-a r2-b1-b r2-repeat
compare r1-cuda-control r1-b1-a r1-vs-cuda
compare r2-cuda-control r2-b1-a r2-vs-cuda

python3 "$s24_dir/build_route_profiles.py" \
    --r0-run "$campaign_dir/runs/r0-b1-a/runtime.json" \
    --r0-validation "$validation_dir/sessions/r0-b1-a.json" \
    --r0-run "$campaign_dir/runs/r0-b1-b/runtime.json" \
    --r0-validation "$validation_dir/sessions/r0-b1-b.json" \
    --r1-run "$campaign_dir/runs/r1-b1-a/runtime.json" \
    --r1-validation "$validation_dir/sessions/r1-b1-a.json" \
    --r1-run "$campaign_dir/runs/r1-b1-b/runtime.json" \
    --r1-validation "$validation_dir/sessions/r1-b1-b.json" \
    --r2-run "$campaign_dir/runs/r2-b1-a/runtime.json" \
    --r2-validation "$validation_dir/sessions/r2-b1-a.json" \
    --r2-run "$campaign_dir/runs/r2-b1-b/runtime.json" \
    --r2-validation "$validation_dir/sessions/r2-b1-b.json" \
    --calibration-output "$validation_dir/route-calibration.json" \
    --profiles-output "$validation_dir/route-profiles.json" \
    > "$validation_dir/route-profiles.stdout" \
    2> "$validation_dir/route-profiles.stderr"

python3 "$s24_dir/verify_cp4.py" \
    --campaign "$campaign_dir" \
    --validation-dir "$validation_dir" \
    --output "$validation_dir/cp4-gates.json" \
    > "$validation_dir/cp4-gates.stdout" \
    2> "$validation_dir/cp4-gates.stderr"

(
    cd "$campaign_dir"
    find . -type f ! -name SHA256SUMS.txt -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum > SHA256SUMS.txt
)

printf '%s\n' "$validation_dir/cp4-gates.json"
