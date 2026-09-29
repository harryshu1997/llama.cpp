#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <new-absolute-output-root> <adb-port>" >&2
    exit 2
fi

output=$1
adb_port=$2
if [[ $output != /* || -e $output ]]; then
    echo "output must be a new absolute path: $output" >&2
    exit 2
fi
if [[ ! $adb_port =~ ^[0-9]+$ ]]; then
    echo "invalid adb port: $adb_port" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
arm=$here/run_fp16_small_overlay_arm.sh
fitter=$here/fit_phase_contention_profile.py
base_profile=$here/SCHEDULER_PROFILE_COMPOSITE_USB_V2.json
for path in "$arm" "$fitter" "$base_profile"; do
    [[ -f $path ]] || {
        echo "missing contention calibration dependency: $path" >&2
        exit 1
    }
done

mkdir -p "$output"

run_static() {
    local name=$1
    local large_policy=$2
    local variant=$3
    echo "starting $name"
    S42_RUNTIME_PROFILE="$base_profile" \
    S42_FORCED_STATIC_ROUTE=desktop-cpu \
        "$arm" "$output/$name" "$large_policy" static-cpu \
        "$adb_port" "$variant"
}

# Symmetric order limits monotonic thermal and background-load drift.
run_static cpu-phase-r1 cpu-overflow phase-calibration
run_static cpu-natural-r1 cpu-overflow natural-validation
run_static op15-phase-r1 op15-assistance phase-calibration
run_static op15-natural-r1 op15-assistance natural-validation
run_static op15-natural-r2 op15-assistance natural-validation
run_static op15-phase-r2 op15-assistance phase-calibration
run_static cpu-natural-r2 cpu-overflow natural-validation
run_static cpu-phase-r2 cpu-overflow phase-calibration

python3 "$fitter" \
    --cpu-overflow-result "$output/cpu-phase-r1/combined/RESULT.json" \
    --cpu-overflow-result "$output/cpu-phase-r2/combined/RESULT.json" \
    --op15-result "$output/op15-phase-r1/combined/RESULT.json" \
    --op15-result "$output/op15-phase-r2/combined/RESULT.json" \
    --cpu-overflow-natural-result \
        "$output/cpu-natural-r1/combined/RESULT.json" \
    --cpu-overflow-natural-result \
        "$output/cpu-natural-r2/combined/RESULT.json" \
    --op15-natural-result \
        "$output/op15-natural-r1/combined/RESULT.json" \
    --op15-natural-result \
        "$output/op15-natural-r2/combined/RESULT.json" \
    --base-profile "$base_profile" \
    --output-profile "$output/PHASE_PROFILE_V2.json" \
    --output-audit "$output/PHASE_PROFILE_AUDIT_V2.json"
