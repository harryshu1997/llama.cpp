#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "usage: $0 <new-absolute-output-root> <adb-port>" \
        "<qualified-cpu-profile> <physical-shape-calibration>" \
        "<marginal-system-profile> <direct-energy-profile>" \
        "<precompiled-ffn-policy>" >&2
    exit 2
fi

output=$1
adb_port=$2
base_profile=$3
physical_calibration=$4
marginal_profile=$5
direct_energy_profile=$6
compiled_policy=$7
if [[ $output != /* || -e $output ]]; then
    echo "output must be a new absolute path: $output" >&2
    exit 2
fi
if [[ ! $adb_port =~ ^[0-9]+$ ]]; then
    echo "invalid adb port: $adb_port" >&2
    exit 2
fi
if [[ $base_profile != /* || ! -f $base_profile ]]; then
    echo "qualified CPU profile must be an existing absolute path" >&2
    exit 2
fi
if [[ $physical_calibration != /* || ! -f $physical_calibration ]]; then
    echo "physical calibration must be an existing absolute path" >&2
    exit 2
fi
for path in "$marginal_profile" "$direct_energy_profile" \
        "$compiled_policy"; do
    if [[ $path != /* || ! -f $path ]]; then
        echo "energy input must be an existing absolute path: $path" >&2
        exit 2
    fi
done

here=$(cd -- "$(dirname -- "$0")" && pwd)
arm=$here/run_fp16_small_overlay_arm.sh
materializer=$here/materialize_ffn_split_route.py
resident_layers=${S42_LLAMA_FFN_RESIDENT_LAYERS:-1}
natural_variant=natural-validation
if [[ ! $resident_layers =~ ^[0-9]+$ ]] \
        || (( resident_layers < 1 || resident_layers > 16 )); then
    echo "invalid S42_LLAMA_FFN_RESIDENT_LAYERS" >&2
    exit 2
fi
for path in "$arm" "$materializer"; do
    [[ -f $path ]] || {
        echo "missing split qualification dependency: $path" >&2
        exit 1
    }
done

mkdir -p "$output"

run_static() {
    local name=$1
    local route=$2
    local variant=$3
    local large_policy=${4:-cpu-overflow}
    local run=$output/$name
    echo "starting $name"
    S42_RUNTIME_PROFILE="$base_profile" \
    S42_MARGINAL_SYSTEM_PROFILE="$marginal_profile" \
    S42_FORCED_STATIC_ROUTE="$route" \
    S42_FFN_PHYSICAL_CALIBRATION="$physical_calibration" \
    S42_FFN_COMPILED_POLICY="$compiled_policy" \
    S42_LLAMA_FFN_RESIDENT_LAYERS="$resident_layers" \
        "$arm" "$run" "$large_policy" static-cpu "$adb_port" "$variant"
}

# Natural arrivals use A-B-B-A. The phase-anchored split runs are a separate
# train/holdout acquisition and are not folded into the paired energy result.
run_static cpu-natural-r1 desktop-cpu "$natural_variant"
run_static split-natural-r1 cpu-phone-ffn-split "$natural_variant"
run_static split-natural-r2 cpu-phone-ffn-split "$natural_variant"
run_static cpu-natural-r2 desktop-cpu "$natural_variant"
run_static split-calibration-r1 cpu-phone-ffn-split phase-calibration
run_static split-calibration-r2 cpu-phone-ffn-split phase-calibration
run_static op15-idle-r1 cpu-phone-ffn-split \
    idle-split-calibration op15-assistance
run_static op15-idle-r2 cpu-phone-ffn-split \
    idle-split-calibration op15-assistance

ffn_manifest=$output/split-calibration-r1/capture/LLAMA_FFN_MANIFEST.json
ffn_policy=$output/split-calibration-r1/capture/LLAMA_FFN_COMPILED_POLICY.json
profile=$output/SCHEDULER_PROFILE_WITH_FFN_SPLIT.json
audit=$output/FFN_SPLIT_ROUTE_CALIBRATION.json

materialize_command=(
    python3 "$materializer"
    --base-profile "$base_profile"
    --direct-energy-profile "$direct_energy_profile"
    --ffn-manifest "$ffn_manifest"
    --ffn-policy "$ffn_policy"
    --output-profile "$profile"
    --output-audit "$audit"
)
for repeat in 1 2; do
    materialize_command+=(
        --cpu-natural-result "$output/cpu-natural-r$repeat/combined/RESULT.json"
        --cpu-natural-phone "$output/cpu-natural-r$repeat/capture/PHONE_ENERGY.json"
        --cpu-natural-qualification "$output/cpu-natural-r$repeat/capture/QUALIFICATION.json"
        --split-natural-result "$output/split-natural-r$repeat/combined/RESULT.json"
        --split-natural-phone "$output/split-natural-r$repeat/capture/PHONE_ENERGY.json"
        --split-natural-qualification "$output/split-natural-r$repeat/capture/QUALIFICATION.json"
        --split-natural-log "$output/split-natural-r$repeat/combined/llama1-cpu-phone-ffn.stderr"
        --split-calibration-result "$output/split-calibration-r$repeat/combined/RESULT.json"
        --split-calibration-phone "$output/split-calibration-r$repeat/capture/PHONE_ENERGY.json"
        --split-calibration-qualification "$output/split-calibration-r$repeat/capture/QUALIFICATION.json"
        --split-calibration-log "$output/split-calibration-r$repeat/combined/llama1-cpu-phone-ffn.stderr"
        --op15-idle-result "$output/op15-idle-r$repeat/combined/RESULT.json"
        --op15-idle-phone "$output/op15-idle-r$repeat/capture/PHONE_ENERGY.json"
        --op15-idle-qualification "$output/op15-idle-r$repeat/capture/QUALIFICATION.json"
        --op15-idle-log "$output/op15-idle-r$repeat/combined/llama1-cpu-phone-ffn.stderr"
    )
done
"${materialize_command[@]}"

runtime=$output/runtime-selected-split
S42_RUNTIME_PROFILE="$profile" \
S42_MARGINAL_SYSTEM_PROFILE="$marginal_profile" \
S42_REQUIRED_RUNTIME_ROUTE=cpu-phone-ffn-split \
S42_FORCED_STATIC_ROUTE=desktop-cpu \
S42_FFN_PHYSICAL_CALIBRATION="$physical_calibration" \
S42_FFN_COMPILED_POLICY="$compiled_policy" \
S42_LLAMA_FFN_RESIDENT_LAYERS="$resident_layers" \
    "$arm" "$runtime" cpu-overflow runtime-scheduler \
    "$adb_port" "$natural_variant"

python3 - "$runtime/combined/RESULT.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="ascii"))
count = result["scheduler_runtime"]["route_counts"].get(
    "cpu-phone-ffn-split", 0
)
if count <= 0:
    raise SystemExit("runtime scheduler selected no physical FFN split request")
print(json.dumps({
    "physical_split_requests": count,
    "result": sys.argv[1],
    "status": "PASS",
}, sort_keys=True))
PY
