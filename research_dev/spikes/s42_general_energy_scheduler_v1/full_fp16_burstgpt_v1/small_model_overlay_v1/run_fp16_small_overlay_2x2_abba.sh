#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <new-absolute-output-root> <adb-port>" \
        "<qualified-runtime-profile> <marginal-system-profile>" >&2
    exit 2
fi

output=$1
adb_port=$2
runtime_profile=$3
marginal_profile=$4
overlay_variant=natural-validation
if [[ $output != /* || -e $output ]]; then
    echo "output must be a new absolute path: $output" >&2
    exit 2
fi
if [[ ! $adb_port =~ ^[0-9]+$ ]]; then
    echo "invalid adb port: $adb_port" >&2
    exit 2
fi
if [[ $runtime_profile != /* || ! -f $runtime_profile ]]; then
    echo "runtime profile must be an existing absolute path" >&2
    exit 2
fi
if [[ $marginal_profile != /* || ! -f $marginal_profile ]]; then
    echo "marginal profile must be an existing absolute path" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
arm=$here/run_fp16_small_overlay_arm.sh
abba=$here/compare_fp16_small_overlay_abba.py
factorial=$here/compare_fp16_small_overlay_2x2.py
marginal_fitter=$here/fit_marginal_system_profile.py
for path in "$arm" "$abba" "$factorial" "$marginal_fitter"; do
    [[ -f $path ]] || {
        echo "missing experiment dependency: $path" >&2
        exit 1
    }
done

mkdir -p "$output"

run_cell() {
    local name=$1
    local large_policy=$2
    local small_policy=$3
    local run=$output/$name
    echo "starting $name"
    S42_RUNTIME_PROFILE="$runtime_profile" \
    S42_MARGINAL_SYSTEM_PROFILE="$marginal_profile" \
    S42_FORCED_STATIC_ROUTE=desktop-cpu \
        "$arm" "$run" "$large_policy" "$small_policy" \
        "$adb_port" "$overlay_variant"
    for receipt in \
            "$run/combined/RESULT.json" \
            "$run/capture/PHONE_ENERGY.json" \
            "$run/capture/QUALIFICATION.json"; do
        [[ -s $receipt ]] || {
            echo "missing completed receipt: $receipt" >&2
            exit 1
        }
    done
}

run_cell cpu-static-r1 cpu-overflow static-cpu
run_cell cpu-runtime-r1 cpu-overflow runtime-scheduler
run_cell op15-static-r1 op15-assistance static-cpu
run_cell op15-runtime-r1 op15-assistance runtime-scheduler
run_cell op15-runtime-r2 op15-assistance runtime-scheduler
run_cell op15-static-r2 op15-assistance static-cpu
run_cell cpu-runtime-r2 cpu-overflow runtime-scheduler
run_cell cpu-static-r2 cpu-overflow static-cpu

python3 "$abba" \
    --static-r1-result "$output/cpu-static-r1/combined/RESULT.json" \
    --static-r1-phone "$output/cpu-static-r1/capture/PHONE_ENERGY.json" \
    --static-r1-qualification "$output/cpu-static-r1/capture/QUALIFICATION.json" \
    --runtime-r1-result "$output/cpu-runtime-r1/combined/RESULT.json" \
    --runtime-r1-phone "$output/cpu-runtime-r1/capture/PHONE_ENERGY.json" \
    --runtime-r1-qualification "$output/cpu-runtime-r1/capture/QUALIFICATION.json" \
    --runtime-r2-result "$output/cpu-runtime-r2/combined/RESULT.json" \
    --runtime-r2-phone "$output/cpu-runtime-r2/capture/PHONE_ENERGY.json" \
    --runtime-r2-qualification "$output/cpu-runtime-r2/capture/QUALIFICATION.json" \
    --static-r2-result "$output/cpu-static-r2/combined/RESULT.json" \
    --static-r2-phone "$output/cpu-static-r2/capture/PHONE_ENERGY.json" \
    --static-r2-qualification "$output/cpu-static-r2/capture/QUALIFICATION.json" \
    --output "$output/CPU_OVERFLOW_ABBA.json"

python3 "$abba" \
    --static-r1-result "$output/op15-static-r1/combined/RESULT.json" \
    --static-r1-phone "$output/op15-static-r1/capture/PHONE_ENERGY.json" \
    --static-r1-qualification "$output/op15-static-r1/capture/QUALIFICATION.json" \
    --runtime-r1-result "$output/op15-runtime-r1/combined/RESULT.json" \
    --runtime-r1-phone "$output/op15-runtime-r1/capture/PHONE_ENERGY.json" \
    --runtime-r1-qualification "$output/op15-runtime-r1/capture/QUALIFICATION.json" \
    --runtime-r2-result "$output/op15-runtime-r2/combined/RESULT.json" \
    --runtime-r2-phone "$output/op15-runtime-r2/capture/PHONE_ENERGY.json" \
    --runtime-r2-qualification "$output/op15-runtime-r2/capture/QUALIFICATION.json" \
    --static-r2-result "$output/op15-static-r2/combined/RESULT.json" \
    --static-r2-phone "$output/op15-static-r2/capture/PHONE_ENERGY.json" \
    --static-r2-qualification "$output/op15-static-r2/capture/QUALIFICATION.json" \
    --output "$output/OP15_ASSISTANCE_ABBA.json"

python3 "$factorial" \
    --cpu-overflow-abba "$output/CPU_OVERFLOW_ABBA.json" \
    --op15-assistance-abba "$output/OP15_ASSISTANCE_ABBA.json" \
    --output "$output/FACTORIAL_2X2_ABBA.json"

python3 "$marginal_fitter" \
    --experiment-root "$output" \
    --output "$output/NEXT_MARGINAL_SYSTEM_PROFILE.json"
