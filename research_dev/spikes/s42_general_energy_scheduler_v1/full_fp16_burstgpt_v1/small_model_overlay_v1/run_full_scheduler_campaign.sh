#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <new-absolute-output-root> <adb-port>" \
        "<physical-shape-calibration> <initial-marginal-profile>" >&2
    exit 2
fi

output=$1
adb_port=$2
physical_calibration=$3
initial_marginal=$4
resident_layers=${S42_LLAMA_FFN_RESIDENT_LAYERS:-1}
if [[ $output != /* || -e $output ]]; then
    echo "output must be a new absolute path: $output" >&2
    exit 2
fi
if [[ ! $adb_port =~ ^[0-9]+$ ]]; then
    echo "invalid adb port: $adb_port" >&2
    exit 2
fi
if [[ ! $resident_layers =~ ^[0-9]+$ ]] \
        || (( resident_layers < 1 || resident_layers > 16 )); then
    echo "invalid S42_LLAMA_FFN_RESIDENT_LAYERS" >&2
    exit 2
fi
for path in "$physical_calibration" "$initial_marginal"; do
    if [[ $path != /* || ! -f $path ]]; then
        echo "campaign input must be an existing absolute file: $path" >&2
        exit 2
    fi
done

here=$(cd -- "$(dirname -- "$0")" && pwd)
phase_runner=$here/run_phase_contention_calibration_abba.sh
qualification_runner=$here/run_full_scheduler_qualification_pipeline.sh
arm=$here/run_fp16_small_overlay_arm.sh
for path in "$phase_runner" "$qualification_runner" "$arm"; do
    [[ -f $path ]] || {
        echo "missing campaign dependency: $path" >&2
        exit 1
    }
done

mkdir -p "$output"
direct_root=$output/direct-energy
phase_root=$output/phase-contention
qualification_root=$output/qualification

set +e
S42_FFN_PHYSICAL_CALIBRATION="$physical_calibration" \
S42_LLAMA_FFN_RESIDENT_LAYERS="$resident_layers" \
S42_MARGINAL_SYSTEM_PROFILE="$initial_marginal" \
    bash "$arm" "$direct_root" cpu-overflow static-cpu "$adb_port" \
    direct-energy-calibration
direct_rc=$?
set -e
direct_energy_profile=$direct_root/capture/LLAMA_FFN_DIRECT_ENERGY.json
compiled_policy=$direct_root/capture/LLAMA_FFN_COMPILED_POLICY.json
if [[ ! -f $direct_energy_profile || ! -f $compiled_policy ]]; then
    echo "direct energy acquisition did not produce bounded artifacts" >&2
    exit 1
fi
direct_status=$(python3 - "$direct_energy_profile" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="ascii"))
if value.get("schema") != "s42-llama1b-ffn-direct-energy-v1":
    raise SystemExit("invalid direct energy artifact")
print(value.get("status", ""))
PY
)
if [[ $direct_status == PASS && $direct_rc -ne 0 ]]; then
    echo "qualified direct energy acquisition exited unsuccessfully" >&2
    exit 1
fi
if [[ $direct_status == FAIL && $direct_rc -ne 2 ]]; then
    echo "rejected direct energy acquisition has unexpected status" >&2
    exit 1
fi
if [[ $direct_status != PASS && $direct_status != FAIL ]]; then
    echo "invalid direct energy qualification status" >&2
    exit 1
fi

bash "$phase_runner" "$phase_root" "$adb_port"
S42_LLAMA_FFN_RESIDENT_LAYERS="$resident_layers" \
    bash "$qualification_runner" "$qualification_root" "$adb_port" \
    "$phase_root" - "$physical_calibration" "$initial_marginal" \
    "$direct_energy_profile" "$compiled_policy"
