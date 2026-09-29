#!/bin/bash
# Post-run analysis of a set of arms on the desktop (reads run artifacts only).
#   bash post_eval.sh OUT_PREFIX BASELINE_LABEL label=inputs-dir ...
# Runs, per arm set: compare_trace_energy, analyze_int2 (energy, calls per device/session, layout timeline,
# token identity vs the baseline), analyze_allon (loads, pairs, dispatch, probe reasons), interval_energy_int2
# (per-model windows), analyze_per_device (device sets per batch, drops) and analyze_phone_power (1 Hz sysfs
# diagnostic), plus the per-helper per-layer RPC by batch from the Qwen server stderr.
set -u
R=/mnt/storage/s43-two-phone-eval-20260925
A=$R/analysis/reports
OUT=$1; BASE=$2; shift 2
ARMS=(); RUNS=(); POWER=()
for spec in "$@"; do
  label=${spec%%=*}; dir=${spec#*=}
  ARMS+=(--arm "$label=$dir/run-eval/run"); RUNS+=(--run "$label=$dir/run-eval/run")
  POWER+=(--run "$label=$dir")
done
mkdir -p "$(dirname "$OUT")"
python3 $A/20260921-fast-path-trace-v2a/compare_trace_energy.py "${RUNS[@]}" --out "${OUT}_ENERGY.json" > "${OUT}_ENERGY.txt" 2>&1
python3 $A/20260924-pixel-integration-2/analyze_int2.py "${ARMS[@]}" --baseline "$BASE" --out "${OUT}_ANALYSIS.json" > "${OUT}_ANALYSIS.txt" 2>&1
(cd $A/20260924-coherent-policy-coalesced && python3 analyze_allon.py "${ARMS[@]}" --out "${OUT}_ALLON_STYLE.json" --md "${OUT}_ALLON_STYLE.md" > /dev/null 2>"${OUT}_ALLON_STYLE.err")
(cd $A/20260924-pixel-integration-2 && python3 interval_energy_int2.py "${ARMS[@]}" --out "${OUT}_MODEL_WINDOW_ENERGY.json" > /dev/null 2>"${OUT}_MODEL_WINDOW_ENERGY.err")
python3 $A/20260925-two-phone-eval/analyze_phone_power.py "${POWER[@]}" --out "${OUT}_PHONE_POWER.json" > /dev/null 2>"${OUT}_PHONE_POWER.err"
for spec in "$@"; do
  label=${spec%%=*}; dir=${spec#*=}
  python3 $A/20260924-per-device-policies/analyze_per_device.py "$dir/run-eval/run" --label "$label" \
    --out "${OUT}_PER_DEVICE_${label}.json" > "${OUT}_PER_DEVICE_${label}.txt" 2>&1
  for f in "$dir"/run-eval/run/large-model-*-hot-*.stderr; do
    [ -f "$f" ] || continue
    grep -a S41SERVERFFNSHAPE "$f" | python3 -c '
import sys, json
for line in sys.stdin:
    d = json.loads(line.split(" ", 1)[1])
    print(sys.argv[1], sys.argv[2], d.get("helper", "op15"), "B" + str(d["tokens"]), d["calls"], round(d["rpc_mean_ms"], 1), round(d["compute_mean_ms"], 1))
' "$label" "$(basename "$f")"
  done
done > "${OUT}_SHAPES.txt" 2>&1
echo "done ${OUT}"
