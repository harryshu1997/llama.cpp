#!/usr/bin/env bash
# Post-run analysis of the S43 dev-trace pair (desktop files only; no phone access).
set -uo pipefail
A=/mnt/storage/s43-dual-prep/analysis
B=/home/zhihao/s42-trace-longtaildev-baseline-20260923-inputs/run-baseline-${1:-1}
T=/home/zhihao/s42-trace-longtaildev-treatment-20260923-inputs/run-treatment-${2:-1}
OUT=$A/pair-${1:-1}-${2:-1}
mkdir -p "$OUT"
cd "$A"
for d in "$B" "$T"; do
  echo "=== $d"; cat "$d/runner.log" 2>/dev/null | tail -2 | cut -c1-600
  ls "$d/run/FAILURE.json" "$d/FAILURE.json" 2>/dev/null && cat "$d"/run/FAILURE.json "$d"/FAILURE.json 2>/dev/null | head -c 3000
done
echo "=== energy"
python3 compare_trace_energy.py --run baseline="$B/run" --run dual="$T/run" --out "$OUT/energy.json" | tee "$OUT/energy.txt"
echo "=== exactness"
rm -f "$OUT/pair.json"
python3 analyze_longdecode_pair.py --baseline "$B/run" --treatment "$T/run" --output "$OUT/pair.json" > "$OUT/pair.stdout" 2>&1; echo "analyze exit=$?"
python3 - "$OUT/pair.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("pair.json unavailable:", e); raise SystemExit
print("status", d["status"], "checks", d["checks"])
print("host_saving_percent %.2f duration_change_percent %.2f identical %d" % (d["host_saving_percent"], d["duration_change_percent"], d["identical_outputs"]))
print("differences", json.dumps(d["output_differences"]))
print("treatment_by_model", json.dumps(d["treatment_by_model"]))
PY
echo "=== phone session logs"
for d in "$B" "$T"; do
  name=$(basename "$d"); mkdir -p "$OUT/$name-phone"
  if [ -s "$d/phone-session-root.tar" ]; then tar -xf "$d/phone-session-root.tar" -C "$OUT/$name-phone" 2>/dev/null; fi
  echo "$name: $(find "$OUT/$name-phone" -name worker.log | wc -l) worker.log, $(find "$OUT/$name-phone" -name router.log | wc -l) router.log"
done
TW=$(find "$OUT/$(basename "$T")-phone" -name worker.log | sort)
args=(); for f in $TW; do args+=(--worker-log "$f"); done
echo "=== treatment timings"
python3 extract_phone_timings.py --run-dir "$T" "${args[@]}" --reference 17408=9.72 --reference 15360=6.59 \
  --reference 13056=6.643 --reference 8704=4.404 --reference 4352=2.303 \
  --reference 11520=4.915 --reference 7680=3.309 --reference 3840=1.694 --out "$OUT/treatment_timings.json" | tee "$OUT/treatment_timings.md"
echo "=== baseline host shapes (expect none)"
python3 extract_phone_timings.py --run-dir "$B" --out "$OUT/baseline_timings.json" | head -12
echo "=== dual config / warmup lines per worker log"
for f in $TW; do echo "-- $f"; grep -a "dual secondary=\|dual warmup\|ready backend\|dual execution failed\|secondary.*failed\|RESIDENTSHARDS" "$f" | cut -c1-300; done
echo "=== phone power diagnostics summary"
python3 - "$T/run/phone-power-diagnostics.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("unavailable", e); raise SystemExit
print({k: (v if not isinstance(v, (list, dict)) else type(v).__name__ + "[%d]" % len(v)) for k, v in d.items()})
PY
