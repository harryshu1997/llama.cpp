#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$ROOT/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
PY=${PYTHON:-python}

# ---- E1 must be byte-identical BEFORE and AFTER E2 runs ----------------------
E1_MANIFEST="$ROOT/E1_BASELINE_MANIFEST.txt"
check_e1() {
    local when="$1"
    ( cd "$REPO" && sha256sum -c "$E1_MANIFEST" >/dev/null 2>&1 ) || {
        echo "E1 BASELINE CHANGED ($when): the frozen E1 tree is not byte-identical" >&2
        ( cd "$REPO" && sha256sum -c "$E1_MANIFEST" 2>&1 | grep -v ': OK' | head -5 ) >&2
        exit 1; }
    echo "E1 baseline byte-identical ($when): $(wc -l < "$E1_MANIFEST") files"
}
check_e1 "before"

# ---- fixtures are a deterministic function of their generator ----------------
"$PY" "$ROOT/tests/make_e2_fixtures.py" --check

# ---- draft-2020 schema validation of every fixture ---------------------------
/usr/bin/jsonschema -i "$ROOT/fixtures/control_timeline.json" \
    "$ROOT/schemas/realized_timeline.v1.schema.json"
/usr/bin/jsonschema -i "$ROOT/fixtures/treatment_timeline.json" \
    "$ROOT/schemas/realized_timeline.v1.schema.json"
/usr/bin/jsonschema -i "$ROOT/fixtures/repetition_set.json" \
    "$ROOT/schemas/repetition_set.v1.schema.json"
/usr/bin/jsonschema -i "$ROOT/fixtures/wall_capability.json" \
    "$ROOT/schemas/server_wall_capability.v1.schema.json"

# ---- unit and adversarial suite ---------------------------------------------
"$PY" -m unittest discover -s "$ROOT/tests" -p 'test_e2.py' -v

# ---- the synthetic pair emits NO physical result -----------------------------
OUT=$("$PY" "$ROOT/src/comparator.py" \
    --control "$ROOT/fixtures/control_timeline.json" \
    --treatment "$ROOT/fixtures/treatment_timeline.json" \
    --trusted-root "$ROOT/fixtures" 2>/dev/null || true)
echo "$OUT" | "$PY" -c '
import json, sys
doc = json.load(sys.stdin)
assert doc["result_label"] == "MEASUREMENT_INVALID", doc
assert doc["reason_code"] == "SYNTHETIC_NO_PHYSICAL_CLAIM", doc
assert doc["detail"]["relief"] is True, "the arithmetic should clear the bar"
print("synthetic pair: relief arithmetic true, label", doc["result_label"],
      "-", doc["reason_code"])
'

# ---- the existing A6000 trace is rejected, not imported ----------------------
TRACE="$ROOT/../s10_power_frontier/artifacts/cp2_a6000_power_trace.csv"
if [ -f "$TRACE" ]; then
    REPORT=$("$PY" "$ROOT/src/import_nvml_trace.py" --csv "$TRACE" --json || true)
    echo "$REPORT" | "$PY" -c '
import json, sys
r = json.load(sys.stdin)
assert r["accepted"] is False, r
assert r["independent_updates"] == 57, r
assert r["rows"] == 323, r
assert r["scope"] == "GPU_BOARD", r
joined = " ".join(r["failures"])
for code in ("E_UPDATES", "E_STATUS_CHANGE", "E_PAIRS"):
    assert code in joined, (code, r["failures"])
print("existing A6000 trace: rows", r["rows"], "independent_updates",
      r["independent_updates"], "-> REJECTED")
'
fi

# ---- negative CLI cases ------------------------------------------------------
PYTHON="$PY" bash "$ROOT/scripts/e2_negative.sh"

# ---- determinism across processes and hash seeds -----------------------------
REF=""
for seed in 0 1 42 12345 random; do
    DIG=$(PYTHONHASHSEED=$seed "$PY" - "$ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root / "src")]
import e2_canon as canon
import comparator
control = canon.load_strict(str(root / "fixtures" / "control_timeline.json"))
treatment = canon.load_strict(str(root / "fixtures" / "treatment_timeline.json"))
label, reason, detail, _f = comparator.compare(control, treatment,
                                               root / "fixtures")
print(canon.digest({"label": label, "reason": reason, "detail": detail}))
PY
)
    if [ -z "$REF" ]; then REF="$DIG"; elif [ "$REF" != "$DIG" ]; then
        echo "E2 output is not deterministic at PYTHONHASHSEED=$seed" >&2
        exit 1; fi
done
echo "E2 determinism across 5 processes/seeds: $REF"

check_e1 "after"

find "$ROOT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -name '*.pyc' -delete 2>/dev/null || true

echo S10_E2_MATCHED_TIMELINE_TESTS_PASS
