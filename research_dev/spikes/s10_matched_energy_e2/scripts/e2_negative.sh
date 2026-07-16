#!/usr/bin/env bash
# Invalid E2 CLI cases. Each must exit nonzero with a stable diagnostic prefix and
# NO Python traceback. A traceback means an unhandled path, which is where
# fail-open bugs hide.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
WORK=$(mktemp -d /tmp/s10_e2_neg.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

PY=${PYTHON:-python}
C="$ROOT/fixtures/control_timeline.json"
T="$ROOT/fixtures/treatment_timeline.json"
FAILURES=0
CASES=0

expect_fail() {
    local name="$1"; shift
    CASES=$((CASES + 1))
    local out; local rc
    out=$("$@" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
        echo "NEGATIVE CASE PASSED WHEN IT MUST FAIL: $name" >&2
        FAILURES=$((FAILURES + 1)); return
    fi
    if ! grep -qE '^(E2_FAIL|E2_IMPORT_REJECTED|E2_FIXTURE_DRIFT|usage:|.*error:)' \
            <<<"$out"; then
        echo "case $name exited $rc without a stable diagnostic:" >&2
        head -3 <<<"$out" >&2
        FAILURES=$((FAILURES + 1)); return
    fi
    if grep -q 'Traceback (most recent call last)' <<<"$out"; then
        echo "case $name leaked a traceback:" >&2
        head -5 <<<"$out" >&2
        FAILURES=$((FAILURES + 1)); return
    fi
}

CMP=("$PY" "$ROOT/src/comparator.py")

# --- missing / unreadable inputs ---------------------------------------------
expect_fail "missing control" "${CMP[@]}" --control "$WORK/nope.json" \
    --treatment "$T" --trusted-root "$ROOT/fixtures"
expect_fail "missing treatment" "${CMP[@]}" --control "$C" \
    --treatment "$WORK/nope.json" --trusted-root "$ROOT/fixtures"
expect_fail "control is a directory" "${CMP[@]}" --control "$WORK" \
    --treatment "$T" --trusted-root "$ROOT/fixtures"
expect_fail "no trusted root argument" "${CMP[@]}" --control "$C" --treatment "$T"

# --- malformed JSON -----------------------------------------------------------
printf '{"a": 1' > "$WORK/truncated.json"
expect_fail "truncated JSON" "${CMP[@]}" --control "$WORK/truncated.json" \
    --treatment "$T" --trusted-root "$ROOT/fixtures"
printf '{"a": 1, "a": 2}' > "$WORK/dupkeys.json"
expect_fail "duplicate keys" "${CMP[@]}" --control "$WORK/dupkeys.json" \
    --treatment "$T" --trusted-root "$ROOT/fixtures"
printf '{"energy_nj": NaN}' > "$WORK/nan.json"
expect_fail "NaN constant" "${CMP[@]}" --control "$WORK/nan.json" \
    --treatment "$T" --trusted-root "$ROOT/fixtures"
printf '{"energy_nj": Infinity}' > "$WORK/inf.json"
expect_fail "Infinity constant" "${CMP[@]}" --control "$WORK/inf.json" \
    --treatment "$T" --trusted-root "$ROOT/fixtures"
printf '[]' > "$WORK/array.json"
expect_fail "top-level array" "${CMP[@]}" --control "$WORK/array.json" \
    --treatment "$T" --trusted-root "$ROOT/fixtures"

# --- semantic mutations -------------------------------------------------------
"$PY" - "$C" "$T" "$WORK" <<'PY'
import json, pathlib, sys
control = json.load(open(sys.argv[1]))
treatment = json.load(open(sys.argv[2]))
work = pathlib.Path(sys.argv[3])
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parents[1] / "src"))
import e2_canon as canon

def dump(name, payload):
    payload["record_sha256"] = canon.record_digest(payload)
    (work / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="ascii")

def fresh(base):
    return json.loads(json.dumps(base))

bad = fresh(treatment); bad["schema_version"] = 2
dump("version.json", bad)
bad = fresh(treatment); bad["surprise"] = 1
dump("unknown_field.json", bad)
bad = fresh(treatment); bad["energy_nj"] = 1
dump("forged_energy.json", bad)
bad = fresh(treatment); bad["independent_updates"] = 57
dump("few_updates.json", bad)
bad = fresh(treatment); bad["workload_digest"] = "0" * 64
dump("workload_mismatch.json", bad)
bad = fresh(treatment); bad["outcomes"] = {"met": 990, "tardy": 10, "rejected": 0,
                                           "canceled": 0}
dump("slo_mismatch.json", bad)
bad = fresh(treatment); bad["offered_work"] = 999
dump("work_mismatch.json", bad)
bad = fresh(treatment); bad["scope"] = "SERVER_WALL"; bad["board_uuids"] = []
dump("scope_mismatch.json", bad)
bad = fresh(treatment); bad["clock_epoch_id"] = "boot-other"
dump("clock_mismatch.json", bad)
bad = fresh(treatment); bad["role"] = "OPTIMIZED_SERVER_ONLY_CONTROL"
dump("role.json", bad)
bad = fresh(treatment); bad["raw_artifact_path"] = "../../../../etc/passwd"
dump("traversal.json", bad)
bad = fresh(treatment); bad["raw_artifact_path"] = "/etc/passwd"
dump("absolute.json", bad)
bad = fresh(treatment); bad["reason_code"] = "SYSTEM_ENERGY_SAVING"
dump("system_claim.json", bad)
bad = fresh(treatment); bad["uncertainty_nj"] = 0
bad["instrument_kind"] = "NVML_BOARD"; bad["provenance"] = "MEASURED"
dump("no_uncertainty.json", bad)
bad = fresh(treatment); bad["synchronization"] = "NONE"
dump("unsynced.json", bad)
# NVML relabelled as a server wall meter by editing the free-form label only
bad = fresh(treatment); bad["instrument_kind"] = "NVML_BOARD"
bad["scope"] = "SERVER_WALL"; bad["board_uuids"] = []
bad["instrument_label"] = "total server wall power"
dump("nvml_as_wall.json", bad)
bad = fresh(treatment); bad["instrument_kind"] = "RAPL_PACKAGE"
bad["scope"] = "SERVER_WALL"; bad["board_uuids"] = []
dump("rapl_as_wall.json", bad)
PY

for case in version unknown_field forged_energy few_updates workload_mismatch \
            slo_mismatch work_mismatch scope_mismatch clock_mismatch role \
            traversal absolute system_claim no_uncertainty unsynced \
            nvml_as_wall rapl_as_wall; do
    expect_fail "semantic $case" "${CMP[@]}" --control "$C" \
        --treatment "$WORK/$case.json" --trusted-root "$ROOT/fixtures"
done

# --- a SERVER_WALL pair with no capability record -----------------------------
expect_fail "server wall without capability" "${CMP[@]}" \
    --control "$WORK/scope_mismatch.json" --treatment "$WORK/scope_mismatch.json" \
    --trusted-root "$ROOT/fixtures"

# --- modified raw artifact bytes ---------------------------------------------
cp -r "$ROOT/fixtures" "$WORK/badroot"
sed -i 's/300001/100001/' "$WORK/badroot/raw/control.json"
expect_fail "modified artifact bytes" "${CMP[@]}" --control "$C" \
    --treatment "$T" --trusted-root "$WORK/badroot"

# --- the existing A6000 trace is a negative, not evidence ---------------------
TRACE="$ROOT/../s10_power_frontier/artifacts/cp2_a6000_power_trace.csv"
if [ -f "$TRACE" ]; then
    expect_fail "existing A6000 trace import" "$PY" \
        "$ROOT/src/import_nvml_trace.py" --csv "$TRACE"
fi

# --- fixture drift ------------------------------------------------------------
cp "$ROOT/fixtures/control_timeline.json" "$WORK/backup.json"
"$PY" - "$ROOT" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "fixtures" / "control_timeline.json"
doc = json.loads(path.read_text())
doc["timeline_id"] = "tampered"
path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
PY
expect_fail "fixture drift detected" "$PY" "$ROOT/tests/make_e2_fixtures.py" --check
cp "$WORK/backup.json" "$ROOT/fixtures/control_timeline.json"

if [ "$FAILURES" -ne 0 ]; then
    echo "S10_E2_NEGATIVE_FAIL: $FAILURES of $CASES cases did not fail closed" >&2
    exit 1
fi
echo "S10_E2_NEGATIVE_PASS ($CASES cases)"
