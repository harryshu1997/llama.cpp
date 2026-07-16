#!/usr/bin/env bash
# Invalid evidence CLI cases. Each must exit nonzero with a stable diagnostic
# prefix and NO Python traceback. A traceback means an unhandled path, which is
# how fail-open bugs hide.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
WORK=$(mktemp -d /tmp/s10_v0r_evneg.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

BUNDLE="$ROOT/fixtures/evidence/mechanics_bundle.json"
INST="$ROOT/fixtures/evidence/transition_v3.json"
FAILURES=0
CASES=0

expect_fail() {
    local name="$1"; shift
    CASES=$((CASES + 1))
    local out; local rc
    if out=$("$@" 2>&1); then
        rc=0
    else
        rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
        echo "NEGATIVE CASE PASSED WHEN IT MUST FAIL: $name" >&2
        FAILURES=$((FAILURES + 1)); return
    fi
    if ! grep -qE '^(EVIDENCE_FAIL|BIND_FAIL|usage:|.*error:)' <<<"$out"; then
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

require_files() {
    local path
    for path in "$@"; do
        if [ ! -s "$path" ]; then
            echo "EVIDENCE_NEGATIVE_SETUP_FAIL: missing generated fixture $path" >&2
            exit 1
        fi
    done
}

V=(python "$ROOT/evidence/validator.py")
B=(python "$ROOT/evidence/binder.py")

# --- missing and unreadable inputs -------------------------------------------
expect_fail "missing bundle" "${V[@]}" --bundle "$WORK/nope.json"
expect_fail "missing instance" "${V[@]}" --bundle "$BUNDLE" --instance "$WORK/nope.json"
expect_fail "bundle is a directory" "${V[@]}" --bundle "$WORK"
expect_fail "no bundle argument" "${V[@]}" --instance "$INST"

# --- malformed JSON -----------------------------------------------------------
printf '{"a": 1' > "$WORK/truncated.json"
expect_fail "truncated JSON" "${V[@]}" --bundle "$WORK/truncated.json"
printf '{"a": 1, "a": 2}' > "$WORK/dupkeys.json"
expect_fail "duplicate keys" "${V[@]}" --bundle "$WORK/dupkeys.json"
printf '{"schema_version": NaN}' > "$WORK/nan.json"
expect_fail "NaN constant" "${V[@]}" --bundle "$WORK/nan.json"
printf '{"schema_version": Infinity}' > "$WORK/inf.json"
expect_fail "Infinity constant" "${V[@]}" --bundle "$WORK/inf.json"
printf '{"schema_version": 3, "bundle_id": "\xc3\xa9"}' > "$WORK/nonascii.json"
expect_fail "non-ASCII bundle" "${V[@]}" --bundle "$WORK/nonascii.json"
printf '[]' > "$WORK/array.json"
expect_fail "top-level array" "${V[@]}" --bundle "$WORK/array.json"

# --- structural bundle failures ----------------------------------------------
python - "$BUNDLE" "$WORK" <<'PY'
import json, pathlib, sys
bundle = json.load(open(sys.argv[1]))
work = pathlib.Path(sys.argv[2])

def dump(name, payload):
    (work / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="ascii")

bad = json.loads(json.dumps(bundle)); bad["schema_version"] = 4
dump("version.json", bad)
bad = json.loads(json.dumps(bundle)); bad["unknown_field"] = "x"
dump("unknown.json", bad)
bad = json.loads(json.dumps(bundle)); del bad["power"]
dump("missing_section.json", bad)
bad = json.loads(json.dumps(bundle)); bad["bundle_sha256"] = "0" * 64
dump("badhash.json", bad)
bad = json.loads(json.dumps(bundle)); bad["provenance"] = "TOTALLY_REAL"
dump("badprov.json", bad)
bad = json.loads(json.dumps(bundle))
bad["routes"][0]["process_count"] = 1
dump("fewproc.json", bad)
bad = json.loads(json.dumps(bundle))
bad["routes"][0]["artifacts"] = []
dump("noartifacts.json", bad)
PY

require_files \
    "$WORK/version.json" "$WORK/unknown.json" "$WORK/missing_section.json" \
    "$WORK/badhash.json" "$WORK/badprov.json" "$WORK/fewproc.json" \
    "$WORK/noartifacts.json"

for case in version unknown missing_section badhash badprov fewproc noartifacts; do
    expect_fail "bundle $case" "${V[@]}" --bundle "$WORK/$case.json"
done

# --- binding failures ---------------------------------------------------------
python - "$INST" "$WORK" <<'PY'
import json, pathlib, sys
inst = json.load(open(sys.argv[1]))
work = pathlib.Path(sys.argv[2])

def dump(name, payload):
    (work / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="ascii")

bad = json.loads(json.dumps(inst)); bad["evidence"]["bindings"] = []
dump("nobindings.json", bad)
bad = json.loads(json.dumps(inst))
bad["evidence"]["bindings"] = [b for b in bad["evidence"]["bindings"]
                               if b["target"] != "server_power.p0_mw"]
dump("dropbinding.json", bad)
bad = json.loads(json.dumps(inst))
bad["nodes"][0]["routes"]["SERVER"]["duration_us"] = 101
dump("offbyone.json", bad)
bad = json.loads(json.dumps(inst)); bad["evidence"]["bundle_sha256"] = "f" * 64
dump("wrongbundle.json", bad)
bad = json.loads(json.dumps(inst)); bad["evidence"]["scope"] = "MEASURED"
dump("fakemeasured.json", bad)
bad = json.loads(json.dumps(inst)); bad["schema_version"] = 2
dump("v2asv3.json", bad)
bad = json.loads(json.dumps(inst))
bad["evidence"]["bindings"][0]["record_sha256"] = "a" * 64
dump("staledigest.json", bad)
PY

require_files \
    "$WORK/nobindings.json" "$WORK/dropbinding.json" "$WORK/offbyone.json" \
    "$WORK/wrongbundle.json" "$WORK/fakemeasured.json" "$WORK/v2asv3.json" \
    "$WORK/staledigest.json"

for case in nobindings dropbinding offbyone wrongbundle fakemeasured v2asv3 staledigest; do
    expect_fail "binding $case" "${V[@]}" --bundle "$BUNDLE" --instance "$WORK/$case.json"
    expect_fail "solve $case" "${B[@]}" --instance "$WORK/$case.json" --bundle "$BUNDLE" --solve
done

# --- certificate failures -----------------------------------------------------
python "$ROOT/evidence/binder.py" --instance "$INST" --bundle "$BUNDLE" --solve \
    --out "$WORK/cert.json" --quiet
require_files "$WORK/cert.json"
python - "$WORK" <<'PY'
import hashlib, json, pathlib, sys
work = pathlib.Path(sys.argv[1])
cert = json.load(open(work / "cert.json"))

def dump(name, payload):
    (work / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="ascii")

bad = json.loads(json.dumps(cert)); bad["objective"][3] = 1
dump("cert_tampered.json", bad)
bad = json.loads(json.dumps(cert)); bad["instance_sha256"] = "0" * 64
dump("cert_wrong_instance.json", bad)
bad = json.loads(json.dumps(cert)); bad["evidence"]["bundle_sha256"] = "0" * 64
dump("cert_wrong_bundle.json", bad)
bad = json.loads(json.dumps(cert))
bad["evidence"]["energy_claim"] = "SYSTEM_ENERGY_SAVING"
dump("cert_overclaim.json", bad)
bad = json.loads(json.dumps(cert)); bad["schema_version"] = 2
dump("cert_v2.json", bad)

def reseal(payload):
    body = {key: value for key, value in payload.items()
            if key != "certificate_sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    payload["certificate_sha256"] = hashlib.sha256(raw).hexdigest()

bad = json.loads(json.dumps(cert)); bad["search"]["complete"] = False; reseal(bad)
dump("cert_incomplete.json", bad)
bad = json.loads(json.dumps(cert)); bad["forged"] = 1; reseal(bad)
dump("cert_unknown.json", bad)
PY

require_files \
    "$WORK/cert_tampered.json" "$WORK/cert_wrong_instance.json" \
    "$WORK/cert_wrong_bundle.json" "$WORK/cert_overclaim.json" \
    "$WORK/cert_v2.json" "$WORK/cert_incomplete.json" "$WORK/cert_unknown.json"

for case in cert_tampered cert_wrong_instance cert_wrong_bundle cert_overclaim cert_v2 \
            cert_incomplete cert_unknown; do
    expect_fail "certificate $case" "${B[@]}" --instance "$INST" --bundle "$BUNDLE" \
        --certificate "$WORK/$case.json"
done

# --- state-cap exhaustion must fail closed, never emit a best-so-far ----------
expect_fail "state cap exhaustion" "${B[@]}" --instance "$INST" --bundle "$BUNDLE" \
    --solve --max-states 10

if [ "$FAILURES" -ne 0 ]; then
    echo "S10_V0R_EVIDENCE_NEGATIVE_FAIL: $FAILURES of $CASES cases did not fail closed" >&2
    exit 1
fi
echo "S10_V0R_EVIDENCE_NEGATIVE_PASS ($CASES cases)"
