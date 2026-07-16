#!/usr/bin/env bash
# Every invalid CLI case must exit nonzero with a stable diagnostic prefix and no
# Python traceback. A traceback is a failure.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
TMP=$(mktemp -d /tmp/s10_v0r_cli.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

FIX="$ROOT/fixtures/transition_delay_counterexample.json"
GOOD="$TMP/good_cert.json"
python "$ROOT/oracle/exact.py" --instance "$FIX" --out "$GOOD" >/dev/null

fails=0

expect_fail() {
    local name="$1"; shift
    local want="$1"; shift
    local out="$TMP/$name.err"
    if "$@" >/dev/null 2>"$out"; then
        echo "CLI_NEGATIVE_FAIL: $name unexpectedly exited 0" >&2
        fails=$((fails + 1))
        return
    fi
    if grep -q 'Traceback (most recent call last)' "$out"; then
        echo "CLI_NEGATIVE_FAIL: $name leaked a traceback" >&2
        sed -n '1,4p' "$out" >&2
        fails=$((fails + 1))
        return
    fi
    if ! grep -q "$want" "$out"; then
        echo "CLI_NEGATIVE_FAIL: $name missing stable diagnostic '$want'" >&2
        sed -n '1,3p' "$out" >&2
        fails=$((fails + 1))
        return
    fi
    echo "  ok: $name"
}

# missing files
expect_fail missing_instance ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/nope.json"
expect_fail missing_cert CHECK_FAIL \
    python "$ROOT/checker/checker.py" --instance "$FIX" --certificate "$TMP/nope.json"

# malformed / hostile JSON
printf '{"schema_version":2,"schema_version":2}' > "$TMP/dupkey.json"
expect_fail duplicate_keys ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/dupkey.json"
expect_fail duplicate_keys_checker CHECK_FAIL \
    python "$ROOT/checker/checker.py" --instance "$TMP/dupkey.json" --certificate "$GOOD"

printf '{"horizon_us": NaN}' > "$TMP/nan.json"
expect_fail nan_constant ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/nan.json"
printf '{"horizon_us": Infinity}' > "$TMP/inf.json"
expect_fail infinity_constant ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/inf.json"
printf '{"broken"' > "$TMP/trunc.json"
expect_fail truncated_json ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/trunc.json"

# semantic holes reached through the CLI
python - "$FIX" "$TMP/bool_field.json" <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))
inst["horizon_us"] = True          # a bool must never pass as an integer
json.dump(inst, open(sys.argv[2], "w"))
PY
expect_fail bool_as_integer ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/bool_field.json"

python - "$FIX" "$TMP/scope.json" <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))
inst["evidence"]["scope"] = "MEASURED"      # wrong evidence scope
json.dump(inst, open(sys.argv[2], "w"))
PY
expect_fail wrong_evidence_scope ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/scope.json"

python - "$FIX" "$TMP/unknown.json" <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))
inst["surprise"] = 1                        # unknown instance field
json.dump(inst, open(sys.argv[2], "w"))
PY
expect_fail unknown_instance_field ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/unknown.json"

python - "$FIX" "$TMP/second_server.json" <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))
inst["devices"]["SERVER2"] = {"kind": "server", "active_mw": 1}
json.dump(inst, open(sys.argv[2], "w"))
PY
expect_fail alternate_server_kind ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/second_server.json"

python - "$FIX" "$TMP/badprofile.json" <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))
inst["batch_profiles"]["unused"] = {"x": False}   # malformed unused profile
json.dump(inst, open(sys.argv[2], "w"))
PY
expect_fail malformed_batch_profile ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$TMP/badprofile.json"

python - "$FIX" "$TMP/noncanonical_profile.json" <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))
inst["batch_profiles"]["unused"] = {"016": 1}
json.dump(inst, open(sys.argv[2], "w"))
PY
expect_fail noncanonical_batch_profile 'instance schema violation' \
    python "$ROOT/oracle/exact.py" --instance "$TMP/noncanonical_profile.json"
expect_fail noncanonical_batch_profile_checker 'instance schema violation' \
    python "$ROOT/checker/checker.py" --instance "$TMP/noncanonical_profile.json" \
        --certificate "$GOOD" --feasibility-only

python - "$FIX" "$TMP/oversized_horizon.json" <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))
inst["horizon_us"] = 1000000000001
json.dump(inst, open(sys.argv[2], "w"))
PY
expect_fail oversized_horizon 'instance schema violation' \
    python "$ROOT/oracle/exact.py" --instance "$TMP/oversized_horizon.json"
expect_fail oversized_horizon_checker 'instance schema violation' \
    python "$ROOT/checker/checker.py" --instance "$TMP/oversized_horizon.json" \
        --certificate "$GOOD" --feasibility-only

# state-cap exhaustion must fail closed, never emit a best-so-far certificate
expect_fail state_cap_exhaustion ORACLE_FAIL \
    python "$ROOT/oracle/exact.py" --instance "$FIX" --max-states 10

# tampered certificate body without a reseal
python - "$GOOD" "$TMP/tampered.json" <<'PY'
import json, sys
cert = json.load(open(sys.argv[1]))
cert["energy"]["total_nj"] += 1
json.dump(cert, open(sys.argv[2], "w"))
PY
expect_fail tampered_certificate CHECK_FAIL \
    python "$ROOT/checker/checker.py" --instance "$FIX" --certificate "$TMP/tampered.json"

if [ "$fails" -ne 0 ]; then
    echo "CLI negative suite failed: $fails case(s)" >&2
    exit 1
fi
echo "S10_V0R_CLI_NEGATIVE_PASS"
