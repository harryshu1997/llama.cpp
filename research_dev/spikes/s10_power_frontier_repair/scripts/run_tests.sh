#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
WORK=$(mktemp -d /tmp/s10_v0r_tests.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
CERT="$WORK/partial_certificate.json"
TCERT="$WORK/transition_certificate.json"
ACERT="$WORK/activation_certificate.json"
DIFF_SUMMARY="$WORK/differential.jsonl"

python -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v

# ---- schema validation of every frozen instance -----------------------------
for fixture in partial_partition transition_delay_counterexample \
               activation_delay_counterexample; do
    /usr/bin/jsonschema -i "$ROOT/fixtures/$fixture.json" \
        "$ROOT/schemas/instance.schema.json"
done

# ---- EARLIEST-mode instance: accounting passes, optimality is NOT certified --
# partial_partition sits outside the independent reference's declared tiny domain,
# so default mode must fail closed rather than trust the solver's own counters.
python "$ROOT/oracle/exact.py" \
    --instance "$ROOT/fixtures/partial_partition.json" \
    --out "$CERT"
/usr/bin/jsonschema -i "$CERT" "$ROOT/schemas/certificate.schema.json"
python "$ROOT/checker/checker.py" \
    --instance "$ROOT/fixtures/partial_partition.json" \
    --certificate "$CERT" \
    --feasibility-only
PPERR="$WORK/partial.err"
if python "$ROOT/checker/checker.py" \
    --instance "$ROOT/fixtures/partial_partition.json" \
    --certificate "$CERT" \
    --quiet 2>"$PPERR"; then
    echo "exact checker certified an optimum outside the independent reference domain" >&2
    rm -f "$PPERR"; exit 1
fi
# Assert the SPECIFIC fail-closed reason. A bare "it exited nonzero" guard would
# stay green even if the whole optimality-comparison block were deleted.
grep -q 'not independently verifiable inside the reference domain' "$PPERR" || {
    echo "partial_partition did not fail closed for the expected domain reason" >&2
    sed -n '1,3p' "$PPERR" >&2; rm -f "$PPERR"; exit 1; }
rm -f "$PPERR"

# ---- TEMPORAL-mode frozen fixtures: default mode proves the optimum ----------
# transition: the solver must pick the delayed 147250000 nJ placement, and the
# independent reference must confirm it.
python "$ROOT/oracle/exact.py" \
    --instance "$ROOT/fixtures/transition_delay_counterexample.json" \
    --out "$TCERT"
/usr/bin/jsonschema -i "$TCERT" "$ROOT/schemas/certificate.schema.json"
python - "$TCERT" <<'PY'
import json, sys
cert = json.load(open(sys.argv[1]))
assert cert["energy"]["total_nj"] == 147250000, cert["energy"]["total_nj"]
assert cert["search"]["complete"] is True
PY
python "$ROOT/checker/checker.py" \
    --instance "$ROOT/fixtures/transition_delay_counterexample.json" \
    --certificate "$TCERT"

# activation: earliest placement is infeasible; the delayed one is proven optimal.
python "$ROOT/oracle/exact.py" \
    --instance "$ROOT/fixtures/activation_delay_counterexample.json" \
    --out "$ACERT"
/usr/bin/jsonschema -i "$ACERT" "$ROOT/schemas/certificate.schema.json"
python "$ROOT/checker/checker.py" \
    --instance "$ROOT/fixtures/activation_delay_counterexample.json" \
    --certificate "$ACERT"

# ---- a feasible but SUBOPTIMAL resealed certificate must be rejected ---------
SUB="$WORK/suboptimal.json"
SUBERR="$WORK/suboptimal.err"
python "$ROOT/tests/make_suboptimal.py" --out "$SUB"
if python "$ROOT/checker/checker.py" \
        --instance "$ROOT/fixtures/transition_delay_counterexample.json" \
        --certificate "$SUB" --quiet 2>"$SUBERR"; then
    echo "default checker accepted a feasible suboptimal certificate" >&2
    rm -f "$SUB" "$SUBERR"; exit 1
fi
grep -q 'SUBOPTIMAL' "$SUBERR" || {
    echo "suboptimal rejection lacked a stable SUBOPTIMAL diagnostic" >&2
    rm -f "$SUB" "$SUBERR"; exit 1; }
grep -q 'Traceback (most recent call last)' "$SUBERR" && {
    echo "checker leaked a traceback while rejecting a suboptimal certificate" >&2
    rm -f "$SUB" "$SUBERR"; exit 1; }
# the same schedule still passes accounting-only mode, which never claims optimality
python "$ROOT/checker/checker.py" \
    --instance "$ROOT/fixtures/transition_delay_counterexample.json" \
    --certificate "$SUB" --feasibility-only
rm -f "$SUB" "$SUBERR"

# ---- invalid CLI cases exit nonzero with a stable diagnostic, no traceback ---
bash "$ROOT/scripts/cli_negative.sh"

# ============================================================================
# S10-V0-R-E1 typed evidence gate
# ============================================================================

# ---- the fixtures are a deterministic function of the generator --------------
python "$ROOT/tests/make_evidence_fixtures.py" --check

# ---- v3 schema validation of the bundle, instances, and certificates ---------
# ajv draft-2020 is NOT installed on this host, so only /usr/bin/jsonschema runs.
# That is recorded as a deviation in RESULTS.md rather than papered over.
/usr/bin/jsonschema -i "$ROOT/fixtures/evidence/mechanics_bundle.json" \
    "$ROOT/schemas/evidence_bundle.v3.schema.json"
for fixture in transition_v3 activation_v3; do
    /usr/bin/jsonschema -i "$ROOT/fixtures/evidence/$fixture.json" \
        "$ROOT/schemas/instance.v3.schema.json"
done

# ---- the synthetic MECHANICS_ONLY bundle validates ---------------------------
python "$ROOT/evidence/validator.py" --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" \
    --quiet
for fixture in transition_v3 activation_v3; do
    python "$ROOT/evidence/validator.py" \
        --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" \
        --instance "$ROOT/fixtures/evidence/$fixture.json" --quiet
done

# ---- the evidence-bound path reproduces BOTH frozen temporal optima ----------
EV_T="$WORK/evidence_transition.json"
EV_A="$WORK/evidence_activation.json"

python "$ROOT/evidence/binder.py" \
    --instance "$ROOT/fixtures/evidence/transition_v3.json" \
    --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" \
    --solve --out "$EV_T" --quiet
/usr/bin/jsonschema -i "$EV_T" "$ROOT/schemas/certificate.v3.schema.json"
python - "$EV_T" <<'PY'
import json, sys
cert = json.load(open(sys.argv[1]))
assert cert["energy"]["total_nj"] == 147250000, cert["energy"]["total_nj"]
assert cert["energy"]["server_p0_intervals"] == [[800, 1150]], cert["energy"]
assert cert["evidence"]["energy_claim"] == "NONE_MECHANICS_ONLY", cert["evidence"]
assert cert["search"]["complete"] is True
PY
python "$ROOT/evidence/binder.py" \
    --instance "$ROOT/fixtures/evidence/transition_v3.json" \
    --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" \
    --certificate "$EV_T" --quiet

python "$ROOT/evidence/binder.py" \
    --instance "$ROOT/fixtures/evidence/activation_v3.json" \
    --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" \
    --solve --out "$EV_A" --quiet
/usr/bin/jsonschema -i "$EV_A" "$ROOT/schemas/certificate.v3.schema.json"
python - "$EV_A" <<'PY'
import json, sys
cert = json.load(open(sys.argv[1]))
assert cert["energy"]["total_nj"] == 2974000, cert["energy"]["total_nj"]
assert cert["activation_peak_bytes"] == 100, cert["activation_peak_bytes"]
PY
python "$ROOT/evidence/binder.py" \
    --instance "$ROOT/fixtures/evidence/activation_v3.json" \
    --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" \
    --certificate "$EV_A" --quiet

# ---- dropping a single binding must fail closed with the SPECIFIC reason -----
# A bare "it exited nonzero" guard would stay green if the whole binding check
# were deleted, so assert the diagnostic itself.
DROPERR="$WORK/drop.err"
DROPINST="$WORK/drop.json"
python - "$ROOT/fixtures/evidence/transition_v3.json" "$DROPINST" <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))
inst["evidence"]["bindings"] = [b for b in inst["evidence"]["bindings"]
                                if b["target"] != "server_power.transition_nj"]
json.dump(inst, open(sys.argv[2], "w"), indent=2, sort_keys=True)
PY
if python "$ROOT/evidence/binder.py" --instance "$DROPINST" \
        --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" \
        --solve --quiet 2>"$DROPERR"; then
    echo "binder solved an instance with an unbound evidence-derived field" >&2
    rm -f "$DROPERR" "$DROPINST"; exit 1
fi
grep -q 'E_BINDING_MISSING' "$DROPERR" || {
    echo "dropping a binding did not raise E_BINDING_MISSING" >&2
    sed -n '1,3p' "$DROPERR" >&2; rm -f "$DROPERR" "$DROPINST"; exit 1; }
rm -f "$DROPERR" "$DROPINST"

# ---- evidence output is deterministic across processes and hash seeds --------
# The bundle digest, binding digest, and certificate digest must not depend on
# dict iteration order. Each run below is a separate process.
EVREF=""
for seed in 0 1 42 12345 random; do
    LINE=$(PYTHONHASHSEED=$seed python "$ROOT/evidence/binder.py" \
        --instance "$ROOT/fixtures/evidence/transition_v3.json" \
        --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" --solve)
    DIG=$(echo "$LINE" | python -c 'import json,sys; d=json.load(sys.stdin); print(d["certificate_sha256"])')
    BIND=$(PYTHONHASHSEED=$seed python "$ROOT/evidence/validator.py" \
        --bundle "$ROOT/fixtures/evidence/mechanics_bundle.json" \
        --instance "$ROOT/fixtures/evidence/transition_v3.json" \
        | python -c 'import json,sys; d=json.load(sys.stdin); print(d["bundle_sha256"], d["binding_sha256"])')
    if [ -z "$EVREF" ]; then
        EVREF="$DIG $BIND"
    elif [ "$EVREF" != "$DIG $BIND" ]; then
        echo "evidence digests are not deterministic at PYTHONHASHSEED=$seed" >&2
        echo "  expected: $EVREF" >&2
        echo "  got     : $DIG $BIND" >&2
        exit 1
    fi
done
echo "evidence determinism: $EVREF"

# ---- invalid evidence CLI cases ---------------------------------------------
bash "$ROOT/scripts/evidence_negative.sh"

# ---- >=1000 generated differential cases across processes and hash seeds -----
DIGESTS=()
SEEDS=(0 1 42 12345)
STARTS=(0 300 600 900)
for index in 0 1 2 3; do
    OUT=$(PYTHONHASHSEED=${SEEDS[$index]} python "$ROOT/tests/differential.py" \
              --start "${STARTS[$index]}" --count 300)
    echo "differential slice ${STARTS[$index]}: $OUT"
    echo "$OUT" >> "$DIFF_SUMMARY"
    DIGESTS+=("$(echo "$OUT" | python -c 'import json,sys; print(json.load(sys.stdin)["digest"])')")
done

python - "$DIFF_SUMMARY" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="ascii") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
assert len(rows) == 4, len(rows)
assert sum(row["count"] for row in rows) == 1200
assert sum(row["compared"] for row in rows) >= 1000
assert sum(row["skipped_out_of_domain"] for row in rows) == 0
for row in rows:
    assert (row["compared"] + row["agreed_infeasible"] +
            row["skipped_out_of_domain"] == row["count"]), row
PY

# determinism: the SAME slice under a different PYTHONHASHSEED and process must
# reproduce byte-identical objectives and certificate hashes.
REPEAT=$(PYTHONHASHSEED=99 python "$ROOT/tests/differential.py" --start 0 --count 300 \
         | python -c 'import json,sys; print(json.load(sys.stdin)["digest"])')
if [ "$REPEAT" != "${DIGESTS[0]}" ]; then
    echo "differential digest is not deterministic across PYTHONHASHSEED values" >&2
    exit 1
fi
REPEAT2=$(PYTHONHASHSEED=random python "$ROOT/tests/differential.py" --start 0 --count 300 \
          | python -c 'import json,sys; print(json.load(sys.stdin)["digest"])')
if [ "$REPEAT2" != "${DIGESTS[0]}" ]; then
    echo "differential digest is not deterministic under PYTHONHASHSEED=random" >&2
    exit 1
fi

find "$ROOT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -name '*.pyc' -delete 2>/dev/null || true

# The foundation marker is kept so a reader can see the temporal gate is still
# green, and the evidence marker is added rather than replacing it.
echo S10_V0R_TEMPORAL_FOUNDATION_TESTS_PASS
echo S10_V0R_TYPED_EVIDENCE_TESTS_PASS
