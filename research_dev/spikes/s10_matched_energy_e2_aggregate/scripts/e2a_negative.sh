#!/usr/bin/env bash
# CLI-level negatives for E2A. Each case mutates a VALID bundle by exactly one
# thing and asserts the gate refuses it with a specific E_* code.
#
# These run the real entry points as a reviewer would, not the Python API: a gate
# that only fails closed when driven from its own test harness is not a gate.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0

# Run the aggregate over a bundle and require a specific E_* code on stderr.
expect_code() {
    local name="$1" code="$2" bundle="$3"
    local out
    out="$(cd "$ROOT" && PYTHONPATH="$ROOT/src" python3 - "$bundle" <<'PY' 2>&1
import sys
sys.path.insert(0, "src")
import aggregate, resolver
try:
    aggregate.evaluate(sys.argv[1])
    print("NO_ERROR")
except (aggregate.AggregateError, resolver.ResolveError) as exc:
    print(exc.code)
except Exception as exc:
    print(f"UNEXPECTED:{type(exc).__name__}:{exc}")
PY
)"
    if [ "$out" = "$code" ]; then
        echo "  ok: $name -> $code"
        pass=$((pass + 1))
    else
        echo "  FAIL: $name expected $code got: $out" >&2
        fail=$((fail + 1))
    fi
}

# Mutate one JSON field in a copied bundle and reindex it.
mutate() {
    local dst="$1" rel="$2" expr="$3"
    python3 - "$ROOT" "$dst" "$rel" "$expr" <<'PY'
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
import e2a_canon as canon
root, rel, expr = pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
if rel != "-":
    record = canon.load_strict(root / rel)
    exec(expr, {"record": record, "canon": canon})
    if "record_sha256" in record and "NOSEAL" not in expr:
        canon.seal(record)
    (root / rel).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                            encoding="ascii")
index = canon.load_strict(root / "bundle.json")
for key, name in (("plan", "plan.json"), ("plan_anchor", "plan_anchor.json"),
                  ("ledger", "ledger.json"),
                  ("ledger_close", "ledger_close.json"),
                  ("manifest", "manifest.json")):
    index[f"{key}_sha256"] = canon.sha256_bytes((root / name).read_bytes())
for slot in index["slots"]:
    for key in ("timeline", "lifecycle", "outcomes"):
        slot[f"{key}_sha256"] = canon.sha256_bytes(
            (root / slot[f"{key}_path"]).read_bytes())
(root / "bundle.json").write_text(
    json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
}

fresh() {
    local dst="$WORK/$1"
    rm -rf "$dst"
    cp -r "$ROOT/fixtures" "$dst"
    echo "$dst"
}

# An anchor record is referenced by the first ledger entry and by the close
# receipt. Repair those dependent digests so an anchor mutation reaches the
# intended verifier or policy gate instead of stopping at E_LEDGER_BINDING.
rebind_anchor_chain() {
    local dst="$1"
    python3 - "$ROOT" "$dst" <<'PY'
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
import e2a_canon as canon

root = pathlib.Path(sys.argv[2])
anchor = canon.load_strict(root / "plan_anchor.json")
ledger = canon.load_strict(root / "ledger.json")
ledger["entries"][0]["plan_anchor_record_sha256"] = anchor["record_sha256"]
previous = "0" * 64
for entry in ledger["entries"]:
    entry["prev_entry_sha256"] = previous
    entry["entry_sha256"] = canon.digest(
        {key: value for key, value in entry.items()
         if key != "entry_sha256"})
    previous = entry["entry_sha256"]
ledger["head_sha256"] = previous
canon.seal(ledger)
(root / "ledger.json").write_text(
    json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="ascii")

close = canon.load_strict(root / "ledger_close.json")
close["anchor_kind"] = anchor["anchor_kind"]
close["plan_anchor_id"] = anchor["anchor_id"]
close["plan_anchor_record_sha256"] = anchor["record_sha256"]
close["attempt_ledger_sha256"] = ledger["record_sha256"]
close["ledger_head_sha256"] = ledger["head_sha256"]
close["anchored_digest"] = ledger["record_sha256"]
close["message_imprint_sha256"] = ledger["record_sha256"]
canon.seal(close)
(root / "ledger_close.json").write_text(
    json.dumps(close, indent=2, sort_keys=True) + "\n", encoding="ascii")

index = canon.load_strict(root / "bundle.json")
for key, name in (("plan", "plan.json"),
                  ("plan_anchor", "plan_anchor.json"),
                  ("ledger", "ledger.json"),
                  ("ledger_close", "ledger_close.json"),
                  ("manifest", "manifest.json")):
    index[f"{key}_sha256"] = canon.sha256_bytes((root / name).read_bytes())
(root / "bundle.json").write_text(
    json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
}

echo "S10-E2A CLI negatives"

# --- the headline: the untouched bundle stops at the anchor gate ---------------
B="$(fresh baseline)"
expect_code "valid_bundle_has_no_pinned_trust_root" "E_ANCHOR_TRUST_ROOT" "$B/bundle.json"

# --- anchors ------------------------------------------------------------------
B="$(fresh anchor_hmac)"
mutate "$B" plan_anchor.json 'record["anchor_kind"]="LOCAL_HMAC"'
rebind_anchor_chain "$B"
expect_code "anchor_local_hmac_has_no_verifier" "E_ANCHOR_NO_VERIFIER" "$B/bundle.json"

B="$(fresh anchor_selfsigned)"
mutate "$B" plan_anchor.json 'record["anchor_kind"]="SELF_SIGNED_TSA"'
rebind_anchor_chain "$B"
expect_code "anchor_self_signed_tsa_has_no_trust_root" "E_ANCHOR_TRUST_ROOT" "$B/bundle.json"

B="$(fresh anchor_git)"
mutate "$B" plan_anchor.json 'record["anchor_kind"]="GIT_COMMIT_LOCAL"'
rebind_anchor_chain "$B"
expect_code "anchor_git_commit_has_no_verifier" "E_ANCHOR_NO_VERIFIER" "$B/bundle.json"

B="$(fresh anchor_clock)"
mutate "$B" plan_anchor.json 'record["anchor_kind"]="LOCAL_CLOCK_ASSERTION"'
rebind_anchor_chain "$B"
expect_code "anchor_local_clock_has_no_verifier" "E_ANCHOR_NO_VERIFIER" "$B/bundle.json"

B="$(fresh anchor_tpm)"
mutate "$B" plan_anchor.json 'record["anchor_kind"]="TPM_NV_QUOTE"'
rebind_anchor_chain "$B"
expect_code "anchor_tpm_quote_has_no_verifier" "E_ANCHOR_NO_VERIFIER" "$B/bundle.json"

B="$(fresh anchor_synthetic)"
mutate "$B" plan_anchor.json 'record["anchor_kind"]="SYNTHETIC_ANCHOR"'
rebind_anchor_chain "$B"
expect_code "anchor_synthetic_has_no_verifier" "E_ANCHOR_NO_VERIFIER" "$B/bundle.json"

B="$(fresh anchor_unsealed)"
mutate "$B" plan_anchor.json 'record["anchor_id"]="rewritten" # NOSEAL'
expect_code "anchor_edited_without_resealing" "E_DIGEST" "$B/bundle.json"

B="$(fresh anchor_missing)"
rm -f "$B/plan_anchor.json"
expect_code "anchor_file_absent" "E_MISSING" "$B/bundle.json"

B="$(fresh anchor_token_missing)"
mutate "$B" plan_anchor.json 'record["anchor_kind"]="TRANSPARENCY_LOG_INCLUSION"'
rebind_anchor_chain "$B"
rm -f "$B/anchors/plan_token.der"
expect_code "anchor_token_absent" "E_MISSING" "$B/bundle.json"

# --- plan ---------------------------------------------------------------------
B="$(fresh plan_gates)"
mutate "$B" plan.json 'record["gate_constants_digest"]=canon.digest({"MIN_PAIRS":2})'
expect_code "plan_retunes_the_gates" "E_GATE_RETUNED" "$B/bundle.json"

B="$(fresh plan_order)"
mutate "$B" plan.json 'record["declared_order"][1]="OPTIMIZED_SERVER_ONLY_CONTROL"; record["order_digest"]=canon.digest(record["declared_order"])'
expect_code "plan_breaks_abba_rotation" "E_ORDER" "$B/bundle.json"

B="$(fresh plan_orderdigest)"
mutate "$B" plan.json 'record["order_digest"]="0"*64'
expect_code "plan_order_digest_uncovered" "E_ORDER" "$B/bundle.json"

B="$(fresh plan_policy)"
mutate "$B" plan.json 'record["policy_digest_treatment"]=record["policy_digest_control"]'
expect_code "plan_same_policy_both_arms" "E_SAME_POLICY" "$B/bundle.json"

B="$(fresh plan_retry)"
mutate "$B" plan.json 'record["max_attempts_per_slot"]=3'
expect_code "plan_permits_retries" "E_SCHEMA" "$B/bundle.json"

B="$(fresh plan_nonce)"
mutate "$B" plan.json 'record["slots"][1]["run_nonce"]=record["slots"][0]["run_nonce"]'
expect_code "plan_reuses_a_run_nonce" "E_PLAN" "$B/bundle.json"

B="$(fresh plan_unsealed)"
mutate "$B" plan.json 'record["experiment_identity"]="rewritten" # NOSEAL'
expect_code "plan_edited_without_resealing" "E_DIGEST" "$B/bundle.json"

B="$(fresh plan_scope)"
mutate "$B" plan.json 'record["instrument_kind"]="NVML_BOARD"; record["scope"]="SERVER_WALL"; record["board_uuids"]=[]'
expect_code "plan_promotes_nvml_to_server_wall" "E_SCOPE" "$B/bundle.json"

B="$(fresh plan_rapl)"
mutate "$B" plan.json 'record["instrument_kind"]="RAPL_PACKAGE"'
expect_code "plan_claims_scope_for_rapl" "E_SCOPE" "$B/bundle.json"

# --- ledger -------------------------------------------------------------------
B="$(fresh ledger_chain)"
mutate "$B" ledger.json 'record["entries"][5]["prev_entry_sha256"]="0"*64'
expect_code "ledger_broken_chain_link" "E_LEDGER_CHAIN" "$B/bundle.json"

B="$(fresh ledger_body)"
mutate "$B" ledger.json 'record["entries"][5]["reason_code"]="REWRITTEN"'
expect_code "ledger_edited_entry_body" "E_LEDGER_CHAIN" "$B/bundle.json"

B="$(fresh ledger_head)"
mutate "$B" ledger.json 'record["head_sha256"]="0"*64'
expect_code "ledger_head_is_not_last_entry" "E_LEDGER_HEAD" "$B/bundle.json"

B="$(fresh ledger_plan)"
mutate "$B" ledger.json 'record["plan_sha256"]="0"*64'
expect_code "ledger_bound_to_another_plan" "E_LEDGER_BINDING" "$B/bundle.json"

B="$(fresh ledger_retry)"
mutate "$B" ledger.json 'record["entries"][3]["attempt_ordinal"]=1
prev="0"*64
for e in record["entries"]:
    e["prev_entry_sha256"]=prev
    e["entry_sha256"]=canon.digest({k:v for k,v in e.items() if k!="entry_sha256"})
    prev=e["entry_sha256"]
record["head_sha256"]=prev'
expect_code "ledger_undeclared_retry" "E_UNDECLARED_RETRY" "$B/bundle.json"

for status in FAILED CANCELED CRASHED INELIGIBLE ABORTED_BY_GATE; do
    B="$(fresh "ledger_$status")"
    mutate "$B" ledger.json "record[\"entries\"][3][\"status\"]=\"$status\"
prev=\"0\"*64
for e in record[\"entries\"]:
    e[\"prev_entry_sha256\"]=prev
    e[\"entry_sha256\"]=canon.digest({k:v for k,v in e.items() if k!=\"entry_sha256\"})
    prev=e[\"entry_sha256\"]
record[\"head_sha256\"]=prev"
    expect_code "ledger_attempt_$status" "E_NON_OK_ATTEMPT" "$B/bundle.json"
done

B="$(fresh ledger_truncated)"
mutate "$B" ledger.json 'record["entries"]=[e for e in record["entries"] if not (e["entry_kind"]=="SLOT_ATTEMPT_END" and e["slot_index"]==15)]
prev="0"*64
for i,e in enumerate(record["entries"]):
    e["seq"]=i
    e["prev_entry_sha256"]=prev
    e["entry_sha256"]=canon.digest({k:v for k,v in e.items() if k!="entry_sha256"})
    prev=e["entry_sha256"]
record["head_sha256"]=prev'
expect_code "ledger_truncated_tail_rechained" "E_SLOT_UNCOVERED" "$B/bundle.json"

# --- bundle -------------------------------------------------------------------
B="$(fresh bundle_slot_missing)"
mutate "$B" - 'pass'
python3 - "$B" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
index = json.loads((root / "bundle.json").read_text())
index["slots"] = index["slots"][:-1]
(root / "bundle.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
PY
expect_code "bundle_omits_a_planned_slot" "E_SLOT_UNCOVERED" "$B/bundle.json"

B="$(fresh timeline_tampered)"
python3 - "$B" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
path = root / "timelines" / "tl.slot00.json"
record = json.loads(path.read_text())
record["energy_nj"] = 1
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
expect_code "timeline_forged_energy_not_reindexed" "E_HASH" "$B/bundle.json"

echo
if [ "$fail" -ne 0 ]; then
    echo "S10_E2A_NEGATIVE_FAIL ($fail of $((pass + fail)) cases)" >&2
    exit 1
fi
echo "S10_E2A_NEGATIVE_PASS ($pass cases)"
