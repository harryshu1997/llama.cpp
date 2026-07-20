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
                  ("manifest", "manifest.json"),
                  ("route_control", "routes/route_control.json"),
                  ("route_treatment", "routes/route_treatment.json")):
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
                  ("manifest", "manifest.json"),
                  ("route_control", "routes/route_control.json"),
                  ("route_treatment", "routes/route_treatment.json")):
    index[f"{key}_sha256"] = canon.sha256_bytes((root / name).read_bytes())
(root / "bundle.json").write_text(
    json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
}

rebind_slot_evidence() {
    local dst="$1" slot="$2"
    python3 - "$ROOT" "$dst" "$slot" <<'PY'
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
import e2a_canon as canon

root = pathlib.Path(sys.argv[2])
slot = int(sys.argv[3])
timeline = canon.load_strict(root / f"timelines/tl.slot{slot:02d}.json")
lifecycle = canon.load_strict(root / f"lifecycle/lc.slot{slot:02d}.json")
outcomes = canon.load_strict(root / f"outcomes/outcomes.slot{slot:02d}.json")
ledger = canon.load_strict(root / "ledger.json")
end = next(entry for entry in ledger["entries"]
           if entry["slot_index"] == slot and
           entry["entry_kind"] == "SLOT_ATTEMPT_END")
end["timeline_record_sha256"] = timeline["record_sha256"]
end["lifecycle_record_sha256"] = lifecycle["record_sha256"]
end["request_outcome_record_sha256"] = outcomes["record_sha256"]
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
                  ("manifest", "manifest.json"),
                  ("route_control", "routes/route_control.json"),
                  ("route_treatment", "routes/route_treatment.json")):
    index[f"{key}_sha256"] = canon.sha256_bytes((root / name).read_bytes())
for item in index["slots"]:
    for key in ("timeline", "lifecycle", "outcomes"):
        item[f"{key}_sha256"] = canon.sha256_bytes(
            (root / item[f"{key}_path"]).read_bytes())
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

B="$(fresh r3_warmup)"
mutate "$B" plan.json 'record["warmup_count"]=1'
expect_code "r3_nonzero_warmup_is_unbound" "E_WARMUP_UNBOUND" "$B/bundle.json"

B="$(fresh r3_same_route)"
mutate "$B" plan.json 'record["route_schedule_digest_treatment"]=record["route_schedule_digest_control"]'
expect_code "r3_control_and_treatment_routes_must_differ" "E_ROUTE_BINDING" "$B/bundle.json"

B="$(fresh r3_wall_capability)"
mutate "$B" plan.json 'record["scope"]="SERVER_WALL"; record["instrument_kind"]="EXTERNAL_WALL_METER"; record["board_uuids"]=[]; record["included_rails"]=["SERVER/AC_INPUT"]; record["excluded_rails"]=[]; record["server_wall_capability_record_sha256"]="a"*64'
expect_code "r3_server_wall_requires_capability" "E_INCOMPLETE_WALL" "$B/bundle.json"

B="$(fresh r3_duplicate_slot)"
python3 - "$B" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
path = root / "bundle.json"
record = json.loads(path.read_text(encoding="ascii"))
record["slots"].append(dict(record["slots"][0]))
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="ascii")
PY
expect_code "r3_duplicate_slot_is_not_last_wins" "E_SLOT_DUPLICATE" "$B/bundle.json"

B="$(fresh r3_lifecycle_set)"
mutate "$B" lifecycle/lc.slot00.json 'record["set_id"]="set.unrelated"'
rebind_slot_evidence "$B" 0
expect_code "r3_lifecycle_set_is_bound" "E_LIFECYCLE_BINDING" "$B/bundle.json"

B="$(fresh r3_phone_exec)"
mutate "$B" lifecycle/lc.slot01.json 'a=next(x for x in record["actions"] if x["action_kind"]=="EXEC"); a["execution_domain"]="SERVER"; a["device_identity"]="GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"; a["backend_kind"]="CUDA"'
rebind_slot_evidence "$B" 1
expect_code "r3_treatment_requires_phone_exec" "E_ROUTE_NODE_MISMATCH" "$B/bundle.json"

B="$(fresh r3_anchor_identity)"
mutate "$B" plan_anchor.json 'record["experiment_identity"]="experiment.other"'
rebind_anchor_chain "$B"
expect_code "r3_anchor_binds_experiment_identity" "E_ANCHOR_IDENTITY" "$B/bundle.json"

B="$(fresh r4_route_artifact)"
mutate "$B" routes/route_treatment.json 'record["route_id"]="route.changed"'
expect_code "r4_route_digest_resolves_an_artifact" "E_ROUTE_ARTIFACT" "$B/bundle.json"

B="$(fresh r4_extra_server_exec)"
mutate "$B" lifecycle/lc.slot01.json 'a=dict(next(x for x in record["actions"] if x["action_kind"]=="EXEC")); a.update({"action_id":"a.server.real","execution_domain":"SERVER","device_identity":"GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f","backend_kind":"CUDA"}); record["actions"].append(a)'
rebind_slot_evidence "$B" 1
expect_code "r4_unplanned_server_exec_is_rejected" "E_ROUTE_NODE_EXTRA" "$B/bundle.json"

B="$(fresh r4_zero_phone_exec)"
mutate "$B" lifecycle/lc.slot01.json 'a=next(x for x in record["actions"] if x["action_kind"]=="EXEC"); a["end_us"]=a["start_us"]; a["ack_us"]=a["start_us"]'
rebind_slot_evidence "$B" 1
expect_code "r4_phone_exec_requires_duration" "E_ACTION_DURATION" "$B/bundle.json"

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
