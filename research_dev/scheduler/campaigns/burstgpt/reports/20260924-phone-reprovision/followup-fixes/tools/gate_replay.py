"""Offline estimate: apply the boundary gate + RETAINED coalescing to a run's recorded EVALUATED stream.

A recorded RETAINED event stands for one boundary evaluation. State key (approximation of the gate's
signature from what the event records): arrived-work buckets, in-use sessions, selected generation and
state, followed/split models, commitment source, route-evidence reasons. Evaluate when the key changed
or >= 10 s since the last evaluation; record when the decision key (same minus nothing) changed."""
import json, sys, collections
run, interval = sys.argv[1], 10_000_000
d = json.load(open(run + "/RESULT.json"))
def bucket(v): return 0 if v == 0 else 1 << (v.bit_length() - 1)
evals = [e for e in d["phone_residency_events"] if e.get("kind") == "EVALUATED"]
kept = evaluated = 0; last_state = None; last_at = None; last_decision = None
other = 0
for e in evals:
    rp = e.get("desktop_reprovision")
    if e.get("reason") != "PHONE_RESIDENCY_REPROVISION_RETAINED" or rp is None:
        other += 1
        last_decision = None  # a different record breaks coalescing
        continue
    state = json.dumps([{k: bucket(v) for k, v in sorted(rp["arrived_work_by_artifact"].items())},
                        rp["in_use_session_ids"], e.get("selected_layout_generation"), e.get("selected_layout_state"),
                        rp["followed_artifact_sha256s"], rp["split_artifact_sha256s"], rp["desktop_commitment_source"],
                        sorted((k, v.get("reason")) for k, v in e.get("route_evidence_by_artifact", {}).items())])
    if state == last_state and e["observed_at_us"] - last_at < interval:
        continue
    evaluated += 1; last_state, last_at = state, e["observed_at_us"]
    if state != last_decision:
        kept += 1; last_decision = state
print(run, "RETAINED recorded", sum(1 for e in evals if e.get("reason") == "PHONE_RESIDENCY_REPROVISION_RETAINED"),
      "-> evaluated", evaluated, "-> recorded", kept, "; other EVALUATED", other)
