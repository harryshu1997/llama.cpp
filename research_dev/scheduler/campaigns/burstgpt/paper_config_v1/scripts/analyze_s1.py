import json, sys, collections
RUN = sys.argv[1]
d = json.load(open(RUN + "/RESULT.json"))
dp = d.get("dispatch_policy") or {}
print("statistics:", dp.get("statistics"))
print("bypass_counts:", dp.get("bypass_counts"))
for r in (dp.get("refusals") or [])[:12]:
    print("  refusal:", r.get("kind"), r.get("request_id", "").split(":")[-1], "t=%.0f" % (r.get("observed_at_us", 0) / 1e6), "|", (r.get("reason") or "")[:200])
txt = open(RUN + "/SCHEDULER_DECISION_LOG.json").read()
res = open(RUN + "/RESULT.json").read()
keys = ("RESIDENCY_HYSTERESIS_HELD", "residency_hysteresis_released", "CONTINUOUS_JOIN_BARRIER_BYPASS", "CONTINUOUS_JOIN_DESKTOP_PARENT",
        "HELPER_ADOPTED_LATE", "SERVER_HELPER_LEASES_SHARED", "PHONE_HELPER_UNAVAILABLE", "DECIDED_TRANSITION_LAYOUT_REEVALUATION_FAILED")
print("markers (decision log / RESULT):", {k: (txt.count(k), res.count(k)) for k in keys})
for rr in d["request_results"]:
    fh = rr.get("fraction_history") or {}
    ex = fh.get("explored_split_fractions_ppm") if isinstance(fh, dict) else None
    print("  ", rr["request_id"].split(":")[-1], rr["model_id"][:5], "explored", ex)
pre = d.get("phone_residency_events") or []
print("residency events:", collections.Counter(e.get("kind") for e in pre).most_common(8))
src = collections.Counter()
for e in pre:
    s = json.dumps(e)
    if '"source": "queued"' in s: src["queued"] += 1
    if "REASON_REPROVISION" in s or '"reprovision"' in s: src["reprovision"] += 1
print("residency sources:", dict(src))
# --- step-1-fix markers ---
st = (d.get("dispatch_policy") or {}).get("statistics") or {}
print("hysteresis:", {k: v for k, v in st.items() if "hysteresis" in k})
for row in ((d.get("dispatch_policy") or {}).get("residency_hysteresis_decisions") or [])[:10]:
    print("  hyst decision:", {k: row.get(k) for k in ("request_id", "decision", "reason", "arrival_probability_ppm", "held_until_us") if k in row})
print("late adoption / retention:", {k: res.count(k) for k in ("HELPER_ADOPTED_LATE", "HELPER_MATERIALIZATION_RETAINED", "HELPER_EXPANDED", "history changed identity", "READY_LAYOUT_PUBLISHED")})
print("re-provision queued source rows:", res.count('"desktop_commitment_source": "queued"'))
# --- publication re-plan markers ---
print("publication:", {k: st.get(k) for k in ("continuous_join_publication_replans", "continuous_join_bypasses", "affinity_displacements", "publication_replans")})
print("wake continuous_join_server_live:", res.count("continuous_join_server_live"), "| MODEL_AFFINITY_DISPLACEMENT in log:", txt.count("MODEL_AFFINITY_DISPLACEMENT"))
