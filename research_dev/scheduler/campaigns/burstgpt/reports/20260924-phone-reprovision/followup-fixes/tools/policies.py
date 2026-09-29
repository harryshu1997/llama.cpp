import json, sys, collections, hashlib
run = sys.argv[1]
obs = json.load(open(run + "/ADAPTIVE_DECODE_OBSERVATIONS.json"))
groups = obs["groups"]
for g in groups:
    rid = g.get("request_id", "")
    pols = {}
    for w in g.get("windows", []):
        p = w["policy"]
        ident = (p.get("columns"), p.get("desktop_placement_sha256", "")[7:15], p.get("executor_id"), p.get("layer_mask"), tuple(p.get("resource_ids", ())), p.get("split_fraction_ppm"))
        pols.setdefault(ident, set()).add(p.get("policy_hash", "")[7:15])
    print(rid[-3:], "model", g.get("model_artifact_sha256","")[7:15], "geom", (g.get("helper_layout_geometry_sha256") or "")[7:17], "comp", (g.get("component_capability_sha256") or "")[7:15], "plan", (g.get("planning_profile_sha256") or "")[7:15], "ctx", sorted({w["context_length"] for w in g["windows"]})[:3])
    for ident, hs in sorted(pols.items(), key=str):
        print("     ", ident, sorted(hs))
