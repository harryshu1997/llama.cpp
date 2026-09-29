import json, sys, collections
run = sys.argv[1]
d = json.load(open(run + "/RESULT.json"))
he = d["request_helper_events"]
print(collections.Counter(e.get("kind") for e in he).most_common(40))
rows = [e for e in he if e.get("kind") == "ASSISTANCE_DECISION"]
print("ASSISTANCE_DECISION", len(rows))
if rows:
    print(sorted(rows[0].keys()))
by = collections.defaultdict(list)
for e in rows: by[e["request_id"][-3:]].append(e)
for rid in sorted(by):
    rs = by[rid]
    first = rs[0]
    sp = first.get("server_policy") or {}
    gens = collections.Counter((e.get("phone_layout_generation") or e.get("helper_layout_generation")) for e in rs)
    reasons = collections.Counter((e.get("reason"), e.get("zero_assistance_reason"), e.get("fraction_ppm") if "fraction_ppm" in e else e.get("selected_fraction_ppm")) for e in rs)
    print(rid, "n", len(rs), "gens", dict(gens))
    print("   first server_policy", {k: sp.get(k) for k in ("owner_request_id","verdicts","reason","records","layout_identity_sha256")})
    last = rs[-1].get("server_policy") or {}
    print("   last  server_policy", {k: last.get(k) for k in ("owner_request_id","verdicts","reason","records")})
    print("   reasons", reasons.most_common(6))
