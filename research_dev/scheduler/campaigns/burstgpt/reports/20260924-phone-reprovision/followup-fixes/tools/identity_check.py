"""Per-model layout identity of every READY generation of a run (root function, duck-typed rows)."""
import json, sys, collections
from types import SimpleNamespace
from research_dev.scheduler._internal.phone_shards import artifact_layout_identity_sha256
run = sys.argv[1]
d = json.load(open(run + "/RESULT.json"))
t0 = None
for e in d["phone_residency_events"]:
    if e.get("kind") != "READY":
        continue
    lay = e.get("layout") or {}
    shards = lay.get("shards") or e.get("shards")
    rows = tuple(SimpleNamespace(**s) for s in shards)
    layout = SimpleNamespace(shards=rows)
    arts = sorted({r.artifact_sha256 for r in rows})
    ids = {a[7:15]: artifact_layout_identity_sha256(layout, a)[7:19] for a in arts}
    per = {a[7:15]: sum(bin(r.layer_mask).count("1") for r in rows if r.artifact_sha256 == a) for a in arts}
    gen = e.get("layout_generation") or e.get("generation") or lay.get("generation")
    print("gen", gen, "geom", (lay.get("geometry_sha256") or e.get("geometry_sha256") or "")[7:17], per, ids)
