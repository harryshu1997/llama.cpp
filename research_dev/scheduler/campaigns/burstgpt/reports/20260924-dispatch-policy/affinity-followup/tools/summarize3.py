import json, sys, glob, os
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for path in sorted(glob.glob(W + "/" + (sys.argv[1] if len(sys.argv) > 1 else "sim") + "/*.json")):
    name = os.path.basename(path)[:-5]
    if name.endswith("_trace"):
        continue
    d = json.load(open(path))
    if "loads" not in d:
        continue
    large = [x for x in d["loads"] if x["model"] != "llama"]
    st = (d.get("dispatch_policy_state") or {}).get("statistics", {})
    reqs = d["requests"]
    lat = [v["done_us"] - v["arrival_us"] for v in reqs.values()]
    print(f"{name:22s} makespan {d['makespan_us']/1e6:8.1f} s  large loads {len(large):2d} {''.join(x['model'][0].upper() for x in d['loads'])}  mean/max lat {sum(lat)/len(lat)/1e6:7.1f}/{max(lat)/1e6:7.1f}  disp {st.get('affinity_displacements','-')} ({st.get('affinity_displaced_attempts','-')}) ref {st.get('affinity_refusals','-')}")
