import json, sys, statistics
def model_of(log, rid):
    return None
for path in sys.argv[1:]:
    d = json.load(open(path))
    reqs = d["requests"]
    loads = [x for x in d["loads"] if x["model"] != "llama"]
    # model per request from ACQUIRE lines
    kind = {}
    for line in d["log"]:
        p = line.split()
        if p and p[0] == "ACQUIRE":
            kind[p[2]] = "g" if ":cold:" in p[3] or "cold:desktop" in p[3] else ("q" if "hot:desktop" in p[3] else "l")
    lat = {k: (v["done_us"] - v["arrival_us"]) / 1e6 for k, v in reqs.items()}
    wait = {k: (v["acquired_us"] - v["arrival_us"]) / 1e6 for k, v in reqs.items()}
    large = [k for k in reqs if "llama" not in k]
    print(f"{path.split('/')[-1]:<34} n={len(reqs)} makespan={d['makespan_us']/1e6:7.1f}s loads={len(loads):2d} "
          f"mean_latency={statistics.mean(lat[k] for k in large):7.1f}s p95/max_latency={sorted(lat[k] for k in large)[int(0.95*len(large))-1]:7.1f}/{max(lat[k] for k in large):7.1f}s "
          f"mean_wait={statistics.mean(wait[k] for k in large):7.1f}s max_wait={max(wait[k] for k in large):7.1f}s")
