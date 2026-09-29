import json, sys
run, sim = sys.argv[1], sys.argv[2]
recs = json.load(open(f"runs/{run}/SCHEDULER_DECISION_LOG.json"))["records"]
real = {}
for r in recs:
    rid = r["request_ids"][0].split(":", 1)[1]
    if r["event_kind"] == "ACQUIRED":
        real.setdefault(rid, {})["acq"] = r["event_time_us"]
    if r["event_kind"] == "COMPLETED":
        real.setdefault(rid, {})["done"] = r["event_time_us"]
s = json.load(open(sim))["requests"]
rows = sorted(s, key=lambda k: s[k]["acquired_us"])
print(f"{'req':<26}{'real acq':>10}{'sim acq':>10}{'real done':>11}{'sim done':>10}")
for k in rows:
    print(f"{k:<26}{real[k]['acq']/1e6:10.1f}{s[k]['acquired_us']/1e6:10.1f}{real[k]['done']/1e6:11.1f}{s[k]['done_us']/1e6:10.1f}")
