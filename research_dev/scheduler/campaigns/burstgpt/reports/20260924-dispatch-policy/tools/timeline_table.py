import json, sys
run = sys.argv[1]; names = sys.argv[2:]
runs = {n: json.load(open(f"sim/final/{run}_{n}.json")) for n in names}
real = {}
for r in json.load(open(f"runs/{run}/SCHEDULER_DECISION_LOG.json"))["records"]:
    k = r["request_ids"][0].rsplit(":", 1)[-1]
    if r["event_kind"] == "COMPLETED": real[k] = r["event_time_us"]/1e6
model = {}
for line in runs[names[0]]["log"]:
    p = line.split()
    if p and p[0] == "ACQUIRE":
        model[p[2]] = "Gemma" if "cold:desktop" in p[3] else "Qwen" if "hot:desktop" in p[3] else "Llama"
print("| request | model | arrival s | real done s | " + " | ".join(f"{n} start-done s" for n in names) + " |")
print("|---|---|---:|---:|" + "---:|" * len(names))
base = runs[names[0]]["requests"]
for k in sorted(base, key=lambda x: base[x]["arrival_us"]):
    short = k.rsplit(":", 1)[-1]
    cells = [f"{runs[n]['requests'][k]['acquired_us']/1e6:.0f}-{runs[n]['requests'][k]['done_us']/1e6:.0f}" for n in names]
    print(f"| {k} | {model.get(short, '?')} | {base[k]['arrival_us']/1e6:.0f} | {real[short]:.0f} | " + " | ".join(cells) + " |")
for n, d in runs.items():
    print(n, "loads:", ", ".join(x['model'] + ' ' + x['request'].rsplit(':',1)[-1] + ' @' + '%.0f' % (x['start_us']/1e6) for x in d['loads']))
