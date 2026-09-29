import json, sys
for path in sys.argv[1:]:
    d = json.load(open(path))
    model = {}
    for line in d["log"]:
        p = line.split()
        if p and p[0] == "ACQUIRE":
            model[p[2]] = "g" if "cold:desktop" in p[3] else "q" if "hot:desktop" in p[3] else "l"
    rows = {k.rsplit(":", 1)[-1]: v for k, v in d["requests"].items()}
    large = [k for k in rows if model.get(k) in ("g", "q")]
    # execution interval: after load for loaders (start_us from 'LOADED' lines)
    loaded = {}
    for line in d["log"]:
        p = line.split()
        if p and p[0] == "LOADED":
            loaded[p[2]] = int(p[1])
    ex = {k: (loaded.get(k, rows[k]["acquired_us"]), rows[k]["done_us"]) for k in rows}
    alone_with_queue = paired = 0
    for k in large:
        s, e = ex[k]
        partners = [o for o in large if o != k and model[o] == model[k] and ex[o][0] < e and s < ex[o][1]]
        if partners:
            paired += 1
            continue
        queued = [o for o in large if o != k and model[o] == model[k] and rows[o]["arrival_us"] < e and ex[o][0] > s]
        if queued:
            alone_with_queue += 1
    print(f"{path.split('/')[-1]:<32} large={len(large)} decoded_with_partner={paired} alone_while_same_model_queued={alone_with_queue}")
