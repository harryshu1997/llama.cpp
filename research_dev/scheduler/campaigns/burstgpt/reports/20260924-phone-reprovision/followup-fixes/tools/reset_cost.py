"""Host-policy window tokens each request spent before its first phone window (the verdict reset)."""
import json, sys
run = sys.argv[1]
obs = json.load(open(run + "/ADAPTIVE_DECODE_OBSERVATIONS.json"))
d = json.load(open(run + "/RESULT.json"))
current = {r["request_id"] for r in d["request_results"]} if isinstance(d.get("request_results"), list) else None
for g in obs["groups"]:
    rid = g["request_id"]
    if current is not None and rid not in current:
        continue
    windows = sorted(g["windows"], key=lambda w: w["token_start"])
    host_before = 0; host_total = 0; phone_total = 0; first_phone = None
    for w in windows:
        n = w["token_end"] - w["token_start"]
        if w["policy"]["baseline"]:
            host_total += n
            if first_phone is None:
                host_before += n
        else:
            phone_total += n
            if first_phone is None:
                first_phone = w["token_start"]
    print(rid[-3:], "windows", len(windows), "host tokens before first phone window", host_before,
          "host total", host_total, "phone total", phone_total, "first phone token", first_phone,
          "final", g["final_policy"]["split_fraction_ppm"], g.get("terminal_reason"))
