import sys, copy
sys.path.insert(0, "sim")
import residency_sim as R1

def base_cfg():
    return {
        "schema_version": 3, "config_id": "t", "seed": 1, "horizon_us": 600000000,
        "link_goodput_sweep_bytes_per_s": [262144000], "baselines": ["relief_predictive"],
        "interference": {"htp_gpu_slowdown_permille": 1000, "transfer_compute_slowdown_permille": 1000, "measured": False},
        "queues": {"prefetch_depth": 4, "server_depth": 16, "lane_depth": 8},
        "server": {"gpu_lane_slots": 4, "hbm_credit_bytes": 4000000000,
                   "per_class_gpu_us": {"decode": 2000}, "per_class_hbm_bytes": {"decode": 1000000}},
        "phones": [{"device_id": "op15", "soc": "op15", "contention_domain": "d15", "boot_epoch": 7,
                    "lpddr_total_bytes": 10000000000, "ufs_total_bytes": 128000000000,
                    "ufs_write_bytes_per_s": 800000000, "verify_bytes_per_s": 1500000000,
                    "materialize_bytes_per_s": 6000000000, "prepare_htp_bytes_per_s": 4000000000,
                    "prepare_gpu_bytes_per_s": 3000000000, "warmup_us": 50000, "htp_lane_slots": 2,
                    "gpu_lane_slots": 1, "thermal_eligible": True}],
        "models": [{"model_id": "m", "weight_set_id": "wsA", "canonical_bytes": 100000000,
                    "derived_bytes_htp": 0, "derived_bytes_gpu": 0, "scratch_bytes": 10000000,
                    "partial_load_supported": True, "eligible_backends": ["htp"], "phone_compute_us": 3000,
                    "state_policy": "stateless"}],
        "workload": {"requests": []}, "failure_schedule": [],
        "policy_params": {"reuse_horizon_us": 600000000, "min_hold_us": 30000000, "ttl_us": 45000000,
                          "predictor_ewma_permille": 500},
    }

def mkreq(i, t, ws="wsA", model="m", state=4096, out=983040, cls="decode"):
    return {"arrival_us": t, "rank": i, "request_id": f"r{i}", "service_class": cls, "model_id": model,
            "weight_set_id": ws, "input_bytes": 983040, "output_bytes": out, "state_bytes": state, "deadline_us": None}

cfg = base_cfg()
cfg["models"][0]["canonical_bytes"] = 900000000
cfg["workload"]["requests"] = [mkreq(0, 0)]
cfg["failure_schedule"] = [{"at_us": 100, "kind": "partial_transfer", "target": "wsA", "verified_offset": 1400000000}]
s = R1.Sim(cfg, 262144000, "static_placement")
res = s.run()
r = s.res.get(("op15","wsA"))
print("state=", r.state if r else None, "resumed=", r.resumed if r else None, "retry_bytes=", res["retry_bytes"])
print("canonical=", 900000000, "verified_offset=", 1400000000)

print("--- full accounting ---")
res2 = res
keys = ["arrivals","completed","fallback","rejected","timed_out","retry_bytes","transfer_bytes","ufs_bytes","link_drop_resumed"]
for k in keys:
    if k in res2: print(k, res2[k])
tot = res2.get("completed",0)+res2.get("fallback",0)+res2.get("rejected",0)+res2.get("timed_out",0)
print("sum term=", tot, "arrivals=", res2.get("arrivals"))

# Compare: offset within canonical (case-15 style) vs the impossible offset
print("--- control: verified_offset=200000000 (<canonical) ---")
cfg2 = base_cfg(); cfg2["models"][0]["canonical_bytes"]=900000000
cfg2["workload"]["requests"]=[mkreq(0,0)]
cfg2["failure_schedule"]=[{"at_us":100,"kind":"partial_transfer","target":"wsA","verified_offset":200000000}]
s2=R1.Sim(cfg2,262144000,"static_placement"); r2res=s2.run()
print("retry_bytes=", r2res["retry_bytes"], "expected 900000000-200000000=", 900000000-200000000)

print("--- all result keys ---")
for k in sorted(res2.keys()):
    print(k, "=", res2[k])
