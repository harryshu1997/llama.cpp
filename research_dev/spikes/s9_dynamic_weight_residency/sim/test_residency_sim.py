#!/usr/bin/env python3
"""S9-V0-R FROZEN behavior suite (preserved under R1).

Exercises the ten required behaviors and the V0-R additions against the FROZEN V0-R
simulator (golden/v0r_historical/residency_sim_v0r.py) and its frozen v2 config snapshot,
so the V0-R suite results are preserved byte-for-byte after the R1 repairs. R1's own
mechanics are covered by sim/test_r1_regressions.py. Deterministic; no network.
"""
import copy
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location(
    "residency_sim_v0r", os.path.join(SPIKE, "golden", "v0r_historical", "residency_sim_v0r.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)  # FROZEN V0-R simulator

JSONSCHEMA = "/usr/bin/jsonschema"
SCHEMAS = os.path.join(SPIKE, "schemas", "v2")
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))


def js_ok(schema, obj):
    p = os.path.join("/tmp", "s9_sim_inst.json")
    open(p, "w").write(json.dumps(obj))
    r = subprocess.run([JSONSCHEMA, "-i", p, os.path.join(SCHEMAS, schema)], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def base_cfg():
    return {
        "schema_version": 2, "config_id": "t", "seed": 1, "horizon_us": 600000000,
        "link_goodput_sweep_bytes_per_s": [262144000],
        "baselines": ["relief_predictive"],
        "interference": {"htp_gpu_slowdown_permille": 1000, "transfer_compute_slowdown_permille": 1000, "measured": False},
        "queues": {"prefetch_depth": 4, "server_depth": 16},
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
        "workload": {"requests": []},
        "failure_schedule": [],
        "policy_params": {"reuse_horizon_us": 600000000, "min_hold_us": 30000000, "ttl_us": 45000000,
                          "predictor_ewma_permille": 500},
    }


def mkreq(i, t, ws="wsA", model="m", state=4096, deadline=None, cls="decode", out=983040):
    return {"arrival_us": t, "rank": i, "request_id": f"r{i}", "service_class": cls,
            "model_id": model, "weight_set_id": ws, "input_bytes": 983040, "output_bytes": out,
            "state_bytes": state, "deadline_us": deadline}


def sums(r):
    return r["completed_phone"] + r["completed_server"] + r["fallback"] + r["rejected"] + r["timed_out"]


# ---------- 1. determinism + schema validity of the frozen V0-R scenario ----------
scen_path = os.path.join(SPIKE, "golden", "v0r", "baseline_sweep.v0r.config.json")
scen = json.load(open(scen_path))
ok, det = js_ok("sim_config.schema.json", scen)
check("committed scenario validates against sim_config v2 schema", ok, det)
res1 = R.run_sweep(scen)
res2 = R.run_sweep(copy.deepcopy(scen))
man1 = R.build_manifest(scen, scen_path, res1)
man2 = R.build_manifest(scen, scen_path, res2)
check("byte-identical deterministic replay (in-process)",
      man1["deterministic_replay_sha256"] == man2["deterministic_replay_sha256"],
      man1["deterministic_replay_sha256"])
ok, det = js_ok("sim_run_manifest.schema.json", man1)
check("emitted manifest validates against sim_run_manifest v2 schema", ok, det)
# every baseline of every goodput row terminalizes exactly the request count
allrows = [(row["goodput_bytes_per_s"], b) for row in res1 for b in row["baselines"]]
check("every baseline: 5 terminal buckets sum to request count",
      all(sums(b) == b["requests"] for _, b in allrows),
      f"{len(allrows)} baseline runs")
check("server_only never dispatches to a phone",
      all(b["dispatched_to_phone"] == 0 for _, b in allrows if b["policy"] == "server_only"))
check("clairvoyant labeled upper bound; per_request_fetch labeled diagnostic",
      all(b["is_upper_bound"] for _, b in allrows if b["policy"] == "clairvoyant") and
      all(b["is_diagnostic_losing"] for _, b in allrows if b["policy"] == "per_request_fetch"))

# ---------- 2. hash mismatch: exact outcomes + ledger rolled back ----------
c = base_cfg(); c["config_id"] = "hashmm"
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 40000000)]
c["failure_schedule"] = [{"at_us": 0, "kind": "hash_mismatch", "target": "wsA"}]
s = R.Sim(c, 262144000, "relief_predictive"); r = s.run()
check("hash mismatch -> wsA QUARANTINED", "wsA" in s.quarantined)
check("hash mismatch -> exact outcomes (fallback==2, no phone, ledger canonical rolled back)",
      r["fallback"] == 2 and r["completed_phone"] == 0 and r["rejected"] == 0 and sums(r) == 2
      and s.ledger["op15"].canonical == 0 and s.ledger["op15"].ufs_used == 0,
      f"outc cp={r['completed_phone']} fb={r['fallback']}")

# ---------- 3. partial transfer recovery (resume, retry bytes, reaches READY, later hit) ----------
c = base_cfg(); c["config_id"] = "partial"
c["models"][0]["canonical_bytes"] = 400000000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 40000000)]
c["failure_schedule"] = [{"at_us": 100, "kind": "partial_transfer", "target": "wsA"}]
s = R.Sim(c, 262144000, "relief_predictive"); r = s.run()
rr = s.res[("op15", "wsA")]
check("partial transfer -> resumed, exact retry bytes, reaches READY, later request hits phone",
      "wsA" in s.resumed_sets and r["retry_bytes"] == 400000000 // 4 and
      r["transfer_bytes"] == 400000000 + 400000000 // 4 and rr.state in ("READY_HTP", "READY_GPU")
      and r["completed_phone"] >= 1 and sums(r) == 2,
      f"retry={r['retry_bytes']} cp={r['completed_phone']}")

# ---------- 4. frame gate: duplicate / stale / reordered rejection ----------
fs = R.FrameSequencer()
def frame(seq, be=7, re=2, idem=None):
    return {"conn": "c0", "channel": "control", "seq": seq, "boot_epoch": be,
            "residency_epoch": re, "idempotency_key": idem or f"k{seq}"}
a0, _ = fs.accept(frame(0)); a1, _ = fs.accept(frame(1))
dup, dr = fs.accept(frame(1, idem="k1b"))
reo, rr2 = fs.accept(frame(5))
a2, _ = fs.accept(frame(2))
stale, sr = fs.accept(frame(3, be=6))
idem, ir = fs.accept(frame(3, idem="k0"))
check("frame gate: in-order accepted; dup/reorder/stale/replay all rejected fail-closed",
      a0 and a1 and a2 and not dup and dr == "duplicate_message" and not reo and rr2 == "reordered_message"
      and not stale and sr == "stale_epoch" and not idem and ir == "duplicate_message")

# ---------- 5. lease-safe eviction + exact accounting + no live-state eviction ----------
c = base_cfg(); c["config_id"] = "evict"; c["baselines"] = ["lru"]
c["phones"][0]["lpddr_total_bytes"] = 1000000000
c["models"] = [
    {"model_id": "mA", "weight_set_id": "wsA", "canonical_bytes": 600000000, "derived_bytes_htp": 0,
     "derived_bytes_gpu": 0, "scratch_bytes": 50000000, "partial_load_supported": True,
     "eligible_backends": ["htp"], "phone_compute_us": 3000, "state_policy": "stateless"},
    {"model_id": "mB", "weight_set_id": "wsB", "canonical_bytes": 600000000, "derived_bytes_htp": 0,
     "derived_bytes_gpu": 0, "scratch_bytes": 50000000, "partial_load_supported": True,
     "eligible_backends": ["htp"], "phone_compute_us": 3000, "state_policy": "stateless"}]
c["workload"]["requests"] = [
    mkreq(0, 0, ws="wsA", model="mA"), mkreq(1, 30000000, ws="wsA", model="mA"),
    mkreq(2, 45000000, ws="wsB", model="mB"), mkreq(3, 90000000, ws="wsB", model="mB")]
s = R.Sim(c, 262144000, "lru"); r = s.run()
s.ledger["op15"].check()
check("tight RAM forces >=1 eviction; no live-state eviction; ledger partitions valid",
      r["evictions"] >= 1 and s.live_state_evictions == 0 and sums(r) == 4)

# stale_epoch while a lease is LIVE -> drain-before-eviction (real, non-vacuous guard)
c3 = base_cfg(); c3["config_id"] = "stale_pin"
c3["models"][0]["phone_compute_us"] = 20000000
c3["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 3000000)]
c3["failure_schedule"] = [{"at_us": 6000000, "kind": "stale_epoch", "target": "op15"}]
s3 = R.Sim(c3, 262144000, "relief_predictive"); r3 = s3.run()
check("stale_epoch during a live lease never frees pinned bytes (real guard)",
      s3.live_state_evictions == 0 and sums(r3) == 2)

# ---------- 6. no dispatch from partial / merely on-disk weights ----------
c = base_cfg(); c["config_id"] = "ondisk"; c["models"][0]["canonical_bytes"] = 900000000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 1000000)]
s = R.Sim(c, 262144000, "relief_predictive"); r = s.run()
check("weights merely in-pipeline never dispatched (both fall back)",
      r["completed_phone"] == 0 and r["fallback"] == 2 and sums(r) == 2)
# the guard is non-vacuous: calling dispatch on a non-ready set returns False and trips the counter
s2 = R.Sim(base_cfg(), 262144000, "relief_predictive")
before = s2.dispatched_from_on_disk
did = s2.dispatch_phone("op15", "htp", mkreq(0, 0))
check("dispatch_phone on a non-ready set is refused (guard non-vacuous)",
      did is False and s2.dispatched_from_on_disk == before + 1)

# ---------- 7. activation preempts bulk prefetch (retains link ownership) ----------
c = base_cfg(); c["config_id"] = "preempt"; c["models"][0]["canonical_bytes"] = 900000000
s = R.Sim(c, 41943040, "relief_predictive"); s.t = 0
s.start_prefetch("op15", "wsA")
before = s.preemptions
s._link_send("op15", 983040, is_result=False)
check("activation preempts in-flight bulk; link ownership retained",
      s.preemptions == before + 1 and s.domain_bulk["d15"] is not None)

# ---------- 8. unknown profile / unsupported backend fail closed (exact outcomes) ----------
for kind, tgt, cid in [("unknown_profile", "op15", "unk"), ("unsupported_backend", "wsA", "unsup")]:
    c = base_cfg(); c["config_id"] = cid
    c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 40000000)]
    c["failure_schedule"] = [{"at_us": 0, "kind": kind, "target": tgt}]
    s = R.Sim(c, 262144000, "relief_predictive"); r = s.run()
    check(f"{kind} -> no phone dispatch, all fall back (sum exact)",
          r["completed_phone"] == 0 and r["fallback"] == 2 and sums(r) == 2)

# ---------- 9. D2H is an explicit completion blocker ----------
c = base_cfg(); c["config_id"] = "d2h"
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 5000000, out=983040)]
s = R.Sim(c, 262144000, "relief_predictive"); r = s.run()
# at least one phone completion happened, and D2H bytes were charged for the returned outputs
check("D2H result transfer on the completion path (d2h_bytes charged per phone completion)",
      r["completed_phone"] >= 1 and r["d2h_bytes"] == r["completed_phone"] * 983040,
      f"cp={r['completed_phone']} d2h={r['d2h_bytes']}")

# ---------- 10. bounded queues: prefetch drop + server reject ----------
c = base_cfg(); c["config_id"] = "server_bound"
c["server"]["gpu_lane_slots"] = 1
c["queues"]["server_depth"] = 1
c["server"]["per_class_gpu_us"] = {"decode": 50000000}      # long server jobs so the queue saturates
c["workload"]["requests"] = [mkreq(i, i * 100000) for i in range(6)]
s = R.Sim(c, 262144000, "server_only"); r = s.run()
check("bounded server queue rejects overflow (fail-closed terminal)",
      r["rejected"] >= 1 and sums(r) == 6)
# prefetch queue bound: many distinct sets contend on one domain, some prefetches dropped
c = base_cfg(); c["config_id"] = "pf_bound"; c["queues"]["prefetch_depth"] = 1
c["models"] = [{"model_id": f"m{i}", "weight_set_id": f"ws{i}", "canonical_bytes": 300000000,
                "derived_bytes_htp": 0, "derived_bytes_gpu": 0, "scratch_bytes": 10000000,
                "partial_load_supported": True, "eligible_backends": ["htp"], "phone_compute_us": 3000,
                "state_policy": "stateless"} for i in range(5)]
c["workload"]["requests"] = [mkreq(i, 0, ws=f"ws{i}", model=f"m{i}") for i in range(5)]
s = R.Sim(c, 41943040, "relief_predictive"); r = s.run()
check("bounded prefetch queue drops overflow (logged, not silent)",
      r["prefetch_drops"] >= 1 and sums(r) == 5, f"prefetch_drops={r['prefetch_drops']}")

# ---------- 11. interference ONLY over the actual overlap window ----------
c = base_cfg(); c["config_id"] = "interf"
c["interference"] = {"htp_gpu_slowdown_permille": 2000, "transfer_compute_slowdown_permille": 2000, "measured": True}
s = R.Sim(c, 262144000, "relief_predictive"); s.t = 1000
no_overlap = s.overlap_extra("op15", 5000, ("gpu",), 2000)
s.lane_busy_until[("op15", "gpu")] = 1000 + 3000            # competitor busy for 3000 of the 5000us window
partial = s.overlap_extra("op15", 5000, ("gpu",), 2000)
check("interference applies only over the overlap window (0 when disjoint; 3000 when 3000 overlaps)",
      no_overlap == 0 and partial == 3000, f"none={no_overlap} partial={partial}")

# ---------- 12. policy distinctness (each of the four has a distinguishing behavior) ----------
# 12a static spreads across devices; fastest_ready concentrates on the tiebreak device
c = base_cfg(); c["config_id"] = "place"
c["phones"].append({**copy.deepcopy(c["phones"][0]), "device_id": "op12", "soc": "op12", "contention_domain": "d12"})
c["models"].append({"model_id": "mB", "weight_set_id": "wsB", "canonical_bytes": 100000000,
                    "derived_bytes_htp": 0, "derived_bytes_gpu": 0, "scratch_bytes": 10000000,
                    "partial_load_supported": True, "eligible_backends": ["htp"], "phone_compute_us": 3000,
                    "state_policy": "stateless"})
reqs = []
for i in range(6):
    reqs.append(mkreq(2 * i, i * 8000000, ws="wsA", model="m"))
    reqs.append(mkreq(2 * i + 1, i * 8000000 + 1000, ws="wsB", model="mB"))
c["workload"]["requests"] = reqs
r_static = R.Sim(copy.deepcopy(c), 262144000, "static_placement").run()
r_fast = R.Sim(copy.deepcopy(c), 262144000, "fastest_ready").run()
check("static_placement spreads across both phones (distinct multi-device behavior)",
      r_static["dispatched_by_device"].get("op12", 0) >= 1 and r_static["dispatched_by_device"].get("op15", 0) >= 1)
check("fastest_ready concentrates differently from static (distinct placement)",
      r_fast["dispatched_by_device"] != r_static["dispatched_by_device"],
      f"static={r_static['dispatched_by_device']} fast={r_fast['dispatched_by_device']}")

# 12b relief_predictive skips prefetch when server relief is 0; fastest_ready does not
c = base_cfg(); c["config_id"] = "relief0"
c["server"]["per_class_gpu_us"] = {"decode": 0}             # zero relief
c["server"]["per_class_hbm_bytes"] = {"decode": 0}
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 20000000)]
r_relief = R.Sim(copy.deepcopy(c), 262144000, "relief_predictive").run()
r_fast2 = R.Sim(copy.deepcopy(c), 262144000, "fastest_ready").run()
check("relief_predictive declines to prefetch at zero relief; fastest_ready still warms the phone",
      r_relief["dispatched_to_phone"] == 0 and r_fast2["dispatched_to_phone"] >= 1,
      f"relief_disp={r_relief['dispatched_to_phone']} fast_disp={r_fast2['dispatched_to_phone']}")

# 12c clairvoyant (Belady) is at least as good as lru under the same eviction pressure
c = base_cfg(); c["config_id"] = "belady"; c["phones"][0]["lpddr_total_bytes"] = 1000000000
c["models"] = [{"model_id": f"m{i}", "weight_set_id": f"ws{i}", "canonical_bytes": 300000000,
                "derived_bytes_htp": 0, "derived_bytes_gpu": 0, "scratch_bytes": 10000000,
                "partial_load_supported": True, "eligible_backends": ["htp"], "phone_compute_us": 3000,
                "state_policy": "stateless"} for i in range(3)]
pattern = [0, 1, 2, 0, 2, 0, 1, 2]
c["workload"]["requests"] = [mkreq(i, 20000000 + i * 20000000, ws=f"ws{pattern[i]}", model=f"m{pattern[i]}")
                             for i in range(len(pattern))]
r_lru = R.Sim(copy.deepcopy(c), 262144000, "lru").run()
r_clv = R.Sim(copy.deepcopy(c), 262144000, "clairvoyant").run()
check("clairvoyant (Belady) evicts no worse than lru under pressure (distinct eviction policy)",
      r_clv["completed_phone"] >= r_lru["completed_phone"] and sums(r_clv) == len(pattern),
      f"lru cp={r_lru['completed_phone']} clv cp={r_clv['completed_phone']}")

# ---------- 13. horizon termination exact taxonomy ----------
c = base_cfg(); c["config_id"] = "horizon"; c["horizon_us"] = 1000000
c["models"][0]["canonical_bytes"] = 900000000
c["workload"]["requests"] = [mkreq(0, 0)]
s = R.Sim(c, 41943040, "per_request_fetch"); r = s.run()
check("request that cannot finish by horizon -> timed_out (exact taxonomy)",
      r["timed_out"] == 1 and r["completed_phone"] == 0 and sums(r) == 1)

n_fail = sum(1 for _, ok in results if not ok)
print(f"\nsim tests: {len(results)}  failures: {n_fail}")
sys.exit(1 if n_fail else 0)
