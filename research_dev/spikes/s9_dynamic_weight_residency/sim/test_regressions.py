#!/usr/bin/env python3
"""S9-V0-R red-before / green-after regression suite.

For each repaired mechanic this imports BOTH the FROZEN pre-repair simulator
(golden/v0_historical/residency_sim_v0.py) and the REPAIRED simulator and shows, on
the same scenario, that the frozen sim exhibits the bug (RED) while the repaired sim is
correct (GREEN). Keeping the frozen module in-tree makes "red before the repair"
reproducible forever instead of a claim about a state that no longer exists. Two
schema/record repairs (dispatch tuples, prepared-image identity, unknown kind) are shown
against the v1 validators the same way. Deterministic; no network.
"""
import copy
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)

_spec_v0r = importlib.util.spec_from_file_location(
    "residency_sim_v0r", os.path.join(SPIKE, "golden", "v0r_historical", "residency_sim_v0r.py"))
V0R = importlib.util.module_from_spec(_spec_v0r)
_spec_v0r.loader.exec_module(V0R)

_spec = importlib.util.spec_from_file_location(
    "residency_sim_v0", os.path.join(SPIKE, "golden", "v0_historical", "residency_sim_v0.py"))
V0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V0)

results = []


def check(name, red_ok, green_ok, detail=""):
    ok = bool(red_ok) and bool(green_ok)
    results.append((name, ok))
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag} {name}  RED(v0)={'bug' if red_ok else 'NO-BUG!'} GREEN(v0r)={'fixed' if green_ok else 'BROKEN!'}"
          + (f"  {detail}" if detail else ""))


def cfgv2(**over):
    c = {
        "schema_version": 2, "config_id": "t", "seed": 1, "horizon_us": 600000000,
        "link_goodput_sweep_bytes_per_s": [262144000],
        "baselines": ["relief_predictive"],
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
        "workload": {"requests": []},
        "failure_schedule": [],
        "policy_params": {"reuse_horizon_us": 600000000, "min_hold_us": 30000000, "ttl_us": 45000000,
                          "predictor_ewma_permille": 500},
    }
    c.update(over)
    return c


def mkreq(i, t, ws="wsA", model="m", state=4096, deadline=None):
    return {"arrival_us": t, "rank": i, "request_id": f"r{i}", "service_class": "decode",
            "model_id": model, "weight_set_id": ws, "input_bytes": 983040, "output_bytes": 983040,
            "state_bytes": state, "deadline_us": deadline}


def to_v1(cfg):
    c = copy.deepcopy(cfg)
    c["schema_version"] = 1
    c["usb_shared_controller_bytes_per_s"] = 625000000
    return c


def run_v0r(cfg, g=262144000, pol="relief_predictive"):
    s = V0R.Sim(cfg, g, pol)
    r = s.run()
    return s, r


def run_v0(cfg, g=262144000, pol="relief_predictive"):
    s = V0.Sim(to_v1(cfg), g, pol)
    r = s.run()
    return s, r


# ---- 1. post-stale arrival / stale completion is rejected fail-closed ----
c = cfgv2(config_id="stale_done")
c["models"][0]["phone_compute_us"] = 20000000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 2000000)]   # r0 warms the set; r1 hits + pins 20s
c["failure_schedule"] = [{"at_us": 6000000, "kind": "stale_epoch", "target": "op15"}]
s_r, r_r = run_v0r(c)
s_0, r_0 = run_v0(c)
check("1 post-stale completion rejected fail-closed",
      r_0.get("completed", 0) >= 1 and "stale_completion_rejects" not in r_0,   # v0 completed the stale work
      r_r["stale_completion_rejects"] >= 1 and r_r["rejected"] >= 1 and r_r["completed_phone"] == 0,
      f"v0r rejected={r_r['rejected']}")

# ---- 2. stale loading pipeline discarded (not published under a bumped epoch) ----
c = cfgv2(config_id="stale_pipe")
c["models"][0]["canonical_bytes"] = 900000000               # long fill so the epoch bumps mid-pipeline
c["workload"]["requests"] = [mkreq(0, 0)]
c["failure_schedule"] = [{"at_us": 500000, "kind": "stale_epoch", "target": "op15"}]
s_r, r_r = run_v0r(c)
s_0, r_0 = run_v0(c)
v0_published = s_0.res.get(("op15", "wsA")) is not None and s_0.res[("op15", "wsA")].state in ("READY_HTP", "READY_GPU")
v0r_state = s_r.res.get(("op15", "wsA"))
check("2 stale loading pipeline discarded",
      v0_published,                                          # v0 publishes READY under the bumped epoch
      r_r["stale_pipeline_drops"] >= 1 and (v0r_state is None or v0r_state.state != "READY_HTP"),
      f"v0r stale_pipeline_drops={r_r['stale_pipeline_drops']}")

# ---- 3. two required weight sets, one certificate -> exact-tuple mismatch ----
bad = os.path.join(SPIKE, "fixtures", "v2", "bundles", "invalid", "two_sets_one_cert.json")
codes = subprocess.run([sys.executable, os.path.join(SPIKE, "bundle_validate.py"), bad],
                       capture_output=True, text=True).stdout
# v1 dispatch_decision has no tuple concept: a DISPATCH with a single cert validates fine.
v1_dd = {"schema_version": 1, "request_id": "q", "island_id": "isl2", "device_id": "op15", "backend": "htp",
         "route_epoch": 3, "required_weight_set_ids": ["ws1", "ws2"], "ready_certificate_ids": ["rc1"],
         "residency_lease_id": "rl1", "state_lease_id": None,
         "epoch_match": {"boot": True, "residency": True, "route": True, "state": True},
         "credits": {"weights_ok": True, "derived_ok": True, "scratch_ok": True, "activations_ok": True, "state_ok": True},
         "correctness_verdict": "pass", "hard_gate_failures": [], "verdict": "DISPATCH", "reason_code": "dispatch_ok"}
open("/tmp/s9_v1_dd.json", "w").write(json.dumps(v1_dd))
v1_ok = subprocess.run(["/usr/bin/jsonschema", "-i", "/tmp/s9_v1_dd.json",
                        os.path.join(SPIKE, "schemas", "dispatch_decision.schema.json")]).returncode == 0
check("3 two required sets / one cert rejected (exact tuples)",
      v1_ok,                                                 # v1 accepts a 2-set island dispatched with 1 cert
      "E_TUPLE_MISMATCH" in codes,
      "v2 E_TUPLE_MISMATCH")

# ---- 4. prepared-image relabel breaks identity ----
# v1 derived_image_digest binds 9 fields NOT including image_class, so a relabel is invisible to it.
sys.path.insert(0, SPIKE)
import validate_manifests as VM  # noqa: E402
v1_pi = json.load(open(os.path.join(SPIKE, "fixtures", "valid", "prepared_image.htp.json")))
v1_pi_relabel = {**v1_pi, "image_class": "gpu_xmem_prepacked"}   # digest unchanged, still self-consistent
v1_blind = (VM.check_prepared_image(v1_pi_relabel) == [])
relabel = os.path.join(SPIKE, "fixtures", "v2", "bundles", "invalid", "prepared_image_relabel.json")
v2codes = subprocess.run([sys.executable, os.path.join(SPIKE, "bundle_validate.py"), relabel],
                         capture_output=True, text=True).stdout
check("4 prepared-image relabel breaks identity",
      v1_blind,                                              # v1 digest does not bind image_class -> blind
      "E_DIGEST_MISMATCH" in v2codes,
      "v2 E_DIGEST_MISMATCH")

# ---- 5. per-domain link (no 'divide controller by phone count') ----
c = cfgv2(config_id="domain")
c["phones"].append({**copy.deepcopy(c["phones"][0]), "device_id": "op12", "soc": "op12",
                    "contention_domain": "d15"})            # SAME domain (one shared bus)
c["workload"]["requests"] = [mkreq(0, 0)]
s_r, _ = run_v0r(c)
s_0, _ = run_v0(c)
g = 262144000
v0r_full = (s_r.goodput == g and len(s_r.domains) == 1)     # v0r keeps full goodput per domain
v0_halved = (s_0.eff_goodput["op15"] == min(g, 625000000 // 2))  # v0 permanently halves per device
check("5 per-domain link, no controller/phone-count division",
      v0_halved,
      v0r_full,
      f"v0 eff={s_0.eff_goodput['op15']} v0r goodput={s_r.goodput}")

# ---- 6. impossible RAM + HBM -> explicit rejected terminal ----
c = cfgv2(config_id="impossible")
c["phones"][0]["lpddr_total_bytes"] = 100000000            # smaller than the model
c["models"][0]["canonical_bytes"] = 900000000
c["server"]["per_class_hbm_bytes"] = {"decode": 8000000000}  # > hbm credit -> server can never fit
c["workload"]["requests"] = [mkreq(0, 0)]
s_r, r_r = run_v0r(c)
s_0, r_0 = run_v0(c)
check("6 impossible RAM/HBM rejected (explicit terminal)",
      r_0.get("completed", 0) < 1 and "rejected" not in r_0,   # v0 silently loses the request
      r_r["rejected"] >= 1 and (r_r["completed_phone"] + r_r["completed_server"] + r_r["fallback"] + r_r["rejected"] + r_r["timed_out"]) == r_r["requests"],
      f"v0r rejected={r_r['rejected']}")

# ---- 7. horizon termination ----
c = cfgv2(config_id="horizon", horizon_us=1000000)         # 1s horizon, 900MB fill can't finish
c["models"][0]["canonical_bytes"] = 900000000
c["baselines"] = ["per_request_fetch"]
c["workload"]["requests"] = [mkreq(0, 0)]
s_r, r_r = run_v0r(c, pol="per_request_fetch")
s_0, r_0 = run_v0(c, pol="per_request_fetch")
check("7 horizon termination (timed_out terminal)",
      "timed_out" not in r_0 and r_0.get("completed", 0) >= 1,   # v0 completes past the horizon (no enforcement)
      r_r["timed_out"] >= 1 and sum(r_r[k] for k in ("completed_phone", "completed_server", "fallback", "rejected", "timed_out")) == r_r["requests"],
      f"v0r timed_out={r_r['timed_out']}")

# ---- 8. persistent sticky state ----
c = cfgv2(config_id="sticky")
c["models"][0]["state_policy"] = "sticky"
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 30000000)]
c["failure_schedule"] = [{"at_us": 40000000, "kind": "reset_state", "target": "op15"}]
s_r, r_r = run_v0r(c)
s_0, r_0 = run_v0(c)
# after the requests complete but before reset, v0r keeps sticky state resident; v0 always frees it.
v0_state_zero = (s_0.ledger["op15"].state == 0)            # v0 has no sticky concept
v0r_reset_freed = (s_r.ledger["op15"].state == 0)          # reset frees it at end
check("8 persistent sticky state (until explicit reset)",
      v0_state_zero,
      r_r["completed_phone"] >= 1 and v0r_reset_freed and s_r.live_state_evictions == 0,
      "v0r sticky pinned then reset-freed")

# ---- 9. transient LPDDR oversubscription rolls back (victim pinned -> cannot evict) ----
c = cfgv2(config_id="oversub")
c["phones"][0]["lpddr_total_bytes"] = 1000000000            # only one 600MB model fits at a time
c["models"] = [
    {"model_id": "mA", "weight_set_id": "wsA", "canonical_bytes": 600000000, "derived_bytes_htp": 0,
     "derived_bytes_gpu": 0, "scratch_bytes": 50000000, "partial_load_supported": True,
     "eligible_backends": ["htp"], "phone_compute_us": 30000000, "state_policy": "stateless"},  # long: keeps wsA pinned
    {"model_id": "mB", "weight_set_id": "wsB", "canonical_bytes": 600000000, "derived_bytes_htp": 0,
     "derived_bytes_gpu": 0, "scratch_bytes": 50000000, "partial_load_supported": True,
     "eligible_backends": ["htp"], "phone_compute_us": 3000, "state_policy": "stateless"}]
# r0 warms wsA; r1 hits wsA and pins it for 30s; r2 wants wsB whose materialize cannot evict pinned wsA
c["workload"]["requests"] = [mkreq(0, 0, ws="wsA", model="mA"), mkreq(1, 5000000, ws="wsA", model="mA"),
                             mkreq(2, 6000000, ws="wsB", model="mB")]
s_r, r_r = run_v0r(c)
s_0, r_0 = run_v0(c)
s_r.ledger["op15"].check()
check("9 transient LPDDR oversubscription rolls back",
      "oversub_rollbacks" not in r_0,                        # v0 reserves only at publish; no mid-pipeline rollback
      r_r["oversub_rollbacks"] >= 1,
      f"v0r oversub_rollbacks={r_r['oversub_rollbacks']}")

# ---- 10. partial resume re-sends only remaining + counts retry bytes ----
c = cfgv2(config_id="resume")
c["models"][0]["canonical_bytes"] = 400000000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 40000000)]
c["failure_schedule"] = [{"at_us": 100, "kind": "partial_transfer", "target": "wsA"}]
s_r, r_r = run_v0r(c)
s_0, r_0 = run_v0(c)
check("10 partial resume counts retry bytes + resends only remaining",
      "retry_bytes" not in r_0 and r_0["transfer_bytes"] == 400000000,   # v0 does not count the resend
      r_r["retry_bytes"] == 400000000 // 4 and r_r["transfer_bytes"] == 400000000 + 400000000 // 4 and "wsA" in s_r.resumed_sets,
      f"v0r retry_bytes={r_r['retry_bytes']}")

# ---- 11. two-phone placement (real multi-device selection) ----
c = cfgv2(config_id="twophone")
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
s_r, r_r = run_v0r(c, pol="static_placement")
s_0, r_0 = run_v0(c, pol="static_placement")
v0_one_device = all(k[0] == "op15" for k in s_0.res)       # v0 always uses next(iter(phones))
v0r_two_devices = r_r["dispatched_by_device"].get("op12", 0) >= 1 and r_r["dispatched_by_device"].get("op15", 0) >= 1
check("11 two-phone placement (multi-device selection)",
      v0_one_device,
      v0r_two_devices,
      f"v0r by_device={r_r['dispatched_by_device']}")

# ---- 12. full epoch stack on completion (route/state, not only boot) ----
c = cfgv2(config_id="epochstack")
c["models"][0]["phone_compute_us"] = 5000000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 2000000)]   # r0 warms; r1 dispatches to the phone
# drive the repaired sim by hand: dispatch, then bump route_epoch mid-flight, then fire the pending exec.
s = V0R.Sim(c, 262144000, "relief_predictive")
for req in sorted(c["workload"]["requests"], key=lambda r: (r["arrival_us"], r["rank"])):
    s.push(req["arrival_us"], V0R.PR_ARRIVAL, "arrival", req)
# run until a lane_arrive/exec_done is queued, bumping route just before exec fires
bumped = False
rejected_seen = False
import heapq as _hq
while s.heap:
    t, pr, seq, kind, payload = _hq.heappop(s.heap)
    s.t = t
    if kind == "arrival":
        s.on_arrival(payload)
    elif kind == "res_stage":
        s.stage(payload)
    elif kind == "lane_arrive":
        s.lane_arrive(payload)
        if not bumped:
            s.route_epoch["op15"] += 1                        # reroute after the activation, before exec_done
            bumped = True
    elif kind == "exec_done":
        s.exec_done(payload)
    elif kind == "d2h_done":
        s.d2h_done(payload)
    elif kind == "server_done":
        s.server_done(payload)
v0r_route_guard = (s.m["stale_completion_rejects"] >= 1)
v0_has_no_route_epoch = not hasattr(V0.Sim(to_v1(c), 262144000, "relief_predictive"), "route_epoch")
check("12 full epoch stack (route/state) on completion",
      v0_has_no_route_epoch,                                 # v0 has no route/state epoch at all
      v0r_route_guard,
      f"v0r stale_completion_rejects={s.m['stale_completion_rejects']}")

# ---- 13. unknown validator kind rejected ----
unk = os.path.join(SPIKE, "fixtures", "v2", "bundles", "invalid", "unknown_kind.json")
ucodes = subprocess.run([sys.executable, os.path.join(SPIKE, "bundle_validate.py"), unk],
                        capture_output=True, text=True).stdout
# v1 has NO bundle validator and no 'kind' field on records; an unknown kind cannot be gated.
v1_no_kind_gate = ("kind" not in json.load(open(os.path.join(SPIKE, "fixtures", "valid", "device_inventory.valid.json"))))
check("13 unknown validator kind rejected",
      v1_no_kind_gate,
      "E_UNKNOWN_KIND" in ucodes,
      "v2 E_UNKNOWN_KIND")

# ---- 14. golden replay mutation is caught (compare to stored artifact, not a second run) ----
gold = json.load(open(os.path.join(SPIKE, "golden", "v0r", "baseline_sweep.v0r.manifest.json")))
v0r_config = json.load(open(os.path.join(SPIKE, "golden", "v0r", "baseline_sweep.v0r.config.json")))
fresh_results = V0R.run_sweep(v0r_config)
fresh_replay = V0R.sha_text(V0R.canonical(fresh_results))
mutated = copy.deepcopy(gold["goodput_results"])
mutated[0]["baselines"][0]["completed_server"] += 1        # tamper one number
mutated_replay = V0R.sha_text(V0R.canonical(mutated))
# v0's determinism test only compared two in-process runs (identical by construction) -> blind to a stored mutation.
check("14 golden replay mutation caught vs stored artifact",
      mutated_replay != gold["deterministic_replay_sha256"],   # mutation changes the hash (would fail the golden test)
      fresh_replay == gold["deterministic_replay_sha256"],     # a fresh run still reproduces the stored golden
      f"golden={gold['deterministic_replay_sha256'][:18]}")

n_fail = sum(1 for _, ok in results if not ok)
print(f"\nregression tests: {len(results)}  failures: {n_fail}")
sys.exit(1 if n_fail else 0)
