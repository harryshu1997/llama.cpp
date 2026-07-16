#!/usr/bin/env python3
"""S9-V0-R1 second-pass hole closure (adversarial verification findings).

An independent multi-agent pass found ten fail-open holes that survived the first R1
round. A later review added two validator closures and six simulator regression checks.

  Validator holes (bundle_validate.py cross_record_r1) -- each shows the check was genuinely
  new: the frozen pre-R1 validator ACCEPTS the equivalent v2 bundle, and R1 REJECTS the v3
  fixture with a specific stable code.
  Simulator holes (sim/residency_sim.py) -- each executes the current implementation; where
  no historical snapshot exists, an exact regression mutant proves that the oracle turns red.

Deterministic; no network beyond local jsonschema.
"""
import copy
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SPIKE)
import residency_sim as R1SIM       # noqa: E402
import bundle_validate as R1VAL     # noqa: E402
import make_fixtures_v2 as F2       # noqa: E402
sys.path.insert(0, os.path.join(SPIKE, "golden", "v0r_historical"))
_spec = importlib.util.spec_from_file_location(
    "bundle_validate_v0r", os.path.join(SPIKE, "golden", "v0r_historical", "bundle_validate_v0r.py"))
V0VAL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V0VAL)

results = []
RV2 = F2.build_records()


def check(name, red_ok, green_ok, detail=""):
    ok = (red_ok is None or bool(red_ok)) and bool(green_ok)
    results.append((name, ok))
    red = "n/a" if red_ok is None else ("hole" if red_ok else "NO-HOLE!")
    print(f"  {'PASS' if ok else 'FAIL'} {name}  RED(reference/mutant)={red} "
          f"GREEN(R1)={'closed' if green_ok else 'OPEN!'}" + (f"  {detail}" if detail else ""))


def frozen_accepts_v2(records):
    b = {"schema_version": 2, "kind": "bundle", "bundle_version": 2, "bundle_id": "red",
         "records": copy.deepcopy(records)}
    fd, p = tempfile.mkstemp(prefix="s9_hole_red_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(b, f)
        return V0VAL.validate_bundle(p) == []
    finally:
        os.unlink(p)


def r1_codes(fixture):
    errs = R1VAL.validate_bundle(os.path.join(SPIKE, "fixtures", "v3", "bundles", "invalid", fixture))
    return sorted({c for c, _ in errs})


V2CORE = [RV2["MM"], RV2["SEGa"], RV2["SEGb"], RV2["WS1"], RV2["WS2"], RV2["ALLOC1"], RV2["ALLOC2"],
          RV2["PI1"], RV2["PI2"], RV2["CC1"], RV2["ISL1"], RV2["RC1"], RV2["RL1"], RV2["SL1"], RV2["DD1"],
          RV2["TT1"], RV2["TT2"], RV2["TFbulk"], RV2["TFexec"], RV2["DI"]]


# ---------- validator hole H1: island required_weight_set_digests coverage ----------
red = copy.deepcopy(V2CORE)
for r in red:
    if r.get("kind") == "island_executable":
        r["required_weight_set_digests"] = [F2.h("ghost")]
        r["island_digest"] = R1VAL.s9lib.island_digest(r)
check("H1 island weight-set-digest coverage", frozen_accepts_v2(red),
      "E_DISPATCH_MISMATCH" in r1_codes("island_ws_digest_ghost.json"))

# ---------- H2: island required_prepared_image coverage ----------
red = copy.deepcopy(V2CORE)
for r in red:
    if r.get("kind") == "island_executable":
        r["required_prepared_image_ids"] = ["pi-ghost"]
        r["required_prepared_image_digests"] = [F2.h("ghost")]
        r["island_digest"] = R1VAL.s9lib.island_digest(r)
check("H2 island prepared-image coverage", frozen_accepts_v2(red),
      "E_DISPATCH_MISMATCH" in r1_codes("island_pi_ghost.json"))

# ---------- H3: residency lease weight-set vs source-allocation mismatch ----------
red = [RV2["MM"], RV2["SEGa"], RV2["SEGb"], RV2["WS1"], RV2["WS2"],
       {**copy.deepcopy(RV2["ALLOC1"]), "lease_refcount": 2, "allocation_digest": RV2["ALLOC1"]["allocation_digest"]},
       RV2["ALLOC2"], RV2["PI1"], RV2["PI2"], RV2["CC1"], RV2["CC2"], RV2["ISL1"], RV2["RC1"], RV2["RC2"],
       RV2["RL1"], {**copy.deepcopy(RV2["RL2"]), "source_allocation_id": "alloc-ws1"}]
# fix the two mutated records' digests so the frozen (v2) validator only sees the semantic gap
red[5]["allocation_digest"] = R1VAL.s9lib.alloc_digest(red[5])
red[-1]["residency_lease_digest"] = R1VAL.s9lib.residency_lease_digest(red[-1])
check("H3 lease weight-set != source allocation", frozen_accepts_v2(red),
      "E_CHAIN_BROKEN" in r1_codes("lease_alloc_ws_mismatch.json"))

# ---------- H4: StateLease stale boot generation ----------
red = copy.deepcopy(V2CORE)
for r in red:
    if r.get("kind") == "state_lease":
        r["boot_epoch"] = 999
        r["state_lease_digest"] = R1VAL.s9lib.state_lease_digest(r)
check("H4 state-lease stale boot", frozen_accepts_v2(red),
      "E_STATE_LEASE" in r1_codes("state_lease_stale_boot.json"))

# ---------- H5: DISPATCH against a DRAINING residency lease ----------
red = copy.deepcopy(V2CORE)
for r in red:
    if r.get("kind") == "residency_lease":
        r["state"] = "DRAINING"
        r["residency_lease_digest"] = R1VAL.s9lib.residency_lease_digest(r)
check("H5 DRAINING lease not dispatchable", frozen_accepts_v2(red),
      "E_CHAIN_BROKEN" in r1_codes("draining_lease.json"))

# ---------- H6: StateLease dependency must be one of the dispatched residency leases ----------
red = [RV2["MM"], RV2["SEGa"], RV2["SEGb"], RV2["WS1"], RV2["WS2"], RV2["ALLOC1"], RV2["ALLOC2"],
       RV2["PI1"], RV2["PI2"], RV2["CC1"], RV2["CC2"], RV2["ISL1"], RV2["RC1"], RV2["RC2"],
       RV2["RL1"], RV2["RL2"], copy.deepcopy(RV2["SL1"]), RV2["DD1"]]
red[-2]["depends_on_residency_lease_id"] = "rl2"
red[-2]["depends_on_residency_generation"] = RV2["RL2"]["residency_generation"]
red[-2]["boot_epoch"] = RV2["RL2"]["boot_epoch"]
red[-2]["state_lease_digest"] = R1VAL.s9lib.state_lease_digest(red[-2])
check("H6 state lease depends on dispatched residency", frozen_accepts_v2(red),
      "E_STATE_LEASE" in r1_codes("state_lease_foreign_residency.json"))

# ---------- H7: ReadyCertificate correctness must be for the dispatched island ----------
rc = copy.deepcopy(RV2["RC1"])
rc["correctness_id"] = RV2["CC2"]["correctness_id"]
rc["correctness_digest"] = RV2["CC2"]["correctness_digest"]
rc["correctness"]["metric_digest"] = RV2["CC2"]["metric_digest"]
rc["ready_certificate_digest"] = R1VAL.s9lib.ready_cert_digest(rc)
rl = copy.deepcopy(RV2["RL1"])
rl["ready_certificate_digest"] = rc["ready_certificate_digest"]
rl["residency_lease_digest"] = R1VAL.s9lib.residency_lease_digest(rl)
red = [RV2["MM"], RV2["SEGa"], RV2["SEGb"], RV2["WS1"], RV2["WS2"], RV2["ALLOC1"], RV2["ALLOC2"],
       RV2["PI1"], RV2["PI2"], RV2["CC1"], RV2["CC2"], RV2["ISL1"], rc, RV2["RC2"], rl, RV2["RL2"],
       RV2["SL1"], RV2["DD1"]]
check("H7 ready correctness belongs to dispatched island", frozen_accepts_v2(red),
      "E_CHAIN_BROKEN" in r1_codes("ready_cert_foreign_correctness.json"))


# ---------- simulator holes: repro against the LIVE R1 sim + invariants ----------
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
                    "prepare_gpu_bytes_per_s": 3000000000, "warmup_us": 50000, "htp_lane_slots": 1,
                    "gpu_lane_slots": 1, "thermal_eligible": True}],
        "models": [{"model_id": "m", "weight_set_id": "wsA", "canonical_bytes": 100000000,
                    "derived_bytes_htp": 0, "derived_bytes_gpu": 0, "scratch_bytes": 10000000,
                    "partial_load_supported": True, "eligible_backends": ["htp"], "phone_compute_us": 3000,
                    "state_policy": "stateless"}],
        "workload": {"requests": []}, "failure_schedule": [],
        "policy_params": {"reuse_horizon_us": 600000000, "min_hold_us": 30000000, "ttl_us": 45000000,
                          "predictor_ewma_permille": 500},
    }


def mkreq(i, t, out=983040):
    return {"arrival_us": t, "rank": i, "request_id": f"r{i}", "service_class": "decode", "model_id": "m",
            "weight_set_id": "wsA", "input_bytes": 983040, "output_bytes": out, "state_bytes": 4096, "deadline_us": None}


# ---------- H8: lane double-release on compute-phase link_drop ----------
class DoubleReleaseMutant(R1SIM.Sim):
    def exec_done(self, p):
        if p["req"]["request_id"] in self.terminal:
            self._release_lane(p["dev"], p["back"])
            return
        super().exec_done(p)


c = base_cfg(); c["config_id"] = "h8"; c["phones"][0]["htp_lane_slots"] = 1
c["models"][0]["phone_compute_us"] = 20000000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 5000000)]
c["failure_schedule"] = [{"at_us": 6000000, "kind": "link_drop", "target": "op15", "phase": "compute"}]
try:
    DoubleReleaseMutant(c, 262144000, "relief_predictive").run()
    red = False
except AssertionError:
    red = True
s = R1SIM.Sim(c, 262144000, "relief_predictive"); r = s.run()
green = (s.htp_free["op15"] == 1 and r["link_drop_terminal"] == 1)
check("H8 terminal exec event cannot double-release lane", red, green,
      f"htp_free={s.htp_free['op15']} link_drop_terminal={r['link_drop_terminal']}")

class FoldQueuedMutant(R1SIM.Sim):
    def _link_drop(self, dev, phase):
        if phase == "bulk":
            return super()._link_drop(dev, phase)
        for rid, rr in list(self.req_res.items()):
            if rr["dev"] == dev and rr.get("phase") == phase and rid not in self.terminal:
                self._reject_inflight(self.req_index[rid], "link_drop_terminal")


c = base_cfg(); c["config_id"] = "h7"; c["phones"][0]["htp_lane_slots"] = 1; c["queues"]["lane_depth"] = 8
c["models"][0]["phone_compute_us"] = 20000000
c["workload"]["requests"] = [mkreq(0, 0)] + [mkreq(i, 5000000 + i * 1000) for i in range(1, 5)]
c["failure_schedule"] = [{"at_us": 6000000, "kind": "link_drop", "target": "op15", "phase": "compute"}]
s_bad = FoldQueuedMutant(c, 262144000, "relief_predictive"); r_bad = s_bad.run()
s = R1SIM.Sim(c, 262144000, "relief_predictive"); r = s.run()
# only the single compute-phase holder (r1) is rejected by the drop; queued r2..r4 are not folded in
rejects = [rid for rid, o in s.terminal.items() if o == "rejected"]
red = r_bad["link_drop_terminal"] > 1
green = (r["link_drop_terminal"] == 1 and rejects == ["r1"]
         and (r["completed_phone"] + r["completed_server"] + r["fallback"] + r["rejected"] + r["timed_out"]) == 5)
check("H9 compute link_drop does not fold queued requests", red, green,
      f"link_drop_terminal={r['link_drop_terminal']} rejects={rejects}")

class SupersededRollbackMutant(R1SIM.Sim):
    def stage(self, p):
        r = self.resolve(p["dev"], p["ws_id"])
        if p["ver"] != r.pipeline_ver or p["gen"] != r.generation or p["boot"] != self.boot_epoch[p["dev"]]:
            self.clean_stale_pipeline(p["dev"], p["ws_id"], r)
            return
        super().stage(p)


c = base_cfg(); c["config_id"] = "h8"; c["models"][0]["canonical_bytes"] = 900000000
c["workload"]["requests"] = [mkreq(0, 0)]
c["failure_schedule"] = [{"at_us": 1000000, "kind": "link_drop", "target": "op15", "phase": "bulk"}]
try:
    SupersededRollbackMutant(c, 262144000, "relief_predictive").run()
    red = False
except AssertionError:
    red = True
s = R1SIM.Sim(c, 262144000, "relief_predictive"); r = s.run()
wsA = s.res.get(("op15", "wsA"))
# the resumed set reaches READY and STILL holds its UFS reservation (no leak)
green = (wsA is not None and wsA.state == "READY_HTP" and wsA.res_ufs == 900000000
         and s.ledger["op15"].ufs_used == 900000000 and r["link_drop_resumed"] >= 1)
check("H10 link_drop bulk resume keeps UFS reservation", red, green,
      f"state={wsA.state if wsA else None} res_ufs={wsA.res_ufs if wsA else None} ufs_used={s.ledger['op15'].ufs_used}")

# ---------- H11: stale D2H rejection executes deferred residency eviction ----------
c = base_cfg(); c["config_id"] = "h11"; c["models"][0]["phone_compute_us"] = 1000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 2000000, out=900000000)]
c["failure_schedule"] = [{"at_us": 3000000, "kind": "stale_epoch", "target": "op15"}]
s = R1SIM.Sim(c, 262144000, "relief_predictive"); r = s.run(); rr = s.res.get(("op15", "wsA"))
green = (s.terminal.get("r1") == "rejected" and rr is not None and rr.state == "ABSENT"
         and rr.pinned == 0 and s.ledger["op15"].canonical == 0 and s.ledger["op15"].ufs_used == 0)
check("H11 stale D2H rejection drains deferred residency", None, green,
      f"state={rr.state if rr else None} pinned={rr.pinned if rr else None}")

# ---------- H12: a phone-scoped bulk fault cannot affect a sibling in the same domain ----------
c = base_cfg(); c["config_id"] = "h12"
c["phones"].append({**copy.deepcopy(c["phones"][0]), "device_id": "op12", "soc": "op12"})
s = R1SIM.Sim(c, 262144000, "relief_predictive")
s.start_prefetch("op12", "wsA"); owner_before = dict(s.domain_bulk["d15"])
s._link_drop("op15", "bulk"); owner_after = dict(s.domain_bulk["d15"])
rr = s.res[("op12", "wsA")]
green = (owner_after == owner_before and rr.resumed == 0 and s.m["link_drop_resumed"] == 0)
check("H12 bulk link_drop is scoped to target phone", None, green,
      f"owner={owner_after['dev']} resumed={rr.resumed}")

# ---------- H13: stale cleanup cannot release a successor domain owner ----------
c = base_cfg(); c["config_id"] = "h13"
c["phones"].append({**copy.deepcopy(c["phones"][0]), "device_id": "op12", "soc": "op12"})
s = R1SIM.Sim(c, 262144000, "relief_predictive")
s.start_prefetch("op12", "wsA"); s.start_prefetch("op15", "wsA")
assert s.link_free("d15", ("op12", "wsA"))
rr = s.res[("op12", "wsA")]; s.clean_stale_pipeline("op12", "wsA", rr)
green = (s.domain_bulk["d15"] is not None and s.domain_bulk["d15"]["dev"] == "op15")
check("H13 stale cleanup preserves successor transfer owner", None, green,
      f"owner={s.domain_bulk['d15']['dev'] if s.domain_bulk['d15'] else None}")

# ---------- H14: horizon timeout releases transient phone resources ----------
c = base_cfg(); c["config_id"] = "h14"; c["horizon_us"] = 2000000
c["phones"][0]["htp_lane_slots"] = 1; c["models"][0]["phone_compute_us"] = 20000000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 1800000)]
s = R1SIM.Sim(c, 100000000, "relief_predictive"); r = s.run(); rr = s.res.get(("op15", "wsA"))
green = (s.terminal.get("r1") == "timed_out" and s.htp_free["op15"] == 1
         and s.ledger["op15"].activations == 0 and s.ledger["op15"].state == 0
         and rr is not None and rr.pinned == 0)
check("H14 horizon timeout releases lane/activation/state/pin", None, green,
      f"free={s.htp_free['op15']} act={s.ledger['op15'].activations} pin={rr.pinned if rr else None}")

# ---------- H15: one sticky session has at most one in-flight mutation ----------
c = base_cfg(); c["config_id"] = "h15"; c["phones"][0]["htp_lane_slots"] = 2
c["models"][0]["state_policy"] = "sticky"; c["models"][0]["phone_compute_us"] = 2000000
r0 = mkreq(0, 0); r1 = mkreq(1, 2000000); r2 = mkreq(2, 2000000)
r1["session_id"] = r2["session_id"] = "same"
c["workload"]["requests"] = [r0, r1, r2]
s = R1SIM.Sim(c, 262144000, "relief_predictive"); r = s.run(); rr = s.res.get(("op15", "wsA"))
green = (rr is not None and rr.sticky_sessions == {"same": 4096}
         and s.ledger["op15"].state == 4096 and not s.session_inflight)
check("H15 sticky session mutations are single-owner", None, green,
      f"sessions={rr.sticky_sessions if rr else None} phone={r['completed_phone']} fallback={r['fallback']}")

# ---------- H16: server work is charged only when admitted ----------
c = base_cfg(); c["config_id"] = "h16"; c["queues"]["server_depth"] = 0
c["server"]["gpu_lane_slots"] = 1; c["server"]["per_class_gpu_us"]["decode"] = 10000
c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 0)]
s = R1SIM.Sim(c, 262144000, "server_only"); r = s.run()
green = (r["completed_server"] == 1 and r["rejected"] == 1 and r["server_gpu_us_used"] == 10000)
check("H16 rejected server work is not charged as GPU execution", None, green,
      f"used={r['server_gpu_us_used']} completed={r['completed_server']} rejected={r['rejected']}")

n_fail = sum(1 for _, ok in results if not ok)
print(f"\nR1 hole-closure tests: {len(results)}  failures: {n_fail}")
sys.exit(1 if n_fail else 0)
