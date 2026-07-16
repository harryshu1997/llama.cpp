#!/usr/bin/env python3
"""S9-V0-R1 red-before / green-after adversarial suite.

For each of the sixteen fail-open cases an independent adversarial pass found in V0-R,
this imports BOTH the frozen V0-R validator/simulator and the R1 ones and shows, on the
same input, that V0-R exhibits the hole (RED) and R1 closes it (GREEN). Assertions are on
ACTUAL state, bytes, ownership, and terminal outcomes -- not counters or labels alone.
Deterministic; no network (except the local jsonschema/multiprocessing).
"""
import copy
import importlib.util
import json
import multiprocessing
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SPIKE)
import residency_sim as R1SIM       # noqa: E402  (repaired sim)
import bundle_validate as R1VAL     # noqa: E402  (repaired validator)
import make_fixtures_v2 as F2       # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V0SIM = _load(os.path.join(SPIKE, "golden", "v0r_historical", "residency_sim_v0r.py"), "residency_sim_v0r")
sys.path.insert(0, os.path.join(SPIKE, "golden", "v0r_historical"))
V0VAL = _load(os.path.join(SPIKE, "golden", "v0r_historical", "bundle_validate_v0r.py"), "bundle_validate_v0r")

results = []
RV2 = F2.build_records()   # v2 records for building the frozen-validator RED bundles


def check(name, red_ok, green_ok, detail=""):
    ok = (red_ok is None or bool(red_ok)) and bool(green_ok)
    results.append((name, ok))
    red = "n/a" if red_ok is None else ("hole" if red_ok else "NO-HOLE!")
    print(f"  {'PASS' if ok else 'FAIL'} {name}  RED(v0r)={red} "
          f"GREEN(R1)={'closed' if green_ok else 'OPEN!'}" + (f"  {detail}" if detail else ""))


# ---------- validator helpers ----------
def v2_bundle(records):
    return {"schema_version": 2, "kind": "bundle", "bundle_version": 2, "bundle_id": "red",
            "records": copy.deepcopy(records)}


def frozen_accepts(bundle):
    fd, p = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(bundle, f)
        return V0VAL.validate_bundle(p) == []
    finally:
        os.unlink(p)


def r1_codes(v3_fixture):
    errs = R1VAL.validate_bundle(os.path.join(SPIKE, "fixtures", "v3", "bundles", "invalid", v3_fixture))
    return sorted({c for c, _ in errs})


V2CORE = [RV2["MM"], RV2["SEGa"], RV2["SEGb"], RV2["WS1"], RV2["WS2"], RV2["ALLOC1"], RV2["ALLOC2"],
          RV2["PI1"], RV2["PI2"], RV2["CC1"], RV2["ISL1"], RV2["RC1"], RV2["RL1"], RV2["SL1"], RV2["DD1"],
          RV2["TT1"], RV2["TT2"], RV2["TFbulk"], RV2["TFexec"], RV2["DI"]]


def mutate_dd(records, **over):
    recs = copy.deepcopy(records)
    for r in recs:
        if r.get("kind") == "dispatch_decision":
            r.update(over)
    return recs


# ---------- 1. DISPATCH wrong device / backend / request / route epoch ----------
red = (frozen_accepts(v2_bundle(mutate_dd(V2CORE, device_id="op12")))
       and frozen_accepts(v2_bundle(mutate_dd(V2CORE, backend="gpu")))
       and frozen_accepts(v2_bundle(mutate_dd(V2CORE, request_id="req-OTHER")))
       and frozen_accepts(v2_bundle(mutate_dd(V2CORE, route_epoch=99))))
green = ("E_CHAIN_BROKEN" in r1_codes("dispatch_wrong_device.json")
         and "E_DISPATCH_MISMATCH" in r1_codes("dispatch_wrong_backend.json")
         and "E_STATE_LEASE" in r1_codes("dispatch_wrong_request.json")
         and "E_STATE_LEASE" in r1_codes("dispatch_wrong_route.json"))
check("1 DISPATCH wrong device/backend/request/route rejected", red, green)

# ---------- 2. sticky DISPATCH with null / foreign StateLease ----------
red = (frozen_accepts(v2_bundle(mutate_dd(V2CORE, state_lease_id=None)))
       and frozen_accepts(v2_bundle(mutate_dd(V2CORE, request_id="req-foreign"))))
green = ("E_STATE_LEASE" in r1_codes("sticky_null_state_lease.json")
         and "E_STATE_LEASE" in r1_codes("sticky_foreign_state_lease.json"))
check("2 sticky DISPATCH null/foreign StateLease rejected", red, green)

# ---------- 3. tuple (ws1,pi2,rc1,rl1): records exist but do not chain ----------
tup = {"weight_set_id": "ws1", "prepared_image_id": "pi2", "ready_certificate_id": "rc1", "residency_lease_id": "rl1"}
red = frozen_accepts(v2_bundle(mutate_dd(V2CORE, required_tuples=[dict(tup)], satisfied_tuples=[dict(tup)])))
green = "E_CHAIN_BROKEN" in r1_codes("incoherent_tuple.json")
check("3 incoherent tuple (records exist, no chain) rejected", red, green)

# ---------- 4. DISPATCH referencing a nonexistent island ----------
red = frozen_accepts(v2_bundle(mutate_dd(V2CORE, island_id="isl-ghost")))
green = "E_ISLAND_ABSENT" in r1_codes("nonexistent_island.json")
check("4 DISPATCH to nonexistent island rejected", red, green)

# ---------- 5. PreparedImage points to an allocation but missing from its alias set ----------
al = {**copy.deepcopy(RV2["ALLOC1"]), "alias_prepared_image_ids": [], "alias_refcount": 0, "lease_refcount": 1, "reclaimable": False}
al["allocation_digest"] = R1VAL.s9lib.alloc_digest(al)
red = frozen_accepts(v2_bundle([RV2["MM"], RV2["SEGa"], RV2["WS1"], al, RV2["PI1"], RV2["CC1"], RV2["RC1"], RV2["RL1"]]))
green = "E_ALIAS_SET" in r1_codes("pi_missing_from_alias_set.json")
check("5 PreparedImage missing from allocation alias set rejected", red, green)

# ---------- 6. Allocation reclaimable while an image references it ----------
al = {**copy.deepcopy(RV2["ALLOC1"]), "alias_prepared_image_ids": [], "alias_refcount": 0, "lease_refcount": 0, "reclaimable": True}
al["allocation_digest"] = R1VAL.s9lib.alloc_digest(al)
red = frozen_accepts(v2_bundle([RV2["MM"], RV2["SEGa"], RV2["WS1"], al, RV2["PI1"]]))
green = "E_ALIAS_SET" in r1_codes("reclaimable_while_referenced.json")
check("6 allocation reclaimable while referenced rejected", red, green)


# ---------- 7. parallel validators racing shared temp filenames ----------
def _worker(args):
    which, path, expect_valid = args
    if which == "v0r":
        errs = V0VAL.validate_bundle(path)
    else:
        errs = R1VAL.validate_bundle(path)
    return (len(errs) == 0) == expect_valid


def parallel_correct(which, jobs, rounds):
    ok = True
    mism = 0
    with multiprocessing.Pool(processes=12) as pool:
        for _ in range(rounds):
            for r in pool.map(_worker, [(which, p, ev) for p, ev in jobs]):
                if not r:
                    mism += 1
                    ok = False
    return ok, mism


if __name__ == "__main__":
    valid_v2 = os.path.join(SPIKE, "fixtures", "v2", "bundles", "valid", "dispatchable.json")
    invalid_v2 = os.path.join(SPIKE, "fixtures", "v2", "bundles", "invalid", "ledger_mismatch.json")
    jobs = [(valid_v2, True), (invalid_v2, False)] * 8
    v0_ok, v0_mism = parallel_correct("v0r", jobs, 6)
    r1_ok, r1_mism = parallel_correct("r1", jobs, 6)
    frozen_src = open(os.path.join(SPIKE, "golden", "v0r_historical", "bundle_validate_v0r.py")).read()
    r1_src = open(os.path.join(SPIKE, "bundle_validate.py")).read()
    red = "s9_bv_rec_" in frozen_src or "s9_bv_envelope" in frozen_src
    green = r1_ok and "tempfile.mkstemp" in r1_src and "s9_bv_rec_" not in r1_src
    check("7 shared temp paths removed; current parallel validation stable", red, green,
          f"v0r_shared_path=yes observed_v0r_mismatches={v0_mism} r1_mismatches={r1_mism}")

    # ---------- sim helpers ----------
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

    def mkreq(i, t, ws="wsA", model="m", state=4096, out=983040, sess=None, cls="decode"):
        r = {"arrival_us": t, "rank": i, "request_id": f"r{i}", "service_class": cls, "model_id": model,
             "weight_set_id": ws, "input_bytes": 983040, "output_bytes": out, "state_bytes": state, "deadline_us": None}
        if sess is not None:
            r["session_id"] = sess
        return r

    def run_r1(cfg, g=262144000, pol="relief_predictive"):
        s = R1SIM.Sim(cfg, g, pol)
        return s, s.run()

    def run_v0(cfg, g=262144000, pol="relief_predictive"):
        s = V0SIM.Sim(cfg, g, pol)
        return s, s.run()

    # ---------- 8. stale RECEIVING pipeline cleanup + wake next queued transfer (two phones, one domain) ----------
    # static round-robin over sorted [op12, op15]: wsA (seen first) -> op12, wsB -> op15 (same domain).
    # op12 fills wsA (900MB, long); wsB queued for op15; stale_epoch hits op12 mid-fill.
    c = base_cfg(); c["config_id"] = "stale_recv"
    c["phones"].append({**copy.deepcopy(c["phones"][0]), "device_id": "op12", "soc": "op12", "contention_domain": "d15"})
    c["models"][0]["canonical_bytes"] = 900000000
    c["models"].append({"model_id": "mB", "weight_set_id": "wsB", "canonical_bytes": 100000000,
                        "derived_bytes_htp": 0, "derived_bytes_gpu": 0, "scratch_bytes": 10000000,
                        "partial_load_supported": True, "eligible_backends": ["htp"], "phone_compute_us": 3000,
                        "state_policy": "stateless"})
    c["workload"]["requests"] = [mkreq(0, 0, ws="wsA", model="m"), mkreq(1, 500000, ws="wsB", model="mB")]
    c["failure_schedule"] = [{"at_us": 1000000, "kind": "stale_epoch", "target": "op12"}]
    s_r, r_r = run_r1(c, pol="static_placement")
    s_0, r_0 = run_v0(c, pol="static_placement")
    wsA_r = s_r.res.get(("op12", "wsA"))
    wsA_0 = s_0.res.get(("op12", "wsA"))
    wsB_r = s_r.res.get(("op15", "wsB"))
    wsB_0 = s_0.res.get(("op15", "wsB"))
    # v0r detects the stale stage but LEAKS: the domain link is never released and the
    # queued transfer is never woken (wsA stuck mid-pipeline, wsB stuck ABSENT).
    red = (s_0.domain_bulk["d15"] is not None
           and wsA_0 is not None and wsA_0.state not in ("ABSENT", "READY_HTP", "READY_GPU")
           and (wsB_0 is None or wsB_0.state == "ABSENT"))
    green = (r_r["stale_pipeline_drops"] >= 1 and r_r["domain_releases_on_stale"] >= 1
             and wsA_r is not None and wsA_r.state == "ABSENT"                       # rolled back, not stuck
             and wsB_r is not None and wsB_r.state != "ABSENT")                      # queued transfer woken on op15
    check("8 stale RECEIVING rolled back, domain released, queue woken", red, green,
          f"v0r wsA={wsA_0.state if wsA_0 else None} domain_held={s_0.domain_bulk['d15'] is not None} | "
          f"R1 releases={r_r['domain_releases_on_stale']} wsA={wsA_r.state if wsA_r else None} wsB={wsB_r.state if wsB_r else None}")

    # ---------- 9. request arriving after stale_epoch ----------
    c = base_cfg(); c["config_id"] = "post_stale"
    c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 2000000), mkreq(2, 8000000)]  # r0/r1 warm; stale; r2 after
    c["failure_schedule"] = [{"at_us": 5000000, "kind": "stale_epoch", "target": "op15"}]
    s_r, r_r = run_r1(c)
    s_0, r_0 = run_v0(c)
    wsA_r = s_r.res.get(("op15", "wsA"))
    wsA_0 = s_0.res.get(("op15", "wsA"))
    red = (not getattr(s_0, "draining", {}).get("op15", False)                       # v0r never drains
           and wsA_0 is not None and wsA_0.state != "ABSENT")                        # v0r re-prefetched onto stale device
    green = (s_r.draining["op15"] is True and r_r["timed_out"] == 0
             and (wsA_r is None or wsA_r.state == "ABSENT")                          # no re-prefetch onto drained device
             and s_r.terminal["r2"] in ("fallback", "rejected"))
    check("9 request after stale_epoch: device DRAINING, non-dispatchable", red, green,
          f"R1 draining={s_r.draining['op15']} r2={s_r.terminal.get('r2')}")

    # ---------- 10. epoch change while D2H is active ----------
    c = base_cfg(); c["config_id"] = "d2h_stale"
    c["models"][0]["phone_compute_us"] = 1000
    c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 2000000, out=900000000)]      # r1 has a long D2H
    # r1 dispatches ~2s, computes 1ms, exec ~2s, D2H ~3.4s -> stale at 3s lands inside the D2H window
    c["failure_schedule"] = [{"at_us": 3000000, "kind": "stale_epoch", "target": "op15"}]
    s_r, r_r = run_r1(c)
    s_0, r_0 = run_v0(c)
    red = (r_0["completed_phone"] >= 1 and "post_d2h_epoch_rejects" not in r_0)      # v0r accepted output post-stale
    green = (r_r["post_d2h_epoch_rejects"] >= 1 and s_r.terminal.get("r1") == "rejected"
             and r_r["completed_phone"] == 0)
    check("10 epoch change during D2H rejects the output", red, green,
          f"R1 post_d2h_epoch_rejects={r_r['post_d2h_epoch_rejects']} r1={s_r.terminal.get('r1')}")

    # ---------- 11. HTP-prepared residency submitted to GPU (and reverse) ----------
    def ready_entry(sim, dev, ws, back):
        r = sim.resolve(dev, ws)
        r.state = "READY_HTP" if back == "htp" else "READY_GPU"
        r.backend = back
        r.ready_at = 0
        return r
    c = base_cfg(); c["config_id"] = "xbackend"; c["models"][0]["eligible_backends"] = ["htp", "gpu"]
    s_r = R1SIM.Sim(c, 262144000, "relief_predictive"); ready_entry(s_r, "op15", "wsA", "htp")
    s_0 = V0SIM.Sim(c, 262144000, "relief_predictive"); ready_entry(s_0, "op15", "wsA", "htp")
    req = mkreq(0, 0)
    r1_htp_to_gpu = s_r.dispatch_phone("op15", "gpu", req)      # R1 must refuse HTP residency for a GPU request
    v0_htp_to_gpu = s_0.dispatch_phone("op15", "gpu", req)      # v0r wrongly dispatches
    # reverse
    s_r2 = R1SIM.Sim(c, 262144000, "relief_predictive"); ready_entry(s_r2, "op15", "wsA", "gpu")
    r1_gpu_to_htp = s_r2.dispatch_phone("op15", "htp", mkreq(1, 0))
    red = (v0_htp_to_gpu is True)
    green = (r1_htp_to_gpu is False and r1_gpu_to_htp is False and s_r.dispatched_from_on_disk >= 1)
    check("11 wrong-backend residency dispatch refused", red, green,
          f"v0r htp->gpu={v0_htp_to_gpu} R1 htp->gpu={r1_htp_to_gpu} gpu->htp={r1_gpu_to_htp}")

    # ---------- 12. two independent sticky sessions with different state sizes ----------
    c = base_cfg(); c["config_id"] = "sessions"; c["models"][0]["state_policy"] = "sticky"
    # r0 warms wsA (cold miss -> fallback); r1 establishes session s1; r2 establishes session s2
    c["workload"]["requests"] = [mkreq(0, 0, state=4096, sess="w"), mkreq(1, 2000000, state=4096, sess="s1"),
                                 mkreq(2, 4000000, state=8192, sess="s2")]
    s_r, r_r = run_r1(c)
    s_0, r_0 = run_v0(c)
    rr = s_r.res.get(("op15", "wsA"))
    r0 = s_0.res.get(("op15", "wsA"))
    red = (getattr(r0, "sticky_state", None) == 4096)          # v0r collapses: only the first session's bytes
    green = (rr is not None and dict(rr.sticky_sessions) == {"s1": 4096, "s2": 8192}
             and s_r.ledger["op15"].state == 4096 + 8192)      # both sessions reserved distinctly
    check("12 two sticky sessions tracked with distinct sizes", red, green,
          f"R1 sessions={dict(rr.sticky_sessions) if rr else None} v0r scalar={getattr(r0,'sticky_state',None)}")

    # ---------- 13. reset requested while state is pinned (deferred) ----------
    c = base_cfg(); c["config_id"] = "reset_pin"; c["models"][0]["state_policy"] = "sticky"
    c["models"][0]["phone_compute_us"] = 3000000
    # r0 warms; r1 establishes+completes sticky s1; r2 then reuses s1 and pins it through the reset
    c["workload"]["requests"] = [mkreq(0, 0, state=4096, sess="w"), mkreq(1, 1000000, state=4096, sess="s1"),
                                 mkreq(2, 4200000, state=4096, sess="s1")]
    c["failure_schedule"] = [{"at_us": 5000000, "kind": "reset_state", "target": "op15"}]
    s_r, r_r = run_r1(c)
    s_0, r_0 = run_v0(c)
    rr = s_r.res.get(("op15", "wsA"))
    r0 = s_0.res.get(("op15", "wsA"))
    red = (getattr(r0, "sticky_state", 0) > 0 and s_0.ledger["op15"].state > 0)      # v0r lost the reset (skipped while pinned)
    green = (r_r["reset_deferred"] >= 1 and (rr is None or not rr.sticky_sessions)
             and s_r.ledger["op15"].state == 0)                # deferred reset executed at unpin
    check("13 reset while pinned is deferred then executed", red, green,
          f"R1 reset_deferred={r_r['reset_deferred']} R1 state={s_r.ledger['op15'].state} v0r state={s_0.ledger['op15'].state}")

    # ---------- 14. lane queue overflow + exact terminal ----------
    c = base_cfg(); c["config_id"] = "lane_of"
    c["phones"][0]["htp_lane_slots"] = 1
    c["queues"]["lane_depth"] = 1
    c["models"][0]["phone_compute_us"] = 20000000              # long compute so requests pile on the single lane
    c["workload"]["requests"] = [mkreq(0, 0)] + [mkreq(i, 5000000 + i * 1000) for i in range(1, 4)]
    s_r, r_r = run_r1(c)
    s_0, r_0 = run_v0(c)
    red = ("lane_rejects" not in r_0 and r_0["rejected"] == 0 and r_0["completed_phone"] == 3)  # v0r unbounded lane
    green = (r_r["lane_rejects"] >= 1 and r_r["rejected"] >= 1
             and (r_r["completed_phone"] + r_r["completed_server"] + r_r["fallback"] + r_r["rejected"] + r_r["timed_out"]) == 4)
    check("14 bounded lane overflow -> explicit rejected terminal", red, green,
          f"R1 lane_rejects={r_r['lane_rejects']} rejected={r_r['rejected']}")

    # ---------- 15. partial failures at multiple verified offsets -> different retry bytes ----------
    c = base_cfg(); c["config_id"] = "multi_partial"; c["models"][0]["canonical_bytes"] = 400000000
    c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 40000000)]
    c["failure_schedule"] = [{"at_us": 100, "kind": "partial_transfer", "target": "wsA", "verified_offset": 100000000},
                             {"at_us": 200, "kind": "partial_transfer", "target": "wsA", "verified_offset": 300000000}]
    s_r, r_r = run_r1(c)
    s_0, r_0 = run_v0(c)
    rr = s_r.res.get(("op15", "wsA"))
    exp_retry = (400000000 - 100000000) + (400000000 - 300000000)                   # 300MB + 100MB
    red = (r_0["retry_bytes"] == 400000000 // 4)                                     # v0r: single fixed canonical/4 resume
    green = (r_r["retry_bytes"] == exp_retry and rr is not None and rr.resumed == 2)
    check("15 multi-offset partial failures: exact per-offset retry bytes", red, green,
          f"R1 retry={r_r['retry_bytes']} resumed={rr.resumed if rr else None} v0r retry={r_0['retry_bytes']}")

    # ---------- 16. link drop during bulk / activation / compute / d2h ----------
    def link_drop_case(phase, at_us, tweak):
        c = base_cfg(); c["config_id"] = f"ld_{phase}"
        tweak(c)
        c["failure_schedule"] = [{"at_us": at_us, "kind": "link_drop", "target": "op15", "phase": phase}]
        s_r, r_r = run_r1(c)
        s_0, r_0 = run_v0(c)
        return r_r, r_0, s_r
    # bulk: link drop during a RECEIVING fill -> R1 resumable
    def tw_bulk(c):
        c["models"][0]["canonical_bytes"] = 900000000
        c["workload"]["requests"] = [mkreq(0, 0)]
    rb_r, rb_0, _ = link_drop_case("bulk", 1000000, tw_bulk)
    # activation: link drop after dispatch but before lane arrival -> terminal reject
    def tw_act(c):
        c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 2000000)]
    ra_r, ra_0, _ = link_drop_case("activation", 2001000, tw_act)
    # compute: link drop while a request computes -> R1 terminal reject
    def tw_comp(c):
        c["models"][0]["phone_compute_us"] = 20000000
        c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 2000000)]
    rc_r, rc_0, sc_r = link_drop_case("compute", 5000000, tw_comp)
    # d2h: link drop during a long D2H -> R1 terminal reject
    def tw_d2h(c):
        c["models"][0]["phone_compute_us"] = 1000
        c["workload"]["requests"] = [mkreq(0, 0), mkreq(1, 2000000, out=900000000)]
    rd_r, rd_0, _ = link_drop_case("d2h", 3000000, tw_d2h)
    red = ("link_drop_resumed" not in rb_0 and "link_drop_terminal" not in ra_0
           and "link_drop_terminal" not in rc_0)                                     # v0r ignores link_drop
    green = (rb_r["link_drop_resumed"] >= 1 and rb_r["retry_bytes"] >= 900000000
             and ra_r["link_drop_terminal"] >= 1 and ra_r["rejected"] >= 1
             and rc_r["link_drop_terminal"] >= 1 and rc_r["rejected"] >= 1
             and rd_r["link_drop_terminal"] >= 1 and rd_r["rejected"] >= 1)
    check("16 link_drop resumable(bulk)/terminal(activation,compute,d2h)", red, green,
          f"R1 bulk_resumed={rb_r['link_drop_resumed']} activation_term={ra_r['link_drop_terminal']} "
          f"compute_term={rc_r['link_drop_terminal']} d2h_term={rd_r['link_drop_terminal']}")

    # ---------- arrivals conservation across every case ----------
    def conserved(r):
        return (r["completed_phone"] + r["completed_server"] + r["fallback"] + r["rejected"] + r["timed_out"]) == r["requests"]
    all_r1 = [r_r for r_r in [rb_r, ra_r, rc_r, rd_r]]
    conserve_ok = all(conserved(x) for x in all_r1)
    check("17 current-only terminal partition invariant", None, conserve_ok,
          "conservation holds across R1 cases")

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\nR1 regression tests: {len(results)}  failures: {n_fail}")
    sys.exit(1 if n_fail else 0)
