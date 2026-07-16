#!/usr/bin/env python3
"""S9-V0-R2 v4 static-coherence regressions (red-v3 / green-v4).

Two families of evidence, mirroring the R1 methodology:

  A. red-v3 / green-v4, one per REVIEW_2026-07-14.md blocker: the FROZEN pre-R2 validator
     (golden/v0r1_historical/bundle_validate_v0r1.py) ACCEPTS the equivalent v3 bundle
     (fail-open), while the LIVE v4 validator REJECTS the v4 fixture with the exact stable
     code recorded in fixtures/v4/bundles/index.json. Where the incoherence is a v4-only
     field (device_status_ref / decision_ts_us / device_id), the v3 side simply lacks the
     gate, so a fully-v3-coherent bundle is accepted -- that IS the fail-open.

  B. digest-field effectiveness: each of the SIX fields the v4 digest family newly binds is
     shown effective -- changing it CHANGES the v4 digest and does NOT change the v3 digest
     (proving v3 left it unbound and v4 now covers it).

Deterministic; the only external process is the pinned local jsonschema (via the validators).
"""
import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)
sys.path.insert(0, SPIKE)
import s9lib                       # noqa: E402
import bundle_validate as V4VAL    # noqa: E402
import make_fixtures_v2 as F2      # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bundle_validate_v0r1", os.path.join(SPIKE, "golden", "v0r1_historical", "bundle_validate_v0r1.py"))
sys.path.insert(0, os.path.join(SPIKE, "golden", "v0r1_historical"))
V3VAL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V3VAL)

results = []


def check(name, red_ok, green_ok, detail=""):
    ok = bool(red_ok) and bool(green_ok)
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name}  RED(frozen-v3)={'accepts' if red_ok else 'REJECTS!'} "
          f"GREEN(v4)={'closed' if green_ok else 'OPEN!'}" + (f"  {detail}" if detail else ""))


# ---------- v3 red-side helpers (v2 records bumped to schema_version 3) ----------
def refresh_chain_v3(R):
    """Recompute every digest + cross-referenced digest field with the v2/v3 family (identical
    to the frozen contract). Same topology as make_fixtures_v4.refresh_chain_v4."""
    D = s9lib.DIGEST_BUILDERS
    for k in ("ALLOC1", "ALLOC2"):
        if k in R:
            R[k]["allocation_digest"] = D["canonical_allocation"][1](R[k])
    for k in ("PI1", "PI2", "PIgpu"):
        if k in R:
            R[k]["derived_image_digest"] = D["prepared_image"][1](R[k])
    for k in ("CC1", "CC2"):
        if k in R:
            R[k]["correctness_digest"] = D["correctness_certificate"][1](R[k])
    if "ISL1" in R:
        R["ISL1"]["required_prepared_image_digests"] = [R["PI1"]["derived_image_digest"]]
        R["ISL1"]["correctness_digest"] = R["CC1"]["correctness_digest"]
        R["ISL1"]["island_digest"] = D["island_executable"][1](R["ISL1"])
    if "ISL2" in R:
        R["ISL2"]["required_prepared_image_digests"] = [R["PI1"]["derived_image_digest"], R["PI2"]["derived_image_digest"]]
        R["ISL2"]["correctness_digest"] = R["CC2"]["correctness_digest"]
        R["ISL2"]["island_digest"] = D["island_executable"][1](R["ISL2"])
    for rck, cck, pik in (("RC1", "CC1", "PI1"), ("RC2", "CC2", "PI2")):
        if rck in R:
            R[rck]["prepared_image_digest"] = R[pik]["derived_image_digest"]
            R[rck]["correctness_digest"] = R[cck]["correctness_digest"]
            R[rck]["ready_certificate_digest"] = D["ready_certificate"][1](R[rck])
    for rlk, rck in (("RL1", "RC1"), ("RL2", "RC2")):
        if rlk in R:
            R[rlk]["ready_certificate_digest"] = R[rck]["ready_certificate_digest"]
            R[rlk]["residency_lease_digest"] = D["residency_lease"][1](R[rlk])
    if "SL1" in R:
        R["SL1"]["state_lease_digest"] = D["state_lease"][1](R["SL1"])
    for k in ("TT1", "TT2"):
        if k in R:
            R[k]["transfer_ticket_digest"] = D["transfer_ticket"][1](R[k])
    return R


V3_CORE_KEYS = ["MM", "SEGa", "WS1", "ALLOC1", "PI1", "CC1", "ISL1", "RC1", "RL1", "SL1",
                "DD1", "TT1", "TT2", "TFbulk", "TFexec", "DI"]


def v3_records():
    """v2 records with the toy-ledger state reservations + coherent DI (v3-shaped: no v4 fields)."""
    R = copy.deepcopy(F2.build_records())
    for r in R.values():
        r["schema_version"] = 3
    R["SL1"]["reserved_state_bytes"] = 60
    R["SL1"]["reserved_activation_bytes"] = 40
    R["DI"]["physical_byte_accounting"] = {"weights_resident": 400, "derived_images": 0,
        "scratch": 100, "activations": 40, "mutable_state": 60, "free": 9400}
    refresh_chain_v3(R)
    return R


def frozen_v3_accepts(records):
    b = {"schema_version": 3, "kind": "bundle", "bundle_version": 3, "bundle_id": "v3-red",
         "records": [copy.deepcopy(r) for r in records]}
    import tempfile
    fd, p = tempfile.mkstemp(prefix="s9_v4red_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(b, f)
        return V3VAL.validate_bundle(p) == []
    finally:
        os.unlink(p)


def v4_codes(fixture):
    errs = V4VAL.validate_bundle(os.path.join(SPIKE, "fixtures", "v4", "bundles", "invalid", fixture))
    return sorted({c for c, _ in errs})


# expected v4 code sets (must match fixtures/v4/bundles/index.json exactly)
IDX = {r["file"]: r.get("expected_codes", [])
       for r in json.load(open(os.path.join(SPIKE, "fixtures", "v4", "bundles", "index.json")))}


def core(R):
    return [R[k] for k in V3_CORE_KEYS]


# ================= A. red-v3 / green-v4, one per review blocker =================

# blocker 1a: DeviceInventory absent
R = v3_records()
red = frozen_v3_accepts([R[k] for k in V3_CORE_KEYS if k != "DI"])
check("1a device absent", red, v4_codes("device_absent.json") == IDX["invalid/device_absent.json"])

# blocker 1b: wrong boot epoch pinned (v4-only device_status_ref; v3 lacks the gate)
R = v3_records()
red = frozen_v3_accepts(core(R))
check("1b device wrong boot", red, v4_codes("device_boot_mismatch.json") == IDX["invalid/device_boot_mismatch.json"])

# blocker 1c: wrong status_seq pinned (v4-only; v3 lacks the gate)
R = v3_records()
red = frozen_v3_accepts(core(R))
check("1c device wrong status_seq", red, v4_codes("device_status_seq_mismatch.json") == IDX["invalid/device_status_seq_mismatch.json"])

# blocker 2a: draining
R = v3_records(); R["DI"]["draining"] = True
check("2a device draining", frozen_v3_accepts(core(R)),
      v4_codes("device_draining.json") == IDX["invalid/device_draining.json"])

# blocker 2b: thermal ineligible
R = v3_records(); R["DI"]["thermal"]["eligible"] = False
check("2b device thermal ineligible", frozen_v3_accepts(core(R)),
      v4_codes("device_thermal_ineligible.json") == IDX["invalid/device_thermal_ineligible.json"])

# blocker 2c: backend unsupported
R = v3_records(); R["DI"]["backends"] = ["gpu", "cpu"]
check("2c device backend unsupported", frozen_v3_accepts(core(R)),
      v4_codes("device_backend_unsupported.json") == IDX["invalid/device_backend_unsupported.json"])

# blocker 3: lease expired (v4-only decision_ts_us; v3 lacks the gate)
R = v3_records()
check("3 residency lease expired", frozen_v3_accepts(core(R)),
      v4_codes("lease_expired.json") == IDX["invalid/lease_expired.json"])

# blocker 4: correctness backend_build != prepared image backend_build
R = v3_records(); R["CC1"]["backend_build"] = F2.BUILD_GPU; refresh_chain_v3(R)
check("4 backend_build mismatch", frozen_v3_accepts(core(R)),
      v4_codes("backend_build_mismatch.json") == IDX["invalid/backend_build_mismatch.json"])

# blocker 5: island graph identity mismatch
R = v3_records(); R["ISL1"]["graph_hash"] = F2.h("graph:other-model"); refresh_chain_v3(R)
check("5 island graph identity mismatch", frozen_v3_accepts(core(R)),
      v4_codes("island_graph_mismatch.json") == IDX["invalid/island_graph_mismatch.json"])

# blocker 6: prepared image source_weight_set_id foreign
R = v3_records(); R["PI1"]["source_weight_set_id"] = "ws2"; refresh_chain_v3(R)
check("6 prepared-image source foreign", frozen_v3_accepts(core(R)),
      v4_codes("pi_source_foreign.json") == IDX["invalid/pi_source_foreign.json"])

# blocker 7a: bulk frame residency epoch foreign (v3 has no frame-epoch check)
R = v3_records(); R["TFbulk"]["residency_epoch"] = 99
check("7a bulk frame stale epoch", frozen_v3_accepts(core(R)),
      v4_codes("bulk_frame_stale_epoch.json") == IDX["invalid/bulk_frame_stale_epoch.json"])

# blocker 7b: EXECUTE frame route epoch foreign
R = v3_records(); R["TFexec"]["route_epoch"] = 99
check("7b exec frame stale route", frozen_v3_accepts(core(R)),
      v4_codes("exec_frame_stale_route.json") == IDX["invalid/exec_frame_stale_route.json"])

# blocker 8: ticket issued_boot_epoch now bound in the v4 digest (v3 digest ignores it)
R = v3_records(); R["TT1"]["issued_boot_epoch"] = 8; R["TFbulk"]["boot_epoch"] = 8
R["TT1"]["transfer_ticket_digest"] = s9lib.transfer_ticket_digest(R["TT1"])   # v3 digest excludes issued
check("8 ticket issued_boot bound (v4) / ignored (v3)", frozen_v3_accepts(core(R)),
      v4_codes("ticket_issued_boot_unbound.json") == IDX["invalid/ticket_issued_boot_unbound.json"])

# blocker 9: state lease in_flight_mutation_seq now bound in the v4 digest
R = v3_records(); R["SL1"]["in_flight_mutation_seq"] = 5
R["SL1"]["state_lease_digest"] = s9lib.state_lease_digest(R["SL1"])           # v3 digest excludes it
check("9 state mutation_seq bound (v4) / ignored (v3)", frozen_v3_accepts(core(R)),
      v4_codes("state_mutation_seq_unbound.json") == IDX["invalid/state_mutation_seq_unbound.json"])

# blocker 10: state reservation exceeds the device ledger
R = v3_records(); R["SL1"]["reserved_activation_bytes"] = 20000; refresh_chain_v3(R)
check("10 state reservation exceeds ledger", frozen_v3_accepts(core(R)),
      v4_codes("state_reservation_exceeds_ledger.json") == IDX["invalid/state_reservation_exceeds_ledger.json"])

# blocker 11: DeviceInventory declares a zero-live ledger while records are live
R = v3_records()
R["DI"]["physical_byte_accounting"] = {"weights_resident": 0, "derived_images": 0,
    "scratch": 0, "activations": 0, "mutable_state": 0, "free": 10000}
check("11 zero-live ledger vs live records", frozen_v3_accepts(core(R)),
      v4_codes("ledger_zero_live.json") == IDX["invalid/ledger_zero_live.json"])


# ===== A2. hole-hunt closures (round 2): red-v3 / green-v4 =====

# RESULT frame is a live-payload frame too (v3 has no frame-epoch gate)
R = v3_records()
res = copy.deepcopy(R["TFexec"]); res["msg_type"] = "RESULT"; res["priority_class"] = "result"
res["route_epoch"] = 99; res["idempotency_key"] = "idem-result-1"; res["seq"] = 200
check("HH RESULT frame stale route", frozen_v3_accepts(core(R) + [res]),
      v4_codes("result_frame_stale.json") == IDX["invalid/result_frame_stale.json"])

# island model identity not anchored to a present manifest (v3 does not require one)
R = v3_records()
check("HH island without manifest", frozen_v3_accepts([R[k] for k in V3_CORE_KEYS if k != "MM"]),
      v4_codes("island_no_manifest.json") == IDX["invalid/island_no_manifest.json"])

# live bytes on a device with NO DeviceInventory (v3 has no record-derived ledger)
R = v3_records()
al = copy.deepcopy(R["ALLOC1"]); al["allocation_id"] = "alloc-op12"; al["device_id"] = "op12"
al["canonical_bytes"] = 9000; al["alias_prepared_image_ids"] = []; al["alias_refcount"] = 0
al["lease_refcount"] = 0; al["reclaimable"] = True
al["allocation_digest"] = s9lib.DIGEST_BUILDERS["canonical_allocation"][1](al)
check("HH DI-less device bytes", frozen_v3_accepts(core(R) + [al]),
      v4_codes("ledger_diless_device.json") == IDX["invalid/ledger_diless_device.json"])

# ReadyCertificate attests a footprint larger than the device LPDDR (v3 checks only total==sum)
R = v3_records()
R["RC1"]["physical_bytes"] = {"canonical": 400, "derived": 0, "scratch": 100,
    "activations_reserved": 40, "state_reserved": 500000, "total": 500540}
refresh_chain_v3(R)
check("HH RC footprint > lpddr", frozen_v3_accepts(core(R)),
      v4_codes("rc_footprint_exceeds_lpddr.json") == IDX["invalid/rc_footprint_exceeds_lpddr.json"])

# ResidencyLease weight reservation != its single-copy allocation
R = v3_records(); R["RL1"]["reserved_bytes"]["weights"] = 9999; refresh_chain_v3(R)
check("HH RL weights != allocation", frozen_v3_accepts(core(R)),
      v4_codes("rl_weights_mismatch.json") == IDX["invalid/rl_weights_mismatch.json"])

# arch identity (v3 does not cross-check arch)
R = v3_records(); R["PI1"]["arch"] = "llama3"; refresh_chain_v3(R)
check("HH prepared-image arch foreign", frozen_v3_accepts(core(R)),
      v4_codes("pi_arch_foreign.json") == IDX["invalid/pi_arch_foreign.json"])

# layout_version identity (v3 does not cross-check layout_version)
R = v3_records(); R["PI1"]["layout_version"] = 4; refresh_chain_v3(R)
check("HH prepared-image layout foreign", frozen_v3_accepts(core(R)),
      v4_codes("pi_layout_foreign.json") == IDX["invalid/pi_layout_foreign.json"])

# soc identity on the prepared image (v3 does not cross-check soc)
R = v3_records(); R["PI1"]["soc"] = "op12"; refresh_chain_v3(R)
check("HH prepared-image soc foreign", frozen_v3_accepts(core(R)),
      v4_codes("pi_soc_foreign.json") == IDX["invalid/pi_soc_foreign.json"])

# device inventory soc contradicts the served chain (v3 does not reconcile DI.soc)
R = v3_records(); R["DI"]["soc"] = "op12"
check("HH device inventory soc mismatch", frozen_v3_accepts(core(R)),
      v4_codes("di_soc_mismatch.json") == IDX["invalid/di_soc_mismatch.json"])

# two dispatch decisions for one request (v3 does not enforce request->decision uniqueness).
# v3 has no decision_ts_us field, so the red-side duplicate is a byte-identical second decision.
R = v3_records()
dd2 = copy.deepcopy(R["DD1"])
check("HH duplicate dispatch for request", frozen_v3_accepts(core(R) + [dd2]),
      v4_codes("duplicate_dispatch_request.json") == IDX["invalid/duplicate_dispatch_request.json"])


# ===== A3. second-round hole-hunt closures: red-v3 / green-v4 =====
V2D = s9lib.DIGEST_BUILDERS

# served SoC not a member of manifest.compatible_soc (v3 checks neither membership nor soc)
R = v3_records(); R["MM"]["compatible_soc"] = ["op12"]
check("HH2 manifest compatible_soc", frozen_v3_accepts(core(R)),
      v4_codes("manifest_compatible_soc.json") == IDX["invalid/manifest_compatible_soc.json"])

# image_class not legal for the serving backend (v3 does not bind format to backend)
R = v3_records(); R["PI1"]["image_class"] = "gpu_xmem_prepacked"; refresh_chain_v3(R)
check("HH2 image_class vs backend", frozen_v3_accepts(core(R)),
      v4_codes("image_class_backend.json") == IDX["invalid/image_class_backend.json"])

# served backend_build not in manifest.required_backends (v3 does not check the manifest)
R = v3_records(); R["CC1"]["backend_build"] = F2.h("build:rogue"); R["PI1"]["backend_build"] = F2.h("build:rogue")
refresh_chain_v3(R)
check("HH2 backend_build vs manifest", frozen_v3_accepts(core(R)),
      v4_codes("backend_build_not_in_manifest.json") == IDX["invalid/backend_build_not_in_manifest.json"])

# ResidencyLease derived reservation phantom (v3 has no ledger reconciliation)
R = v3_records(); R["RL1"]["reserved_bytes"]["derived"] = 500; refresh_chain_v3(R)
check("HH2 RL derived phantom", frozen_v3_accepts(core(R)),
      v4_codes("rl_derived_phantom.json") == IDX["invalid/rl_derived_phantom.json"])

# allocation under-reservation vs weight set total (v3 does not compare them)
R = v3_records(); R["ALLOC1"]["canonical_bytes"] = 200; refresh_chain_v3(R)
check("HH2 alloc under-reservation", frozen_v3_accepts(core(R)),
      v4_codes("alloc_underreservation.json") == IDX["invalid/alloc_underreservation.json"])

# duplicate single-copy allocation (v3 has no single-copy dedup)
R = v3_records()
al2 = copy.deepcopy(R["ALLOC1"]); al2["allocation_id"] = "alloc-ws1-copy"
al2["alias_prepared_image_ids"] = []; al2["alias_refcount"] = 0; al2["lease_refcount"] = 0
al2["reclaimable"] = True; al2["allocation_digest"] = V2D["canonical_allocation"][1](al2)
check("HH2 duplicate allocation", frozen_v3_accepts(core(R) + [al2]),
      v4_codes("duplicate_allocation.json") == IDX["invalid/duplicate_allocation.json"])

# dtype disagreement across allocation / weight set / manifest (v3 does not cross-check dtype)
R = v3_records(); R["ALLOC1"]["dtype"] = "bf16"; refresh_chain_v3(R)
check("HH2 dtype mismatch", frozen_v3_accepts(core(R)),
      v4_codes("dtype_mismatch.json") == IDX["invalid/dtype_mismatch.json"])

# ReadyCertificate embedded correctness metric != its correctness certificate (v3 checks only the digest)
R = v3_records(); R["RC1"]["correctness"]["metric_digest"] = F2.h("metric:rogue"); refresh_chain_v3(R)
check("HH2 cert embedded metric mismatch", frozen_v3_accepts(core(R)),
      v4_codes("cert_metric_mismatch.json") == IDX["invalid/cert_metric_mismatch.json"])

# TransferTicket issued boot != the device it targets (v3 has no ticket/device anchor)
R = v3_records(); R["TT1"]["issued_boot_epoch"] = 8; R["TFbulk"]["boot_epoch"] = 8; refresh_chain_v3(R)
check("HH2 ticket issued boot vs device", frozen_v3_accepts(core(R)),
      v4_codes("ticket_issued_boot_device.json") == IDX["invalid/ticket_issued_boot_device.json"])


# ================= B. digest-field effectiveness (v4 binds, v3 does not) =================
def eff(name, kind, v4b, v2b, rec, field, newval):
    m = {**copy.deepcopy(rec), field: newval}
    v4_binds = v4b(rec) != v4b(m)
    v3_ignores = v2b(rec) == v2b(m)
    results.append((name, v4_binds and v3_ignores))
    print(f"  {'PASS' if (v4_binds and v3_ignores) else 'FAIL'} {name}  "
          f"v4-binds={v4_binds} v3-ignores={v3_ignores}")


RB = F2.build_records()
eff("digest source_weight_set_id", "prepared_image", s9lib.prepared_image_digest_v4,
    s9lib.prepared_image_digest, RB["PI1"], "source_weight_set_id", "ws-other")
eff("digest issued_boot_epoch", "transfer_ticket", s9lib.transfer_ticket_digest_v4,
    s9lib.transfer_ticket_digest, RB["TT1"], "issued_boot_epoch", 999)
eff("digest issued_residency_generation", "transfer_ticket", s9lib.transfer_ticket_digest_v4,
    s9lib.transfer_ticket_digest, RB["TT1"], "issued_residency_generation", 999)
eff("digest reserved_state_bytes", "state_lease", s9lib.state_lease_digest_v4,
    s9lib.state_lease_digest, RB["SL1"], "reserved_state_bytes", 12345)
eff("digest reserved_activation_bytes", "state_lease", s9lib.state_lease_digest_v4,
    s9lib.state_lease_digest, RB["SL1"], "reserved_activation_bytes", 12345)
eff("digest in_flight_mutation_seq", "state_lease", s9lib.state_lease_digest_v4,
    s9lib.state_lease_digest, RB["SL1"], "in_flight_mutation_seq", 7)

# also confirm the LIVE v4 validator still ACCEPTS the coherent v4 bundle
valid = V4VAL.validate_bundle(os.path.join(SPIKE, "fixtures", "v4", "bundles", "valid", "dispatchable.json"))
results.append(("valid v4 dispatchable accepted", valid == []))
print(f"  {'PASS' if valid == [] else 'FAIL'} valid v4 dispatchable accepted  errs={valid}")

n_fail = sum(1 for _, ok in results if not ok)
print(f"\nv4 regression + effectiveness tests: {len(results)}  failures: {n_fail}")
sys.exit(1 if n_fail else 0)
