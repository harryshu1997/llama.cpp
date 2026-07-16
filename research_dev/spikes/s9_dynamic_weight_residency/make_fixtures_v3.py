#!/usr/bin/env python3
"""S9-V0-R1 v3 fixture generator (deterministic).

Emits:
  fixtures/v3/{valid,invalid}/         one valid record per kind (schema_version 3)
  fixtures/v3/bundles/{valid,invalid}/ a coherent-chain bundle + R1 adversarials that the
                                       frozen V0-R validator ACCEPTS (fail-open) and R1 rejects
  fixtures/v3/index.json, fixtures/v3/bundles/index.json

Records are the v2 records bumped to schema_version 3 (record shapes are identical; the
digest pre-images do not bind schema_version, so digests stay valid). No clock/PRNG.
ASCII only.
"""
import copy
import json
import os

import s9lib
import make_fixtures_v2 as F2

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures", "v3")


def v3(rec):
    r = copy.deepcopy(rec)
    r["schema_version"] = 3
    return r


def realloc(rec):
    """recompute allocation_digest after editing an allocation record."""
    rec = copy.deepcopy(rec)
    rec["allocation_digest"] = s9lib.alloc_digest(rec)
    return rec


def write(sub, name, obj):
    d = os.path.join(FX, sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write(s9lib.canonical(obj) + "\n")


def bundle(bid, records):
    return {"schema_version": 3, "kind": "bundle", "bundle_version": 3, "bundle_id": bid,
            "records": [v3(r) for r in records]}


SCHEMA = {k: f"schemas/v3/{v}" for k, v in s9lib.V3_KINDS.items()}


def emit_schema_fixtures(R):
    valid = []

    def v(name, kind, rec):
        write("valid", name, v3(rec))
        valid.append((f"valid/{name}", SCHEMA[kind]))

    v("weight_segment.json", "weight_segment", R["SEGa"])
    v("weight_set.json", "weight_set", R["WS1"])
    v("model_manifest.json", "model_manifest", R["MM"])
    v("canonical_allocation.json", "canonical_allocation", R["ALLOC1"])
    v("prepared_image.json", "prepared_image", R["PI1"])
    v("correctness_certificate.json", "correctness_certificate", R["CC1"])
    v("island_executable.json", "island_executable", R["ISL1"])
    v("ready_certificate.json", "ready_certificate", R["RC1"])
    v("residency_lease.json", "residency_lease", R["RL1"])
    v("state_lease.json", "state_lease", R["SL1"])
    v("dispatch_decision.json", "dispatch_decision", R["DD1"])
    v("transfer_ticket.json", "transfer_ticket", R["TT1"])
    v("transport_frame.json", "transport_frame", R["TFbulk"])
    v("device_inventory.json", "device_inventory", R["DI"])
    idx = [{"file": f, "schema": s, "expect": "valid"} for f, s in valid]
    with open(os.path.join(FX, "index.json"), "w") as f:
        f.write(json.dumps(idx, indent=2) + "\n")
    return len(valid)


def emit_bundle_fixtures(R):
    valid, invalid = [], []

    def bv(name, obj):
        write(os.path.join("bundles", "valid"), name, obj)
        valid.append(f"valid/{name}")

    def bi(name, obj):
        write(os.path.join("bundles", "invalid"), name, obj)
        invalid.append(f"invalid/{name}")

    core = [R["MM"], R["SEGa"], R["WS1"], R["ALLOC1"], R["PI1"], R["CC1"], R["ISL1"],
            R["RC1"], R["RL1"], R["SL1"], R["DD1"], R["TT1"], R["TT2"], R["TFbulk"], R["TFexec"], R["DI"]]
    bv("dispatchable.json", bundle("b3-dispatchable", core))

    # (1) DISPATCH wrong device
    b = copy.deepcopy(core)
    for r in b:
        if r.get("kind") == "dispatch_decision":
            r["device_id"] = "op12"
    bi("dispatch_wrong_device.json", bundle("b3-wrong-device", b))

    # (1) DISPATCH wrong backend (island is htp)
    b = copy.deepcopy(core)
    for r in b:
        if r.get("kind") == "dispatch_decision":
            r["backend"] = "gpu"
    bi("dispatch_wrong_backend.json", bundle("b3-wrong-backend", b))

    # (1) DISPATCH wrong route epoch (state lease route is 3)
    b = copy.deepcopy(core)
    for r in b:
        if r.get("kind") == "dispatch_decision":
            r["route_epoch"] = 99
    bi("dispatch_wrong_route.json", bundle("b3-wrong-route", b))

    # (1) DISPATCH wrong request id (foreign vs state lease request)
    b = copy.deepcopy(core)
    for r in b:
        if r.get("kind") == "dispatch_decision":
            r["request_id"] = "req-OTHER"
    bi("dispatch_wrong_request.json", bundle("b3-wrong-request", b))

    # (2) sticky DISPATCH with null state lease
    b = copy.deepcopy(core)
    for r in b:
        if r.get("kind") == "dispatch_decision":
            r["state_lease_id"] = None
    bi("sticky_null_state_lease.json", bundle("b3-null-sl", b))

    # (2) sticky DISPATCH with a foreign state lease (its request_id differs)
    b = copy.deepcopy(core)
    for r in b:
        if r.get("kind") == "state_lease":
            r["request_id"] = "req-foreign"
            r["state_lease_digest"] = s9lib.state_lease_digest({**r, "request_id": "req-foreign"})
    bi("sticky_foreign_state_lease.json", bundle("b3-foreign-sl", b))

    # (3) incoherent tuple (ws1, pi2, rc1, rl1): records exist but do not form one chain
    b = [R["MM"], R["SEGa"], R["SEGb"], R["WS1"], R["WS2"], R["ALLOC1"], R["ALLOC2"], R["PI1"], R["PI2"],
         R["CC1"], R["CC2"], R["ISL1"], R["RC1"], R["RC2"], R["RL1"], R["RL2"], R["SL1"]]
    tup = {"weight_set_id": "ws1", "prepared_image_id": "pi2", "ready_certificate_id": "rc1",
           "residency_lease_id": "rl1"}
    dd = {**copy.deepcopy(R["DD1"]), "required_tuples": [dict(tup)], "satisfied_tuples": [dict(tup)]}
    b.append(dd)
    bi("incoherent_tuple.json", bundle("b3-incoherent", b))

    # (4) DISPATCH referencing a nonexistent island
    b = copy.deepcopy(core)
    for r in b:
        if r.get("kind") == "dispatch_decision":
            r["island_id"] = "isl-ghost"
    bi("nonexistent_island.json", bundle("b3-ghost-island", b))

    # (5) PreparedImage points to an allocation but is missing from its alias set
    al = realloc({**copy.deepcopy(R["ALLOC1"]), "alias_prepared_image_ids": [], "alias_refcount": 0,
                  "lease_refcount": 1, "reclaimable": False})
    b = [R["MM"], R["SEGa"], R["WS1"], al, R["PI1"], R["CC1"], R["RC1"], R["RL1"]]
    bi("pi_missing_from_alias_set.json", bundle("b3-alias-missing", b))

    # (6) Allocation reclaimable while an image still references it
    al = realloc({**copy.deepcopy(R["ALLOC1"]), "alias_prepared_image_ids": [], "alias_refcount": 0,
                  "lease_refcount": 0, "reclaimable": True})
    b = [R["MM"], R["SEGa"], R["WS1"], al, R["PI1"]]
    bi("reclaimable_while_referenced.json", bundle("b3-reclaimable", b))

    # --- holes found by the adversarial verification pass (workflow + review) ---

    # island required_weight_set_digests not covered by the dispatched tuple (served content != certified content)
    isl = copy.deepcopy(R["ISL1"]); isl["required_weight_set_digests"] = [F2.h("ghost-ws-digest")]
    isl["island_digest"] = s9lib.island_digest(isl)
    b = [r for r in core if r.get("kind") != "island_executable"] + [isl]
    bi("island_ws_digest_ghost.json", bundle("b3-isl-wsdig", b))

    # island required_prepared_image_ids/digests not covered by the dispatched tuple
    isl = copy.deepcopy(R["ISL1"]); isl["required_prepared_image_ids"] = ["pi-ghost"]
    isl["required_prepared_image_digests"] = [F2.h("ghost-pi-digest")]
    isl["island_digest"] = s9lib.island_digest(isl)
    b = [r for r in core if r.get("kind") != "island_executable"] + [isl]
    bi("island_pi_ghost.json", bundle("b3-isl-pi", b))

    # DISPATCH against a DRAINING residency lease (state not dispatchable)
    rl = copy.deepcopy(R["RL1"]); rl["state"] = "DRAINING"
    rl["residency_lease_digest"] = s9lib.residency_lease_digest(rl)
    b = [r for r in core if not (r.get("kind") == "residency_lease" and r.get("residency_lease_id") == "rl1")] + [rl]
    bi("draining_lease.json", bundle("b3-draining", b))

    # StateLease minted in a stale boot generation attaches to a live chain
    sl = copy.deepcopy(R["SL1"]); sl["boot_epoch"] = 999
    sl["state_lease_digest"] = s9lib.state_lease_digest(sl)
    b = [r for r in core if r.get("kind") != "state_lease"] + [sl]
    bi("state_lease_stale_boot.json", bundle("b3-sl-boot", b))

    # residency lease for ws2 charged to the ws1 allocation (would let ws2 allocation reclaim while live)
    rl2 = copy.deepcopy(R["RL2"]); rl2["source_allocation_id"] = "alloc-ws1"
    rl2["residency_lease_digest"] = s9lib.residency_lease_digest(rl2)
    al2 = realloc({**copy.deepcopy(R["ALLOC2"]), "lease_refcount": 0})
    b = [R["MM"], R["SEGa"], R["SEGb"], R["WS1"], R["WS2"], R["ALLOC1"], al2, R["PI1"], R["PI2"],
         R["CC1"], R["CC2"], R["ISL1"], R["RC1"], R["RC2"], R["RL1"], rl2, R["SL1"]]
    bi("lease_alloc_ws_mismatch.json", bundle("b3-lease-alloc", b))

    # StateLease is coherent on its own but depends on an unused residency chain, not the dispatched tuple.
    sl = copy.deepcopy(R["SL1"])
    sl["depends_on_residency_lease_id"] = "rl2"
    sl["depends_on_residency_generation"] = R["RL2"]["residency_generation"]
    sl["boot_epoch"] = R["RL2"]["boot_epoch"]
    sl["state_lease_digest"] = s9lib.state_lease_digest(sl)
    b = [R["MM"], R["SEGa"], R["SEGb"], R["WS1"], R["WS2"], R["ALLOC1"], R["ALLOC2"],
         R["PI1"], R["PI2"], R["CC1"], R["CC2"], R["ISL1"], R["RC1"], R["RC2"],
         R["RL1"], R["RL2"], sl, R["DD1"]]
    bi("state_lease_foreign_residency.json", bundle("b3-sl-foreign-rl", b))

    # ReadyCertificate is a valid chain member but cites correctness for a different island.
    rc = copy.deepcopy(R["RC1"])
    rc["correctness_id"] = R["CC2"]["correctness_id"]
    rc["correctness_digest"] = R["CC2"]["correctness_digest"]
    rc["correctness"]["metric_digest"] = R["CC2"]["metric_digest"]
    rc["ready_certificate_digest"] = s9lib.ready_cert_digest(rc)
    rl = copy.deepcopy(R["RL1"])
    rl["ready_certificate_digest"] = rc["ready_certificate_digest"]
    rl["residency_lease_digest"] = s9lib.residency_lease_digest(rl)
    b = [R["MM"], R["SEGa"], R["SEGb"], R["WS1"], R["WS2"], R["ALLOC1"], R["ALLOC2"],
         R["PI1"], R["PI2"], R["CC1"], R["CC2"], R["ISL1"], rc, R["RC2"], rl, R["RL2"],
         R["SL1"], R["DD1"]]
    bi("ready_cert_foreign_correctness.json", bundle("b3-rc-foreign-cc", b))

    expected_codes = {
        "invalid/dispatch_wrong_device.json": ["E_CHAIN_BROKEN", "E_STATE_LEASE"],
        "invalid/dispatch_wrong_backend.json": ["E_CHAIN_BROKEN", "E_DISPATCH_MISMATCH", "E_STATE_LEASE"],
        "invalid/dispatch_wrong_route.json": ["E_STATE_LEASE"],
        "invalid/dispatch_wrong_request.json": ["E_STATE_LEASE"],
        "invalid/sticky_null_state_lease.json": ["E_STATE_LEASE"],
        "invalid/sticky_foreign_state_lease.json": ["E_STATE_LEASE"],
        "invalid/incoherent_tuple.json": ["E_CHAIN_BROKEN", "E_DISPATCH_MISMATCH"],
        "invalid/nonexistent_island.json": ["E_ISLAND_ABSENT"],
        "invalid/pi_missing_from_alias_set.json": ["E_ALIAS_SET"],
        "invalid/reclaimable_while_referenced.json": ["E_ALIAS_SET"],
        "invalid/island_ws_digest_ghost.json": ["E_DISPATCH_MISMATCH"],
        "invalid/island_pi_ghost.json": ["E_DISPATCH_MISMATCH"],
        "invalid/draining_lease.json": ["E_CHAIN_BROKEN"],
        "invalid/state_lease_stale_boot.json": ["E_STATE_LEASE"],
        "invalid/lease_alloc_ws_mismatch.json": ["E_ALLOC_REFCOUNT", "E_CHAIN_BROKEN"],
        "invalid/state_lease_foreign_residency.json": ["E_STATE_LEASE"],
        "invalid/ready_cert_foreign_correctness.json": ["E_CHAIN_BROKEN"],
    }
    assert set(expected_codes) == set(invalid)
    idx = ([{"file": f, "expect": "valid", "expected_codes": []} for f in valid] +
           [{"file": f, "expect": "invalid", "expected_codes": expected_codes[f]} for f in invalid])
    with open(os.path.join(FX, "bundles", "index.json"), "w") as f:
        f.write(json.dumps(idx, indent=2) + "\n")
    return len(valid), len(invalid)


def main():
    R = F2.build_records()
    sv = emit_schema_fixtures(R)
    bvn, bin_ = emit_bundle_fixtures(R)
    print(f"v3 schema fixtures: {sv} valid")
    print(f"v3 bundle fixtures: {bvn} valid + {bin_} invalid = {bvn + bin_}")


if __name__ == "__main__":
    main()
