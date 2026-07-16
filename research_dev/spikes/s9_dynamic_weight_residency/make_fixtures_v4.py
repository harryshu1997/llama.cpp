#!/usr/bin/env python3
"""S9-V0-R2 v4 fixture generator (deterministic).

Emits:
  fixtures/v4/{valid}/                 one valid record per kind (schema_version 4)
  fixtures/v4/bundles/{valid,invalid}/ a coherent-chain bundle + one adversarial per review
                                       blocker that the frozen v3 validator ACCEPTS (fail-open)
                                       and the v4 validator REJECTS with a stable code.
  fixtures/v4/index.json, fixtures/v4/bundles/index.json

v4 records are the v2 records bumped to schema_version 4, re-digested with the v4 family
(s9lib.*_v4), with the dispatch carrying decision_ts_us + device_status_ref, the transport
frames carrying device_id, and a ledger-coherent DeviceInventory. The state-lease physical
reservations are scaled down so the toy LPDDR ledger can hold them and be derived exactly.
No clock/PRNG. ASCII only.
"""
import copy
import json
import os

import s9lib
import make_fixtures_v2 as F2

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures", "v4")
GHOST = F2.h("ghost")
OTHER_GRAPH = F2.h("graph:other-model")


def refresh_chain_v4(R):
    """Recompute EVERY digest + cross-referenced digest field, in dependency order, with the
    v4 family, for the known fixture topology. After this the whole record set is a coherent
    chain of v4 digests (so a mutation to a NON-digest-bound field is the only thing a
    cross-record check can still see)."""
    for k in ("ALLOC1", "ALLOC2"):
        if k in R:
            R[k]["allocation_digest"] = s9lib.alloc_digest_v4(R[k])
    for k in ("PI1", "PI2", "PIgpu"):
        if k in R:
            R[k]["derived_image_digest"] = s9lib.prepared_image_digest_v4(R[k])
    for k in ("CC1", "CC2"):
        if k in R:
            R[k]["correctness_digest"] = s9lib.correctness_digest_v4(R[k])
    if "ISL1" in R:
        R["ISL1"]["required_prepared_image_digests"] = [R["PI1"]["derived_image_digest"]]
        R["ISL1"]["correctness_digest"] = R["CC1"]["correctness_digest"]
        R["ISL1"]["island_digest"] = s9lib.island_digest_v4(R["ISL1"])
    if "ISL2" in R:
        R["ISL2"]["required_prepared_image_digests"] = [R["PI1"]["derived_image_digest"], R["PI2"]["derived_image_digest"]]
        R["ISL2"]["correctness_digest"] = R["CC2"]["correctness_digest"]
        R["ISL2"]["island_digest"] = s9lib.island_digest_v4(R["ISL2"])
    for rck, cck, pik in (("RC1", "CC1", "PI1"), ("RC2", "CC2", "PI2")):
        if rck in R:
            R[rck]["prepared_image_digest"] = R[pik]["derived_image_digest"]
            R[rck]["correctness_digest"] = R[cck]["correctness_digest"]
            R[rck]["ready_certificate_digest"] = s9lib.ready_cert_digest_v4(R[rck])
    for rlk, rck in (("RL1", "RC1"), ("RL2", "RC2")):
        if rlk in R:
            R[rlk]["ready_certificate_digest"] = R[rck]["ready_certificate_digest"]
            R[rlk]["residency_lease_digest"] = s9lib.residency_lease_digest_v4(R[rlk])
    if "SL1" in R:
        R["SL1"]["state_lease_digest"] = s9lib.state_lease_digest_v4(R["SL1"])
    for k in ("TT1", "TT2"):
        if k in R:
            R[k]["transfer_ticket_digest"] = s9lib.transfer_ticket_digest_v4(R[k])
    return R


def build_records_v4():
    R = copy.deepcopy(F2.build_records())
    for r in R.values():
        r["schema_version"] = 4
    # v4-only: state reservations scaled to fit the toy ledger (match RC1's reserved parts)
    R["SL1"]["reserved_state_bytes"] = 60
    R["SL1"]["reserved_activation_bytes"] = 40
    # v4-only dispatch fields
    R["DD1"]["decision_ts_us"] = 2000000
    R["DD1"]["device_status_ref"] = {"device_id": "op15", "boot_epoch": 7, "status_seq": 42}
    # v4-only transport-frame device binding
    R["TFbulk"]["device_id"] = "op15"
    R["TFexec"]["device_id"] = "op15"
    # DeviceInventory ledger derived EXACTLY from the v4 core records on op15:
    #   weights_resident=ALLOC1(400) derived_images=PI1 shared(0) scratch=RL1(100)
    #   activations=SL1(40) mutable_state=SL1(60) free=10000-600
    R["DI"]["physical_byte_accounting"] = {"weights_resident": 400, "derived_images": 0,
        "scratch": 100, "activations": 40, "mutable_state": 60, "free": 9400}
    refresh_chain_v4(R)
    return R


def v4(rec):
    return copy.deepcopy(rec)


def write(sub, name, obj):
    d = os.path.join(FX, sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write(s9lib.canonical(obj) + "\n")


def bundle(bid, records):
    return {"schema_version": 4, "kind": "bundle", "bundle_version": 4, "bundle_id": bid,
            "records": [v4(r) for r in records]}


SCHEMA = {k: f"schemas/v4/{v}" for k, v in s9lib.V4_KINDS.items()}


def emit_schema_fixtures(R):
    valid = []

    def v(name, kind, rec):
        write("valid", name, v4(rec))
        valid.append((f"valid/{name}", SCHEMA[kind]))

    v("weight_segment.json", "weight_segment", R["SEGa"])
    v("weight_set.json", "weight_set", R["WS1"])
    v("model_manifest.json", "model_manifest", R["MM"])
    v("canonical_allocation.json", "canonical_allocation", R["ALLOC1"])
    v("prepared_image_htp.json", "prepared_image", R["PI1"])
    v("prepared_image_gpu.json", "prepared_image", R["PIgpu"])
    v("correctness_certificate.json", "correctness_certificate", R["CC1"])
    v("island_executable.json", "island_executable", R["ISL1"])
    v("ready_certificate.json", "ready_certificate", R["RC1"])
    v("residency_lease.json", "residency_lease", R["RL1"])
    v("state_lease.json", "state_lease", R["SL1"])
    v("dispatch_decision.json", "dispatch_decision", R["DD1"])
    v("transfer_ticket.json", "transfer_ticket", R["TT1"])
    v("transport_frame_bulk.json", "transport_frame", R["TFbulk"])
    v("transport_frame_exec.json", "transport_frame", R["TFexec"])
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

    core_keys = ["MM", "SEGa", "WS1", "ALLOC1", "PI1", "CC1", "ISL1", "RC1", "RL1", "SL1",
                 "DD1", "TT1", "TT2", "TFbulk", "TFexec", "DI"]

    def fresh():
        return {k: copy.deepcopy(R[k]) for k in R}

    def core_from(rr):
        return [rr[k] for k in core_keys]

    bv("dispatchable.json", bundle("b4-dispatchable", core_from(R)))

    # --- blocker 1: authoritative DeviceInventory snapshot ---
    rr = fresh()
    b = [rr[k] for k in core_keys if k != "DI"]           # (1a) DeviceInventory absent
    bi("device_absent.json", bundle("b4-dev-absent", b))

    rr = fresh(); rr["DD1"]["device_status_ref"]["boot_epoch"] = 8  # (1b) pinned wrong boot
    bi("device_boot_mismatch.json", bundle("b4-dev-boot", core_from(rr)))

    rr = fresh(); rr["DD1"]["device_status_ref"]["status_seq"] = 99  # (1c) pinned wrong status_seq
    bi("device_status_seq_mismatch.json", bundle("b4-dev-seq", core_from(rr)))

    # --- blocker 2: DeviceInventory eligibility ---
    rr = fresh(); rr["DI"]["draining"] = True
    bi("device_draining.json", bundle("b4-dev-drain", core_from(rr)))

    rr = fresh(); rr["DI"]["thermal"]["eligible"] = False
    bi("device_thermal_ineligible.json", bundle("b4-dev-thermal", core_from(rr)))

    rr = fresh(); rr["DI"]["backends"] = ["gpu", "cpu"]   # htp not supported
    bi("device_backend_unsupported.json", bundle("b4-dev-backend", core_from(rr)))

    # --- blocker 3: residency lease expired at the decision timestamp ---
    rr = fresh(); rr["DD1"]["decision_ts_us"] = 61000001   # RL1 horizon.end_us is 61000000
    bi("lease_expired.json", bundle("b4-lease-expired", core_from(rr)))

    # --- blocker 4: CorrectnessCertificate.backend_build != PreparedImage.backend_build ---
    rr = fresh(); rr["CC1"]["backend_build"] = F2.BUILD_GPU
    refresh_chain_v4(rr)
    bi("backend_build_mismatch.json", bundle("b4-backend-build", core_from(rr)))

    # --- blocker 5: island graph identity != manifest / prepared image ---
    rr = fresh(); rr["ISL1"]["graph_hash"] = OTHER_GRAPH
    refresh_chain_v4(rr)
    bi("island_graph_mismatch.json", bundle("b4-isl-graph", core_from(rr)))

    # --- blocker 6: PreparedImage.source_weight_set_id names a foreign set ---
    rr = fresh(); rr["PI1"]["source_weight_set_id"] = "ws2"
    refresh_chain_v4(rr)
    bi("pi_source_foreign.json", bundle("b4-pi-source", core_from(rr)))

    # --- blocker 7: transport frame epoch stacks foreign vs ticket / dispatch ---
    rr = fresh(); rr["TFbulk"]["residency_epoch"] = 99     # (7a) bulk frame vs ticket issued gen
    bi("bulk_frame_stale_epoch.json", bundle("b4-bulk-epoch", core_from(rr)))

    rr = fresh(); rr["TFexec"]["route_epoch"] = 99          # (7b) EXECUTE frame vs dispatch route
    bi("exec_frame_stale_route.json", bundle("b4-exec-route", core_from(rr)))

    # --- blocker 8: TransferTicket issued boot/gen now bound in the v4 digest ---
    # coherent v4 ticket, then bump issued_boot_epoch WITHOUT recomputing (align the frame so
    # only the ticket digest is affected) -> v4 recompute binds issued_boot_epoch -> mismatch.
    rr = fresh(); rr["TT1"]["issued_boot_epoch"] = 8; rr["TFbulk"]["boot_epoch"] = 8
    bi("ticket_issued_boot_unbound.json", bundle("b4-tt-issued", core_from(rr)))

    # --- blocker 9: StateLease in_flight_mutation_seq now bound in the v4 digest ---
    rr = fresh(); rr["SL1"]["in_flight_mutation_seq"] = 5   # digest NOT refreshed
    bi("state_mutation_seq_unbound.json", bundle("b4-sl-seq", core_from(rr)))

    # --- blocker 10: StateLease reservation exceeds the device ledger ---
    rr = fresh(); rr["SL1"]["reserved_activation_bytes"] = 20000
    rr["SL1"]["state_lease_digest"] = s9lib.state_lease_digest_v4(rr["SL1"])  # keep digest valid
    bi("state_reservation_exceeds_ledger.json", bundle("b4-sl-exceeds", core_from(rr)))

    # --- blocker 11: DeviceInventory declares a zero-live ledger while records are live ---
    rr = fresh(); rr["DI"]["physical_byte_accounting"] = {"weights_resident": 0, "derived_images": 0,
        "scratch": 0, "activations": 0, "mutable_state": 0, "free": 10000}
    bi("ledger_zero_live.json", bundle("b4-ledger-zero", core_from(rr)))

    # --- holes found by the independent adversarial hole hunt (8-lens workflow + verify pass) ---

    # (7) a RESULT frame is a live-payload frame too; a foreign one dodged the EXECUTE-only check
    rr = fresh()
    res = copy.deepcopy(rr["TFexec"]); res["msg_type"] = "RESULT"; res["priority_class"] = "result"
    res["route_epoch"] = 99; res["idempotency_key"] = "idem-result-1"; res["seq"] = 200
    bi("result_frame_stale.json", bundle("b4-result-frame", core_from(rr) + [res]))

    # (4/5) island model identity is only anchored if a manifest for its model_id is PRESENT
    rr = fresh()
    bi("island_no_manifest.json", bundle("b4-no-manifest", [rr[k] for k in core_keys if k != "MM"]))

    # (10/11) a live allocation on a device that has NO DeviceInventory must not escape the ledger
    rr = fresh()
    al = copy.deepcopy(rr["ALLOC1"]); al["allocation_id"] = "alloc-op12"; al["device_id"] = "op12"
    al["canonical_bytes"] = 9000; al["alias_prepared_image_ids"] = []; al["alias_refcount"] = 0
    al["lease_refcount"] = 0; al["reclaimable"] = True; al["allocation_digest"] = s9lib.alloc_digest_v4(al)
    bi("ledger_diless_device.json", bundle("b4-diless", core_from(rr) + [al]))

    # (10) a ReadyCertificate cannot attest a footprint larger than the device LPDDR
    rr = fresh()
    rr["RC1"]["physical_bytes"] = {"canonical": 400, "derived": 0, "scratch": 100,
        "activations_reserved": 40, "state_reserved": 500000, "total": 500540}
    rr["RC1"]["ready_certificate_digest"] = s9lib.ready_cert_digest_v4(rr["RC1"])
    rr["RL1"]["ready_certificate_digest"] = rr["RC1"]["ready_certificate_digest"]
    rr["RL1"]["residency_lease_digest"] = s9lib.residency_lease_digest_v4(rr["RL1"])
    bi("rc_footprint_exceeds_lpddr.json", bundle("b4-rc-footprint", core_from(rr)))

    # (10) a ResidencyLease's weight reservation must equal its single-copy canonical allocation
    rr = fresh()
    rr["RL1"]["reserved_bytes"]["weights"] = 9999
    rr["RL1"]["residency_lease_digest"] = s9lib.residency_lease_digest_v4(rr["RL1"])
    bi("rl_weights_mismatch.json", bundle("b4-rl-weights", core_from(rr)))

    # (4/5) arch is a digest-bound identity field; a foreign arch on the prepared image must be caught
    rr = fresh(); rr["PI1"]["arch"] = "llama3"; refresh_chain_v4(rr)
    bi("pi_arch_foreign.json", bundle("b4-pi-arch", core_from(rr)))

    # (4/5) layout_version identity must agree across prepared image / weight set / allocation / manifest
    rr = fresh(); rr["PI1"]["layout_version"] = 4; refresh_chain_v4(rr)
    bi("pi_layout_foreign.json", bundle("b4-pi-layout", core_from(rr)))

    # (4/5) soc must agree across chain + device: a v75 image cannot run on a v81 device and vice versa
    rr = fresh(); rr["PI1"]["soc"] = "op12"; refresh_chain_v4(rr)
    bi("pi_soc_foreign.json", bundle("b4-pi-soc", core_from(rr)))

    # (1/4) the dispatch DeviceInventory soc must match the chain the dispatch serves
    rr = fresh(); rr["DI"]["soc"] = "op12"
    bi("di_soc_mismatch.json", bundle("b4-di-soc", core_from(rr)))

    # a request has exactly one dispatch decision: two DISPATCH records for req-1 are contradictory
    rr = fresh()
    dd2 = copy.deepcopy(rr["DD1"]); dd2["decision_ts_us"] = 3000000
    bi("duplicate_dispatch_request.json", bundle("b4-dup-dispatch", core_from(rr) + [dd2]))

    # --- second-round hole-hunt closures (coherence-field family) ---

    # the served SoC must be a MEMBER of the manifest's compatible_soc (equality alone is not enough)
    rr = fresh(); rr["MM"]["compatible_soc"] = ["op12"]
    bi("manifest_compatible_soc.json", bundle("b4-compat-soc", core_from(rr)))

    # the derived-image format must be legal for the serving backend (gpu format on an htp image)
    rr = fresh(); rr["PI1"]["image_class"] = "gpu_xmem_prepacked"; refresh_chain_v4(rr)
    bi("image_class_backend.json", bundle("b4-imgclass", core_from(rr)))

    # the served (backend, backend_build) must be sanctioned by the manifest's required_backends
    rr = fresh(); rr["CC1"]["backend_build"] = F2.h("build:rogue"); rr["PI1"]["backend_build"] = F2.h("build:rogue")
    refresh_chain_v4(rr)
    bi("backend_build_not_in_manifest.json", bundle("b4-build-manifest", core_from(rr)))

    # a ResidencyLease's derived reservation must equal the derived bytes of images on its allocation
    rr = fresh(); rr["RL1"]["reserved_bytes"]["derived"] = 500
    rr["RL1"]["residency_lease_digest"] = s9lib.residency_lease_digest_v4(rr["RL1"])
    bi("rl_derived_phantom.json", bundle("b4-rl-derived", core_from(rr)))

    # a canonical allocation must hold the FULL weight set (no under-reservation of physical bytes)
    rr = fresh(); rr["ALLOC1"]["canonical_bytes"] = 200
    rr["ALLOC1"]["allocation_digest"] = s9lib.alloc_digest_v4(rr["ALLOC1"])
    bi("alloc_underreservation.json", bundle("b4-alloc-under", core_from(rr)))

    # two canonical allocations of the SAME content-addressed weight set on one device (single-copy)
    rr = fresh()
    al2 = copy.deepcopy(rr["ALLOC1"]); al2["allocation_id"] = "alloc-ws1-copy"
    al2["alias_prepared_image_ids"] = []; al2["alias_refcount"] = 0; al2["lease_refcount"] = 0
    al2["reclaimable"] = True; al2["allocation_digest"] = s9lib.alloc_digest_v4(al2)
    bi("duplicate_allocation.json", bundle("b4-dup-alloc", core_from(rr) + [al2]))

    # dtype must agree across allocation / weight set / manifest
    rr = fresh(); rr["ALLOC1"]["dtype"] = "bf16"; rr["ALLOC1"]["allocation_digest"] = s9lib.alloc_digest_v4(rr["ALLOC1"])
    bi("dtype_mismatch.json", bundle("b4-dtype", core_from(rr)))

    # the ReadyCertificate's embedded correctness attestation must match its correctness certificate
    rr = fresh(); rr["RC1"]["correctness"]["metric_digest"] = F2.h("metric:rogue")
    rr["RC1"]["ready_certificate_digest"] = s9lib.ready_cert_digest_v4(rr["RC1"])
    rr["RL1"]["ready_certificate_digest"] = rr["RC1"]["ready_certificate_digest"]
    rr["RL1"]["residency_lease_digest"] = s9lib.residency_lease_digest_v4(rr["RL1"])
    bi("cert_metric_mismatch.json", bundle("b4-cert-metric", core_from(rr)))

    # a TransferTicket's issued boot generation must match the device it targets
    rr = fresh(); rr["TT1"]["issued_boot_epoch"] = 8; rr["TFbulk"]["boot_epoch"] = 8
    rr["TT1"]["transfer_ticket_digest"] = s9lib.transfer_ticket_digest_v4(rr["TT1"])
    bi("ticket_issued_boot_device.json", bundle("b4-tt-device", core_from(rr)))

    expected_codes = {
        "invalid/device_absent.json": ["E_DEVICE_ABSENT", "E_GATE_UNDERIVED", "E_LEDGER_DERIVED"],
        "invalid/device_boot_mismatch.json": ["E_DEVICE_STALE", "E_GATE_UNDERIVED"],
        "invalid/device_status_seq_mismatch.json": ["E_DEVICE_STALE", "E_GATE_UNDERIVED"],
        "invalid/device_draining.json": ["E_DEVICE_INELIGIBLE"],
        "invalid/device_thermal_ineligible.json": ["E_DEVICE_INELIGIBLE"],
        "invalid/device_backend_unsupported.json": ["E_DEVICE_INELIGIBLE"],
        "invalid/lease_expired.json": ["E_LEASE_EXPIRED"],
        "invalid/backend_build_mismatch.json": ["E_IDENTITY_MISMATCH"],
        "invalid/island_graph_mismatch.json": ["E_IDENTITY_MISMATCH"],
        "invalid/pi_source_foreign.json": ["E_PI_SOURCE"],
        "invalid/bulk_frame_stale_epoch.json": ["E_FRAME_EPOCH"],
        "invalid/exec_frame_stale_route.json": ["E_FRAME_EPOCH"],
        "invalid/ticket_issued_boot_unbound.json": ["E_DIGEST_MISMATCH", "E_FRAME_EPOCH"],
        "invalid/state_mutation_seq_unbound.json": ["E_DIGEST_MISMATCH"],
        "invalid/state_reservation_exceeds_ledger.json": ["E_GATE_UNDERIVED", "E_LEDGER_DERIVED"],
        "invalid/ledger_zero_live.json": ["E_GATE_UNDERIVED", "E_LEDGER_DERIVED"],
        # independent hole-hunt closures (round 2):
        "invalid/result_frame_stale.json": ["E_FRAME_EPOCH"],
        "invalid/island_no_manifest.json": ["E_IDENTITY_MISMATCH"],
        "invalid/ledger_diless_device.json": ["E_LEDGER_DERIVED"],
        "invalid/rc_footprint_exceeds_lpddr.json": ["E_LEDGER_DERIVED"],
        "invalid/rl_weights_mismatch.json": ["E_LEDGER_DERIVED"],
        "invalid/pi_arch_foreign.json": ["E_IDENTITY_MISMATCH"],
        "invalid/pi_layout_foreign.json": ["E_IDENTITY_MISMATCH"],
        "invalid/pi_soc_foreign.json": ["E_IDENTITY_MISMATCH"],
        "invalid/di_soc_mismatch.json": ["E_IDENTITY_MISMATCH"],
        "invalid/duplicate_dispatch_request.json": ["E_DISPATCH_MISMATCH"],
        # second-round hole-hunt closures:
        "invalid/manifest_compatible_soc.json": ["E_IDENTITY_MISMATCH"],
        "invalid/image_class_backend.json": ["E_IDENTITY_MISMATCH"],
        "invalid/backend_build_not_in_manifest.json": ["E_IDENTITY_MISMATCH"],
        "invalid/rl_derived_phantom.json": ["E_LEDGER_DERIVED"],
        "invalid/alloc_underreservation.json": ["E_GATE_UNDERIVED", "E_LEDGER_DERIVED"],
        "invalid/duplicate_allocation.json": ["E_GATE_UNDERIVED", "E_LEDGER_DERIVED"],
        "invalid/dtype_mismatch.json": ["E_IDENTITY_MISMATCH"],
        "invalid/cert_metric_mismatch.json": ["E_CORRECTNESS_BINDING"],
        "invalid/ticket_issued_boot_device.json": ["E_FRAME_EPOCH"],
    }
    assert set(expected_codes) == set(invalid), \
        f"index drift missing={set(expected_codes) - set(invalid)} phantom={set(invalid) - set(expected_codes)}"
    idx = ([{"file": f, "expect": "valid", "expected_codes": []} for f in valid] +
           [{"file": f, "expect": "invalid", "expected_codes": expected_codes[f]} for f in invalid])
    with open(os.path.join(FX, "bundles", "index.json"), "w") as f:
        f.write(json.dumps(idx, indent=2) + "\n")
    return len(valid), len(invalid)


def main():
    R = build_records_v4()
    sv = emit_schema_fixtures(R)
    bvn, bin_ = emit_bundle_fixtures(R)
    print(f"v4 schema fixtures: {sv} valid")
    print(f"v4 bundle fixtures: {bvn} valid + {bin_} invalid = {bvn + bin_}")


if __name__ == "__main__":
    main()
