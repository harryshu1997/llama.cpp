#!/usr/bin/env python3
"""S9-V0-R v2 fixture generator (deterministic).

Emits:
  fixtures/v2/{valid,invalid}/         one record per kind + schema-level adversarials
  fixtures/v2/bundles/{valid,invalid}/ one internally-consistent bundle + cross-record
                                       adversarials (each breaks exactly one bundle rule)
  fixtures/v2/index.json               schema-fixture index (for run_schema_tests.py)
  fixtures/v2/bundles/index.json       bundle-fixture index (for bundle_validate.py --selftest)

All digests are computed via s9lib (the single source of truth shared with the bundle
validator), so a valid record/bundle is self-consistent and an adversarial one fails
exactly the intended check. No clock, no PRNG. ASCII only.
"""
import copy
import hashlib
import json
import os

import s9lib
from s9lib import canonical, set_digest

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures", "v2")


def h(label):
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def write(sub, name, obj):
    d = os.path.join(FX, sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write(canonical(obj) + "\n")


def with_digest(kind, rec):
    """Fill in the domain-separated identity field for a record via s9lib."""
    field, builder = s9lib.DIGEST_BUILDERS[kind]
    rec[field] = builder(rec)
    return rec


def chunks(label, total, chunk_bytes):
    out, off, i = [], 0, 0
    while off < total:
        n = min(chunk_bytes, total - off)
        out.append({"index": i, "offset": off, "bytes": n, "sha256": h(f"{label}:c{i}")})
        off += n
        i += 1
    return out


# ---- shared content identities ----
MV = h("model:gemma4-12b-f16")
SRC = h("source:gemma4-12b-f16.gguf")
GRAPH = h("graph:gemma4")
BUILD_HTP = h("build:htp-v81")
BUILD_GPU = h("build:opencl-a840")
METRIC = h("metric:pass")
PROFILE1 = h("profile:isl1")
PROFILE2 = h("profile:isl2")

SEGA_H = h("seg:blk0")
SEGB_H = h("seg:blk1")
D1 = set_digest([SEGA_H])          # weight-set 1 content digest
D2 = set_digest([SEGB_H])          # weight-set 2 content digest

SHAPE = {"input_dtype": "f16", "output_dtype": "f16", "max_input_bytes": 983040,
         "max_output_bytes": 983040, "max_batch": 8, "max_context": 4096}
PAY_EMPTY = h("payload:shared_canonical_empty")


def build_records():
    R = {}

    # ---- weight segments + atomic weight sets ----
    R["SEGa"] = {"schema_version": 2, "kind": "weight_segment", "segment_id": "seg-blk0",
                 "model_id": "gemma4-12b-f16", "model_version": MV, "logical_name": "blk.0.weight",
                 "file": "op15/blk0.gguf", "bytes": 400, "sha256": SEGA_H, "weight_set_id": "ws1",
                 "layer_range": {"start": 0, "end": 1, "n_layer_total": 2}, "chunk_bytes": 256,
                 "chunks": chunks("seg:blk0", 400, 256)}
    R["SEGb"] = {"schema_version": 2, "kind": "weight_segment", "segment_id": "seg-blk1",
                 "model_id": "gemma4-12b-f16", "model_version": MV, "logical_name": "blk.1.weight",
                 "file": "op15/blk1.gguf", "bytes": 300, "sha256": SEGB_H, "weight_set_id": "ws2",
                 "layer_range": {"start": 1, "end": 2, "n_layer_total": 2}, "chunk_bytes": 256,
                 "chunks": chunks("seg:blk1", 300, 256)}
    R["WS1"] = {"schema_version": 2, "kind": "weight_set", "weight_set_id": "ws1",
                "model_id": "gemma4-12b-f16", "model_version": MV, "layout_version": 3,
                "arch": "gemma4", "dtype": "f16", "layer_range": {"start": 0, "end": 1, "n_layer_total": 2},
                "segments": [{"segment_id": "seg-blk0", "sha256": SEGA_H, "bytes": 400}],
                "set_digest": D1, "total_bytes": 400, "atomicity": "all_or_none"}
    R["WS2"] = {"schema_version": 2, "kind": "weight_set", "weight_set_id": "ws2",
                "model_id": "gemma4-12b-f16", "model_version": MV, "layout_version": 3,
                "arch": "gemma4", "dtype": "f16", "layer_range": {"start": 1, "end": 2, "n_layer_total": 2},
                "segments": [{"segment_id": "seg-blk1", "sha256": SEGB_H, "bytes": 300}],
                "set_digest": D2, "total_bytes": 300, "atomicity": "all_or_none"}

    # ---- model manifest (contiguous partition over [0,2)) ----
    R["MM"] = {"schema_version": 2, "kind": "model_manifest", "model_id": "gemma4-12b-f16",
               "model_version": MV, "source_sha256": SRC, "arch": "gemma4", "n_layer_total": 2,
               "dtype": "f16", "layout_version": 3, "graph_hash": GRAPH,
               "partial_load_supported": True, "coverage_policy": "contiguous_partition",
               "weight_sets": [
                   {"weight_set_id": "ws1", "set_digest": D1, "total_bytes": 400,
                    "layer_range": {"start": 0, "end": 1, "n_layer_total": 2}},
                   {"weight_set_id": "ws2", "set_digest": D2, "total_bytes": 300,
                    "layer_range": {"start": 1, "end": 2, "n_layer_total": 2}}],
               "required_backends": [{"backend": "htp", "backend_build": BUILD_HTP, "min_soc": "op12"}],
               "compatible_soc": ["op12", "op15"], "resident_ram_estimate_bytes": 700,
               "warmup_cmd": "warm --sentinel"}

    # ---- canonical allocations (one byte-charge each; alias + lease pinned) ----
    R["ALLOC1"] = with_digest("canonical_allocation", {
        "schema_version": 2, "kind": "canonical_allocation", "allocation_id": "alloc-ws1",
        "device_id": "op15", "soc": "op15", "weight_set_id": "ws1", "weight_set_digest": D1,
        "dtype": "f16", "layout_version": 3, "boot_epoch": 7, "residency_generation": 2,
        "canonical_bytes": 400, "alias_refcount": 1, "alias_prepared_image_ids": ["pi1"],
        "lease_refcount": 1, "reclaimable": False})
    R["ALLOC2"] = with_digest("canonical_allocation", {
        "schema_version": 2, "kind": "canonical_allocation", "allocation_id": "alloc-ws2",
        "device_id": "op15", "soc": "op15", "weight_set_id": "ws2", "weight_set_digest": D2,
        "dtype": "f16", "layout_version": 3, "boot_epoch": 7, "residency_generation": 2,
        "canonical_bytes": 300, "alias_refcount": 1, "alias_prepared_image_ids": ["pi2"],
        "lease_refcount": 1, "reclaimable": False})

    # ---- prepared images (htp linear, shared canonical base) ----
    R["PI1"] = with_digest("prepared_image", {
        "schema_version": 2, "kind": "prepared_image", "prepared_image_id": "pi1",
        "image_class": "htp_linear", "backend": "htp", "preparation_algorithm": "htp_repack.v1",
        "model_version": MV, "tensor_digest": D1, "graph_hash": GRAPH, "backend_build": BUILD_HTP,
        "soc": "op15", "arch": "gemma4", "layout_version": 3, "boot_epoch": 7,
        "residency_generation": 2, "source_allocation_id": "alloc-ws1", "source_weight_set_id": "ws1",
        "source_weight_set_digest": D1, "derived_bytes": 0, "sharing_mode": "shared_canonical",
        "derived_payload_sha256": PAY_EMPTY})
    R["PI2"] = with_digest("prepared_image", {
        "schema_version": 2, "kind": "prepared_image", "prepared_image_id": "pi2",
        "image_class": "htp_linear", "backend": "htp", "preparation_algorithm": "htp_repack.v1",
        "model_version": MV, "tensor_digest": D2, "graph_hash": GRAPH, "backend_build": BUILD_HTP,
        "soc": "op15", "arch": "gemma4", "layout_version": 3, "boot_epoch": 7,
        "residency_generation": 2, "source_allocation_id": "alloc-ws2", "source_weight_set_id": "ws2",
        "source_weight_set_digest": D2, "derived_bytes": 0, "sharing_mode": "shared_canonical",
        "derived_payload_sha256": PAY_EMPTY})
    # a gpu xmem private image (schema-fixture coverage of the private_derived branch)
    R["PIgpu"] = with_digest("prepared_image", {
        "schema_version": 2, "kind": "prepared_image", "prepared_image_id": "pi-gpu",
        "image_class": "gpu_xmem_prepacked", "backend": "gpu", "preparation_algorithm": "gpu_xmem_os8.v1",
        "model_version": MV, "tensor_digest": D1, "graph_hash": GRAPH, "backend_build": BUILD_GPU,
        "soc": "op15", "arch": "gemma4", "layout_version": 3, "boot_epoch": 7,
        "residency_generation": 2, "source_allocation_id": "alloc-ws1", "source_weight_set_id": "ws1",
        "source_weight_set_digest": D1, "derived_bytes": 400, "sharing_mode": "private_derived",
        "derived_payload_sha256": h("payload:pi-gpu")})

    # ---- correctness certificates ----
    R["CC1"] = with_digest("correctness_certificate", {
        "schema_version": 2, "kind": "correctness_certificate", "correctness_id": "cc1",
        "island_id": "isl1", "backend": "htp", "backend_build": BUILD_HTP,
        "required_kernel_path": "hmx.decode.fa_off", "profile_row_digest": PROFILE1,
        "shape_envelope": copy.deepcopy(SHAPE), "verdict": "pass", "metric_digest": METRIC,
        "rel_l2_micro": 3300})
    R["CC2"] = with_digest("correctness_certificate", {
        "schema_version": 2, "kind": "correctness_certificate", "correctness_id": "cc2",
        "island_id": "isl2", "backend": "htp", "backend_build": BUILD_HTP,
        "required_kernel_path": "hmx.decode.fa_off", "profile_row_digest": PROFILE2,
        "shape_envelope": copy.deepcopy(SHAPE), "verdict": "pass", "metric_digest": METRIC,
        "rel_l2_micro": 3300})

    # ---- islands ----
    R["ISL1"] = with_digest("island_executable", {
        "schema_version": 2, "kind": "island_executable", "island_id": "isl1",
        "service_class": "decode", "model_id": "gemma4-12b-f16", "model_version": MV,
        "graph_hash": GRAPH, "backend": "htp",
        "layer_range": {"start": 0, "end": 1, "n_layer_total": 2},
        "required_weight_set_ids": ["ws1"], "required_weight_set_digests": [D1],
        "required_prepared_image_ids": ["pi1"], "required_prepared_image_digests": [R["PI1"]["derived_image_digest"]],
        "io_schema": {"input_dtype": "f16", "output_dtype": "f16", "max_input_bytes": 983040, "max_output_bytes": 983040},
        "state_policy": "sticky", "correctness_id": "cc1", "correctness_digest": R["CC1"]["correctness_digest"],
        "required_kernel_path": "hmx.decode.fa_off", "fallback": {"kind": "server", "available": True}})
    R["ISL2"] = with_digest("island_executable", {
        "schema_version": 2, "kind": "island_executable", "island_id": "isl2",
        "service_class": "decode", "model_id": "gemma4-12b-f16", "model_version": MV,
        "graph_hash": GRAPH, "backend": "htp",
        "layer_range": {"start": 0, "end": 2, "n_layer_total": 2},
        "required_weight_set_ids": ["ws1", "ws2"], "required_weight_set_digests": [D1, D2],
        "required_prepared_image_ids": ["pi1", "pi2"],
        "required_prepared_image_digests": [R["PI1"]["derived_image_digest"], R["PI2"]["derived_image_digest"]],
        "io_schema": {"input_dtype": "f16", "output_dtype": "f16", "max_input_bytes": 983040, "max_output_bytes": 983040},
        "state_policy": "sticky", "correctness_id": "cc2", "correctness_digest": R["CC2"]["correctness_digest"],
        "required_kernel_path": "hmx.decode.fa_off", "fallback": {"kind": "server", "available": True}})

    # ---- ready certificates ----
    R["RC1"] = with_digest("ready_certificate", {
        "schema_version": 2, "kind": "ready_certificate", "cert_id": "rc1", "device_id": "op15",
        "soc": "op15", "boot_epoch": 7, "residency_generation": 2, "backend": "htp",
        "state": "READY_HTP", "weight_set_id": "ws1", "weight_set_digest": D1,
        "prepared_image_id": "pi1", "prepared_image_digest": R["PI1"]["derived_image_digest"],
        "correctness_id": "cc1", "correctness_digest": R["CC1"]["correctness_digest"],
        "correctness": {"verdict": "pass", "metric_digest": METRIC}, "warmup_passed": True,
        "physical_bytes": {"canonical": 400, "derived": 0, "scratch": 100,
                           "activations_reserved": 40, "state_reserved": 60, "total": 600},
        "free_ram_after_bytes": 9400, "issued_receiver_ts_us": 1000000})
    R["RC2"] = with_digest("ready_certificate", {
        "schema_version": 2, "kind": "ready_certificate", "cert_id": "rc2", "device_id": "op15",
        "soc": "op15", "boot_epoch": 7, "residency_generation": 2, "backend": "htp",
        "state": "READY_HTP", "weight_set_id": "ws2", "weight_set_digest": D2,
        "prepared_image_id": "pi2", "prepared_image_digest": R["PI2"]["derived_image_digest"],
        "correctness_id": "cc2", "correctness_digest": R["CC2"]["correctness_digest"],
        "correctness": {"verdict": "pass", "metric_digest": METRIC}, "warmup_passed": True,
        "physical_bytes": {"canonical": 300, "derived": 0, "scratch": 100,
                           "activations_reserved": 40, "state_reserved": 60, "total": 500},
        "free_ram_after_bytes": 9500, "issued_receiver_ts_us": 1000000})

    # ---- residency leases ----
    R["RL1"] = with_digest("residency_lease", {
        "schema_version": 2, "kind": "residency_lease", "residency_lease_id": "rl1",
        "device_id": "op15", "backend": "htp", "weight_set_id": "ws1", "weight_set_digest": D1,
        "source_allocation_id": "alloc-ws1", "model_id": "gemma4-12b-f16", "model_version": MV,
        "boot_epoch": 7, "residency_generation": 2, "share_registry_generation": 5,
        "ready_certificate_id": "rc1", "ready_certificate_digest": R["RC1"]["ready_certificate_digest"],
        "state": "LEASED", "horizon": {"start_us": 1000000, "end_us": 61000000},
        "min_hold_us": 30000000, "reserved_bytes": {"weights": 400, "derived": 0, "scratch": 100}})
    R["RL2"] = with_digest("residency_lease", {
        "schema_version": 2, "kind": "residency_lease", "residency_lease_id": "rl2",
        "device_id": "op15", "backend": "htp", "weight_set_id": "ws2", "weight_set_digest": D2,
        "source_allocation_id": "alloc-ws2", "model_id": "gemma4-12b-f16", "model_version": MV,
        "boot_epoch": 7, "residency_generation": 2, "share_registry_generation": 5,
        "ready_certificate_id": "rc2", "ready_certificate_digest": R["RC2"]["ready_certificate_digest"],
        "state": "LEASED", "horizon": {"start_us": 1000000, "end_us": 61000000},
        "min_hold_us": 30000000, "reserved_bytes": {"weights": 300, "derived": 0, "scratch": 100}})

    # ---- state lease ----
    R["SL1"] = with_digest("state_lease", {
        "schema_version": 2, "kind": "state_lease", "state_lease_id": "sl1", "request_id": "req-1",
        "device_id": "op15", "backend": "htp", "island_id": "isl1", "boot_epoch": 7,
        "route_epoch": 3, "lease_epoch": 4, "seq_slot_epoch": 1,
        "depends_on_residency_lease_id": "rl1", "depends_on_residency_generation": 2,
        "state_policy": "sticky", "reserved_state_bytes": 4096, "reserved_activation_bytes": 983040,
        "in_flight_mutation_seq": 0})

    # ---- dispatch decision (single-tuple DISPATCH) ----
    tup1 = {"weight_set_id": "ws1", "prepared_image_id": "pi1", "ready_certificate_id": "rc1",
            "residency_lease_id": "rl1"}
    R["DD1"] = {"schema_version": 2, "kind": "dispatch_decision", "request_id": "req-1",
                "island_id": "isl1", "device_id": "op15", "backend": "htp", "route_epoch": 3,
                "required_tuples": [dict(tup1)], "satisfied_tuples": [dict(tup1)], "state_lease_id": "sl1",
                "epoch_match": {"boot": True, "residency": True, "route": True, "state": True},
                "credits": {"weights_ok": True, "derived_ok": True, "scratch_ok": True,
                            "activations_ok": True, "state_ok": True},
                "correctness_verdict": "pass", "hard_gate_failures": [], "verdict": "DISPATCH",
                "reason_code": "dispatch_ok"}

    # ---- transfer tickets ----
    R["TT1"] = with_digest("transfer_ticket", {
        "schema_version": 2, "kind": "transfer_ticket", "ticket_id": "tt1",
        "idempotency_key": "idem-tt1", "device_id": "op15", "channel": "bulk",
        "priority_class": "bulk_weight_background", "model_id": "gemma4-12b-f16", "model_version": MV,
        "weight_set_id": "ws1", "segment_id": "seg-blk0", "expected_sha256": SEGA_H,
        "byte_range": {"offset": 0, "length": 400}, "chunk_range": {"first": 0, "last": 1},
        "resumable_from_verified_offset": 0, "resume_partial_sha256": None, "credits_bytes": 1048576,
        "deadline_us": None, "issued_boot_epoch": 7, "issued_residency_generation": 2})
    R["TT2"] = with_digest("transfer_ticket", {
        "schema_version": 2, "kind": "transfer_ticket", "ticket_id": "tt2",
        "idempotency_key": "idem-tt2", "device_id": "op15", "channel": "bulk",
        "priority_class": "bulk_weight_background", "model_id": "gemma4-12b-f16", "model_version": MV,
        "weight_set_id": "ws1", "segment_id": "seg-blk0", "expected_sha256": SEGA_H,
        "byte_range": {"offset": 256, "length": 144}, "chunk_range": {"first": 1, "last": 1},
        "resumable_from_verified_offset": 256, "resume_partial_sha256": h("resume:seg-blk0:256"),
        "credits_bytes": 1048576, "deadline_us": None, "issued_boot_epoch": 7,
        "issued_residency_generation": 2})

    # ---- transport frames ----
    R["TFbulk"] = {"schema_version": 2, "kind": "transport_frame", "magic": "S9WR",
                   "protocol_version": 1, "channel": "bulk", "msg_type": "BULK_CHUNK",
                   "priority_class": "bulk_weight_background", "header_len": 64, "payload_bytes": 256,
                   "header_crc32": 305419896, "payload_sha256": h("seg:blk0:c0"), "request_id": "tt1",
                   "batch_id": None, "seq": 5, "idempotency_key": "idem-chunk-0", "boot_epoch": 7,
                   "residency_epoch": 2, "route_epoch": None, "state_epoch": None, "deadline_us": None,
                   "cancellable": True, "bulk_binding": {"ticket_id": "tt1", "segment_id": "seg-blk0",
                   "chunk_index": 0, "chunk_offset": 0, "chunk_length": 256, "chunk_sha256": h("seg:blk0:c0")}}
    R["TFexec"] = {"schema_version": 2, "kind": "transport_frame", "magic": "S9WR",
                   "protocol_version": 1, "channel": "control", "msg_type": "EXECUTE",
                   "priority_class": "activation", "header_len": 64, "payload_bytes": 4096,
                   "header_crc32": 305419896, "payload_sha256": h("exec:req-1"), "request_id": "req-1",
                   "batch_id": 10, "seq": 100, "idempotency_key": "idem-exec-1", "boot_epoch": 7,
                   "residency_epoch": 2, "route_epoch": 3, "state_epoch": 1, "deadline_us": 2000000,
                   "cancellable": False, "bulk_binding": None}

    # ---- device inventory ----
    R["DI"] = {"schema_version": 2, "kind": "device_inventory", "device_id": "op15", "soc": "op15",
               "boot_epoch": 7, "status_seq": 42, "backends": ["htp", "gpu", "cpu"],
               "ufs": {"total_bytes": 128000000000, "free_bytes": 60000000000},
               "lpddr": {"total_bytes": 10000},
               "physical_byte_accounting": {"weights_resident": 700, "derived_images": 0, "scratch": 200,
                                            "activations": 40, "mutable_state": 60, "free": 9000},
               "thermal": {"temp_milli_c": 41000, "slope_milli_c_per_s": 120, "eligible": True},
               "link": {"contention_domain": "op15-bus008", "h2d_goodput_bytes_per_s": 274614277,
                        "d2h_goodput_bytes_per_s": 237813760, "rtt_us": 500, "measured": True},
               "accepting": True, "draining": False, "stale": False, "receiver_ts_us": 1000000}
    return R


# schema-fixture kind -> schema path (v2)
SCHEMA = {k: f"schemas/v2/{v}" for k, v in s9lib.V2_KINDS.items()}


def emit_schema_fixtures(R):
    valid, invalid = [], []

    def v(name, kind, obj):
        write("valid", name, obj)
        valid.append((f"valid/{name}", SCHEMA[kind]))

    def iv(name, kind, obj):
        write("invalid", name, obj)
        invalid.append((f"invalid/{name}", SCHEMA[kind]))

    # one valid record per kind
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
    v("transfer_ticket_fresh.json", "transfer_ticket", R["TT1"])
    v("transfer_ticket_resume.json", "transfer_ticket", R["TT2"])
    v("transport_frame_bulk.json", "transport_frame", R["TFbulk"])
    v("transport_frame_exec.json", "transport_frame", R["TFexec"])
    v("device_inventory.json", "device_inventory", R["DI"])

    # schema-level adversarials (each violates the SCHEMA, independent of digests)
    iv("canonical_allocation.reclaimable_while_aliased.json", "canonical_allocation",
       {**R["ALLOC1"], "reclaimable": True})                       # alias_refcount>=1 -> must be false
    iv("prepared_image.shared_nonzero_bytes.json", "prepared_image",
       {**R["PI1"], "derived_bytes": 400})                         # shared_canonical -> derived_bytes const 0
    iv("prepared_image.private_zero_bytes.json", "prepared_image",
       {**R["PIgpu"], "derived_bytes": 0})                         # private_derived -> derived_bytes>=1
    iv("correctness_certificate.verdict_fail.json", "correctness_certificate",
       {**R["CC1"], "verdict": "fail"})                            # verdict const pass
    iv("ready_certificate.warmup_false.json", "ready_certificate",
       {**R["RC1"], "warmup_passed": False})                       # warmup_passed const true
    iv("ready_certificate.backend_mismatch.json", "ready_certificate",
       {**R["RC1"], "backend": "gpu"})                             # READY_HTP -> backend htp
    iv("model_manifest.partial_false_multi.json", "model_manifest",
       {**R["MM"], "partial_load_supported": False})               # false -> maxItems 1 (has 2 sets)
    iv("state_lease.stateless_nonzero.json", "state_lease",
       {**R["SL1"], "state_policy": "stateless"})                  # stateless -> reserved_state const 0 (has 4096)
    iv("transfer_ticket.resume_no_prefix.json", "transfer_ticket",
       {**R["TT2"], "resume_partial_sha256": None})                # nonzero resume -> prefix digest required
    iv("transport_frame.exec_no_hash.json", "transport_frame",
       {**R["TFexec"], "payload_sha256": None})                    # EXECUTE -> payload_sha256 required
    iv("transport_frame.bulk_no_binding.json", "transport_frame",
       {**R["TFbulk"], "bulk_binding": None})                      # bulk channel -> bulk_binding required
    iv("dispatch_decision.dispatch_empty_tuples.json", "dispatch_decision",
       {**R["DD1"], "required_tuples": [], "satisfied_tuples": []})  # DISPATCH -> minItems 1
    iv("dispatch_decision.dispatch_stale.json", "dispatch_decision",
       {**R["DD1"], "epoch_match": {"boot": True, "residency": False, "route": True, "state": True}})
    iv("island_executable.bad_service.json", "island_executable",
       {**R["ISL1"], "service_class": "teleport"})
    iv("weight_set.bad_atomicity.json", "weight_set", {**R["WS1"], "atomicity": "best_effort"})
    iv("prepared_image.oversize_id.json", "prepared_image",
       {**R["PI1"], "prepared_image_id": "x" * 300})               # maxLength 256

    idx = ([{"file": f, "schema": s, "expect": "valid"} for f, s in valid] +
           [{"file": f, "schema": s, "expect": "invalid"} for f, s in invalid])
    with open(os.path.join(FX, "index.json"), "w") as f:
        f.write(json.dumps(idx, indent=2) + "\n")
    return len(valid), len(invalid)


def bundle(bid, records):
    return {"schema_version": 2, "kind": "bundle", "bundle_version": 2, "bundle_id": bid,
            "records": records}


def emit_bundle_fixtures(R):
    valid, invalid = [], []

    def bv(name, obj):
        write(os.path.join("bundles", "valid"), name, obj)
        valid.append(f"valid/{name}")

    def bi(name, obj):
        write(os.path.join("bundles", "invalid"), name, obj)
        invalid.append(f"invalid/{name}")

    core = [R["MM"], R["SEGa"], R["SEGb"], R["WS1"], R["WS2"], R["ALLOC1"], R["ALLOC2"],
            R["PI1"], R["PI2"], R["CC1"], R["ISL1"], R["RC1"], R["RL1"], R["SL1"], R["DD1"],
            R["TT1"], R["TT2"], R["TFbulk"], R["TFexec"], R["DI"]]
    bv("dispatchable.json", bundle("b-dispatchable", copy.deepcopy(core)))

    # --- cross-record / semantic adversarials (each breaks exactly one bundle rule) ---
    # 1. unknown record kind
    b = copy.deepcopy(core)
    bad = copy.deepcopy(R["DI"]); bad["kind"] = "teleporter"
    b.append(bad)
    bi("unknown_kind.json", bundle("b-unknown-kind", b))

    # 2. unknown schema_version for a known kind
    b = copy.deepcopy(core); b[0] = {**copy.deepcopy(R["MM"]), "schema_version": 99}
    bi("unknown_version.json", bundle("b-unknown-version", b))

    # 3. prepared-image relabel (image_class changed, digest not recomputed) -> digest mismatch
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "prepared_image" and rec["prepared_image_id"] == "pi1":
            b[i] = {**copy.deepcopy(R["PI1"]), "image_class": "gpu_plain"}  # digest now stale
    bi("prepared_image_relabel.json", bundle("b-relabel", b))

    # 4. prepared-image payload corruption (derived_payload_sha256 changed) -> digest mismatch
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "prepared_image" and rec["prepared_image_id"] == "pi1":
            b[i] = {**copy.deepcopy(R["PI1"]), "derived_payload_sha256": h("payload:tampered")}
    bi("prepared_image_corrupt.json", bundle("b-corrupt", b))

    # 5. two required weight sets but only one ready certificate -> dispatch tuple mismatch
    #    ISL2 requires ws1+ws2; DD claims DISPATCH but only supplies ws1's tuple.
    tup1 = {"weight_set_id": "ws1", "prepared_image_id": "pi1", "ready_certificate_id": "rc1",
            "residency_lease_id": "rl1"}
    tup2 = {"weight_set_id": "ws2", "prepared_image_id": "pi2", "ready_certificate_id": "rc2",
            "residency_lease_id": "rl2"}
    dd2 = {**copy.deepcopy(R["DD1"]), "request_id": "req-2", "island_id": "isl2",
           "required_tuples": [dict(tup1), dict(tup2)], "satisfied_tuples": [dict(tup1)],
           "state_lease_id": None}
    b = [R["MM"], R["SEGa"], R["SEGb"], R["WS1"], R["WS2"], R["ALLOC1"], R["ALLOC2"], R["PI1"], R["PI2"],
         R["CC1"], R["CC2"], R["ISL1"], R["ISL2"], R["RC1"], R["RL1"], dd2]
    bi("two_sets_one_cert.json", bundle("b-two-sets-one-cert", copy.deepcopy(b)))

    # 6. missing referenced record: DD references rl1 but the residency_lease is absent
    b = [r for r in copy.deepcopy(core) if not (r.get("kind") == "residency_lease")]
    bi("missing_residency_lease.json", bundle("b-missing-record", b))

    # 7. ledger partition wrong on the device inventory
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "device_inventory":
            di = copy.deepcopy(R["DI"]); di["physical_byte_accounting"]["free"] += 1
            b[i] = di
    bi("ledger_mismatch.json", bundle("b-ledger", b))

    # 8. partial-range overlap under a contiguous_partition policy
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "model_manifest":
            mm = copy.deepcopy(R["MM"])
            mm["weight_sets"][1]["layer_range"] = {"start": 0, "end": 1, "n_layer_total": 2}  # overlaps ws1
            b[i] = mm
    bi("range_overlap.json", bundle("b-range-overlap", b))

    # 9. weight_set total_bytes disagrees with the manifest's per-set total
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "weight_set" and rec["weight_set_id"] == "ws1":
            b[i] = {**copy.deepcopy(R["WS1"]), "total_bytes": 400, "segments":
                    [{"segment_id": "seg-blk0", "sha256": SEGA_H, "bytes": 400}]}
    # (this one is actually consistent; craft a real mismatch instead:)
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "model_manifest":
            mm = copy.deepcopy(R["MM"]); mm["weight_sets"][0]["total_bytes"] = 401
            b[i] = mm
    bi("total_mismatch.json", bundle("b-total-mismatch", b))

    # 10. canonical allocation alias_refcount disagrees with alias_prepared_image_ids length
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "canonical_allocation" and rec["allocation_id"] == "alloc-ws1":
            al = copy.deepcopy(R["ALLOC1"]); al["alias_refcount"] = 2   # list still has 1
            al["allocation_digest"] = s9lib.alloc_digest(al)           # digest stays consistent
            b[i] = al
    bi("alloc_refcount_mismatch.json", bundle("b-alloc-refcount", b))

    # 11. resume ticket byte_range.offset != resumable_from_verified_offset
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "transfer_ticket" and rec["ticket_id"] == "tt2":
            tt = copy.deepcopy(R["TT2"]); tt["byte_range"] = {"offset": 128, "length": 144}
            tt["transfer_ticket_digest"] = s9lib.transfer_ticket_digest(tt)
            b[i] = tt
    bi("resume_offset_mismatch.json", bundle("b-resume-offset", b))

    # 12. bulk frame binding disagrees with the segment's chunk (chunk_sha256 wrong)
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "transport_frame" and rec.get("channel") == "bulk":
            tf = copy.deepcopy(R["TFbulk"])
            tf["bulk_binding"]["chunk_sha256"] = h("wrong:chunk")
            tf["payload_sha256"] = h("wrong:chunk")
            b[i] = tf
    bi("bulk_binding_mismatch.json", bundle("b-bulk-binding", b))

    # 13. correctness digest does not match island's referenced correctness_digest
    b = copy.deepcopy(core)
    for i, rec in enumerate(b):
        if rec.get("kind") == "correctness_certificate":
            cc = copy.deepcopy(R["CC1"]); cc["required_kernel_path"] = "hmx.decode.fa_on"  # digest stale
            b[i] = cc
    bi("correctness_relabel.json", bundle("b-correctness", b))

    # 14. duplicate JSON key (raw text; json.dumps cannot emit one) -> parse-time reject
    raw = canonical(bundle("b-dupkey", copy.deepcopy(core)))
    raw = raw.replace('"bundle_version":2,', '"bundle_version":2,"bundle_version":2,', 1)
    with open(os.path.join(FX, "bundles", "invalid", "dup_key.json"), "w") as f:
        f.write(raw + "\n")
    invalid.append("invalid/dup_key.json")

    idx = ([{"file": f, "expect": "valid"} for f in valid] +
           [{"file": f, "expect": "invalid"} for f in invalid])
    with open(os.path.join(FX, "bundles", "index.json"), "w") as f:
        f.write(json.dumps(idx, indent=2) + "\n")
    return len(valid), len(invalid)


def main():
    R = build_records()
    sv, si = emit_schema_fixtures(R)
    bvn, bin_ = emit_bundle_fixtures(R)
    print(f"v2 schema fixtures: {sv} valid + {si} invalid = {sv + si}")
    print(f"v2 bundle fixtures: {bvn} valid + {bin_} invalid = {bvn + bin_}")


if __name__ == "__main__":
    main()
