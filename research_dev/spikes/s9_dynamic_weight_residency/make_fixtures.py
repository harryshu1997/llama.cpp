#!/usr/bin/env python3
"""S9 fixture generator (deterministic).

Emits every schema fixture (fixtures/{valid,invalid}) and every semantic fixture
(fixtures/semantic/{valid,invalid}) plus their index files. Digest-dependent
fixtures (weight_set.set_digest, prepared_image.derived_image_digest, ledger sums,
chunk tiling) are COMPUTED so valid fixtures pass validate_manifests.py and
adversarial ones fail exactly one check. Re-runnable and deterministic (no clock,
no PRNG). ASCII only.
"""
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def h(label):
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def sha_over(text):
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_digest(seg_hashes):
    return sha_over("\n".join(sorted(seg_hashes)))


def derived_digest(model_version, tensor_digest, graph_hash, backend_build, soc, arch, layout_version, boot_epoch, residency_generation):
    fields = {
        "arch": arch, "backend_build": backend_build, "boot_epoch": boot_epoch,
        "graph_hash": graph_hash, "layout_version": layout_version,
        "model_version": model_version, "residency_generation": residency_generation,
        "soc": soc, "tensor_digest": tensor_digest,
    }
    return sha_over(canonical(fields))


def chunks(segment_label, total, chunk_bytes):
    out, off, i = [], 0, 0
    while off < total:
        n = min(chunk_bytes, total - off)
        out.append({"index": i, "offset": off, "bytes": n, "sha256": h(f"{segment_label}:c{i}")})
        off += n
        i += 1
    return out


def write(sub, name, obj):
    d = os.path.join(FX, sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write(canonical(obj) + "\n")


# ---- shared identities ----
MV = h("model:gemma4-12b-f16")          # model_version
SRC = h("source:gemma4-12b-f16.gguf")
GRAPH = h("graph:gemma4")
BUILD_HTP = h("build:htp-v81")
BUILD_GPU = h("build:opencl-a840")
METRIC = h("metric:pass")

SEG_A = h("seg:blk45.weight")
SEG_B = h("seg:blk46.weight")
WS_SEGS = [SEG_A, SEG_B]
WS_DIGEST = set_digest(WS_SEGS)          # correct atomic set digest

DERIVED_HTP = derived_digest(MV, WS_DIGEST, GRAPH, BUILD_HTP, "op15", "gemma4", 3, 7, 2)
DERIVED_GPU = derived_digest(MV, WS_DIGEST, GRAPH, BUILD_GPU, "op15", "gemma4", 3, 7, 2)

valid, invalid = [], []   # (file, schema)


def v(name, schema, obj):
    write("valid", name, obj)
    valid.append((name, schema))


def iv(name, schema, obj):
    write("invalid", name, obj)
    invalid.append((name, schema))


# ============ WeightSegment ============
seg_valid = {
    "schema_version": 1, "segment_id": "seg-blk45", "model_id": "gemma4-12b-f16",
    "model_version": MV, "logical_name": "blk.45.weight", "file": "op15/blk45.gguf",
    "bytes": 300, "sha256": SEG_A,
    "layer_range": {"start": 45, "end": 46, "n_layer_total": 48},
    "chunk_bytes": 128, "chunks": chunks("seg:blk45.weight", 300, 128),
}
v("weight_segment.valid.json", "schemas/weight_segment.schema.json", seg_valid)
iv("weight_segment.zero_bytes.json", "schemas/weight_segment.schema.json",
   {**seg_valid, "bytes": 0})

# ============ WeightSet ============
ws_valid = {
    "schema_version": 1, "weight_set_id": "ws-op15-4546", "model_id": "gemma4-12b-f16",
    "model_version": MV, "layout_version": 3, "arch": "gemma4", "dtype": "f16",
    "layer_range": {"start": 45, "end": 47, "n_layer_total": 48},
    "segments": [
        {"segment_id": "seg-blk45", "sha256": SEG_A, "bytes": 300},
        {"segment_id": "seg-blk46", "sha256": SEG_B, "bytes": 200},
    ],
    "set_digest": WS_DIGEST, "total_bytes": 500, "atomicity": "all_or_none",
}
v("weight_set.valid.json", "schemas/weight_set.schema.json", ws_valid)
iv("weight_set.bad_atomicity.json", "schemas/weight_set.schema.json",
   {**ws_valid, "atomicity": "best_effort"})

# ============ ModelManifest ============
mm_valid = {
    "schema_version": 1, "model_id": "gemma4-12b-f16", "model_version": MV,
    "source_sha256": SRC, "arch": "gemma4", "n_layer_total": 48, "dtype": "f16",
    "layout_version": 3, "graph_hash": GRAPH, "partial_load_supported": True,
    "weight_sets": [
        {"weight_set_id": "ws-op15-4546", "set_digest": WS_DIGEST, "total_bytes": 500,
         "layer_range": {"start": 45, "end": 47, "n_layer_total": 48}},
        {"weight_set_id": "ws-op12-47", "set_digest": h("ws:op12-47"), "total_bytes": 250,
         "layer_range": {"start": 47, "end": 48, "n_layer_total": 48}},
    ],
    "required_backends": [
        {"backend": "htp", "backend_build": BUILD_HTP, "min_soc": "op12"},
        {"backend": "gpu", "backend_build": BUILD_GPU, "min_soc": "op15"},
    ],
    "compatible_soc": ["op12", "op15"],
    "resident_ram_estimate_bytes": 800, "warmup_cmd": "warm --sentinel",
}
v("model_manifest.valid.json", "schemas/model_manifest.schema.json", mm_valid)

mm_full = {
    "schema_version": 1, "model_id": "plain-7b-f16", "model_version": h("model:plain7b"),
    "source_sha256": h("src:plain7b"), "arch": "llama", "n_layer_total": 32, "dtype": "f16",
    "layout_version": 1, "graph_hash": h("graph:llama"), "partial_load_supported": False,
    "weight_sets": [
        {"weight_set_id": "ws-full", "set_digest": h("ws:full"), "total_bytes": 9000,
         "layer_range": {"start": 0, "end": 32, "n_layer_total": 32}},
    ],
    "required_backends": [{"backend": "htp", "backend_build": h("build:htp")}],
    "compatible_soc": ["op15"], "resident_ram_estimate_bytes": None, "warmup_cmd": "",
}
v("model_manifest.full.json", "schemas/model_manifest.schema.json", mm_full)
iv("model_manifest.unknown_soc.json", "schemas/model_manifest.schema.json",
   {**mm_valid, "compatible_soc": ["op12", "op99"]})
iv("model_manifest.missing_setdigest.json", "schemas/model_manifest.schema.json",
   {**mm_valid, "weight_sets": [{"weight_set_id": "ws-x", "total_bytes": 500,
                                 "layer_range": {"start": 45, "end": 47, "n_layer_total": 48}}]})

# ============ PreparedImage ============
pi_htp = {
    "schema_version": 1, "prepared_image_id": "pi-htp-4546", "image_class": "htp_linear",
    "backend": "htp", "model_version": MV, "tensor_digest": WS_DIGEST, "graph_hash": GRAPH,
    "backend_build": BUILD_HTP, "soc": "op15", "arch": "gemma4", "layout_version": 3,
    "boot_epoch": 7, "residency_generation": 2, "derived_image_digest": DERIVED_HTP,
    "source_weight_set_id": "ws-op15-4546", "source_weight_set_digest": WS_DIGEST,
    "derived_bytes": 0, "shares_canonical_base": True,
}
v("prepared_image.htp.json", "schemas/prepared_image.schema.json", pi_htp)
pi_gpu = {
    "schema_version": 1, "prepared_image_id": "pi-gpu-4546", "image_class": "gpu_xmem_prepacked",
    "backend": "gpu", "model_version": MV, "tensor_digest": WS_DIGEST, "graph_hash": GRAPH,
    "backend_build": BUILD_GPU, "soc": "op15", "arch": "gemma4", "layout_version": 3,
    "boot_epoch": 7, "residency_generation": 2, "derived_image_digest": DERIVED_GPU,
    "source_weight_set_id": "ws-op15-4546", "source_weight_set_digest": WS_DIGEST,
    "derived_bytes": 500, "shares_canonical_base": False,
}
v("prepared_image.gpu_xmem.json", "schemas/prepared_image.schema.json", pi_gpu)
iv("prepared_image.bad_digest_pattern.json", "schemas/prepared_image.schema.json",
   {**pi_htp, "derived_image_digest": "not-a-sha"})

# ============ IslandExecutable ============
isl = {
    "schema_version": 1, "island_id": "isl-decode-op15", "service_class": "decode",
    "model_id": "gemma4-12b-f16", "model_version": MV, "graph_hash": GRAPH, "backend": "htp",
    "layer_range": {"start": 45, "end": 47, "n_layer_total": 48},
    "required_weight_sets": [{"weight_set_id": "ws-op15-4546", "set_digest": WS_DIGEST}],
    "required_prepared_images": [{"prepared_image_id": "pi-htp-4546", "derived_image_digest": DERIVED_HTP}],
    "io_schema": {"input_dtype": "f16", "output_dtype": "f16", "max_input_bytes": 983040, "max_output_bytes": 983040},
    "state_policy": "sticky",
    "correctness": {"verdict": "pass", "metric_digest": METRIC, "rel_l2_micro": 3300},
    "required_kernel_path": "hmx.decode.fa_off", "fallback": {"kind": "server", "available": True},
}
v("island_executable.valid.json", "schemas/island_executable.schema.json", isl)
iv("island_executable.bad_service.json", "schemas/island_executable.schema.json",
   {**isl, "service_class": "teleport"})

# ============ TransferTicket ============
tt = {
    "schema_version": 1, "ticket_id": "tt-1", "idempotency_key": "idem-tt-1",
    "device_id": "op15", "channel": "bulk", "priority_class": "bulk_weight_background",
    "model_id": "gemma4-12b-f16", "model_version": MV, "weight_set_id": "ws-op15-4546",
    "segment_id": "seg-blk45", "expected_sha256": SEG_A,
    "byte_range": {"offset": 0, "length": 300},
    "chunk_range": {"first": 0, "last": 2},
    "resumable_from_verified_offset": 0, "resume_partial_sha256": None,
    "credits_bytes": 1048576, "deadline_us": None,
    "issued_boot_epoch": 7, "issued_residency_generation": 2,
}
v("transfer_ticket.fresh.json", "schemas/transfer_ticket.schema.json", tt)
tt_resume = {**tt, "ticket_id": "tt-2", "idempotency_key": "idem-tt-2",
             "resumable_from_verified_offset": 128,
             "resume_partial_sha256": h("resume:seg-blk45:128"),
             "chunk_range": {"first": 1, "last": 2}}
v("transfer_ticket.resume.json", "schemas/transfer_ticket.schema.json", tt_resume)
iv("transfer_ticket.priority_wrong.json", "schemas/transfer_ticket.schema.json",
   {**tt, "priority_class": "activation"})
iv("transfer_ticket.extra_field.json", "schemas/transfer_ticket.schema.json",
   {**tt, "surprise": 1})

# ============ ReadyCertificate ============
rc_htp = {
    "schema_version": 1, "cert_id": "rc-htp-1", "device_id": "op15", "soc": "op15",
    "boot_epoch": 7, "residency_generation": 2, "backend": "htp", "state": "READY_HTP",
    "weight_set_id": "ws-op15-4546", "weight_set_digest": WS_DIGEST,
    "prepared_image_id": "pi-htp-4546", "prepared_image_digest": DERIVED_HTP,
    "correctness": {"verdict": "pass", "metric_digest": METRIC}, "warmup_passed": True,
    "physical_bytes": {"canonical": 500, "derived": 0, "scratch": 100,
                       "activations_reserved": 40, "state_reserved": 60, "total": 700},
    "free_ram_after_bytes": 9300, "issued_receiver_ts_us": 1000000,
}
v("ready_certificate.htp.json", "schemas/ready_certificate.schema.json", rc_htp)
rc_gpu = {**rc_htp, "cert_id": "rc-gpu-1", "backend": "gpu", "state": "READY_GPU",
          "prepared_image_id": "pi-gpu-4546", "prepared_image_digest": DERIVED_GPU,
          "physical_bytes": {"canonical": 500, "derived": 500, "scratch": 100,
                             "activations_reserved": 40, "state_reserved": 60, "total": 1200}}
v("ready_certificate.gpu.json", "schemas/ready_certificate.schema.json", rc_gpu)
iv("ready_certificate.warmup_false.json", "schemas/ready_certificate.schema.json",
   {**rc_htp, "warmup_passed": False})
iv("ready_certificate.verdict_fail.json", "schemas/ready_certificate.schema.json",
   {**rc_htp, "correctness": {"verdict": "fail", "metric_digest": METRIC}})
iv("ready_certificate.backend_mismatch.json", "schemas/ready_certificate.schema.json",
   {**rc_htp, "backend": "gpu"})   # READY_HTP but backend gpu
iv("ready_certificate.overflow.json", "schemas/ready_certificate.schema.json",
   {**rc_htp, "free_ram_after_bytes": 9007199254740992})

# ============ ResidencyLease ============
rl = {
    "schema_version": 1, "residency_lease_id": "rl-1", "device_id": "op15", "backend": "htp",
    "weight_set_id": "ws-op15-4546", "weight_set_digest": WS_DIGEST,
    "model_id": "gemma4-12b-f16", "model_version": MV, "boot_epoch": 7,
    "residency_generation": 2, "share_registry_generation": 5, "ready_certificate_id": "rc-htp-1",
    "ready_certificate_digest": h("cert:rc-htp-1"), "state": "LEASED",
    "horizon": {"start_us": 1000000, "end_us": 61000000}, "min_hold_us": 30000000,
    "reserved_bytes": {"weights": 500, "derived": 0, "scratch": 100},
}
v("residency_lease.valid.json", "schemas/residency_lease.schema.json", rl)
iv("residency_lease.bad_state.json", "schemas/residency_lease.schema.json",
   {**rl, "state": "HOVERING"})

# ============ StateLease ============
sl_stateless = {
    "schema_version": 1, "state_lease_id": "sl-0", "request_id": "req-0", "device_id": "op15",
    "backend": "gpu", "island_id": "isl-prefill", "boot_epoch": 7, "route_epoch": 3,
    "lease_epoch": 4, "seq_slot_epoch": 1, "depends_on_residency_lease_id": "rl-1",
    "depends_on_residency_generation": 2, "state_policy": "stateless",
    "reserved_state_bytes": 0, "reserved_activation_bytes": 983040, "in_flight_mutation_seq": 0,
}
v("state_lease.stateless.json", "schemas/state_lease.schema.json", sl_stateless)
sl_sticky = {**sl_stateless, "state_lease_id": "sl-1", "request_id": "req-1", "backend": "htp",
             "island_id": "isl-decode-op15", "state_policy": "sticky", "reserved_state_bytes": 4096}
v("state_lease.sticky.json", "schemas/state_lease.schema.json", sl_sticky)
iv("state_lease.stateless_nonzero.json", "schemas/state_lease.schema.json",
   {**sl_stateless, "reserved_state_bytes": 4096})
iv("state_lease.sticky_zero.json", "schemas/state_lease.schema.json",
   {**sl_sticky, "reserved_state_bytes": 0})
iv("state_lease.rebuildable_zero.json", "schemas/state_lease.schema.json",
   {**sl_sticky, "state_policy": "rebuildable", "reserved_state_bytes": 0})

# ============ DeviceInventory ============
di = {
    "schema_version": 1, "device_id": "op15", "soc": "op15", "boot_epoch": 7, "status_seq": 42,
    "backends": ["htp", "gpu", "cpu"],
    "ufs": {"total_bytes": 128000000000, "free_bytes": 60000000000},
    "lpddr": {"total_bytes": 10000},
    "physical_byte_accounting": {"weights_resident": 500, "derived_images": 500, "scratch": 100,
                                 "activations": 40, "mutable_state": 60, "free": 8800},
    "thermal": {"temp_milli_c": 41000, "slope_milli_c_per_s": 120, "eligible": True},
    "link": {"goodput_bytes_per_s": 104857600, "rtt_us": 500, "measured": False},
    "accepting": True, "draining": False, "stale": False, "receiver_ts_us": 1000000,
}
v("device_inventory.valid.json", "schemas/device_inventory.schema.json", di)
iv("device_inventory.unknown_soc.json", "schemas/device_inventory.schema.json",
   {**di, "soc": "op99"})

# ============ DispatchDecision ============
dd_ok = {
    "schema_version": 1, "request_id": "req-1", "island_id": "isl-decode-op15", "device_id": "op15",
    "backend": "htp", "route_epoch": 3, "required_weight_set_ids": ["ws-op15-4546"],
    "ready_certificate_ids": ["rc-htp-1"], "residency_lease_id": "rl-1", "state_lease_id": "sl-1",
    "epoch_match": {"boot": True, "residency": True, "route": True, "state": True},
    "credits": {"weights_ok": True, "derived_ok": True, "scratch_ok": True, "activations_ok": True, "state_ok": True},
    "correctness_verdict": "pass", "hard_gate_failures": [], "verdict": "DISPATCH", "reason_code": "dispatch_ok",
}
v("dispatch_decision.dispatch.json", "schemas/dispatch_decision.schema.json", dd_ok)
dd_fb = {
    "schema_version": 1, "request_id": "req-2", "island_id": "isl-decode-op15", "device_id": "op15",
    "backend": "cpu", "route_epoch": 3, "required_weight_set_ids": ["ws-op15-4546"],
    "ready_certificate_ids": [], "residency_lease_id": None, "state_lease_id": None,
    "epoch_match": {"boot": True, "residency": False, "route": True, "state": True},
    "credits": {"weights_ok": False, "derived_ok": True, "scratch_ok": True, "activations_ok": True, "state_ok": True},
    "correctness_verdict": "unknown", "hard_gate_failures": ["not_resident"],
    "verdict": "FALLBACK_SERVER", "reason_code": "not_resident",
}
v("dispatch_decision.fallback.json", "schemas/dispatch_decision.schema.json", dd_fb)
iv("dispatch_decision.dispatch_no_cert.json", "schemas/dispatch_decision.schema.json",
   {**dd_ok, "ready_certificate_ids": []})
iv("dispatch_decision.dispatch_stale.json", "schemas/dispatch_decision.schema.json",
   {**dd_ok, "epoch_match": {"boot": True, "residency": False, "route": True, "state": True}})
iv("dispatch_decision.dispatch_credit.json", "schemas/dispatch_decision.schema.json",
   {**dd_ok, "credits": {"weights_ok": True, "derived_ok": True, "scratch_ok": True, "activations_ok": True, "state_ok": False}})
iv("dispatch_decision.dispatch_gate.json", "schemas/dispatch_decision.schema.json",
   {**dd_ok, "hard_gate_failures": ["thermal"]})
iv("dispatch_decision.dispatch_bad_reason.json", "schemas/dispatch_decision.schema.json",
   {**dd_ok, "reason_code": "not_resident"})
iv("dispatch_decision.fallback_dispatch_ok.json", "schemas/dispatch_decision.schema.json",
   {**dd_fb, "reason_code": "dispatch_ok"})

# ============ TransportFrame ============
tf_ctrl = {
    "schema_version": 1, "magic": "S9WR", "protocol_version": 1, "channel": "control",
    "msg_type": "EXECUTE", "priority_class": "activation", "header_len": 64, "payload_bytes": 4096,
    "header_crc32": 305419896, "payload_sha256": None, "request_id": "req-1", "batch_id": 10,
    "seq": 100, "idempotency_key": "idem-exec-1", "boot_epoch": 7, "residency_epoch": 2,
    "route_epoch": 3, "state_epoch": 1, "deadline_us": 2000000, "cancellable": False,
}
v("transport_frame.control_execute.json", "schemas/transport_frame.schema.json", tf_ctrl)
tf_hb = {**tf_ctrl, "msg_type": "HEARTBEAT", "priority_class": "control", "payload_bytes": 0,
         "batch_id": None, "idempotency_key": "idem-hb-1", "deadline_us": None}
v("transport_frame.heartbeat.json", "schemas/transport_frame.schema.json", tf_hb)
tf_bulk = {
    "schema_version": 1, "magic": "S9WR", "protocol_version": 1, "channel": "bulk",
    "msg_type": "BULK_CHUNK", "priority_class": "bulk_weight_background", "header_len": 64,
    "payload_bytes": 1048576, "header_crc32": 4009750016, "payload_sha256": h("chunk:0"),
    "request_id": "tt-1", "batch_id": None, "seq": 5, "idempotency_key": "idem-chunk-0",
    "boot_epoch": 7, "residency_epoch": 2, "route_epoch": None, "state_epoch": None,
    "deadline_us": None, "cancellable": True,
}
v("transport_frame.bulk_chunk.json", "schemas/transport_frame.schema.json", tf_bulk)
iv("transport_frame.bad_magic.json", "schemas/transport_frame.schema.json",
   {**tf_bulk, "magic": "XXXX"})
iv("transport_frame.bulk_no_hash.json", "schemas/transport_frame.schema.json",
   {**tf_bulk, "payload_sha256": None})
iv("transport_frame.bulk_wrong_msg.json", "schemas/transport_frame.schema.json",
   {**tf_bulk, "msg_type": "READY"})
iv("transport_frame.oversize.json", "schemas/transport_frame.schema.json",
   {**tf_bulk, "payload_bytes": 16777217})
iv("transport_frame.control_oversize.json", "schemas/transport_frame.schema.json",
   {**tf_ctrl, "payload_bytes": 70000})
iv("transport_frame.execute_background.json", "schemas/transport_frame.schema.json",
   {**tf_ctrl, "priority_class": "bulk_weight_background"})

# ============ SimConfig (small valid) ============
sim_cfg = {
    "schema_version": 1, "config_id": "fx-min", "seed": 1, "horizon_us": 10000000,
    "link_goodput_sweep_bytes_per_s": [41943040, 104857600, 576716800],
    "usb_shared_controller_bytes_per_s": 625000000,
    "baselines": ["server_only", "relief_predictive", "clairvoyant"],
    "interference": {"htp_gpu_slowdown_permille": 1000, "transfer_compute_slowdown_permille": 1000, "measured": False},
    "server": {"gpu_lane_slots": 4, "hbm_credit_bytes": 1000000000,
               "per_class_gpu_us": {"decode": 2000, "prefill": 8000},
               "per_class_hbm_bytes": {"decode": 1000000, "prefill": 4000000}},
    "phones": [{"device_id": "op15", "soc": "op15", "boot_epoch": 7, "lpddr_total_bytes": 10000000000,
                "ufs_write_bytes_per_s": 800000000, "verify_bytes_per_s": 1500000000,
                "materialize_bytes_per_s": 6000000000, "prepare_htp_bytes_per_s": 4000000000,
                "prepare_gpu_bytes_per_s": 3000000000, "warmup_us": 50000, "htp_lane_slots": 1,
                "gpu_lane_slots": 1, "thermal_eligible": True}],
    "models": [{"model_id": "gemma4-12b-f16", "weight_set_id": "ws-op15-4546", "canonical_bytes": 900000000,
                "derived_bytes_htp": 0, "derived_bytes_gpu": 900000000, "scratch_bytes": 100000000,
                "partial_load_supported": True, "eligible_backends": ["htp", "gpu"], "phone_compute_us": 3000}],
    "workload": {"requests": [
        {"arrival_us": 0, "rank": 0, "request_id": "r0", "service_class": "decode", "model_id": "gemma4-12b-f16",
         "weight_set_id": "ws-op15-4546", "input_bytes": 983040, "output_bytes": 983040, "state_bytes": 4096, "deadline_us": None},
        {"arrival_us": 500000, "rank": 1, "request_id": "r1", "service_class": "decode", "model_id": "gemma4-12b-f16",
         "weight_set_id": "ws-op15-4546", "input_bytes": 983040, "output_bytes": 983040, "state_bytes": 4096, "deadline_us": None}]},
    "failure_schedule": [],
    "policy_params": {"reuse_horizon_us": 60000000, "min_hold_us": 30000000, "ttl_us": 45000000, "predictor_ewma_permille": 500},
}
v("sim_config.valid.json", "schemas/sim_config.schema.json", sim_cfg)
iv("sim_config.bad_baseline.json", "schemas/sim_config.schema.json",
   {**sim_cfg, "baselines": ["server_only", "teleport"]})

# ============ SimRunManifest (small valid) ============
srm = {
    "schema_version": 1, "run_id": "run-fx", "kind": "residency_sim", "fixtures_only": True,
    "config_id": "fx-min", "config_hash": h("cfg:fx-min"), "code_version": h("code:sim"), "seed": 1,
    "inputs": [{"role": "sim_config", "path": "fixtures/valid/sim_config.valid.json", "sha256": h("in:cfg")}],
    "goodput_results": [{"goodput_bytes_per_s": 104857600, "baselines": [
        {"policy": "server_only", "is_upper_bound": False, "is_diagnostic_losing": False, "completed": 2,
         "dispatched_to_phone": 0, "fell_back_to_server": 2, "weight_misses": 0, "server_gpu_us_used": 4000,
         "server_gpu_us_freed": 0, "transfer_bytes": 0, "prepare_us_total": 0, "evictions": 0,
         "deadline_hits": 0, "deadline_misses": 0, "completion_p50_us": 2000, "completion_p95_us": 2000,
         "causal_score_total": 0, "waited_for_weights": 0}]}],
    "deterministic_replay_sha256": h("replay:fx"), "gate_results": {},
}
v("sim_run_manifest.valid.json", "schemas/sim_run_manifest.schema.json", srm)
iv("sim_run_manifest.not_fixtures_only.json", "schemas/sim_run_manifest.schema.json",
   {**srm, "fixtures_only": False})

# ---- write schema-fixture index ----
index = ([{"file": f"valid/{n}", "schema": s, "expect": "valid"} for n, s in valid] +
         [{"file": f"invalid/{n}", "schema": s, "expect": "invalid"} for n, s in invalid])
with open(os.path.join(FX, "index.json"), "w") as f:
    f.write(json.dumps(index, indent=2) + "\n")

print(f"schema fixtures: {len(valid)} valid + {len(invalid)} invalid = {len(index)}")

# ============================================================
# Semantic fixtures (validate_manifests.py --selftest)
# ============================================================
sem_valid, sem_invalid = [], []


def sv(name, kind, obj):
    write(os.path.join("semantic", "valid"), name, obj)
    sem_valid.append((name, kind))


def si(name, kind, obj):
    write(os.path.join("semantic", "invalid"), name, obj)
    sem_invalid.append((name, kind))


# weight_segment tiling
sv("weight_segment.ok.json", "weight_segment", seg_valid)
si("weight_segment.gap.json", "weight_segment",
   {**seg_valid, "chunks": [{"index": 0, "offset": 0, "bytes": 128, "sha256": h("a")},
                            {"index": 1, "offset": 200, "bytes": 100, "sha256": h("b")}]})  # gap
si("weight_segment.sum.json", "weight_segment",
   {**seg_valid, "chunks": [{"index": 0, "offset": 0, "bytes": 128, "sha256": h("a")}]})  # sum 128 != 300
si("weight_segment.index.json", "weight_segment",
   {**seg_valid, "chunks": [{"index": 5, "offset": 0, "bytes": 300, "sha256": h("a")}]})  # index != 0

# weight_set digest + total
sv("weight_set.ok.json", "weight_set", ws_valid)
si("weight_set.bad_digest.json", "weight_set", {**ws_valid, "set_digest": h("wrong")})
si("weight_set.bad_total.json", "weight_set", {**ws_valid, "total_bytes": 999})

# model manifest partial-load rule
sv("model_manifest.partial.json", "model_manifest", mm_valid)
sv("model_manifest.full.json", "model_manifest", mm_full)
si("model_manifest.full_multi.json", "model_manifest",
   {**mm_full, "weight_sets": mm_full["weight_sets"] + [
       {"weight_set_id": "ws-2", "set_digest": h("ws2"), "total_bytes": 10,
        "layer_range": {"start": 0, "end": 16, "n_layer_total": 32}}]})  # partial False but 2 sets
si("model_manifest.full_notfull.json", "model_manifest",
   {**mm_full, "weight_sets": [{"weight_set_id": "ws-full", "set_digest": h("ws:full"), "total_bytes": 9000,
                                "layer_range": {"start": 0, "end": 16, "n_layer_total": 32}}]})  # not full range

# prepared image derived digest + tensor binding
sv("prepared_image.ok.json", "prepared_image", pi_htp)
si("prepared_image.bad_derived.json", "prepared_image", {**pi_htp, "derived_image_digest": h("wrong-derived")})
si("prepared_image.tensor_mismatch.json", "prepared_image",
   {**pi_htp, "tensor_digest": h("other"), "derived_image_digest": derived_digest(
       MV, h("other"), GRAPH, BUILD_HTP, "op15", "gemma4", 3, 7, 2)})  # digest self-consistent but != source_weight_set_digest

# ready certificate total
sv("ready_certificate.ok.json", "ready_certificate", rc_htp)
si("ready_certificate.total.json", "ready_certificate",
   {**rc_htp, "physical_bytes": {**rc_htp["physical_bytes"], "total": 701}})

# device inventory ledger
sv("device_inventory.ok.json", "device_inventory", di)
si("device_inventory.ledger.json", "device_inventory",
   {**di, "physical_byte_accounting": {**di["physical_byte_accounting"], "free": 8801}})  # sum != total
si("device_inventory.over.json", "device_inventory",
   {**di, "physical_byte_accounting": {"weights_resident": 11000, "derived_images": 0, "scratch": 0,
                                       "activations": 0, "mutable_state": 0, "free": 0}})  # a field exceeds total (and sum != total)

# residency lease horizon
sv("residency_lease.ok.json", "residency_lease", rl)
si("residency_lease.horizon.json", "residency_lease",
   {**rl, "horizon": {"start_us": 1000000, "end_us": 1000001}})  # end < start+min_hold

# transfer ticket chunk range
sv("transfer_ticket.ok.json", "transfer_ticket", tt)
si("transfer_ticket.chunk_range.json", "transfer_ticket",
   {**tt, "chunk_range": {"first": 3, "last": 1}})  # first > last

sem_index = ([{"file": f"valid/{n}", "kind": k, "expect": "valid"} for n, k in sem_valid] +
             [{"file": f"invalid/{n}", "kind": k, "expect": "invalid"} for n, k in sem_invalid])
with open(os.path.join(FX, "semantic", "index.json"), "w") as f:
    f.write(json.dumps(sem_index, indent=2) + "\n")

print(f"semantic fixtures: {len(sem_valid)} valid + {len(sem_invalid)} invalid = {len(sem_index)}")
