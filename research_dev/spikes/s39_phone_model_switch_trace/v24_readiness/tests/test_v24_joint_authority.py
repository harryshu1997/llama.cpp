#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load_module("v24_joint_evidence", HERE / "cp0_r1_evidence_v24.py")
common = evidence.common
base_tests = load_module("v24_base_tests", HERE / "tests" / "test_v24_readiness.py")
producer_tests = load_module(
    "v24_joint_producer_tests",
    HERE / "producers_v1" / "tests" / "test_joint_phone_cuda_v1.py",
)


def artifact(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": common.sha256_bytes(raw),
    }


def execution_groups(history: dict) -> list[dict]:
    requests = {
        request["item_index"]: request
        for request in history["requests"]
    }
    result = []
    frame_index = 0
    for expected in history["quality_groups"]:
        wires = [
            10000 + expected["group_index"] * 8 + index
            for index in range(8)
        ]
        receipts = []
        current = {}
        continuations = [[] for _ in range(8)]
        for partition in expected["prefill_partitions"]:
            rows = []
            for source in partition["rows"]:
                output = source["token_id"] + 100
                rows.append({
                    "input_token": source["token_id"],
                    "item_index": source["item_index"],
                    "output_token": output,
                    "position": source["position"],
                    "request_id": source["request_id"],
                    "route_epoch": 7,
                    "seq_id": source["seq_id"],
                    "wire_request_id": wires[source["seq_id"]],
                })
                request = requests[source["item_index"]]
                if source["position"] == len(request["token_ids"]) - 1:
                    current[source["seq_id"]] = output
            receipts.append({
                "call_index": partition["call_index"],
                "frame_call_index": frame_index,
                "phase": "prefill",
                "rows": rows,
            })
            frame_index += 1
        first = copy.deepcopy(current)
        for call in expected["decode_calls"]:
            rows = []
            next_tokens = {}
            for source in call["rows"]:
                sequence = source["seq_id"]
                output = current[sequence] + 1
                rows.append({
                    "input_token": current[sequence],
                    "item_index": source["item_index"],
                    "output_token": output,
                    "position": source["position"],
                    "request_id": source["request_id"],
                    "route_epoch": 7,
                    "seq_id": sequence,
                    "wire_request_id": wires[sequence],
                })
                next_tokens[sequence] = output
            receipts.append({
                "call_index": call["call_index"],
                "frame_call_index": frame_index,
                "phase": "decode",
                "rows": rows,
            })
            frame_index += 1
            current = next_tokens
            for sequence in range(8):
                continuations[sequence].append(current[sequence])
        for sequence in range(8):
            continuations[sequence].insert(0, first[sequence])
        result.append({
            "call_receipts": receipts,
            "continuations": continuations,
            "group_index": expected["group_index"],
            "item_indices": expected["item_indices"],
            "wire_request_ids": wires,
        })
    return result


class JointAuthorityFixture:
    def __init__(self, root: Path):
        self.root = root
        self.bundle = root / "bundle"
        self.bundle.mkdir()
        base_root = root / "base"
        base_root.mkdir()
        self.base = base_tests.Fixture(base_root)
        self.contract = self.base.contract
        self.candidate = self.base.candidate
        self.history = self.base.history
        self.history_raw = self.base.history_raw
        self.runtime_plan = self.base.plan
        self.runtime = self.base.runtime
        self.artifact_root = self.base.artifact_root
        self.phase_id = self.runtime["phase_id"]
        self.started_ns = 1100
        self.completed_ns = 1900
        self.route_epoch = 7
        self.model = next(
            row for row in self.candidate["models"] if row["slot"] == "A"
        )
        self.groups = execution_groups(self.history)
        self.mechanism = {"route": "OP15_DIRECT_OP12_CUDA"}
        self.mechanism_sha256 = common.sha256_bytes(
            common.canonical_bytes(self.mechanism)
        )
        self.runtime_processes = self._runtime_processes()
        self.phone_launch = self._phone_launch()
        self.cuda_launch = self._cuda_launch()
        self.phone_fragment = self._phone_fragment()
        self.cuda_fragment = self._cuda_fragment()
        self.receipt = self._outer_receipt()
        self.rows_by_role = self._rows_by_role()

    def write_json(self, relative: str, value: dict) -> dict:
        path = self.bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(common.canonical_bytes(value))
        return artifact(path)

    def write_bytes(self, relative: str, raw: bytes) -> dict:
        path = self.bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return artifact(path)

    def _runtime_processes(self) -> list[dict]:
        result = []
        for row in self.runtime["processes"]:
            if row["bundle_id"] == "cuda_monolithic":
                continue
            result.append({
                "boot_id": row["boot_id"],
                "bundle_id": row["bundle_id"],
                "endpoint": row["endpoint"],
                "launcher_path": row["launcher_path"],
                "loaded_repo_component_ids": row["loaded_repo_component_ids"],
                "observed_ns": row["observed_ns"],
                "pid": row["pid"],
                "start_ticks": row["start_ticks"],
                "system_dependencies": [{
                    "build_id": None,
                    "ctime_ns": 1,
                    "device_id": 1,
                    "inode": row["pid"],
                    "mode": 0o100755,
                    "mtime_ns": 1,
                    "path": f"/runtime/{row['bundle_id']}.so",
                    "size": 1,
                }],
            })
        return sorted(result, key=lambda value: value["bundle_id"])

    def process(self, bundle_id: str) -> dict:
        return next(
            row for row in self.runtime_processes
            if row["bundle_id"] == bundle_id
        )

    def _phone_launch(self) -> dict:
        devices = self.contract["devices"]
        route = self.contract["incumbent_route_lock"]
        processes = {
            row["bundle_id"]: row for row in self.runtime["processes"]
        }
        phones = {}
        for phone, bundle, executed, stored, local, peer in (
            ("op15", "op15_stagenet", [0, 30], [0, 32], "10.0.0.15", "10.0.0.12"),
            ("op12", "op12_stagenet", [30, 40], [24, 40], "10.0.0.12", "10.0.0.15"),
        ):
            component = next(
                row for row in self.artifact_root["components"]
                if row["component_id"]
                == ("op15-stage" if phone == "op15" else "op12-stage")
            )
            shard = self.contract["model_geometry"][evidence.MODEL_ID][
                "known_shards"
            ][phone]
            phones[phone] = {
                **devices[phone],
                "boot_id": processes[bundle]["boot_id"],
                "direct_peer_ipv4": peer,
                "executed_layers": executed,
                "expected_worker_executable_path": processes[bundle][
                    "launcher_path"
                ],
                "expected_worker_executable_sha256": component["sha256"],
                "interface": "wlan0",
                "loaded_shard_path": shard["path"],
                "loaded_shard_sha256": (
                    route["op15_shard_sha256"]
                    if phone == "op15"
                    else route["op12_shard_sha256"]
                ),
                "local_ipv4": local,
                "stored_layers": stored,
            }
        specs = {}
        for bundle_id in ("op12_stagenet", "op15_direct_relay", "op15_stagenet"):
            process = processes[bundle_id]
            specs[bundle_id] = {
                "runtime_component_ids": process["loaded_repo_component_ids"],
                "runtime_executable_path": process["launcher_path"],
            }
        return {
            "expected_file_type": 15,
            "expected_max_streams": 8,
            "expected_n_batch": 64,
            "expected_n_ctx_seq": 512,
            "expected_n_embd": 5120,
            "expected_n_layer": 40,
            "expected_n_ubatch": 64,
            "history_path": self.runtime_plan["token_history"]["artifact_path"],
            "history_sha256": common.sha256_bytes(self.history_raw),
            "model_id": evidence.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "phones": phones,
            "processes": specs,
            "route_epoch": self.route_epoch,
            "schema": "s39-cp0-r1-v24-phone-route-launch-v1",
        }

    def _cuda_launch(self) -> dict:
        component = next(
            row for row in self.artifact_root["components"]
            if row["component_id"] == "model.cuda"
        )
        process = next(
            row for row in self.runtime["processes"]
            if row["bundle_id"] == "cuda_route"
        )
        return {
            "expected_capabilities": 0x3F,
            "expected_file_type": 15,
            "expected_max_streams": 8,
            "expected_n_batch": 64,
            "expected_n_ctx_seq": 512,
            "expected_n_embd": 5120,
            "expected_n_layer": 40,
            "expected_n_ubatch": 64,
            "history_path": self.runtime_plan["token_history"]["artifact_path"],
            "history_sha256": common.sha256_bytes(self.history_raw),
            "mechanism_commands": self.mechanism,
            "model_artifact": {
                "bytes": component["bytes"],
                "path": component["path"],
                "sha256": component["sha256"],
                "stat": component["stat"],
            },
            "model_id": evidence.MODEL_ID,
            "model_sha256": component["sha256"],
            "route_epoch": self.route_epoch,
            "schema": "s39-cp0-r1-v24-cuda-route-launch-v1",
            "worker": {
                "argv": [
                    process["launcher_path"],
                    "--mode",
                    "monov3",
                    "--backend",
                    "CUDA0",
                    "--layer-start",
                    "0",
                    "--layer-end",
                    "40",
                    "--model",
                    component["path"],
                ],
                "environment": {
                    "LAYERSPLIT_MEMORY_CERT": "1",
                    "LAYERSPLIT_MODEL_SHA256": component["sha256"],
                    "LAYERSPLIT_PLACEMENT_CERT": "1",
                },
                "runtime_component_ids": process["loaded_repo_component_ids"],
                "runtime_executable": {"path": process["launcher_path"]},
            },
        }

    def _mechanics_rows(self, backend: str, event_ns: int) -> list[dict]:
        rows = producer_tests.mechanics_rows(
            self.history,
            self.groups,
            backend,
        )
        for index, row in enumerate(rows):
            row["event_ns"] = event_ns + index
        return rows

    def _quality_rows(self, event_ns: int) -> list[dict]:
        return [
            {
                "event_ns": event_ns + request["item_index"],
                "item_index": request["item_index"],
                "kind": "output",
                "prompt_sha256": request["prompt_sha256"],
                "raw_output": "A",
            }
            for request in self.history["requests"]
        ]

    def _phone_probe(self, phone: str, tx_bytes: int, rx_bytes: int) -> dict:
        launch = self.phone_launch["phones"][phone]
        process = self.process(
            "op15_stagenet" if phone == "op15" else "op12_stagenet"
        )
        return {
            "active_sequences": 0,
            "available_bytes": 1024 * 1024 * 1024,
            "boot_id": process["boot_id"],
            "device": launch["device"],
            "direct_peer": {
                "interface": launch["interface"],
                "local_ipv4": launch["local_ipv4"],
                "peer_ipv4": launch["direct_peer_ipv4"],
                "socket_peer_observed": True,
            },
            "gpu_max_millic": 50000,
            "interface": {
                "ipv4": launch["local_ipv4"],
                "name": launch["interface"],
                "rx_bytes": rx_bytes,
                "tx_bytes": tx_bytes,
            },
            "loaded_shard_path": launch["loaded_shard_path"],
            "loaded_shard_sha256": launch["loaded_shard_sha256"],
            "model": launch["model"],
            "model_id": evidence.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "process_swap_bytes": 0,
            "product": launch["product"],
            "schema": "s39-cp0-r1-v24-phone-runtime-probe-v1",
            "serial": launch["serial"],
            "system_swap_used_bytes": 0,
            "thermal_status": 0,
            "worker_executable_path": launch["expected_worker_executable_path"],
            "worker_executable_sha256": launch[
                "expected_worker_executable_sha256"
            ],
            "worker_pid": process["pid"],
            "worker_start_ticks": process["start_ticks"],
        }

    def _phone_fragment(self) -> dict:
        calls = [
            call
            for group in self.groups
            for call in group["call_receipts"]
        ]
        frames = []
        total_payload = 0
        for index, call in enumerate(calls):
            rows = len(call["rows"])
            payload = rows * 5120 * 4
            total_payload += payload
            frames.append({
                "activation_payload_bytes": payload,
                "call_index": index,
                "hidden_width": 5120,
                "payload_sha256": f"{index + 1:064x}",
                "positions": [row["position"] for row in call["rows"]],
                "request_ids": [
                    row["wire_request_id"] for row in call["rows"]
                ],
                "route_epochs": [self.route_epoch] * rows,
                "rows": rows,
                "schema": "ls-stage-direct-frame-v1",
                "seq_ids": [row["seq_id"] for row in call["rows"]],
            })
        mechanics = self._mechanics_rows("PHONE_COLLECTIVE", 1400)
        publications = []
        for row in mechanics[1:]:
            normalized = {
                "acquisition_id": self.phase_id,
                **{
                    key: value
                    for key, value in row.items()
                    if key != "event_ns"
                },
                "role": f"model.{evidence.MODEL_ID}.mechanics.phone",
            }
            publications.append({
                "event_ns": 1500 + row["request_id"],
                "kind": "phone_publication_received",
                "phone_request_sha256": common.sha256_bytes(
                    common.canonical_bytes(normalized)
                ),
                "request_id": row["request_id"],
                "token_ids": row["continuation_tokens"],
            })
        mechanics_calls = self.groups[0]["call_receipts"]
        transfers = [{
            "batch": 8,
            "cut_layer": 30,
            "event_ns": 1420,
            "kind": "meta",
            "model_id": evidence.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "request_ids": list(range(8)),
        }]
        for index, (frame, _) in enumerate(
            zip(frames, mechanics_calls)
        ):
            transfers.append({
                "call_index": index,
                "event_ns": 1421 + index,
                "host_payload_bytes": 0,
                "kind": "transfer",
                "path": "WIFI_TCP_DIRECT",
                "payload_bytes": frame["activation_payload_bytes"],
                "payload_sha256": frame["payload_sha256"],
                "receiver": "op12",
                "row_count": frame["rows"],
                "sender": "op15",
            })
        sessions = {}
        placements = {}
        for phone, layers, bundle in (
            ("op15", [0, 30], "op15_stagenet"),
            ("op12", [30, 40], "op12_stagenet"),
        ):
            process = self.process(bundle)
            op_map = {"MUL_MAT": {"OpenCL": 10}}
            sessions[phone] = {
                "compute_by_op_and_buffer": op_map,
                "device_boot_id": process["boot_id"],
                "expected_backend": "GPUOpenCL",
                "layer_end": layers[1],
                "layer_start": layers[0],
                "missing_buffer_compute_nodes": 0,
                "n_layer": 40,
                "placement_status": "SCHEDULED_PLACEMENT_OK",
                "proto_version": 2,
                "reset_applied": False,
                "schema": "ls-stagenet-session-v2",
                "session_end": "STOP",
                "session_id": 1,
                "steps_session": len(calls),
                "steps_total": len(calls),
                "worker_boot_nonce": "0123456789abcdef",
                "worker_pid": process["pid"],
            }
            placements[phone] = {
                "compute_by_buffer_type": {"OpenCL": 10},
                "compute_by_op": {"MUL_MAT": 10},
                "compute_by_op_and_buffer": op_map,
                "compute_nodes": 10,
                "copy_by_buffer_type": {},
                "copy_nodes": 0,
                "layer_end": layers[1],
                "layer_start": layers[0],
                "metadata_nodes": 0,
                "missing_buffer_compute_nodes": 0,
                "mode": "stagenet" if phone == "op15" else "tailv3",
                "n_layer": 40,
                "pid": process["pid"],
                "role": "phone_stage" if phone == "op15" else "host_tail_v3",
                "run_rc": 0,
                "schema": "layersplit-scheduled-placement-v2",
                "status": "SCHEDULED_PLACEMENT_OK",
            }
        placement_rows = {}
        for phone, layers in (("op15", [0, 30]), ("op12", [30, 40])):
            placement_rows[phone] = [{
                "event_ns": 1550,
                "executed_layers": layers,
                "kind": "meta",
            }]
        before = {
            "op15": self._phone_probe("op15", 0, 0),
            "op12": self._phone_probe("op12", 0, 0),
        }
        after = {
            "op15": self._phone_probe("op15", total_payload, 0),
            "op12": self._phone_probe("op12", 0, total_payload),
        }
        log_artifact = self.write_bytes("phone/phone.log", b"phone\n")
        return {
            "bridge_publication_rows": publications,
            "completed_ns": 1750,
            "direct_certificate": {
                "activation_payload_bytes": total_payload,
                "batches": len(frames),
                "cut_layer": 30,
                "file_type": 15,
                "head_endpoint": "op15",
                "host_activation_payload_bytes": 0,
                "layer_end": 40,
                "layer_start": 0,
                "model_sha256": self.model["artifact"]["sha256"],
                "n_embd": 5120,
                "n_layer": 40,
                "rows": sum(frame["rows"] for frame in frames),
                "run_rc": 0,
                "schema": "ls-stage-direct-relay-v1",
                "status": "DIRECT_RELAY_OK",
                "tail_endpoint": "op12",
            },
            "direct_frames": frames,
            "evidence_artifacts": [log_artifact],
            "execution_groups": copy.deepcopy(self.groups),
            "history_sha256": common.sha256_bytes(self.history_raw),
            "launch_plan_sha256": "",
            "mechanics_rows": mechanics,
            "mechanism_commands_sha256": self.mechanism_sha256,
            "model_id": evidence.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "op12_runtime": {"active_sequences_after_cleanup": 0},
            "op15_runtime": {"active_sequences_after_cleanup": 0},
            "phase_id": self.phase_id,
            "placement_certificates": placements,
            "placement_op12_rows": placement_rows["op12"],
            "placement_op15_rows": placement_rows["op15"],
            "producer_sha256": self.contract["producer_requirements"][
                "source_programs"
            ]["phone_route"]["sha256"],
            "quality_phone_rows": self._quality_rows(1450),
            "raw_probes": {
                "op12": {
                    "after": after["op12"],
                    "after_ns": 1701,
                    "before": before["op12"],
                    "before_ns": 1201,
                },
                "op15": {
                    "after": after["op15"],
                    "after_ns": 1700,
                    "before": before["op15"],
                    "before_ns": 1200,
                },
            },
            "route_epoch": self.route_epoch,
            "route_transfer_rows": transfers,
            "runtime_processes": [
                self.process("op12_stagenet"),
                self.process("op15_direct_relay"),
                self.process("op15_stagenet"),
            ],
            "schema": "s39-cp0-r1-v24-phone-route-raw-v1",
            "session_certificates": sessions,
            "started_ns": 1150,
        }

    def _cuda_placement(self) -> dict:
        process = self.process("cuda_route")
        return {
            "compute_by_buffer_type": {"CUDA0": 10},
            "compute_by_op": {"MUL_MAT": 10},
            "compute_by_op_and_buffer": {"MUL_MAT": {"CUDA0": 10}},
            "compute_nodes": 10,
            "copy_by_buffer_type": {},
            "copy_nodes": 0,
            "layer_end": 40,
            "layer_start": 0,
            "metadata_nodes": 0,
            "missing_buffer_compute_nodes": 0,
            "mode": "monov3",
            "n_layer": 40,
            "pid": process["pid"],
            "role": "monov3",
            "run_rc": 0,
            "schema": "layersplit-scheduled-placement-v2",
            "status": "SCHEDULED_PLACEMENT_OK",
        }

    def _cuda_fragment(self) -> dict:
        process = self.process("cuda_route")
        placement = self._cuda_placement()
        memory = {
            "compute_buffer_bytes": 100,
            "host_compute_buffer_bytes": 20,
            "host_context_buffer_bytes": 30,
            "host_model_buffer_bytes": 40,
            "kv_buffer_bytes": 1_000_000_000,
            "model_buffer_bytes": 6_000_000_000,
            "pid": process["pid"],
            "role": "monov3",
            "schema": "layersplit-memory-breakdown-v1",
        }
        rows = []
        samples = []
        for index, kind in enumerate(("before", "ready", "after")):
            row = {
                "event_ns": 1300 + index,
                "kind": kind,
                "kv_buffer_bytes": (
                    memory["kv_buffer_bytes"] if kind == "ready" else 0
                ),
                "model_buffer_bytes": (
                    memory["model_buffer_bytes"] if kind == "ready" else 0
                ),
                "placement_compute_nodes": (
                    placement["compute_nodes"] if kind == "ready" else 0
                ),
                "process_pid": process["pid"] if kind == "ready" else 0,
                "process_used_bytes": (
                    8_000_000_000 if kind == "ready" else 0
                ),
            }
            rows.append(row)
            record = self.write_json(f"cuda/memory-{kind}.json", row)
            samples.append({**record, "row": copy.deepcopy(row)})
        component = next(
            row for row in self.artifact_root["components"]
            if row["component_id"] == "model.cuda"
        )
        model_rows = []
        for index, expected in enumerate(
            self.contract["cuda_monolithic_identity"]["maps_exact_rows"]
        ):
            model_rows.append({
                "address_range": f"{0x60000000 + index * 0x200000:x}-"
                f"{0x60100000 + index * 0x200000:x}",
                "device_major": 0,
                "device_minor": 7,
                "inode": component["stat"]["inode"],
                "offset_bytes": expected["offset_bytes"],
                "path": component["path"],
                "permissions": expected["permissions"],
            })
        return {
            "bridge_ready_row": {"event_ns": 1600, "kind": "cuda_ready"},
            "bridge_start_row": {"event_ns": 1250, "kind": "cuda_load_start"},
            "completed_ns": 1800,
            "cuda_memory_rows": rows,
            "cuda_route_rows": self._mechanics_rows("CUDA0", 1400),
            "evidence_artifacts": [
                self.write_bytes("cuda/cuda.log", b"cuda\n")
            ],
            "execution_groups": copy.deepcopy(self.groups),
            "gpu_runtime": {"device": "CUDA0"},
            "history_sha256": common.sha256_bytes(self.history_raw),
            "launch_plan_sha256": "",
            "mechanism_commands_sha256": self.mechanism_sha256,
            "memory_certificate": memory,
            "model_id": evidence.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "phase_id": self.phase_id,
            "placement_certificate": placement,
            "producer_sha256": self.contract["producer_requirements"][
                "source_programs"
            ]["cuda_route"]["sha256"],
            "protocol_identity": {
                "capabilities": 0x3F,
                "file_type": 15,
                "layer_end": 40,
                "layer_start": 0,
                "max_streams": 8,
                "model_sha256": self.model["artifact"]["sha256"],
                "n_batch": 64,
                "n_ctx_seq": 512,
                "n_embd": 5120,
                "n_layer": 40,
                "n_ubatch": 64,
                "schema": "layersplit-stage-v3-identity-v1",
                "stage_identity_version": 1,
                "stage_protocol_version": 3,
            },
            "quality_cuda_rows": self._quality_rows(1450),
            "raw_memory_samples": samples,
            "route_epoch": self.route_epoch,
            "runtime_model_binding": {
                "model_mapping_rows": model_rows,
                "model_path": component["path"],
                "model_sha256": component["sha256"],
                "other_gguf_mapping_paths": [],
                "pid": process["pid"],
                "start_ticks": process["start_ticks"],
            },
            "runtime_process": process,
            "schema": "s39-cp0-r1-v24-cuda-route-raw-v1",
            "started_ns": 1151,
        }

    def _command(
        self,
        name: str,
        launch: dict,
    ) -> tuple[dict, dict, dict]:
        source_name = "cuda_route" if name == "cuda" else "phone_route"
        source_path = (
            HERE / "producers_v1" / f"{source_name}_v1.py"
        ).resolve()
        source_raw = source_path.read_bytes()
        source_copy = self.write_bytes(
            f"joint/evidence/executed/{name}/000-{source_path.name}",
            source_raw,
        )
        launch_record = self.write_json(
            f"joint/evidence/executed/{name}/001-launch.json",
            launch,
        )
        template = [
            str(source_path),
            "--output",
            "{output_path}",
            "--phase-id",
            "{phase_id}",
            "--pre-dir",
            "{pre_dir}",
            "--started",
            "{acquisition_started_ns}",
            "--plan",
            "{command_plan_sha256}",
            "--mechanism-commands-sha256",
            self.mechanism_sha256,
            "--model-sha256",
            self.model["artifact"]["sha256"],
            "--launch-plan",
            "/original/launch.json",
        ]
        records = [
            {
                "argv_index": 0,
                "bytes": len(source_raw),
                "path": str(source_path),
                "sha256": common.sha256_bytes(source_raw),
            },
            {
                "argv_index": len(template) - 1,
                "bytes": launch_record["bytes"],
                "path": "/original/launch.json",
                "sha256": launch_record["sha256"],
            },
        ]
        history_copy = None
        if name == "cuda":
            template.extend(
                ["--histories", "/original/token-history.json"]
            )
            history_copy = self.write_bytes(
                f"joint/evidence/executed/{name}/002-token-history.json",
                self.history_raw,
            )
            records.append({
                "argv_index": len(template) - 1,
                "bytes": len(self.history_raw),
                "path": "/original/token-history.json",
                "sha256": common.sha256_bytes(self.history_raw),
            })
        template.extend([
            "--execute",
            "--confirm",
            (
                "RUN_V24_CUDA_ROUTE_A_ONLY"
                if name == "cuda"
                else "RUN_V24_PHONE_ROUTE_A_ONLY"
            ),
        ])
        command = {
            "argv_template": template,
            "cwd": str(self.bundle),
            "environment": {"LC_ALL": "C"},
            "executed_files": records,
            "launch_plan_argv_index": records[1]["argv_index"],
            "launch_plan_sha256": launch_record["sha256"],
            "producer_sha256": records[0]["sha256"],
            "result_filename": f"{name}.result.json",
            "timeout_seconds": 10,
        }
        copied_by_index = {
            0: source_copy["path"],
            records[1]["argv_index"]: launch_record["path"],
        }
        if history_copy is not None:
            copied_by_index[records[2]["argv_index"]] = history_copy["path"]
        replacements = {
            "{acquisition_started_ns}": "1000",
            "{command_plan_sha256}": common.sha256_bytes(
                common.canonical_bytes(self.runtime_plan)
            ),
            "{output_path}": str(
                self.bundle / f"joint/evidence/{name}.result.json"
            ),
            "{phase_id}": self.phase_id,
            "{pre_dir}": str(self.bundle / "pre"),
        }
        argv = [
            copied_by_index.get(index, replacements.get(value, value))
            for index, value in enumerate(template)
        ]
        receipt = {
            "argv": argv,
            "completed_ns": 1780 if name == "cuda" else 1740,
            "cwd": command["cwd"],
            "environment": command["environment"],
            "launch_plan_sha256": launch_record["sha256"],
            "producer_sha256": records[0]["sha256"],
            "returncode": 0,
            "schema": "s39-cp0-r1-v24-subproducer-receipt-v1",
            "started_ns": 1121 if name == "cuda" else 1120,
        }
        return command, receipt, launch_record

    def _outer_receipt(self) -> dict:
        cuda_command, cuda_receipt, _ = self._command(
            "cuda",
            self.cuda_launch,
        )
        phone_command, phone_receipt, _ = self._command(
            "phone",
            self.phone_launch,
        )
        self.cuda_fragment["launch_plan_sha256"] = cuda_command[
            "launch_plan_sha256"
        ]
        self.phone_fragment["launch_plan_sha256"] = phone_command[
            "launch_plan_sha256"
        ]
        cuda_fragment_record = self.write_json(
            "joint/evidence/cuda.result.json",
            self.cuda_fragment,
        )
        phone_fragment_record = self.write_json(
            "joint/evidence/phone.result.json",
            self.phone_fragment,
        )
        cuda_receipt_record = self.write_json(
            "joint/evidence/cuda.receipt.json",
            cuda_receipt,
        )
        phone_receipt_record = self.write_json(
            "joint/evidence/phone.receipt.json",
            phone_receipt,
        )
        capture_plan = {
            "commands": {
                "cuda": cuda_command,
                "phone": phone_command,
            },
            "history": {
                "bytes": len(self.history_raw),
                "path": self.runtime_plan["token_history"]["artifact_path"],
                "sha256": common.sha256_bytes(self.history_raw),
            },
            "mechanism_commands": self.mechanism,
            "model_id": evidence.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "phase": "A_ONLY",
            "schema": "s39-cp0-r1-v24-joint-capture-plan-v1",
        }
        capture_plan_record = self.write_json(
            "joint/capture-plan.json",
            capture_plan,
        )
        joint_source = (
            HERE / "producers_v1" / "joint_phone_cuda_v1.py"
        ).read_bytes()
        joint_source_record = self.write_bytes(
            "joint/evidence/joint_phone_cuda_v1.py",
            joint_source,
        )
        cuda_evidence = {
            "fragment": cuda_fragment_record,
            "launch_plan_sha256": cuda_command["launch_plan_sha256"],
            "memory_certificate": self.cuda_fragment["memory_certificate"],
            "placement_certificate": self.cuda_fragment[
                "placement_certificate"
            ],
            "protocol_identity": self.cuda_fragment["protocol_identity"],
            "raw_memory_samples": self.cuda_fragment["raw_memory_samples"],
            "receipt": cuda_receipt,
            "receipt_artifact": cuda_receipt_record,
            "runtime_model_binding": self.cuda_fragment[
                "runtime_model_binding"
            ],
            "runtime_process": self.cuda_fragment["runtime_process"],
        }
        phone_evidence = {
            "direct_certificate": self.phone_fragment["direct_certificate"],
            "direct_frames": self.phone_fragment["direct_frames"],
            "fragment": phone_fragment_record,
            "launch_plan_sha256": phone_command["launch_plan_sha256"],
            "placement_certificates": self.phone_fragment[
                "placement_certificates"
            ],
            "raw_probes": self.phone_fragment["raw_probes"],
            "receipt": phone_receipt,
            "receipt_artifact": phone_receipt_record,
            "runtime_processes": self.phone_fragment["runtime_processes"],
            "session_certificates": self.phone_fragment[
                "session_certificates"
            ],
        }
        return {
            "bridge_rows": [
                self.cuda_fragment["bridge_start_row"],
                *self.phone_fragment["bridge_publication_rows"],
                self.cuda_fragment["bridge_ready_row"],
            ],
            "capture_plan_artifact": capture_plan_record,
            "capture_plan_sha256": capture_plan_record["sha256"],
            "command_plan_sha256": common.sha256_bytes(
                common.canonical_bytes(self.runtime_plan)
            ),
            "completed_ns": self.completed_ns,
            "cuda_evidence": cuda_evidence,
            "cuda_memory_rows": self.cuda_fragment["cuda_memory_rows"],
            "cuda_route_rows": self.cuda_fragment["cuda_route_rows"],
            "executed_file_artifacts": {
                "cuda": cuda_command["executed_files"],
                "phone": phone_command["executed_files"],
            },
            "fragment_sha256": {
                "cuda": cuda_fragment_record["sha256"],
                "phone": phone_fragment_record["sha256"],
            },
            "gpu_runtime": self.cuda_fragment["gpu_runtime"],
            "history_sha256": common.sha256_bytes(self.history_raw),
            "joint_producer_artifact": joint_source_record,
            "joint_producer_sha256": joint_source_record["sha256"],
            "mechanics_rows": self.phone_fragment["mechanics_rows"],
            "mechanism_commands_sha256": self.mechanism_sha256,
            "model_id": evidence.MODEL_ID,
            "model_sha256": self.model["artifact"]["sha256"],
            "op12_runtime": self.phone_fragment["op12_runtime"],
            "op15_runtime": self.phone_fragment["op15_runtime"],
            "phase_id": self.phase_id,
            "phone_evidence": phone_evidence,
            "placement_op12_rows": self.phone_fragment[
                "placement_op12_rows"
            ],
            "placement_op15_rows": self.phone_fragment[
                "placement_op15_rows"
            ],
            "quality_cuda_rows": self.cuda_fragment["quality_cuda_rows"],
            "quality_phone_rows": self.phone_fragment["quality_phone_rows"],
            "route_epoch": self.route_epoch,
            "route_transfer_rows": self.phone_fragment[
                "route_transfer_rows"
            ],
            "runtime_processes": self.runtime_processes,
            "schema": "s39-cp0-r1-v24-joint-phone-cuda-raw-v1",
            "started_ns": self.started_ns,
            "subproducer_bindings": {
                "cuda": {
                    "launch_plan_sha256": cuda_command[
                        "launch_plan_sha256"
                    ],
                    "producer_sha256": cuda_command["producer_sha256"],
                },
                "phone": {
                    "launch_plan_sha256": phone_command[
                        "launch_plan_sha256"
                    ],
                    "producer_sha256": phone_command["producer_sha256"],
                },
            },
        }

    def _rows_by_role(self) -> dict[str, list[dict]]:
        fields = {
            "model.qwen3-14b-q4_k_m.bridge": "bridge_rows",
            "model.qwen3-14b-q4_k_m.cuda_memory": "cuda_memory_rows",
            "model.qwen3-14b-q4_k_m.mechanics.phone": "mechanics_rows",
            "model.qwen3-14b-q4_k_m.oracle.cuda_route": "cuda_route_rows",
            "model.qwen3-14b-q4_k_m.placement.op12": "placement_op12_rows",
            "model.qwen3-14b-q4_k_m.placement.op15": "placement_op15_rows",
            "model.qwen3-14b-q4_k_m.quality.cuda": "quality_cuda_rows",
            "model.qwen3-14b-q4_k_m.quality.phone": "quality_phone_rows",
            "model.qwen3-14b-q4_k_m.route_transfer": "route_transfer_rows",
        }
        return {
            role: [
                {
                    "acquisition_id": self.phase_id,
                    **row,
                    "phase": "A_ONLY",
                    "phase_id": self.phase_id,
                    "role": role,
                }
                for row in self.receipt[key]
            ]
            for role, key in fields.items()
        }

    def validate(self) -> None:
        evidence.validate_joint_phone_cuda_receipt(
            self.receipt,
            self.bundle,
            self.contract,
            self.candidate,
            self.history,
            self.history_raw,
            self.runtime_plan,
            self.runtime,
            self.artifact_root,
            self.rows_by_role,
            1000,
        )


class JointAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = JointAuthorityFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_producer_shaped_joint_receipt_passes(self):
        self.fixture.validate()

    def test_legacy_cuda_launch_schema_is_rejected(self):
        launch = copy.deepcopy(self.fixture.cuda_launch)
        launch["schema"] = "s39-cp0-r1-a-only-cuda-route-launch-v1"
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_JOINT_CUDA_LAUNCH_SCHEMA",
        ):
            evidence._validate_joint_cuda_launch(
                launch,
                next(
                    row for row in self.fixture.artifact_root["components"]
                    if row["component_id"] == "model.cuda"
                ),
                self.fixture.history_raw,
                self.fixture.runtime_plan,
                self.fixture.runtime,
                self.fixture.receipt,
            )

    def test_joint_cuda_pid_cannot_reuse_monolithic_pid(self):
        monolithic = next(
            row for row in self.fixture.runtime["processes"]
            if row["bundle_id"] == "cuda_monolithic"
        )
        joint = next(
            row for row in self.fixture.receipt["runtime_processes"]
            if row["bundle_id"] == "cuda_route"
        )
        joint["pid"] = monolithic["pid"]
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_JOINT_RUNTIME_PROCESS|E_JOINT_CUDA_MONOLITHIC_PID_REUSE",
        ):
            self.fixture.validate()

    def test_joint_memory_certificate_pid_is_load_bearing(self):
        self.fixture.receipt["cuda_evidence"]["memory_certificate"]["pid"] += 1
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_JOINT_CUDA_MEMORY_CERT_PROJECTION|E_MEMORY_PID",
        ):
            self.fixture.validate()

    def test_joint_protocol_identity_is_load_bearing(self):
        self.fixture.receipt["cuda_evidence"]["protocol_identity"][
            "n_batch"
        ] = 32
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_JOINT_CUDA_PROTOCOL_PROJECTION|E_CUDA_PROTOCOL_IDENTITY",
        ):
            self.fixture.validate()

    def test_joint_copied_source_is_load_bearing(self):
        path = Path(
            self.fixture.receipt["executed_file_artifacts"]["cuda"][0]["path"]
        )
        copied = Path(
            self.fixture.receipt["cuda_evidence"]["receipt"]["argv"][0]
        )
        self.assertNotEqual(path, copied)
        copied.write_bytes(copied.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_JOINT_EXECUTED_FILE_(BYTES|SHA256)|E_JOINT_SOURCE_(BYTES|SHA256)",
        ):
            self.fixture.validate()

    def test_joint_launch_plan_is_load_bearing(self):
        receipt = self.fixture.receipt["cuda_evidence"]["receipt"]
        launch_path = Path(
            receipt["argv"][receipt["argv"].index("--launch-plan") + 1]
        )
        launch_path.write_bytes(launch_path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_JOINT_EXECUTED_FILE_(BYTES|SHA256)|E_JOINT_LAUNCH_CANONICAL",
        ):
            self.fixture.validate()

    def test_joint_raw_role_projection_is_load_bearing(self):
        role = f"model.{evidence.MODEL_ID}.quality.phone"
        self.fixture.rows_by_role[role][0]["raw_output"] = "B"
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CAPTURE_RAW_PROJECTION",
        ):
            self.fixture.validate()

    def test_joint_capture_plan_artifact_is_load_bearing(self):
        path = Path(self.fixture.receipt["capture_plan_artifact"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_CAPTURE_(BYTES|SHA256)|E_CAPTURE_CANONICAL",
        ):
            self.fixture.validate()


if __name__ == "__main__":
    unittest.main()
