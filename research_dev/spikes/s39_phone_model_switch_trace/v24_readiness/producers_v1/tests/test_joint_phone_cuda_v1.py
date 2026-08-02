#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "joint_phone_cuda_v1.py"
SPEC = importlib.util.spec_from_file_location("joint_phone_cuda_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
joint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(joint)


def build_history() -> dict:
    requests = []
    groups = []
    for item_index in range(64):
        requests.append({
            "item_index": item_index,
            "prompt_sha256": f"{item_index + 1:064x}",
            "request_id": item_index % 8 + 1,
            "seq_id": item_index % 8,
            "token_ids": [1000 + item_index, 2000 + item_index],
        })
    for group_index in range(8):
        items = list(range(group_index * 8, group_index * 8 + 8))
        prefill_rows = []
        for position in range(2):
            for item_index in items:
                request = requests[item_index]
                prefill_rows.append({
                    "item_index": item_index,
                    "position": position,
                    "request_id": request["request_id"],
                    "seq_id": request["seq_id"],
                    "token_id": request["token_ids"][position],
                })
        prefill = [{"call_index": 0, "rows": prefill_rows}]
        decode = [
            {
                "call_index": ordinal + 1,
                "continuation_input_ordinal": ordinal,
                "continuation_output_ordinal": ordinal + 1,
                "rows": [
                    {
                        "item_index": item_index,
                        "position": 2 + ordinal,
                        "request_id": requests[item_index]["request_id"],
                        "seq_id": requests[item_index]["seq_id"],
                    }
                    for item_index in items
                ],
            }
            for ordinal in range(7)
        ]
        groups.append({
            "decode_calls": decode,
            "group_index": group_index,
            "item_indices": items,
            "prefill_partitions": prefill,
        })
    return {
        "mechanics_b8": groups[0],
        "quality_groups": groups,
        "requests": requests,
    }


def execution_groups(history: dict) -> list[dict]:
    result = []
    frame_index = 0
    for group in history["quality_groups"]:
        wires = [10000 + group["group_index"] * 8 + index for index in range(8)]
        receipts = []
        current = {}
        continuations = [[] for _ in range(8)]
        for partition in group["prefill_partitions"]:
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
                if source["position"] == 1:
                    current[source["seq_id"]] = output
            receipts.append({
                "call_index": partition["call_index"],
                "frame_call_index": frame_index,
                "phase": "prefill",
                "rows": rows,
            })
            frame_index += 1
        first = copy.deepcopy(current)
        for call in group["decode_calls"]:
            rows = []
            next_tokens = {}
            for source in call["rows"]:
                seq_id = source["seq_id"]
                output = current[seq_id] + 1
                rows.append({
                    "input_token": current[seq_id],
                    "item_index": source["item_index"],
                    "output_token": output,
                    "position": source["position"],
                    "request_id": source["request_id"],
                    "route_epoch": 7,
                    "seq_id": seq_id,
                    "wire_request_id": wires[seq_id],
                })
                next_tokens[seq_id] = output
            receipts.append({
                "call_index": call["call_index"],
                "frame_call_index": frame_index,
                "phase": "decode",
                "rows": rows,
            })
            frame_index += 1
            current = next_tokens
            for seq_id in range(8):
                continuations[seq_id].append(current[seq_id])
        for seq_id in range(8):
            continuations[seq_id].insert(0, first[seq_id])
        result.append({
            "call_receipts": receipts,
            "continuations": continuations,
            "group_index": group["group_index"],
            "item_indices": group["item_indices"],
            "wire_request_ids": wires,
        })
    return result


def mechanics_rows(history: dict, groups: list[dict], backend: str) -> list[dict]:
    rows = [{
        "backend": backend,
        "call_shapes": joint.expected_call_shapes(history["mechanics_b8"]),
        "event_ns": 100,
        "kind": "meta",
    }]
    for seq_id, item_index in enumerate(history["mechanics_b8"]["item_indices"]):
        source = history["requests"][item_index]
        rows.append({
            "continuation_tokens": groups[0]["continuations"][seq_id],
            "event_ns": 101 + seq_id,
            "input_tokens": source["token_ids"],
            "kind": "request",
            "positions": [0, 1],
            "request_id": seq_id + 1,
        })
    return rows


class JointTests(unittest.TestCase):
    def test_capture_plan_digest_is_checked_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture-plan.json"
            original = joint.canonical_bytes({"original": True})
            path.write_bytes(original)
            expected = joint.sha256_bytes(original)
            path.write_bytes(joint.canonical_bytes({"mutated": True}))
            with self.assertRaisesRegex(
                joint.CaptureError,
                "E_CAPTURE_PLAN_SHA256",
            ):
                joint.load_plan(path, expected)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.history = build_history()
        self.groups = execution_groups(self.history)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evidence_file(self, name: str, raw: bytes = b"evidence\n") -> dict:
        path = self.root / name
        path.write_bytes(raw)
        return {
            "bytes": len(raw),
            "path": str(path),
            "sha256": joint.sha256_bytes(raw),
        }

    def quality_rows(self) -> list[dict]:
        return [
            {
                "event_ns": 300 + item_index,
                "item_index": item_index,
                "kind": "output",
                "prompt_sha256": self.history["requests"][item_index][
                    "prompt_sha256"
                ],
                "raw_output": "A",
            }
            for item_index in range(64)
        ]

    def phone_value(self) -> dict:
        rows = mechanics_rows(self.history, self.groups, "PHONE_COLLECTIVE")
        publications = []
        for request in rows[1:]:
            normalized = {
                "acquisition_id": "phase",
                **{
                    key: value
                    for key, value in request.items()
                    if key != "event_ns"
                },
                "role": "model.qwen3-14b-q4_k_m.mechanics.phone",
            }
            publications.append({
                "event_ns": 200 + request["request_id"],
                "kind": "phone_publication_received",
                "phone_request_sha256": joint.sha256_bytes(
                    joint.canonical_bytes(normalized)
                ),
                "request_id": request["request_id"],
                "token_ids": request["continuation_tokens"],
            })
        receipts = [
            call
            for group in self.groups
            for call in group["call_receipts"]
        ]
        frames = []
        payload = 0
        for index, call in enumerate(receipts):
            size = len(call["rows"]) * 5120 * 4
            payload += size
            frames.append({
                "activation_payload_bytes": size,
                "call_index": index,
                "payload_sha256": f"{index + 1:064x}",
                "positions": [row["position"] for row in call["rows"]],
                "request_ids": [row["wire_request_id"] for row in call["rows"]],
                "route_epochs": [row["route_epoch"] for row in call["rows"]],
                "rows": len(call["rows"]),
                "seq_ids": [row["seq_id"] for row in call["rows"]],
            })
        def probe(tx: int, rx: int) -> dict:
            return {
                "available_bytes": 1024 * 1024 * 1024,
                "interface": {"rx_bytes": rx, "tx_bytes": tx},
                "process_swap_bytes": 0,
                "system_swap_used_bytes": 10,
                "thermal_status": 0,
            }
        def placement(stored, executed, shard):
            return [
                {
                    "available_after_bytes": 1024 * 1024 * 1024,
                    "available_before_bytes": 1024 * 1024 * 1024,
                    "event_ns": 400,
                    "executed_layers": executed,
                    "kind": "meta",
                    "process_swap_bytes": 0,
                    "shard_sha256": shard,
                    "stored_layers": stored,
                    "system_swap_after_bytes": 10,
                    "system_swap_before_bytes": 10,
                },
                {
                    "backend": "GPUOpenCL",
                    "event_ns": 401,
                    "kind": "node",
                    "missing_buffer": False,
                    "op": "MUL_MAT",
                },
            ]
        return {
            "bridge_publication_rows": publications,
            "direct_certificate": {
                "activation_payload_bytes": payload,
                "cut_layer": 30,
                "host_activation_payload_bytes": 0,
                "status": "DIRECT_RELAY_OK",
            },
            "direct_frames": frames,
            "evidence_artifacts": [self.evidence_file("phone.log")],
            "execution_groups": copy.deepcopy(self.groups),
            "mechanics_rows": rows,
            "op12_runtime": {
                "active_sequences_after_cleanup": 0,
                "process_swap_bytes": 0,
            },
            "op15_runtime": {
                "active_sequences_after_cleanup": 0,
                "process_swap_bytes": 0,
            },
            "phase_id": "phase",
            "placement_op12_rows": placement(
                [24, 40],
                [30, 40],
                "72e312af745160dc33a0ba39ba94fbbc"
                "e6112950d0409d39c42ddc3b25e756ab",
            ),
            "placement_op15_rows": placement(
                [0, 32],
                [0, 30],
                "ba56b9c5e19b3a4512777e6a47803cc"
                "03261c2d3c2734965cd5ec96b7c6c59fb",
            ),
            "quality_phone_rows": self.quality_rows(),
            "raw_probes": {
                "op12": {
                    "after": probe(0, payload),
                    "after_ns": 500,
                    "before": probe(0, 0),
                    "before_ns": 90,
                },
                "op15": {
                    "after": probe(payload, 0),
                    "after_ns": 500,
                    "before": probe(0, 0),
                    "before_ns": 90,
                },
            },
        }

    def cuda_value(self) -> dict:
        rows = []
        samples = []
        evidence = [self.evidence_file("cuda.log")]
        for index, kind in enumerate(("before", "ready", "after")):
            used = 8 * 1024 * 1024 * 1024 if kind == "ready" else 0
            process_used = 7 * 1024 * 1024 * 1024 if kind == "ready" else 0
            event_ns = 100 + index
            device_raw = (
                "NVIDIA GeForce RTX 4060 Ti, "
                "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08, "
                f"16380, {used // (1024 * 1024)}\n"
            ).encode("ascii")
            process_raw = (
                f"123, {process_used // (1024 * 1024)}\n".encode("ascii")
                if kind == "ready"
                else b"\n"
            )
            swap_raw = b"SwapTotal:       100 kB\nSwapFree:        100 kB\n"
            sample_bundle = {
                "device_stdout_sha256": joint.sha256_bytes(device_raw),
                "kind": kind,
                "process_stdout_sha256": joint.sha256_bytes(process_raw),
                "sample_completed_ns": event_ns,
                "sample_started_ns": event_ns - 1,
                "swap_raw_sha256": joint.sha256_bytes(swap_raw),
            }
            for suffix, raw in (
                ("device.stdout", device_raw),
                ("process.stdout", process_raw),
                ("meminfo", swap_raw),
                ("sample.json", joint.canonical_bytes(sample_bundle)),
            ):
                evidence.append(self.evidence_file(f"{kind}.{suffix}", raw))
            row = {
                "clock_id": "HOST_MONOTONIC_RAW",
                "device_name": "NVIDIA GeForce RTX 4060 Ti",
                "device_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
                "event_ns": event_ns,
                "free_bytes": 17_175_674_880 - used,
                "host_swap_used_bytes": 0,
                "kind": kind,
                "kv_buffer_bytes": 1_000_000_000 if kind == "ready" else 0,
                "model_buffer_bytes": 6_000_000_000 if kind == "ready" else 0,
                "placement_compute_nodes": 1 if kind == "ready" else 0,
                "process_pid": 123 if kind == "ready" else 0,
                "process_used_bytes": process_used,
                "sample_id": joint.sha256_bytes(
                    joint.canonical_bytes(sample_bundle)
                ),
                "sampler_sha256": "b" * 64,
                "timestamp_ns": event_ns,
                "used_bytes": used,
                "memory_total_bytes": 17_175_674_880,
            }
            path = self.root / f"cuda-memory-{kind}.json"
            raw = joint.canonical_bytes(row)
            path.write_bytes(raw)
            rows.append(row)
            samples.append({
                "bytes": len(raw),
                "path": str(path),
                "row": copy.deepcopy(row),
                "sha256": joint.sha256_bytes(raw),
            })
        return {
            "bridge_ready_row": {"event_ns": 20, "kind": "cuda_ready"},
            "bridge_start_row": {"event_ns": 10, "kind": "cuda_load_start"},
            "cuda_memory_rows": rows,
            "cuda_route_rows": mechanics_rows(
                self.history,
                self.groups,
                "CUDA0",
            ),
            "evidence_artifacts": evidence,
            "execution_groups": copy.deepcopy(self.groups),
            "memory_certificate": {
                "compute_buffer_bytes": 100,
                "host_compute_buffer_bytes": 20,
                "host_context_buffer_bytes": 30,
                "host_model_buffer_bytes": 40,
                "kv_buffer_bytes": 1_000_000_000,
                "model_buffer_bytes": 6_000_000_000,
                "pid": 123,
                "role": "monov3",
                "schema": "layersplit-memory-breakdown-v1",
            },
            "model_sha256": "a" * 64,
            "placement_certificate": {
                "compute_by_buffer_type": {"CUDA0": 1},
                "layer_end": 40,
                "layer_start": 0,
                "missing_buffer_compute_nodes": 0,
                "status": "SCHEDULED_PLACEMENT_OK",
            },
            "protocol_identity": {
                "capabilities": 0x3F,
                "file_type": 15,
                "layer_end": 40,
                "layer_start": 0,
                "max_streams": 8,
                "model_sha256": "a" * 64,
                "n_batch": 64,
                "n_ctx_seq": 512,
                "n_embd": 5120,
                "n_layer": 40,
                "n_ubatch": 64,
                "schema": "layersplit-stage-v3-identity-v1",
                "stage_identity_version": 1,
                "stage_protocol_version": 3,
            },
            "quality_cuda_rows": self.quality_rows(),
            "raw_memory_samples": samples,
            "runtime_process": {
                "bundle_id": "cuda_route",
                "endpoint": "cuda",
                "pid": 123,
                "start_ticks": 1,
            },
        }

    def test_valid_phone_receipt(self):
        continuations = joint.validate_phone_rich(
            self.phone_value(),
            self.history,
            "phone",
        )
        self.assertEqual(sorted(continuations), list(range(64)))

    def test_wrong_wire_identity_is_rejected(self):
        value = self.phone_value()
        value["execution_groups"][0]["call_receipts"][0]["rows"][0][
            "wire_request_id"
        ] += 1
        with self.assertRaisesRegex(joint.CaptureError, "wire_request_id"):
            joint.validate_phone_rich(value, self.history, "phone")

    def test_fabricated_continuation_is_rejected(self):
        value = self.phone_value()
        value["execution_groups"][0]["continuations"][0][0] += 1
        with self.assertRaisesRegex(joint.CaptureError, "continuations"):
            joint.validate_phone_rich(value, self.history, "phone")

    def test_cpu_fallback_is_rejected(self):
        value = self.phone_value()
        value["placement_op12_rows"][1]["backend"] = "CPU"
        with self.assertRaisesRegex(joint.CaptureError, "E_PHONE_FALLBACK"):
            joint.validate_phone_rich(value, self.history, "phone")

    def test_link_byte_shortfall_is_rejected(self):
        value = self.phone_value()
        value["raw_probes"]["op12"]["after"]["interface"]["rx_bytes"] = 1
        with self.assertRaisesRegex(joint.CaptureError, "E_OP12_LINK_COUNTER"):
            joint.validate_phone_rich(value, self.history, "phone")

    def test_cleanup_failure_is_rejected(self):
        value = self.phone_value()
        value["op15_runtime"]["active_sequences_after_cleanup"] = 1
        with self.assertRaisesRegex(joint.CaptureError, "cleanup"):
            joint.validate_phone_rich(value, self.history, "phone")

    def test_memory_summary_without_raw_file_is_rejected(self):
        value = {
            "bridge_ready_row": {"event_ns": 20, "kind": "cuda_ready"},
            "bridge_start_row": {"event_ns": 10, "kind": "cuda_load_start"},
            "cuda_memory_rows": [],
            "cuda_route_rows": mechanics_rows(
                self.history,
                self.groups,
                "CUDA0",
            ),
            "evidence_artifacts": [self.evidence_file("cuda.log")],
            "execution_groups": copy.deepcopy(self.groups),
            "placement_certificate": {
                "compute_by_buffer_type": {"CUDA0": 1},
                "layer_end": 40,
                "layer_start": 0,
                "missing_buffer_compute_nodes": 0,
                "status": "SCHEDULED_PLACEMENT_OK",
            },
            "quality_cuda_rows": self.quality_rows(),
            "raw_memory_samples": [],
            "runtime_process": {
                "bundle_id": "cuda_route",
                "endpoint": "cuda",
                "pid": 1,
                "start_ticks": 1,
            },
        }
        with self.assertRaisesRegex(joint.CaptureError, "E_MEMORY_SAMPLES"):
            joint.validate_cuda_rich(value, self.history, "cuda")

    def test_cuda_protocol_and_memory_certificate_are_load_bearing(self):
        value = self.cuda_value()
        continuations = joint.validate_cuda_rich(value, self.history, "cuda")
        self.assertEqual(sorted(continuations), list(range(64)))

        value = self.cuda_value()
        value["protocol_identity"]["n_batch"] = 32
        with self.assertRaisesRegex(joint.CaptureError, "protocol_identity.n_batch"):
            joint.validate_cuda_rich(value, self.history, "cuda")

        value = self.cuda_value()
        value["protocol_identity"]["file_type"] = 7
        with self.assertRaisesRegex(
            joint.CaptureError,
            "protocol_identity.file_type",
        ):
            joint.validate_cuda_rich(value, self.history, "cuda")

        value = self.cuda_value()
        value["memory_certificate"]["model_buffer_bytes"] += 1
        with self.assertRaisesRegex(joint.CaptureError, "memory_certificate.model"):
            joint.validate_cuda_rich(value, self.history, "cuda")

        value = self.cuda_value()
        value["memory_certificate"]["pid"] += 1
        with self.assertRaisesRegex(joint.CaptureError, "memory_certificate.pid"):
            joint.validate_cuda_rich(value, self.history, "cuda")

        value = self.cuda_value()
        artifact = next(
            record
            for record in value["evidence_artifacts"]
            if record["path"].endswith("ready.process.stdout")
        )
        raw = b"123, 6144\n"
        Path(artifact["path"]).write_bytes(raw)
        artifact["bytes"] = len(raw)
        artifact["sha256"] = joint.sha256_bytes(raw)
        with self.assertRaisesRegex(joint.CaptureError, "raw_memory.ready.process.used"):
            joint.validate_cuda_rich(value, self.history, "cuda")

    def test_program_digest_binding_is_load_bearing(self):
        source = self.evidence_file("producer.py", b"#!/usr/bin/python3\n")
        launch = self.evidence_file("launch.json", b"{}\n")
        template = [
            source["path"],
            launch["path"],
            "{acquisition_started_ns}",
            "{command_plan_sha256}",
            "{output_path}",
            "{phase_id}",
            "{pre_dir}",
        ]
        command = {
            "argv_template": template,
            "cwd": str(self.root),
            "environment": {"LC_ALL": "C"},
            "executed_files": [
                {"argv_index": 0, **source},
                {"argv_index": 1, **launch},
            ],
            "launch_plan_argv_index": 1,
            "launch_plan_sha256": launch["sha256"],
            "producer_sha256": source["sha256"],
            "result_filename": "result.json",
            "timeout_seconds": 1,
        }
        joint.validate_command(command, "command")
        command["producer_sha256"] = "f" * 64
        with self.assertRaisesRegex(joint.CaptureError, "producer_binding"):
            joint.validate_command(command, "command")


if __name__ == "__main__":
    unittest.main()
