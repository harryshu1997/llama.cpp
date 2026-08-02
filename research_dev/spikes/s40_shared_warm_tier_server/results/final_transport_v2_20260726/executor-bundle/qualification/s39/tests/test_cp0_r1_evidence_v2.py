import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


S39 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S39))

import build_cp0_r1_v2 as builder
import cp0_r1_evidence_v2 as evidence
import cp0_r1_preflight_v2 as preflight
import validate_cp0_r1_preflight_v2 as preflight_validator


def marker(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class BundleFixture:
    def __init__(self, root, contract, contract_raw, candidate, candidate_raw):
        self.root = root
        self.contract = contract
        self.contract_raw = contract_raw
        self.candidate = candidate
        self.candidate_raw = candidate_raw
        self.acquisition_id = "cp0-r1-v2-test"
        self.clock_id = "HOST_MONOTONIC_RAW"
        self.started_ns = 1_000_000
        self.rows = {}
        self.locks = {}
        self.allocations = {}
        self._build()

    def common(self, role, kind):
        return {
            "acquisition_id": self.acquisition_id,
            "kind": kind,
            "role": role,
        }

    def route_lock(self, model, index):
        role = f"model.{model['model_id']}.route_lock"
        cut = 30 if index == 0 else 24
        op15_end = 32 if index == 0 else 28
        op12_start = 24 if index == 0 else 20
        row = {
            **self.common(role, "route_lock"),
            "backend": "GPUOpenCL",
            "batch_config_sha256": evidence.digest_json(
                self.contract["serving_envelope"]
            ),
            "clock_id": self.clock_id,
            "cut_layer": cut,
            "frozen_ns": self.started_ns - 1,
            "model_id": model["model_id"],
            "model_sha256": model["artifact"]["sha256"],
            "n_layer": model["n_layer"],
            "op12_shard_sha256": marker(f"op12-shard-{index}"),
            "op12_stored_layers": [op12_start, model["n_layer"]],
            "op15_shard_sha256": marker(f"op15-shard-{index}"),
            "op15_stored_layers": [0, op15_end],
        }
        self.locks[model["model_id"]] = row
        self.rows[role] = [row]

    def execution(self, model, role_suffix, backend, program_label, token_offset=0):
        role = f"model.{model['model_id']}.{role_suffix}"
        rows = [
            {
                **self.common(role, "meta"),
                "backend": backend,
                "call_shapes": [
                    {
                        "call_index": 0,
                        "n_seqs": 8,
                        "n_tokens": 16,
                        "phase": "prefill",
                    },
                    {
                        "call_index": 1,
                        "n_seqs": 8,
                        "n_tokens": 8,
                        "phase": "decode",
                    },
                ],
                "model_id": model["model_id"],
                "model_sha256": model["artifact"]["sha256"],
                "program_sha256": marker(program_label),
                "state_count_after": 0,
                "state_count_before": 0,
            }
        ]
        owner = "PHONE" if backend == "PHONE_COLLECTIVE" else "CUDA"
        for request_id in range(8):
            rows.append(
                {
                    **self.common(role, "request"),
                    "continuation_tokens": [
                        300 + request_id + token_offset,
                        400 + request_id + token_offset,
                    ],
                    "input_tokens": [100 + request_id, 200 + request_id],
                    "model_id": model["model_id"],
                    "model_sha256": model["artifact"]["sha256"],
                    "owner_after": "RELEASED",
                    "owner_before": owner,
                    "ownership_epoch_after": 2,
                    "ownership_epoch_before": 1,
                    "positions": [0, 1],
                    "request_id": request_id,
                }
            )
        self.rows[role] = rows

    def cuda_memory(self, model, index):
        role = f"model.{model['model_id']}.cuda_memory"
        total = self.contract["devices"]["cuda"]["memory_total_bytes"]
        idle = 300_000_000
        model_bytes = 8_800_000_000 if index == 0 else 8_400_000_000
        kv_bytes = 200_000_000
        ready_used = idle + model_bytes + kv_bytes
        self.allocations[model["model_id"]] = {
            "kv_buffer_bytes": kv_bytes,
            "model_buffer_bytes": model_bytes,
        }
        rows = []
        for kind, timestamp, used in (
            ("before", 2_000_000 + index * 10_000, idle),
            ("ready", 2_100_000 + index * 10_000, ready_used),
            ("after", 2_200_000 + index * 10_000, idle),
        ):
            ready = kind == "ready"
            rows.append(
                {
                    **self.common(role, kind),
                    "batch": 8 if ready else 0,
                    "clock_id": self.clock_id,
                    "completed_requests": 8 if ready else 0,
                    "config_sha256": (
                        evidence.digest_json(self.contract["serving_envelope"])
                        if ready
                        else "NONE"
                    ),
                    "device_name": self.contract["devices"]["cuda"]["name"],
                    "device_uuid": self.contract["devices"]["cuda"]["uuid"],
                    "free_bytes": total - used,
                    "host_swap_used_bytes": 100,
                    "kv_buffer_bytes": kv_bytes if ready else 0,
                    "memory_total_bytes": total,
                    "model_buffer_bytes": model_bytes if ready else 0,
                    "model_id": model["model_id"],
                    "model_sha256": model["artifact"]["sha256"],
                    "placement_compute_nodes": 100 if ready else 0,
                    "state_count": 8 if ready else 0,
                    "timestamp_ns": timestamp,
                    "used_bytes": used,
                }
            )
        self.rows[role] = rows

    def quality_outputs(self, model, role_suffix):
        role = f"model.{model['model_id']}.quality.{role_suffix}"
        rows = []
        for index in range(64):
            answer = "ABCD"[index % 4]
            prompt = self.corpus_prompt(index)
            rows.append(
                {
                    **self.common(role, "output"),
                    "item_index": index,
                    "model_id": model["model_id"],
                    "model_sha256": model["artifact"]["sha256"],
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "raw_output": answer,
                }
            )
        self.rows[role] = rows

    def corpus_prompt(self, index):
        return self.candidate["task_suite"]["prompt_format"].format(
            question=f"Question {index}?",
            choice0=f"A{index}",
            choice1=f"B{index}",
            choice2=f"C{index}",
            choice3=f"D{index}",
        )

    def corpus(self):
        role = "quality.corpus"
        self.rows[role] = [
            {
                **self.common(role, "item"),
                "choices": [f"A{index}", f"B{index}", f"C{index}", f"D{index}"],
                "dataset": self.candidate["task_suite"]["dataset"],
                "dataset_revision": self.candidate["task_suite"]["revision"],
                "expected_answer": "ABCD"[index % 4],
                "item_index": index,
                "question": f"Question {index}?",
                "source_row": index,
                "subject": f"subject-{index % 4}",
            }
            for index in range(64)
        ]

    def bridge(self, model, index):
        role = f"model.{model['model_id']}.bridge"
        start = 3_000_000 + index * 100_000
        rows = [
            {
                **self.common(role, "cuda_load_start"),
                "clock_id": self.clock_id,
                "model_id": model["model_id"],
                "model_sha256": model["artifact"]["sha256"],
                "timestamp_ns": start,
            }
        ]
        for request_id in range(8):
            rows.append(
                {
                    **self.common(role, "phone_publication_received"),
                    "clock_id": self.clock_id,
                    "model_id": model["model_id"],
                    "model_sha256": model["artifact"]["sha256"],
                    "request_id": request_id,
                    "timestamp_ns": start + 10_000 + request_id,
                    "token_ids": [500 + request_id],
                }
            )
        rows.append(
            {
                **self.common(role, "cuda_ready"),
                "clock_id": self.clock_id,
                "model_id": model["model_id"],
                "model_sha256": model["artifact"]["sha256"],
                "timestamp_ns": start + 50_000,
            }
        )
        self.rows[role] = rows

    def placement(self, model, phone):
        role = f"model.{model['model_id']}.placement.{phone}"
        lock = self.locks[model["model_id"]]
        device = self.contract["devices"][phone]
        executed = (
            [0, lock["cut_layer"]]
            if phone == "op15"
            else [lock["cut_layer"], model["n_layer"]]
        )
        self.rows[role] = [
            {
                **self.common(role, "meta"),
                "available_after_bytes": 700_000_000,
                "available_before_bytes": 800_000_000,
                "batch": 8,
                "boot_id": (
                    "11111111-1111-4111-8111-111111111111"
                    if phone == "op15"
                    else "22222222-2222-4222-8222-222222222222"
                ),
                "device": device["device"],
                "executed_layers": executed,
                "model": device["model"],
                "model_id": model["model_id"],
                "model_sha256": model["artifact"]["sha256"],
                "process_swap_bytes": 0,
                "product": device["product"],
                "serial": device["serial"],
                "shard_sha256": lock[f"{phone}_shard_sha256"],
                "stored_layers": lock[f"{phone}_stored_layers"],
                "system_swap_after_bytes": 100,
                "system_swap_before_bytes": 100,
            },
            {
                **self.common(role, "node"),
                "backend": "GPUOpenCL",
                "compute": True,
                "missing_buffer": False,
                "node_id": 0,
                "op": "MUL_MAT",
            },
            {
                **self.common(role, "node"),
                "backend": "CPU",
                "compute": True,
                "missing_buffer": False,
                "node_id": 1,
                "op": "GET_ROWS",
            },
        ]

    def transfer(self, model):
        role = f"model.{model['model_id']}.route_transfer"
        lock = self.locks[model["model_id"]]
        self.rows[role] = [
            {
                **self.common(role, "meta"),
                "batch": 8,
                "cut_layer": lock["cut_layer"],
                "model_id": model["model_id"],
                "model_sha256": model["artifact"]["sha256"],
                "request_ids": list(range(8)),
            },
            {
                **self.common(role, "transfer"),
                "host_payload_bytes": 0,
                "path": "WIFI_TCP_DIRECT",
                "payload_bytes": 1_000_000,
                "payload_sha256": marker(f"payload-{model['model_id']}"),
                "receiver": "op12",
                "sender": "op15",
            },
        ]

    def pair_memory(self):
        role = "pair.cuda_memory"
        total = self.contract["devices"]["cuda"]["memory_total_bytes"]
        before = {
            **self.common(role, "before"),
            "clock_id": self.clock_id,
            "device_uuid": self.contract["devices"]["cuda"]["uuid"],
            "free_bytes": total - 300_000_000,
            "host_swap_used_bytes": 100,
            "memory_total_bytes": total,
            "timestamp_ns": 4_000_000,
            "used_bytes": 300_000_000,
        }
        after = copy.deepcopy(before)
        after["kind"] = "after"
        after["timestamp_ns"] = 4_200_000
        model_ids = [model["model_id"] for model in self.candidate["models"]]
        self.rows[role] = [
            before,
            {
                **self.common(role, "attempt"),
                "clock_id": self.clock_id,
                "config_sha256": evidence.digest_json(
                    self.contract["serving_envelope"]
                ),
                "device_uuid": self.contract["devices"]["cuda"]["uuid"],
                "exit_code": 1,
                "free_bytes": 100_000_000,
                "kv_buffer_bytes": [
                    self.allocations[item]["kv_buffer_bytes"] for item in model_ids
                ],
                "memory_total_bytes": total,
                "model_buffer_bytes": [
                    self.allocations[item]["model_buffer_bytes"] for item in model_ids
                ],
                "model_ids": model_ids,
                "model_sha256s": [
                    model["artifact"]["sha256"] for model in self.candidate["models"]
                ],
                "outcome": "CUDA_OOM",
                "timestamp_ns": 4_100_000,
            },
            after,
        ]

    def reprepare(self, from_model, to_model, direction, start):
        role = f"reprepare.{direction}"
        rows = [
            {
                **self.common(role, "start"),
                "clock_id": self.clock_id,
                "from_model_id": from_model["model_id"],
                "timestamp_ns": start,
                "to_model_id": to_model["model_id"],
            }
        ]
        for phone in ("op15", "op12"):
            device = self.contract["devices"][phone]
            rows.append(
                {
                    **self.common(role, "phone_before"),
                    "boot_id": (
                        "11111111-1111-4111-8111-111111111111"
                        if phone == "op15"
                        else "22222222-2222-4222-8222-222222222222"
                    ),
                    "local_ufs_read_bytes": 1_000,
                    "network_weight_bytes": 2_000,
                    "phone": phone,
                    "ready_generation": 1,
                    "serial": device["serial"],
                    "state_count": 8,
                    "usb_weight_bytes": 3_000,
                }
            )
        to_lock = self.locks[to_model["model_id"]]
        for offset, phone in enumerate(("op15", "op12"), start=1):
            device = self.contract["devices"][phone]
            rows.append(
                {
                    **self.common(role, "phone_ready"),
                    "boot_id": (
                        "11111111-1111-4111-8111-111111111111"
                        if phone == "op15"
                        else "22222222-2222-4222-8222-222222222222"
                    ),
                    "host_received_ns": start + 1_000_000 + offset,
                    "local_ufs_read_bytes": 101_000,
                    "model_sha256": to_model["artifact"]["sha256"],
                    "network_weight_bytes": 2_000,
                    "phone": phone,
                    "ready_generation": 2,
                    "serial": device["serial"],
                    "shard_sha256": to_lock[f"{phone}_shard_sha256"],
                    "state_count": 0,
                    "usb_weight_bytes": 3_000,
                }
            )
        rows.append(
            {
                **self.common(role, "end"),
                "clock_id": self.clock_id,
                "from_model_id": from_model["model_id"],
                "timestamp_ns": start + 2_000_000,
                "to_model_id": to_model["model_id"],
            }
        )
        self.rows[role] = rows

    def _build(self):
        self.corpus()
        for index, model in enumerate(self.candidate["models"]):
            self.route_lock(model, index)
            self.execution(
                model,
                "mechanics.phone",
                "PHONE_COLLECTIVE",
                f"phone-program-{index}",
            )
            self.execution(
                model,
                "oracle.cuda_route",
                "CUDA0",
                f"cuda-route-program-{index}",
            )
            self.execution(
                model,
                "oracle.cuda_monolithic",
                "CUDA0",
                f"cuda-oracle-program-{index}",
            )
            self.cuda_memory(model, index)
            self.quality_outputs(model, "cuda")
            self.quality_outputs(model, "phone")
            self.bridge(model, index)
            self.placement(model, "op15")
            self.placement(model, "op12")
            self.transfer(model)
        self.pair_memory()
        self.reprepare(
            self.candidate["models"][0],
            self.candidate["models"][1],
            "A_to_B",
            5_000_000,
        )
        self.reprepare(
            self.candidate["models"][1],
            self.candidate["models"][0],
            "B_to_A",
            8_000_000,
        )
        self.write()

    def write(self):
        raw_dir = self.root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for index, role in enumerate(self.contract["raw_evidence"]["required_roles"]):
            raw = b"".join(evidence.canonical_line(row) for row in self.rows[role])
            path = f"raw/{index:02d}.jsonl"
            (self.root / path).write_bytes(raw)
            artifacts.append(
                {
                    "bytes": len(raw),
                    "format": "CANONICAL_ASCII_JSONL",
                    "path": path,
                    "role": role,
                    "sha256": evidence.sha256_bytes(raw),
                }
            )
        manifest = {
            "acquisition_id": self.acquisition_id,
            "acquisition_started_ns": self.started_ns,
            "artifacts": artifacts,
            "candidate_attempt": 1,
            "candidate_sha256": evidence.sha256_bytes(self.candidate_raw),
            "clock_id": self.clock_id,
            "contract_sha256": evidence.sha256_bytes(self.contract_raw),
            "schema": "s39-cp0-r1-evidence-bundle-v2",
        }
        (self.root / "EVIDENCE_BUNDLE.json").write_bytes(
            evidence.canonical_bytes(manifest)
        )

    def mutate(self, role, callback):
        callback(self.rows[role])
        self.write()


class Cp0R1EvidenceV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.contract_raw = evidence.load_canonical_path(
            S39 / "CP0_R1_EVIDENCE_CONTRACT_V2.json"
        )
        cls.candidate, cls.candidate_raw = evidence.load_canonical_path(
            S39 / "CP0_R1_CANDIDATE.json"
        )

    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        fixture = BundleFixture(
            root,
            self.contract,
            self.contract_raw,
            self.candidate,
            self.candidate_raw,
        )
        return fixture

    def evaluate(self, fixture, candidate_path=None):
        candidate = self.candidate
        candidate_raw = self.candidate_raw
        if candidate_path is not None:
            candidate, candidate_raw = evidence.load_canonical_path(candidate_path)
        manifest, rows = evidence.load_bundle(
            fixture.root,
            "EVIDENCE_BUNDLE.json",
            self.contract,
            self.contract_raw,
            candidate_raw,
        )
        return evidence.evaluate(
            self.contract,
            self.contract_raw,
            candidate,
            candidate_raw,
            manifest,
            rows,
        )

    def assert_mutation_fails(self, role, callback, message):
        fixture = self.fixture()
        fixture.mutate(role, callback)
        with self.assertRaisesRegex(evidence.EvidenceError, message):
            self.evaluate(fixture)

    def test_builder_is_byte_deterministic(self):
        self.assertEqual(
            (S39 / "CP0_R1_EVIDENCE_CONTRACT_V2.json").read_bytes(),
            builder.canonical_bytes(builder.build_contract()),
        )

    def test_parent_v1_files_are_immutable(self):
        for relative, expected in builder.PARENT_FILES.items():
            self.assertEqual(builder.sha256_file(S39 / relative), expected, relative)

    def test_parent_candidate_digest_cannot_be_rebound(self):
        contract = copy.deepcopy(self.contract)
        contract["parent"]["candidate_sha256"] = marker("other-candidate")
        with self.assertRaisesRegex(evidence.EvidenceError, "parent.candidate_sha256"):
            evidence.validate_contract(contract)

    def test_quality_gate_cannot_be_weakened(self):
        contract = copy.deepcopy(self.contract)
        contract["gates"]["quality_maximum_new_errors"] = 64
        with self.assertRaisesRegex(evidence.EvidenceError, "gates"):
            evidence.validate_contract(contract)

    def test_complete_raw_bundle_passes(self):
        result = self.evaluate(self.fixture())
        self.assertEqual(result["status"], "TWO_ROUTE_ELIGIBILITY_PASS")
        self.assertEqual(
            result["derived"]["models"]["qwen3-14b-q4_k_m"]["quality"][
                "phone_correct"
            ],
            64,
        )

    def test_contract_only_cli_is_not_eligibility(self):
        proc = subprocess.run(
            [sys.executable, str(S39 / "cp0_r1_evidence_v2.py")],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("RAW_EVIDENCE_CONTRACT_READY_ACQUISITION_NOT_RUN", proc.stdout)
        self.assertNotIn('"status":"TWO_ROUTE_ELIGIBILITY_PASS"', proc.stdout)

    def test_candidate_rebinding_is_rejected(self):
        fixture = self.fixture()
        candidate = copy.deepcopy(self.candidate)
        candidate["models"][1]["model_id"] = "replacement-candidate"
        path = fixture.root / "candidate.json"
        path.write_bytes(evidence.canonical_bytes(candidate))
        manifest = json.loads((fixture.root / "EVIDENCE_BUNDLE.json").read_bytes())
        manifest["candidate_sha256"] = evidence.sha256_bytes(path.read_bytes())
        (fixture.root / "EVIDENCE_BUNDLE.json").write_bytes(
            evidence.canonical_bytes(manifest)
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "candidate.sha256"):
            self.evaluate(fixture, path)

    def test_missing_role_is_rejected(self):
        fixture = self.fixture()
        manifest = json.loads((fixture.root / "EVIDENCE_BUNDLE.json").read_bytes())
        manifest["artifacts"].pop()
        (fixture.root / "EVIDENCE_BUNDLE.json").write_bytes(
            evidence.canonical_bytes(manifest)
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "artifact count"):
            self.evaluate(fixture)

    def test_role_path_reuse_is_rejected(self):
        fixture = self.fixture()
        manifest = json.loads((fixture.root / "EVIDENCE_BUNDLE.json").read_bytes())
        manifest["artifacts"][1]["path"] = manifest["artifacts"][0]["path"]
        manifest["artifacts"][1]["sha256"] = manifest["artifacts"][0]["sha256"]
        manifest["artifacts"][1]["bytes"] = manifest["artifacts"][0]["bytes"]
        (fixture.root / "EVIDENCE_BUNDLE.json").write_bytes(
            evidence.canonical_bytes(manifest)
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "PATH_REUSE"):
            self.evaluate(fixture)

    def test_missing_file_is_rejected(self):
        fixture = self.fixture()
        (fixture.root / "raw/00.jsonl").unlink()
        with self.assertRaisesRegex(evidence.EvidenceError, "E_PATH"):
            self.evaluate(fixture)

    def test_digest_rebinding_does_not_hide_bad_oracle(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.oracle.cuda_monolithic"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[1].__setitem__("continuation_tokens", [999]),
            "oracle.continuation",
        )

    def test_oracle_program_alias_is_rejected(self):
        model_id = self.candidate["models"][0]["model_id"]
        route_role = f"model.{model_id}.oracle.cuda_route"
        oracle_role = f"model.{model_id}.oracle.cuda_monolithic"
        fixture = self.fixture()
        fixture.rows[oracle_role][0]["program_sha256"] = fixture.rows[route_role][0][
            "program_sha256"
        ]
        fixture.write()
        with self.assertRaisesRegex(evidence.EvidenceError, "ORACLE_INDEPENDENCE"):
            self.evaluate(fixture)

    def test_cross_geometry_cuda_oracle_is_rejected(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.oracle.cuda_monolithic"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[0]["call_shapes"][0].__setitem__("n_tokens", 15),
            "oracle.call_shapes",
        )

    def test_route_lock_coverage_gap_is_rejected(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.route_lock"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[0].__setitem__("op15_stored_layers", [0, 20]),
            "COVERAGE",
        )

    def test_phone_cross_backend_token_difference_is_diagnostic(self):
        fixture = self.fixture()
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.mechanics.phone"
        fixture.rows[role][1]["continuation_tokens"][0] = 999
        fixture.write()
        result = self.evaluate(fixture)
        diagnostic = result["derived"]["models"][model_id]["oracle"]
        self.assertLess(
            diagnostic["cross_backend_greedy_matches"],
            diagnostic["cross_backend_greedy_total"],
        )

    def test_fabricated_cuda_headroom_summary_cannot_override_raw(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.cuda_memory"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[1].__setitem__("free_bytes", 1),
            "HEADROOM",
        )

    def test_pair_success_cannot_be_labeled_nonresident(self):
        self.assert_mutation_fails(
            "pair.cuda_memory",
            lambda rows: rows[1].__setitem__("outcome", "SUCCESS"),
            "pair.cuda_memory.outcome",
        )

    def test_per_item_quality_regression_is_rejected(self):
        model_id = self.candidate["models"][1]["model_id"]
        role = f"model.{model_id}.quality.phone"

        def mutate(rows):
            rows[0]["raw_output"] = "B"
            rows[1]["raw_output"] = "C"

        self.assert_mutation_fails(role, mutate, "QUALITY_NEW_ERRORS")

    def test_quality_output_cannot_change_prompt(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.quality.phone"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[0].__setitem__("prompt_sha256", marker("other")),
            "prompt_sha256",
        )

    def test_bridge_mixed_clock_is_rejected(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.bridge"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[1].__setitem__("clock_id", "PHONE_BOOTTIME"),
            "clock",
        )

    def test_late_publication_is_rejected(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.bridge"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[1].__setitem__(
                "timestamp_ns", rows[-1]["timestamp_ns"]
            ),
            "BRIDGE_ORDER",
        )

    def test_cpu_compute_fallback_is_rejected(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.placement.op15"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[1].update({"backend": "CPU", "op": "MUL_MAT"}),
            "CPU_FALLBACK",
        )

    def test_false_phone_headroom_is_rejected(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.placement.op12"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[0].__setitem__("available_after_bytes", 1),
            "PHONE_HEADROOM",
        )

    def test_host_relay_is_rejected(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.route_transfer"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[1].__setitem__("host_payload_bytes", 1),
            "host_payload",
        )

    def test_reprepare_usb_bytes_are_rejected(self):
        self.assert_mutation_fails(
            "reprepare.A_to_B",
            lambda rows: rows[3].__setitem__("usb_weight_bytes", 3_001),
            "usb_weight_bytes",
        )

    def test_reprepare_without_live_state_is_rejected(self):
        self.assert_mutation_fails(
            "reprepare.A_to_B",
            lambda rows: rows[1].__setitem__("state_count", 0),
            "RELEASE_UNEXERCISED",
        )

    def test_reprepare_wrong_shard_is_rejected(self):
        self.assert_mutation_fails(
            "reprepare.B_to_A",
            lambda rows: rows[3].__setitem__("shard_sha256", marker("wrong")),
            "shard",
        )

    def test_reprepare_dwell_bound_is_rejected(self):
        self.assert_mutation_fails(
            "reprepare.A_to_B",
            lambda rows: rows[-1].__setitem__(
                "timestamp_ns", rows[0]["timestamp_ns"] + 31_000_000_000
            ),
            "REPREPARE_DWELL",
        )

    def test_float_cannot_pass_integer_gate(self):
        model_id = self.candidate["models"][0]["model_id"]
        role = f"model.{model_id}.cuda_memory"
        self.assert_mutation_fails(
            role,
            lambda rows: rows[1].__setitem__("batch", 8.0),
            "expected integer",
        )

    def test_summary_field_is_rejected(self):
        fixture = self.fixture()
        manifest = json.loads((fixture.root / "EVIDENCE_BUNDLE.json").read_bytes())
        manifest["reported_status"] = "TWO_ROUTE_ELIGIBILITY_PASS"
        (fixture.root / "EVIDENCE_BUNDLE.json").write_bytes(
            evidence.canonical_bytes(manifest)
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "unknown=.*reported_status"):
            self.evaluate(fixture)

    def test_symlink_artifact_is_rejected(self):
        fixture = self.fixture()
        target = fixture.root / "raw/00.jsonl"
        copy_path = fixture.root / "copy.jsonl"
        copy_path.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(copy_path)
        with self.assertRaisesRegex(evidence.EvidenceError, "E_PATH"):
            self.evaluate(fixture)

    def test_hardlink_artifact_is_rejected(self):
        fixture = self.fixture()
        target = fixture.root / "raw/00.jsonl"
        os.link(target, fixture.root / "alias.jsonl")
        with self.assertRaisesRegex(evidence.EvidenceError, "E_HARDLINK"):
            self.evaluate(fixture)

    def test_duplicate_raw_json_key_is_rejected_after_rehash(self):
        fixture = self.fixture()
        manifest_path = fixture.root / "EVIDENCE_BUNDLE.json"
        manifest = json.loads(manifest_path.read_bytes())
        artifact = manifest["artifacts"][0]
        raw = (
            b'{"acquisition_id":"cp0-r1-v2-test",'
            b'"acquisition_id":"cp0-r1-v2-test",'
            b'"kind":"item","role":"quality.corpus"}\n'
        )
        (fixture.root / artifact["path"]).write_bytes(raw)
        artifact["bytes"] = len(raw)
        artifact["sha256"] = evidence.sha256_bytes(raw)
        manifest_path.write_bytes(evidence.canonical_bytes(manifest))
        with self.assertRaisesRegex(evidence.EvidenceError, "DUPLICATE_KEY"):
            self.evaluate(fixture)

    def test_preflight_parsers_require_exact_identities(self):
        adb = preflight.parse_adb_devices(
            "List of devices attached\n"
            "5ae7a43d device product:CPH2583 model:CPH2583 device:OP595DL1\n"
        )
        self.assertEqual(adb["5ae7a43d"]["device"], "OP595DL1")
        stdout = (
            "zhihao-Z690-C-ac\n"
            "NVIDIA GeForce RTX 4060 Ti, "
            "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08, 16380\n"
        )
        matched, parsed = preflight.parse_gpu(stdout, self.contract)
        self.assertTrue(matched)
        self.assertEqual(parsed["gpus"][0]["memory_total_bytes"], 17_175_674_880)

    def test_checked_in_preflight_is_independently_validated(self):
        run_dir = (
            S39
            / "results"
            / "cp0_r1_preflight_v2"
            / "run_20260725T163055Z"
        )
        result_raw = evidence.secure_read(run_dir, "PREFLIGHT.json")
        result = evidence.parse_json(result_raw, "PREFLIGHT.json")
        manifest_raw = evidence.secure_read(run_dir, "SHA256SUMS.txt")
        validated = preflight_validator.validate(
            result,
            result_raw,
            manifest_raw,
            self.contract,
            self.contract_raw,
            self.candidate_raw,
        )
        self.assertEqual(validated["status"], "NO_MODEL_PREFLIGHT_VALIDATED")

    def test_preflight_gpu_summary_cannot_override_raw_stdout(self):
        run_dir = (
            S39
            / "results"
            / "cp0_r1_preflight_v2"
            / "run_20260725T163055Z"
        )
        result = json.loads((run_dir / "PREFLIGHT.json").read_bytes())
        result["probes"]["cuda"]["stdout"] = result["probes"]["cuda"][
            "stdout"
        ].replace(
            "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "GPU-00000000-0000-0000-0000-000000000000",
        )
        raw = evidence.canonical_bytes(result)
        manifest = f"{evidence.sha256_bytes(raw)}  PREFLIGHT.json\n".encode("ascii")
        with self.assertRaisesRegex(evidence.EvidenceError, "cuda.derived"):
            preflight_validator.validate(
                result,
                raw,
                manifest,
                self.contract,
                self.contract_raw,
                self.candidate_raw,
            )

    def test_preflight_pass_label_cannot_override_phone_failure(self):
        run_dir = (
            S39
            / "results"
            / "cp0_r1_preflight_v2"
            / "run_20260725T163055Z"
        )
        result = json.loads((run_dir / "PREFLIGHT.json").read_bytes())
        result["probes"]["op12"]["stdout"] = ""
        raw = evidence.canonical_bytes(result)
        manifest = f"{evidence.sha256_bytes(raw)}  PREFLIGHT.json\n".encode("ascii")
        with self.assertRaisesRegex(evidence.EvidenceError, "PREFLIGHT_PHONE"):
            preflight_validator.validate(
                result,
                raw,
                manifest,
                self.contract,
                self.contract_raw,
                self.candidate_raw,
            )


if __name__ == "__main__":
    unittest.main()
