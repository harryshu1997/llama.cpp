#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ELIGIBILITY = load_module("s39_cp0_r1_eligibility", S39 / "cp0_r1_eligibility.py")
BUILDER = load_module("s39_cp0_r1_builder", S39 / "build_cp0_r1_contract.py")


def marker(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class Cp0R1EligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.contract_raw = ELIGIBILITY.load_canonical(
            S39 / "CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json"
        )
        cls.candidate, cls.candidate_raw = ELIGIBILITY.load_canonical(
            S39 / "CP0_R1_CANDIDATE.json"
        )

    def finalized_candidate(self):
        value = copy.deepcopy(self.candidate)
        value["status"] = "PAIR_FROZEN_BEFORE_PAID_ACQUISITION"
        model = value["models"][1]
        model["readiness_at_freeze"] = "ROUTE_BINDING_FROZEN_EVIDENCE_PENDING"
        model["route_binding"] = {
            "backend": "GPUOpenCL",
            "executed_cut_layer": 24,
            "op15_stored_layers": [0, 28],
            "op12_stored_layers": [20, 36],
            "op15_shard_sha256": marker("qwen8-op15"),
            "op12_shard_sha256": marker("qwen8-op12"),
        }
        return value

    def cuda_record(self, model, index):
        model_bytes = 8_000_000_000 if index == 0 else 7_000_000_000
        kv_bytes = 1_000_000_000
        peak = 10_000_000_000 if index == 0 else 9_000_000_000
        return {
            "artifacts": [marker(f"cuda-{index}")],
            "batch": 8,
            "completed_requests": 8,
            "config": copy.deepcopy(self.contract["serving_envelope"]),
            "device_name": self.contract["target_server"]["device_name"],
            "device_uuid": self.contract["target_server"]["device_uuid"],
            "free_vram_bytes": (
                self.contract["target_server"]["memory_total_bytes"] - peak
            ),
            "host_swap_used_after_bytes": 0,
            "host_swap_used_before_bytes": 0,
            "kv_buffer_bytes": kv_bytes,
            "model_buffer_bytes": model_bytes,
            "model_sha256": model["artifact"]["sha256"],
            "peak_used_vram_bytes": peak,
            "placement_compute_nodes": 100,
            "placement_status": "SCHEDULED_PLACEMENT_OK",
            "state_count_after": 0,
        }

    def phone_route(self, model, index):
        cut = model["route_binding"]["executed_cut_layer"]
        history = marker(f"history-{index}")
        positions = marker(f"positions-{index}")
        oracle = marker(f"oracle-vector-{index}")
        call_shapes = marker(f"call-shapes-{index}")
        memory = {
            "available_bytes": 700_000_000,
            "process_swap_bytes": 0,
            "system_swap_used_after_bytes": 100,
            "system_swap_used_before_bytes": 100,
        }
        placement = {
            "backend": "GPUOpenCL",
            "compute_nodes": 100,
            "cpu_ops": ["GET_ROWS"],
            "status": "SCHEDULED_PLACEMENT_OK",
        }
        return {
            "activation": {
                "direct_payload_bytes": 1_000_000,
                "host_payload_bytes": 0,
                "path": "OP15_TO_OP12_WIFI_TCP",
            },
            "artifacts": [marker(f"phone-route-{index}")],
            "batch": 8,
            "coverage": {
                "cut_layer": cut,
                "n_layer": model["n_layer"],
                "op12_layers": [cut, model["n_layer"]],
                "op15_layers": [0, cut],
            },
            "mechanics": {
                "cross_backend_greedy_matches": 0,
                "cross_backend_greedy_total": 64,
                "cross_geometry_exact": False,
                "duplicate_tokens": 0,
                "expected_history_sha256": history,
                "expected_positions_sha256": positions,
                "missing_tokens": 0,
                "observed_history_sha256": history,
                "observed_positions_sha256": positions,
                "oracle_expected_sha256": oracle,
                "oracle_backend": "CUDA0",
                "oracle_call_shapes_sha256": call_shapes,
                "oracle_observed_sha256": oracle,
                "oracle_program_sha256": marker(f"oracle-program-{index}"),
                "ownership_transition_count": 1,
                "route_program_sha256": marker(f"route-program-{index}"),
                "route_call_shapes_sha256": call_shapes,
                "stale_tokens": 0,
                "terminal_state_counts": {"cuda": 0, "op12": 0, "op15": 0},
            },
            "memory": {
                "op12": copy.deepcopy(memory),
                "op15": copy.deepcopy(memory),
            },
            "model_sha256": model["artifact"]["sha256"],
            "placement": {
                "op12": copy.deepcopy(placement),
                "op15": copy.deepcopy(placement),
            },
            "quality": {
                "artifacts": [marker(f"quality-{index}")],
                "cross_backend_greedy_matches": 0,
                "cross_backend_greedy_total": 64,
                "cuda_correct": 48,
                "cuda_parsed": 64,
                "dataset": self.candidate["task_suite"]["dataset"],
                "dataset_revision": self.candidate["task_suite"]["revision"],
                "phone_correct": 47,
                "phone_new_errors": 1,
                "phone_parsed": 64,
                "phone_recovered_errors": 0,
                "total": 64,
            },
        }

    def evidence(self, candidate):
        candidate_raw = ELIGIBILITY.canonical_bytes(candidate)
        models = {}
        allocations = {}
        for index, model in enumerate(candidate["models"]):
            cuda = self.cuda_record(model, index)
            allocations[model["model_id"]] = {
                "kv_buffer_bytes": cuda["kv_buffer_bytes"],
                "model_buffer_bytes": cuda["model_buffer_bytes"],
            }
            models[model["model_id"]] = {
                "bridge": {
                    "artifacts": [marker(f"bridge-{index}")],
                    "cuda_ready_ns": 2_000_000_000,
                    "live_phone_publication_ns": 1_000_000_000,
                    "requests_published_before_cuda_ready": 8,
                    "useful_phone_tokens": 8,
                },
                "cuda_b8": cuda,
                "phone_route": self.phone_route(model, index),
            }
        idle = 300_000_000
        headroom = self.contract["target_server"]["minimum_free_vram_bytes"]
        lower_bound = idle + headroom + sum(
            item["kv_buffer_bytes"] + item["model_buffer_bytes"]
            for item in allocations.values()
        )
        directions = [
            (candidate["models"][0]["model_id"], candidate["models"][1]["model_id"]),
            (candidate["models"][1]["model_id"], candidate["models"][0]["model_id"]),
        ]
        return {
            "candidate_attempts": 1,
            "candidate_sha256": ELIGIBILITY.sha256(candidate_raw),
            "contract_sha256": ELIGIBILITY.sha256(self.contract_raw),
            "models": models,
            "pair_capacity": {
                "artifacts": [marker("capacity")],
                "idle_used_vram_bytes": idle,
                "lower_bound_used_vram_bytes": lower_bound,
                "models": allocations,
                "required_headroom_bytes": headroom,
                "target_device_uuid": self.contract["target_server"]["device_uuid"],
                "total_vram_bytes": self.contract["target_server"]["memory_total_bytes"],
            },
            "reported_status": "TWO_ROUTE_ELIGIBILITY_PASS",
            "reprepare": [
                {
                    "artifacts": [marker(f"reprepare-{index}")],
                    "ended_ns": 21_000_000_000,
                    "from_model_id": direction[0],
                    "local_ufs_bytes_read": 1_000_000,
                    "network_weight_bytes": 0,
                    "op12_shard_sha256": candidate["models"][1 - index][
                        "route_binding"
                    ]["op12_shard_sha256"],
                    "op15_shard_sha256": candidate["models"][1 - index][
                        "route_binding"
                    ]["op15_shard_sha256"],
                    "ready_generation_after": 2,
                    "ready_generation_before": 1,
                    "ready_model_sha256": candidate["models"][1 - index]["artifact"][
                        "sha256"
                    ],
                    "released_state_count": 0,
                    "source": "PHONE_LOCAL_UFS_ONLY",
                    "started_ns": 1_000_000_000,
                    "to_model_id": direction[1],
                    "usb_weight_bytes": 0,
                }
                for index, direction in enumerate(directions)
            ],
            "schema": "s39-two-route-eligibility-evidence-v1",
        }

    def evaluate(self, candidate, evidence):
        candidate_raw = ELIGIBILITY.canonical_bytes(candidate)
        return ELIGIBILITY.evaluate(
            self.contract,
            self.contract_raw,
            candidate,
            candidate_raw,
            evidence,
        )

    def test_checked_in_contract_and_candidate_validate(self):
        ELIGIBILITY.validate_contract(self.contract)
        models = ELIGIBILITY.validate_candidate(self.candidate, self.contract_raw)
        self.assertEqual([model["slot"] for model in models], ["A", "B"])

    def test_builder_is_byte_deterministic(self):
        self.assertEqual(
            (S39 / "CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json").read_bytes(),
            BUILDER.canonical_bytes(BUILDER.build_contract()),
        )
        self.assertEqual(
            (S39 / "CP0_R1_CANDIDATE.json").read_bytes(),
            BUILDER.canonical_bytes(BUILDER.build_candidate()),
        )
        prompt = self.candidate["task_suite"]["prompt_format"]
        self.assertIn("\n", prompt)
        self.assertNotIn("\\n", prompt)

    def test_current_cli_reports_not_acquired(self):
        proc = subprocess.run(
            ["python3", str(S39 / "cp0_r1_eligibility.py")],
            cwd=S39,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("CONTRACT_VALID_CANDIDATE_NOT_ACQUIRED", proc.stdout)

    def test_complete_synthetic_evidence_passes(self):
        candidate = self.finalized_candidate()
        result = self.evaluate(candidate, self.evidence(candidate))
        self.assertEqual(result["status"], "TWO_ROUTE_ELIGIBILITY_PASS")
        self.assertTrue(result["cycle_authorized"])
        self.assertFalse(result["trace_authorized"])
        self.assertFalse(result["energy_authorized"])

    def test_current_incomplete_candidate_cannot_evaluate(self):
        with self.assertRaisesRegex(ELIGIBILITY.EligibilityError, "incomplete"):
            self.evaluate(self.candidate, {})

    def mutation_fails(self, mutate, message):
        candidate = self.finalized_candidate()
        evidence = self.evidence(candidate)
        mutate(candidate, evidence)
        if evidence.get("candidate_sha256") is not None:
            evidence["candidate_sha256"] = ELIGIBILITY.sha256(
                ELIGIBILITY.canonical_bytes(candidate)
            )
        with self.assertRaisesRegex(ELIGIBILITY.EligibilityError, message):
            self.evaluate(candidate, evidence)

    def test_second_candidate_attempt_is_rejected(self):
        self.mutation_fails(
            lambda _candidate, evidence: evidence.__setitem__("candidate_attempts", 2),
            "candidate_attempts",
        )

    def test_cuda_batch_below_eight_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][0]["model_id"]
            ]["cuda_b8"].__setitem__("batch", 7),
            "cuda_b8.batch",
        )

    def test_cuda_headroom_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][0]["model_id"]
            ]["cuda_b8"].__setitem__("free_vram_bytes", 1),
            "insufficient GPU headroom",
        )

    def test_phone_coverage_gap_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][1]["model_id"]
            ]["phone_route"]["coverage"].__setitem__("op12_layers", [25, 36]),
            "coverage.op12",
        )

    def test_host_relay_payload_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][0]["model_id"]
            ]["phone_route"]["activation"].__setitem__("host_payload_bytes", 1),
            "activation.host",
        )

    def test_cpu_fallback_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][0]["model_id"]
            ]["phone_route"]["placement"]["op15"]["cpu_ops"].append("MUL_MAT"),
            "cpu_ops",
        )

    def test_phone_memory_headroom_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][1]["model_id"]
            ]["phone_route"]["memory"]["op12"].__setitem__("available_bytes", 1),
            "insufficient memory headroom",
        )

    def test_phone_swap_growth_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][0]["model_id"]
            ]["phone_route"]["memory"]["op15"].__setitem__(
                "system_swap_used_after_bytes", 101
            ),
            "system swap grew",
        )

    def test_history_mismatch_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][0]["model_id"]
            ]["phone_route"]["mechanics"].__setitem__(
                "observed_history_sha256", marker("wrong-history")
            ),
            "history mismatch",
        )

    def test_shared_oracle_program_is_rejected(self):
        def mutate(candidate, evidence):
            mechanics = evidence["models"][candidate["models"][0]["model_id"]][
                "phone_route"
            ]["mechanics"]
            mechanics["oracle_program_sha256"] = mechanics["route_program_sha256"]

        self.mutation_fails(mutate, "oracle is not independent")

    def test_cross_geometry_cuda_oracle_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][0]["model_id"]
            ]["phone_route"]["mechanics"].__setitem__(
                "oracle_call_shapes_sha256", marker("wrong-call-shapes")
            ),
            "CUDA oracle is not path-matched",
        )

    def test_task_regression_is_rejected(self):
        def mutate(candidate, evidence):
            quality = evidence["models"][candidate["models"][1]["model_id"]][
                "phone_route"
            ]["quality"]
            quality["phone_correct"] = 40
            quality["phone_new_errors"] = 8

        self.mutation_fails(mutate, "too many new task errors")

    def test_inconsistent_paired_task_counts_are_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][0]["model_id"]
            ]["phone_route"]["quality"].__setitem__("phone_recovered_errors", 1),
            "paired task counts are inconsistent",
        )

    def test_cross_backend_and_geometry_disagreement_are_diagnostic(self):
        candidate = self.finalized_candidate()
        evidence = self.evidence(candidate)
        for record in evidence["models"].values():
            mechanics = record["phone_route"]["mechanics"]
            mechanics["cross_backend_greedy_matches"] = 0
            mechanics["cross_geometry_exact"] = False
        self.assertEqual(
            self.evaluate(candidate, evidence)["status"],
            "TWO_ROUTE_ELIGIBILITY_PASS",
        )

    def test_late_phone_publication_is_rejected(self):
        self.mutation_fails(
            lambda candidate, evidence: evidence["models"][
                candidate["models"][1]["model_id"]
            ]["bridge"].__setitem__("live_phone_publication_ns", 3_000_000_000),
            "not before CUDA readiness",
        )

    def test_pair_that_can_coexist_is_rejected(self):
        def mutate(_candidate, evidence):
            for item in evidence["pair_capacity"]["models"].values():
                item["model_buffer_bytes"] = 1
                item["kv_buffer_bytes"] = 1
            for record in evidence["models"].values():
                record["cuda_b8"]["model_buffer_bytes"] = 1
                record["cuda_b8"]["kv_buffer_bytes"] = 1
            evidence["pair_capacity"]["lower_bound_used_vram_bytes"] = (
                evidence["pair_capacity"]["idle_used_vram_bytes"]
                + evidence["pair_capacity"]["required_headroom_bytes"]
                + 4
            )

        self.mutation_fails(mutate, "models can coexist")

    def test_reprepare_over_dwell_is_rejected(self):
        self.mutation_fails(
            lambda _candidate, evidence: evidence["reprepare"][0].__setitem__(
                "ended_ns", 40_000_000_000
            ),
            "dwell bound exceeded",
        )

    def test_reprepare_usb_weight_transfer_is_rejected(self):
        self.mutation_fails(
            lambda _candidate, evidence: evidence["reprepare"][1].__setitem__(
                "usb_weight_bytes", 1
            ),
            "usb_weight_bytes",
        )

    def test_reprepare_wrong_model_identity_is_rejected(self):
        self.mutation_fails(
            lambda _candidate, evidence: evidence["reprepare"][0].__setitem__(
                "ready_model_sha256", marker("wrong-ready-model")
            ),
            "ready_model_sha256",
        )

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_cp0_r1_") as directory:
            path = Path(directory) / "bad.json"
            path.write_bytes(b'{"schema":"a","schema":"b"}\n')
            with self.assertRaisesRegex(ELIGIBILITY.EligibilityError, "duplicate"):
                ELIGIBILITY.load_canonical(path)


if __name__ == "__main__":
    unittest.main()
