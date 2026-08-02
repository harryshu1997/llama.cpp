#!/usr/bin/env python3

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from build_runtime_config import build_runtime_config, main  # noqa: E402
from evidence_common import (  # noqa: E402
    EvidenceError,
    canonical_bytes,
    digest_file,
)


class RuntimeConfigTests(unittest.TestCase):
    def profile_lock(self, root: Path) -> Path:
        placements = {}
        for model_id in ("qwen3-14b-q4_k_m", "qwen3-8b-q8_0"):
            placement = root / f"{model_id}.placement.json"
            placement.write_bytes(canonical_bytes({
                "model_id": model_id,
                "schema": "fixture-placement-v1",
            }))
            placements[model_id] = digest_file(placement)
        path = root / "c3_profile_lock.json"
        path.write_bytes(canonical_bytes({
            "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "measured_gpu_headroom_bytes": 536870912,
            "models": {
                "qwen3-14b-q4_k_m": {
                    "b1_service_pass": True,
                    "b8_service_pass": True,
                    "n_gpu_layers": 20,
                    "placement_evidence_sha256":
                        placements["qwen3-14b-q4_k_m"],
                },
                "qwen3-8b-q8_0": {
                    "b1_service_pass": True,
                    "b8_service_pass": True,
                    "n_gpu_layers": 18,
                    "placement_evidence_sha256":
                        placements["qwen3-8b-q8_0"],
                },
            },
            "schema": "s40-c3-profile-lock-v1",
            "system_swap_growth_bytes": 0,
        }))
        return path

    def plan(self, mode: str, root: Path) -> dict:
        kinds = {
            "C1_GPU_ONLY_OPTIMIZED": ["GPU_PRIMARY"],
            "C2_GPU_PLUS_CPU_WARM_EXECUTOR": [
                "GPU_PRIMARY", "CPU_WARM"],
            "C3_DUAL_PARTIAL_OFFLOAD": ["GPU_CPU_PARTIAL"],
            "T1_PHONE_WARM_TIER": ["GPU_PRIMARY", "PHONE_WARM"],
            "T2_PHONE_NO_PROMOTION": ["GPU_PRIMARY", "PHONE_WARM"],
        }[mode]
        profile = (
            self.profile_lock(root)
            if mode == "C3_DUAL_PARTIAL_OFFLOAD" else None)
        executors = [{
            "credits": 8,
            "execute_concurrency": 8,
            "executor_id": kind.lower(),
            "executor_instance_id": f"fixture-{index}-{kind.lower()}",
            "expected_peer_pid": 1000 + index,
            "expected_peer_start_time_ticks": 2000 + index,
            "kind": kind,
            "output_limit_bytes": 65536,
            "queue_capacity": 74,
            "socket_path": str(root / f"{kind.lower()}.sock"),
            "timeout_ms": 300000,
            "transport": "UNIX_SOCKET",
        } for index, kind in enumerate(kinds)]
        executor_configs = []
        for executor in executors:
            config = root / (
                f"{mode}-{executor['executor_id']}-gateway.json")
            config.write_bytes(canonical_bytes({
                "executor_id": executor["executor_id"],
                "schema": "fixture-gateway-config-v1",
            }))
            executor_configs.append({
                "executor_id": executor["executor_id"],
                "gateway_config_path": str(config),
                "gateway_config_sha256": digest_file(config),
            })
        placements = {}
        if profile is not None:
            for model_id in ("qwen3-14b-q4_k_m", "qwen3-8b-q8_0"):
                placement = root / f"{model_id}.placement.json"
                placements[model_id] = {
                    "path": str(placement),
                    "sha256": digest_file(placement),
                }
        evidence_root = root / f"{mode}-evidence-root.json"
        evidence_root.write_bytes(canonical_bytes({
            "c3_placement_artifacts": placements,
            "configuration": mode,
            "executor_configs": executor_configs,
            "experiment_contract_sha256": digest_file(
                S40 / "EXPERIMENT_CONTRACT.json"),
            "schema": "s40-runtime-evidence-root-v1",
        }))
        return {
            "c3_profile_lock_path": (
                str(profile) if profile is not None else None),
            "c3_profile_lock_sha256": (
                digest_file(profile) if profile is not None else None),
            "evidence_root_path": str(evidence_root),
            "evidence_root_sha256": digest_file(evidence_root),
            "event_log_path": "/tmp/s40-events.jsonl",
            "executors": executors,
            "hot_model_id": "qwen3-8b-q8_0",
            "mode": mode,
            "run_id": "fixture-run",
            "schema": "s40-runtime-config-plan-v3",
        }

    def test_c1_has_no_fabricated_warm_executor(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C1_GPU_ONLY_OPTIMIZED", root)
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            plan_sha256 = digest_file(path)
            result = build_runtime_config(path)
        runtime = result["runtime"]
        self.assertEqual(
            runtime["schema"], "llama-server-warm-tier-runtime-v4")
        self.assertEqual(
            runtime["runtime_plan_sha256"], plan_sha256)
        self.assertTrue(runtime["promotion_enabled"])
        self.assertEqual(len(runtime["executors"]), 1)
        self.assertEqual(
            [row["state"] for row in runtime["initial_models"]],
            ["READY", "ABSENT"],
        )

    def test_c2_rotates_gpu_and_cpu_residency(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(
                self.plan("C2_GPU_PLUS_CPU_WARM_EXECUTOR", root)))
            result = build_runtime_config(path)
        runtime = result["runtime"]
        self.assertEqual(
            [row["role"] for row in runtime["executors"]],
            ["GPU", "CPU"],
        )
        self.assertEqual(
            [row["state"] for row in runtime["initial_models"]],
            ["READY", "ABSENT", "ABSENT", "READY"],
        )

    def test_c3_summary_only_profile_is_blocked(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(
                self.plan("C3_DUAL_PARTIAL_OFFLOAD", root)))
            with self.assertRaisesRegex(
                    EvidenceError, "summary-only v1"):
                build_runtime_config(path)

    def test_t1_promotes_but_t2_does_not(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            t1_path = root / "t1.json"
            t2_path = root / "t2.json"
            t1_path.write_bytes(canonical_bytes(
                self.plan("T1_PHONE_WARM_TIER", root)))
            t2_path.write_bytes(canonical_bytes(
                self.plan("T2_PHONE_NO_PROMOTION", root)))
            t1 = build_runtime_config(t1_path)["runtime"]
            t2 = build_runtime_config(t2_path)["runtime"]
        self.assertTrue(t1["promotion_enabled"])
        self.assertFalse(t2["promotion_enabled"])
        self.assertEqual(
            [row["role"] for row in t1["executors"]],
            ["GPU", "PHONE"],
        )

    def test_executor_topology_is_exact(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C1_GPU_ONLY_OPTIMIZED", root)
            plan["executors"].append(copy.deepcopy(plan["executors"][0]))
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "topology mismatch"):
                build_runtime_config(path)

    def test_c3_profile_lock_is_required(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C3_DUAL_PARTIAL_OFFLOAD", root)
            plan["c3_profile_lock_sha256"] = None
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(
                    EvidenceError, "c3_profile_lock_sha256|digest"):
                build_runtime_config(path)

    def test_non_c3_profile_lock_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C1_GPU_ONLY_OPTIMIZED", root)
            plan["c3_profile_lock_sha256"] = "1" * 64
            plan["c3_profile_lock_path"] = "/tmp/foreign.json"
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "unexpected C3"):
                build_runtime_config(path)

    def test_credits_cannot_exceed_concurrency(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("T1_PHONE_WARM_TIER", root)
            plan["executors"][1]["credits"] = 9
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "credits exceed"):
                build_runtime_config(path)

    def test_runtime_identity_is_required_and_unique(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C2_GPU_PLUS_CPU_WARM_EXECUTOR", root)
            plan["executors"][1]["executor_instance_id"] = (
                plan["executors"][0]["executor_instance_id"])
            path = root / "duplicate-instance.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "duplicate executor"):
                build_runtime_config(path)
            plan = self.plan("C1_GPU_ONLY_OPTIMIZED", root)
            plan["executors"][0]["expected_peer_start_time_ticks"] = 0
            path = root / "zero-start.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "expected >= 1"):
                build_runtime_config(path)

    def test_queue_and_promotion_credit_bounds_are_enforced(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C1_GPU_ONLY_OPTIMIZED", root)
            plan["executors"][0]["queue_capacity"] = 7
            path = root / "short-queue.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "below credits"):
                build_runtime_config(path)
            plan = self.plan("C2_GPU_PLUS_CPU_WARM_EXECUTOR", root)
            plan["executors"][0]["credits"] = 7
            path = root / "short-gpu-credit.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "GPU credits"):
                build_runtime_config(path)

    def test_process_transport_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C1_GPU_ONLY_OPTIMIZED", root)
            plan["executors"][0]["transport"] = "PROCESS"
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "UNIX_SOCKET"):
                build_runtime_config(path)

    def test_relative_or_oversize_socket_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C1_GPU_ONLY_OPTIMIZED", root)
            plan["executors"][0]["socket_path"] = "relative.sock"
            path = root / "relative.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "absolute"):
                build_runtime_config(path)
            plan["executors"][0]["socket_path"] = "/" + "s" * 103
            path = root / "oversize.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "invalid socket"):
                build_runtime_config(path)

    def test_bridge_argv_is_rejected_by_exact_schema(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C1_GPU_ONLY_OPTIMIZED", root)
            plan["executors"][0]["bridge_argv"] = ["/bin/false"]
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "key set mismatch"):
                build_runtime_config(path)

    def test_c3_profile_lock_hash_is_load_bearing(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C3_DUAL_PARTIAL_OFFLOAD", root)
            plan["c3_profile_lock_sha256"] = "f" * 64
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "SHA-256 mismatch"):
                build_runtime_config(path)

    def test_c3_summary_headroom_cannot_bypass_raw_evidence(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C3_DUAL_PARTIAL_OFFLOAD", root)
            lock_path = Path(plan["c3_profile_lock_path"])
            lock = __import__("json").loads(lock_path.read_text())
            lock["measured_gpu_headroom_bytes"] = 536870911
            lock_path.write_bytes(canonical_bytes(lock))
            plan["c3_profile_lock_sha256"] = digest_file(lock_path)
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "summary-only v1"):
                build_runtime_config(path)

    def test_evidence_root_hash_is_load_bearing(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C2_GPU_PLUS_CPU_WARM_EXECUTOR", root)
            plan["evidence_root_sha256"] = "f" * 64
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "root SHA-256"):
                build_runtime_config(path)

    def test_c3_opaque_placement_cannot_bypass_raw_evidence(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan = self.plan("C3_DUAL_PARTIAL_OFFLOAD", root)
            evidence_path = Path(plan["evidence_root_path"])
            evidence = __import__("json").loads(
                evidence_path.read_text(encoding="ascii"))
            evidence["c3_placement_artifacts"]["qwen3-8b-q8_0"][
                "sha256"] = "f" * 64
            evidence_path.write_bytes(canonical_bytes(evidence))
            plan["evidence_root_sha256"] = digest_file(evidence_path)
            path = root / "plan.json"
            path.write_bytes(canonical_bytes(plan))
            with self.assertRaisesRegex(EvidenceError, "summary-only v1"):
                build_runtime_config(path)

    def test_cli_refuses_preexisting_runtime_config(self):
        with tempfile.TemporaryDirectory(prefix="s40_config_") as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_bytes(canonical_bytes(
                self.plan("C1_GPU_ONLY_OPTIMIZED", root)))
            output = root / "runtime.json"
            output.write_bytes(b"sentinel\n")
            with patch.object(sys, "argv", [
                    "build_runtime_config.py",
                    "--plan", str(plan_path),
                    "--output", str(output),
            ]):
                self.assertEqual(main(), 2)
            self.assertEqual(output.read_bytes(), b"sentinel\n")


if __name__ == "__main__":
    unittest.main()
