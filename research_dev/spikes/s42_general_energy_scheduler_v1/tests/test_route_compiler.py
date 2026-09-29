#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "runtime_routes_v1"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from route_compiler import (  # noqa: E402
    RouteCompileError,
    compile_bundle,
    object_sha256,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def fixture() -> dict[str, object]:
    epoch = {
        "schema": "s42-runtime-epoch-v1",
        "epoch_label": "test-epoch",
        "bindings": {
            "artifacts": {"worker": {"sha256": SHA_D}},
            "concurrency": {
                "cold_parallel_slots": 8,
                "cold_ubatch_size": 512,
                "hot_model_active": True,
                "hot_parallel_slots": 4,
                "request_count": 6,
            },
            "hardware": {
                "desktop_cpu": "test-cpu",
                "desktop_gpu_name": "test-gpu",
                "desktop_gpu_uuid": "gpu-0",
                "phone_model": "test-phone",
                "phone_serial": "phone-0",
            },
            "models": {
                "cold": {
                    "architecture": "gemma4",
                    "file_bytes": 110,
                    "file_sha256": SHA_A,
                    "n_embd": 3840,
                    "n_layer": 48,
                    "n_vocab": 262144,
                    "quantization": "Q4_0",
                    "tensor_bytes": 100,
                },
                "hot": {
                    "architecture": "qwen3",
                    "file_bytes": 210,
                    "file_sha256": SHA_B,
                    "n_embd": 5120,
                    "n_layer": 40,
                    "n_vocab": 151936,
                    "quantization": "Q4_K_M",
                    "tensor_bytes": 200,
                },
            },
            "policy": {
                "column_quantum": 2048,
                "id": "i3-hidden-wait",
                "io": "f16",
                "max_columns": 11136,
                "table": "1:9664,3:8192,8:4096,128:8192,512:11136",
                "weight_layout": "view_safe_dense_ffn",
            },
            "residency": {
                "cold": "cpu-and-phone",
                "hot": "cuda",
            },
            "runtimes": {
                "cold": {
                    "manifest": [
                        {"name": "cold-server", "sha256": SHA_A, "size_bytes": 10}
                    ]
                },
                "hot": {
                    "manifest": [
                        {"name": "hot-server", "sha256": SHA_B, "size_bytes": 20}
                    ]
                },
            },
            "transport": {
                "allocator": "devmem",
                "desktop_endpoint": "libusb-bulk",
                "io_type": "f16",
                "phone_endpoint": "functionfs-dmabuf",
                "protocol": "test-v1",
                "reset_recovery": "qualified",
            },
            "workload": {
                "cold_requests": 2,
                "hot_requests": 4,
                "input_tokens": 100,
                "output_tokens": 20,
                "requests": 6,
                "sha256": SHA_C,
            },
        },
    }
    graph = {
        "schema": "s42-llama-placement-graph-v1",
        "status": "PASS",
        "manifest_sha256": SHA_D,
        "model": {
            "architecture": "gemma4",
            "n_embd": 3840,
            "n_layer": 48,
            "n_vocab": 262144,
            "tensor_bytes": 100,
        },
        "qualification": {
            "energy_bound": False,
            "graph_observed": True,
            "placement_bound": False,
            "runtime_route": False,
        },
        "operators": [
            {
                "family": "dense_ffn",
                "shape": {"k": 3840, "m": 8, "n": 15360},
                "split_options": [
                    {
                        "maximum": 15328,
                        "minimum": 32,
                        "quantum": 32,
                    }
                ],
            },
            {
                "family": "attn_q",
                "shape": {"k": 3840, "m": 8, "n": 4096},
                "split_options": [],
            },
        ],
    }
    profile = {
        "schema": "s42-kernel-energy-profile-v1",
        "profile_id": "test-profile",
        "devices": {
            "desktop_cpu": "test-cpu",
            "desktop_gpu": "gpu-0",
            "phone": "phone-0",
        },
        "qualification": {
            "composition_status": "pending_held_out_full_model_validation",
            "enforcement": "fail_closed",
        },
        "kernel_rows": [
            {
                "backend": "cpu",
                "kernel_family": "gemma4-dense-ffn-swiglu",
                "profile_id": "cpu-m8",
                "resident_type": "q4_0",
                "shape": {"k": 3840, "m": 8, "n": 15360},
                "status": "measured_shape_bucket",
            },
            {
                "backend": "htp",
                "kernel_family": "gemma4-dense-ffn-swiglu",
                "profile_id": "htp-m8",
                "resident_type": "q4_0",
                "shape": {"k": 3840, "m": 8, "n": 9664},
                "status": "measured_shape_bucket",
            },
        ],
        "one_time_cost_rows": [],
    }

    def repeat(index: int, duration: float, energy: float, digest: str) -> dict[str, object]:
        return {
            "duration_s": duration,
            "fleet_j": energy,
            "phone_energy_sha256": digest,
            "repeat_index": index,
            "result_sha256": digest,
        }

    control_id = "test-control"
    certificate = {
        "schema": "s42-route-certificates-v1",
        "certificate_set_id": "test-certificates",
        "required_epoch_binding_sha256": object_sha256(epoch["bindings"]),
        "profile": {"file_sha256": SHA_A, "profile_id": "test-profile"},
        "graphs": {
            "gemma": {
                "file_sha256": SHA_B,
                "manifest_sha256": SHA_D,
                "model_binding": "cold",
            }
        },
        "campaign": {
            "aggregate_file_sha256": SHA_A,
            "aggregate_record_sha256": SHA_B,
            "gates": {"all_pairs": True},
            "phone_work": {
                "mean_exposed_join_wait_fraction": 0.02,
                "reset_recoveries": 0,
            },
            "repetitions": 3,
            "trace_sha256": SHA_C,
            "verdict": "PASS",
        },
        "routes": [
            {
                "dispatch_contract": {
                    "cold_model": "cpu",
                    "hot_model": "cuda",
                    "phone_dense_ffn_split": False,
                    "route_mode": "cpu-control",
                },
                "fallback_route_id": None,
                "quality": {"class": "exact"},
                "repeats": [
                    repeat(1, 10.0, 100.0, SHA_A),
                    repeat(2, 11.0, 110.0, SHA_B),
                    repeat(3, 12.0, 120.0, SHA_C),
                ],
                "role": "control",
                "route_id": control_id,
                "scope": "task",
            },
            {
                "dispatch_contract": {
                    "cold_model": "cpu-plus-op15-htp",
                    "hot_model": "cuda",
                    "operator_family": "gemma4-dense-ffn-swiglu",
                    "phone_dense_ffn_split": True,
                    "policy_id": "i3-hidden-wait",
                    "route_mode": "cpu-htp-operator-split",
                    "split_table": "1:9664,3:8192,8:4096,128:8192,512:11136",
                    "transport": "functionfs-dmabuf-f16",
                },
                "fallback_route_id": control_id,
                "quality": {"class": "approximate"},
                "repeats": [
                    repeat(1, 8.0, 80.0, SHA_A),
                    repeat(2, 9.0, 90.0, SHA_B),
                    repeat(3, 10.0, 95.0, SHA_C),
                ],
                "role": "treatment",
                "route_id": "test-treatment",
                "scope": "operator",
            },
        ],
    }
    return {
        "certificate": certificate,
        "epoch": epoch,
        "graph": graph,
        "profile": profile,
    }


def compile_fixture(value: dict[str, object]) -> dict[str, object]:
    return dict(
        compile_bundle(
            profile=value["profile"],
            profile_sha256=SHA_A,
            certificate=value["certificate"],
            certificate_sha256=SHA_C,
            epoch=value["epoch"],
            epoch_sha256=SHA_D,
            graphs={"gemma": (value["graph"], SHA_B)},
        )
    )


class RouteCompilerTests(unittest.TestCase):
    def test_compiles_only_direct_cohort_routes(self) -> None:
        output = compile_fixture(fixture())
        self.assertEqual(output["status"], "PASS")
        self.assertEqual(len(output["compiled_routes"]), 2)
        self.assertFalse(output["runtime_activation_ready"])
        self.assertEqual(
            output["general_operator_enforcement"]["certified_route_count"], 0
        )
        matches = output["operator_profile_matches"][0]
        self.assertEqual(matches["isolated_kernel_matches_by_backend"]["cpu"], 1)
        self.assertEqual(matches["isolated_kernel_matches_by_backend"]["htp"], 1)
        self.assertEqual(matches["certified_general_operator_routes"], 0)

    def test_epoch_mutation_fails_closed(self) -> None:
        value = fixture()
        value["epoch"]["bindings"]["hardware"]["desktop_gpu_uuid"] = "other-gpu"
        with self.assertRaisesRegex(RouteCompileError, "epoch binding SHA-256"):
            compile_fixture(value)

    def test_graph_file_mutation_fails_closed(self) -> None:
        value = fixture()
        value["certificate"]["graphs"]["gemma"]["file_sha256"] = SHA_C
        with self.assertRaisesRegex(RouteCompileError, "graph gemma file SHA-256"):
            compile_fixture(value)

    def test_prior_graph_qualification_claim_is_rejected(self) -> None:
        value = fixture()
        value["graph"]["qualification"]["runtime_route"] = True
        with self.assertRaisesRegex(RouteCompileError, "prior runtime_route"):
            compile_fixture(value)

    def test_unqualified_profile_composition_is_required(self) -> None:
        value = fixture()
        value["profile"]["qualification"]["enforcement"] = "enabled"
        with self.assertRaisesRegex(RouteCompileError, "fail closed"):
            compile_fixture(value)

    def test_treatment_must_beat_every_control_repeat(self) -> None:
        value = fixture()
        value["certificate"]["routes"][1]["repeats"][1]["duration_s"] = 11.5
        with self.assertRaisesRegex(RouteCompileError, "every treatment repeat"):
            compile_fixture(value)

    def test_dispatch_policy_must_match_epoch(self) -> None:
        value = fixture()
        value["certificate"]["routes"][1]["dispatch_contract"]["split_table"] = "1:1"
        with self.assertRaisesRegex(RouteCompileError, "split table differs"):
            compile_fixture(value)


if __name__ == "__main__":
    unittest.main()
