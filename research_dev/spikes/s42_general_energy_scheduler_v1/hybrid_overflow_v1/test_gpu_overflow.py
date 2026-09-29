#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
TRACE = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1"
    / "REQUESTS_SEMANTIC_SOURCE.jsonl"
)

from research_dev.scheduler import (  # noqa: E402
    CapacityError,
    load_execution_plan,
    write_execution_plan,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = load_module("plan_gpu_overflow", HERE / "plan_gpu_overflow.py")
analyzer = load_module(
    "analyze_gpu_overflow_pair", HERE / "analyze_gpu_overflow_pair.py"
)


def plan(mode: str, split_policy_variant: str = "qualified"):
    roles = ["cold_model", "cold_server"]
    if mode == "shadow":
        roles += [
            "bridge",
            "phone_model",
            "phone_session",
            "phone_worker",
            "restore_usb",
        ]
    paths = {role: f"/tmp/{role}" for role in roles}
    hashes = {role: "a" * 64 for role in roles}
    hashes["cold_model"] = planner.MODEL_SHA256
    if "phone_worker" in hashes:
        hashes["phone_worker"] = "b" * 64
    return planner.build_plan(
        trace_path=TRACE,
        mode=mode,
        artifact_paths=paths,
        artifact_hashes=hashes,
        lib_dir="/tmp/lib",
        adb_port=5037,
        phone_serial="phone",
        split_policy_variant=split_policy_variant,
    )


class OverflowPlannerTests(unittest.TestCase):
    def test_bridge_allocator_capability_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            legacy = root / "legacy-bridge"
            legacy.write_text(
                "#!/bin/sh\n"
                "echo 'usage: bridge <malloc|devmem>' >&2\n"
                "exit 2\n",
                encoding="ascii",
            )
            legacy.chmod(0o755)
            with self.assertRaisesRegex(
                planner.PlanError, "bridge allocator capability: malloc-split"
            ):
                planner.require_bridge_allocator(legacy, "malloc-split")

            split = root / "split-bridge"
            split.write_text(
                "#!/bin/sh\n"
                "echo 'usage: bridge <malloc|devmem|malloc-split|devmem-split>' >&2\n"
                "exit 2\n",
                encoding="ascii",
            )
            split.chmod(0o755)
            planner.require_bridge_allocator(split, "malloc-split")

    def test_phone_split_capabilities_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            planner.PlanError, "phone worker split protocol capabilities"
        ):
            planner.require_phone_worker_capabilities(
                "--f16-io --max-requests"
            )
        planner.require_phone_worker_capabilities(
            "--alternate-columns --column-quantum --max-tokens "
            "--staged-dmabuf"
        )
        with self.assertRaisesRegex(
            planner.PlanError, "phone session split protocol capabilities"
        ):
            planner.require_phone_session_capabilities("S41_FFN_F16_IO")
        planner.require_phone_session_capabilities(
            "S41_FFN_ALTERNATE_COLUMNS S41_FFN_COLUMN_QUANTUM "
            "S41_FFN_MAX_TOKENS S41_FFN_STAGED_DMABUF"
        )

    def test_control_and_shadow_share_cpu_gpu_placement(self) -> None:
        control = plan("control")
        treatment = plan("shadow")
        self.assertEqual(control.layer_placement.selected, treatment.layer_placement.selected)
        self.assertEqual(control.layer_placement.cpu_layer_spec, "0-22")
        self.assertEqual(control.layer_placement.gpu_layer_spec, "23-47")
        self.assertIn(
            ("gemma-f16-cuda-full-v1", "GPU_CAPACITY"),
            control.layer_placement.rejected,
        )
        self.assertEqual(control.decision.reason, "CONTROL_BASELINE")
        self.assertEqual(
            treatment.decision.reason, "SHADOW_FASTEST_MEASURED_COHORT"
        )
        self.assertEqual(tuple(treatment.offload.split.layer_ids), tuple(range(23)))

    def test_display_occupancy_can_make_route_fail_closed(self) -> None:
        with self.assertRaisesRegex(CapacityError, "no verified"):
            planner.build_plan(
                trace_path=TRACE,
                mode="control",
                artifact_paths={
                    "cold_model": "/tmp/model",
                    "cold_server": "/tmp/server",
                },
                artifact_hashes={
                    "cold_model": planner.MODEL_SHA256,
                    "cold_server": "a" * 64,
                },
                lib_dir="/tmp/lib",
                adb_port=5037,
                phone_serial="phone",
                gpu_occupied_bytes=3_000_000_000,
            )

    def test_plan_round_trip(self) -> None:
        expected = plan("shadow")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "plan.json"
            write_execution_plan(path, expected)
            self.assertEqual(load_execution_plan(path), expected)

    def test_shape_balanced_plan_is_scheduler_materialized(self) -> None:
        expected = plan("shadow", "shape-balanced")
        self.assertEqual(
            expected.offload.split.table,
            "4:6144,6:5632,8:5120,10:6144,11:5120,"
            "13:6144,14:5120,15:6144,16:5120,512:0",
        )
        self.assertEqual(
            expected.admission_phase,
            "shadow_shape_balance_energy_calibration",
        )


def result_value(plan_value, arm: str, server_j: float) -> dict:
    treatment = arm == "treatment"
    rows = [
        {
            "input_tokens": 11_460 if index == 0 else 1,
            "output_tokens": 6_903 if index == 0 else 1,
        }
        for index in range(17)
    ]
    phone = {"bridge": None, "ffn": None, "shapes": []}
    if treatment:
        calls = 19_550
        transfer_bytes = 1_201_152_000
        phone = {
            "bridge": {
                "allocator": "malloc-split",
                "calls": calls,
                "download_bytes": transfer_bytes,
                "max_wire_bytes": 3_932_288,
                "reset_recoveries": 0,
                "status": "ok",
                "upload_bytes": transfer_bytes,
            },
            "ffn": {
                "calls": calls,
                "download_bytes": transfer_bytes,
                "status": "ok",
                "upload_bytes": transfer_bytes,
            },
            "shapes": [
                {"calls": calls, "columns": 6144, "tokens": 8}
            ],
        }
    return {
        "metrics": {
            "completed": 17,
            "makespan_s": 100.0,
            "output_tokens": 6_919,
        },
        "mode": "cuda-cpu-op15" if treatment else "cuda-cpu",
        "model": {"sha256": planner.MODEL_SHA256},
        "n_gpu_layers": 25,
        "paid_end_ns": 101_000_000_000,
        "paid_start_ns": 1_000_000_000,
        "phone": phone,
        "preflight": {
            "scheduler": {"plan_sha256": plan_value.plan_sha256}
        },
        "repeat_index": 1,
        "request_results": rows,
        "resources": {
            "gpu_memory_total_bytes": 17_175_674_880,
            "process_swap_max_bytes": 0,
        },
        "scheduler_plan_sha256": plan_value.plan_sha256,
        "schema": "s41-gemma-gpu-trace-v2",
        "server_energy": {
            "boundary": "paid_trace_interval",
            "cpu_package_energy_j": server_j * 0.7,
            "gpu_board_energy_j": server_j * 0.3,
            "server_compute_device_energy_j": server_j,
        },
        "status": "PASS",
    }


def phone_value(energy_j: float) -> dict:
    return {
        "boundary": "paid_trace_interval",
        "duration_s": 100.0,
        "schema": "s41-phone-energy-v3",
        "status": "PASS",
        "whole_phone_energy_j": energy_j,
    }


class OverflowAnalyzerTests(unittest.TestCase):
    def test_dynamic_phone_shapes_conserve_work(self) -> None:
        treatment_plan = plan("shadow")
        shapes = [
            {"tokens": 1, "columns": 6144, "calls": 2047},
            {"tokens": 2, "columns": 6144, "calls": 1702},
            {"tokens": 3, "columns": 6144, "calls": 3197},
            {"tokens": 4, "columns": 6144, "calls": 391},
            {"tokens": 6, "columns": 6144, "calls": 1242},
            {"tokens": 7, "columns": 6144, "calls": 5520},
            {"tokens": 8, "columns": 6144, "calls": 11638},
            {"tokens": 11, "columns": 6144, "calls": 207},
            {"tokens": 16, "columns": 6144, "calls": 23},
        ]
        summary = {
            "bridge": {
                "allocator": "malloc-split",
                "calls": 25_967,
                "download_bytes": 1_216_872_960,
                "max_wire_bytes": 3_932_288,
                "reset_recoveries": 0,
                "status": "ok",
                "upload_bytes": 1_216_872_960,
            },
            "ffn": {
                "calls": 25_967,
                "download_bytes": 1_216_872_960,
                "status": "ok",
                "upload_bytes": 1_216_872_960,
            },
            "shapes": shapes,
        }
        analyzer.validate_phone_work(treatment_plan, summary)
        broken = json.loads(json.dumps(summary))
        broken["bridge"]["upload_bytes"] += 1
        with self.assertRaisesRegex(
            analyzer.AnalysisError, "treatment phone work"
        ):
            analyzer.validate_phone_work(treatment_plan, broken)

    def test_fleet_saving_includes_control_idle_phone(self) -> None:
        control_plan = plan("control")
        treatment_plan = plan("shadow")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = {
                "control_result": root / "control-result.json",
                "control_phone": root / "control-phone.json",
                "control_plan": root / "control-plan.json",
                "treatment_result": root / "treatment-result.json",
                "treatment_phone": root / "treatment-phone.json",
                "treatment_plan": root / "treatment-plan.json",
            }
            write_execution_plan(paths["control_plan"], control_plan)
            write_execution_plan(paths["treatment_plan"], treatment_plan)
            paths["control_result"].write_text(
                json.dumps(result_value(control_plan, "control", 1000.0)),
                encoding="ascii",
            )
            paths["treatment_result"].write_text(
                json.dumps(result_value(treatment_plan, "treatment", 800.0)),
                encoding="ascii",
            )
            paths["control_phone"].write_text(
                json.dumps(phone_value(50.0)), encoding="ascii"
            )
            paths["treatment_phone"].write_text(
                json.dumps(phone_value(70.0)), encoding="ascii"
            )
            value = analyzer.analyze(**paths)
            self.assertEqual(value["energy_verdict"], "SAVING")
            self.assertAlmostEqual(
                value["comparison"]["fleet_energy_change_pct"],
                (870.0 / 1050.0 - 1.0) * 100.0,
            )


if __name__ == "__main__":
    unittest.main()
