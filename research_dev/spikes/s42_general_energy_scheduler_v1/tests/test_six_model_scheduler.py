#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "mixed_model_trace_v1/run_six_model_trace.py"
SPEC = importlib.util.spec_from_file_location("six_model_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class SixModelSchedulerTests(unittest.TestCase):
    def test_dynamic_sampler_exposes_a_bounded_gpu_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sampler = RUNNER.DynamicSampler(Path(directory))
            sample_t_ns = time.monotonic_ns()
            sampler.rows.append({
                "gpu": {
                    "memory_free_bytes": 1,
                    "sample_t_ns": sample_t_ns,
                },
                "t_ns": sample_t_ns,
            })
            snapshot = sampler.latest_gpu_snapshot(1_000_000_000)
            self.assertEqual(snapshot["sample_t_ns"], sample_t_ns)
            snapshot["memory_free_bytes"] = 2
            self.assertEqual(
                sampler.rows[0]["gpu"]["memory_free_bytes"], 1
            )

    def rows(self) -> list[dict[str, object]]:
        models = (
            RUNNER.QWEN14,
            RUNNER.QWEN8,
            RUNNER.GEMMA12,
            RUNNER.QWEN06,
            RUNNER.LLAMA1,
            RUNNER.GEMMA_E2B,
        )
        return [
            {
                "arrival_us": index * 100,
                "execution_model_id": model_id,
                "input_tokens": 100,
                "mixed_request_index": index,
                "output_tokens": 10,
                "slo_us": 30_000_000,
            }
            for index, model_id in enumerate(models)
        ]

    def control(
        self, rows: list[dict[str, object]]
    ) -> dict[str, object]:
        backend = {
            RUNNER.QWEN14: "cuda",
            RUNNER.QWEN8: "cuda",
            RUNNER.GEMMA12: "cuda",
            RUNNER.QWEN06: "cpu",
            RUNNER.LLAMA1: "cpu",
            RUNNER.GEMMA_E2B: "cpu",
        }
        slots = {
            RUNNER.QWEN14: 4,
            RUNNER.QWEN8: 4,
            RUNNER.GEMMA12: 8,
            RUNNER.QWEN06: 4,
            RUNNER.LLAMA1: 4,
            RUNNER.GEMMA_E2B: 2,
        }
        return {
            "request_results": [
                {
                    "execution_model_id": row["execution_model_id"],
                    "mixed_request_index": row["mixed_request_index"],
                    "predicted_ms": 2000.0,
                    "prompt_ms": 1000.0,
                    "route": (
                        f"{row['execution_model_id']}-"
                        f"{backend[row['execution_model_id']]}"
                    ),
                }
                for row in rows
            ],
            "server_records": [
                {
                    "backend": backend[model_id],
                    "load_ms": 100.0,
                    "model_id": model_id,
                    "props": {"total_slots": slots[model_id]},
                    "stage": (
                        "promotion"
                        if model_id in {RUNNER.QWEN8, RUNNER.GEMMA12}
                        else "initial"
                    ),
                    **(
                        {"warm_ms": 10.0}
                        if model_id in {RUNNER.QWEN8, RUNNER.GEMMA12}
                        else {}
                    ),
                }
                for model_id in (
                    RUNNER.QWEN14,
                    RUNNER.QWEN06,
                    RUNNER.LLAMA1,
                    RUNNER.GEMMA_E2B,
                    RUNNER.QWEN8,
                    RUNNER.GEMMA12,
                )
            ],
        }

    def plan(self) -> dict[str, object]:
        routes = {
            "0": f"{RUNNER.QWEN14}-cuda",
            "1": f"{RUNNER.QWEN8}-cuda",
            "2": RUNNER.PHONE_ROUTE,
            "3": f"{RUNNER.QWEN06}-cpu",
            "4": f"{RUNNER.LLAMA1}-cpu",
            "5": f"{RUNNER.GEMMA_E2B}-cpu",
        }
        plan: dict[str, object] = {
            "gpu": {
                "sequence": [RUNNER.QWEN14, RUNNER.QWEN8, RUNNER.GEMMA12]
            },
            "phone": {
                "desktop_companion_retire_when_idle_before_cuda": (
                    RUNNER.GEMMA12
                ),
                "model_id": RUNNER.GEMMA12,
                "preloaded_before_paid_start": True,
                "route_id": RUNNER.PHONE_ROUTE,
                "scheduler_effective_lanes": 2,
                "selected_mixed_request_indices": [2],
                "split_policy": {
                    "id": "i3-hidden-wait",
                    "io": "f16",
                    "layer_mask": "0x0000ffffffffffff",
                    "max_columns": 11136,
                    "timeout_ms": 35000,
                    "table": (
                        "1:9664,3:8192,8:4096,128:8192,512:11136"
                    ),
                },
            },
            "request_routes": routes,
            "schema": RUNNER.PLAN_SCHEMA,
            "status": "PASS",
            "trace_sha256": "trace",
        }
        plan["plan_sha256"] = hashlib.sha256(
            RUNNER.run_trace.canonical(plan)
        ).hexdigest()
        return plan

    def write(self, plan: dict[str, object]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "plan.json"
        path.write_bytes(RUNNER.run_trace.canonical(plan))
        return path

    def live_plan(self, mode: str = "adaptive") -> dict[str, object]:
        rows = self.rows()
        control = self.control(rows)
        source_path = (
            ROOT
            / "whole_task_phone_v1/results/4060ti_op15_20260807"
            / "SCHEDULER_PROFILE_TAIL_REUSED.json"
        )
        source = json.loads(source_path.read_text(encoding="ascii"))
        model_bytes = {
            row["execution_model_id"]: 1000 for row in rows
        }
        bindings = RUNNER.plan_live_vq_trace.executor_bindings(
            rows, control, model_bytes
        )
        profile, executors = RUNNER.plan_live_vq_trace.make_runtime_profile(
            rows,
            control,
            source,
            "0" * 64,
            bindings,
            mode,
        )
        residency_problem, residency_plan = (
            RUNNER.plan_live_vq_trace.plan_gpu_residency(
                rows, control, bindings, "sha256:" + "0" * 64
            )
        )
        routes = {str(index): RUNNER.UNIFIED_ROUTE for index in range(6)}
        scheduled_model_ids = list(
            RUNNER.plan_live_vq_trace.trace_model_ids(rows)
        )
        plan: dict[str, object] = {
            "gpu": {
                "residency_plan": residency_plan,
                "residency_problem": residency_problem,
                "sequence": residency_plan["model_order"],
            },
            "mode": mode,
            "phone": (
                {
                    "model_id": None,
                    "preloaded_before_paid_start": False,
                    "route_id": None,
                    "route_kind": "none",
                    "scheduler_effective_lanes": 0,
                    "predicted_selected_mixed_request_indices": [],
                }
                if mode == "control"
                else {
                    "model_id": RUNNER.LLAMA1,
                    "preloaded_before_paid_start": True,
                    "route_id": RUNNER.TASK_PHONE_ROUTE,
                    "route_kind": "whole-task-adreno",
                    "scheduler_effective_lanes": 1,
                    "predicted_selected_mixed_request_indices": [4],
                }
            ),
            "request_routes": routes,
            "scheduler": {
                "alternative_model_ids": [RUNNER.LLAMA1],
                "deferred_until_ready_model_ids": residency_plan[
                    "model_order"
                ][1:],
                "enabled": True,
                "policy_mode": mode,
                "profile": profile,
                "route_executors": executors,
                "scheduled_model_ids": scheduled_model_ids,
                "single_route_model_ids": [
                    model_id
                    for model_id in scheduled_model_ids
                    if model_id != RUNNER.LLAMA1
                ],
            },
            "schema": RUNNER.LIVE_PLAN_SCHEMA,
            "status": "PASS",
            "trace_sha256": "trace",
        }
        plan["plan_sha256"] = hashlib.sha256(
            RUNNER.run_trace.canonical(plan)
        ).hexdigest()
        return plan

    def test_accepts_bound_preloaded_phone_route(self) -> None:
        plan = self.plan()
        loaded = RUNNER.read_scheduler_plan(
            self.write(plan), self.rows(), "trace"
        )
        self.assertEqual(
            loaded["phone"]["selected_mixed_request_indices"], [2]
        )

    def test_rejects_route_change_after_hashing(self) -> None:
        plan = self.plan()
        plan["request_routes"]["2"] = f"{RUNNER.GEMMA12}-cuda"
        with self.assertRaisesRegex(
            RUNNER.run_trace.RunError, "scheduler plan identity"
        ):
            RUNNER.read_scheduler_plan(
                self.write(plan), self.rows(), "trace"
            )

    def test_accepts_live_virtual_queue_plan(self) -> None:
        plan = self.live_plan()
        loaded = RUNNER.read_scheduler_plan(
            self.write(plan), self.rows(), "trace"
        )
        self.assertTrue(loaded["scheduler"]["enabled"])
        self.assertEqual(
            loaded["phone"]["route_kind"], "whole-task-adreno"
        )

    def test_control_arm_also_uses_the_live_scheduler(self) -> None:
        plan = self.live_plan("control")
        loaded = RUNNER.read_scheduler_plan(
            self.write(plan), self.rows(), "trace"
        )
        self.assertTrue(loaded["scheduler"]["enabled"])
        self.assertEqual(loaded["scheduler"]["policy_mode"], "control")
        self.assertNotIn(
            RUNNER.TASK_PHONE_ROUTE,
            set(loaded["scheduler"]["route_executors"].values()),
        )

    def test_live_plan_executor_map_is_closed(self) -> None:
        plan = self.live_plan()
        plan["scheduler"]["route_executors"][next(iter(
            plan["scheduler"]["route_executors"]
        ))] = "unknown"
        plan.pop("plan_sha256")
        plan["plan_sha256"] = hashlib.sha256(
            RUNNER.run_trace.canonical(plan)
        ).hexdigest()
        with self.assertRaisesRegex(
            RUNNER.run_trace.RunError, "executor compatibility"
        ):
            RUNNER.read_scheduler_plan(
                self.write(plan), self.rows(), "trace"
            )

    def test_live_profile_schedules_every_request(self) -> None:
        plan = self.live_plan()
        decisions, summary = RUNNER.plan_live_vq_trace.simulate(
            self.rows(),
            plan["scheduler"]["profile"],
            plan["scheduler"]["route_executors"],
            plan["gpu"]["residency_plan"],
            "adaptive",
        )
        self.assertEqual(summary["decision_count"], 6)
        self.assertEqual(summary["energy_request_count"], 1)
        self.assertEqual(summary["energy_unknown_request_count"], 5)
        self.assertEqual(
            [row["mixed_request_index"] for row in decisions],
            list(range(6)),
        )

    def test_real_trace_has_full_scheduler_coverage(self) -> None:
        trace_dir = ROOT / "mixed_model_trace_v1"
        rows = [
            json.loads(line)
            for line in (
                trace_dir / "REQUESTS_MIXED_114.jsonl"
            ).read_text(encoding="ascii").splitlines()
        ]
        manifest = json.loads(
            (trace_dir / "TRACE_MANIFEST.json").read_text(encoding="ascii")
        )
        control = self.control(rows)
        bindings = RUNNER.plan_live_vq_trace.executor_bindings(
            rows,
            control,
            RUNNER.plan_live_vq_trace.manifest_model_bytes(manifest),
        )
        source = json.loads((
            ROOT
            / "whole_task_phone_v1/results/4060ti_op15_20260807"
            / "SCHEDULER_PROFILE_TAIL_REUSED.json"
        ).read_text(encoding="ascii"))
        profile, executors = RUNNER.plan_live_vq_trace.make_runtime_profile(
            rows, control, source, "0" * 64, bindings, "adaptive"
        )
        _, residency = RUNNER.plan_live_vq_trace.plan_gpu_residency(
            rows, control, bindings, "sha256:" + "0" * 64
        )
        decisions, summary = RUNNER.plan_live_vq_trace.simulate(
            rows, profile, executors, residency, "adaptive"
        )
        self.assertEqual(len(profile["routes"]), 124)
        self.assertEqual(summary["decision_count"], 114)
        self.assertEqual(summary["energy_request_count"], 10)
        self.assertEqual(summary["energy_unknown_request_count"], 104)
        self.assertEqual(
            {row["mixed_request_index"] for row in decisions},
            set(range(114)),
        )

    def test_build_plan_uses_scheduler_for_real_trace(self) -> None:
        trace_dir = ROOT / "mixed_model_trace_v1"
        requests_path = trace_dir / "REQUESTS_MIXED_114.jsonl"
        manifest_path = trace_dir / "TRACE_MANIFEST.json"
        rows = [
            json.loads(line)
            for line in requests_path.read_text(
                encoding="ascii"
            ).splitlines()
        ]
        paid_start_ns = 1_000_000_000
        control = self.control(rows)
        for row, record in zip(rows, control["request_results"], strict=True):
            record["completion_ns"] = (
                paid_start_ns + (row["arrival_us"] + 3_000_000) * 1000
            )
            record["slo_met"] = True
        control.update({
            "paid_start_ns": paid_start_ns,
            "schema": RUNNER.RESULT_SCHEMA,
            "status": "PASS",
            "trace_sha256": RUNNER.digest_file(requests_path),
        })
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        control_path = Path(directory.name) / "control.json"
        control_path.write_bytes(RUNNER.run_trace.canonical(control))
        plan = RUNNER.plan_live_vq_trace.build_plan(
            requests_path,
            manifest_path,
            control_path,
            (
                ROOT
                / "whole_task_phone_v1/results/4060ti_op15_20260807"
                / "SCHEDULER_PROFILE_TAIL_REUSED.json"
            ),
            "adaptive",
        )
        self.assertEqual(plan["schema"], RUNNER.LIVE_PLAN_SCHEMA)
        self.assertEqual(plan["prediction"]["decision_count"], 114)
        self.assertEqual(
            set(plan["request_routes"].values()),
            {RUNNER.UNIFIED_ROUTE},
        )

    def test_live_plan_recomputes_residency_decision(self) -> None:
        plan = self.live_plan()
        plan["gpu"]["residency_plan"]["phases"][1]["ready_us"] += 1
        plan.pop("plan_sha256")
        plan["plan_sha256"] = hashlib.sha256(
            RUNNER.run_trace.canonical(plan)
        ).hexdigest()
        with self.assertRaisesRegex(
            RUNNER.run_trace.RunError, "live scheduler configuration"
        ):
            RUNNER.read_scheduler_plan(
                self.write(plan), self.rows(), "trace"
            )

    def test_completed_summary_ignores_ready_line(self) -> None:
        value = RUNNER.completed_prefixed_json(
            [
                "S41SERVERFFN ready host=127.0.0.1",
                'S41SERVERFFN {"calls":48,"status":"ok"}',
            ],
            "S41SERVERFFN ",
        )
        self.assertEqual(value, {"calls": 48, "status": "ok"})


if __name__ == "__main__":
    unittest.main()
