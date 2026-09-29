#!/usr/bin/env python3

from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
OVERLAY = ROOT / "full_fp16_burstgpt_v1/small_model_overlay_v1"
CAMPAIGN = REPO_ROOT / "research_dev/scheduler/campaigns/burstgpt"
HARDWARE_PROFILE = (
    REPO_ROOT
    / "research_dev/scheduler/profiles/"
      "MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json"
)
for path in (REPO_ROOT, ROOT, OVERLAY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from full_fp16_burstgpt_v1.plan_full_fp16_burstgpt import (  # noqa: E402
    runtime_candidates,
)
from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    PhoneResidencyPlan,
    ProfileBundle,
    Request,
    RuntimeExecutorBinding,
    RuntimeExecutorObservation,
    RuntimeExecutionFailure,
    RuntimeCapabilityCatalog,
    RuntimeModelArtifact,
    RuntimePhaseObservation,
    RuntimePlacementSnapshot,
    UnifiedScheduler,
    assess_runtime_completion,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    build_static_split_prewarm,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullFp16SmallOverlayTests(unittest.TestCase):
    def test_physical_split_contract_is_exact_and_request_scoped(
        self,
    ) -> None:
        materializer = load_module(
            "materialize_automated_runtime_catalog_contract_test",
            CAMPAIGN / "overlay_catalog.py",
        )
        model_sha256 = "a" * 64
        manifest = {
            "geometry": {
                "n_ff": 8192,
                "resident_layer_ids": [0, 3],
            },
            "model": {
                "sha256": model_sha256,
                "size_bytes": 123456,
            },
            "schema": materializer.FFN_MANIFEST_SCHEMA,
        }
        manifest["record_sha256"] = hashlib.sha256(
            materializer.canonical(manifest)
        ).hexdigest()
        policy = {
            "compiled_buckets": [
                {"max_tokens": 512, "phone_columns": 4096},
            ],
            "policy_text": "4096",
            "schema": materializer.FFN_POLICY_SCHEMA,
        }
        policy["record_sha256"] = hashlib.sha256(
            materializer.canonical(policy)
        ).hexdigest()
        binding = {
            "compiled_buckets_sha256": hashlib.sha256(
                materializer.canonical(policy["compiled_buckets"])
            ).hexdigest(),
            "manifest_record_sha256": manifest["record_sha256"],
            "policy_text": policy["policy_text"],
            "route_admission": "QUALIFIED_FOR_RUNTIME_SELECTION",
            "route_id": materializer.SPLIT_ROUTE,
        }
        route_profile = {
            "ffn_split_binding": binding,
            "routes": [{"route_id": materializer.SPLIT_ROUTE}],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            policy_path = Path(directory) / "policy.json"
            manifest_path.write_bytes(materializer.canonical(manifest))
            policy_path.write_bytes(materializer.canonical(policy))
            contract = materializer.physical_split_contract(
                route_profile,
                manifest_path=manifest_path,
                policy_path=policy_path,
                endpoint="http://127.0.0.1:18486",
                model_sha256=model_sha256,
                model_bytes=123456,
            )

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract["fraction_ppm"], 500_000)
        self.assertEqual(
            contract["operator_ids"],
            ("layer:0:ffn", "layer:3:ffn"),
        )

    def test_automated_catalog_materializes_only_audited_physical_routes(
        self,
    ) -> None:
        materializer = load_module(
            "materialize_automated_runtime_catalog_test",
            CAMPAIGN / "overlay_catalog.py",
        )
        evidence = (
            OVERLAY
            / "results/4060ti_op15_20260813/"
            "natural_runtime_v3_matched_v1"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.json"
            arguments = [
                "overlay_catalog.py",
                "--route-profile", str(evidence / "RUNTIME_PROFILE.json"),
                "--route-profile-audit",
                str(evidence / "RUNTIME_PROFILE_AUDIT.json"),
                "--kernel-profile",
                str(HARDWARE_PROFILE),
                "--requests", str(OVERLAY / "REQUESTS_LLAMA1B_10.jsonl"),
                "--model-sha256",
                "4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad",
                "--model-bytes", "770928288",
                "--cpu-endpoint", "http://127.0.0.1:18484",
                "--gpu-endpoint", "http://127.0.0.1:18485",
                "--phone-endpoint", "http://192.0.2.1:18382",
                "--host-memory-bytes", str(64 * 1024**3),
                "--gpu-memory-bytes", str(16 * 1024**3),
                "--phone-memory-bytes", str(10 * 1024**3),
                "--marginal-system-profile",
                str(OVERLAY / "MARGINAL_SYSTEM_PROFILE_V1.json"),
                "--large-model-policy", "cpu-overflow",
                "--control-overhead-profile",
                str(OVERLAY / "AUTOMATED_CONTROL_OVERHEAD_PROFILE_V1.json"),
                "--output", str(output),
            ]
            with mock.patch.object(sys, "argv", arguments):
                self.assertEqual(materializer.main(), 0)
            catalog = RuntimeCapabilityCatalog.from_json(
                json.loads(output.read_text(encoding="ascii"))
            )

        self.assertEqual(
            {row.device_id for row in catalog.executors},
            {"desktop-cpu", "desktop-cuda", "op15-phone"},
        )
        self.assertEqual(
            catalog.executor_by_device[
                "desktop-cuda"
            ].exclusive_residency_resource_id,
            "cuda0",
        )
        self.assertEqual(
            {row.device_ids for row in catalog.route_shape_profiles},
            {("desktop-cpu",), ("op15-phone",)},
        )
        self.assertTrue({
            "desktop-usb-root",
            "op15-adreno",
            "op15-functionfs",
            "op15-htp",
            "op15-ncm",
        }.issubset(catalog.resources))
        self.assertTrue(all(
            not row.coordinated_route_families
            and row.operator_plan_protocol is None
            for row in catalog.executors
        ))
        self.assertEqual(len(catalog.system_cost_profiles), 4)
        self.assertTrue(all(
            row.maturity == "QUALIFIED"
            for row in catalog.system_cost_profiles
        ))
        self.assertTrue(all(
            set(row.interference_ppm_by_resource) == set(catalog.resources)
            for row in catalog.system_cost_profiles
        ))
        self.assertEqual(
            catalog.resources["link:pcie-d2h-fit"].capacity,
            materializer.CUDA_REQUEST_SLOTS,
        )
        self.assertEqual(
            catalog.resources["link:pcie-h2d-fit"].capacity,
            materializer.CUDA_REQUEST_SLOTS,
        )
        canonical_materializer = (
            CAMPAIGN / "catalog.py"
        ).read_text(encoding="ascii")
        self.assertIn('parser.add_argument("--campaign"', canonical_materializer)
        self.assertIn("load_scheduler_configuration", canonical_materializer)

    def test_unqualified_split_metadata_does_not_require_executor(self) -> None:
        raw = json.loads(
            (OVERLAY / "SCHEDULER_PROFILE_COMPOSITE_USB_V2.json")
                .read_text(encoding="ascii")
        )
        scheduler = UnifiedScheduler(
            (ProfileBundle.from_json(raw),), "enforce"
        )
        workload_id = raw["trace_workload_map"][
            next(iter(raw["trace_workload_map"]))
        ]
        self.assertFalse(scheduler.runtime_route_configured(
            workload_id, "cpu-phone-ffn-split"
        ))
        split = dict(raw["routes"][0])
        split["baseline"] = False
        split["route_id"] = "cpu-phone-ffn-split"
        raw["routes"].append(split)
        scheduler = UnifiedScheduler(
            (ProfileBundle.from_json(raw),), "enforce"
        )
        self.assertTrue(scheduler.runtime_route_configured(
            workload_id, "cpu-phone-ffn-split"
        ))

    def test_split_prewarm_uses_largest_phone_bearing_shape(self) -> None:
        rows = [{
            "input_tokens": 3,
            "output_tokens": 7,
            "overlay_request_index": 0,
            "prompt_tokens": [10, 11, 12],
        }]
        policy = {
            "compiled_buckets": [
                {"max_tokens": 8, "phone_columns": 0},
                {"max_tokens": 32, "phone_columns": 4096},
                {"max_tokens": 512, "phone_columns": 8192},
            ],
        }
        row, shape = build_static_split_prewarm(
            rows,
            policy,
            event_id="synthetic-prewarm",
            request_index=99,
        )
        self.assertEqual(shape, {
            "input_tokens": 512,
            "phone_columns": 8192,
        })
        self.assertEqual(row["input_tokens"], 512)
        self.assertEqual(row["output_tokens"], 2)
        self.assertEqual(len(row["prompt_tokens"]), 512)
        self.assertEqual(row["prompt_tokens"][:5], [10, 11, 12, 10, 11])

    def test_direct_energy_campaign_uses_two_abba_cycles(self) -> None:
        runner = load_module(
            "measure_ffn_direct_energy_plan_test",
            OVERLAY / "measure_ffn_direct_energy.py",
        )
        self.assertEqual(
            runner.batch_plan(2),
            [
                (1, 1, "desktop-cpu"),
                (1, 1, "cpu-phone-ffn-split"),
                (1, 2, "cpu-phone-ffn-split"),
                (1, 2, "desktop-cpu"),
                (2, 3, "desktop-cpu"),
                (2, 3, "cpu-phone-ffn-split"),
                (2, 4, "cpu-phone-ffn-split"),
                (2, 4, "desktop-cpu"),
            ],
        )

    def test_direct_energy_fit_keeps_last_cycle_held_out(self) -> None:
        analyzer = load_module(
            "analyze_ffn_direct_energy_fit_test",
            OVERLAY / "analyze_ffn_direct_energy.py",
        )
        shapes = [(23, 513), (146, 44), (722, 26), (915, 292)]
        rows = []
        for cycle in (1, 2):
            for repeat in (1, 2):
                for index, (input_tokens, output_tokens) in enumerate(shapes):
                    expected = 100 + 10 * input_tokens + 100 * output_tokens
                    rows.append({
                        "case_id": f"c{cycle}-r{repeat}-q{index}",
                        "cycle": cycle,
                        "input_tokens": input_tokens,
                        "marginal_energy_uj": round(
                            expected * (1.0 if cycle == 1 else 1.05)
                        ),
                        "output_tokens": output_tokens,
                        "overlay_request_index": index,
                    })
        energy, heldout, violations = analyzer.route_fit(rows, 1)
        self.assertEqual(violations, 0)
        self.assertEqual(len(heldout), 8)
        self.assertEqual(
            energy["boundary_id"],
            analyzer.BOUNDARY,
        )
        self.assertGreaterEqual(energy["upper_error_ppm"], 100_000)

    def test_direct_energy_ratio_confidence_is_fail_closed(self) -> None:
        analyzer = load_module(
            "analyze_ffn_direct_energy_ci_test",
            OVERLAY / "analyze_ffn_direct_energy.py",
        )
        qualified = analyzer.paired_ratio_ci95([0.70, 0.69, 0.71, 0.70])
        noisy = analyzer.paired_ratio_ci95([0.60, 0.90, 0.60, 0.90])
        self.assertLess(qualified["ci95_high"], 0.95)
        self.assertGreater(noisy["ci95_high"], 0.95)

    def test_live_large_model_phase_counts_only_executing_role(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_phase_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        rows = [
            {
                "cuda_owner": "qwen",
                "functionfs_owner": "qwen",
                "kind": "large_model_phase",
                "large_model_arm": "op15",
                "phase": "qwen",
                "t_ns": 100,
            },
            {"kind": "request_arrival", "role": "hot", "t_ns": 110},
            {"kind": "request_arrival", "role": "hot", "t_ns": 120},
            {"kind": "request_arrival", "role": "cold", "t_ns": 130},
            {"kind": "request_complete", "role": "hot", "t_ns": 140},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "events.jsonl").write_bytes(b"".join(
                runner.canonical(row) for row in rows
            ))
            phase = runner.live_large_model_phase(base, 150)

        self.assertEqual(phase["phase"], "qwen")
        self.assertEqual(phase["phase_id"], 1)
        self.assertEqual(phase["active_large_requests"], 1)
        self.assertEqual(phase["functionfs_owner"], "qwen")

    def test_cpu_times_includes_iowait_as_idle(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_cpu_times_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        self.assertEqual(
            runner.cpu_times("cpu 10 20 30 40 50 60 70 80\n"),
            (360, 90),
        )

    def test_cpu_utilization_reuses_same_tick_observation(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_cpu_utilization_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        self.assertEqual(
            runner.cpu_utilization_pct((100, 40), (200, 70), None),
            70,
        )
        self.assertEqual(
            runner.cpu_utilization_pct((200, 70), (200, 70), 70),
            70,
        )
        self.assertEqual(
            runner.cpu_utilization_pct((200, 70), (200, 70), None),
            100,
        )
        with self.assertRaises(runner.OverlayError):
            runner.cpu_utilization_pct((200, 70), (199, 70), 70)

    def test_phase_power_uses_rapl_and_gpu_board_samples(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_power_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        sampler = mock.Mock()
        sampler.rows = [
            {
                "gpu": {"power_mw": 30_000},
                "rapl_package": {
                    "energy_uj": 100_000_000,
                    "max_energy_range_uj": 1_000_000_000,
                    "sample_t_ns": 1_000_000_000,
                },
            },
            {
                "gpu": {"power_mw": 50_000},
                "rapl_package": {
                    "energy_uj": 150_000_000,
                    "max_energy_range_uj": 1_000_000_000,
                    "sample_t_ns": 2_000_000_000,
                },
            },
        ]
        value = runner.phase_power_observation(sampler)
        self.assertEqual(value["cpu_package_power_mw"], 50_000)
        self.assertEqual(value["gpu_board_power_mw"], 40_000)
        self.assertEqual(value["total_server_power_mw"], 90_000)

    def test_phone_power_integration_uses_host_aligned_bounds(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_phone_power_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        rows = [
            {"host_sample_t_ns": 0, "power_mw": 500},
            {"host_sample_t_ns": 1_000_000_000, "power_mw": 1_500},
            {"host_sample_t_ns": 2_000_000_000, "power_mw": 500},
        ]
        self.assertEqual(
            runner.integrate_milliwatt_samples(
                rows,
                "power_mw",
                500_000_000,
                1_500_000_000,
            ),
            1_250_000,
        )

    def test_marginal_context_adds_live_server_and_phone_power(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_marginal_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        routes = {
            "cpu-phone-ffn-split": 200_000,
            "desktop-cpu": 200_000,
            "desktop-cuda": 0,
            "phone-adreno": 0,
        }
        profile = {
            "arms": {
                "cpu-overflow": {
                    "causal_tail_power_mw": 50_000,
                    "evidence": {"test": True},
                    "gpu_idle_power_mw": 20_000,
                    "lower_error_ppm": 100_000,
                    "measured": True,
                    "phase_power_mw": 80_000,
                    "phone_phase_power_mw": 10_000,
                    "phases": {
                        name: {"route_cpu_interference_ppm": routes}
                        for name in ("gemma", "idle", "qwen", "switching")
                    },
                    "sample_count": 2,
                    "trace_duration_upper_us": 3_000_000,
                    "upper_error_ppm": 100_000,
                },
            },
            "profile_id": "test-marginal",
            "schema": "s42-fp16-overlay-marginal-system-profile-v1",
        }
        context, receipt = runner.marginal_system_context(
            profile,
            "cpu-overflow",
            {"phase": "qwen"},
            1_000,
            {
                "sample_count": 4,
                "status": "MEASURED",
                "total_server_power_mw": 90_000,
            },
        )
        self.assertIsNotNone(context)
        self.assertEqual(context.phase_power_mw, 100_000)
        self.assertEqual(receipt["phone_phase_power_mw"], 10_000)
        observation, _ = runner.protected_work_observation(
            profile,
            "cpu-overflow",
            {"phase": "qwen"},
            1_000,
            {
                "sample_count": 4,
                "status": "MEASURED",
                "total_server_power_mw": 90_000,
            },
        )
        self.assertEqual(observation.phase_power_mw, 100_000)
        self.assertEqual(observation.stranded_idle_power_mw, 20_000)
        self.assertTrue(observation.measured)

    def test_marginal_fit_uses_cpu_service_delta(self) -> None:
        fitter = load_module(
            "fit_marginal_system_profile_test",
            OVERLAY / "fit_marginal_system_profile.py",
        )

        def run(name: str, duration_s: float, service_s: int):
            return {
                "base_result": {
                    "paid_end_ns": round(duration_s * 1_000_000_000),
                    "paid_start_ns": 0,
                    "switch": {
                        "gpu_ready_s": duration_s - 5,
                        "hot_end_s": duration_s - 10,
                    },
                },
                "name": name,
                "result": {
                    "metrics": {"duration_s": duration_s},
                    "request_results": [{
                        "completion_ns": service_s * 1_000_000_000,
                        "dispatch_ns": 0,
                        "predicted_ms": service_s * 1_000,
                        "prompt_ms": 0.0,
                        "route": "desktop-cpu",
                        "runtime_context": {
                            "large_model": {"phase": "qwen"},
                        },
                    }],
                },
            }

        coefficient, receipt = fitter.pair_interference_ppm(
            run("static", 100.0, 10),
            run("runtime", 98.0, 4),
        )
        self.assertEqual(coefficient, 333_333)
        self.assertEqual(receipt["service_delta_us"], 6_000_000)
        phase_coefficient, phase_receipt = (
            fitter.pair_phase_interference_ppm(
                run("static", 100.0, 10),
                run("runtime", 98.0, 4),
                "qwen",
            )
        )
        self.assertEqual(phase_coefficient, 333_333)
        self.assertEqual(phase_receipt["phase"], "qwen")
        self.assertEqual(
            phase_receipt["static_phase_duration_us"],
            90_000_000,
        )

    def test_contention_observation_separates_endpoint_queue(self) -> None:
        fitter = load_module(
            "fit_phase_contention_service_test",
            OVERLAY / "fit_phase_contention_profile.py",
        )
        row = {
            "completion_ns": 10_000_000_000,
            "dispatch_ns": 0,
            "event_id": "queued-request",
            "input_tokens": 100,
            "output_tokens": 20,
            "overlay_request_index": 0,
            "predicted_ms": 2.0,
            "prompt_ms": 1.0,
            "route": "cpu-phone-ffn-split",
            "runtime_context": {
                "cost_features": {
                    "active_cpu_slots": 2,
                    "actual_batch_size": 100,
                    "large_model_op15": 0,
                    "large_phase_id": 1,
                    "memory_bandwidth_pressure_basis_points": 3,
                },
                "phase_power_observation": {
                    "status": "MEASURED",
                    "total_server_power_mw": 100_000,
                },
            },
        }
        value = fitter.physical_observation(
            row, "holdout", "cpu-phone-ffn-split"
        )
        self.assertEqual(value["latency_us"], 3_000)
        self.assertEqual(value["controller_wall_us"], 10_000_000)
        self.assertEqual(value["endpoint_queue_us"], 9_997_000)

    def test_split_route_requires_isolated_incremental_energy(self) -> None:
        materializer = load_module(
            "materialize_ffn_split_energy_test",
            OVERLAY / "materialize_ffn_split_route.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "split-shapes.stderr"
            split_path.write_text(
                "\n".join((
                    'S41SERVERFFN {"status":"ok","calls":3}',
                    "S41SERVERFFNSHAPE "
                    '{"tokens":210,"columns":8192,"calls":3,'
                    '"overlap_mean_ms":40.0,"wait_mean_ms":39.0}',
                    "",
                )),
                encoding="ascii",
            )
            split_log = materializer.parse_split_log(split_path)
            self.assertEqual(split_log["summary"]["calls"], 3)
            self.assertEqual(split_log["shapes"][0]["columns"], 8192)
            self.assertEqual(
                materializer.exposed_join_wait_upper_ppm(
                    [split_log], 8192
                ),
                0,
            )
            partial_log = {
                "shapes": [{
                    "columns": 4096,
                    "overlap_mean_ms": 8.0,
                    "wait_mean_ms": 2.0,
                }],
            }
            self.assertEqual(
                materializer.exposed_join_wait_upper_ppm(
                    [partial_log], 8192
                ),
                250_000,
            )

        manifest = {
            "model": {"sha256": "1" * 64},
            "record_sha256": "2" * 64,
        }
        policy = {
            "compiled_buckets": [
                {"max_tokens": 512, "phone_columns": 8192}
            ],
            "policy_text": "512:8192",
            "record_sha256": "3" * 64,
        }
        direct = {
            "binding": {
                "manifest_record_sha256": manifest["record_sha256"],
                "model_sha256": manifest["model"]["sha256"],
                "policy_execution_sha256": (
                    materializer.policy_execution_sha256(policy)
                ),
                "policy_record_sha256": policy["record_sha256"],
                "route_id": "cpu-phone-ffn-split",
            },
            "qualification": {
                "heldout_upper_bound_violations": 0,
                "idle_baseline_sample_count": 3,
                "idle_baseline_subtracted": True,
                "max_concurrent_measured_requests": 1,
                "method": materializer.DIRECT_ENERGY_METHOD,
                "repeat_count": 2,
                "request_windows_non_overlapping": True,
                "split_over_cpu_ratio_ci95": {
                    "ci95_high": 0.8,
                    "ci95_low": 0.6,
                    "mean": 0.7,
                    "samples": 2,
                },
                "status": "PASS",
            },
            "route_energy": {
                "boundary_id": materializer.BOUNDARY,
                "cost_uj": {
                    "fixed": 0,
                    "input_token": 100,
                    "kind": "affine_tokens_v1",
                    "output_token": 1_000,
                },
                "lower_error_ppm": 100_000,
                "status": "measured",
                "upper_error_ppm": 200_000,
            },
            "schema": materializer.DIRECT_ENERGY_SCHEMA,
            "status": "PASS",
        }
        direct["record_sha256"] = hashlib.sha256(
            materializer.canonical(direct)
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            energy_path = Path(directory) / "DIRECT_ENERGY.json"
            energy_path.write_bytes(materializer.canonical(direct))
            model, audit = materializer.direct_route_energy(
                energy_path, manifest, policy
            )
            equivalent_policy = {
                **policy,
                "record_sha256": "4" * 64,
            }
            materializer.direct_route_energy(
                energy_path, manifest, equivalent_policy
            )
            invalid = json.loads(json.dumps(direct))
            invalid["qualification"]["max_concurrent_measured_requests"] = 2
            invalid.pop("record_sha256")
            invalid["record_sha256"] = hashlib.sha256(
                materializer.canonical(invalid)
            ).hexdigest()
            invalid_path = Path(directory) / "INVALID_DIRECT_ENERGY.json"
            invalid_path.write_bytes(materializer.canonical(invalid))
            with self.assertRaises(materializer.MaterializationError):
                materializer.direct_route_energy(
                    invalid_path, manifest, policy
                )

        self.assertEqual(model["cost_uj"]["kind"], "affine_tokens_v1")
        self.assertEqual(
            audit["method"], materializer.DIRECT_ENERGY_METHOD
        )
        raw = json.loads((
            OVERLAY / "SCHEDULER_PROFILE_COMPOSITE_USB_V2.json"
        ).read_text(encoding="ascii"))
        raw["resources"].append({
            "capacity": 1,
            "identity": "test-htp3",
            "kind": "npu",
            "ready": True,
            "resource_id": "op15-htp",
        })
        raw["routes"].append({
            "baseline": False,
            "energy": model,
            "evidence_ids": ["sha256:" + "1" * 64],
            "granularity": "operator",
            "latency": {
                "kind": "conditioned_affine_features_v1",
                "selector_feature": "contention_class_id",
                "variants": [{
                    "cost_us": {
                        "coefficients": {
                            name: 1 for name in (
                                "active_cpu_slots",
                                "actual_batch_size",
                                "input_tokens",
                                "memory_bandwidth_pressure_basis_points",
                                "output_tokens",
                            )
                        },
                        "fixed": 1,
                        "kind": "affine_features_v1",
                    },
                    "label": f"phase-{value}",
                    "measured": True,
                    "sample_count": 20,
                    "selector_value": value,
                    "ucb_add_us": 1,
                } for value in range(1, 5)],
            },
            "overlap": {
                "exposed_join_wait_ppm": 0,
                "sample_count": 20,
                "status": "measured",
                "upper_error_ppm": 1,
            },
            "placement_verified": True,
            "quality_class": "bounded_numeric",
            "resident": True,
            "resource_slots": {
                "desktop-cpu": 1,
                "desktop-usb-root": 1,
                "op15-htp": 1,
                "op15-ncm": 1,
            },
            "route_id": "cpu-phone-ffn-split",
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": 770_928_288,
            "workload_id": "llama-1b-resident-task",
        })
        parsed = ProfileBundle.from_json(raw)
        self.assertIn(
            "cpu-phone-ffn-split",
            {route.route_id for route in parsed.routes},
        )

    def test_phone_readiness_admits_a_healthy_busy_endpoint(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_probe_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        with mock.patch.object(runner, "endpoint_ready") as health, \
                mock.patch.object(runner, "endpoint_slots") as slots:
            health.return_value = True
            slots.return_value = [{
                "id_task": 7,
                "is_processing": True,
            }]
            observed = runner.executor_runtime_snapshot("phone", 1)

        health.assert_called_once_with("phone", 1)
        slots.assert_called_once_with("phone", 1, 1)
        self.assertEqual(observed["health"], "healthy")
        self.assertEqual(observed["free_slots"], 0)
        self.assertEqual(observed["processing_task_ids"], [7])

    def test_live_phone_snapshot_retries_a_transient_probe_failure(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_snapshot_retry_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        expected = {"schema": "s42-op15-live-snapshot-v1"}
        with mock.patch.object(
            runner,
            "live_phone_snapshot",
            side_effect=(None, expected),
        ) as probe, mock.patch.object(runner.time, "sleep") as sleep:
            observed = runner.wait_for_live_phone_snapshot(
                "phone", 18383, 805_306_368, 1
            )

        self.assertIs(observed, expected)
        self.assertEqual(probe.call_count, 2)
        sleep.assert_called_once()

    def test_functionfs_reservation_leaves_a_composite_lane_for_ncm(self) -> None:
        raw_profile = json.loads(
            (OVERLAY / "SCHEDULER_PROFILE_COMPOSITE_USB_V2.json").read_text(
                encoding="ascii"
            )
        )
        profile = ProfileBundle.from_json(raw_profile)
        scheduler = UnifiedScheduler((profile,), "enforce")
        scheduler.set_resource_ready("cuda0", False, 0)
        functionfs = scheduler.reserve_external_resource(
            "op15-functionfs", "fp16-functionfs", 0, 7_200_000_000
        )
        root = scheduler.reserve_external_resource(
            "desktop-usb-root", "fp16-root-lane", 0, 7_200_000_000
        )

        inventory = json.loads(
            (OVERLAY / "TRACE_MANIFEST.json").read_text(encoding="ascii")
        )["model_inventory"]["llama-3.2-1b-instruct-q4_0"]
        model = RuntimeModelArtifact(
            "llama-3.2-1b-instruct-q4_0",
            inventory["artifact_sha256"],
            inventory["artifact_bytes"],
        )
        request = Request(
            request_id="composite-usb-phone-route",
            workload_id="llama-1b-resident-task",
            arrival_us=1_000,
            deadline_us=30_001_000,
            input_tokens=500,
            output_tokens=100,
            quality_requirement="bounded_numeric",
        )
        snapshot = RuntimePlacementSnapshot(
            snapshot_id="composite-usb-live-snapshot",
            captured_at_us=900,
            valid_until_us=2_000,
            capacities={
                "cuda0-vram": DeviceMemoryCapacity(
                    "cuda0-vram", 16 * 1024**3, 15 * 1024**3, 512 * 1024**2
                ),
                "host-ram": DeviceMemoryCapacity(
                    "host-ram", 64 * 1024**3, 20 * 1024**3, 2 * 1024**3
                ),
                "op15-ram": DeviceMemoryCapacity(
                    "op15-ram", 15 * 1024**3, 11 * 1024**3, 768 * 1024**2
                ),
            },
        )
        bindings = (
            RuntimeExecutorBinding(
                executor_id="http://127.0.0.1:18484",
                route_id="desktop-cpu",
                model_id=model.model_id,
                artifact_sha256=model.artifact_sha256,
                artifact_bytes=model.artifact_bytes,
                backend="cpu",
                resource_ids=("desktop-cpu",),
                memory_resource_id="host-ram",
                resident=True,
                ready=True,
            ),
            RuntimeExecutorBinding(
                executor_id="http://[op15-ncm]:18382",
                route_id="phone-adreno",
                model_id=model.model_id,
                artifact_sha256=model.artifact_sha256,
                artifact_bytes=model.artifact_bytes,
                backend="phone-adreno-ncm",
                resource_ids=(
                    "op15-adreno",
                    "op15-ncm",
                    "desktop-usb-root",
                ),
                memory_resource_id="op15-ram",
                resident=True,
                ready=True,
            ),
        )

        estimates, decision = scheduler.schedule_runtime(
            request,
            model,
            bindings,
            snapshot=snapshot,
            now_us=1_000,
        )

        phone = next(
            estimate for estimate in estimates.estimates
            if estimate.route_id == "phone-adreno"
        )
        root_lease = next(
            lease for lease in decision.leases
            if lease.resource_id == "desktop-usb-root"
        )
        self.assertTrue(phone.admitted)
        self.assertEqual(decision.route_id, "phone-adreno")
        self.assertEqual(decision.reason, "VERIFIED_ENERGY_SAVING")
        self.assertEqual(functionfs[0].resource_id, "op15-functionfs")
        self.assertEqual(root[0].lanes, (0,))
        self.assertEqual(root_lease.lanes, (1,))

    def test_slot_probe_reports_occupancy_separately_from_health(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_slots_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        connection = mock.Mock()
        response = mock.Mock()
        response.status = 200
        response.read.return_value = b'[{"id":0,"is_processing":true}]'
        connection.getresponse.return_value = response
        with mock.patch.object(
            runner.http.client,
            "HTTPConnection",
            return_value=connection,
        ):
            self.assertFalse(runner.endpoint_has_idle_slot("phone", 1))

        response.read.return_value = b'[{"id":0,"is_processing":false}]'
        with mock.patch.object(
            runner.http.client,
            "HTTPConnection",
            return_value=connection,
        ):
            self.assertTrue(runner.endpoint_has_idle_slot("phone", 1))

    def test_completion_recovers_task_id_after_a_short_request(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_completion_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        chunks = [
            {
                "tokens": [7],
                "tokens_predicted": 1,
                "stop": False,
            },
            {
                "content": "ok",
                "id_slot": 3,
                "model": runner.LLAMA1,
                "timings": {
                    "predicted_ms": 2.0,
                    "predicted_n": 2,
                    "prompt_ms": 1.0,
                    "prompt_n": 2,
                },
                "tokens": [8],
                "tokens_predicted": 2,
                "stop": True,
            },
        ]
        response = mock.Mock()
        response.status = 200
        response.readline.side_effect = [
            f"data: {json.dumps(chunk)}\n".encode("utf-8")
            for chunk in chunks
        ] + [b""]
        connection = mock.Mock()
        connection.getresponse.return_value = response
        first_tokens = []
        row = {
            "event_id": "short-request",
            "input_tokens": 2,
            "output_tokens": 2,
            "overlay_request_index": 0,
            "prompt_tokens": [1, 2],
        }
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(
                    runner.http.client,
                    "HTTPConnection",
                    return_value=connection,
                ),
                mock.patch.object(
                    runner,
                    "endpoint_slots",
                    side_effect=[
                        TimeoutError("busy"),
                        [{"id": 3, "id_task": 9}],
                    ],
                ) as slots,
            ):
                receipt = runner.endpoint_completion(
                    "host",
                    1,
                    row,
                    Path(temporary) / "stream.raw",
                    first_tokens.append,
                )
        self.assertEqual(receipt["endpoint_slot_id"], 3)
        self.assertEqual(receipt["endpoint_task_id"], 9)
        self.assertEqual(
            receipt["endpoint_slot_probe_errors"],
            ["active:TimeoutError:busy"],
        )
        self.assertEqual(slots.call_count, 2)
        self.assertEqual(len(first_tokens), 1)

    def test_phone_snapshot_separates_health_capacity_and_thermal(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_phone_snapshot_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        value = {
            "android_thermal_status": 0,
            "captured_epoch_s": 100,
            "mem_available_kib": 2_000_000,
            "mem_total_kib": 8_000_000,
            "schema": "s42-op15-live-snapshot-v1",
            "sequence": 9,
            "task_server_alive": True,
            "temperature_max_millic": 70_000,
            "thermal_state": "nominal",
            "throttling_state": "not-observed",
        }
        with mock.patch.object(
            runner, "endpoint_json", return_value=value
        ), mock.patch.object(runner.time, "time", return_value=102):
            snapshot = runner.live_phone_snapshot(
                "phone", 18383, 768 * 1024**2
            )
        assert snapshot is not None
        self.assertTrue(snapshot["memory_reserve_met"])
        self.assertTrue(snapshot["thermal_qualified"])
        self.assertEqual(snapshot["sample_age_s"], 2)

        value.update({
            "android_thermal_status": 1,
            "thermal_state": "hot",
            "throttling_state": "android-thermal-throttling",
        })
        with mock.patch.object(
            runner, "endpoint_json", return_value=value
        ), mock.patch.object(runner.time, "time", return_value=102):
            snapshot = runner.live_phone_snapshot(
                "phone", 18383, 768 * 1024**2
            )
        assert snapshot is not None
        self.assertFalse(snapshot["thermal_qualified"])

    def test_missing_or_low_phone_memory_has_no_schedulable_capacity(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_memory_fallback_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        missing = runner.phone_memory_capacity(None, 16_000, 800)
        self.assertEqual(missing.available_bytes, 0)
        low = runner.phone_memory_capacity({
            "available_bytes": 500,
            "capacity_bytes": 16_000,
        }, 16_000, 800)
        self.assertEqual(low.available_bytes, 0)
        enough = runner.phone_memory_capacity({
            "available_bytes": 1_200,
            "capacity_bytes": 16_000,
        }, 16_000, 800)
        self.assertEqual(enough.available_bytes, 400)

    def test_phase_transport_lease_releases_on_owner_change(self) -> None:
        profile = ProfileBundle.from_json(json.loads(
            (OVERLAY / "SCHEDULER_PROFILE_COMPOSITE_USB_V2.json")
                .read_text(encoding="ascii")
        ))
        scheduler = UnifiedScheduler((profile,), "enforce")
        resources = (
            "desktop-usb-root", "op15-adreno", "op15-functionfs"
        )
        active = scheduler.observe_runtime_phase(
            RuntimePhaseObservation(
                "synthetic-large-work", "active", "model-a", 0, resources
            ),
            observed_at_us=0,
            renewal_us=5_000,
        )
        self.assertEqual(active["status"], "LEASED")
        idle = scheduler.observe_runtime_phase(
            RuntimePhaseObservation(
                "synthetic-large-work", "idle", None, 10_000, ()
            ),
            observed_at_us=10_000,
            renewal_us=5_000,
        )
        self.assertEqual(idle["status"], "AVAILABLE")
        history = scheduler.runtime_phase_history("synthetic-large-work")
        self.assertEqual(history[0]["released_at_us"], 10_000)
        self.assertEqual(
            scheduler.resource_snapshot(10_000)["op15-functionfs"]
                ["free_slots"],
            1,
        )

    def test_large_phone_phase_leases_profiled_htp(self) -> None:
        raw = json.loads(
            (OVERLAY / "SCHEDULER_PROFILE_COMPOSITE_USB_V2.json")
                .read_text(encoding="ascii")
        )
        raw["resources"].append({
            "capacity": 1,
            "identity": "test-htp3",
            "kind": "npu",
            "ready": True,
            "resource_id": "op15-htp",
        })
        scheduler = UnifiedScheduler((ProfileBundle.from_json(raw),), "enforce")
        resources = (
            "desktop-usb-root",
            "op15-adreno",
            "op15-functionfs",
            "op15-htp",
        )
        state = scheduler.observe_runtime_phase(
            RuntimePhaseObservation(
                "synthetic-large-work", "active", "model-a", 0, resources
            ),
            observed_at_us=0,
            renewal_us=5_000,
        )
        active = scheduler.resource_snapshot(0)
        self.assertEqual(active["op15-htp"]["free_slots"], 0)
        self.assertIn("op15-htp", state["resource_ids"])

        scheduler.observe_runtime_phase(
            RuntimePhaseObservation(
                "synthetic-large-work", "idle", None, 10_000, ()
            ),
            observed_at_us=10_000,
            renewal_us=5_000,
        )
        self.assertEqual(
            scheduler.resource_snapshot(10_000)["op15-htp"]["free_slots"],
            1,
        )

    def test_split_binding_uses_idle_op15_window_and_profiled_class(self) -> None:
        def observation(profiled: bool) -> RuntimeExecutorObservation:
            return RuntimeExecutorObservation(
                executor_id="memory://split",
                route_id="synthetic-split",
                backend="composite",
                resource_ids=("host-compute", "helper-compute"),
                memory_resource_id="helper-memory",
                resident=True,
                health="healthy",
                qualification_facts={
                    "contention_class_profiled": profiled,
                    "thermal_qualified": True,
                },
                route_family="cpu-phone",
            )

        self.assertEqual(observation(True).eligibility_reasons, ())
        self.assertEqual(
            observation(False).eligibility_reasons,
            ("QUALIFICATION:contention_class_profiled",),
        )

    def test_phone_binding_waits_for_op15_phase_release(self) -> None:
        qualified = RuntimeExecutorObservation(
            executor_id="memory://helper",
            route_id="synthetic-helper",
            backend="accelerator",
            resource_ids=("helper-compute", "helper-link"),
            memory_resource_id="helper-memory",
            resident=True,
            health="healthy",
            qualification_facts={"thermal_qualified": True},
            route_family="phone",
        )
        unqualified = RuntimeExecutorObservation(
            **{
                **qualified.__dict__,
                "qualification_facts": {"thermal_qualified": False},
            }
        )
        self.assertEqual(qualified.eligibility_reasons, ())
        self.assertEqual(
            unqualified.eligibility_reasons,
            ("QUALIFICATION:thermal_qualified",),
        )

    def test_lease_coverage_is_distinct_from_latency_prediction(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_lease_assessment_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        profile = ProfileBundle.from_json(json.loads(
            (OVERLAY / "SCHEDULER_PROFILE_COMPOSITE_USB_V2.json")
                .read_text(encoding="ascii")
        ))
        scheduler = UnifiedScheduler((profile,), "control")
        decision = scheduler.schedule(Request(
            request_id="lease-assessment",
            workload_id=runner.WORKLOAD_ID,
            arrival_us=1_000,
            deadline_us=30_000_000,
            input_tokens=32,
            output_tokens=8,
            quality_requirement="bounded_numeric",
        ))
        actual_end_us = decision.finish_upper_us + 1_000
        final_ends = {
            lease.token: actual_end_us + 1_000
            for lease in decision.leases
        }

        latency, coverage = assess_runtime_completion(
            decision, final_ends, actual_end_us
        )

        self.assertTrue(coverage.covered)
        self.assertEqual(coverage.status, "COVERED")
        self.assertFalse(latency.met)
        self.assertEqual(latency.overrun_us, 1_000)

    def test_legacy_release_uses_final_renewal_for_lease_coverage(self) -> None:
        qualifier = load_module(
            "qualify_fp16_small_overlay_lease_test",
            OVERLAY / "qualify_fp16_small_overlay.py",
        )
        row = {
            "overlay_request_index": 0,
            "scheduler_final_decision": {
                "finish_upper_us": 100,
                "leases": [{
                    "reserved_until_us": 100,
                    "token": "lease-1",
                }],
            },
            "scheduler_release": {
                "actual_end_us": 105,
                "upper_bound_violation": True,
            },
        }
        runtime = {"renewal_receipts": {"0": [{
            "extended_leases": [{
                "reserved_until_us": 110,
                "token": "lease-1",
            }],
        }]}}

        self.assertTrue(
            qualifier.release_has_final_lease_coverage(row, runtime)
        )
        self.assertFalse(
            qualifier.release_meets_original_latency_upper_bound(row)
        )

    def test_phone_connect_failure_is_retry_safe_before_request(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_connect_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        connection = mock.Mock()
        connection.connect.side_effect = OSError(
            errno.ENETUNREACH, "Network is unreachable"
        )
        row = {
            "output_tokens": 1,
            "overlay_request_index": 0,
            "prompt_tokens": [1],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner.http.client,
            "HTTPConnection",
            return_value=connection,
        ):
            with self.assertRaises(runner.EndpointTransportError) as raised:
                runner.endpoint_completion(
                    "phone", 1, row, Path(temporary) / "stream.raw", lambda _: None
                )

        self.assertTrue(raised.exception.retry_safe)
        self.assertEqual(raised.exception.phase, "connect")
        self.assertTrue(RuntimeExecutionFailure(
            phase=raised.exception.phase,
            retry_safe=raised.exception.retry_safe,
            execution_started=False,
        ).fallback_allowed)

    def test_failure_after_connect_is_not_retry_safe(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_request_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        connection = mock.Mock()
        connection.request.side_effect = OSError(errno.EPIPE, "Broken pipe")
        row = {
            "output_tokens": 1,
            "overlay_request_index": 0,
            "prompt_tokens": [1],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner.http.client,
            "HTTPConnection",
            return_value=connection,
        ):
            with self.assertRaises(runner.EndpointTransportError) as raised:
                runner.endpoint_completion(
                    "phone", 1, row, Path(temporary) / "stream.raw", lambda _: None
                )

        self.assertFalse(raised.exception.retry_safe)
        self.assertFalse(RuntimeExecutionFailure(
            phase=raised.exception.phase,
            retry_safe=raised.exception.retry_safe,
            execution_started=False,
        ).fallback_allowed)

    def test_runtime_wait_never_passes_a_negative_delay(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        clock = iter((90, 101))
        delays = []
        with mock.patch.object(
            runner.time,
            "monotonic_ns",
            side_effect=lambda: next(clock),
        ), mock.patch.object(
            runner.time,
            "sleep",
            side_effect=delays.append,
        ):
            runner.sleep_until_ns(100)

        self.assertEqual(delays, [10 / 1e9])

    def test_resident_release_publish_is_atomic(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_atomic_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident-release.json"
            value = {
                "released_at_ns": 1,
                "schema": "s42-resident-release-v1",
                "status": "OVERLAY_COMPLETE",
            }
            runner.write_new_atomic(path, value)
            self.assertEqual(json.loads(path.read_text(encoding="ascii")), value)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_overlay_is_exact_and_reproducible(self) -> None:
        builder = load_module(
            "build_fp16_small_overlay_test",
            OVERLAY / "build_fp16_small_overlay.py",
        )
        trace, manifest = builder.derive()
        self.assertEqual(trace, (OVERLAY / "REQUESTS_LLAMA1B_10.jsonl").read_bytes())
        self.assertEqual(manifest, (OVERLAY / "TRACE_MANIFEST.json").read_bytes())
        value = json.loads(manifest)
        self.assertEqual(
            value["combined_work"],
            {"input_tokens": 38948, "output_tokens": 13132, "record_count": 84},
        )

    def test_phase_calibration_overlay_is_exact_and_split(self) -> None:
        builder = load_module(
            "build_phase_calibration_overlay_test",
            OVERLAY / "build_phase_calibration_overlay.py",
        )
        trace, manifest = builder.derive()
        self.assertEqual(
            trace,
            (OVERLAY / "REQUESTS_LLAMA1B_PHASE_CALIBRATION_40.jsonl")
                .read_bytes(),
        )
        self.assertEqual(
            manifest,
            (OVERLAY / "TRACE_PHASE_CALIBRATION_MANIFEST.json").read_bytes(),
        )
        rows = [json.loads(line) for line in trace.splitlines()]
        self.assertEqual(len(rows), 40)
        self.assertEqual(
            {row["target_large_phase"] for row in rows},
            {"qwen", "switching", "gemma", "idle"},
        )
        self.assertEqual(
            sum(row["calibration_split"] == "holdout" for row in rows),
            12,
        )

    def test_natural_validation_overlay_spans_full_horizon(self) -> None:
        builder = load_module(
            "build_natural_validation_overlay_test",
            OVERLAY / "build_natural_validation_overlay.py",
        )
        trace, manifest = builder.derive()
        self.assertEqual(
            trace,
            (OVERLAY / "REQUESTS_LLAMA1B_NATURAL_VALIDATION_40.jsonl")
                .read_bytes(),
        )
        self.assertEqual(
            manifest,
            (OVERLAY / "TRACE_NATURAL_VALIDATION_MANIFEST.json")
                .read_bytes(),
        )
        rows = [json.loads(line) for line in trace.splitlines()]
        arrivals = [row["arrival_us"] for row in rows]
        self.assertEqual(len(rows), 40)
        self.assertEqual(arrivals, sorted(set(arrivals)))
        self.assertGreater(arrivals[-1], 3_000_000_000)
        self.assertTrue(all(
            "target_large_phase" not in row
            and "calibration_split" not in row
            for row in rows
        ))
        value = json.loads(manifest)
        self.assertFalse(value["arrival_contract"]["phase_labels_used"])
        self.assertEqual(
            value["derivation"]["donor_repeats"],
            4,
        )

    def test_idle_split_calibration_targets_only_op15_idle(self) -> None:
        builder = load_module(
            "build_idle_split_calibration_overlay_test",
            OVERLAY / "build_idle_split_calibration_overlay.py",
        )
        trace, manifest = builder.derive()
        self.assertEqual(
            trace,
            (OVERLAY / "REQUESTS_LLAMA1B_IDLE_SPLIT_CALIBRATION_40.jsonl")
                .read_bytes(),
        )
        self.assertEqual(
            manifest,
            (OVERLAY / "TRACE_IDLE_SPLIT_CALIBRATION_MANIFEST.json")
                .read_bytes(),
        )
        rows = [json.loads(line) for line in trace.splitlines()]
        self.assertEqual(len(rows), 40)
        self.assertEqual(
            {row["target_large_phase"] for row in rows}, {"idle"}
        )
        self.assertEqual(
            sum(row["calibration_split"] == "holdout" for row in rows),
            12,
        )

    def test_gguf_ffn_manifest_selects_exact_dense_weights(self) -> None:
        exporter = load_module(
            "derive_llama_ffn_manifest_test",
            OVERLAY / "derive_llama_ffn_manifest.py",
        )
        descriptors = []
        for layer in range(2):
            for role in exporter.FFN_ROLES:
                descriptors.append({
                    "n_bytes": 32,
                    "name": exporter.dense_ffn_name(layer, role),
                    "shape": (
                        [8, 4] if role == "ffn_down" else [4, 8]
                    ),
                    "tensor_type": "Q4_0",
                })
        selected = exporter.select_dense_ffn_tensors(
            descriptors, block_count=2, n_embd=4, n_ff=8
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["layer_id"] for row in selected}, {0, 1})
        with self.assertRaisesRegex(
            exporter.ManifestError, "missing dense FFN tensor"
        ):
            exporter.select_dense_ffn_tensors(
                descriptors[:-1], block_count=2, n_embd=4, n_ff=8
            )

    def test_unified_vq_compiles_a_physical_split_table(self) -> None:
        exporter = load_module(
            "derive_llama_ffn_manifest_compile_test",
            OVERLAY / "derive_llama_ffn_manifest.py",
        )
        compiler = load_module(
            "compile_llama_ffn_split_test",
            OVERLAY / "compile_llama_ffn_split.py",
        )
        calibrator = load_module(
            "calibrate_llama_ffn_split_policy_test",
            OVERLAY / "calibrate_llama_ffn_split_policy.py",
        )
        manifest = {
            "geometry": {
                "block_count": 16,
                "n_embd": 2048,
                "n_ff": 8192,
                "quantization": "q4_0",
                "resident_layer_ids": list(range(12)),
            },
            "model": {
                "id": exporter.MODEL_ID,
                "sha256": "b" * 64,
            },
            "resident_slice": {
                "raw_bytes": 475_000_000,
                "weight_sha256": "a" * 64,
            },
            "schema": compiler.MANIFEST_SCHEMA,
            "split_contract": {
                "column_quantum": 1024,
                "max_tokens": 512,
            },
        }
        manifest["aggregate_ops"] = [
            exporter.aggregate_op(manifest, m)
            for m in (1, 8, 32, 128, 512)
        ]
        self.assertTrue(all(
            row["layer_id"] == "layers-0-11-dense-ffn"
            for row in manifest["aggregate_ops"]
        ))
        manifest["record_sha256"] = hashlib.sha256(
            compiler.canonical(manifest)
        ).hexdigest()
        campaign = json.loads((
            HARDWARE_PROFILE
        ).read_text(encoding="ascii"))
        value = compiler.compile_policy(
            manifest,
            campaign,
            phone_capacity_bytes=15_846_404_096,
            phone_available_bytes=1_500_000_000,
            phone_reserve_bytes=805_306_368,
        )
        self.assertEqual(
            [row["max_tokens"] for row in value["compiled_buckets"]],
            [1, 8, 32, 128, 512],
        )
        self.assertTrue(all(
            row["phone_columns"] % 1024 == 0
            for row in value["compiled_buckets"]
        ))
        self.assertEqual(
            value["qualification"]["route_admission"],
            "SHADOW_ONLY_UNTIL_HELDOUT_PHYSICAL_PROFILE",
        )
        lines = [
            'S41SERVERFFN {"status":"ok","calls":12}',
        ]
        for tokens, columns, calls, overlap_ms in (
            (1, 5120, 1, 2.5),
            (146, 8192, 1, 1000.0),
            (210, 8192, 2, 25.0),
            (345, 8192, 3, 40.0),
            (512, 8192, 5, 50.0),
        ):
            lines.append(
                "S41SERVERFFNSHAPE "
                + json.dumps({
                    "calls": calls,
                    "columns": columns,
                    "overlap_mean_ms": overlap_ms,
                    "tokens": tokens,
                }, separators=(",", ":"), sort_keys=True)
            )
        calibration = calibrator.calibrate(
            manifest,
            value,
            ("\n".join(lines) + "\n").encode("ascii"),
            "synthetic-contention",
            "c" * 64,
        )
        self.assertEqual(calibration["policy_text"], "209:0,512:8192")
        guarded = compiler.compile_policy(
            manifest,
            campaign,
            phone_capacity_bytes=15_846_404_096,
            phone_available_bytes=1_500_000_000,
            phone_reserve_bytes=805_306_368,
            physical_calibration=calibration,
        )
        self.assertEqual(guarded["policy_text"], "209:0,512:8192")
        self.assertTrue(
            guarded["qualification"]["physical_shape_calibrated"]
        )
        mismatched = json.loads(json.dumps(manifest))
        mismatched["resident_slice"]["raw_bytes"] += 1
        mismatched.pop("record_sha256")
        mismatched["record_sha256"] = hashlib.sha256(
            compiler.canonical(mismatched)
        ).hexdigest()
        with self.assertRaisesRegex(
            compiler.CompileError, "physical calibration contract"
        ):
            compiler.compile_policy(
                mismatched,
                campaign,
                phone_capacity_bytes=15_846_404_096,
                phone_available_bytes=1_500_000_000,
                phone_reserve_bytes=805_306_368,
                physical_calibration=calibration,
            )

    def test_contention_fit_qualifies_only_bounded_holdout(self) -> None:
        fitter = load_module(
            "fit_phase_contention_profile_test",
            OVERLAY / "fit_phase_contention_profile.py",
        )
        rows = []
        for index in range(20):
            features = {
                "active_cpu_slots": index % 4,
                "actual_batch_size": min(512, 100 + 10 * index),
                "input_tokens": 100 + 10 * index,
                "memory_bandwidth_pressure_basis_points": index % 3,
                "output_tokens": 20 + index,
            }
            latency_us = (
                10_000
                + 3 * features["input_tokens"]
                + 11 * features["output_tokens"]
                + 1_000 * features["active_cpu_slots"]
                + 2 * features["actual_batch_size"]
                + 100
                    * features["memory_bandwidth_pressure_basis_points"]
            )
            rows.append({
                "event_id": f"r{index}",
                "features": features,
                "latency_us": latency_us,
                "phase_power_mw": 100_000 + index,
                "split": "train" if index < 14 else "holdout",
            })
        profile, audit = fitter.variant(1, rows)
        self.assertTrue(profile["measured"])
        self.assertEqual(audit["holdout_upper_violations"], 0)

        rows[-1] = {**rows[-1], "latency_us": 10**9}
        profile, audit = fitter.variant(1, rows)
        self.assertFalse(profile["measured"])
        self.assertEqual(audit["holdout_upper_violations"], 1)

    def test_contention_fit_audit_fails_if_a_variant_is_unmeasured(self) -> None:
        fitter = load_module(
            "fit_phase_contention_audit_test",
            OVERLAY / "fit_phase_contention_profile.py",
        )
        audits = [{"measured": True}] * 7 + [{"measured": False}]
        self.assertFalse(all(row["measured"] for row in audits))
        source = (OVERLAY / "fit_phase_contention_profile.py").read_text(
            encoding="ascii"
        )
        self.assertIn(
            '"status": "PASS" if all_variants_measured else "FAIL"',
            source,
        )
        self.assertIn(
            'return 0 if audit["status"] == "PASS" else 2',
            source,
        )
        self.assertIn('audit["profile_sha256"]', source)

    def test_natural_route_fit_normalizes_legacy_contention(self) -> None:
        fitter = load_module(
            "fit_runtime_route_profile_features_test",
            OVERLAY / "fit_runtime_route_profile.py",
        )
        features = fitter.normalized_features({
            "input_tokens": 700,
            "output_tokens": 20,
            "runtime_context": {"cost_features": {
                "active_large_cpu_requests": 12,
                "active_small_cpu_requests": 3,
                "contention_class_id": 1,
                "large_model_op15": 0,
                "large_phase_id": 1,
                "memory_stall_avg10_basis_points": 17,
            }},
        })
        self.assertEqual(features["active_cpu_slots"], 7)
        self.assertEqual(features["actual_batch_size"], 512)
        self.assertEqual(features["contention_class_id"], 1)
        self.assertEqual(features["source_feature_contract"], 1)

    def test_natural_route_fit_updates_service_and_phone_leases(self) -> None:
        fitter = load_module(
            "fit_runtime_route_profile_integration_test",
            OVERLAY / "fit_runtime_route_profile.py",
        )
        results = (
            OVERLAY
            / "results/4060ti_op15_20260812/"
            "isolated_cpu_overflow_abba_v1"
        )
        profile, audit = fitter.fit(
            [
                results / "static_r1/RESULT.json",
                results / "runtime_r1/RESULT.json",
            ],
            [
                results / "static_r2/RESULT.json",
                results / "runtime_r2/RESULT.json",
            ],
            OVERLAY / "SCHEDULER_PROFILE_COMPOSITE_USB_V2.json",
            "cpu-overflow",
        )
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(audit["all_variants_measured"])
        self.assertTrue(all(
            row["holdout_upper_violations"] == 0
            for row in audit["variants"]
        ))
        phone = next(
            row for row in profile["routes"]
            if row["route_id"] == "phone-adreno"
        )
        variant = phone["latency"]["variants"][0]
        self.assertEqual(
            {lease["duration_ucb_add_us"] for lease in phone["resource_leases"]},
            {variant["ucb_add_us"]},
        )
        self.assertTrue(all(
            lease["duration_us"] == variant["cost_us"]
            for lease in phone["resource_leases"]
        ))
        parsed = ProfileBundle.from_json(profile)
        self.assertEqual(parsed.profile_id, profile["profile_id"])

    def test_natural_route_fit_requires_disjoint_repeats(self) -> None:
        fitter = load_module(
            "fit_runtime_route_profile_disjoint_test",
            OVERLAY / "fit_runtime_route_profile.py",
        )
        result = (
            OVERLAY
            / "results/4060ti_op15_20260812/"
            "isolated_cpu_overflow_abba_v1/static_r1/RESULT.json"
        )
        with self.assertRaisesRegex(
            fitter.FitError, "disjoint calibration inputs"
        ):
            fitter.fit(
                [result],
                [result],
                OVERLAY / "SCHEDULER_PROFILE_COMPOSITE_USB_V2.json",
                "cpu-overflow",
            )

    def test_runtime_auto_profile_combines_qualified_evidence(self) -> None:
        builder = load_module(
            "build_runtime_auto_profile_test",
            OVERLAY / "build_runtime_auto_profile.py",
        )
        result_root = (
            OVERLAY / "results/4060ti_op15_20260812/"
            "isolated_cpu_overflow_abba_v1/calibration"
        )
        natural_root = (
            OVERLAY / "results/4060ti_op15_20260813/"
            "natural_runtime_v3_matched_v1"
        )
        profile, audit = builder.build(
            result_root / "PHASE_PROFILE_V1.json",
            result_root / "PHASE_PROFILE_AUDIT_V1.json",
            natural_root / "RUNTIME_PROFILE.json",
            natural_root / "RUNTIME_PROFILE_AUDIT.json",
        )
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(
            audit["supported_contention_classes"], list(range(1, 9))
        )
        cpu = next(
            row for row in profile["routes"]
            if row["route_id"] == "desktop-cpu"
        )
        self.assertEqual(
            {row["selector_value"] for row in cpu["latency"]["variants"]},
            set(range(1, 9)),
        )
        phone = next(
            row for row in profile["routes"]
            if row["route_id"] == "phone-adreno"
        )
        self.assertNotIn("kind", phone["latency"])
        self.assertEqual(phone["latency"]["ucb_add_us"], 3_575_788)
        self.assertEqual(
            {lease["duration_ucb_add_us"]
             for lease in phone["resource_leases"]},
            {3_575_788},
        )
        ProfileBundle.from_json(profile)

    def test_physical_qualifier_matches_server_task_receipts(self) -> None:
        qualifier = load_module(
            "qualify_fp16_small_overlay_task_test",
            OVERLAY / "qualify_fp16_small_overlay.py",
        )
        automated_row = {
            "route": "auto:whole:helper-c:residency:hot",
            "scheduler_final_binding": {
                "backend": "phone-adreno-ncm",
                "route_id": "auto:whole:helper-c:residency:hot",
            },
        }
        self.assertEqual(
            qualifier.endpoint_log_hash(
                automated_row, {"phone-adreno": "sha256:physical-log"}
            ),
            "sha256:physical-log",
        )
        self.assertTrue(qualifier.task_in_log(
            b"slot launch id_task = 42\n", 42
        ))
        self.assertTrue(qualifier.task_in_log(
            b"task id = 91 complete\n", 91
        ))
        self.assertTrue(qualifier.task_in_log(
            b"slot launch: id 0 | task 17 | processing\n", 17
        ))
        self.assertFalse(qualifier.task_in_log(
            b"slot launch id_task = 420\n", 42
        ))
        summary = qualifier.split_summary(
            b'S41SERVERFFN {"status":"ok","calls":1537}\n'
        )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["calls"], 1537)
        auto_base = {
            "fp16_resident_scheduler": {
                "arm": "op15",
                "arm_source": "runtime_placement",
                "planned_arm": "op15",
            },
            "request_results": [
                {"route": "qwen_cuda_cpu_op15_f16"},
                {"route": "gemma_cuda_cpu_op15_f16"},
                {"route": "local"},
            ],
        }
        self.assertEqual(
            qualifier.runtime_selected_large_nonbaseline_count(auto_base),
            2,
        )
        auto_base["fp16_resident_scheduler"]["arm_source"] = (
            "experimental_override"
        )
        self.assertEqual(
            qualifier.runtime_selected_large_nonbaseline_count(auto_base),
            0,
        )

    def test_physical_qualifier_writes_fail_receipt(self) -> None:
        qualifier = load_module(
            "qualify_fp16_small_overlay_failure_test",
            OVERLAY / "qualify_fp16_small_overlay.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "RESULT.json"
            release = root / "release.json"
            output = root / "QUALIFICATION.json"
            result.write_text("{}", encoding="ascii")
            release.write_text("{}", encoding="ascii")
            with mock.patch.object(sys, "argv", [
                "qualify_fp16_small_overlay.py",
                "--result", str(result),
                "--cpu-log", str(root / "cpu.log"),
                "--resident-release", str(release),
                "--output", str(output),
            ]):
                self.assertEqual(qualifier.main(), 2)
            value = json.loads(output.read_text(encoding="ascii"))
        self.assertEqual(value["status"], "FAIL")
        self.assertEqual(value["error"], "combined result")
        self.assertEqual(
            set(value["input_sha256"]), {"result", "resident_release"}
        )

    def test_abba_interval_requires_two_matched_pairs(self) -> None:
        abba = load_module(
            "compare_fp16_small_overlay_abba_test",
            OVERLAY / "compare_fp16_small_overlay_abba.py",
        )
        interval = abba.mean_ci95([10.0, 10.0])
        self.assertEqual(interval["ci95_low"], 10.0)
        self.assertEqual(interval["ci95_high"], 10.0)
        wide = abba.mean_ci95([5.0, 15.0])
        self.assertLess(wide["ci95_low"], 0)

    def test_2x2_report_keeps_large_policy_fixed_for_each_effect(self) -> None:
        reporter = load_module(
            "compare_fp16_small_overlay_2x2_test",
            OVERLAY / "compare_fp16_small_overlay_2x2.py",
        )

        def metrics(energy: float) -> dict[str, object]:
            return {
                "cpu_package_energy_j": energy / 2,
                "duration_s": energy / 100,
                "fleet_energy_j": energy,
                "gpu_board_energy_j": energy / 3,
                "phone_energy_j": energy / 20,
                "route_counts": {"desktop-cpu": 10},
                "server_energy_j": energy * 0.95,
                "slo_met": 10,
                "throughput_tokens_s": 4.0,
            }

        def policy(name: str, saving: float) -> dict[str, object]:
            pairs = []
            for _ in range(2):
                pairs.append({
                    "delta": {
                        "duration_saving_pct": saving,
                        "fleet_energy_saving_pct": saving,
                    },
                    "runtime_scheduler": metrics(90.0),
                    "static_cpu": metrics(100.0),
                })
            interval = {
                "ci95_high": saving,
                "ci95_low": saving,
                "mean": saving,
                "samples": 2,
            }
            return {
                "confidence_intervals": {
                    "duration_saving_pct": interval,
                    "fleet_energy_saving_pct": interval,
                },
                "large_model_policy": name,
                "pair_results": pairs,
                "schema": reporter.ABBA_SCHEMA,
                "status": "PASS",
                "work_identity": {"trace": "same"},
                "work_receipts": {"requests": 84},
            }

        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "cpu.json"
            second = Path(temporary) / "op15.json"
            first.write_text("cpu", encoding="ascii")
            second.write_text("op15", encoding="ascii")
            value = reporter.combine(
                policy("cpu-overflow", 5.0),
                policy("op15-assistance", 3.0),
                first,
                second,
            )
        self.assertEqual(len(value["cells"]), 4)
        self.assertEqual(
            value["interaction"]
                ["fleet_energy_saving_pct_cpu_minus_op15"]["mean"],
            2.0,
        )
        self.assertTrue(all(
            value["scheduler_energy_claim_eligible"].values()
        ))
        self.assertEqual(value["energy_qualification_status"], "PASS")
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "cpu.json"
            second = Path(temporary) / "op15.json"
            first.write_text("cpu", encoding="ascii")
            second.write_text("op15", encoding="ascii")
            failed = reporter.combine(
                policy("cpu-overflow", 5.0),
                policy("op15-assistance", -1.0),
                first,
                second,
            )
        self.assertEqual(failed["energy_qualification_status"], "FAIL")
        self.assertEqual(failed["status"], "FAIL")
        self.assertFalse(
            failed["scheduler_energy_claim_eligible"]["op15-assistance"]
        )

    def test_stable_work_identity_excludes_execution_receipt(self) -> None:
        comparator = load_module(
            "compare_fp16_small_overlay_identity_test",
            OVERLAY / "compare_fp16_small_overlay.py",
        )
        result = {
            "input_sha256": {
                "base_trace": "a" * 64,
                "manifest": "b" * 64,
                "overlay_trace": "c" * 64,
                "resident_release": "d" * 64,
                "runtime_profile": "e" * 64,
            },
            "model_identities": {"model": {"artifact_sha256": "f" * 64}},
            "scheduler_runtime": {
                "marginal_system_profile": {
                    "profile_id": "heldout-marginal-v1",
                    "sha256": "0" * 64,
                },
            },
        }
        identity = comparator.stable_work_identity(result)
        self.assertNotIn("resident_release", identity["input_sha256"])
        self.assertEqual(
            set(identity["input_sha256"]),
            {"base_trace", "manifest", "overlay_trace", "runtime_profile"},
        )
        self.assertEqual(
            identity["marginal_system_profile"],
            result["scheduler_runtime"]["marginal_system_profile"],
        )

    def test_stable_work_identity_excludes_scheduler_catalog(self) -> None:
        comparator = load_module(
            "compare_fp16_small_overlay_catalog_identity_test",
            OVERLAY / "compare_fp16_small_overlay.py",
        )
        result = {
            "input_sha256": {
                "automated_catalog": "0" * 64,
                "base_trace": "a" * 64,
                "manifest": "b" * 64,
                "overlay_trace": "c" * 64,
                "runtime_profile": "d" * 64,
            },
            "model_identities": {"model": {"artifact_sha256": "e" * 64}},
            "scheduler_runtime": {"marginal_system_profile": None},
        }
        identity = comparator.stable_work_identity(result)
        self.assertNotIn("automated_catalog", identity["input_sha256"])
        self.assertEqual(
            set(identity["input_sha256"]),
            {"base_trace", "manifest", "overlay_trace", "runtime_profile"},
        )

    def test_combined_residency_is_capacity_accounted(self) -> None:
        raw = json.loads(
            (OVERLAY / "OP15_COMBINED_RESIDENCY_PLAN_V1.json").read_text(
                encoding="ascii"
            )
        )
        plan = PhoneResidencyPlan.from_json(raw)
        llama = raw["combined_task_residency"]
        self.assertEqual(plan.minimum_available_bytes, 768 * 1024**2)
        self.assertLessEqual(
            plan.resident_bytes
            + llama["artifact_bytes"]
            + plan.minimum_available_bytes,
            plan.memory_capacity_bytes,
        )

    def test_runtime_placement_charges_small_model_phone_memory(self) -> None:
        placement = json.loads(
            (
                ROOT
                / "full_fp16_burstgpt_v1/results/FULL_FP16_BURSTGPT_ABBA_V2.json"
            ).read_text(encoding="ascii")
        )
        residency = PhoneResidencyPlan.from_json(json.loads(
            (OVERLAY / "OP15_COMBINED_RESIDENCY_PLAN_V1.json").read_text(
                encoding="ascii"
            )
        ))
        candidates = runtime_candidates(
            placement, residency, 770_928_288, 770_928_288
        )
        treatment = next(
            candidate for candidate in candidates
            if candidate.candidate_id == "fp16-server-gpu-cpu-op15-switch-v1"
        )
        resource = f"phone:{residency.phone_serial}:dram"
        self.assertEqual(
            treatment.additional_bytes[resource],
            residency.resident_bytes + 770_928_288,
        )
        self.assertEqual(
            treatment.runtime_bindings["additional_phone_resident_bytes"],
            770_928_288,
        )
        self.assertEqual(
            treatment.runtime_bindings["additional_host_resident_bytes"],
            770_928_288,
        )
        self.assertEqual(
            treatment.additional_bytes["desktop-host-ram"],
            19_734_474_752 + 770_928_288,
        )

    def test_runner_commits_runtime_cost_admissions(self) -> None:
        source = (OVERLAY / "run_fp16_small_overlay.py").read_text(
            encoding="ascii"
        )
        self.assertIn("submit_automated_request", source)
        self.assertIn("CanonicalPhysicalAdapter", source)
        self.assertIn("CanonicalHttpExecutionBackend", source)
        self.assertIn("RuntimeSnapshotBuilder", source)
        self.assertIn("RuntimeActivityTracker", source)
        self.assertNotIn("scheduler.wait_runtime_request", source)
        self.assertNotIn("scheduler.fail_runtime_request", source)
        self.assertNotIn("scheduler.fail_automated_request", source)
        self.assertNotIn("scheduler.complete_runtime_request", source)
        self.assertNotIn("scheduler.complete_automated_request", source)
        self.assertNotIn("scheduler.replan_runtime_request", source)
        self.assertNotIn("scheduler.replan_automated_request", source)
        self.assertNotIn("scheduler.cancel_runtime_request", source)
        self.assertIn("observe_runtime_phase", source)
        self.assertIn("RuntimePhaseObservation", source)
        self.assertNotIn("RuntimeExecutorObservation", source)
        self.assertNotIn("start_runtime_lease_renewal", source)
        self.assertIn("runtime_decision_log", source)
        self.assertNotIn("RuntimeExecutorBinding(", source)
        self.assertNotIn("PhaseResourceManager", source)
        self.assertIn("cuda_executor_ready", source)
        self.assertIn('"desktop-usb-root"', source)
        self.assertNotIn('"op15-ncm"', source)
        self.assertNotIn('"usb-token-rpc"', source)
        self.assertNotIn(
            '"skipped_external_usb_reservation"',
            source,
        )
        self.assertNotIn('"runtime_scheduler_recovery"', source)
        self.assertIn("result.recoveries", source)
        self.assertIn("BackgroundRuntimeMonitor", source)
        self.assertNotIn("RuntimeDispatchQueue", source)
        self.assertNotIn("schedule_cpu_recovery", source)
        self.assertNotIn("quarantined_routes", source)
        self.assertNotIn("binding_endpoint", source)
        self.assertNotIn("S42EndpointBackend", source)
        self.assertNotIn("run_scheduled_endpoint", source)
        self.assertIn("split_route_profiled", source)
        self.assertNotIn("args.forced_static_route == SPLIT_ROUTE", source)
        self.assertNotIn("lease_upper_bound_overrun", source)
        controller_source = (
            REPO_ROOT
            / "research_dev/scheduler/_internal/runtime_controller_ops/leases.py"
        ).read_text(encoding="ascii")
        self.assertIn("lease_upper_bound_overrun", controller_source)
        self.assertIn('"active_cpu_requests": (', source)
        self.assertIn('"cpu_utilization_pct": host[', source)
        self.assertIn('"memory_stall_avg10_basis_points":', source)
        self.assertNotIn("execution_condition.wait(timeout=0.5)", source)
        self.assertIn('"overhead_ns": overhead', source)
        arm = (OVERLAY / "run_fp16_small_overlay_arm.sh").read_text(
            encoding="ascii"
        )
        self.assertIn("large_model_policy=$2", arm)
        self.assertIn("small_model_policy=$3", arm)
        self.assertIn("overlay_variant=${5:-qualification}", arm)
        self.assertIn(
            "REQUESTS_LLAMA1B_PHASE_CALIBRATION_40.jsonl", arm
        )
        self.assertIn(
            "REQUESTS_LLAMA1B_NATURAL_VALIDATION_40.jsonl", arm
        )
        self.assertIn(
            "REQUESTS_LLAMA1B_IDLE_SPLIT_CALIBRATION_40.jsonl", arm
        )
        self.assertIn("phone_run_parent=${output%/*}", arm)
        self.assertIn(
            "phone_run_tag=${phone_run_parent##*/}-${output##*/}", arm
        )
        self.assertIn(
            "[[ $large_model_policy == op15-assistance ]] && large_arm=op15",
            arm,
        )
        self.assertIn(
            "[[ $large_model_policy == runtime-auto ]] && large_arm=auto",
            arm,
        )
        self.assertIn(
            '--large-model-policy "$effective_large_model_policy"', arm
        )
        self.assertIn('--small-model-policy "$small_model_policy"', arm)
        self.assertIn("--overlay-manifest", arm)
        self.assertIn("S42_USB_NCM=1", arm)
        self.assertIn("$restore_usb 7200 3", arm)
        self.assertIn("S42_MIN_AVAILABLE_KIB=$phone_reserve_kib", arm)
        self.assertIn(
            "S42_COMBINED_MIN_AVAILABLE_KIB=$phone_reserve_kib", arm
        )
        self.assertIn("--fp16-arm \"$large_arm\"", arm)
        self.assertIn("S42_MIN_PHONE_BATTERY_LEVEL", arm)
        self.assertIn("phone battery below safe run floor", arm)
        self.assertIn("$large_model_policy != runtime-auto", arm)
        self.assertIn("SCHEDULER_PROFILE_RUNTIME_AUTO_V1.json", arm)
        self.assertIn("recover_resident_usb", arm)
        self.assertIn("close_resident_bridge_session qwen", arm)
        self.assertIn("close_resident_bridge_session gemma", arm)
        self.assertIn(
            "phone_power_active=$phone_root/power.active", arm
        )
        self.assertIn(
            "$phone_logger $phone_samples $phone_power_active", arm
        )
        self.assertIn("touch $phone_power_active", arm)
        self.assertIn("wait_for_phone_power_done()", arm)
        self.assertIn("wait_for_phone_power_done", arm)
        self.assertIn("phone power logger did not finish", arm)
        self.assertIn(
            "rm -f $phone_active $phone_power_active", arm
        )
        self.assertNotIn(
            "$phone_logger $phone_samples $phone_active", arm
        )
        self.assertIn("interrupt_controller", arm)
        self.assertIn("phone_session_pid_file", arm)
        self.assertIn("kill \\$phone_session_pid", arm)
        self.assertIn("ffn_python=${S42_FFN_PYTHON:-python3}", arm)
        self.assertIn("S42_LLAMA_FFN_RESIDENT_LAYERS", arm)
        self.assertIn("profile_has_split=$(python3", arm)
        self.assertIn("ffn_route_required=0", arm)
        self.assertIn(
            "S42_LLAMA_FFN_MODEL=$phone_llama_ffn_model", arm
        )
        self.assertIn("if (( ffn_route_required )); then", arm)
        self.assertIn('hash_inputs+=("$ffn_manifest"', arm)
        self.assertIn("qualification_rc=$?", arm)
        self.assertIn("$qualification_status != FAIL", arm)
        self.assertIn('exit "$qualification_rc"', arm)
        self.assertIn('six_model_runner=$repo_root/', arm)
        self.assertIn('"$six_model_runner"', arm)
        self.assertIn("if [[ $base_rc -ne 0 ]]", arm)
        self.assertIn('--resident-release-file "$resident_release"', arm)
        self.assertIn('--resident-release "$resident_release"', arm)
        self.assertIn(
            '"cpu_executor_state": "deferred_until_paid_start"',
            source,
        )
        self.assertIn('"status": "OVERLAY_COMPLETE"', source)
        self.assertIn(
            '"final_observed_before_resident_release": True', source
        )
        self.assertNotIn(
            'or not (args.base_output / "RESULT.json").is_file()', source
        )
        self.assertLess(
            source.index("paid_start_ns = wait_for_trace_start"),
            source.index("cpu_process, cpu_load_ms, cpu_props ="),
        )
        factorial = (
            OVERLAY / "run_fp16_small_overlay_2x2_abba.sh"
        ).read_text(encoding="ascii")
        self.assertIn("run_cell cpu-static-r1", factorial)
        self.assertIn("run_cell cpu-runtime-r1", factorial)
        self.assertIn("run_cell op15-static-r1", factorial)
        self.assertIn("run_cell op15-runtime-r1", factorial)
        self.assertIn("FACTORIAL_2X2_ABBA.json", factorial)
        self.assertIn(
            'S42_MARGINAL_SYSTEM_PROFILE="$marginal_profile"',
            factorial,
        )
        self.assertIn("NEXT_MARGINAL_SYSTEM_PROFILE.json", factorial)
        self.assertIn("overlay_variant=natural-validation", factorial)
        split_qualification = (
            OVERLAY / "run_ffn_split_qualification_abba.sh"
        ).read_text(encoding="ascii")
        self.assertIn(
            "resident_layers=${S42_LLAMA_FFN_RESIDENT_LAYERS:-1}",
            split_qualification,
        )
        self.assertIn(
            'S42_LLAMA_FFN_RESIDENT_LAYERS="$resident_layers"',
            split_qualification,
        )
        self.assertIn(
            'S42_FFN_PHYSICAL_CALIBRATION="$physical_calibration"',
            split_qualification,
        )
        self.assertIn(
            'S42_MARGINAL_SYSTEM_PROFILE="$marginal_profile"',
            split_qualification,
        )
        self.assertIn(
            "natural_variant=natural-validation",
            split_qualification,
        )
        self.assertIn("op15-idle-r1", split_qualification)
        self.assertIn("--direct-energy-profile", split_qualification)
        self.assertIn(
            'S42_FFN_COMPILED_POLICY="$compiled_policy"',
            split_qualification,
        )
        pipeline = (
            OVERLAY / "run_full_scheduler_qualification_pipeline.sh"
        ).read_text(encoding="ascii")
        self.assertIn('audit.get("profile_sha256")', pipeline)
        self.assertIn(
            'heldout_result.get("energy_qualification_status") == "PASS"',
            pipeline,
        )
        self.assertIn(
            'qualification.get("status") == "PASS"', pipeline
        )
        self.assertIn("seed-2x2", pipeline)
        self.assertIn("heldout-2x2", pipeline)
        self.assertIn("NEXT_MARGINAL_SYSTEM_PROFILE.json", pipeline)
        self.assertIn("FFN_SPLIT_REJECTION.json", pipeline)
        self.assertIn("REJECTED_BY_DIRECT_ENERGY_GATE", pipeline)
        campaign = (
            OVERLAY / "run_full_scheduler_campaign.sh"
        ).read_text(encoding="ascii")
        self.assertIn("run_phase_contention_calibration_abba.sh", campaign)
        self.assertIn(
            "run_full_scheduler_qualification_pipeline.sh", campaign
        )
        self.assertIn('"$phase_root" -', campaign)
        self.assertIn("direct-energy-calibration", campaign)
        self.assertIn("LLAMA_FFN_DIRECT_ENERGY.json", campaign)
        self.assertIn("direct_rc=$?", campaign)
        self.assertIn("direct_status == FAIL", campaign)
        self.assertNotIn("direct_energy_profile=$5", campaign)

    def test_phone_energy_analyzer_accepts_combined_result_v2(self) -> None:
        analyzer = ROOT.parent / (
            "s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/"
            "burstgpt_gpu_cpu_op15_v1/analyze_phone_energy.py"
        )
        source = analyzer.read_text(encoding="ascii")
        self.assertIn(
            '"s42-full-fp16-llama1b-combined-result-v2"', source
        )

    def test_comparison_rejects_a_large_model_policy_mismatch(self) -> None:
        comparer = load_module(
            "compare_fp16_small_overlay_policy_test",
            OVERLAY / "compare_fp16_small_overlay.py",
        )
        static = {
            "base": {"arm": "control"},
            "input_sha256": {"trace": "same"},
            "model_identities": {"model": "same"},
            "policy": {"large_model_policy": "cpu-overflow"},
            "work_receipts": {"outputs": "same"},
        }
        runtime = {
            **static,
            "base": {"arm": "op15"},
            "policy": {"large_model_policy": "op15-assistance"},
        }
        phone = {"boundary": "paid_trace_interval"}
        with mock.patch.object(
            comparer,
            "validate",
            side_effect=((static, phone), (runtime, phone)),
        ):
            with self.assertRaisesRegex(
                comparer.ComparisonError, "large-model policy differs"
            ):
                comparer.compare(
                    Path("static-result"),
                    Path("static-phone"),
                    Path("runtime-result"),
                    Path("runtime-phone"),
                )

    def test_comparison_accepts_current_physical_gate_schema(self) -> None:
        comparer = load_module(
            "compare_fp16_small_overlay_gate_schema_test",
            OVERLAY / "compare_fp16_small_overlay.py",
        )
        policy = {
            "large_model_policy": "cpu-overflow",
            "small_model_policy": "static-cpu",
        }
        result = {"policy": policy, "scheduler_runtime": {}}
        gates = {
            "all_endpoint_tasks_in_selected_server_log": True,
            "all_leases_cover_physical_execution": True,
            "all_original_latency_upper_bounds_met": True,
            "all_runtime_snapshots_live": True,
            "decision_matches_physical_endpoint": True,
            "large_request_receipts_causal": True,
            "nonbaseline_physical_execution": False,
            "overlay_request_receipts_causal": True,
            "phase_transport_leases_released": True,
            "resident_release_after_final_phone_snapshot": True,
            "selected_split_route_has_phone_ffn_calls": True,
            "trace_model_energy_identity": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "RESULT.json"
            result_path.write_text("{}\n", encoding="ascii")
            qualification_path = root / "QUALIFICATION.json"
            qualification_path.write_text(json.dumps({
                "gates": gates,
                "input_sha256": {
                    "result": hashlib.sha256(
                        result_path.read_bytes()
                    ).hexdigest(),
                },
                "nonbaseline_physical_executions": 0,
                "policy": policy,
                "schema": comparer.QUALIFICATION_SCHEMA,
                "status": "PASS",
            }), encoding="ascii")
            qualification = comparer.validate_qualification(
                qualification_path,
                result_path,
                result,
                False,
            )
        self.assertEqual(qualification["gates"], gates)

    def test_comparison_separates_output_shape_from_token_content(self) -> None:
        comparer = load_module(
            "compare_fp16_small_overlay_receipt_test",
            OVERLAY / "compare_fp16_small_overlay.py",
        )
        static = {
            "work_receipts": {
                "large_model_outputs": {
                    "actual_output_tokens": 11_605,
                    "output_tokens": 11_605,
                    "request_count": 74,
                    "shape_sha256": "a" * 64,
                    "token_sha256": "b" * 64,
                },
                "small_model_outputs": {
                    "actual_output_tokens": 1_527,
                    "output_tokens": 1_527,
                    "request_count": 10,
                    "shape_sha256": "c" * 64,
                    "token_sha256": "d" * 64,
                },
            },
        }
        runtime = json.loads(json.dumps(static))
        runtime["work_receipts"]["large_model_outputs"][
            "token_sha256"
        ] = "e" * 64
        runtime["work_receipts"]["small_model_outputs"][
            "token_sha256"
        ] = "f" * 64

        self.assertEqual(
            comparer.stable_work_receipts(static),
            comparer.stable_work_receipts(runtime),
        )
        self.assertEqual(
            comparer.output_shape_receipts(static),
            comparer.output_shape_receipts(runtime),
        )
        runtime["work_receipts"]["small_model_outputs"][
            "shape_sha256"
        ] = "0" * 64
        self.assertEqual(
            comparer.stable_work_receipts(static),
            comparer.stable_work_receipts(runtime),
        )
        self.assertNotEqual(
            comparer.output_shape_receipts(static),
            comparer.output_shape_receipts(runtime),
        )

    def test_output_receipt_requires_every_requested_token(self) -> None:
        runner = load_module(
            "run_fp16_small_overlay_output_receipt_test",
            OVERLAY / "run_fp16_small_overlay.py",
        )
        row = {
            "event_id": "shape:0",
            "input_tokens": 2,
            "output_tokens": 2,
            "overlay_request_index": 0,
            "tokens": [1, 2],
        }
        receipt = runner.output_token_receipt(
            [row], "overlay_request_index"
        )
        self.assertEqual(receipt["actual_output_tokens"], 2)
        self.assertEqual(len(receipt["shape_sha256"]), 64)

        with self.assertRaisesRegex(
            runner.OverlayError, "exact output shape receipt"
        ):
            runner.output_token_receipt(
                [{**row, "tokens": [1]}], "overlay_request_index"
            )

    def test_physical_all_request_audit_is_complete(self) -> None:
        builder = load_module(
            "build_all_request_scheduling_log_test",
            OVERLAY / "build_all_request_scheduling_log.py",
        )
        run_dir = (
            OVERLAY
            / "results/4060ti_op15_20260812/runtime_scheduler"
        )
        records, summary = builder.build(run_dir)

        self.assertEqual(len(records), 84)
        self.assertEqual(
            [row["combined_request_index"] for row in records],
            list(range(84)),
        )
        self.assertEqual(
            summary["audit"]["scope_counts"],
            {
                "per_request_runtime_cost_and_lease": 10,
                "runtime_placement_inherited": 74,
            },
        )
        self.assertEqual(
            summary["audit"]["per_request_runtime_scheduler_count"], 10
        )
        self.assertTrue(all(
            row["runtime_scheduling"]["scheduler_invoked_at_arrival"]
            for row in records
            if row["stream"] == "llama1b-overlay"
        ))
        self.assertTrue(all(
            not row["runtime_scheduling"]["scheduler_invoked_at_arrival"]
            for row in records
            if row["stream"] == "fp16-burstgpt"
        ))

    def test_all_request_audit_accepts_launcher_layout(self) -> None:
        builder = load_module(
            "build_all_request_launcher_layout_test",
            OVERLAY / "build_all_request_scheduling_log.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                "base_events": root / "base/events.jsonl",
                "base_result": root / "base/RESULT.json",
                "combined_result": root / "combined/RESULT.json",
                "execution_plan": root / "capture/EXECUTION_PLAN.json",
                "overlay_events": root / "combined/events.jsonl",
            }
            for path in expected.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(builder.artifact_paths(root), expected)

    def test_all_request_audit_groups_runtime_replans(self) -> None:
        builder = load_module(
            "build_all_request_replan_test",
            OVERLAY / "build_all_request_scheduling_log.py",
        )
        events = [
            {
                "attempt": {"attempt_kind": "arrival"},
                "kind": "runtime_scheduler_decision",
                "overlay_request_index": 3,
            },
            {
                "attempt": {
                    "attempt_kind": "lease-renewal-replan-1",
                },
                "kind": "runtime_scheduler_decision",
                "overlay_request_index": 3,
            },
        ]
        grouped = builder.grouped_by(
            events,
            "runtime_scheduler_decision",
            "overlay_request_index",
        )
        self.assertEqual(len(grouped[3]), 2)
        self.assertEqual(
            builder.decision_attempt(grouped[3][-1])["attempt_kind"],
            "lease-renewal-replan-1",
        )


if __name__ == "__main__":
    unittest.main()
