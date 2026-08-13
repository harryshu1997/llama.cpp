#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import unittest

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    EnergyProfile,
    LatencyProfile,
    MarginalSystemCostContext,
    ProfileBundle,
    Request,
    RuntimeExecutorBinding,
    RuntimeMemoryDemand,
    RuntimeModelArtifact,
    RuntimePlacementSnapshot,
    UnifiedScheduleError,
    UnifiedScheduler,
)


MODEL_HASH = "sha256:" + "a" * 64
MODEL_BYTES = 800_000_000
WORKLOAD = "small-model-task"


def profile() -> ProfileBundle:
    def route(
        route_id: str,
        resource_slots: dict[str, int],
        *,
        baseline: bool,
        latency_input: int,
        latency_output: int,
        energy_input: int,
        energy_output: int,
    ) -> dict[str, object]:
        return {
            "baseline": baseline,
            "energy": {
                "boundary_id": "cpu-package+gpu-board+whole-phone",
                "cost_uj": {
                    "fixed": 0,
                    "input_token": energy_input,
                    "kind": "affine_tokens_v1",
                    "output_token": energy_output,
                },
                "lower_error_ppm": 0,
                "status": "measured",
                "upper_error_ppm": 0,
            },
            "evidence_ids": ["qualified-route-r1"],
            "granularity": "task",
            "latency": {
                "cost_us": {
                    "fixed": 0,
                    "input_token": latency_input,
                    "kind": "affine_tokens_v1",
                    "output_token": latency_output,
                },
                "measured": True,
                "sample_count": 3,
                "ucb_add_us": 100,
            },
            "overlap": {"status": "not_applicable"},
            "placement_verified": True,
            "quality_class": "bounded_numeric",
            "resident": True,
            "resource_slots": resource_slots,
            "route_id": route_id,
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": MODEL_BYTES,
            "workload_id": WORKLOAD,
        }

    return ProfileBundle.from_json({
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 20_000_000,
        },
        "profile_id": "runtime-cost-test",
        "resources": [
            {
                "capacity": 1,
                "identity": "cpu",
                "kind": "cpu",
                "ready": True,
                "resource_id": "desktop-cpu",
            },
            {
                "capacity": 1,
                "identity": "gpu",
                "kind": "gpu",
                "ready": True,
                "resource_id": "cuda0",
            },
            {
                "capacity": 1,
                "identity": "phone",
                "kind": "phone-gpu",
                "ready": True,
                "resource_id": "op15-adreno",
            },
            {
                "capacity": 1,
                "identity": "usb",
                "kind": "transport",
                "ready": True,
                "resource_id": "usb-token-rpc",
            },
        ],
        "routes": [
            route(
                "desktop-cpu",
                {"desktop-cpu": 1},
                baseline=True,
                latency_input=10,
                latency_output=100,
                energy_input=100,
                energy_output=1_000,
            ),
            route(
                "desktop-cuda",
                {"cuda0": 1},
                baseline=False,
                latency_input=1,
                latency_output=5,
                energy_input=5,
                energy_output=50,
            ),
            route(
                "phone-adreno",
                {"op15-adreno": 1, "usb-token-rpc": 1},
                baseline=False,
                latency_input=2,
                latency_output=20,
                energy_input=10,
                energy_output=100,
            ),
        ],
        "schema": "s42-general-scheduler-profile-v1",
        "trace_workload_map": {"small-model": WORKLOAD},
    })


def request(request_id: str = "r0") -> Request:
    return Request(
        request_id=request_id,
        workload_id=WORKLOAD,
        arrival_us=1_000,
        deadline_us=1_000_000,
        input_tokens=100,
        output_tokens=20,
        quality_requirement="bounded_numeric",
    )


def model() -> RuntimeModelArtifact:
    return RuntimeModelArtifact("small-model", MODEL_HASH, MODEL_BYTES)


def snapshot() -> RuntimePlacementSnapshot:
    return RuntimePlacementSnapshot(
        snapshot_id="live-memory-1",
        captured_at_us=900,
        valid_until_us=2_000,
        capacities={
            "host-ram": DeviceMemoryCapacity(
                "host-ram", 32_000_000_000, 10_000_000_000, 2_000_000_000
            ),
            "cuda0-vram": DeviceMemoryCapacity(
                "cuda0-vram", 16_000_000_000, 15_000_000_000, 500_000_000
            ),
            "op15-ram": DeviceMemoryCapacity(
                "op15-ram", 12_000_000_000, 5_000_000_000, 2_000_000_000
            ),
        },
    )


def binding(
    route_id: str,
    resources: tuple[str, ...],
    memory: str,
    *,
    ready: bool = True,
    artifact_hash: str = MODEL_HASH,
) -> RuntimeExecutorBinding:
    return RuntimeExecutorBinding(
        executor_id="executor-" + route_id,
        route_id=route_id,
        model_id="small-model",
        artifact_sha256=artifact_hash,
        artifact_bytes=MODEL_BYTES,
        backend=route_id,
        resource_ids=resources,
        memory_resource_id=memory,
        resident=True,
        ready=ready,
    )


def cpu_binding(*, ready: bool = True) -> RuntimeExecutorBinding:
    return binding(
        "desktop-cpu", ("desktop-cpu",), "host-ram", ready=ready
    )


def phone_binding(
    *, ready: bool = True, artifact_hash: str = MODEL_HASH
) -> RuntimeExecutorBinding:
    return binding(
        "phone-adreno",
        ("op15-adreno", "usb-token-rpc"),
        "op15-ram",
        ready=ready,
        artifact_hash=artifact_hash,
    )


def contention_profile() -> ProfileBundle:
    base = profile()
    latency = LatencyProfile.from_json({
        "kind": "conditioned_affine_features_v1",
        "selector_feature": "large_phase_id",
        "variants": [
            {
                "cost_us": {
                    "coefficients": {
                        "active_cpu_requests": 500,
                        "input_tokens": 10,
                        "output_tokens": 100,
                    },
                    "fixed": 0,
                    "kind": "affine_features_v1",
                },
                "label": "idle-desktop",
                "measured": True,
                "sample_count": 4,
                "selector_value": 0,
                "ucb_add_us": 100,
            },
            {
                "cost_us": {
                    "coefficients": {
                        "active_cpu_requests": 5_000,
                        "input_tokens": 100,
                        "output_tokens": 1_000,
                    },
                    "fixed": 0,
                    "kind": "affine_features_v1",
                },
                "label": "qwen-phase",
                "measured": True,
                "sample_count": 5,
                "selector_value": 1,
                "ucb_add_us": 1_000,
            },
        ],
    })
    return replace(base, routes=tuple(
        replace(route, latency=latency)
        if route.route_id == "desktop-cpu" else route
        for route in base.routes
    ))


def marginal_profile() -> ProfileBundle:
    base = profile()
    phone_energy = EnergyProfile.from_json({
        "boundary_id": "cpu-package+gpu-board+whole-phone",
        "cost_uj": {
            "fixed": 0,
            "input_token": 500,
            "kind": "affine_tokens_v1",
            "output_token": 2_500,
        },
        "lower_error_ppm": 0,
        "status": "measured",
        "upper_error_ppm": 0,
    })
    return replace(base, routes=tuple(
        replace(route, energy=phone_energy)
        if route.route_id == "phone-adreno"
        else route
        for route in base.routes
    ))


class RuntimeCostTests(unittest.TestCase):
    def test_conditioned_gate_uses_selected_variant_qualification(self) -> None:
        base = contention_profile()
        cpu = next(
            route for route in base.routes
            if route.route_id == "desktop-cpu"
        )
        variants = dict(cpu.latency.variants)
        variants[1] = replace(variants[1], measured=False)
        latency = replace(
            cpu.latency,
            measured=False,
            variants=variants,
        )
        qualified = replace(base, routes=tuple(
            replace(route, latency=latency)
            if route.route_id == "desktop-cpu" else route
            for route in base.routes
        ))
        scheduler = UnifiedScheduler((qualified,), "enforce")
        idle_request = replace(request(), features={
            "active_cpu_requests": 0,
            "large_phase_id": 0,
        })
        estimates = scheduler.estimate_runtime_costs(
            idle_request,
            model(),
            (cpu_binding(), phone_binding(ready=False)),
            snapshot=snapshot(),
            now_us=1_000,
        )
        decision = scheduler.schedule_runtime_costs(
            idle_request, estimates, runtime_now_us=1_000
        )
        self.assertEqual(decision.route_id, "desktop-cpu")

    def test_runtime_cost_selects_the_live_contention_variant(self) -> None:
        scheduler = UnifiedScheduler((contention_profile(),), "enforce")
        live_request = replace(request(), features={
            "active_cpu_requests": 2,
            "large_phase_id": 1,
        })
        estimates = scheduler.estimate_runtime_costs(
            live_request,
            model(),
            (cpu_binding(), phone_binding()),
            snapshot=snapshot(),
            now_us=1_000,
        )
        cpu = next(
            item for item in estimates.estimates
            if item.route_id == "desktop-cpu"
        )
        self.assertEqual(cpu.service_us, 40_000)
        self.assertEqual(cpu.service_upper_us, 41_000)
        self.assertEqual(cpu.latency_profile_label, "qwen-phase")
        self.assertEqual(cpu.latency_sample_count, 5)
        self.assertTrue(cpu.latency_measured)
        self.assertEqual(
            estimates.to_json()["schema"],
            "research-scheduler-runtime-cost-v3",
        )

    def test_multi_resource_memory_demands_are_all_accounted(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        phone = RuntimeExecutorBinding(
            executor_id="executor-phone-multi-memory",
            route_id="phone-adreno",
            model_id="small-model",
            artifact_sha256=MODEL_HASH,
            artifact_bytes=MODEL_BYTES,
            backend="phone-adreno",
            resource_ids=("op15-adreno", "usb-token-rpc"),
            memory_resource_id=None,
            resident=True,
            ready=True,
            memory_demands=(
                RuntimeMemoryDemand(
                    "host-weights", "host-ram", "model_weights",
                    MODEL_BYTES, MODEL_BYTES, "resident",
                ),
                RuntimeMemoryDemand(
                    "phone-slice", "op15-ram", "weight_slice",
                    400_000_000, 400_000_000, "resident",
                ),
                RuntimeMemoryDemand(
                    "kv-cache", "host-ram", "kv_cache",
                    100_000_000, 0, "request",
                ),
                RuntimeMemoryDemand(
                    "phone-workspace", "op15-ram", "workspace",
                    200_000_000, 0, "request",
                ),
            ),
        )

        estimates = scheduler.estimate_runtime_costs(
            request(), model(), (cpu_binding(), phone),
            snapshot=snapshot(), now_us=1_000,
        )
        estimate = next(
            row for row in estimates.estimates
            if row.route_id == "phone-adreno"
        )

        self.assertTrue(estimate.admitted)
        self.assertEqual(estimate.additional_bytes, 300_000_000)
        self.assertEqual(
            dict(estimate.additional_bytes_by_resource),
            {"host-ram": 100_000_000, "op15-ram": 200_000_000},
        )
        self.assertEqual(
            len(estimate.to_json()["memory_demands"]), 4
        )

    def test_unknown_live_contention_variant_fails_closed(self) -> None:
        scheduler = UnifiedScheduler((contention_profile(),), "enforce")
        with self.assertRaisesRegex(
            UnifiedScheduleError, "unknown latency variant"
        ):
            scheduler.estimate_runtime_costs(
                replace(request(), features={
                    "active_cpu_requests": 0,
                    "large_phase_id": 9,
                }),
                model(),
                (cpu_binding(), phone_binding()),
                snapshot=snapshot(),
                now_us=1_000,
            )

    def test_estimates_shape_from_live_resident_bindings(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        estimates = scheduler.estimate_runtime_costs(
            request(),
            model(),
            (cpu_binding(), phone_binding()),
            snapshot=snapshot(),
            now_us=1_000,
        )
        routes = {item.route_id: item for item in estimates.estimates}
        self.assertTrue(routes["desktop-cpu"].admitted)
        self.assertTrue(routes["phone-adreno"].admitted)
        self.assertEqual(routes["desktop-cuda"].reason, "EXECUTOR_ABSENT")
        self.assertEqual(routes["phone-adreno"].service_us, 600)
        self.assertEqual(routes["phone-adreno"].fleet_energy_uj, 3_000)
        self.assertEqual(
            estimates.to_json()["model"]["artifact_sha256"], MODEL_HASH
        )

    def test_scheduler_selects_live_phone_candidate(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.set_resource_ready("cuda0", False, 1_000)
        estimates, decision = scheduler.schedule_runtime(
            request(),
            model(),
            (cpu_binding(), phone_binding()),
            snapshot=snapshot(),
            now_us=1_000,
        )
        self.assertEqual(estimates.request_id, "r0")
        self.assertEqual(decision.route_id, "phone-adreno")
        self.assertEqual(decision.reason, "VERIFIED_ENERGY_SAVING")

    def test_missing_phone_falls_back_to_cpu(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.set_resource_ready("cuda0", False, 1_000)
        scheduler.set_resource_ready("op15-adreno", False, 1_000)
        scheduler.set_resource_ready("usb-token-rpc", False, 1_000)
        estimates = scheduler.estimate_runtime_costs(
            request(),
            model(),
            (cpu_binding(), phone_binding(ready=False)),
            snapshot=snapshot(),
            now_us=1_000,
        )
        phone = next(
            item for item in estimates.estimates
            if item.route_id == "phone-adreno"
        )
        self.assertEqual(phone.reason, "EXECUTOR_NOT_READY")
        decision = scheduler.schedule(request(), runtime_now_us=1_000)
        self.assertEqual(decision.route_id, "desktop-cpu")
        self.assertEqual(decision.reason, "FAIL_CLOSED_BASELINE")

    def test_cancelled_phone_owner_can_be_replanned_to_cpu(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.set_resource_ready("cuda0", False, 1_000)
        _, phone = scheduler.schedule_runtime(
            request(),
            model(),
            (cpu_binding(), phone_binding()),
            snapshot=snapshot(),
            now_us=1_000,
        )
        self.assertEqual(phone.route_id, "phone-adreno")
        cancelled = scheduler.cancel(phone.request_id, 1_100)
        self.assertEqual(set(cancelled), {lease.token for lease in phone.leases})

        estimates = scheduler.estimate_runtime_costs(
            request(),
            model(),
            (cpu_binding(), phone_binding(ready=False)),
            snapshot=snapshot(),
            now_us=1_100,
        )
        fallback = scheduler.schedule_runtime_costs(
            request(), estimates, runtime_now_us=1_100
        )
        self.assertEqual(fallback.route_id, "desktop-cpu")
        self.assertEqual(fallback.reason, "FAIL_CLOSED_BASELINE")

    def test_external_usb_owner_blocks_phone_until_after_deadline(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.set_resource_ready("cuda0", False, 1_000)
        leases = scheduler.reserve_external_resource(
            "usb-token-rpc", "large-model-functionfs", 0, 2_000_000
        )
        self.assertEqual(len(leases), 1)
        scheduler.set_resource_ready("op15-adreno", False, 1_000)
        scheduler.set_resource_ready("op15-adreno", True, 1_001)

        _, decision = scheduler.schedule_runtime(
            request(),
            model(),
            (cpu_binding(), phone_binding()),
            snapshot=snapshot(),
            now_us=1_000,
        )
        self.assertEqual(decision.route_id, "desktop-cpu")
        self.assertIn(("phone-adreno", "SLO_INFEASIBLE"), decision.rejected)

    def test_wrong_phone_model_hash_is_not_a_candidate(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        estimates = scheduler.estimate_runtime_costs(
            request(),
            model(),
            (cpu_binding(), phone_binding(artifact_hash="b" * 64)),
            snapshot=snapshot(),
            now_us=1_000,
        )
        phone = next(
            item for item in estimates.estimates
            if item.route_id == "phone-adreno"
        )
        self.assertFalse(phone.admitted)
        self.assertEqual(phone.reason, "MODEL_HASH_MISMATCH")

        decision = scheduler.schedule_runtime_costs(
            request(), estimates, runtime_now_us=1_000
        )
        self.assertEqual(decision.route_id, "desktop-cpu")
        self.assertIn(
            ("phone-adreno", "MODEL_HASH_MISMATCH"), decision.rejected
        )

    def test_runtime_estimate_request_mismatch_is_rejected(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        estimates = scheduler.estimate_runtime_costs(
            request(),
            model(),
            (cpu_binding(), phone_binding()),
            snapshot=snapshot(),
            now_us=1_000,
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "does not match the request"
        ):
            scheduler.schedule_runtime_costs(
                request("r1"), estimates, runtime_now_us=1_000
            )

    def test_missing_baseline_binding_fails_closed(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        with self.assertRaisesRegex(
            UnifiedScheduleError, "baseline fallback.*EXECUTOR_ABSENT"
        ):
            scheduler.estimate_runtime_costs(
                request(),
                model(),
                (phone_binding(),),
                snapshot=snapshot(),
                now_us=1_000,
            )

    def test_marginal_system_cost_can_avoid_cpu_interference(self) -> None:
        without = UnifiedScheduler((marginal_profile(),), "enforce")
        without.set_resource_ready("cuda0", False, 1_000)
        _, direct_decision = without.schedule_runtime(
            request(),
            model(),
            (cpu_binding(), phone_binding()),
            snapshot=snapshot(),
            now_us=1_000,
        )
        self.assertEqual(direct_decision.route_id, "desktop-cpu")

        context = MarginalSystemCostContext(
            context_id="qwen-cpu-overflow-r1",
            critical_path_end_us=1_000_000,
            phase_power_mw=100_000,
            gpu_idle_power_mw=20_000,
            causal_tail_power_mw=30_000,
            route_cpu_interference_ppm={
                "desktop-cpu": 500_000,
                "desktop-cuda": 0,
                "phone-adreno": 0,
            },
            lower_error_ppm=0,
            upper_error_ppm=0,
            sample_count=8,
            measured=True,
        )
        with_marginal = UnifiedScheduler((marginal_profile(),), "enforce")
        with_marginal.set_resource_ready("cuda0", False, 1_000)
        _, system_decision = with_marginal.schedule_runtime(
            request(),
            model(),
            (cpu_binding(), phone_binding()),
            snapshot=snapshot(),
            now_us=1_000,
            marginal_system_context=context,
        )

        self.assertEqual(system_decision.route_id, "phone-adreno")
        self.assertEqual(
            system_decision.system_finish_upper_us, 1_000_000
        )
        self.assertEqual(
            system_decision.energy_breakdown["kind"],
            "marginal_system_v1",
        )
        self.assertEqual(
            system_decision.marginal_system_cost["context_id"],
            "qwen-cpu-overflow-r1",
        )

    def test_marginal_tail_excludes_phase_interference_time(self) -> None:
        context = MarginalSystemCostContext(
            context_id="qwen-tail-accounting",
            critical_path_end_us=1_000,
            phase_power_mw=100_000,
            gpu_idle_power_mw=20_000,
            causal_tail_power_mw=30_000,
            route_cpu_interference_ppm={"desktop-cpu": 500_000},
            lower_error_ppm=0,
            upper_error_ppm=0,
            sample_count=8,
            measured=True,
        )
        value = context.route_cost(
            "desktop-cpu",
            service_us=100,
            service_upper_us=100,
            finish_us=1_100,
            finish_upper_us=1_100,
        )
        self.assertEqual(value["interference_us"], 50)
        self.assertEqual(value["critical_path_extension_us"], 100)
        self.assertEqual(value["causal_tail_us"], 50)
        self.assertEqual(value["phase_interference_uj"], 5_000)
        self.assertEqual(value["causal_tail_uj"], 1_500)
        self.assertEqual(value["gpu_idle_uj"], 1_000)
        self.assertEqual(value["total_uj"], 7_500)
        self.assertEqual(value["lower_uj"], 7_500)
        self.assertEqual(value["upper_uj"], 7_500)


if __name__ == "__main__":
    unittest.main()
