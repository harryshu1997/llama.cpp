#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import research_dev.scheduler as scheduler_api
from research_dev.scheduler import (
    LifecycleProfileSet,
    LifecycleReceipt,
    MatmulOp,
    MatmulSystemProfile,
    ModelProgram,
    ProfileBundle,
    Request,
    UnifiedScheduleError,
    UnifiedScheduler,
)


def route(
    route_id: str,
    workload_id: str,
    resource_id: str,
    *,
    baseline: bool,
    latency_us: int,
    energy_uj: int | None,
) -> dict[str, object]:
    energy = {
        "status": "unknown",
        "cost_uj": None,
        "lower_error_ppm": 0,
        "upper_error_ppm": 0,
    }
    if energy_uj is not None:
        energy = {
            "status": "measured",
            "cost_uj": {
                "fixed": energy_uj,
                "input_token": 0,
                "kind": "affine_tokens_v1",
                "output_token": 0,
            },
            "boundary_id": "test-fleet",
            "lower_error_ppm": 0,
            "upper_error_ppm": 0,
        }
    return {
        "route_id": route_id,
        "workload_id": workload_id,
        "granularity": "task",
        "baseline": baseline,
        "resource_slots": {resource_id: 1},
        "latency": {
            "cost_us": {
                "fixed": latency_us,
                "input_token": 0,
                "kind": "affine_tokens_v1",
                "output_token": 0,
            },
            "ucb_add_us": 0,
            "sample_count": 3,
            "measured": True,
        },
        "energy": energy,
        "overlap": {"status": "not_applicable"},
        "quality_class": "exact",
        "placement_verified": True,
        "resident": True,
        "server_busy_ppm": 1_000_000,
        "server_memory_bytes": 1,
        "evidence_ids": ["test-control-plane"],
    }


def profile(
    profile_id: str,
    workload_id: str,
    routes: list[dict[str, object]],
    resources: list[dict[str, object]] | None = None,
    policy_overrides: dict[str, object] | None = None,
) -> ProfileBundle:
    policy = {
        "energy_saving_ppm": 50_000,
        "latency_limit_ppm": 2_000_000,
    }
    policy.update(policy_overrides or {})
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": profile_id,
        "resources": resources or [
            {
                "resource_id": "gpu",
                "kind": "gpu",
                "capacity": 1,
                "ready": True,
                "identity": "test-gpu",
            },
            {
                "resource_id": "phone",
                "kind": "phone",
                "capacity": 1,
                "ready": True,
                "identity": "test-phone",
            },
        ],
        "routes": routes,
        "trace_workload_map": {workload_id: workload_id},
        "policy": policy,
    })


def request(
    request_id: str,
    workload_id: str,
    arrival_us: int = 0,
    features: dict[str, int] | None = None,
) -> Request:
    return Request(
        request_id=request_id,
        workload_id=workload_id,
        arrival_us=arrival_us,
        deadline_us=arrival_us + 10_000,
        input_tokens=1,
        output_tokens=1,
        quality_requirement="exact",
        features=features or {},
    )


def matmul_profile() -> MatmulSystemProfile:
    return MatmulSystemProfile.from_json({
        "schema": "s42-matmul-vq-profile-v1",
        "profile_id": "test-matmul",
        "energy_boundary_id": "test-fleet",
        "domains": [
            {"domain_id": "cpu-package", "idle_power_mw": 1_000},
        ],
        "devices": [
            {
                "device_id": "cpu",
                "kind": "desktop_cpu",
                "resource_id": "compute:cpu",
                "memory_capacity_bytes": 0,
                "reserved_bytes": 0,
                "ready": True,
            },
            {
                "device_id": "gpu",
                "kind": "desktop_gpu",
                "resource_id": "compute:gpu",
                "memory_capacity_bytes": 1024,
                "reserved_bytes": 0,
                "ready": True,
            },
            {
                "device_id": "phone",
                "kind": "phone_accelerator",
                "resource_id": "compute:phone",
                "memory_capacity_bytes": 1024,
                "reserved_bytes": 0,
                "ready": True,
            },
        ],
        "kernels": [
            {
                "profile_id": "cpu-mm",
                "device_id": "cpu",
                "kernel_family": "dense",
                "quantization": "q4_0",
                "shape": {"m": 1, "k": 4, "n": 4},
                "effective_ops_per_s": 1_000_000,
                "effective_bytes_per_s": 1_000_000,
                "launch_us": 0,
                "domain_power_mw": {"cpu-package": 2_000},
                "minimum_n": 1,
                "maximum_n": 0,
                "n_quantum": 1,
                "status": "measured",
                "evidence_ids": ["test-cpu-mm"],
            },
        ],
        "links": [],
        "policy": {
            "host_device_id": "cpu",
            "final_device_id": "cpu",
            "host_domain_id": "cpu-package",
            "host_active_power_mw": 2_000,
            "latency_limit_ppm": 1_000_000,
            "split_search_points": 2,
            "queue_limit": 8,
            "require_measured": True,
        },
    })


class UnifiedSchedulerTests(unittest.TestCase):
    def test_public_api_has_one_scheduler_class(self) -> None:
        scheduler_classes = sorted(
            name
            for name in scheduler_api.__all__
            if name.endswith("Scheduler")
            and inspect.isclass(getattr(scheduler_api, name))
        )
        self.assertEqual(scheduler_classes, ["UnifiedScheduler"])

    def test_profiles_share_one_resource_timeline(self) -> None:
        first = profile(
            "first-profile",
            "first",
            [route(
                "first-gpu",
                "first",
                "gpu",
                baseline=True,
                latency_us=100,
                energy_uj=None,
            )],
        )
        second = profile(
            "second-profile",
            "second",
            [route(
                "second-gpu",
                "second",
                "gpu",
                baseline=True,
                latency_us=50,
                energy_uj=None,
            )],
        )
        scheduler = UnifiedScheduler((first, second), "control")
        first_decision = scheduler.schedule(request("r0", "first"))
        second_decision = scheduler.schedule(request("r1", "second"))
        self.assertEqual(first_decision.start_us, 0)
        self.assertEqual(first_decision.finish_us, 100)
        self.assertEqual(second_decision.start_us, 100)
        self.assertEqual(second_decision.finish_us, 150)

    def test_gpu_switch_window_uses_cpu_phone_then_returns_to_gpu(self) -> None:
        gpu = route(
            "cold-gpu",
            "cold-model",
            "gpu",
            baseline=True,
            latency_us=300,
            energy_uj=100,
        )
        helper = route(
            "cold-cpu-phone",
            "cold-model",
            "phone",
            baseline=False,
            latency_us=500,
            energy_uj=70,
        )
        helper["resource_slots"] = {"cpu": 1, "phone": 1, "usb": 1}
        helper["finish_before_feature"] = "gpu_switch_start_us"
        scheduler_profile = profile(
            "gpu-switch-helper",
            "cold-model",
            [gpu, helper],
            [
                {
                    "resource_id": "gpu",
                    "kind": "gpu",
                    "capacity": 1,
                    "ready": True,
                    "identity": "test-gpu",
                },
                {
                    "resource_id": "cpu",
                    "kind": "cpu",
                    "capacity": 1,
                    "ready": True,
                    "identity": "test-cpu",
                },
                {
                    "resource_id": "phone",
                    "kind": "phone",
                    "capacity": 1,
                    "ready": True,
                    "identity": "test-phone",
                },
                {
                    "resource_id": "usb",
                    "kind": "transport",
                    "capacity": 1,
                    "ready": True,
                    "identity": "test-usb",
                },
            ],
            {
                "offload_requires_baseline_queue": True,
                "offload_min_finish_saving_us": 1,
            },
        )
        scheduler = UnifiedScheduler((scheduler_profile,), "adaptive")
        scheduler.reserve_external_resource(
            "gpu", "hot-model-and-switch", 0, 1000
        )

        hidden = scheduler.schedule(request(
            "cold-early",
            "cold-model",
            100,
            {"gpu_switch_start_us": 900},
        ))
        self.assertEqual(hidden.route_id, "cold-cpu-phone")
        self.assertEqual(hidden.reason, "ADAPTIVE_ENERGY_SAVING")
        self.assertEqual(hidden.finish_us, 600)

        available = scheduler.schedule(request(
            "cold-late",
            "cold-model",
            600,
            {"gpu_switch_start_us": 900},
        ))
        self.assertEqual(available.route_id, "cold-gpu")
        self.assertEqual(available.reason, "ADAPTIVE_NO_QUALIFIED_ALTERNATIVE")
        self.assertIn(
            ("cold-cpu-phone", "FINISH_WINDOW_EXCEEDED"),
            available.rejected,
        )

    def test_lifecycle_state_changes_route_without_caller_policy(self) -> None:
        open_profile = profile(
            "epoch-open",
            "embed",
            [
                route(
                    "gpu",
                    "embed",
                    "gpu",
                    baseline=True,
                    latency_us=50,
                    energy_uj=100,
                ),
                route(
                    "phone",
                    "embed",
                    "phone",
                    baseline=False,
                    latency_us=80,
                    energy_uj=40,
                ),
            ],
        )
        reused_profile = profile(
            "epoch-reused",
            "embed",
            [
                route(
                    "gpu",
                    "embed",
                    "gpu",
                    baseline=True,
                    latency_us=50,
                    energy_uj=20,
                ),
                route(
                    "phone",
                    "embed",
                    "phone",
                    baseline=False,
                    latency_us=80,
                    energy_uj=40,
                ),
            ],
        )
        profiles = LifecycleProfileSet(
            "embed-cuda-epoch",
            "cuda0",
            {
                "cuda_epoch_open": open_profile,
                "cuda_epoch_reused": reused_profile,
            },
            "cuda_epoch_open",
            frozenset({"cuda_epoch_reused"}),
        )
        scheduler = UnifiedScheduler(
            (), "adaptive", lifecycle_profiles=(profiles,)
        )
        self.assertEqual(
            scheduler.schedule(request("r0", "embed")).route_id,
            "phone",
        )
        scheduler.update_lifecycle(
            "cuda0",
            LifecycleReceipt("cuda_epoch_reused", "tail-1"),
        )
        self.assertEqual(
            scheduler.schedule(request("r1", "embed", 100)).route_id,
            "gpu",
        )
        scheduler.update_lifecycle(
            "cuda0", LifecycleReceipt("cuda_epoch_unknown")
        )
        self.assertEqual(scheduler.profile_for("embed").profile_id, "epoch-open")
        with self.assertRaises(UnifiedScheduleError):
            scheduler.update_lifecycle(
                "cuda0",
                LifecycleReceipt("cuda_epoch_open", "invalid-tail"),
            )

    def test_external_route_and_matmul_work_share_calendar(self) -> None:
        resources = [
            {
                "resource_id": "compute:cpu",
                "kind": "desktop_cpu",
                "capacity": 1,
                "ready": True,
                "identity": "cpu",
            },
        ]
        task_profile = profile(
            "cpu-task-profile",
            "cpu-task",
            [route(
                "cpu-task",
                "cpu-task",
                "compute:cpu",
                baseline=True,
                latency_us=100,
                energy_uj=None,
            )],
            resources,
        )
        scheduler = UnifiedScheduler(
            (task_profile,),
            "control",
            matmul_profile=matmul_profile(),
        )
        scheduler.reserve_external_resource(
            "compute:cpu", "server", 0, 100
        )
        scheduler.enqueue_matmul(ModelProgram(
            program_id="p0",
            model_id="m0",
            arrival_us=0,
            deadline_us=10_000,
            ops=(MatmulOp(
                op_id="mm0",
                layer_id="l0",
                weight_id="w0",
                kernel_family="dense",
                quantization="q4_0",
                m=1,
                k=4,
                n=4,
                input_bytes=16,
                output_bytes=16,
                weight_bytes=16,
                compute_ops=32,
                allowed_devices=("cpu",),
            ),),
        ))
        decision = scheduler.schedule_next_matmul(0)
        assert decision is not None
        self.assertEqual(decision["start_us"], 100)
        task = scheduler.schedule(request("r0", "cpu-task"))
        self.assertGreaterEqual(task.start_us, decision["finish_us"])

    def test_active_lease_extension_blocks_until_physical_release(self) -> None:
        task_profile = profile(
            "active-execution-profile",
            "active-task",
            [route(
                "cpu-task",
                "active-task",
                "cpu",
                baseline=True,
                latency_us=100,
                energy_uj=None,
            )],
            [{
                "resource_id": "cpu",
                "kind": "cpu",
                "capacity": 1,
                "ready": True,
                "identity": "test-cpu",
            }],
        )
        scheduler = UnifiedScheduler((task_profile,), "control")
        first = scheduler.schedule(request("r0", "active-task"))
        token = first.leases[0].token

        self.assertEqual(scheduler.extend_lease(token, 500), 100)
        self.assertEqual(
            scheduler.resource_snapshot(150)["cpu"]["active_until_us"],
            500,
        )
        scheduler.release(token, 175)
        second = scheduler.schedule(request("r1", "active-task", 150))
        self.assertEqual(second.start_us, 175)

    def test_active_lease_extension_rejects_committed_collision(self) -> None:
        task_profile = profile(
            "active-collision-profile",
            "active-task",
            [route(
                "cpu-task",
                "active-task",
                "cpu",
                baseline=True,
                latency_us=100,
                energy_uj=None,
            )],
            [{
                "resource_id": "cpu",
                "kind": "cpu",
                "capacity": 1,
                "ready": True,
                "identity": "test-cpu",
            }],
        )
        scheduler = UnifiedScheduler((task_profile,), "control")
        first = scheduler.schedule(request("r0", "active-task"))
        scheduler.schedule(request("r1", "active-task", 100))

        with self.assertRaisesRegex(
            UnifiedScheduleError, "overlaps committed work"
        ):
            scheduler.extend_lease(first.leases[0].token, 150)


if __name__ == "__main__":
    unittest.main()
