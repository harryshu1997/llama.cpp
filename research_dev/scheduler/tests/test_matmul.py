#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "research_dev/spikes/s42_general_energy_scheduler_v1"
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    EnergyDomainProfile,
    FollowOp,
    MatmulOp,
    MatmulScheduleError,
    MatmulSystemProfile,
    MatmulPlanner,
    ModelProgram,
    PowerInterval,
    PowerTimeline,
    load_matmul_workload as load_workload,
)
from research_dev.scheduler import (  # noqa: E402
    GIB,
    materialize_matmul_profile as materialize,
)


def profile_value(
    *,
    gpu_capacity: int = 10_000_000,
    phone_capacity: int = 10_000_000,
    cpu_minimum_n: int = 100,
    gpu_ops_per_s: int = 20_000_000_000,
    gpu_power_mw: int = 160_000,
    phone_ops_per_s: int = 2_000_000_000,
    queue_limit: int = 100,
    require_measured: bool = False,
) -> dict[str, object]:
    def kernel(
        profile_id: str,
        device_id: str,
        ops_per_s: int,
        domain: str,
        power_mw: int,
        minimum_n: int,
    ) -> dict[str, object]:
        return {
            "profile_id": profile_id,
            "device_id": device_id,
            "kernel_family": "test-mm",
            "quantization": "q4_0",
            "shape": {"m": 1, "k": 100, "n": 1000},
            "effective_ops_per_s": ops_per_s,
            "effective_bytes_per_s": 1_000_000_000_000,
            "launch_us": 0,
            "domain_power_mw": {domain: power_mw},
            "minimum_n": minimum_n,
            "maximum_n": 1000,
            "n_quantum": 100,
            "status": "measured",
            "evidence_ids": [f"sha256:{profile_id}"],
        }

    def link(
        link_id: str,
        source: str,
        target: str,
        resource: str,
    ) -> dict[str, object]:
        return {
            "link_id": link_id,
            "source_device": source,
            "target_device": target,
            "resource_id": resource,
            "fixed_latency_us": 10,
            "bandwidth_bytes_per_s": 1_000_000_000,
            "fixed_energy_uj": 0,
            "dynamic_pj_per_byte": 0,
            "domain_power_mw": {},
            "minimum_bytes": 0,
            "maximum_bytes": 0,
            "status": "measured",
            "ready": True,
            "evidence_ids": [f"sha256:{link_id}"],
        }

    return {
        "schema": "s42-matmul-vq-profile-v1",
        "profile_id": "test-4060ti-op15",
        "energy_boundary_id": "cpu-gpu-phone",
        "domains": [
            {"domain_id": "cpu-package", "idle_power_mw": 1000},
            {"domain_id": "gpu-board", "idle_power_mw": 1000},
            {"domain_id": "phone-system", "idle_power_mw": 500},
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
                "memory_capacity_bytes": gpu_capacity,
                "reserved_bytes": 0,
                "ready": True,
            },
            {
                "device_id": "phone",
                "kind": "phone_npu",
                "resource_id": "compute:phone",
                "memory_capacity_bytes": phone_capacity,
                "reserved_bytes": 0,
                "ready": True,
            },
        ],
        "kernels": [
            kernel(
                "cpu-mm", "cpu", 1_000_000_000,
                "cpu-package", 115_000, cpu_minimum_n,
            ),
            kernel(
                "gpu-mm", "gpu", gpu_ops_per_s,
                "gpu-board", gpu_power_mw, 100,
            ),
            kernel(
                "phone-mm", "phone", phone_ops_per_s,
                "phone-system", 5_000, 100,
            ),
        ],
        "links": [
            link("cpu-gpu", "cpu", "gpu", "link:pcie"),
            link("gpu-cpu", "gpu", "cpu", "link:pcie"),
            link("cpu-phone", "cpu", "phone", "link:usb"),
            link("phone-cpu", "phone", "cpu", "link:usb"),
        ],
        "policy": {
            "host_device_id": "cpu",
            "final_device_id": "cpu",
            "host_domain_id": "cpu-package",
            "host_active_power_mw": 115_000,
            "latency_limit_ppm": 1_000_000,
            "split_search_points": 32,
            "queue_limit": queue_limit,
            "require_measured": require_measured,
        },
    }


def mm(
    op_id: str,
    *,
    weight_bytes: int = 1_000_000,
    allowed_devices: tuple[str, ...] = ("cpu", "gpu", "phone"),
) -> MatmulOp:
    return MatmulOp(
        op_id=op_id,
        layer_id="layer-0",
        weight_id=f"{op_id}.weight",
        kernel_family="test-mm",
        quantization="q4_0",
        m=1,
        k=100,
        n=1000,
        input_bytes=100,
        output_bytes=100,
        weight_bytes=weight_bytes,
        compute_ops=200_000,
        split_quantum_n=100,
        allowed_devices=allowed_devices,
    )


def program(
    program_id: str,
    *ops: MatmulOp | FollowOp,
    arrival_us: int = 0,
    deadline_us: int = 100_000,
) -> ModelProgram:
    return ModelProgram(
        program_id=program_id,
        model_id="test-model",
        arrival_us=arrival_us,
        deadline_us=deadline_us,
        ops=tuple(ops),
    )


class MatmulVirtualQueueTests(unittest.TestCase):
    def scheduler(self, **kwargs: object) -> MatmulPlanner:
        return MatmulPlanner(
            MatmulSystemProfile.from_json(profile_value(**kwargs))
        )

    def test_gpu_route_beats_cpu_energy_and_returns_to_cpu(self) -> None:
        scheduler = self.scheduler(cpu_minimum_n=200)
        scheduler.enqueue(program("p0", mm("mm0", allowed_devices=("cpu", "gpu"))))
        result = scheduler.run()
        matmul = next(row for row in result["decisions"] if row["kind"] == "matmul")
        self.assertGreater(matmul["placement"]["gpu"], 0)
        self.assertLess(matmul["finish_us"], matmul["cpu_baseline_finish_us"])
        self.assertEqual(result["programs"][0]["final_device"], "cpu")
        final = result["decisions"][-1]
        self.assertEqual(final["kind"], "final_cpu_gather")
        self.assertEqual(final["transfers"][0]["source_device"], "gpu")

    def test_cpu_phone_output_split_overlaps_branches(self) -> None:
        scheduler = self.scheduler()
        scheduler.enqueue(
            program("p0", mm("mm0", allowed_devices=("cpu", "phone")))
        )
        result = scheduler.run()
        decision = result["decisions"][0]
        self.assertGreater(decision["placement"]["cpu"], 0)
        self.assertGreater(decision["placement"]["phone"], 0)
        self.assertEqual(
            decision["placement"]["cpu"] + decision["placement"]["phone"],
            1000,
        )
        cpu_kernel = next(
            row for row in decision["kernels"] if row["device_id"] == "cpu"
        )
        phone_kernel = next(
            row for row in decision["kernels"] if row["device_id"] == "phone"
        )
        phone_transfer = decision["transfers"][0]
        self.assertEqual(
            decision["service_us"],
            max(
                cpu_kernel["duration_us"],
                phone_transfer["duration_us"] + phone_kernel["duration_us"],
            ),
        )

    def test_three_engine_cut_is_searchable(self) -> None:
        scheduler = self.scheduler(
            gpu_ops_per_s=2_000_000_000,
            gpu_power_mw=10_000,
            phone_ops_per_s=2_000_000_000,
        )
        scheduler.enqueue(program("p0", mm("mm0")))
        result = scheduler.run()
        placement = result["decisions"][0]["placement"]
        self.assertGreater(placement["cpu"], 0)
        self.assertGreater(placement["gpu"], 0)
        self.assertGreater(placement["phone"], 0)

    def test_shard_safe_chain_avoids_per_matmul_usb_round_trip(self) -> None:
        scheduler = self.scheduler(
            cpu_minimum_n=1000,
            phone_ops_per_s=20_000_000_000,
        )
        follow = FollowOp("silu", "layer-0", "SILU", True, 100)
        scheduler.enqueue(program(
            "p0",
            mm("mm0", allowed_devices=("cpu", "phone")),
            follow,
            mm("mm1", allowed_devices=("cpu", "phone")),
        ))
        result = scheduler.run()
        matmuls = [row for row in result["decisions"] if row["kind"] == "matmul"]
        self.assertEqual(len(matmuls), 2)
        self.assertEqual(matmuls[0]["placement"]["phone"], 1000)
        self.assertEqual(matmuls[1]["placement"]["phone"], 1000)
        self.assertEqual(len(matmuls[0]["transfers"]), 1)
        self.assertEqual(matmuls[1]["transfers"], [])
        inherited = next(
            row for row in result["decisions"]
            if row["kind"] == "non_matmul_follow"
        )
        self.assertEqual(inherited["placement"], ["phone"])
        final = result["decisions"][-1]
        self.assertEqual(len(final["transfers"]), 1)
        self.assertEqual(final["transfers"][0]["source_device"], "phone")

    def test_non_shard_safe_op_forces_cpu_barrier(self) -> None:
        scheduler = self.scheduler(
            cpu_minimum_n=1000,
            phone_ops_per_s=20_000_000_000,
        )
        barrier = FollowOp("softmax", "layer-0", "SOFT_MAX", False, 100)
        scheduler.enqueue(program(
            "p0", mm("mm0", allowed_devices=("cpu", "phone")), barrier
        ))
        result = scheduler.run()
        row = next(
            item for item in result["decisions"]
            if item["kind"] == "non_matmul_barrier"
        )
        self.assertEqual(row["output_residency"]["shards"][0]["device_id"], "cpu")
        self.assertEqual(row["transfers"][0]["source_device"], "phone")

    def test_follow_activation_growth_still_obeys_phone_memory(self) -> None:
        scheduler = self.scheduler(
            cpu_minimum_n=1000,
            phone_ops_per_s=20_000_000_000,
            phone_capacity=1_000_250,
        )
        scheduler.enqueue(program(
            "p0",
            mm("mm0", allowed_devices=("cpu", "phone")),
            FollowOp("grow", "layer-0", "DUP", True, 1000),
        ))
        with self.assertRaisesRegex(
            MatmulScheduleError, "activation exceeds phone memory"
        ):
            scheduler.run()

    def test_phone_memory_capacity_limits_split_columns(self) -> None:
        scheduler = self.scheduler(phone_capacity=450_200)
        scheduler.enqueue(
            program("p0", mm("mm0", allowed_devices=("cpu", "phone")))
        )
        result = scheduler.run()
        decision = result["decisions"][0]
        self.assertGreater(decision["placement"]["phone"], 0)
        self.assertLessEqual(decision["placement"]["phone"], 400)
        self.assertGreater(
            result["candidate_rejections"].get("MEMORY_CAPACITY:phone", 0), 0
        )
        self.assertLessEqual(
            result["memory"]["peak_usage_bytes"]["phone"], 450_200
        )

    def test_last_matmul_prices_mandatory_cpu_return_before_selection(self) -> None:
        scheduler = self.scheduler(phone_capacity=100_000_000)
        op = mm("mm0", allowed_devices=("cpu", "phone"))
        op = replace(op, output_bytes=10_000_000)
        scheduler.enqueue(program("p0", op))
        result = scheduler.run()
        decision = result["decisions"][0]
        self.assertEqual(
            decision["placement"],
            {"cpu": 1000, "gpu": 0, "phone": 0},
        )
        self.assertEqual(decision["lookahead_transfers"], [])
        self.assertEqual(
            decision["lookahead_finish_us"], decision["finish_us"]
        )

    def test_external_hot_model_reservation_can_remove_gpu_route(self) -> None:
        scheduler = self.scheduler(gpu_capacity=1_000_000)
        scheduler.reserve_external_memory("gpu", "hot-model", 950_000)
        scheduler.enqueue(
            program("p0", mm("mm0", allowed_devices=("cpu", "gpu")))
        )
        result = scheduler.run()
        decision = result["decisions"][0]
        self.assertEqual(decision["placement"], {"cpu": 1000, "gpu": 0, "phone": 0})
        self.assertGreater(
            result["candidate_rejections"].get("MEMORY_CAPACITY:gpu", 0), 0
        )

    def test_weight_slice_reservation_grows_but_never_shrinks(self) -> None:
        scheduler = self.scheduler(phone_capacity=2_100_000)
        shared = mm("mm0", allowed_devices=("cpu", "phone"))
        scheduler.enqueue(program("p0", shared))
        scheduler.enqueue(program("p1", shared))
        result = scheduler.run()
        weights = result["memory"]["resident_weight_allocations"]["phone"]
        self.assertEqual(len(weights), 1)
        allocated = next(iter(weights.values()))
        selected = [
            row["placement"]["phone"] for row in result["decisions"]
            if row["kind"] == "matmul"
        ]
        self.assertEqual(allocated, max(selected) * 1000)

    def test_two_programs_observe_resource_queueing(self) -> None:
        scheduler = self.scheduler()
        scheduler.enqueue(
            program("p0", mm("mm0", allowed_devices=("cpu", "phone")))
        )
        scheduler.enqueue(
            program("p1", mm("mm1", allowed_devices=("cpu", "phone")))
        )
        result = scheduler.run()
        matmuls = [row for row in result["decisions"] if row["kind"] == "matmul"]
        self.assertEqual(len(matmuls), 2)
        self.assertEqual(matmuls[0]["queue_us"], 0)
        self.assertGreater(matmuls[1]["queue_us"], 0)
        self.assertTrue(matmuls[1]["blocking_resources"])

    def test_schedule_next_assigns_only_one_ready_queue_entry(self) -> None:
        scheduler = self.scheduler(cpu_minimum_n=1000)
        scheduler.enqueue(program(
            "p0",
            mm("mm0", allowed_devices=("cpu", "gpu")),
            FollowOp("silu", "layer-0", "SILU", True, 100),
            mm("mm1", allowed_devices=("cpu", "gpu")),
        ))

        first = scheduler.schedule_next(0)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first["kind"], "matmul")
        self.assertEqual(first["dispatch_state"], "ASSIGNED")
        partial = scheduler.result()
        self.assertEqual(
            partial["status"], "PARTIAL_PLANNED_NOT_RUNTIME_CERTIFIED"
        )
        self.assertEqual(partial["programs"][0]["scheduled_op_count"], 1)
        self.assertIsNone(scheduler.schedule_next(first["finish_us"] - 1))

        second = scheduler.schedule_next(first["finish_us"])
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second["kind"], "non_matmul_follow")
        result = scheduler.run()
        self.assertEqual(result["status"], "PLANNED_NOT_RUNTIME_CERTIFIED")

    def test_new_ready_task_sees_existing_device_free_forecast(self) -> None:
        scheduler = self.scheduler(cpu_minimum_n=1000)
        scheduler.enqueue(
            program("p0", mm("mm0", allowed_devices=("cpu", "gpu")))
        )
        first = scheduler.schedule_next(0, program_id="p0")
        self.assertIsNotNone(first)
        assert first is not None
        self.assertGreater(first["placement"]["gpu"], 0)

        scheduler.enqueue(
            program("p1", mm("mm1", allowed_devices=("cpu", "gpu")))
        )
        second = scheduler.schedule_next(0, program_id="p1")
        self.assertIsNotNone(second)
        assert second is not None
        self.assertGreater(second["device_free_before_us"]["gpu"], 0)
        self.assertGreater(second["queue_us"], 0)
        self.assertIn("compute:gpu", second["blocking_resources"])

    def test_external_busy_window_and_early_release_update_forecast(self) -> None:
        scheduler = self.scheduler(cpu_minimum_n=1000)
        leases = scheduler.reserve_external_resource(
            "compute:gpu", "hot-model", 0, 500
        )
        before = scheduler.resource_forecast(0)
        self.assertEqual(before["devices"]["gpu"]["predicted_free_us"], 500)
        self.assertEqual(
            before["devices"]["gpu"]["active_owners"],
            ["external:hot-model"],
        )

        scheduler.release_resource_lease(leases[0].token, 50)
        after = scheduler.resource_forecast(50)
        self.assertEqual(after["devices"]["gpu"]["predicted_free_us"], 50)
        scheduler.enqueue(program(
            "p0",
            mm("mm0", allowed_devices=("cpu", "gpu")),
            arrival_us=50,
        ))
        decision = scheduler.schedule_next(50)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertGreater(decision["placement"]["gpu"], 0)

    def test_runtime_gpu_memory_release_changes_only_new_assignment(self) -> None:
        scheduler = self.scheduler(
            gpu_capacity=1_100_000,
            cpu_minimum_n=1000,
        )
        scheduler.reserve_external_memory("gpu", "hot-model", 1_050_000)
        scheduler.enqueue(program(
            "p0", mm("mm0", allowed_devices=("cpu", "gpu"))
        ))
        first = scheduler.schedule_next(0, program_id="p0")
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first["placement"]["gpu"], 0)

        scheduler.update_external_memory(
            "gpu", "hot-model", 0, first["finish_us"]
        )
        scheduler.enqueue(program(
            "p1",
            mm("mm1", allowed_devices=("cpu", "gpu")),
            arrival_us=first["finish_us"],
        ))
        second = scheduler.schedule_next(
            first["finish_us"], program_id="p1"
        )
        self.assertIsNotNone(second)
        assert second is not None
        self.assertGreater(second["placement"]["gpu"], 0)
        self.assertEqual(first["placement"]["gpu"], 0)

    def test_runtime_device_restore_enables_new_gpu_assignment(self) -> None:
        value = profile_value(cpu_minimum_n=1000)
        gpu = next(
            row for row in value["devices"] if row["device_id"] == "gpu"
        )
        gpu["ready"] = False
        scheduler = MatmulPlanner(
            MatmulSystemProfile.from_json(value)
        )
        scheduler.enqueue(program(
            "p0", mm("mm0", allowed_devices=("cpu", "gpu"))
        ))
        first = scheduler.schedule_next(0, program_id="p0")
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first["placement"]["gpu"], 0)

        scheduler.set_device_ready("gpu", True, first["finish_us"])
        scheduler.enqueue(program(
            "p1",
            mm("mm1", allowed_devices=("cpu", "gpu")),
            arrival_us=first["finish_us"],
        ))
        second = scheduler.schedule_next(
            first["finish_us"], program_id="p1"
        )
        self.assertIsNotNone(second)
        assert second is not None
        self.assertGreater(second["placement"]["gpu"], 0)

    def test_measured_only_mode_rejects_interpolated_partial_cuts(self) -> None:
        scheduler = self.scheduler(
            require_measured=True,
            cpu_minimum_n=100,
        )
        scheduler.enqueue(program("p0", mm("mm0", allowed_devices=("cpu", "gpu"))))
        result = scheduler.run()
        decision = result["decisions"][0]
        self.assertIn(
            decision["placement"],
            (
                {"cpu": 1000, "gpu": 0, "phone": 0},
                {"cpu": 0, "gpu": 1000, "phone": 0},
            ),
        )
        self.assertTrue(decision["measured"])

    def test_virtual_queue_capacity_is_finite(self) -> None:
        scheduler = self.scheduler(queue_limit=1)
        scheduler.enqueue(program("p0", mm("mm0")))
        with self.assertRaisesRegex(MatmulScheduleError, "queue is full"):
            scheduler.enqueue(program("p1", mm("mm1")))

    def test_physical_campaign_materializer_sets_requested_memory_caps(self) -> None:
        source_path = ROOT / "MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json"
        source = json.loads(source_path.read_text(encoding="ascii"))
        value = materialize(source, generic_family=True)
        parsed = MatmulSystemProfile.from_json(value)
        self.assertEqual(parsed.devices["gpu"].memory_capacity_bytes, 16 * GIB)
        self.assertEqual(parsed.devices["phone"].memory_capacity_bytes, 10 * GIB)
        self.assertTrue(all(row.status == "estimated" for row in parsed.kernels))
        self.assertTrue(all(row.shape_k == 0 for row in parsed.kernels))
        self.assertEqual(len(parsed.links), 4)

    def test_overlapping_host_activity_uses_one_power_envelope(self) -> None:
        timeline = PowerTimeline(
            {"cpu": EnergyDomainProfile("cpu", 1000)}, 0
        )
        active = (PowerInterval("cpu", 0, 100, 115_000),)
        first = timeline.incremental_energy_nj(active, 0, 100)
        timeline.commit(active, 0, 100)
        second = timeline.incremental_energy_nj(active, 0, 100)
        self.assertEqual(first, 11_500_000)
        self.assertEqual(second, 0)

    def test_checked_in_example_parses_and_runs_on_generic_shadow_profile(self) -> None:
        source = json.loads(
            (ROOT / "MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json").read_text(
                encoding="ascii"
            )
        )
        profile = MatmulSystemProfile.from_json(
            materialize(source, generic_family=True)
        )
        programs = load_workload(ROOT / "MATMUL_VQ_EXAMPLE_WORKLOAD_V1.json")
        scheduler = MatmulPlanner(profile)
        for item in programs:
            scheduler.enqueue(item)
        result = scheduler.run()
        self.assertEqual(result["status"], "PLANNED_NOT_RUNTIME_CERTIFIED")
        self.assertEqual(
            [row["kind"] for row in result["decisions"]],
            ["matmul", "non_matmul_follow", "matmul", "final_cpu_gather"],
        )


if __name__ == "__main__":
    unittest.main()
