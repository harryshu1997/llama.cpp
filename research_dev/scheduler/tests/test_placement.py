#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    ComputeStep,
    ExecutionBranch,
    HierarchicalPlacementPlanner,
    OperatorCandidate,
    OperatorNode,
    PlacementError,
    PlacementHardwareProfile,
    ResidentAllocation,
    TaskRoute,
    TransferStep,
    placement_plan_to_json,
)


def hardware_value(
    *,
    phone_pool_bytes: int = 10_000,
    phone_gpu_status: str = "measured",
) -> dict[str, object]:
    def kernel(
        profile_id: str,
        device_id: str,
        domain_id: str,
        ops: int,
        active_power: int,
        status: str = "measured",
    ) -> dict[str, object]:
        return {
            "active_power_mw": active_power,
            "device_id": device_id,
            "domain_id": domain_id,
            "effective_bytes_per_s": ops,
            "effective_ops_per_s": ops,
            "evidence_ids": [f"sha256:{profile_id}"],
            "kernel_id": f"{profile_id}-kernel",
            "launch_us": 0,
            "profile_id": profile_id,
            "status": status,
        }

    def link(
        link_id: str,
        source: str,
        target: str,
        fixed_us: int,
        bandwidth: int,
        energy_uj: int,
    ) -> dict[str, object]:
        return {
            "bandwidth_bytes_per_s": bandwidth,
            "domain_active_power_mw": {},
            "dynamic_pj_per_byte": 0,
            "evidence_ids": [f"sha256:{link_id}"],
            "fixed_dynamic_uj": energy_uj,
            "fixed_latency_us": fixed_us,
            "link_id": link_id,
            "ready": True,
            "source_device": source,
            "status": "measured",
            "target_device": target,
        }

    return {
        "schema": "s42-placement-hardware-profile-v1",
        "profile_id": "four-device-test",
        "energy_boundary_id": "cpu-gpu-phone",
        "idle_charge_domains": [
            "cpu-package",
            "gpu-board",
            "phone-system",
        ],
        "memory_pools": [
            {"pool_id": "ram", "capacity_bytes": 100_000, "reserved_bytes": 0},
            {"pool_id": "vram", "capacity_bytes": 10_000, "reserved_bytes": 0},
            {
                "pool_id": "phone-dram",
                "capacity_bytes": phone_pool_bytes,
                "reserved_bytes": 0,
            },
        ],
        "devices": [
            {
                "device_id": "cpu",
                "kind": "desktop_cpu",
                "memory_pool_id": "ram",
                "ready": True,
            },
            {
                "device_id": "cuda",
                "kind": "desktop_gpu",
                "memory_pool_id": "vram",
                "ready": True,
            },
            {
                "device_id": "htp",
                "kind": "phone_npu",
                "memory_pool_id": "phone-dram",
                "ready": True,
            },
            {
                "device_id": "phone-gpu",
                "kind": "phone_gpu",
                "memory_pool_id": "phone-dram",
                "ready": True,
            },
        ],
        "domains": [
            {
                "domain_id": "cpu-package",
                "idle_power_mw": 1000,
                "status": "measured",
                "evidence_ids": ["sha256:cpu-idle"],
            },
            {
                "domain_id": "gpu-board",
                "idle_power_mw": 2000,
                "status": "measured",
                "evidence_ids": ["sha256:gpu-idle"],
            },
            {
                "domain_id": "phone-system",
                "idle_power_mw": 500,
                "status": "measured",
                "evidence_ids": ["sha256:phone-idle"],
            },
        ],
        "kernels": [
            kernel("cpu-mm", "cpu", "cpu-package", 1_000_000, 3000),
            kernel("cuda-mm", "cuda", "gpu-board", 10_000_000, 100_000),
            kernel("htp-mm", "htp", "phone-system", 2_000_000, 1000),
            kernel(
                "phone-gpu-mm",
                "phone-gpu",
                "phone-system",
                5_000_000,
                3000,
                phone_gpu_status,
            ),
        ],
        "links": [
            link("cpu-to-cuda", "cpu", "cuda", 10, 100_000_000, 50),
            link("cuda-to-cpu", "cuda", "cpu", 10, 100_000_000, 50),
            link("cpu-to-htp", "cpu", "htp", 100, 1_000_000, 50),
            link("htp-to-cpu", "htp", "cpu", 100, 1_000_000, 50),
            link("htp-to-phone-gpu", "htp", "phone-gpu", 0, 1_000_000_000, 0),
            link("phone-gpu-to-htp", "phone-gpu", "htp", 0, 1_000_000_000, 0),
        ],
    }


def compute(step_id: str, kernel: str, ops: int = 1000) -> ComputeStep:
    return ComputeStep(step_id, kernel, 1, ops, 0)


def candidate(
    operator_id: str,
    candidate_id: str,
    device: str,
    kernel: str,
    *,
    status: str = "measured",
    allocations: tuple[ResidentAllocation, ...] = (),
    workspace: dict[str, int] | None = None,
) -> OperatorCandidate:
    return OperatorCandidate(
        candidate_id=candidate_id,
        operator_id=operator_id,
        input_device=device,
        output_device=device,
        branches=(ExecutionBranch("main", (compute(f"{candidate_id}:mm", kernel),)),),
        resident_allocations=allocations,
        workspace_bytes=workspace or {},
        quality_class="exact",
        status=status,
        placement_verified=True,
        evidence_ids=(f"sha256:{candidate_id}",),
    )


def node(
    operator_id: str,
    layer_id: str,
    *candidates: OperatorCandidate,
    transfer_bytes: int = 100,
) -> OperatorNode:
    return OperatorNode(
        operator_id,
        layer_id,
        transfer_bytes,
        transfer_bytes,
        tuple(candidates),
    )


class PlacementPlannerTests(unittest.TestCase):
    def test_minimizes_energy_instead_of_latency(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(hardware_value())
        )
        work = node(
            "mm",
            "layer-0",
            candidate("mm", "cpu", "cpu", "cpu-mm"),
            candidate("mm", "cuda", "cuda", "cuda-mm"),
        )
        loose = planner.plan_sequence(
            "loose", (work,), "cpu", "cpu", 10_000
        )
        self.assertEqual(loose.operator_decisions[0].candidate_id, "cpu")
        self.assertEqual(loose.latency_us, 1000)
        self.assertTrue(loose.search_optimal)

        tight = planner.plan_sequence(
            "tight", (work,), "cpu", "cpu", 500
        )
        self.assertEqual(tight.operator_decisions[0].candidate_id, "cuda")
        self.assertLessEqual(tight.latency_us, 500)
        self.assertGreater(tight.total_energy_uj, loose.total_energy_uj)

    def test_cuda_to_phone_uses_and_charges_multihop_path(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(hardware_value())
        )
        work = node(
            "mm",
            "layer-0",
            candidate("mm", "htp", "htp", "htp-mm"),
        )
        plan = planner.plan_sequence(
            "gpu-phone", (work,), "cuda", "cuda", 10_000
        )
        transition = plan.operator_decisions[0].transition
        assert transition is not None
        self.assertEqual(
            transition.link_ids, ("cuda-to-cpu", "cpu-to-htp")
        )
        assert plan.final_transfer is not None
        self.assertEqual(
            plan.final_transfer.link_ids, ("htp-to-cpu", "cpu-to-cuda")
        )

    def test_parallel_split_uses_max_branch_time_and_adds_energy(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(hardware_value())
        )
        split = OperatorCandidate(
            candidate_id="cpu-htp-split",
            operator_id="ffn",
            input_device="cpu",
            output_device="cpu",
            branches=(
                ExecutionBranch("host", (compute("host-mm", "cpu-mm"),)),
                ExecutionBranch("phone", (
                    TransferStep("upload", "cpu", "htp", 100),
                    compute("phone-mm", "htp-mm"),
                    TransferStep("download", "htp", "cpu", 100),
                )),
            ),
            quality_class="exact",
            status="measured",
            placement_verified=True,
            evidence_ids=("sha256:split",),
            split_axis="ffn_columns",
            split_amount=6000,
            split_total=15000,
        )
        plan = planner.plan_sequence(
            "split", (node("ffn", "layer-0", split),), "cpu", "cpu", 2000
        )
        self.assertEqual(plan.scope, "operator")
        self.assertEqual(plan.latency_us, 1000)
        self.assertEqual(
            plan.operator_decisions[0].compute_devices, ("cpu", "htp")
        )
        self.assertEqual(
            len(plan.operator_decisions[0].internal_transfers), 2
        )
        self.assertIn("link:cpu-to-htp", plan.resources)
        self.assertIn("link:htp-to-cpu", plan.resources)
        self.assertGreater(plan.dynamic_energy_uj, 2000)
        self.assertEqual(
            sum(plan.energy_by_domain_uj.values()), plan.total_energy_uj
        )

    def test_phone_gpu_and_htp_share_one_memory_pool(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(
                hardware_value(phone_pool_bytes=1300)
            )
        )
        shared = (
            ResidentAllocation("shared-weight", "htp", 800),
            ResidentAllocation("shared-weight", "phone-gpu", 800),
        )
        work = node(
            "mm",
            "layer-0",
            candidate(
                "mm",
                "phone-gpu",
                "phone-gpu",
                "phone-gpu-mm",
                allocations=shared,
                workspace={"htp": 200, "phone-gpu": 300},
            ),
        )
        plan = planner.plan_sequence(
            "shared-memory", (work,), "phone-gpu", "phone-gpu", 1000
        )
        self.assertEqual(plan.memory_by_pool_bytes["phone-dram"], 1300)
        self.assertEqual(plan.memory_by_device_bytes["htp"], 1000)
        self.assertEqual(plan.memory_by_device_bytes["phone-gpu"], 1100)

        distinct = (
            ResidentAllocation("htp-layout", "htp", 800),
            ResidentAllocation("gpu-layout", "phone-gpu", 800),
        )
        blocked = node(
            "other",
            "layer-0",
            candidate(
                "other",
                "too-large",
                "phone-gpu",
                "phone-gpu-mm",
                allocations=distinct,
            ),
        )
        with self.assertRaisesRegex(PlacementError, "memory=1"):
            planner.plan_sequence(
                "distinct-memory", (blocked,), "phone-gpu", "phone-gpu", 1000
            )

    def test_engine_mapping_limit_is_separate_from_shared_pool(self) -> None:
        value = hardware_value(phone_pool_bytes=10_000)
        value["devices"][2]["allocation_limit_bytes"] = 900  # type: ignore[index]
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(value)
        )
        work = node(
            "mm",
            "layer-0",
            candidate(
                "mm",
                "htp",
                "htp",
                "htp-mm",
                allocations=(ResidentAllocation("htp-weight", "htp", 1000),),
            ),
        )
        with self.assertRaisesRegex(PlacementError, "memory=1"):
            planner.plan_sequence(
                "htp-map-limit", (work,), "htp", "htp", 2000
            )

    def test_estimated_kernel_is_planning_only(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(
                hardware_value(phone_gpu_status="estimated")
            )
        )
        estimated = node(
            "mm",
            "layer-0",
            candidate(
                "mm",
                "phone-gpu",
                "phone-gpu",
                "phone-gpu-mm",
                status="estimated",
            ),
        )
        with self.assertRaises(PlacementError):
            planner.plan_sequence(
                "enforce", (estimated,), "phone-gpu", "phone-gpu", 1000
            )
        plan = planner.plan_sequence(
            "planning",
            (estimated,),
            "phone-gpu",
            "phone-gpu",
            1000,
            require_measured=False,
        )
        self.assertFalse(plan.measured)

    def test_estimated_idle_boundary_is_planning_only(self) -> None:
        value = hardware_value()
        value["domains"][2]["status"] = "estimated"  # type: ignore[index]
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(value)
        )
        work = node(
            "mm",
            "layer-0",
            candidate("mm", "cpu", "cpu", "cpu-mm"),
        )
        with self.assertRaisesRegex(PlacementError, "idle energy"):
            planner.plan_sequence(
                "enforce", (work,), "cpu", "cpu", 2000
            )
        plan = planner.plan_sequence(
            "planning",
            (work,),
            "cpu",
            "cpu",
            2000,
            require_measured=False,
        )
        self.assertFalse(plan.measured)

    def test_scope_distinguishes_layer_and_operator_placement(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(hardware_value())
        )
        layer_plan = planner.plan_sequence(
            "layers",
            (
                node("l0-mm", "layer-0", candidate(
                    "l0-mm", "cpu-0", "cpu", "cpu-mm"
                )),
                node("l1-mm", "layer-1", candidate(
                    "l1-mm", "cuda-1", "cuda", "cuda-mm"
                )),
            ),
            "cpu",
            "cpu",
            10_000,
        )
        self.assertEqual(layer_plan.scope, "layer")

        operator_plan = planner.plan_sequence(
            "operators",
            (
                node("gate", "layer-0", candidate(
                    "gate", "cpu-gate", "cpu", "cpu-mm"
                )),
                node("down", "layer-0", candidate(
                    "down", "cuda-down", "cuda", "cuda-mm"
                )),
            ),
            "cpu",
            "cpu",
            10_000,
        )
        self.assertEqual(operator_plan.scope, "operator")

    def test_task_route_includes_load_cost(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(hardware_value())
        )
        cpu_node = node(
            "cpu-mm", "layer-0",
            candidate("cpu-mm", "cpu", "cpu", "cpu-mm"),
        )
        htp_node = node(
            "htp-mm", "layer-0",
            candidate("htp-mm", "htp", "htp", "htp-mm"),
        )
        decision = planner.plan_task(
            "request-0",
            (
                TaskRoute(
                    "cpu-route", (cpu_node,), "cpu", "cpu",
                    load_status="not_applicable",
                ),
                TaskRoute(
                    "phone-route", (htp_node,), "htp", "htp",
                    resident=False,
                    load_latency_us=50,
                    load_energy_uj=10_000,
                    load_status="measured",
                    evidence_ids=("sha256:phone-load",),
                ),
            ),
            deadline_us=5000,
        )
        self.assertEqual(decision.route_id, "cpu-route")
        self.assertGreater(len(decision.placement.operator_decisions), 0)

    def test_task_route_rejects_unadoptable_staging(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(hardware_value())
        )
        cpu_node = node(
            "cpu-mm", "layer-0",
            candidate("cpu-mm", "cpu", "cpu", "cpu-mm"),
        )
        cuda_node = node(
            "cuda-mm", "layer-0",
            candidate("cuda-mm", "cuda", "cuda", "cuda-mm"),
        )
        decision = planner.plan_task(
            "request-0",
            (
                TaskRoute(
                    "cpu-route", (cpu_node,), "cpu", "cpu",
                    load_status="not_applicable",
                ),
                TaskRoute(
                    "staged-cuda-route", (cuda_node,), "cuda", "cuda",
                    resident=False,
                    load_latency_us=50,
                    load_energy_uj=10_000,
                    load_status="measured",
                    staged_allocation_bytes=1024,
                    staged_allocation_adoptable=False,
                    evidence_ids=("sha256:cuda-load",),
                ),
            ),
            deadline_us=5000,
        )
        self.assertEqual(decision.route_id, "cpu-route")
        self.assertIn(
            (
                "staged-cuda-route",
                "STAGED_ALLOCATION_NOT_ADOPTABLE",
            ),
            decision.rejected,
        )

    def test_json_output_contains_operator_layer_and_transfer_breakdown(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(hardware_value())
        )
        work = node(
            "mm", "layer-0",
            candidate("mm", "cuda", "cuda", "cuda-mm"),
        )
        value = placement_plan_to_json(planner.plan_sequence(
            "json", (work,), "cpu", "cpu", 1000
        ))
        self.assertEqual(value["energy_boundary_id"], "cpu-gpu-phone")
        self.assertEqual(value["scope"], "task")
        self.assertEqual(len(value["operators"]), 1)
        self.assertEqual(len(value["layers"]), 1)
        self.assertEqual(
            value["operators"][0]["transition"]["link_ids"],
            ["cpu-to-cuda"],
        )

    def test_frontier_truncation_does_not_claim_optimality(self) -> None:
        planner = HierarchicalPlacementPlanner(
            PlacementHardwareProfile.from_json(hardware_value()),
            beam_width=1,
        )
        work = node(
            "mm",
            "layer-0",
            candidate("mm", "cpu", "cpu", "cpu-mm"),
            candidate("mm", "cuda", "cuda", "cuda-mm"),
        )
        plan = planner.plan_sequence(
            "bounded", (work,), "cpu", "cpu", 10_000
        )
        self.assertFalse(plan.search_optimal)


if __name__ == "__main__":
    unittest.main()
