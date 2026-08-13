#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    EnergyProfile,
    RoutePolicy,
    ProfileBundle,
    Request,
    SchedulerError,
)


def request(features: dict[str, int] | None = None) -> Request:
    return Request(
        "r0", "work", 0, 100_000, 1, 1, "exact", features or {}
    )


def work(value: int | str) -> object:
    if type(value) is int:
        return value
    return {
        "kind": "affine_features_v1",
        "fixed": 0,
        "coefficients": {value: 1},
    }


def operator_cost(
    *,
    fixed_uj: int = 0,
    phone: bool = False,
    compute_ops: int | str = 1000,
    memory_bytes: int | str = 500,
    invocations: int | str = 1,
) -> dict[str, object]:
    domains: list[dict[str, object]] = [{
        "domain_id": "server",
        "idle_power_mw": 1000,
    }]
    kernels: list[dict[str, object]] = [{
        "active_power_mw": 3000,
        "domain_id": "server",
        "effective_bytes_per_s": 1_000_000,
        "effective_ops_per_s": 1_000_000,
        "kernel_id": "server-kernel",
        "launch_us": 10,
    }]
    operators: list[dict[str, object]] = [{
        "compute_ops": work(compute_ops),
        "invocations": work(invocations),
        "kernel_id": "server-kernel",
        "memory_bytes": work(memory_bytes),
        "op_id": "server-op",
    }]
    if phone:
        domains.append({
            "domain_id": "phone",
            "idle_power_mw": 5000,
        })
        kernels.append({
            "active_power_mw": 5000,
            "domain_id": "phone",
            "effective_bytes_per_s": 2_000_000,
            "effective_ops_per_s": 2_000_000,
            "kernel_id": "phone-kernel",
            "launch_us": 5,
        })
        operators.append({
            "compute_ops": 1000,
            "invocations": 1,
            "kernel_id": "phone-kernel",
            "memory_bytes": 500,
            "op_id": "phone-op",
        })
    return {
        "domains": domains,
        "fixed_uj": fixed_uj,
        "kernels": kernels,
        "kind": "operator_sum_v1",
        "operators": operators,
    }


def energy(cost: dict[str, object]) -> EnergyProfile:
    return EnergyProfile.from_json({
        "boundary_id": "cpu-package-plus-gpu-board-plus-phone",
        "cost_uj": cost,
        "lower_error_ppm": 0,
        "status": "measured",
        "upper_error_ppm": 0,
    })


def affine_energy(value: int) -> dict[str, object]:
    return {
        "boundary_id": "cpu-package-plus-gpu-board-plus-phone",
        "cost_uj": {
            "fixed": value,
            "input_token": 0,
            "kind": "affine_tokens_v1",
            "output_token": 0,
        },
        "lower_error_ppm": 0,
        "status": "measured",
        "upper_error_ppm": 0,
    }


def route(
    route_id: str,
    *,
    baseline: bool,
    route_energy: dict[str, object],
    resources: dict[str, int],
) -> dict[str, object]:
    return {
        "baseline": baseline,
        "energy": route_energy,
        "evidence_ids": ["sha256:evidence"],
        "granularity": "task" if baseline else "operator",
        "latency": {
            "cost_us": {
                "fixed": 2000,
                "input_token": 0,
                "kind": "affine_tokens_v1",
                "output_token": 0,
            },
            "measured": True,
            "sample_count": 3,
            "ucb_add_us": 0,
        },
        "overlap": (
            {"status": "not_applicable"}
            if baseline
            else {
                "exposed_join_wait_ppm": 0,
                "sample_count": 3,
                "status": "measured",
                "upper_error_ppm": 0,
            }
        ),
        "placement_verified": True,
        "quality_class": "exact",
        "resident": True,
        "resource_slots": resources,
        "route_id": route_id,
        "server_busy_ppm": 1_000_000,
        "server_memory_bytes": 1000,
        "workload_id": "work",
    }


class OperatorEnergyTests(unittest.TestCase):
    def test_compute_bound_operator_energy(self) -> None:
        profile = energy(operator_cost())
        value = profile.breakdown(request(), service_us=2000)
        assert value is not None
        self.assertEqual(value["operators"][0]["compute_us"], 1000)
        self.assertEqual(value["operators"][0]["memory_us"], 500)
        self.assertEqual(value["operators"][0]["active_us"], 1010)
        self.assertEqual(value["domains"]["server"]["idle_uj"], 2000)
        self.assertEqual(value["domains"]["server"]["dynamic_uj"], 2020)
        self.assertEqual(value["total_uj"], 4020)

    def test_memory_bound_operator_uses_roofline_maximum(self) -> None:
        profile = energy(operator_cost(compute_ops=100, memory_bytes=1500))
        value = profile.breakdown(request(), service_us=2000)
        assert value is not None
        row = value["operators"][0]
        self.assertEqual(row["compute_us"], 100)
        self.assertEqual(row["memory_us"], 1500)
        self.assertEqual(row["active_us"], 1510)

    def test_concurrent_device_energy_is_additive(self) -> None:
        profile = energy(operator_cost(phone=True, fixed_uj=7))
        value = profile.breakdown(request(), service_us=2000)
        assert value is not None
        self.assertEqual(value["domains"]["server"]["total_uj"], 4020)
        self.assertEqual(value["domains"]["phone"]["total_uj"], 10000)
        self.assertEqual(value["total_uj"], 14027)

    def test_work_can_scale_from_request_features(self) -> None:
        profile = energy(operator_cost(
            compute_ops="ops",
            memory_bytes="bytes",
            invocations="layers",
        ))
        value = profile.breakdown(
            request({"bytes": 1000, "layers": 2, "ops": 2000}),
            service_us=3000,
        )
        assert value is not None
        self.assertEqual(value["operators"][0]["active_us"], 2020)

    def test_missing_work_feature_fails_closed(self) -> None:
        profile = energy(operator_cost(compute_ops="ops"))
        with self.assertRaisesRegex(SchedulerError, "lacks cost feature"):
            profile.predict_uj(request(), service_us=2000)

    def test_domain_active_time_cannot_exceed_service(self) -> None:
        profile = energy(operator_cost(compute_ops=3000))
        with self.assertRaisesRegex(SchedulerError, "exceeds service time"):
            profile.predict_uj(request(), service_us=2000)

    def test_active_power_cannot_be_below_idle_power(self) -> None:
        value = operator_cost()
        value["kernels"][0]["active_power_mw"] = 999  # type: ignore[index]
        with self.assertRaisesRegex(SchedulerError, "below domain idle"):
            energy(value)

    def test_scheduler_can_select_operator_energy_model(self) -> None:
        bundle = ProfileBundle.from_json({
            "policy": {
                "energy_saving_ppm": 50_000,
                "latency_limit_ppm": 1_050_000,
            },
            "profile_id": "operator-energy-test",
            "resources": [
                {
                    "capacity": 1,
                    "identity": "server-0",
                    "kind": "cpu",
                    "ready": True,
                    "resource_id": "server",
                },
                {
                    "capacity": 1,
                    "identity": "phone-0",
                    "kind": "npu",
                    "ready": True,
                    "resource_id": "phone",
                },
            ],
            "routes": [
                route(
                    "baseline",
                    baseline=True,
                    route_energy=affine_energy(20_000),
                    resources={"server": 1},
                ),
                route(
                    "offload",
                    baseline=False,
                    route_energy={
                        "boundary_id": "cpu-package-plus-gpu-board-plus-phone",
                        "cost_uj": operator_cost(phone=True),
                        "lower_error_ppm": 0,
                        "status": "measured",
                        "upper_error_ppm": 0,
                    },
                    resources={"phone": 1, "server": 1},
                ),
            ],
            "schema": "s42-general-scheduler-profile-v1",
            "trace_workload_map": {"model": "work"},
        })
        decision = RoutePolicy(bundle, "enforce").schedule(request())
        self.assertEqual(decision.route_id, "offload")
        self.assertEqual(decision.energy_uj, 14020)
        assert decision.energy_breakdown is not None
        self.assertEqual(decision.energy_breakdown["kind"], "operator_sum_v1")
        self.assertEqual(len(decision.energy_breakdown["operators"]), 2)

    def test_scheduler_rejects_different_energy_boundaries(self) -> None:
        candidate_energy = {
            "boundary_id": "phone-only",
            "cost_uj": operator_cost(phone=True),
            "lower_error_ppm": 0,
            "status": "measured",
            "upper_error_ppm": 0,
        }
        bundle = ProfileBundle.from_json({
            "policy": {
                "energy_saving_ppm": 50_000,
                "latency_limit_ppm": 1_050_000,
            },
            "profile_id": "boundary-test",
            "resources": [
                {
                    "capacity": 1,
                    "identity": "server-0",
                    "kind": "cpu",
                    "ready": True,
                    "resource_id": "server",
                },
                {
                    "capacity": 1,
                    "identity": "phone-0",
                    "kind": "npu",
                    "ready": True,
                    "resource_id": "phone",
                },
            ],
            "routes": [
                route(
                    "baseline",
                    baseline=True,
                    route_energy=affine_energy(20_000),
                    resources={"server": 1},
                ),
                route(
                    "offload",
                    baseline=False,
                    route_energy=candidate_energy,
                    resources={"phone": 1, "server": 1},
                ),
            ],
            "schema": "s42-general-scheduler-profile-v1",
            "trace_workload_map": {"model": "work"},
        })
        decision = RoutePolicy(bundle, "enforce").schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(
            ("offload", "ENERGY_BOUNDARY_MISMATCH"), decision.rejected
        )


if __name__ == "__main__":
    unittest.main()
