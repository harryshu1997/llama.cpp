#!/usr/bin/env python3

from __future__ import annotations

import unittest

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    ONLINE_ROUTE_FAMILIES,
    ProfileBundle,
    Request,
    RuntimeExecutorBinding,
    RuntimeModelArtifact,
    RuntimePlacementSnapshot,
    UnifiedScheduleError,
    UnifiedScheduler,
)


MODEL_HASH = "sha256:" + "a" * 64
MODEL_BYTES = 800_000_000
WORKLOAD = "online-model-task"


def profile() -> ProfileBundle:
    def route(
        route_id: str,
        resource_slots: dict[str, int],
        *,
        baseline: bool,
        latency: int,
        energy: int,
    ) -> dict[str, object]:
        return {
            "baseline": baseline,
            "energy": {
                "boundary_id": "cpu+gpu+phone",
                "cost_uj": {
                    "fixed": 0,
                    "input_token": energy,
                    "kind": "affine_tokens_v1",
                    "output_token": energy,
                },
                "lower_error_ppm": 0,
                "status": "measured",
                "upper_error_ppm": 0,
            },
            "evidence_ids": [route_id + "-r1"],
            "granularity": "task",
            "latency": {
                "cost_us": {
                    "fixed": 0,
                    "input_token": latency,
                    "kind": "affine_tokens_v1",
                    "output_token": latency,
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
        "profile_id": "online-placement-test",
        "resources": [
            {
                "capacity": 2,
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
                "resource_id": "op15-usb",
            },
        ],
        "routes": [
            route(
                "desktop-cpu",
                {"desktop-cpu": 1},
                baseline=True,
                latency=100,
                energy=1_000,
            ),
            route(
                "desktop-cuda",
                {"cuda0": 1},
                baseline=False,
                latency=10,
                energy=100,
            ),
            route(
                "phone-adreno",
                {"op15-adreno": 1, "op15-usb": 1},
                baseline=False,
                latency=20,
                energy=200,
            ),
        ],
        "schema": "s42-general-scheduler-profile-v1",
        "trace_workload_map": {"online-model": WORKLOAD},
    })


def model() -> RuntimeModelArtifact:
    return RuntimeModelArtifact("online-model", MODEL_HASH, MODEL_BYTES)


def request(request_id: str, arrival_us: int, output_tokens: int) -> Request:
    return Request(
        request_id=request_id,
        workload_id=WORKLOAD,
        arrival_us=arrival_us,
        deadline_us=arrival_us + 1_000_000,
        input_tokens=100,
        output_tokens=output_tokens,
        quality_requirement="bounded_numeric",
    )


def snapshot(snapshot_id: str, captured_at_us: int) -> RuntimePlacementSnapshot:
    return RuntimePlacementSnapshot(
        snapshot_id=snapshot_id,
        captured_at_us=captured_at_us,
        valid_until_us=captured_at_us + 10_000,
        capacities={
            "host-ram": DeviceMemoryCapacity(
                "host-ram", 32_000_000_000, 10_000_000_000, 2_000_000_000
            ),
            "cuda0-vram": DeviceMemoryCapacity(
                "cuda0-vram", 16_000_000_000, 1_000_000_000, 500_000_000
            ),
            "op15-ram": DeviceMemoryCapacity(
                "op15-ram", 12_000_000_000, 5_000_000_000, 2_000_000_000
            ),
        },
    )


def binding(
    route_id: str,
    resources: tuple[str, ...],
    memory_resource_id: str,
) -> RuntimeExecutorBinding:
    return RuntimeExecutorBinding(
        executor_id="executor-" + route_id,
        route_id=route_id,
        model_id="online-model",
        artifact_sha256=MODEL_HASH,
        artifact_bytes=MODEL_BYTES,
        backend=route_id,
        resource_ids=resources,
        memory_resource_id=memory_resource_id,
        resident=True,
        ready=True,
    )


def bindings() -> tuple[RuntimeExecutorBinding, ...]:
    return (
        binding("desktop-cpu", ("desktop-cpu",), "host-ram"),
        binding("desktop-cuda", ("cuda0",), "cuda0-vram"),
        binding(
            "phone-adreno",
            ("op15-adreno", "op15-usb"),
            "op15-ram",
        ),
    )


def family_routes() -> dict[str, str | None]:
    return {
        "cpu": "desktop-cpu",
        "gpu": "desktop-cuda",
        "phone": "phone-adreno",
        "gpu-cpu": None,
        "gpu-phone": None,
        "cpu-phone": None,
    }


class OnlinePlacementTests(unittest.TestCase):
    def test_receipt_always_lists_six_route_families(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        _, decision, receipt = scheduler.schedule_online_request(
            request("request-0", 1_000, 20),
            model(),
            bindings(),
            snapshot=snapshot("snapshot-0", 900),
            observed_at_us=1_000,
            family_routes=family_routes(),
        )
        self.assertEqual(decision.route_id, "desktop-cuda")
        self.assertEqual(receipt.selected_family, "gpu")
        self.assertEqual(
            tuple(row.family for row in receipt.family_estimates),
            ONLINE_ROUTE_FAMILIES,
        )
        missing = {
            row.family: row.reason
            for row in receipt.family_estimates
            if row.route_id is None
        }
        self.assertEqual(missing, {
            "gpu-cpu": "ROUTE_NOT_PROFILED",
            "gpu-phone": "ROUTE_NOT_PROFILED",
            "cpu-phone": "ROUTE_NOT_PROFILED",
        })
        self.assertEqual(
            receipt.to_json()["schema"],
            "research-scheduler-online-placement-v2",
        )
        causal_input = receipt.to_json()["causal_input"]
        self.assertEqual(
            causal_input["scheduler_state"]["schema"],
            "research-scheduler-online-state-v1",
        )
        self.assertEqual(
            causal_input["scheduler_state"]["resource_timeline"][
                "schema"
            ],
            "research-scheduler-resource-timeline-state-v1",
        )

    def test_equal_observed_prefix_is_independent_of_future_suffix(self) -> None:
        first = request("request-0", 1_000, 20)
        left = UnifiedScheduler((profile(),), "enforce")
        right = UnifiedScheduler((profile(),), "enforce")
        _, _, left_first = left.schedule_online_request(
            first,
            model(),
            bindings(),
            snapshot=snapshot("snapshot-0", 900),
            observed_at_us=1_000,
            family_routes=family_routes(),
        )
        _, _, right_first = right.schedule_online_request(
            first,
            model(),
            bindings(),
            snapshot=snapshot("snapshot-0", 900),
            observed_at_us=1_000,
            family_routes=family_routes(),
        )
        self.assertEqual(left_first.to_json(), right_first.to_json())

        _, _, left_second = left.schedule_online_request(
            request("left-future", 2_000, 10),
            model(),
            bindings(),
            snapshot=snapshot("left-snapshot", 1_900),
            observed_at_us=2_000,
            family_routes=family_routes(),
        )
        _, _, right_second = right.schedule_online_request(
            request("right-future", 2_000, 200),
            model(),
            bindings(),
            snapshot=snapshot("right-snapshot", 1_900),
            observed_at_us=2_000,
            family_routes=family_routes(),
        )
        self.assertEqual(
            left_second.previous_prefix_sha256,
            left_first.prefix_sha256,
        )
        self.assertEqual(
            right_second.previous_prefix_sha256,
            right_first.prefix_sha256,
        )
        self.assertNotEqual(
            left_second.causal_input_sha256,
            right_second.causal_input_sha256,
        )

    def test_duplicate_request_is_rejected_before_another_commit(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        row = request("request-0", 1_000, 20)
        scheduler.schedule_online_request(
            row,
            model(),
            bindings(),
            snapshot=snapshot("snapshot-0", 900),
            observed_at_us=1_000,
            family_routes=family_routes(),
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "already placed"
        ):
            scheduler.schedule_online_request(
                row,
                model(),
                bindings(),
                snapshot=snapshot("snapshot-1", 1_000),
                observed_at_us=1_100,
                family_routes=family_routes(),
            )

    def test_causal_hash_binds_the_resource_calendar(self) -> None:
        left = UnifiedScheduler((profile(),), "enforce")
        right = UnifiedScheduler((profile(),), "enforce")
        right.reserve_external_resource(
            "op15-usb", "phone-owner", 900, 2_000
        )
        _, left_decision, left_receipt = left.schedule_online_request(
            request("request-0", 1_000, 20),
            model(),
            bindings(),
            snapshot=snapshot("snapshot-0", 900),
            observed_at_us=1_000,
            family_routes=family_routes(),
        )
        _, right_decision, right_receipt = right.schedule_online_request(
            request("request-0", 1_000, 20),
            model(),
            bindings(),
            snapshot=snapshot("snapshot-0", 900),
            observed_at_us=1_000,
            family_routes=family_routes(),
        )
        self.assertEqual(left_decision.route_id, right_decision.route_id)
        self.assertNotEqual(
            left_receipt.causal_input_sha256,
            right_receipt.causal_input_sha256,
        )
        left_state = left_receipt.to_json()["causal_input"][
            "scheduler_state"
        ]["resource_timeline"]
        right_state = right_receipt.to_json()["causal_input"][
            "scheduler_state"
        ]["resource_timeline"]
        self.assertNotEqual(left_state, right_state)

    def test_unmapped_profile_route_fails_before_scheduling(self) -> None:
        routes = family_routes()
        routes["gpu"] = None
        scheduler = UnifiedScheduler((profile(),), "enforce")
        with self.assertRaisesRegex(
            UnifiedScheduleError, "runtime routes lack online families"
        ):
            scheduler.schedule_online_request(
                request("request-0", 1_000, 20),
                model(),
                bindings(),
                snapshot=snapshot("snapshot-0", 900),
                observed_at_us=1_000,
                family_routes=routes,
            )

    def test_request_cannot_be_observed_before_arrival(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        with self.assertRaisesRegex(
            UnifiedScheduleError, "before its arrival"
        ):
            scheduler.schedule_online_request(
                request("request-0", 1_000, 20),
                model(),
                bindings(),
                snapshot=snapshot("snapshot-0", 800),
                observed_at_us=999,
                family_routes=family_routes(),
            )


if __name__ == "__main__":
    unittest.main()
