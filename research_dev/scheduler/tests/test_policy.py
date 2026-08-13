#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    RoutePolicy,
    LeaseDemand,
    ProfileBundle,
    Request,
    ResourceProfile,
    ResourceTimeline,
    SchedulerError,
    decision_to_json,
    RequestSemantics,
    RuntimeGateReceipt,
    RuntimeSnapshot,
)


def energy(status: str, fixed: int = 0) -> dict[str, object]:
    return {
        "status": status,
        "cost_uj": (
            None
            if status == "unknown"
            else {
                "kind": "affine_tokens_v1",
                "fixed": fixed,
                "input_token": 0,
                "output_token": 0,
            }
        ),
        "lower_error_ppm": 0,
        "upper_error_ppm": 0,
        **({"boundary_id": "physical-boundary-1"} if status != "unknown" else {}),
    }


def route(
    route_id: str,
    *,
    baseline: bool,
    latency_us: int,
    energy_status: str,
    energy_uj: int,
    resources: dict[str, int],
    quality: str = "exact",
    measured: bool = True,
    placement: bool = True,
    resident: bool = True,
    server_busy_ppm: int = 1_000_000,
    server_memory_bytes: int = 1000,
    overlap_status: str | None = None,
    overlap_wait_ppm: int = 20_000,
    overlap_error_ppm: int = 10_000,
    resource_leases: list[dict[str, object]] | None = None,
    runtime_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_overlap = overlap_status or (
        "not_applicable" if baseline else "measured"
    )
    overlap: dict[str, object] = {"status": selected_overlap}
    if selected_overlap in {"measured", "diagnostic"}:
        overlap.update({
            "exposed_join_wait_ppm": overlap_wait_ppm,
            "upper_error_ppm": overlap_error_ppm,
            "sample_count": 3,
        })
    result = {
        "route_id": route_id,
        "workload_id": "work",
        "granularity": "task" if baseline else "operator",
        "baseline": baseline,
        "resource_slots": resources,
        "latency": {
            "cost_us": {
                "kind": "affine_tokens_v1",
                "fixed": latency_us,
                "input_token": 0,
                "output_token": 0,
            },
            "ucb_add_us": 0,
            "sample_count": 3,
            "measured": measured,
        },
        "energy": energy(energy_status, energy_uj),
        "overlap": overlap,
        "quality_class": quality,
        "placement_verified": placement,
        "resident": resident,
        "server_busy_ppm": server_busy_ppm,
        "server_memory_bytes": server_memory_bytes,
        "evidence_ids": ["sha256:evidence"],
    }
    if resource_leases is not None:
        result["resource_leases"] = resource_leases
    if runtime_contract is not None:
        result["runtime_contract"] = runtime_contract
    return result


def profile(
    alternatives: list[dict[str, object]],
    *,
    phone_ready: bool = True,
    policy_overrides: dict[str, object] | None = None,
) -> ProfileBundle:
    policy = {
        "energy_saving_ppm": 50_000,
        "latency_limit_ppm": 1_050_000,
    }
    policy.update(policy_overrides or {})
    value = {
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "test-profile",
        "resources": [
            {
                "resource_id": "server",
                "kind": "cpu",
                "capacity": 1,
                "ready": True,
                "identity": "server-0",
            },
            {
                "resource_id": "phone",
                "kind": "npu",
                "capacity": 1,
                "ready": phone_ready,
                "identity": "phone-0",
            },
            {
                "resource_id": "usb",
                "kind": "transport",
                "capacity": 1,
                "ready": phone_ready,
                "identity": "usb-0",
            },
            {
                "resource_id": "gpu",
                "kind": "gpu",
                "capacity": 1,
                "ready": True,
                "identity": "gpu-0",
            },
        ],
        "trace_workload_map": {"model": "work"},
        "policy": policy,
        "routes": alternatives,
    }
    return ProfileBundle.from_json(value)


def request(
    request_id: str = "r0",
    *,
    arrival_us: int = 0,
    deadline_us: int = 10_000,
    quality: str = "exact",
    features: dict[str, int] | None = None,
    semantics: RequestSemantics | None = None,
) -> Request:
    return Request(
        request_id,
        "work",
        arrival_us,
        deadline_us,
        10,
        2,
        quality,
        features or {},
        semantics=semantics or RequestSemantics(),
    )


def baseline(status: str = "measured", value: int = 100) -> dict[str, object]:
    return route(
        "baseline", baseline=True, latency_us=1000,
        energy_status=status, energy_uj=value, resources={"server": 1},
    )


def offload(
    status: str = "measured",
    value: int = 80,
    latency_us: int = 900,
    **kwargs: object,
) -> dict[str, object]:
    return route(
        "offload", baseline=False, latency_us=latency_us,
        energy_status=status, energy_uj=value,
        resources={"server": 1, "phone": 1, "usb": 1},
        **kwargs,
    )


def phased_offload(*, phone_ucb_add_us: int = 0) -> dict[str, object]:
    return offload(
        latency_us=500,
        resource_leases=[
            {
                "lease_id": "host-branch",
                "resource_id": "server",
                "slots": 1,
                "start_offset_us": 0,
                "duration_us": 200,
            },
            {
                "lease_id": "phone-branch",
                "resource_id": "phone",
                "slots": 1,
                "start_offset_us": 0,
                "duration_us": 300,
                "duration_ucb_add_us": phone_ucb_add_us,
            },
            {
                "lease_id": "upload",
                "resource_id": "usb",
                "slots": 1,
                "start_offset_us": 0,
                "duration_us": 100,
            },
            {
                "lease_id": "download",
                "resource_id": "usb",
                "slots": 1,
                "start_offset_us": 400,
                "duration_us": 100,
            },
        ],
    )


def adaptive_profile(
    alternatives: list[dict[str, object]],
    *,
    minimum_finish_saving_us: int = 1,
) -> ProfileBundle:
    return profile(
        alternatives,
        policy_overrides={
            "latency_limit_ppm": 20_000_000,
            "offload_requires_baseline_queue": True,
            "offload_min_finish_saving_us": minimum_finish_saving_us,
        },
    )


def gpu_baseline(
    *, latency_us: int = 1000, energy_uj: int = 100
) -> dict[str, object]:
    return route(
        "gpu", baseline=True, latency_us=latency_us,
        energy_status="measured", energy_uj=energy_uj,
        resources={"gpu": 1},
    )


def task_route(
    route_id: str,
    *,
    latency_us: int,
    energy_uj: int,
    resources: dict[str, int],
    energy_status: str = "measured",
) -> dict[str, object]:
    result = route(
        route_id, baseline=False, latency_us=latency_us,
        energy_status=energy_status, energy_uj=energy_uj,
        resources=resources,
    )
    result["granularity"] = "task"
    result["overlap"] = {"status": "not_applicable"}
    return result


RUNTIME_EPOCH = "sha256:" + "e" * 64


def runtime_contract() -> dict[str, object]:
    def requirement(residency: str, heartbeat: int | None) -> dict[str, object]:
        return {
            "heartbeat_max_age_us": heartbeat,
            "max_temperature_millic": 50000,
            "allowed_thermal_buckets": ["nominal"],
            "allowed_contention_buckets": ["qualified"],
            "max_slowdown_ppm": 1050000,
            "max_failure_count": 0,
            "reset_generation": 0,
            "required_residency_ids": [residency],
        }

    return {
        "schema": "s42-route-runtime-contract-v1",
        "epoch_key": RUNTIME_EPOCH,
        "failure_mode": "fallback_before_dispatch",
        "resources": {
            "server": requirement("cold-model", None),
            "phone": requirement("gemma-ffn", 1000),
            "usb": requirement("dmabuf-session", 1000),
        },
        "semantics": {
            "kv_owner": "llama_context",
            "kv_migration": False,
            "context_shift": True,
            "full_logits": True,
            "grammar": True,
            "sampler_locations": ["desktop"],
            "speculative_decode": False,
            "cancellation_modes": ["before_dispatch", "cooperative"],
        },
    }


def runtime_snapshot(
    *,
    generation: int = 1,
    phone_temperature_millic: int = 40000,
    phone_ready: bool = True,
) -> RuntimeSnapshot:
    def state(residency: str, heartbeat: int | None = None) -> dict[str, object]:
        return {
            "ready": True,
            "generation": generation,
            "heartbeat_age_us": heartbeat,
            "temperature_millic": 40000,
            "thermal_bucket": "nominal",
            "contention_bucket": "qualified",
            "slowdown_ppm": 1000000,
            "failure_count": 0,
            "circuit_open": False,
            "reset_generation": 0,
            "residency_ids": [residency],
        }

    value = {
        "schema": "s42-runtime-snapshot-v1",
        "snapshot_id": f"snapshot-{generation}",
        "generation": generation,
        "epoch_key": RUNTIME_EPOCH,
        "captured_at_us": 100,
        "valid_until_us": 200,
        "cancellation_generation": 0,
        "resources": {
            "server": state("cold-model"),
            "phone": state("gemma-ffn", 100),
            "usb": state("dmabuf-session", 100),
        },
    }
    value["resources"]["phone"]["temperature_millic"] = phone_temperature_millic
    value["resources"]["phone"]["ready"] = phone_ready
    return RuntimeSnapshot.from_json(value)


class SchedulerTests(unittest.TestCase):
    def test_resource_capacity_assigns_independent_lanes(self) -> None:
        timeline = ResourceTimeline({
            "phone": ResourceProfile("phone", "npu", 2, True, "phone-0")
        })
        demand = (
            LeaseDemand("work", "phone", 1, 0, 100, 100),
        )
        first = timeline.preview_leases(demand, 0, 100, 100)
        timeline.commit_leases(first, "r0")
        second = timeline.preview_leases(demand, 0, 100, 100)
        timeline.commit_leases(second, "r1")
        third = timeline.preview_leases(demand, 0, 100, 100)
        self.assertEqual(first.start_us, 0)
        self.assertEqual(second.start_us, 0)
        self.assertNotEqual(first.plans[0].lanes, second.plans[0].lanes)
        self.assertEqual(third.start_us, 100)

    def test_control_always_uses_baseline(self) -> None:
        scheduler = RoutePolicy(profile([baseline(), offload()]), "control")
        self.assertEqual(scheduler.schedule(request()).route_id, "baseline")

    def test_enforce_selects_verified_energy_saving(self) -> None:
        scheduler = RoutePolicy(profile([baseline(), offload()]), "enforce")
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "offload")
        self.assertEqual(decision.reason, "VERIFIED_ENERGY_SAVING")

    def test_enforce_rejects_lower_power_but_higher_joules(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(value=100), offload(value=110)]), "enforce"
        )
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(("offload", "ENERGY_MARGIN"), decision.rejected)

    def test_enforce_rejects_unknown_energy(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(status="unknown"), offload(status="unknown")]),
            "enforce",
        )
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(("offload", "ENERGY_NOT_MEASURED"), decision.rejected)

    def test_enforce_rejects_estimated_energy(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(status="estimated"), offload(status="estimated")]),
            "enforce",
        )
        self.assertEqual(scheduler.schedule(request()).route_id, "baseline")

    def test_uncertainty_must_preserve_energy_margin(self) -> None:
        base = baseline(value=100)
        candidate = offload(value=90)
        candidate["energy"]["upper_error_ppm"] = 100_000  # type: ignore[index]
        scheduler = RoutePolicy(profile([base, candidate]), "enforce")
        self.assertEqual(scheduler.schedule(request()).route_id, "baseline")

    def test_latency_limit_rejects_energy_saving(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), offload(latency_us=1100)]), "enforce"
        )
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(("offload", "LATENCY_LIMIT"), decision.rejected)

    def test_enforce_rejects_a_worse_tardy_queue(self) -> None:
        scheduler = RoutePolicy(
            profile(
                [baseline(), offload(latency_us=500)],
                policy_overrides={"latency_limit_ppm": 20_000_000},
            ),
            "enforce",
        )
        _, finish_us, lanes = scheduler.timeline.preview(
            {"phone": 1, "usb": 1}, 0, 2_000
        )
        scheduler.timeline.commit(lanes, finish_us)
        decision = scheduler.schedule(request(
            "tardy", arrival_us=100, deadline_us=200
        ))
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(
            ("offload", "SLO_TARDINESS_REGRESSION"), decision.rejected
        )

    def test_enforce_rejects_unmeasured_split_overlap(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), offload(overlap_status="unknown")]), "enforce"
        )
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(("offload", "OVERLAP_NOT_MEASURED"), decision.rejected)

    def test_enforce_rejects_exposed_join_wait_over_limit(self) -> None:
        scheduler = RoutePolicy(
            profile([
                baseline(),
                offload(overlap_wait_ppm=45_000, overlap_error_ppm=10_000),
            ]),
            "enforce",
        )
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(("offload", "OVERLAP_LIMIT"), decision.rejected)

    def test_adaptive_keeps_idle_gpu_even_when_phone_uses_less_energy(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=800, energy_uj=50,
                    resources={"phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "gpu")
        self.assertEqual(decision.reason, "ADAPTIVE_BASELINE_AVAILABLE")
        self.assertIn(("phone", "BASELINE_NOT_QUEUED"), decision.rejected)

    def test_adaptive_keeps_queued_gpu_without_energy_benefit(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=800, energy_uj=110,
                    resources={"phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=10_000)
        )
        self.assertEqual(decision.route_id, "gpu")
        self.assertEqual(decision.reason, "ADAPTIVE_BASELINE_FEASIBLE")

    def test_adaptive_offloads_for_verified_energy_saving(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=800, energy_uj=80,
                    resources={"phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=10_000)
        )
        self.assertEqual(decision.route_id, "phone")
        self.assertEqual(decision.reason, "ADAPTIVE_ENERGY_SAVING")

    def test_adaptive_helper_must_finish_inside_overflow_window(self) -> None:
        helper = task_route(
            "phone", latency_us=800, energy_uj=80,
            resources={"phone": 1, "usb": 1},
        )
        helper["finish_before_feature"] = "gpu_switch_start_us"
        scheduler = RoutePolicy(
            adaptive_profile([gpu_baseline(), helper]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(request(
            "queued",
            arrival_us=100,
            deadline_us=10_000,
            features={"gpu_switch_start_us": 900},
        ))
        self.assertEqual(decision.route_id, "phone")

    def test_adaptive_rejects_helper_past_overflow_window(self) -> None:
        helper = task_route(
            "phone", latency_us=800, energy_uj=80,
            resources={"phone": 1, "usb": 1},
        )
        helper["finish_before_feature"] = "gpu_switch_start_us"
        scheduler = RoutePolicy(
            adaptive_profile([gpu_baseline(), helper]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(request(
            "queued",
            arrival_us=100,
            deadline_us=10_000,
            features={"gpu_switch_start_us": 899},
        ))
        self.assertEqual(decision.route_id, "gpu")
        self.assertIn(
            ("phone", "FINISH_WINDOW_EXCEEDED"), decision.rejected
        )

    def test_adaptive_rejects_helper_without_overflow_window(self) -> None:
        helper = task_route(
            "phone", latency_us=800, energy_uj=80,
            resources={"phone": 1, "usb": 1},
        )
        helper["finish_before_feature"] = "gpu_switch_start_us"
        scheduler = RoutePolicy(
            adaptive_profile([gpu_baseline(), helper]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=10_000)
        )
        self.assertEqual(decision.route_id, "gpu")
        self.assertIn(
            ("phone", "FINISH_WINDOW_MISSING"), decision.rejected
        )

    def test_baseline_cannot_be_bound_to_helper_finish_window(self) -> None:
        candidate = gpu_baseline()
        candidate["finish_before_feature"] = "gpu_switch_start_us"
        with self.assertRaisesRegex(
            SchedulerError, "baseline route cannot have a finish window"
        ):
            adaptive_profile([candidate])

    def test_adaptive_rejects_energy_saving_route_that_extends_gpu_tail(
        self,
    ) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=1950, energy_uj=50,
                    resources={"phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=10_000)
        )
        self.assertEqual(decision.route_id, "gpu")
        self.assertEqual(decision.reason, "ADAPTIVE_NO_SYSTEM_BENEFIT")
        self.assertIn(("phone", "NO_FINISH_SAVING"), decision.rejected)

    def test_adaptive_recovers_deadline_with_phone(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=800, energy_uj=110,
                    resources={"phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=950)
        )
        self.assertEqual(decision.route_id, "phone")
        self.assertEqual(decision.reason, "ADAPTIVE_DEADLINE_RECOVERY")

    def test_adaptive_accounts_for_all_companion_resource_queues(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "cpu", latency_us=700, energy_uj=90,
                    resources={"server": 1},
                ),
                task_route(
                    "cpu-phone", latency_us=600, energy_uj=60,
                    resources={"server": 1, "phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        _, finish_us, lanes = scheduler.timeline.preview(
            {"phone": 1, "usb": 1}, 0, 2000
        )
        scheduler.timeline.commit(lanes, finish_us)
        decision = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=900)
        )
        self.assertEqual(decision.route_id, "cpu")
        self.assertEqual(decision.reason, "ADAPTIVE_DEADLINE_RECOVERY")

    def test_adaptive_reduces_tardiness_when_no_route_meets_deadline(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=800, energy_uj=110,
                    resources={"phone": 1, "usb": 1},
                ),
            ], minimum_finish_saving_us=200),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=700)
        )
        self.assertEqual(decision.route_id, "phone")
        self.assertEqual(decision.reason, "ADAPTIVE_TARDINESS_REDUCTION")

    def test_adaptive_uses_actual_early_release_for_next_decision(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=800, energy_uj=80,
                    resources={"phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        first = scheduler.schedule(request("gpu-owner"))
        queued = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=10_000)
        )
        self.assertEqual(queued.route_id, "phone")
        scheduler.timeline.release(first.leases[0].token, 100)
        after_release = scheduler.schedule(
            request("after-release", arrival_us=100, deadline_us=10_000)
        )
        self.assertEqual(after_release.route_id, "gpu")
        self.assertEqual(
            after_release.reason, "ADAPTIVE_BASELINE_AVAILABLE"
        )

    def test_adaptive_fails_closed_on_unmeasured_energy(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=800, energy_uj=0,
                    energy_status="unknown",
                    resources={"phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        scheduler.schedule(request("gpu-owner"))
        decision = scheduler.schedule(
            request("queued", arrival_us=100, deadline_us=950)
        )
        self.assertEqual(decision.route_id, "gpu")
        self.assertIn(("phone", "ENERGY_NOT_MEASURED"), decision.rejected)

    def test_adaptive_queued_unknown_single_route_uses_baseline(self) -> None:
        single = gpu_baseline()
        single["energy"] = energy("unknown")
        scheduler = RoutePolicy(
            adaptive_profile([single]),
            "adaptive",
        )
        scheduler.schedule(request("owner"))
        decision = scheduler.schedule(request("queued", arrival_us=100))
        self.assertEqual(decision.route_id, "gpu")
        self.assertEqual(
            decision.reason, "ADAPTIVE_NO_QUALIFIED_ALTERNATIVE"
        )

    def test_approximate_route_rejected_for_exact_request(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), offload(quality="approximate")]), "shadow"
        )
        decision = scheduler.schedule(request(quality="exact"))
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(("offload", "QUALITY_INSUFFICIENT"), decision.rejected)

    def test_approximate_route_allowed_for_approximate_request(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), offload(quality="approximate")]), "shadow"
        )
        self.assertEqual(
            scheduler.schedule(request(quality="approximate")).route_id,
            "offload",
        )

    def test_unready_phone_falls_back(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), offload()], phone_ready=False), "shadow"
        )
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertTrue(any("not ready" in reason for _, reason in decision.rejected))

    def test_unmeasured_or_unplaced_route_falls_back(self) -> None:
        for candidate in (
            offload(measured=False),
            offload(placement=False),
            offload(resident=False),
        ):
            scheduler = RoutePolicy(profile([baseline(), candidate]), "shadow")
            self.assertEqual(scheduler.schedule(request()).route_id, "baseline")

    def test_shadow_selects_fastest_qualified_route(self) -> None:
        scheduler = RoutePolicy(profile([baseline(), offload()]), "shadow")
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "offload")
        self.assertEqual(decision.reason, "SHADOW_FASTEST_QUALIFIED")

    def test_composite_resources_are_queued_once(self) -> None:
        scheduler = RoutePolicy(profile([baseline(), offload()]), "shadow")
        first = scheduler.schedule(request("r0"))
        second = scheduler.schedule(request("r1", arrival_us=10))
        self.assertEqual(first.start_us, 0)
        self.assertEqual(second.start_us, first.finish_us)
        self.assertEqual(second.queue_us, first.finish_us - 10)

    def test_phase_leases_release_resources_before_route_finish(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload()]), "shadow"
        )
        first = scheduler.schedule(request("r0"))
        second = scheduler.schedule(request("r1"))
        self.assertEqual(first.route_id, "offload")
        self.assertEqual(first.finish_us, 500)
        self.assertEqual(second.route_id, "offload")
        self.assertEqual(second.start_us, 300)
        self.assertEqual(second.queue_us, 300)
        self.assertEqual(second.blocking_resources, ("phone",))
        self.assertEqual(
            second.queue_by_resource_us,
            {"phone": 300, "server": 0, "usb": 0},
        )

    def test_lease_ucb_controls_queue_reservation(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload(phone_ucb_add_us=100)]),
            "shadow",
        )
        first = scheduler.schedule(request("r0"))
        second = scheduler.schedule(request("r1"))
        phone = next(lease for lease in first.leases if lease.resource_id == "phone")
        self.assertEqual(phone.predicted_end_us, 300)
        self.assertEqual(phone.reserved_until_us, 400)
        self.assertEqual(second.start_us, 500)
        self.assertEqual(second.blocking_resources, ("phone", "usb"))

    def test_actual_completion_releases_a_lease_early(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload(phone_ucb_add_us=100)]),
            "shadow",
        )
        first = scheduler.schedule(request("r0"))
        phone = next(lease for lease in first.leases if lease.resource_id == "phone")
        scheduler.timeline.release(phone.token, 250)
        second = scheduler.schedule(request("r1"))
        self.assertEqual(second.start_us, 250)

    def test_resource_snapshot_reports_live_and_queued_availability(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload(phone_ucb_add_us=100)]),
            "shadow",
        )
        first = scheduler.schedule(request("r0"))
        second = scheduler.schedule(request("r1"))
        snapshot = scheduler.timeline.resource_snapshot(50)
        self.assertEqual(snapshot["phone"]["active_until_us"], 400)
        self.assertEqual(snapshot["phone"]["reserved_until_us"], 900)
        self.assertEqual(snapshot["phone"]["active_owners"], ["r0"])
        self.assertEqual(snapshot["phone"]["queued_owners"], ["r1"])
        self.assertEqual(snapshot["phone"]["next_free_us"], 400)

        phone = next(lease for lease in first.leases if lease.resource_id == "phone")
        scheduler.timeline.release(phone.token, 250)
        self.assertEqual(
            scheduler.timeline.resource_snapshot(50)["phone"]["next_free_us"],
            250,
        )

        scheduler.timeline.revoke_resource("phone", 50)
        self.assertIsNone(scheduler.timeline.next_available_us("phone", 50))
        scheduler.timeline.restore_resource("phone")
        self.assertEqual(scheduler.timeline.next_available_us("phone", 50), 50)

    def test_decision_serializes_resource_lease_receipts(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload()]), "shadow"
        )
        row = decision_to_json(scheduler.schedule(request("r0")))
        self.assertEqual(len(row["leases"]), 4)
        self.assertEqual(row["blocking_resources"], [])
        self.assertEqual(
            sorted(lease["resource_id"] for lease in row["leases"]),
            ["phone", "server", "usb", "usb"],
        )

    def test_overlapping_internal_leases_fail_closed(self) -> None:
        candidate = offload()
        candidate["resource_leases"] = [
            {
                "lease_id": "phone-a",
                "resource_id": "phone",
                "slots": 1,
                "start_offset_us": 0,
                "duration_us": 700,
            },
            {
                "lease_id": "phone-b",
                "resource_id": "phone",
                "slots": 1,
                "start_offset_us": 100,
                "duration_us": 700,
            },
            {
                "lease_id": "server",
                "resource_id": "server",
                "slots": 1,
                "start_offset_us": 0,
                "duration_us": 100,
            },
            {
                "lease_id": "usb",
                "resource_id": "usb",
                "slots": 1,
                "start_offset_us": 0,
                "duration_us": 100,
            },
        ]
        scheduler = RoutePolicy(profile([baseline(), candidate]), "shadow")
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(
            ("offload", "route lease concurrency exceeds resource capacity"),
            decision.rejected,
        )

    def test_explicit_leases_must_cover_declared_resources(self) -> None:
        candidate = phased_offload()
        candidate["resource_leases"] = [
            lease
            for lease in candidate["resource_leases"]
            if lease["resource_id"] != "usb"
        ]
        with self.assertRaisesRegex(SchedulerError, "cover every declared"):
            profile([baseline(), candidate])

    def test_lease_cannot_exceed_route_service(self) -> None:
        candidate = phased_offload()
        candidate["resource_leases"][1]["duration_us"] = 501
        scheduler = RoutePolicy(profile([baseline(), candidate]), "shadow")
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(
            ("offload", "resource lease exceeds route service time"),
            decision.rejected,
        )

    def test_busy_phone_can_make_baseline_faster(self) -> None:
        scheduler = RoutePolicy(profile([baseline(), offload()]), "shadow")
        _, finish, lanes = scheduler.timeline.preview(
            {"phone": 1, "usb": 1}, 0, 2000
        )
        scheduler.timeline.commit(lanes, finish)
        decision = scheduler.schedule(request("r1", arrival_us=100))
        self.assertEqual(decision.route_id, "baseline")

    def test_capacity_mode_minimizes_server_busy_time(self) -> None:
        candidate = offload(
            latency_us=1000,
            server_busy_ppm=250_000,
            server_memory_bytes=500,
        )
        scheduler = RoutePolicy(profile([baseline(), candidate]), "capacity")
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "offload")
        self.assertEqual(decision.server_busy_us, 250)

    def test_duplicate_or_missing_baseline_is_rejected(self) -> None:
        with self.assertRaises(SchedulerError):
            profile([offload()])
        duplicate = baseline()
        duplicate["route_id"] = "baseline-2"
        with self.assertRaises(SchedulerError):
            profile([baseline(), duplicate])

    def test_route_cannot_exceed_resource_capacity(self) -> None:
        candidate = offload()
        candidate["resource_slots"] = {"phone": 2}
        with self.assertRaises(SchedulerError):
            profile([baseline(), candidate])

    def test_arbitrary_workload_features_do_not_require_scheduler_changes(self) -> None:
        base = baseline()
        base["latency"]["cost_us"] = {  # type: ignore[index]
            "kind": "affine_features_v1",
            "fixed": 10,
            "coefficients": {"batch": 20, "weight_bytes_kib": 2},
        }
        scheduler = RoutePolicy(profile([base]), "control")
        work = Request(
            "feature-work", "work", 0, 10_000, 1, 1, "exact",
            {"batch": 4, "weight_bytes_kib": 100},
        )
        self.assertEqual(scheduler.schedule(work).service_us, 290)

    def test_missing_arbitrary_feature_fails_closed(self) -> None:
        base = baseline()
        base["latency"]["cost_us"] = {  # type: ignore[index]
            "kind": "affine_features_v1",
            "fixed": 10,
            "coefficients": {"context_tokens": 2},
        }
        scheduler = RoutePolicy(profile([base]), "control")
        with self.assertRaisesRegex(SchedulerError, "lacks cost feature"):
            scheduler.schedule(request())

    def test_runtime_readiness_overrides_historical_profile(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), offload()]),
            "shadow",
            {"phone": False, "usb": True},
        )
        decision = scheduler.schedule(request())
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(
            ("offload", "resource is not ready: phone"),
            decision.rejected,
        )

    def test_runtime_contract_admits_exact_snapshot(self) -> None:
        candidate = phased_offload()
        candidate["runtime_contract"] = runtime_contract()
        scheduler = RoutePolicy(
            profile([baseline(), candidate]),
            "shadow",
            runtime_snapshot=runtime_snapshot(),
        )
        decision = scheduler.schedule(request(), runtime_now_us=150)
        self.assertEqual(decision.route_id, "offload")
        self.assertEqual(decision.start_us, 150)
        self.assertIsNotNone(decision.runtime_gate)
        self.assertEqual(decision.runtime_gate.reason, "RUNTIME_GATES_PASS")

    def test_runtime_now_does_not_create_false_baseline_queue(self) -> None:
        scheduler = RoutePolicy(
            adaptive_profile([
                gpu_baseline(),
                task_route(
                    "phone", latency_us=800, energy_uj=50,
                    resources={"phone": 1, "usb": 1},
                ),
            ]),
            "adaptive",
        )
        decision = scheduler.schedule(request(), runtime_now_us=150)
        self.assertEqual(decision.start_us, 150)
        self.assertEqual(decision.route_id, "gpu")
        self.assertEqual(decision.reason, "ADAPTIVE_BASELINE_AVAILABLE")
        self.assertIn(("phone", "BASELINE_NOT_QUEUED"), decision.rejected)

    def test_runtime_now_must_be_nonnegative_integer(self) -> None:
        scheduler = RoutePolicy(profile([baseline()]), "control")
        with self.assertRaisesRegex(SchedulerError, "nonnegative integer"):
            scheduler.schedule(request(), runtime_now_us=-1)

    def test_atomic_recheck_falls_back_without_phone_lease(self) -> None:
        candidate = phased_offload()
        candidate["runtime_contract"] = runtime_contract()
        scheduler = RoutePolicy(
            profile([baseline(), candidate]),
            "shadow",
            runtime_snapshot=runtime_snapshot(),
        )
        original_gate = scheduler._runtime_gate
        calls = 0

        def revoke_on_commit(request_row, route_row, now_us):
            nonlocal calls
            result = original_gate(request_row, route_row, now_us)
            if route_row.runtime_contract is not None:
                calls += 1
                if calls == 2:
                    return RuntimeGateReceipt(
                        False,
                        "RUNTIME_CIRCUIT_OPEN",
                        "snapshot-2",
                        2,
                        RUNTIME_EPOCH,
                        ("phone",),
                    )
            return result

        scheduler._runtime_gate = revoke_on_commit
        decision = scheduler.schedule(request(), runtime_now_us=150)
        self.assertEqual(decision.route_id, "baseline")
        self.assertEqual(decision.reason, "ATOMIC_RUNTIME_FALLBACK")
        self.assertEqual(
            {lease.resource_id for lease in decision.leases},
            {"server"},
        )

    def test_thermal_gate_falls_back_before_dispatch(self) -> None:
        candidate = phased_offload()
        candidate["runtime_contract"] = runtime_contract()
        scheduler = RoutePolicy(
            profile([baseline(), candidate]),
            "shadow",
            runtime_snapshot=runtime_snapshot(phone_temperature_millic=50001),
        )
        decision = scheduler.schedule(request(), runtime_now_us=150)
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(("offload", "RUNTIME_THERMAL_LIMIT"), decision.rejected)

    def test_missing_runtime_snapshot_falls_back(self) -> None:
        candidate = phased_offload()
        candidate["runtime_contract"] = runtime_contract()
        scheduler = RoutePolicy(profile([baseline(), candidate]), "shadow")
        decision = scheduler.schedule(request(), runtime_now_us=150)
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(
            ("offload", "RUNTIME_SNAPSHOT_MISSING"),
            decision.rejected,
        )

    def test_unsupported_sampler_semantics_fall_back(self) -> None:
        candidate = phased_offload()
        candidate["runtime_contract"] = runtime_contract()
        scheduler = RoutePolicy(
            profile([baseline(), candidate]),
            "shadow",
            runtime_snapshot=runtime_snapshot(),
        )
        semantics = RequestSemantics(sampler_location="phone")
        decision = scheduler.schedule(
            request(semantics=semantics),
            runtime_now_us=150,
        )
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(
            ("offload", "SAMPLER_LOCATION_UNSUPPORTED"),
            decision.rejected,
        )

    def test_cancelled_request_never_acquires_leases(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload()]), "shadow"
        )
        with self.assertRaisesRegex(SchedulerError, "cancelled before dispatch"):
            scheduler.schedule(
                request(semantics=RequestSemantics(cancelled=True))
            )
        decision = scheduler.schedule(request("after-cancel"))
        self.assertEqual(decision.start_us, 0)

    def test_owner_cancellation_releases_current_and_future_phases(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload()]), "shadow"
        )
        scheduler.schedule(request("r0"))
        cancelled = scheduler.timeline.cancel_owner("r0", 100)
        self.assertEqual(len(cancelled), 3)
        second = scheduler.schedule(request("r1"))
        self.assertEqual(second.start_us, 100)

    def test_resource_failure_revokes_affected_owner(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload()]), "shadow"
        )
        scheduler.schedule(request("r0"))
        self.assertEqual(
            scheduler.timeline.revoke_resource("phone", 50),
            ("r0",),
        )
        decision = scheduler.schedule(request("r1", arrival_us=50))
        self.assertEqual(decision.route_id, "baseline")
        self.assertIn(
            ("offload", "resource is not ready: phone"),
            decision.rejected,
        )

    def test_runtime_snapshot_generation_must_advance(self) -> None:
        scheduler = RoutePolicy(
            profile([baseline(), phased_offload()]),
            "shadow",
            runtime_snapshot=runtime_snapshot(),
        )
        with self.assertRaisesRegex(SchedulerError, "did not advance"):
            scheduler.update_runtime_snapshot(runtime_snapshot())
        scheduler.update_runtime_snapshot(runtime_snapshot(generation=2))

    def test_runtime_contract_must_cover_every_resource(self) -> None:
        candidate = phased_offload()
        contract = runtime_contract()
        del contract["resources"]["usb"]
        candidate["runtime_contract"] = contract
        with self.assertRaisesRegex(SchedulerError, "cover every declared"):
            profile([baseline(), candidate])

    def test_unknown_runtime_resource_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchedulerError, "unknown resource"):
            RoutePolicy(
                profile([baseline(), offload()]), "shadow", {"other": True}
            )


if __name__ == "__main__":
    unittest.main()
