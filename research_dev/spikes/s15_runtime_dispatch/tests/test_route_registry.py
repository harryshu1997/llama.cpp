#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
S15 = HERE.parent
S14 = S15.parent / "s14_mixed_streaming_scheduler"
sys.path.insert(0, str(S15))
sys.path.insert(0, str(S14))

import route_fixtures as F  # noqa: E402
from op12_profile_adapter import OP12ProfileError, load_op12_head_route  # noqa: E402
from power_frontier_policy import CertifiedBatchPoint  # noqa: E402
from priority_batch_runtime import RouteConfig  # noqa: E402
from route_registry import (  # noqa: E402
    Lease,
    ReadyRouteRegistry,
    RouteRegistryError,
    RouteSnapshot,
    route_content_digest,
)

ENERGY = F.ENERGY


class MeasuredRouteTests(unittest.TestCase):
    def test_op15_and_op12_adapters_load_eligible_b1_routes(self) -> None:
        op15 = F.op15_head_snapshot()
        op12 = F.op12_head_snapshot()
        self.assertEqual(op15.batch_envelope, (1, 1))
        self.assertEqual(op12.batch_envelope, (1, 1))

    def test_validated_b32_profiles_are_explicit_route_variants(self) -> None:
        op15 = F.op15_b32_snapshot()
        op12 = F.op12_b32_snapshot()
        self.assertEqual(op15.batch_envelope, (1, 32))
        self.assertEqual(op12.batch_envelope, (1, 32))
        op15_b32 = next(point for point in op15.config.points if point.batch_size == 32)
        op12_b32 = next(point for point in op12.config.points if point.batch_size == 32)
        self.assertEqual(op15_b32.duration_us, 2_933_672)
        self.assertEqual(op12_b32.duration_us, 9_383_453)
        self.assertLessEqual(op15_b32.duration_us, 5_000_000)
        self.assertGreater(op12_b32.duration_us, 5_000_000)
        registry = ReadyRouteRegistry()
        generation = registry.install([op15, op12])
        self.assertEqual(generation, 1)
        self.assertTrue(registry.get(op15.route_id).dispatchable())
        self.assertTrue(registry.get(op12.route_id).dispatchable())

    def test_persistent_b32_profile_is_a_distinct_measured_epoch(self) -> None:
        old = F.op15_b32_snapshot()
        persistent = F.op15_persistent_b32_snapshot()
        points = {point.batch_size: point.duration_us for point in persistent.config.points}
        self.assertEqual(persistent.route_epoch, 14)
        self.assertEqual(points[32], 3_676_122)
        self.assertNotEqual(persistent.profile_id, old.profile_id)
        self.assertTrue(persistent.dispatchable())

    def test_op12_0_8_cannot_enter_the_registry(self) -> None:
        k8_paths = [ENERGY / f"op12_k8_b1_v2_r{repeat}.json" for repeat in range(7)]
        with self.assertRaises(OP12ProfileError):
            load_op12_head_route(k8_paths, 11)


class ContentDigestTests(unittest.TestCase):
    def test_declared_digest_mismatch_is_rejected(self) -> None:
        snapshot = F.op15_head_snapshot()
        forged = dataclasses.replace(snapshot, content_digest="sha256:" + "00" * 32)
        with self.assertRaisesRegex(RouteRegistryError, "content digest"):
            forged.validate()

    def test_profile_id_mutation_changes_the_digest(self) -> None:
        snapshot = F.op15_head_snapshot()
        mutated = dataclasses.replace(snapshot.config, profile_id="sha256:" + "11" * 32)
        bad = dataclasses.replace(snapshot, config=mutated)  # keeps the old digest
        with self.assertRaisesRegex(RouteRegistryError, "content digest"):
            bad.validate()

    def test_batch_size_mutation_changes_the_digest(self) -> None:
        snapshot = F.op15_head_snapshot()
        point = snapshot.config.points[0]
        moved = CertifiedBatchPoint(
            2, point.duration_us,
            point.correctness_certificate_id, point.placement_certificate_id,
        )
        mutated = dataclasses.replace(snapshot.config, points=(moved,))
        bad = dataclasses.replace(snapshot, config=mutated)
        with self.assertRaisesRegex(RouteRegistryError, "content digest"):
            bad.validate()

    def test_evidence_digest_mutation_changes_the_digest(self) -> None:
        snapshot = F.op15_head_snapshot()
        point = snapshot.config.points[0]
        tampered = CertifiedBatchPoint(
            point.batch_size, point.duration_us,
            "sha256:" + "22" * 32 + "#same-batch-token-correctness-7proc",
            point.placement_certificate_id,
        )
        mutated = dataclasses.replace(snapshot.config, points=(tampered,))
        bad = dataclasses.replace(snapshot, config=mutated)
        with self.assertRaisesRegex(RouteRegistryError, "content digest"):
            bad.validate()


class CompoundRouteTests(unittest.TestCase):
    def test_shared_tail_route_cannot_be_ready(self) -> None:
        ready = dataclasses.replace(F.shared_tail_snapshot(), state="READY")
        with self.assertRaisesRegex(RouteRegistryError, "compound route cannot be READY"):
            ready.validate()

    def test_serial_chain_route_cannot_be_ready(self) -> None:
        ready = dataclasses.replace(F.serial_chain_snapshot(), state="READY")
        with self.assertRaisesRegex(RouteRegistryError, "compound route cannot be READY"):
            ready.validate()

    def test_uncertified_single_route_cannot_be_ready(self) -> None:
        snapshot = F.op15_head_snapshot()
        bad = dataclasses.replace(snapshot, correctness_certified=False)
        with self.assertRaisesRegex(RouteRegistryError, "uncertified route cannot be READY"):
            bad.validate()

    def test_compound_route_is_not_dispatchable_even_when_present(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.shared_tail_snapshot(state="UNAVAILABLE")])
        snapshot = registry.get("twophone-sharedtail-0-12")
        self.assertFalse(snapshot.dispatchable())
        with self.assertRaisesRegex(RouteRegistryError, "not dispatchable"):
            registry.acquire_lease("twophone-sharedtail-0-12", 0)


class DeviceAndThermalTests(unittest.TestCase):
    def test_wrong_device_is_rejected(self) -> None:
        snapshot = F.op15_head_snapshot()
        bad = dataclasses.replace(snapshot, device_id="op99:deadbeef")
        with self.assertRaisesRegex(RouteRegistryError, "unknown device"):
            bad.validate()

    def test_thermal_violation_is_rejected(self) -> None:
        snapshot = F.op15_head_snapshot()
        bad = dataclasses.replace(snapshot, thermal_observed_millic=99_000, thermal_ceiling_millic=95_000)
        with self.assertRaisesRegex(RouteRegistryError, "thermal envelope"):
            bad.validate()


class LeaseTests(unittest.TestCase):
    def _registry(self) -> ReadyRouteRegistry:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot(), F.op12_head_snapshot()])
        return registry

    def test_independent_devices_can_be_leased_simultaneously(self) -> None:
        registry = self._registry()
        lease15 = registry.acquire_lease(F.op15_head_snapshot().route_id, 0)
        lease12 = registry.acquire_lease(F.op12_head_snapshot().route_id, 0)
        self.assertNotEqual(lease15.device_id, lease12.device_id)
        self.assertTrue(registry.validate_lease(lease15))
        self.assertTrue(registry.validate_lease(lease12))
        self.assertEqual(registry.outstanding(lease15.route_id), 1)
        self.assertEqual(registry.outstanding(lease12.route_id), 1)

    def test_execution_credit_cannot_overcommit(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot(credits=1)])
        route_id = F.op15_head_snapshot().route_id
        registry.acquire_lease(route_id, 0)
        with self.assertRaisesRegex(RouteRegistryError, "no free execution credit"):
            registry.acquire_lease(route_id, 0)

    def test_release_returns_the_credit(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot(credits=1)])
        route_id = F.op15_head_snapshot().route_id
        lease = registry.acquire_lease(route_id, 0)
        registry.release_lease(lease)
        self.assertEqual(registry.outstanding(route_id), 0)
        registry.acquire_lease(route_id, 0)  # credit is free again

    def test_stale_snapshot_generation_is_rejected(self) -> None:
        registry = self._registry()
        route_id = F.op15_head_snapshot().route_id
        observed = registry.generation
        registry.install([F.op15_head_snapshot(), F.op12_head_snapshot()])  # bumps generation
        with self.assertRaisesRegex(RouteRegistryError, "stale registry snapshot"):
            registry.acquire_lease(route_id, 0, expected_generation=observed)

    def test_registry_reinstall_invalidates_an_active_lease(self) -> None:
        registry = self._registry()
        route_id = F.op15_head_snapshot().route_id
        lease = registry.acquire_lease(route_id, 0)
        self.assertTrue(registry.validate_lease(lease))
        registry.install([F.op15_head_snapshot(), F.op12_head_snapshot()])
        self.assertFalse(registry.validate_lease(lease))
        registry.release_lease(lease)

    def test_each_epoch_change_invalidates_the_lease(self) -> None:
        route_id = F.op15_head_snapshot().route_id
        for field in ("route_epoch", "residency_epoch", "lease_epoch", "device_boot_epoch"):
            with self.subTest(field=field):
                registry = ReadyRouteRegistry()
                registry.install([F.op15_head_snapshot()])
                lease = registry.acquire_lease(route_id, 0)
                self.assertTrue(registry.validate_lease(lease))
                if field == "route_epoch":
                    bumped = F.op15_head_snapshot(route_epoch=12)
                else:
                    bumped = F.op15_head_snapshot(**{field: 2})
                registry.install([bumped])
                self.assertFalse(registry.validate_lease(lease))

    def test_drained_route_invalidates_the_lease(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        route_id = F.op15_head_snapshot().route_id
        lease = registry.acquire_lease(route_id, 0)
        draining = dataclasses.replace(F.op15_head_snapshot(), state="DRAINING")
        registry.install([draining])
        self.assertFalse(registry.validate_lease(lease))

    def test_released_lease_no_longer_validates(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        route_id = F.op15_head_snapshot().route_id
        lease = registry.acquire_lease(route_id, 0)
        registry.release_lease(lease)
        self.assertFalse(registry.validate_lease(lease))
        with self.assertRaisesRegex(RouteRegistryError, "not active"):
            registry.release_lease(lease)

    def test_foreign_lease_type_is_rejected(self) -> None:
        registry = self._registry()
        with self.assertRaisesRegex(RouteRegistryError, "lease must be a Lease"):
            registry.validate_lease(object())


if __name__ == "__main__":
    unittest.main()
