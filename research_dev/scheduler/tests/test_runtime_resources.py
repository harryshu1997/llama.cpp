#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import unittest

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    ModelResidencyObservation,
    RuntimeMemoryDemand,
    RuntimePlacementSnapshot,
    RuntimeResidencyEviction,
    RuntimeResourceError,
    RuntimeTransitionPlan,
)
from research_dev.scheduler._internal.runtime_resources import (
    RuntimeMemoryLedger,
    preview_runtime_memory,
    transition_adjusted_memory_demands,
)


class RuntimeMemoryLedgerTests(unittest.TestCase):
    @staticmethod
    def snapshot() -> RuntimePlacementSnapshot:
        return RuntimePlacementSnapshot(
            snapshot_id="memory-interval-test",
            captured_at_us=0,
            valid_until_us=1_000,
            capacities={
                "memory-a": DeviceMemoryCapacity(
                    "memory-a", 100, 0, 0
                ),
            },
        )

    @staticmethod
    def demand() -> RuntimeMemoryDemand:
        return RuntimeMemoryDemand(
            demand_id="workspace",
            resource_id="memory-a",
            kind="workspace",
            required_bytes=80,
            resident_bytes=0,
            lifetime="request",
        )

    def test_nonoverlapping_queued_reservations_reuse_capacity(self) -> None:
        ledger = RuntimeMemoryLedger()
        ledger.reserve(
            "request-a",
            (self.demand(),),
            self.snapshot(),
            start_us=10,
            reserved_until_us=20,
        )

        with self.assertRaisesRegex(
            RuntimeResourceError, "memory capacity is insufficient"
        ):
            ledger.preview(
                (self.demand(),),
                self.snapshot(),
                start_us=19,
                reserved_until_us=30,
            )
        ledger.reserve(
            "request-b",
            (self.demand(),),
            self.snapshot(),
            start_us=20,
            reserved_until_us=30,
        )

        intervals = {
            row["owner_id"]: (
                row["start_us"], row["reserved_until_us"]
            )
            for row in ledger.snapshot()["reservations"]
        }
        self.assertEqual(intervals, {
            "request-a": (10, 20),
            "request-b": (20, 30),
        })

    def test_replacement_preview_excludes_owner_without_mutation(self) -> None:
        ledger = RuntimeMemoryLedger()
        ledger.reserve(
            "request-a",
            (self.demand(),),
            self.snapshot(),
            start_us=10,
            reserved_until_us=30,
        )
        before = ledger.snapshot()

        with self.assertRaisesRegex(
            RuntimeResourceError, "memory capacity is insufficient"
        ):
            ledger.preview(
                (self.demand(),),
                self.snapshot(),
                start_us=10,
                reserved_until_us=30,
            )
        self.assertEqual(
            dict(ledger.preview(
                (self.demand(),),
                self.snapshot(),
                start_us=10,
                reserved_until_us=30,
                exclude_owner_id="request-a",
            )),
            {"memory-a": 80},
        )
        self.assertEqual(ledger.snapshot(), before)

    def test_overlapping_preallocated_memory_uses_shared_high_watermark(
        self,
    ) -> None:
        ledger = RuntimeMemoryLedger()
        shared = replace(
            self.demand(),
            lifetime="resident",
            share_key="preallocated:endpoint-a:workspace",
        )
        ledger.reserve(
            "request-a",
            (shared,),
            self.snapshot(),
            start_us=10,
            reserved_until_us=30,
        )
        ledger.reserve(
            "request-b",
            (shared,),
            self.snapshot(),
            start_us=11,
            reserved_until_us=40,
        )

        state = ledger.snapshot()
        self.assertEqual(len(state["reservations"]), 2)
        self.assertEqual(
            {row["share_key"] for row in state["reservations"]},
            {"preallocated:endpoint-a:workspace"},
        )
        self.assertEqual(
            dict(ledger.preview(
                (shared,),
                self.snapshot(),
                start_us=12,
                reserved_until_us=20,
            )),
            {"memory-a": 0},
        )

    def test_projected_resident_memory_does_not_double_count_its_lease(
        self,
    ) -> None:
        ledger = RuntimeMemoryLedger()
        shared = replace(
            self.demand(),
            lifetime="resident",
            share_key="preallocated:endpoint-a:workspace",
        )
        ledger.reserve(
            "request-a",
            (shared,),
            self.snapshot(),
            start_us=10,
            reserved_until_us=30,
        )
        reflected = replace(
            self.snapshot(),
            capacities={
                "memory-a": DeviceMemoryCapacity(
                    "memory-a", 100, 80, 0
                ),
            },
        )
        resident = replace(shared, resident_bytes=80)

        self.assertEqual(
            dict(ledger.preview(
                (resident,),
                reflected,
                start_us=11,
                reserved_until_us=20,
            )),
            {},
        )
        with self.assertRaisesRegex(
            RuntimeResourceError, "memory capacity is insufficient"
        ):
            ledger.preview(
                (self.demand(),),
                reflected,
                start_us=11,
                reserved_until_us=20,
            )

    def test_whole_phone_llama_and_htp_shards_share_one_memory_ledger(
        self,
    ) -> None:
        snapshot = RuntimePlacementSnapshot(
            snapshot_id="phone-portfolio-memory",
            captured_at_us=0,
            valid_until_us=1_000,
            capacities={
                "op15-ram": DeviceMemoryCapacity(
                    "op15-ram", 10_000_000_000, 0, 0
                ),
            },
        )
        llama = RuntimeMemoryDemand(
            demand_id="weights:llama-whole-phone",
            resource_id="op15-ram",
            kind="model_weights",
            required_bytes=760_000_000,
            resident_bytes=0,
            lifetime="resident",
            share_key="llama-whole-phone",
            device_id="op15-phone",
        )

        def htp(session_id: str) -> RuntimeMemoryDemand:
            return RuntimeMemoryDemand(
                demand_id="weights:" + session_id,
                resource_id="op15-ram",
                kind="model_weights",
                required_bytes=3_200_000_000,
                resident_bytes=0,
                lifetime="resident",
                share_key="phone-ffn:" + session_id,
                device_id="op15-phone",
            )

        admitted = RuntimeMemoryLedger().preview(
            (llama, htp("HTP0"), htp("HTP1")),
            snapshot,
        )
        self.assertEqual(admitted["op15-ram"], 7_160_000_000)
        with self.assertRaisesRegex(
            RuntimeResourceError, "memory capacity is insufficient"
        ):
            RuntimeMemoryLedger().preview(
                (llama, htp("HTP0"), htp("HTP1"), htp("HTP2")),
                snapshot,
            )

    @staticmethod
    def replacement_demand() -> RuntimeMemoryDemand:
        return RuntimeMemoryDemand(
            demand_id="weights:accelerator-a",
            resource_id="memory-a",
            kind="model_weights",
            required_bytes=80,
            resident_bytes=0,
            lifetime="resident",
            share_key="sha256:" + "b" * 64 + ":accelerator-a",
            replacement_group="exclusive-accelerator-a",
            device_id="accelerator-a",
        )

    @staticmethod
    def replacement_transition(
        *, generation: int = 7,
    ) -> RuntimeTransitionPlan:
        return RuntimeTransitionPlan(
            transition_id="replace:accelerator-a",
            device_id="accelerator-a",
            source_state="cold",
            target_state="hot",
            latency_us=10,
            energy_uj=20,
            resource_ids=("exclusive-accelerator-a",),
            maturity="QUALIFIED",
            evictions=(RuntimeResidencyEviction(
                model_id="resident-model",
                artifact_sha256="sha256:" + "a" * 64,
                device_id="accelerator-a",
                resident_bytes=70,
                generation=generation,
            ),),
        )

    @staticmethod
    def resident_model() -> tuple[ModelResidencyObservation, ...]:
        return (ModelResidencyObservation(
            model_id="resident-model",
            artifact_sha256="sha256:" + "a" * 64,
            device_id="accelerator-a",
            state="hot",
            resident_tensor_ids=("tensor-a",),
            resident_bytes=70,
            generation=7,
        ),)

    def test_replacement_credit_requires_exact_transition_eviction(
        self,
    ) -> None:
        adjusted = transition_adjusted_memory_demands(
            (self.replacement_demand(),),
            transitions=(self.replacement_transition(),),
            residency=self.resident_model(),
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )
        self.assertEqual(adjusted[0].replaceable_bytes, 70)
        self.assertEqual(adjusted[0].additional_bytes, 10)

        with self.assertRaisesRegex(
            RuntimeResourceError, "transition eviction is stale"
        ):
            transition_adjusted_memory_demands(
                (self.replacement_demand(),),
                transitions=(self.replacement_transition(generation=8),),
                residency=self.resident_model(),
                exclusive_resource_by_device={
                    "accelerator-a": "exclusive-accelerator-a",
                },
            )

    def test_exact_endpoint_allocation_is_reclaimed_by_bound_eviction(
        self,
    ) -> None:
        resident = (replace(
            self.resident_model()[0],
            executor_id="executor:accelerator-a",
            reclaimable_bytes=90,
        ),)
        transition = replace(
            self.replacement_transition(),
            evictions=(replace(
                self.replacement_transition().evictions[0],
                executor_id="executor:accelerator-a",
                reclaimable_bytes=90,
            ),),
        )
        demands = transition_adjusted_memory_demands(
            (
                self.replacement_demand(),
                RuntimeMemoryDemand(
                    demand_id="workspace:accelerator-a",
                    resource_id="memory-a",
                    kind="workspace",
                    required_bytes=10,
                    resident_bytes=0,
                    lifetime="request",
                    device_id="accelerator-a",
                ),
            ),
            transitions=(transition,),
            residency=resident,
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )
        occupied = replace(
            self.snapshot(),
            capacities={
                "memory-a": DeviceMemoryCapacity(
                    "memory-a", 100, 90, 0
                ),
            },
        )

        required = preview_runtime_memory(
            demands,
            occupied,
            transitions=(transition,),
            residency=resident,
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )

        self.assertEqual(required, {"memory-a": 20})

    def test_endpoint_reclaim_requires_exact_executor_identity(self) -> None:
        resident = (replace(
            self.resident_model()[0],
            executor_id="executor:accelerator-a",
            reclaimable_bytes=90,
        ),)
        transition = replace(
            self.replacement_transition(),
            evictions=(replace(
                self.replacement_transition().evictions[0],
                executor_id="executor:other",
                reclaimable_bytes=90,
            ),),
        )

        with self.assertRaisesRegex(
            RuntimeResourceError, "transition eviction is stale"
        ):
            transition_adjusted_memory_demands(
                (self.replacement_demand(),),
                transitions=(transition,),
                residency=resident,
                exclusive_resource_by_device={
                    "accelerator-a": "exclusive-accelerator-a",
                },
            )

    def test_reclaimable_residency_serialization_is_backward_compatible(
        self,
    ) -> None:
        legacy = self.resident_model()[0]
        self.assertNotIn("executor_id", legacy.to_json())
        self.assertNotIn("reclaimable_bytes", legacy.to_json())
        exact = replace(
            legacy,
            executor_id="executor:accelerator-a",
            reclaimable_bytes=90,
        )

        restored = ModelResidencyObservation.from_json(exact.to_json())

        self.assertEqual(restored, exact)
        self.assertEqual(
            exact.to_json()["executor_id"], "executor:accelerator-a"
        )
        self.assertEqual(exact.to_json()["reclaimable_bytes"], 90)

    def test_replacement_without_transition_does_not_reclaim_memory(
        self,
    ) -> None:
        adjusted = transition_adjusted_memory_demands(
            (self.replacement_demand(),),
            transitions=(),
            residency=self.resident_model(),
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )
        self.assertEqual(adjusted[0].replaceable_bytes, 0)
        occupied = replace(
            self.snapshot(),
            capacities={
                "memory-a": DeviceMemoryCapacity(
                    "memory-a", 100, 70, 0
                ),
            },
        )
        with self.assertRaisesRegex(
            RuntimeResourceError, "memory capacity is insufficient"
        ):
            RuntimeMemoryLedger().preview(
                adjusted,
                occupied,
                residency=self.resident_model(),
                exclusive_resource_by_device={
                    "accelerator-a": "exclusive-accelerator-a",
                },
            )

    def test_fully_replaced_weights_retain_atomic_memory_ownership(
        self,
    ) -> None:
        demand = replace(self.replacement_demand(), required_bytes=60)
        adjusted = transition_adjusted_memory_demands(
            (demand,),
            transitions=(self.replacement_transition(),),
            residency=self.resident_model(),
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )
        occupied = replace(
            self.snapshot(),
            capacities={
                "memory-a": DeviceMemoryCapacity(
                    "memory-a", 100, 70, 0
                ),
            },
        )

        reservations = RuntimeMemoryLedger().reserve(
            "request-replacement-owner",
            adjusted,
            occupied,
            transitions=(self.replacement_transition(),),
            residency=self.resident_model(),
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )

        self.assertEqual(len(reservations), 1)
        self.assertEqual(reservations[0].reserved_bytes, 0)
        self.assertEqual(reservations[0].replaced_bytes, 60)
        self.assertEqual(
            reservations[0].replacement_group,
            "exclusive-accelerator-a",
        )

    def test_smaller_replacement_leaves_memory_for_request_state(
        self,
    ) -> None:
        resident = (replace(
            self.resident_model()[0],
            executor_id="executor:accelerator-a",
            reclaimable_bytes=90,
        ),)
        transition = replace(
            self.replacement_transition(),
            evictions=(replace(
                self.replacement_transition().evictions[0],
                executor_id="executor:accelerator-a",
                reclaimable_bytes=90,
            ),),
        )
        demands = transition_adjusted_memory_demands(
            (
                replace(self.replacement_demand(), required_bytes=60),
                RuntimeMemoryDemand(
                    demand_id="kv:accelerator-a",
                    resource_id="memory-a",
                    kind="kv_cache",
                    required_bytes=40,
                    resident_bytes=0,
                    lifetime="request",
                    device_id="accelerator-a",
                ),
            ),
            transitions=(transition,),
            residency=resident,
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )
        occupied = replace(
            self.snapshot(),
            capacities={
                "memory-a": DeviceMemoryCapacity(
                    "memory-a", 100, 90, 0
                ),
            },
        )

        required = preview_runtime_memory(
            demands,
            occupied,
            transitions=(transition,),
            residency=resident,
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )

        self.assertEqual(required, {"memory-a": 40})

    def test_failed_replacement_reservation_is_atomic(self) -> None:
        ledger = RuntimeMemoryLedger()
        demands = transition_adjusted_memory_demands(
            (
                self.replacement_demand(),
                RuntimeMemoryDemand(
                    demand_id="workspace:accelerator-a",
                    resource_id="memory-a",
                    kind="workspace",
                    required_bytes=10,
                    resident_bytes=0,
                    lifetime="request",
                    device_id="accelerator-a",
                ),
            ),
            transitions=(self.replacement_transition(),),
            residency=self.resident_model(),
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )
        occupied = replace(
            self.snapshot(),
            captured_at_us=50,
            capacities={
                "memory-a": DeviceMemoryCapacity(
                    "memory-a", 100, 70, 0
                ),
            },
        )
        before = ledger.snapshot()
        ledger.fail_after_mutations_for_test(1)
        with self.assertRaisesRegex(
            RuntimeResourceError, "injected memory reservation failure"
        ):
            ledger.reserve(
                "request-b",
                demands,
                occupied,
                transitions=(self.replacement_transition(),),
                residency=self.resident_model(),
                exclusive_resource_by_device={
                    "accelerator-a": "exclusive-accelerator-a",
                },
            )
        self.assertEqual(ledger.snapshot(), before)

    def test_exact_composite_eviction_replaces_host_and_accelerator_memory(
        self,
    ) -> None:
        source_hash = "sha256:" + "a" * 64
        target_hash = "sha256:" + "b" * 64
        residency = (
            ModelResidencyObservation(
                model_id="resident-model",
                artifact_sha256=source_hash,
                device_id="host-a",
                state="hot",
                resident_tensor_ids=("host-weight",),
                resident_bytes=60,
                generation=7,
                executor_id="executor:host-accelerator",
            ),
            ModelResidencyObservation(
                model_id="resident-model",
                artifact_sha256=source_hash,
                device_id="accelerator-a",
                state="hot",
                resident_tensor_ids=("accelerator-weight",),
                resident_bytes=70,
                generation=7,
                executor_id="executor:host-accelerator",
                reclaimable_bytes=90,
            ),
        )
        transition = RuntimeTransitionPlan(
            transition_id="replace:host-accelerator",
            device_id="accelerator-a",
            source_state="cold",
            target_state="hot",
            latency_us=10,
            energy_uj=20,
            resource_ids=("exclusive-accelerator-a",),
            maturity="QUALIFIED",
            prepares_device_ids=("host-a", "accelerator-a"),
            evictions=(
                RuntimeResidencyEviction(
                    model_id="resident-model",
                    artifact_sha256=source_hash,
                    device_id="host-a",
                    resident_bytes=60,
                    generation=7,
                    executor_id="executor:host-accelerator",
                    replacement_group="exclusive-accelerator-a",
                ),
                RuntimeResidencyEviction(
                    model_id="resident-model",
                    artifact_sha256=source_hash,
                    device_id="accelerator-a",
                    resident_bytes=70,
                    generation=7,
                    executor_id="executor:host-accelerator",
                    reclaimable_bytes=90,
                    replacement_group="exclusive-accelerator-a",
                ),
            ),
        )
        demands = (
            RuntimeMemoryDemand(
                demand_id="weights:host-a",
                resource_id="host-memory",
                kind="model_weights",
                required_bytes=60,
                resident_bytes=0,
                lifetime="resident",
                share_key=target_hash + ":host-a",
                replacement_group="exclusive-accelerator-a",
                device_id="host-a",
            ),
            RuntimeMemoryDemand(
                demand_id="weights:accelerator-a",
                resource_id="accelerator-memory",
                kind="model_weights",
                required_bytes=80,
                resident_bytes=0,
                lifetime="resident",
                share_key=target_hash + ":accelerator-a",
                replacement_group="exclusive-accelerator-a",
                device_id="accelerator-a",
            ),
        )

        adjusted = transition_adjusted_memory_demands(
            demands,
            transitions=(transition,),
            residency=residency,
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )

        self.assertEqual(
            {row.device_id: row.replaceable_bytes for row in adjusted},
            {"accelerator-a": 70, "host-a": 60},
        )
        occupied = RuntimePlacementSnapshot(
            snapshot_id="composite-memory",
            captured_at_us=0,
            valid_until_us=1_000,
            capacities={
                "host-memory": DeviceMemoryCapacity(
                    "host-memory", 64, 60, 0
                ),
                "accelerator-memory": DeviceMemoryCapacity(
                    "accelerator-memory", 100, 90, 0
                ),
            },
        )
        reservations = RuntimeMemoryLedger().reserve(
            "request-composite-replacement",
            adjusted,
            occupied,
            transitions=(transition,),
            residency=residency,
            exclusive_resource_by_device={
                "accelerator-a": "exclusive-accelerator-a",
            },
        )
        self.assertEqual(
            {
                (row.resource_id, row.reserved_bytes, row.replaced_bytes)
                for row in reservations
            },
            {
                ("host-memory", 0, 60),
                ("accelerator-memory", 10, 70),
            },
        )

    def test_associated_reclaim_requires_exact_exclusive_anchor(self) -> None:
        source_hash = "sha256:" + "a" * 64
        residency = (ModelResidencyObservation(
            model_id="resident-model",
            artifact_sha256=source_hash,
            device_id="host-a",
            state="hot",
            resident_tensor_ids=("host-weight",),
            resident_bytes=60,
            generation=7,
            executor_id="executor:host-accelerator",
        ),)
        transition = RuntimeTransitionPlan(
            transition_id="invalid-associated-reclaim",
            device_id="accelerator-a",
            source_state="cold",
            target_state="hot",
            latency_us=10,
            energy_uj=20,
            resource_ids=("exclusive-accelerator-a",),
            maturity="QUALIFIED",
            prepares_device_ids=("host-a", "accelerator-a"),
            evictions=(RuntimeResidencyEviction(
                model_id="resident-model",
                artifact_sha256=source_hash,
                device_id="host-a",
                resident_bytes=60,
                generation=7,
                executor_id="executor:host-accelerator",
                replacement_group="exclusive-accelerator-a",
            ),),
        )
        demand = RuntimeMemoryDemand(
            demand_id="weights:host-a",
            resource_id="host-memory",
            kind="model_weights",
            required_bytes=60,
            resident_bytes=0,
            lifetime="resident",
            share_key="sha256:" + "b" * 64 + ":host-a",
            replacement_group="exclusive-accelerator-a",
            device_id="host-a",
        )

        with self.assertRaisesRegex(
            RuntimeResourceError, "exclusive anchor"
        ):
            transition_adjusted_memory_demands(
                (demand,),
                transitions=(transition,),
                residency=residency,
                exclusive_resource_by_device={
                    "accelerator-a": "exclusive-accelerator-a",
                },
            )


if __name__ == "__main__":
    unittest.main()
